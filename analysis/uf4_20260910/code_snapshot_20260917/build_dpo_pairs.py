"""Realize the seven declared scalarized-DPO preference sets on the NBPO pool.

The pairs are derived from the NBPO pair file itself, not rebuilt from the pool,
so every prompt context and every candidate's token ids are byte-identical to
the arms this baseline is compared against. A weighting changes only which side
of a pair is preferred, and drops the pairs it cannot order. That is what makes
this "a different label on identical data" rather than a second dataset.

Calibration and the tie threshold are read from dpo/v1/calibration_and_counts.json,
which was fitted on the train pool and declared before any outcome was seen; this
script never refits them. Dev reuses the train statistics unchanged.

The NBPO-specific columns are removed rather than carried along: a solver mass or
an Eq. (26) target sitting in a DPO dataset would be a target nothing optimizes,
and the next reader would have to guess whether it mattered. What replaces them
records how the label was made: the weight vector, the standardized gap, and the
threshold the gap had to clear.

One pass over the 9 GB train file writes all seven sets at once.
"""
from __future__ import annotations

import argparse, hashlib, json, os
from pathlib import Path

import numpy as np

ROOT = Path("/work/uf4_20260910")
OBJECTIVES = ("instruction_following", "truthfulness", "honesty", "helpfulness")
POOL = 8
SIDE_FIELDS = ("", "_input_ids", "_attention_mask", "_labels", "_response_id",
               "_text_sha256", "_token_sha256", "_candidate_index")
# Columns that describe the NBPO teacher, not the data. Dropped for DPO.
DROP_PREFIX = ("nbpo_", "solver_", "opponent_", "lambda_")
DROP_EXACT = ("aggregation", "representation", "target_column", "target_units",
              "target_artifact_hash",
              "target_mode", "eta_already_included", "nbpo_z")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_bt(score_root, shards=4):
    """prompt_id -> r_bt[K, 2, POOL] for the learner and comparator occurrences."""
    out = {}
    for shard in range(shards):
        d = Path(score_root) / f"shard{shard}"
        for path in sorted(d.glob("chunk*.npz")):
            meta = json.loads((d / (path.stem + ".manifest.json")).read_text())
            if file_hash(path) != meta["sha256"]:
                raise ValueError(f"Score chunk hash mismatch: {path}")
            a = np.load(path, allow_pickle=True)
            for i, pid in enumerate([str(x) for x in a["prompt_ids"]]):
                out[pid] = a["r_bt"][:, i]
    return out


def flip(row):
    """Swap the two sides of a pair, field group for field group."""
    for suffix in SIDE_FIELDS:
        a, b = "chosen" + suffix, "rejected" + suffix
        row[a], row[b] = row[b], row[a]
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, choices=("train", "dev"))
    ap.add_argument("--source-targets", default="nash_v1")
    ap.add_argument("--calibration", default=str(ROOT / "dpo/v1/calibration_and_counts.json"))
    ap.add_argument("--out-root", default=str(ROOT / "dpo/v1/pairs"))
    args = ap.parse_args()

    cal = json.loads(Path(args.calibration).read_text())
    if list(cal["weights_order"]) != list(OBJECTIVES):
        raise ValueError("Calibration declares a different objective order")
    weights = {k: np.asarray(v, dtype=np.float64) for k, v in cal["weights"].items()}
    mean = np.asarray(cal["calibration"]["per_objective_mean"], dtype=np.float64)
    std = np.asarray(cal["calibration"]["per_objective_std"], dtype=np.float64)
    tie = float(cal["tie_threshold_standardized"])

    scores = load_bt(ROOT / ("scores/v1" if args.split == "train" else "scores/dev_v1"))
    # standardized learner rewards per prompt: (K, POOL) -> scalarized per weight
    scalarized = {}
    for pid, r in scores.items():
        z = (r[:, 0, :] - mean[:, None]) / std[:, None]          # (K, POOL)
        scalarized[pid] = {name: w @ z for name, w in weights.items()}

    source = ROOT / "targets" / args.source_targets / "pairs" / f"{args.split}.jsonl"
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    handles, counts = {}, {}
    for name in weights:
        path = out_root / name / f"{args.split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
        handles[name] = path.open("w")
        counts[name] = {"kept": 0, "tied": 0, "flipped": 0}

    seen = 0
    with source.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            seen += 1
            pid = str(row["prompt_id"])
            i, j = int(row["chosen_candidate_index"]), int(row["rejected_candidate_index"])
            base = {k: v for k, v in row.items()
                    if not k.startswith(DROP_PREFIX) and k not in DROP_EXACT}
            for name, rw in scalarized[pid].items():
                gap = float(rw[i] - rw[j])
                if abs(gap) < tie:
                    counts[name]["tied"] += 1
                    continue
                out = dict(base)
                if gap < 0:
                    out = flip(out)
                    counts[name]["flipped"] += 1
                counts[name]["kept"] += 1
                out["dpo_weight_name"] = name
                out["dpo_weight_vector"] = [float(x) for x in weights[name]]
                out["dpo_scalarized_gap"] = abs(gap)
                out["dpo_tie_threshold_standardized"] = tie
                out["dpo_label_source"] = "frozen BT head, train-pool standardization"
                handles[name].write(json.dumps(out, ensure_ascii=False) + "\n")
            if seen % 20000 == 0:
                print(json.dumps({"rows_read": seen}), flush=True)
    for h in handles.values():
        h.close()

    report = {"split": args.split, "rows_read": seen, "tie_threshold": tie,
              "source_pairs": str(source), "source_pairs_sha256": file_hash(source),
              "calibration_sha256": file_hash(args.calibration),
              "builder_sha256": file_hash(__file__), "per_weight": {}}
    for name in weights:
        path = out_root / name / f"{args.split}.jsonl"
        c = counts[name]
        report["per_weight"][name] = {
            **c, "usable_fraction": c["kept"] / seen if seen else 0.0,
            "path": str(path), "sha256": file_hash(path),
            "bytes": path.stat().st_size}
        print("%-12s kept %7d  tied %6d  flipped %7d  (%.4f usable)"
              % (name, c["kept"], c["tied"], c["flipped"],
                 c["kept"] / seen if seen else 0.0), flush=True)
    out = out_root / f"build_report_{args.split}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(out)}), flush=True)


if __name__ == "__main__":
    main()
