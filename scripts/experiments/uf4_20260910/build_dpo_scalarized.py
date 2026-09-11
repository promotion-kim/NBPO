"""Build the seven pre-declared scalarized-DPO preference sets from the cached BT rewards.

The weights are fixed before any outcome is seen, in the order
(instruction_following, truthfulness, honesty, helpfulness):

    (1,0,0,0) (0,1,0,0) (0,0,1,0) (0,0,0,1) (.25,.25,.25,.25) (.1,.1,.1,.7) (.1,.7,.1,.1)

Calibration is declared here and fitted on the TRAIN pool only: each objective's
BT reward is standardized by the train mean and standard deviation over every
sampled occurrence, so a weighted sum adds comparable units. Dev reuses the train
statistics and never refits them.

A pair is kept only when the weighted reward gap clears the tie threshold, so a
pair that a weighting cannot order does not become a training signal in that
weighting. Counts are recorded per weight rather than equalized.

This mirrors the NBPO pool exactly: same prompts, same eight learner
occurrences, same 28 unordered pairs. It is a different label on identical data.
"""
from __future__ import annotations

import argparse, hashlib, itertools, json
from pathlib import Path

import numpy as np

ROOT = Path("/work/uf4_20260910")
OBJECTIVES = ("instruction_following", "truthfulness", "honesty", "helpfulness")
POOL = 8
WEIGHTS = {
    "if_only":   (1.0, 0.0, 0.0, 0.0),
    "truth_only": (0.0, 1.0, 0.0, 0.0),
    "honesty_only": (0.0, 0.0, 1.0, 0.0),
    "help_only": (0.0, 0.0, 0.0, 1.0),
    "uniform":   (0.25, 0.25, 0.25, 0.25),
    "help_heavy": (0.1, 0.1, 0.1, 0.7),
    "truth_heavy": (0.1, 0.7, 0.1, 0.1),
}


def file_hash(p):
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_scores(root, shards=4):
    """prompt_id -> r_bt[K, 2, 8] (learners, comparators)."""
    out = {}
    for shard in range(shards):
        d = Path(root) / f"shard{shard}"
        for path in sorted(d.glob("chunk*.npz")):
            meta = json.loads((d / (path.stem + ".manifest.json")).read_text())
            if file_hash(path) != meta["sha256"]:
                raise ValueError(f"Score chunk hash mismatch: {path}")
            a = np.load(path, allow_pickle=True)
            for i, pid in enumerate([str(x) for x in a["prompt_ids"]]):
                out[pid] = a["r_bt"][:, i]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-name", default="v1")
    ap.add_argument("--tie-threshold", type=float, default=0.05,
                    help="Minimum standardized weighted-reward gap for a pair to be usable.")
    args = ap.parse_args()

    out = ROOT / "dpo" / args.out_name
    out.mkdir(parents=True, exist_ok=False)
    train = load_scores(ROOT / "scores/v1")
    dev = load_scores(ROOT / "scores/dev_v1")

    # calibration on the train pool only, over every sampled occurrence
    stack = np.concatenate([np.stack([train[p] for p in sorted(train)])[:, :, 0],
                            np.stack([train[p] for p in sorted(train)])[:, :, 1]], axis=-1)
    mean = stack.reshape(stack.shape[0], -1).mean(axis=1) if stack.ndim == 3 else None
    arr = np.stack([train[p] for p in sorted(train)])          # (X, K, 2, 8)
    flat = arr.transpose(1, 0, 2, 3).reshape(len(OBJECTIVES), -1)
    mean = flat.mean(axis=1)
    std = flat.std(axis=1, ddof=1)
    calibration = {"fitted_on": "policy_train pool, all 16 occurrences per prompt",
                   "per_objective_mean": mean.tolist(),
                   "per_objective_std": std.tolist(),
                   "rule": "r_tilde = (r - mean)/std, then r_w = sum_k w_k r_tilde_k",
                   "dev_refits": False}

    report = {"weights_order": list(OBJECTIVES), "weights": {k: list(v) for k, v in WEIGHTS.items()},
              "tie_threshold_standardized": args.tie_threshold,
              "calibration": calibration, "pool": "identical to the NBPO pool",
              "source_sha256": file_hash(__file__), "splits": {}}

    for split, scores in (("train", train), ("dev", dev)):
        pids = sorted(scores)
        arr = np.stack([scores[p] for p in pids])              # (X, K, 2, 8)
        learners = (arr[:, :, 0, :] - mean[None, :, None]) / std[None, :, None]   # (X, K, 8)
        stats = {}
        for name, w in WEIGHTS.items():
            wv = np.array(w, dtype=np.float64)
            rw = np.einsum("k,xki->xi", wv, learners)          # (X, 8)
            gaps, kept, ties = [], 0, 0
            for x in range(rw.shape[0]):
                for i, j in itertools.combinations(range(POOL), 2):
                    g = rw[x, i] - rw[x, j]
                    gaps.append(abs(g))
                    if abs(g) < args.tie_threshold:
                        ties += 1
                    else:
                        kept += 1
            gaps = np.array(gaps)
            stats[name] = {"total_pairs": int(gaps.size), "usable_pairs": kept,
                           "below_tie_threshold": ties,
                           "usable_fraction": kept / gaps.size,
                           "median_abs_gap": float(np.median(gaps)),
                           "p10_abs_gap": float(np.quantile(gaps, 0.1)),
                           "reward_sd_over_pool": float(rw.std(ddof=1))}
        report["splits"][split] = {"n_prompts": len(pids), "per_weight": stats}
        print(json.dumps({"split": split, "n_prompts": len(pids)}), flush=True)
        for name in WEIGHTS:
            s = stats[name]
            print("   %-12s usable %6d/%6d (%.3f)  median|gap| %.3f  reward sd %.3f"
                  % (name, s["usable_pairs"], s["total_pairs"], s["usable_fraction"],
                     s["median_abs_gap"], s["reward_sd_over_pool"]), flush=True)
    (out / "calibration_and_counts.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"written": str(out / "calibration_and_counts.json")}), flush=True)


if __name__ == "__main__":
    main()
