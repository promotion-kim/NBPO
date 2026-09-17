"""Tokenize a panel's PROSPER fitting pairs into a trainable dataset.

Each row of prosper_targets_panel.py names one learner occurrence and one
reference occurrence of the same prompt, together with the regression target
eta * (g_hat(z) - g_hat(z')) that Algorithm 1 step 7 fits. This turns those rows
into the trainer's pair schema using the pool's OWN immutable token events, so
the sequence the trainer scores is byte-identical to the sequence that was
sampled and judged; nothing is re-tokenized from text.

The learner occurrence is placed as `chosen` and the reference occurrence as
`rejected`. That is a naming convention of the pair schema, not a preference
claim: the target carries the sign, and a negative target means the reference
side is preferred.

Written with the trainer's own datasets version (deps_train first on
PYTHONPATH), because a newer writer emits a feature type the trainer rejects.
"""
from __future__ import annotations

import argparse, glob, hashlib, json
from pathlib import Path

UF = Path("/work/uf4_20260910")


def file_hash(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_events(pool, shards):
    """candidate_id -> the pool's own recorded event for that occurrence."""
    out = {}
    for shard in range(shards):
        d = UF / "pools" / pool / ("shard%d" % shard)
        for path in sorted(d.glob("chunk*.jsonl")):
            manifest = json.loads((d / (path.stem + ".manifest.json")).read_text())
            if file_hash(path) != manifest["sha256"]:
                raise SystemExit("pool chunk hash mismatch: %s" % path)
            with path.open() as stream:
                for line in stream:
                    e = json.loads(line)
                    out[e["candidate_id"]] = e
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--targets", required=True, help="targets dir name, e.g. us1_prosper")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--pool-shards", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import datasets as ds_mod
    from datasets import Dataset, DatasetDict
    from mnpo_scripts.pair_tokenization import pair_from_candidate_events

    events = load_events(args.pool, args.pool_shards)
    tdir = UF / "targets" / args.targets
    report = json.loads((tdir / "targets_report.json").read_text())

    out, stats = {}, {}
    for split, src in (("train", "policy_train"), ("dev", "policy_dev")):
        rows = [json.loads(l) for l in (tdir / ("%s.jsonl" % src)).open() if l.strip()]
        built = []
        for r in rows:
            pid = r["prompt_id"]
            a = events["%s:learner:%d" % (pid, r["learner_index"])]
            b = events["%s:comparator:%d" % (pid, r["reference_index"])]
            if a["prompt_token_ids"] != b["prompt_token_ids"]:
                raise SystemExit("%s: the two occurrences disagree on the prompt tokens"
                                 % pid)
            pair = pair_from_candidate_events(a["prompt_token_ids"], a, b)
            row = {"prompt_id": pid, "prompt": a["prompt"],
                   "chosen": a["response"], "rejected": b["response"],
                   "chosen_response_id": a["candidate_id"],
                   "rejected_response_id": b["candidate_id"],
                   "chosen_text_sha256": a["response_sha256"],
                   "rejected_text_sha256": b["response_sha256"],
                   "ronpo_target": float(r["ronpo_target"]),
                   "selected_objective": int(r["objective"]),
                   "g_learner": float(r["g_learner"]),
                   "g_reference": float(r["g_reference"]),
                   "target_mode": "prosper_blackwell_step7",
                   "target_units": "final_logratio_change",
                   "estimator": report["estimator"],
                   "beta": report["beta"], "eta": report["eta"],
                   "pair_exposure": "learner_by_reference_%dx%d" % (8, 8),
                   "split": split, "panel": args.panel}
            row.update(pair)
            built.append(row)
        if not built:
            raise SystemExit("%s produced no rows" % split)
        cols = set(built[0])
        for need in ("chosen_input_ids", "rejected_input_ids", "chosen_labels",
                     "rejected_labels", "prompt_input_ids"):
            if need not in cols:
                raise SystemExit("pair tokenization did not supply %s" % need)
        ds = Dataset.from_list(built)
        out[split] = ds
        ts = [r["ronpo_target"] for r in built]
        stats[split] = {"rows": len(built),
                        "prompts": len({r["prompt_id"] for r in built}),
                        "target_mean": sum(ts) / len(ts),
                        "pairs_sha256": file_hash(tdir / ("%s.jsonl" % src))}

    DatasetDict(out).save_to_disk(args.out)
    manifest = {"dataset": args.out, "panel": args.panel,
                "rows": {k: v["rows"] for k, v in stats.items()},
                "columns": sorted(out["train"].column_names),
                "targets_report": report["checks"],
                "target_mode": "prosper_blackwell_step7",
                "tokenization": "pair_from_candidate_events on the pool's immutable tokens",
                "datasets_version": ds_mod.__version__,
                "splits": stats}
    Path(args.out, "prosper_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "targets_report", "splits")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
