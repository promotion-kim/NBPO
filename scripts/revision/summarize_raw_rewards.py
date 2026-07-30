"""Raw (un-normalized) reward reporting with fixed anchors (feedback #18).
Reads the controlled-eval scored files and reports, per model and objective,
the mean raw RM score with prompt-level bootstrap CIs, plus the two fixed
anchors requested for scale interpretation: the base-policy mean and the
frozen pool mean (per objective, over the same prompts).
"""
import argparse
import json
from pathlib import Path

import numpy as np

OBJS = ["skywork", "athene", "armo"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_scored", required=True)
    ap.add_argument("--pool_scored", help="optional frozen-pool scored dir for the pool anchor")
    ap.add_argument("--output", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--eval_prefix", default="eval")
    ap.add_argument("--policy_lock", help="run_lock.json; limits variable joint rows to frozen policies")
    args = ap.parse_args()

    scores = {}
    names = None
    for obj in OBJS:
        for line in open(Path(args.eval_scored) / f"{args.eval_prefix}_{obj}.jsonl"):
            r = json.loads(line)
            names = r["response_model_names"]
            scores.setdefault(obj, {})[r["prompt"]] = np.array(r["all_rm_scores"], dtype=float)

    if args.policy_lock:
        names = json.loads(Path(args.policy_lock).read_text())["normalization_policies"]

    pool_anchor = {}
    if args.pool_scored:
        for obj in OBJS:
            vals = []
            for line in open(Path(args.pool_scored) / f"test_{obj}.jsonl"):
                r = json.loads(line)
                if r["prompt"] in scores[obj]:
                    vals.append(np.mean(r["all_rm_scores"]))
            pool_anchor[obj] = float(np.mean(vals))

    rng = np.random.default_rng(42)
    out = {"models": names, "objectives": OBJS, "pool_anchor_mean": pool_anchor, "per_model": {}}
    prompts = sorted(set.intersection(*[set(scores[o]) for o in OBJS]))
    out["num_prompts"] = len(prompts)
    idx = rng.integers(0, len(prompts), size=(args.bootstrap, len(prompts)))
    for mi, m in enumerate(names):
        row = {}
        for obj in OBJS:
            v = np.array([scores[obj][p][mi] for p in prompts])
            b = v[idx].mean(axis=1)
            row[obj] = {"raw_mean": float(v.mean()),
                        "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
        out["per_model"][m] = row
        print(m, {o: round(row[o]["raw_mean"], 4) for o in OBJS})
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(f"[raw] wrote {args.output}")


if __name__ == "__main__":
    main()
