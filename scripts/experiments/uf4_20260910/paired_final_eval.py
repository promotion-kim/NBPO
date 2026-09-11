"""Paired NBPO minus utilitarian on the SAME final prompts and the same judge calls.

Comparing two independently reported win rates throws away the pairing: both arms
were judged against the identical cached base response on the identical prompts,
so the per-prompt difference removes prompt difficulty entirely and is the
comparison with the tighter interval.

Intervals are whole-prompt bootstrap over prompts scored in both arms. They
measure prompt variability under one frozen teacher and one seed; they are not
seed uncertainty, and an interval containing zero is not equivalence.
"""
from __future__ import annotations

import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/work/uf4_20260910")
CRITERIA = ("instruction_following", "truthfulness", "honesty", "helpfulness")


def file_hash(p):
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def per_prompt(arm):
    """(criterion, prompt) -> order-averaged value, only when both orders parsed."""
    path = ROOT / "evaluation/final_eval" / arm / "judgments.jsonl"
    got = defaultdict(dict)
    with path.open() as stream:
        for line in stream:
            r = json.loads(line)
            if r["status"] == "ok":
                got[(r["criterion"], r["prompt_id"])][r["order"]] = r["value_for_policy"]
    return {k: 0.5 * (v[0] + v[1]) for k, v in got.items() if len(v) == 2}, file_hash(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--left", default="nbpo_mse_s42")
    ap.add_argument("--right", default="util_mse_s42")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    left, lh = per_prompt(args.left)
    right, rh = per_prompt(args.right)
    rng = np.random.default_rng(20260911)
    out = {"left": args.left, "right": args.right,
           "left_judgments_sha256": lh, "right_judgments_sha256": rh,
           "estimand": "per-prompt (left - right) of the order-averaged win rate against the same base response",
           "bootstrap": "%d whole-prompt replicates; prompt variability under one seed, not seed uncertainty" % args.bootstrap,
           "criteria": {}}
    for criterion in CRITERIA:
        shared = sorted({p for (c, p) in left if c == criterion} &
                        {p for (c, p) in right if c == criterion})
        d = np.array([left[(criterion, p)] - right[(criterion, p)] for p in shared])
        boot = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(args.bootstrap)])
        lo, hi = float(np.quantile(boot, .025)), float(np.quantile(boot, .975))
        out["criteria"][criterion] = {
            "n_paired_prompts": int(d.size), "mean_difference": float(d.mean()),
            "ci95": [lo, hi], "crosses_zero": bool(lo <= 0 <= hi),
            "fraction_left_better": float(np.mean(d > 0)),
            "fraction_equal": float(np.mean(d == 0)),
            "sd_of_differences": float(d.std(ddof=1))}
        print(json.dumps({"criterion": criterion, "n": int(d.size),
                          "mean_diff": round(float(d.mean()), 5),
                          "ci95": [round(lo, 5), round(hi, 5)],
                          "crosses_zero": bool(lo <= 0 <= hi)}), flush=True)
    dest = ROOT / "analysis" / f"paired_{args.left}_minus_{args.right}.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"written": str(dest)}), flush=True)


if __name__ == "__main__":
    main()
