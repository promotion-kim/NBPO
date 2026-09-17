"""Paired prompt bootstrap between two evaluation runs.

Two arms judged on the same panel against the same static baseline share every
prompt, so comparing their marginal intervals throws away the pairing and gives
an interval far wider than the comparison deserves. This resamples whole
prompts once and differences the two arms inside every replicate.

A prompt enters only if BOTH runs have every scheduled verdict parsed, so the
comparison is on one prompt set; the intersection is reported rather than
assumed. Per prompt an arm's score is the mean over its presentation orders,
which is what the win rate in complete.json averages, so the point estimate
here reduces to the difference of the two published win rates restricted to the
common prompts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/work/sub_20260914/eval_pairwise")


def file_hash(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def per_prompt(tag: str, orders: int = 2):
    """prompt_id -> mean value over orders, only for fully parsed prompts."""
    path = ROOT / tag / "verdicts.jsonl"
    table = defaultdict(list)
    dropped = set()
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["status"] != "ok":
                dropped.add(r["prompt_id"])
                continue
            table[r["prompt_id"]].append(float(r["value_for_arm"]))
    out = {p: float(np.mean(v)) for p, v in table.items()
           if len(v) == orders and p not in dropped}
    return out, file_hash(path), len(table), len(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="eval_pairwise tag, the row of interest")
    ap.add_argument("--b", required=True, help="eval_pairwise tag it is measured against")
    ap.add_argument("--orders", type=int, default=2)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--out", required=True, help="path under analysis/")
    args = ap.parse_args()

    a, a_sha, a_seen, a_kept = per_prompt(args.a, args.orders)
    b, b_sha, b_seen, b_kept = per_prompt(args.b, args.orders)
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit("no prompt is fully parsed in both %s and %s" % (args.a, args.b))
    A = np.array([a[p] for p in common])
    B = np.array([b[p] for p in common])

    point = float(A.mean() - B.mean())
    rng = np.random.default_rng(args.seed)
    n = len(common)
    draws = np.empty(args.bootstrap)
    for k in range(args.bootstrap):
        idx = rng.integers(0, n, n)
        draws[k] = A[idx].mean() - B[idx].mean()
    lo, hi = (float(x) for x in np.percentile(draws, [2.5, 97.5]))

    report = {
        "a": args.a, "b": args.b,
        "n_common_prompts": n,
        "n_prompts_seen": {args.a: a_seen, args.b: b_seen},
        "n_prompts_fully_parsed": {args.a: a_kept, args.b: b_kept},
        "win_rate_on_common": {args.a: float(A.mean()), args.b: float(B.mean())},
        "paired_difference": point,
        "paired_difference_ci95": [lo, hi],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "bootstrap": {"replicates": args.bootstrap,
                      "unit": "whole prompt; both presentation orders move together",
                      "paired": True, "seed": args.seed},
        "verdicts_sha256": {args.a: a_sha, args.b: b_sha},
    }
    out = Path("/work/sub_20260914/analysis") / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("a", "b", "n_common_prompts", "win_rate_on_common",
                       "paired_difference", "paired_difference_ci95", "excludes_zero")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
