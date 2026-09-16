"""Is the arm-minus-base win-rate gap a length effect?

The trained arms answer at roughly the length of the untrained base but far
above the released baseline answers they are judged against, and pairwise judges
are known to prefer longer text. That threatens the absolute win rates, and it
would threaten the arm-minus-base difference too if the arms were systematically
longer than the base.

This conditions the paired arm-minus-base difference on length instead of
assuming it away. Per prompt we take each policy's own response length and form
d = log(arm tokens / base tokens); the comparison is then recomputed inside
quartiles of d and on the subset where the two lengths are within ten percent of
each other. If the gap survives on prompts where the arm is not longer than the
base, length does not explain it. The bootstrap resamples whole prompts, as
elsewhere.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

RESP = Path("/work/sub_20260914/responses")
EVAL = Path("/work/sub_20260914/eval_pairwise")


def values(tag, orders=2):
    table = {}
    for line in (EVAL / tag / "verdicts.jsonl").open():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["status"] != "ok":
            table[r["prompt_id"]] = None
            continue
        table.setdefault(r["prompt_id"], [])
        if table[r["prompt_id"]] is not None:
            table[r["prompt_id"]].append(float(r["value_for_arm"]))
    return {p: float(np.mean(v)) for p, v in table.items() if v and len(v) == orders}


def tokens(tag):
    out = {}
    for line in (RESP / tag / "responses.jsonl").open():
        if line.strip():
            e = json.loads(line)
            out[e["prompt_id"]] = int(e["n_tokens"])
    return out


def paired(A, B, rng, reps):
    a, b = np.asarray(A), np.asarray(B)
    point = float(a.mean() - b.mean())
    n = len(a)
    draws = np.empty(reps)
    for k in range(reps):
        idx = rng.integers(0, n, n)
        draws[k] = a[idx].mean() - b[idx].mean()
    lo, hi = (float(x) for x in np.percentile(draws, [2.5, 97.5]))
    return {"n": n, "difference": point, "ci95": [lo, hi],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-tag", required=True, help="eval_pairwise tag of the arm")
    ap.add_argument("--base-tag", required=True, help="eval_pairwise tag of the base")
    ap.add_argument("--arm-responses", required=True)
    ap.add_argument("--base-responses", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    va, vb = values(args.arm_tag), values(args.base_tag)
    ta, tb = tokens(args.arm_responses), tokens(args.base_responses)
    common = sorted(set(va) & set(vb) & set(ta) & set(tb))
    if not common:
        raise SystemExit("no prompt has both verdicts and both lengths")
    d = np.array([math.log(max(ta[p], 1) / max(tb[p], 1)) for p in common])
    A = np.array([va[p] for p in common])
    B = np.array([vb[p] for p in common])
    rng = np.random.default_rng(args.seed)

    report = {"arm": args.arm_tag, "base": args.base_tag, "n_common_prompts": len(common),
              "length_log_ratio": {"mean": float(d.mean()), "median": float(np.median(d)),
                                   "p10": float(np.percentile(d, 10)),
                                   "p90": float(np.percentile(d, 90))},
              "median_tokens": {"arm": float(np.median([ta[p] for p in common])),
                                "base": float(np.median([tb[p] for p in common]))},
              "overall": paired(A, B, np.random.default_rng(args.seed), args.bootstrap)}
    edges = np.percentile(d, [0, 25, 50, 75, 100])
    quart = []
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        m = (d >= lo) & (d <= hi) if i == 3 else (d >= lo) & (d < hi)
        if m.sum() < 20:
            quart.append({"log_ratio_range": [float(lo), float(hi)], "n": int(m.sum()),
                          "skipped": "fewer than twenty prompts"})
            continue
        q = paired(A[m], B[m], np.random.default_rng(args.seed + 1 + i), args.bootstrap)
        q["log_ratio_range"] = [float(lo), float(hi)]
        quart.append(q)
    report["by_length_quartile"] = quart
    near = np.abs(d) <= math.log(1.10)
    report["within_ten_percent"] = (
        paired(A[near], B[near], np.random.default_rng(args.seed + 9), args.bootstrap)
        if near.sum() >= 20 else {"n": int(near.sum()), "skipped": "too few prompts"})
    shorter = d <= 0
    report["arm_no_longer_than_base"] = (
        paired(A[shorter], B[shorter], np.random.default_rng(args.seed + 17), args.bootstrap)
        if shorter.sum() >= 20 else {"n": int(shorter.sum()), "skipped": "too few prompts"})
    out = Path("/work/sub_20260914/analysis") / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
