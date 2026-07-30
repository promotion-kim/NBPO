#!/usr/bin/env python3
"""Aggregate judge verdicts into per-judge anchored surplus s_k = P_k(pi>mu) - 1/2.

Verdicts come from judge_bpo.py run with policy=<trained arm seeds> and
reference=<base seeds>; swap-averaged over both orders. Reports the primary
BPO metric (per-judge surplus) with a Hoeffding 95% margin, plus the worst
surplus (min_k), Nash welfare (sum_k log s_k, IR-valid iff all positive), and
whether individual rationality holds (every surplus > 0).
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--label", default="policy")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = [json.loads(l) for l in args.verdicts.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["valid"]]
    # swap-average per (pid, obj, pseed, rseed)
    pref = defaultdict(list)
    for r in rows:
        pref[(r["prompt_id"], r["objective"], r["policy_seed"], r["ref_seed"])].append(r["policy_win"])
    pref = {k: sum(v) / len(v) for k, v in pref.items()}
    per_obj = defaultdict(list)
    for (pid, obj, ps, rs), v in pref.items():
        per_obj[obj].append(v)
    surplus, margin = {}, {}
    for obj, vals in per_obj.items():
        n = len(vals)
        surplus[obj] = sum(vals) / n - 0.5
        margin[obj] = math.sqrt(math.log(2 * len(per_obj) / 0.05) / (2 * n))
    worst = min(surplus.values())
    ir = all(s > 0 for s in surplus.values())
    nash = sum(math.log(max(s, 1e-6)) for s in surplus.values())
    out = {
        "label": args.label, "n_objectives": len(surplus),
        "surplus": {k: round(v, 4) for k, v in surplus.items()},
        "hoeffding_margin": {k: round(v, 4) for k, v in margin.items()},
        "worst_surplus": round(worst, 4),
        "individual_rationality": ir,
        "nash_welfare_logsum": round(nash, 4),
    }
    print(json.dumps(out, indent=2))
    if args.out:
        args.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
