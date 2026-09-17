"""The Delta column: a paired difference in seed-wise minimum versus NBPO.

tab:stress_results defines Delta as this row minus NBPO in the minimum-over-
objectives statistic, with a 95% prompt interval. Comparing the two rows'
marginal intervals is not that: the arms are judged on the same prompts against
the same reference bank, so the comparison has to be paired prompt by prompt.

The bootstrap resamples whole prompts -- every objective and both orders of a
prompt move together -- and recomputes each arm's minimum over objectives
inside every replicate before differencing, rather than differencing two point
estimates. Prompts are restricted to the intersection where both arms have
every scheduled verdict parsed, and that intersection is reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SUB = Path("/work/sub_20260914")


def file_hash(p):
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def per_prompt(tag, objectives, refs_per_prompt=4):
    """prompt -> [mean value per objective], only for fully parsed prompts."""
    path = SUB / "fresh_eval" / tag / "verdicts.jsonl"
    table = defaultdict(lambda: defaultdict(list))
    for line in path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["status"] != "ok":
            continue
        table[r["prompt_id"]][r["rubric"]].append(float(r["value_for_fresh"]))
    out = {}
    need = refs_per_prompt * 2
    for pid, per_rubric in table.items():
        if all(len(per_rubric.get(o, [])) == need for o in objectives):
            out[pid] = [float(np.mean(per_rubric[o])) for o in objectives]
    return out, file_hash(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--row-tag", required=True, help="fresh_eval tag of the row")
    ap.add_argument("--reference-tag", default="safe_nbpo_s42",
                    help="fresh_eval tag of NBPO, the row Delta is measured against")
    ap.add_argument("--objectives", nargs="+",
                    default=["harmlessness", "helpfulness"])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--out", required=True, help="path under analysis/")
    args = ap.parse_args()

    row, row_sha = per_prompt(args.row_tag, args.objectives)
    ref, ref_sha = per_prompt(args.reference_tag, args.objectives)
    common = sorted(set(row) & set(ref))
    if not common:
        raise SystemExit("no prompts shared between %s and %s" % (args.row_tag,
                                                                  args.reference_tag))
    A = np.array([row[p] for p in common])
    B = np.array([ref[p] for p in common])

    def statistic(idx):
        a = A[idx].mean(axis=0).min()
        b = B[idx].mean(axis=0).min()
        return a - b

    point = statistic(np.arange(len(common)))
    rng = np.random.default_rng(args.seed)
    draws = np.array([statistic(rng.integers(0, len(common), len(common)))
                      for _ in range(args.bootstrap)])
    lo, hi = np.percentile(draws, [2.5, 97.5])

    report = {
        "row_tag": args.row_tag, "reference_tag": args.reference_tag,
        "objectives": args.objectives,
        "n_common_prompts": len(common),
        "n_row_prompts": len(row), "n_reference_prompts": len(ref),
        "row_min_on_common": float(A.mean(axis=0).min()),
        "reference_min_on_common": float(B.mean(axis=0).min()),
        "delta_min": float(point),
        "delta_min_ci95": [float(lo), float(hi)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "bootstrap": {"replicates": args.bootstrap,
                      "unit": "whole prompt; all objectives and orders resampled together",
                      "minimum_recomputed_per_replicate": True,
                      "paired": True},
        "sources_sha256": {args.row_tag: row_sha, args.reference_tag: ref_sha},
    }
    out = SUB / "analysis" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("row_tag", "n_common_prompts", "row_min_on_common",
                       "reference_min_on_common", "delta_min", "delta_min_ci95",
                       "excludes_zero")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
