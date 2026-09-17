"""Objective-wise win fractions for Table 3, with the declared bootstrap.

For arm a, objective k and prompt x the judge supplies both orders, so

    P_k(a > c | x) = 1/2 [ v_ac/4 + 1 - v_ca/4 ],

which panel_eval_judge already writes as value_for_a. This averages over prompts
first and then takes the minimum, exactly as the table's caption requires:

    W_k(a) = mean_x P_k(a > c | x),      W_min(a) = min_k W_k(a).

Uncertainty is a whole-prompt paired bootstrap: a replicate resamples PROMPTS
with replacement, shared across arms and objectives, and W_min is recomputed
inside every replicate rather than being bootstrapped as if it were a fixed
coordinate. An interval containing 1/2 is reported as containing 1/2 and is
never read as equivalence.

Only prompts whose every (arm, objective, order) cell resolved enter the common
set, so all rows of the table use one prompt set; the planned and retained
counts are both recorded.
"""
from __future__ import annotations

import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path

import numpy as np

SUB = Path("/work/sub_20260914")
ORDERS = 2


def file_hash(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="directory under objectivewise/")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()

    root = SUB / "objectivewise" / args.tag
    settings = json.loads((root / "complete.json").read_text())
    arms = list(settings["bank"])
    K = int(settings["items_per_prompt"])

    # (arm, item) -> prompt -> order -> value. Both orders are required: the
    # judge has a large first-position bias on this panel and averaging the two
    # is what cancels it, so a prompt with one order is not a measurement.
    raw = defaultdict(lambda: defaultdict(dict))
    status = defaultdict(int)
    for line in (root / "verdicts.jsonl").open():
        if not line.strip():
            continue
        r = json.loads(line)
        status[r["status"]] += 1
        if r["value_for_a"] is None:
            continue
        raw[(r["a"], r["item"])][r["prompt_id"]][int(r["order"])] = float(r["value_for_a"])
    cells, dropped_one_order = defaultdict(dict), 0
    for key, byprompt in raw.items():
        for pid, byorder in byprompt.items():
            if len(byorder) != ORDERS:
                dropped_one_order += 1
                continue
            cells[key][pid] = sum(byorder.values()) / len(byorder)

    # one prompt set for every row and column of the table
    need = [(a, k) for a in arms for k in range(K)]
    missing = [key for key in need if key not in cells]
    if missing:
        raise SystemExit("no verdicts for %s" % missing[:4])
    common = sorted(set.intersection(*[set(cells[key]) for key in need]))
    planned = sorted({p for key in need for p in cells[key]})
    if not common:
        raise SystemExit("no prompt resolved for every arm and objective")

    W = np.zeros((len(arms), K, len(common)))
    for ai, a in enumerate(arms):
        for k in range(K):
            W[ai, k] = [cells[(a, k)][p] for p in common]

    means = W.mean(axis=2)                       # W_k(a)
    wmin = means.min(axis=1)                     # W_min(a)

    rng = np.random.default_rng(args.seed)
    n = len(common)
    boot_w = np.zeros((args.bootstrap, len(arms), K))
    boot_min = np.zeros((args.bootstrap, len(arms)))
    for b in range(args.bootstrap):
        idx = rng.integers(0, n, n)              # shared across arms: paired
        m = W[:, :, idx].mean(axis=2)
        boot_w[b] = m
        boot_min[b] = m.min(axis=1)              # recomputed inside the replicate

    def ci(sample):
        lo, hi = np.percentile(sample, [2.5, 97.5], axis=0)
        return lo, hi

    wlo, whi = ci(boot_w)
    mlo, mhi = ci(boot_min)

    table = {}
    for ai, a in enumerate(arms):
        row = {"wmin": {"value": float(wmin[ai]),
                        "ci": [float(mlo[ai]), float(mhi[ai])],
                        "contains_half": bool(mlo[ai] <= 0.5 <= mhi[ai])}}
        for k in range(K):
            row["w%d" % (k + 1)] = {
                "value": float(means[ai, k]),
                "ci": [float(wlo[ai, k]), float(whi[ai, k])],
                "contains_half": bool(wlo[ai, k] <= 0.5 <= whi[ai, k])}
        table[a] = row

    report = {
        "tag": args.tag, "comparator": settings["comparator"],
        "arms": arms, "objectives": K,
        "prompts": {"planned": len(planned), "common_to_every_cell": len(common),
                    "rule": ("a prompt enters only if every arm and objective "
                             "resolved both orders")},
        "status_counts": dict(status),
        "order_handling": ("both presentation orders averaged per prompt, which is "
                           "what cancels the judge's first-position bias; a prompt "
                           "with only one order is not counted"),
        "prompt_cells_dropped_for_single_order": dropped_one_order,
        "bootstrap": {"replicates": args.bootstrap, "unit": "whole prompt",
                      "paired_across_arms": True,
                      "wmin_recomputed_in_each_replicate": True,
                      "seed": args.seed},
        "aggregation": "average over prompts first, then take the minimum",
        "interval_reading": ("an interval containing .5 is not evidence of "
                             "equivalence"),
        "table": table,
        "verdicts_sha256": settings["verdicts_sha256"],
        "source_sha256": file_hash(__file__),
    }
    (root / "matrix.json").write_text(json.dumps(report, indent=1) + "\n")
    brief = {a: {k: round(v["value"], 4) for k, v in sorted(row.items())}
             for a, row in table.items()}
    print(json.dumps({"prompts": report["prompts"], "table": brief}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
