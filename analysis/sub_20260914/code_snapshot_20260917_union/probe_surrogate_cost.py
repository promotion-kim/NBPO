"""Time the screening surrogate solve against panel size, to predict the full cost.

The 195-prompt aggregate max-min has now run far past two estimates, so this
measures the scaling directly instead of asserting an exponent: the same code
path is run on the first N prompts for several N, and the fitted slope in
log-log says what the full panel costs.

This is a timing probe. It writes no result cell and reads only the verdicts
the screening run already produced.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.experiments.iclr2027_table1_v2 import nontransitivity_feasibility as nf

ROOT = Path("/work/sub_20260914")
OBJECTIVES = ["harmlessness", "helpfulness"]


def load(tags):
    table = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for tag in tags:
        path = ROOT / "judgments" / tag / "judgments.jsonl"
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["status"] != "ok":
                    continue
                table[r["prompt_id"]][r["rubric"]][(r["i"], r["j"])].append(
                    (r["order"], float(r["value_for_i"])))
    return table


def p_hat(entries):
    per_order = defaultdict(list)
    for order, value in entries:
        per_order[order].append(value)
    if 0 not in per_order or 1 not in per_order:
        return None
    return 0.5 * (float(np.mean(per_order[0])) + float(np.mean(per_order[1])))


def tensor(table, prompts, responses=8):
    rows = []
    kept = []
    for pid in prompts:
        mat = np.zeros((len(OBJECTIVES), responses, responses))
        ok = True
        for k, rub in enumerate(OBJECTIVES):
            pairs = table[pid].get(rub, {})
            for i in range(responses):
                for j in range(i + 1, responses):
                    p = p_hat(pairs.get((i, j), []))
                    if p is None:
                        ok = False
                        continue
                    mat[k, i, j] = p - 0.5
                    mat[k, j, i] = 0.5 - p
        if ok:
            rows.append(mat)
            kept.append(pid)
    A = np.stack(rows, axis=1)
    return A, kept


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tags", nargs="+",
                    default=["safe_pilot10_panelA", "safe_rest_panelA_s0",
                             "safe_rest_panelA_s1"])
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 40, 80])
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--out", default=str(ROOT / "analysis/surrogate_cost_probe.json"))
    args = ap.parse_args()

    table = load(args.tags)
    prompts = sorted(table)
    A_full, kept = tensor(table, prompts)
    rows = []
    for n in args.sizes:
        if n > len(kept):
            continue
        A = A_full[:, :n]
        mu = np.full((n, A.shape[2]), 1.0 / A.shape[2])
        beta = np.full(A.shape[0], args.beta)
        d = nf.game_values(A, mu, mu, beta)
        t0 = time.time()
        feas = nf.solve_max_min(A, mu, beta, d)
        seconds = time.time() - t0
        rows.append({"prompts": n, "variables": n * A.shape[2],
                     "seconds": seconds, "rho_star": float(feas["rho_star"])})
        print(json.dumps(rows[-1]), flush=True)

    report = {"complete_prompts_available": len(kept), "rows": rows}
    if len(rows) >= 2:
        x = np.log([r["prompts"] for r in rows])
        y = np.log([r["seconds"] for r in rows])
        slope, intercept = np.polyfit(x, y, 1)
        full = len(kept)
        report["fit"] = {
            "log_log_slope": float(slope),
            "predicted_seconds_for_full_panel": float(np.exp(intercept + slope * np.log(full))),
            "full_panel_prompts": full,
            "reading": ("the slope is the empirical exponent of the aggregate solve in "
                        "panel size; the prediction is for solve_max_min alone and "
                        "excludes the two exact aggregation solves that follow")}
        print(json.dumps(report["fit"]), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
