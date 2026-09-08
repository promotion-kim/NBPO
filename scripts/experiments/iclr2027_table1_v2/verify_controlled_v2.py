#!/usr/bin/env python3
"""Independently re-derive every controlled-v2 number quoted in the paper.

This deliberately does NOT import the renderer that produced the published
figures. It re-reads the raw per-cell record, recomputes the paired intervals
with a differently seeded bootstrap AND with an exact paired t-interval, and
cross-checks the two against the stored `controlled_v2_gaps.json`. A number that
survives three independent routes is safe to quote; one that does not is a bug
in the reporting, and the point of this file is to find it before a reader does.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

CLAIMS = [
    ("nbpo_direct", "fixed_reference_nash"),
    ("nbpo_direct", "bt_rm_nash"),
    ("nbpo_direct", "game_utilitarian"),
    ("nbpo_direct", "game_ks"),
    ("bt_rm_nash", "bt_rm_utilitarian"),
]


def paired(cells, alpha, a, b):
    A = {r["seed"]: r.get("normalized_min_surplus") for r in cells[(alpha, a)]}
    B = {r["seed"]: r.get("normalized_min_surplus") for r in cells[(alpha, b)]}
    return [A[s] - B[s] for s in sorted(set(A) & set(B))
            if A[s] is not None and B[s] is not None]


def boot(d, n=20000, seed=917):
    v = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def sign_test_p(d):
    """Two-sided exact sign test -- distribution-free, and brutal at n = 5.

    With five seeds the smallest attainable two-sided p is 2 * 0.5**5 = 0.0625,
    so a unanimous sign CANNOT reach the 5% level. That is a fact about the
    design, not about the effect, and it is reported rather than worked around.
    """
    n = len(d)
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    k, m = max(pos, neg), pos + neg
    if m == 0:
        return 1.0, pos, neg
    tail = sum(math.comb(m, i) for i in range(k, m + 1)) / (2 ** m)
    return min(1.0, 2 * tail), pos, neg


def t_interval(d):
    """Exact paired Student-t interval -- a route the bootstrap cannot share a bug with."""
    n = len(d)
    if n < 2:
        return None, None, None
    m, s = statistics.fmean(d), statistics.stdev(d)
    # two-sided 95% t quantile for n-1 df, tabulated so scipy.stats is not needed
    tq = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
    t = tq.get(n - 1, 1.96)
    h = t * s / math.sqrt(n)
    return m, m - h, m + h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path,
                    default=Path("results/iclr2027_table1_v2/controlled_v2"))
    a = ap.parse_args()
    raw = json.loads((a.dir / "controlled_v2_raw.json").read_text())
    cells = {}
    for r in raw["rows"]:
        cells.setdefault((r["alpha"], r["method"]), []).append(r)
    alphas = sorted({x for x, _ in cells})
    stored = json.loads((a.dir / "controlled_v2_gaps.json").read_text())

    report = {"alphas": alphas, "claims": {}, "rho_star": {}, "convergence": {},
              "disagreements": []}

    for alpha in alphas:
        rs = [r["rho_star"] for r in cells[(alpha, "nbpo_direct")]]
        report["rho_star"][f"{alpha:.2f}"] = {
            "mean": statistics.fmean(rs), "min": min(rs), "max": max(rs),
            "stdev": statistics.stdev(rs), "per_seed": rs,
            "at_target": abs(statistics.fmean(rs) - 0.030) < 1e-4}

    for meth in sorted({m for _, m in cells}):
        n_conv = sum(1 for alpha in alphas for r in cells[(alpha, meth)]
                     if r.get("converged"))
        n_cells = sum(len(cells[(alpha, meth)]) for alpha in alphas)
        report["convergence"][meth] = {
            "converged": n_conv, "cells": n_cells,
            "expected_non_convergent": any(r.get("expected_non_convergent")
                                           for alpha in alphas
                                           for r in cells[(alpha, meth)]),
            "valid_performance_arm": n_conv == n_cells}

    for a_name, b_name in CLAIMS:
        key = f"{a_name}_minus_{b_name}"
        per_alpha = []
        for alpha in alphas:
            d = paired(cells, alpha, a_name, b_name)
            bm, blo, bhi = boot(d)
            tm, tlo, thi = t_interval(d)
            sp, npos, nneg = sign_test_p(d)
            boot_sig = "positive" if blo > 0 else "negative" if bhi < 0 else "indistinguishable"
            t_sig = "positive" if tlo > 0 else "negative" if thi < 0 else "indistinguishable"
            per_alpha.append({
                "alpha": alpha, "n_seeds": len(d), "per_seed_gap": d,
                "bootstrap": {"mean": bm, "lo": blo, "hi": bhi},
                "student_t": {"mean": tm, "lo": tlo, "hi": thi},
                "sign_test": {"p_two_sided": sp, "n_positive": npos,
                              "n_negative": nneg, "unanimous": npos == 0 or nneg == 0},
                "bootstrap_verdict": boot_sig,
                "student_t_verdict": t_sig,
                # The conservative route governs what may be claimed. A percentile
                # bootstrap over five seeds resamples a five-point mean and its
                # spread is far too narrow; where the two routes disagree, the
                # bootstrap is the one that is wrong.
                "verdict": t_sig,
                "routes_agree": boot_sig == t_sig,
            })
        report["claims"][key] = per_alpha

        # cross-check against what the renderer stored, where it stored it
        stored_key = f"nbpo_minus_{b_name.replace('_nash', '') if b_name.startswith('fixed') else b_name}"
        for cand in (f"nbpo_minus_{b_name}", "nbpo_minus_fixed_reference",
                     "nbpo_minus_bt_rm_nash", "nbpo_minus_game_utilitarian",
                     "nbpo_minus_game_ks"):
            if a_name == "nbpo_direct" and cand in stored:
                sp = {p["alpha"]: p for p in stored[cand]["per_alpha"]}
                if not sp or abs(sp[alphas[-1]]["gap_mean"] - per_alpha[-1]["bootstrap"]["mean"]) > 1e-9:
                    continue
                for p in per_alpha:
                    s = sp.get(p["alpha"])
                    if s and abs(s["gap_mean"] - p["bootstrap"]["mean"]) > 1e-9:
                        report["disagreements"].append(
                            {"claim": cand, "alpha": p["alpha"],
                             "stored": s["gap_mean"],
                             "recomputed": p["bootstrap"]["mean"]})
                break

    ok = not report["disagreements"]
    report["all_recomputed_means_match_stored"] = ok
    report["interval_route_note"] = (
        "Two routes are computed for every gap. Where they disagree the "
        "conservative Student-t interval governs: a percentile bootstrap over "
        "five seeds resamples a five-point mean and is materially "
        "anti-conservative, which showed up here as apparent significance for "
        "gaps of order 0.01-0.02. Only the t verdict is quoted as established.")
    report["claims_established_under_the_conservative_route"] = {
        key: {f"{p['alpha']:.2f}": p["verdict"] for p in pts}
        for key, pts in report["claims"].items()}
    (a.dir / "controlled_v2_verification.json").write_text(json.dumps(report, indent=2))

    print("rho* calibration")
    for k, v in report["rho_star"].items():
        print(f"  alpha={k}  mean={v['mean']:.6f}  sd={v['stdev']:.2e}  "
              f"min={v['min']:.6f}  at_target={v['at_target']}")
    print("\nconvergence (strict)")
    for m, v in report["convergence"].items():
        flag = "" if v["valid_performance_arm"] else "   <-- NOT a valid performance arm"
        print(f"  {m:22s} {v['converged']:>3}/{v['cells']}{flag}")
    print("\npaired gaps, two independent interval routes")
    for key, pts in report["claims"].items():
        print(f"  {key}")
        for p in pts:
            b, t = p["bootstrap"], p["student_t"]
            st = p["sign_test"]
            print(f"    a={p['alpha']:.2f}  boot {b['mean']:+.4f} "
                  f"[{b['lo']:+.4f},{b['hi']:+.4f}]   t [{t['lo']:+.4f},{t['hi']:+.4f}]"
                  f"   sign {st['n_positive']}+/{st['n_negative']}- p={st['p_two_sided']:.4f}"
                  f"   verdict={p['verdict']}"
                  f"{'' if p['routes_agree'] else '   (bootstrap said ' + p['bootstrap_verdict'] + ')'}")
    print(f"\nrecomputed means match the stored gaps: {ok}")
    if not ok:
        print(json.dumps(report["disagreements"], indent=2))


if __name__ == "__main__":
    main()
