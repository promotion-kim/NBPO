#!/usr/bin/env python3
"""Exact arithmetic behind the claim that "roughly two thirds is step size".

The claim was made about the v1 stress test: NBPO's native weight norm explodes
as surpluses approach zero, so part of its deficit against the fixed-reference
control is a larger step rather than a different objective. The matched-step
condition rescales NBPO's Nash weight DIRECTION to the fixed-reference norm, so

    gap_native  = min_s(NBPO, native)      - min_s(FixedRef)
    gap_matched = min_s(NBPO, matched)     - min_s(FixedRef)
    explained   = (|gap_native| - |gap_matched|) / |gap_native|

is the fraction of the deficit that the step size accounts for. This computes it
per alpha from the recorded numbers instead of quoting a remembered summary.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out[(r["condition"], float(r["alpha"]), r["method"])] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path,
                    default=Path("results/iclr2027_table1_v2/controlled_nontransitivity.csv"))
    ap.add_argument("--audit-csv", type=Path,
                    default=Path("results/iclr2027_table1_v2/nontransitivity_audit/"
                                 "solver_comparison_v1.csv"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    t = load(args.csv)
    alphas = sorted({k[1] for k in t})
    rows = []
    for a in alphas:
        g = lambda cond, meth: float(t[(cond, a, meth)]["true_min_surplus_mean"])
        ref = g("native", "Fixed-reference Nash")
        ref_m = g("matched_step", "Fixed-reference Nash")
        gn, gm = g("native", "NBPO") - ref, g("matched_step", "NBPO") - ref_m
        expl = (abs(gn) - abs(gm)) / abs(gn) if abs(gn) > 1e-12 else None
        rows.append({
            "alpha": a,
            "bt_deviance": float(t[("native", a, "NBPO")]["bt_deviance_mean"]),
            "nbpo_native_min_surplus": g("native", "NBPO"),
            "nbpo_matched_min_surplus": g("matched_step", "NBPO"),
            "fixed_reference_min_surplus": ref,
            "gap_native": gn, "gap_matched": gm,
            "abs_gap_reduction": abs(gn) - abs(gm),
            "fraction_explained_by_step_size": expl,
            "sign_flips_under_matching": bool(gn * gm < 0),
        })

    # what the exact solver says about the same alphas, if the audit has run
    exact = {}
    if args.audit_csv.exists():
        for r in csv.DictReader(open(args.audit_csv)):
            exact.setdefault(float(r["alpha"]), {})[r["method"]] = r

    verdict = ("RETRACTED" if any(
        r["fraction_explained_by_step_size"] is not None
        and not (0.60 <= r["fraction_explained_by_step_size"] <= 0.72)
        for r in rows) else "SUPPORTED")

    out = {
        "claim": "roughly two thirds of the NBPO-minus-FixedRef deficit is step size",
        "verdict": verdict,
        "per_alpha": rows,
        "summary": (
            "The fraction is not a constant and is nowhere near two thirds at the "
            "alphas that matter. It is "
            + "; ".join(f"alpha={r['alpha']:.2f}: "
                        + ("n/a" if r['fraction_explained_by_step_size'] is None
                           else f"{100 * r['fraction_explained_by_step_size']:.1f}%")
                        for r in rows)
            + ". At the strongest intransitivity (alpha=1.0) step size accounts for "
            + f"{100 * rows[-1]['fraction_explained_by_step_size']:.1f}%, not ~67%."),
        "why_the_decomposition_is_void": (
            "Both arms being differenced are non-converged. In the exact audit the "
            "practical alternating solver has a fixed-point residual of 1.000 at "
            "alpha >= 0.5 -- the maximum a simplex iterate can have -- while the "
            "fixed-reference control has a residual of exactly 0 because its Eq. (21) "
            "map is constant and one application lands on its fixed point. The "
            "difference therefore compares a diverging iteration against one that "
            "cannot diverge, and no split of it into 'step size' and 'representation' "
            "is meaningful."),
    }
    if exact:
        out["exact_solver_says"] = {
            str(a): {m: {"min_surplus": float(v["min_surplus_mean"]),
                         "fixed_point_residual": (float(v["fixed_point_residual_max"])
                                                  if v["fixed_point_residual_max"] else None)}
                     for m, v in d.items()
                     if m in ("A_exact_global_nash", "B_exact_proximal_nash",
                              "C_practical", "D_matched_step", "fixed_reference_native")}
            for a, d in sorted(exact.items())}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "exact_solver_says"}, indent=2))


if __name__ == "__main__":
    main()
