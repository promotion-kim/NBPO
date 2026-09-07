#!/usr/bin/env python3
"""Emit only the paper tables whose every cell is a real measurement.

The revised contract is negative as much as positive: a table that cannot be
filled honestly is not emitted with placeholders, it is reported as blocked with
the gate that blocks it. ``\\pending`` never reaches a final table.

Currently fillable
    Table 2  feasibility-preserving controlled nontransitivity -- alpha, BT
             deviance, rho*, exact and practical NBPO, fixed-reference, BT-RM,
             exploitability and solver residual. Every number comes from the
             exact float64 audit, and each rho* carries a concavity certificate.

Currently blocked
    Table 1  SafeRLHF natural direct-pairwise alignment -- needs trained
             policies, which the Section 4 go/no-go gate has not released.
    Table 3  general capability -- same gate.

The blocked tables are listed with their gate rather than stubbed, so the
manuscript can drop them cleanly instead of carrying dead cells.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROW_LABELS = {
    "A_exact_global_nash": r"\NBPO{} (exact global Nash)",
    "B_exact_proximal_nash": r"\NBPO{} (exact proximal)",
    "C2_exact_inner_solve": r"\NBPO{} (practical, exact inner solve)",
    "C_practical": r"\NBPO{} (practical, $R$-step map)",
    "fixed_reference_native": r"Fixed-reference Nash",
    "bt_rm_nash": r"BT-RM--Nash",
    "reference": r"Reference $\piref$",
}
ORDER = ["A_exact_global_nash", "B_exact_proximal_nash", "C2_exact_inner_solve",
         "C_practical", "fixed_reference_native", "bt_rm_nash", "reference"]


def load(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out.setdefault(float(r["alpha"]), {})[r["method"]] = r
    return out


def num(v, d=4, signed=True):
    if v in (None, ""):
        return "--"
    x = float(v)
    return f"{x:+.{d}f}" if signed else f"{x:.{d}f}"


def sci(v):
    if v in (None, ""):
        return "--"
    x = float(v)
    return "$0$" if x == 0 else f"${x:.0e}$".replace("e-0", r"\mathrm{e}{-}").replace(
        "e-", r"\mathrm{e}{-}").replace("e+0", r"\mathrm{e}{+}")


def table2(by_alpha, feas_csv, benchmark):
    feas = {float(r["alpha"]): r for r in csv.DictReader(open(feas_csv))}
    L = [r"\begin{tabular}{l r r r r r}", r"\toprule",
         r"$\alpha$ & BT deviance & $\rho^\star$ & min surplus & exploitability "
         r"& solver residual \\", r"\midrule"]
    for alpha in sorted(by_alpha):
        f = feas.get(alpha, {})
        L.append(r"\multicolumn{6}{l}{\textit{$\alpha=%.2f$}} \\" % alpha)
        for m in ORDER:
            r = by_alpha[alpha].get(m)
            if r is None:
                continue
            first = m == ORDER[0]
            L.append(" & ".join([
                r"\quad " + ROW_LABELS[m],
                (num(f.get("bt_deviance"), 4, False) if first else ""),
                (num(f.get("rho_star_mean"), 4) if first else ""),
                num(r["min_surplus_mean"], 4),
                num(r["exploitability_mean"], 4, False),
                sci(r.get("fixed_point_residual_max"))]) + r" \\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit-dir", type=Path,
                    default=Path("results/iclr2027_table1_v2/nontransitivity_audit"))
    ap.add_argument("--benchmark", default="v1")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("results/iclr2027_table1_v2"))
    args = ap.parse_args()

    sc = args.audit_dir / f"solver_comparison_{args.benchmark}.csv"
    fc = args.audit_dir / f"feasibility_{args.benchmark}.csv"
    status = {"generated_from": {"solver_comparison": str(sc), "feasibility": str(fc)},
              "tables": {}}

    if sc.exists() and fc.exists():
        tex = table2(load(sc), fc, args.benchmark)
        if r"\pending" in tex:
            raise SystemExit("refusing to write a table containing \\pending")
        (args.out_dir / f"table2_controlled_nontransitivity_{args.benchmark}.tex"
         ).write_text(tex + "\n")
        status["tables"]["table2_controlled_nontransitivity"] = {
            "state": "FILLABLE", "path": str(
                args.out_dir / f"table2_controlled_nontransitivity_{args.benchmark}.tex"),
            "every_cell_measured": True}
    else:
        status["tables"]["table2_controlled_nontransitivity"] = {
            "state": "BLOCKED", "reason": "the audit has not been run"}

    status["tables"]["table1_saferlhf_natural_alignment"] = {
        "state": "BLOCKED",
        "columns": ["helpfulness", "harmlessness", "worst", "average", "HarmBench"],
        "gate": ("Section 4 go/no-go: controlled feasibility positive, exact NBPO "
                 "correct, a practical solver at fixed-point residual < 1e-4 and "
                 "close to the exact proximal policy, and the SafeRLHF GPM passing "
                 "held-out validation. Then a one-seed reduced smoke, then seeds "
                 "42/43/44."),
        "must_not_be_stubbed": True}
    status["tables"]["table3_general_capability"] = {
        "state": "BLOCKED",
        "rows": ["Base", "NBPO", "Fixed-reference Nash", "BT-RM-Nash", "Game-KS"],
        "gate": "same as Table 1", "must_not_be_stubbed": True}
    status["removed_rows"] = {
        "rows": ["TL;DR", "RACO", "Rewarded Soups", "Panacea", "PROSPER/MaxEntBW",
                 "MOPO", "Game-utilitarian", "BT-RM-utilitarian"],
        "reason": ("not completed and faithful under the current protocol. A row "
                   "that cannot be produced end to end from this pipeline is "
                   "dropped rather than carried as a placeholder.")}
    status["ultrafeedback_role"] = (
        "retained ONLY as a transitive score-induced aggregation control. Its four "
        "aspects are per-completion ordinal ratings, so induced pairwise labels "
        "inherit a total order and are transitive by construction; it may not be "
        "cited as cyclicity evidence.")
    (args.out_dir / "table_status_v5.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
