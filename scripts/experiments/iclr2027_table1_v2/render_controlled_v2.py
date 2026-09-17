#!/usr/bin/env python3
"""Table, figure and gap analysis for the final controlled-v2 benchmark.

Everything is derived from `controlled_v2_raw.json`, the per-cell record, so a
change to how results are presented never costs a re-solve.

The figure is the one the claim rests on: BT deviance on x -- how much of the
payoff a scalar representation provably cannot express -- against
``min surplus / rho*`` on y, which is comparable across alpha precisely because
the construction holds ``rho*`` fixed. Bands are bootstrap confidence intervals
over seeds.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

METHOD_ORDER = ["reference", "exact_global_nash", "exact_proximal_nash",
                "nbpo_direct", "nbpo_rstep_legacy", "fixed_reference_nash",
                "bt_rm_nash", "game_utilitarian", "game_ks", "bt_rm_utilitarian"]
LABEL = {
    "reference": r"Reference $\piref$",
    "exact_global_nash": r"Exact global Nash",
    "exact_proximal_nash": r"Exact proximal Nash",
    "nbpo_direct": r"\NBPO{} (direct inner solve)",
    "nbpo_rstep_legacy": r"\NBPO{} ($R$-step map)",
    "fixed_reference_nash": r"Fixed-reference Nash",
    "bt_rm_nash": r"BT-RM--Nash",
    "game_utilitarian": r"Game-utilitarian",
    "game_ks": r"Game-KS",
    "bt_rm_utilitarian": r"BT-RM--utilitarian",
}
PLOT = ["nbpo_direct", "game_ks", "game_utilitarian", "fixed_reference_nash",
        "bt_rm_nash", "bt_rm_utilitarian", "nbpo_rstep_legacy"]
PLOT_LABEL = {
    "nbpo_direct": "NBPO (adaptive game, Nash)",
    "game_ks": "Game-KS (adaptive game)",
    "game_utilitarian": "Game-utilitarian (adaptive game)",
    "fixed_reference_nash": "Fixed-reference Nash",
    "bt_rm_nash": "BT-RM-Nash (scalar)",
    "bt_rm_utilitarian": "BT-RM-utilitarian (scalar)",
    "nbpo_rstep_legacy": "NBPO, $R$-step map (non-converged)",
}


def load(path: Path):
    raw = json.loads(path.read_text())
    cells = {}
    for r in raw["rows"]:
        cells.setdefault((r["alpha"], r["method"]), []).append(r)
    return raw, cells


def boot_ci(vals, n=5000, seed=0):
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else None, None, None)
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def gaps(cells, alphas, a_name, b_name, seed=0):
    """Paired per-seed gap ``a - b``, bootstrapped over seeds."""
    out = []
    for alpha in alphas:
        A = {r["seed"]: r.get("normalized_min_surplus") for r in cells[(alpha, a_name)]}
        B = {r["seed"]: r.get("normalized_min_surplus") for r in cells[(alpha, b_name)]}
        d = [A[s] - B[s] for s in sorted(set(A) & set(B))
             if A[s] is not None and B[s] is not None]
        m, lo, hi = boot_ci(d, seed=seed)
        dev = statistics.fmean([r["bt_deviance_per_edge"]
                                for r in cells[(alpha, a_name)]])
        out.append({"alpha": alpha, "bt_deviance": dev, "n_seeds": len(d),
                    "gap_mean": m, "gap_ci95_low": lo, "gap_ci95_high": hi})
    return out


def slope(points, xkey="bt_deviance", ykey="gap_mean"):
    """OLS slope of the gap against BT deviance -- does the gap OPEN with it?"""
    x = np.asarray([p[xkey] for p in points], float)
    y = np.asarray([p[ykey] for p in points], float)
    if len(x) < 2 or x.std() == 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path,
                    default=Path("results/iclr2027_table1_v2/controlled_v2"))
    ap.add_argument("--tag", default="",
                    help="suffix selecting one construction family or beta value; "
                         "empty means the primary circulant family")
    args = ap.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""
    raw, cells = load(args.dir / f"controlled_v2_raw{sfx}.json")
    alphas = sorted({a for a, _ in cells})

    # ---------------- gap analysis ----------------
    analysis = {}
    for name, ctrl in (("nbpo_minus_fixed_reference", "fixed_reference_nash"),
                       ("nbpo_minus_bt_rm_nash", "bt_rm_nash"),
                       ("nbpo_minus_bt_rm_utilitarian", "bt_rm_utilitarian"),
                       ("nbpo_minus_game_utilitarian", "game_utilitarian"),
                       ("nbpo_minus_game_ks", "game_ks")):
        pts = gaps(cells, alphas, "nbpo_direct", ctrl)
        analysis[name] = {"per_alpha": pts,
                          "slope_vs_bt_deviance": slope(pts),
                          "gap_at_max_alpha": pts[-1]}
    # the pre-registered success condition, evaluated rather than asserted
    fx = analysis["nbpo_minus_fixed_reference"]["gap_at_max_alpha"]
    bt = analysis["nbpo_minus_bt_rm_nash"]["gap_at_max_alpha"]
    analysis["matched_controls_degrade_relative_to_nbpo"] = bool(
        fx["gap_ci95_low"] is not None and fx["gap_ci95_low"] > 0
        and bt["gap_ci95_low"] is not None and bt["gap_ci95_low"] > 0)
    analysis["condition"] = (
        "the pre-registered condition is that the matched controls DEGRADE "
        "relative to converged practical NBPO as intransitivity rises; it is "
        "checked as a strictly positive bootstrap lower bound on the paired gap "
        "at the largest alpha, for BOTH matched controls")
    (args.dir / f"controlled_v2_gaps{sfx}.json").write_text(
        json.dumps(analysis, indent=2))

    # ---------------- LaTeX ----------------
    L = [r"\begin{tabular}{l r r r r r r}", r"\toprule",
         r"$\alpha$ & BT dev. & $\rho^\star$ & min surplus & "
         r"$\min s/\rho^\star$ & exploitability & solver resid. \\", r"\midrule"]
    for alpha in alphas:
        dev = statistics.fmean([r["bt_deviance_per_edge"]
                                for r in cells[(alpha, "nbpo_direct")]])
        rho = statistics.fmean([r["rho_star"] for r in cells[(alpha, "nbpo_direct")]])
        L.append(r"\multicolumn{7}{l}{\textit{$\alpha=%.2f$}} \\" % alpha)
        for meth in METHOD_ORDER:
            rs = cells.get((alpha, meth), [])
            ok = [r for r in rs if r.get("status") == "ok"]
            if not ok:
                continue
            f = lambda k: [r[k] for r in ok if r.get(k) is not None]
            ms, nm = f("min_surplus"), f("normalized_min_surplus")
            ex = f("exploitability")
            res = [r for r in (f("extra_map_residual") or []) if r is not None]
            nc = "" if all(r.get("converged") for r in ok) else r"$^{\dagger}$"
            L.append(" & ".join([
                r"\quad " + LABEL[meth] + nc,
                (f"{dev:.4f}" if meth == METHOD_ORDER[0] else ""),
                (f"{rho:+.4f}" if meth == METHOD_ORDER[0] else ""),
                (f"{statistics.fmean(ms):+.4f}" if ms else "--"),
                (f"{statistics.fmean(nm):+.3f}" if nm else "--"),
                (f"{statistics.fmean(ex):.4f}" if ex else "--"),
                (f"{max(res):.0e}" if res else "--")]) + r" \\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(L)
    if r"\pending" in tex:
        raise SystemExit("refusing to emit a table containing \\pending")
    (args.dir / f"controlled_v2_generated{sfx}.tex").write_text(tex + "\n")

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for meth in PLOT:
        xs, ms, los, his = [], [], [], []
        for alpha in alphas:
            rs = cells[(alpha, meth)]
            v = [r.get("normalized_min_surplus") for r in rs]
            m, lo, hi = boot_ci(v)
            if m is None:
                continue
            xs.append(statistics.fmean([r["bt_deviance_per_edge"] for r in rs]))
            ms.append(m)
            los.append(lo if lo is not None else m)
            his.append(hi if hi is not None else m)
        style = dict(marker="o", lw=2.0)
        if meth == "nbpo_rstep_legacy":
            style.update(ls=":", lw=1.4, marker="x", color="0.45")
        elif meth.startswith("bt_rm") or meth.startswith("fixed"):
            style.update(ls="--")
        ax.plot(xs, ms, label=PLOT_LABEL[meth], **style)
        ax.fill_between(xs, los, his, alpha=0.15,
                        color=ax.lines[-1].get_color(), lw=0)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("Bradley-Terry deviance per edge  (payoff a scalar representation cannot express)")
    ax.set_ylabel(r"min surplus / $\rho^\star$")
    ax.set_title(r"Feasibility-preserving controlled nontransitivity ($\rho^\star$ held fixed)")
    lo = min(min(ms) for ms in [[boot_ci([r.get("normalized_min_surplus")
                                          for r in cells[(al, m)]])[0] or 0
                                for al in alphas] for m in PLOT])
    finite = [v for m in PLOT for v in
              [boot_ci([r.get("normalized_min_surplus") for r in cells[(al, m)]])[0]
               for al in alphas] if v is not None]
    ax.set_ylim(min(-0.05, min(finite) - 0.05), max(1.02, max(finite) + 0.05))
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.dir / f"controlled_v2_gap_figure{sfx}.pdf")
    fig.savefig(args.dir / f"controlled_v2_gap_figure{sfx}.png", dpi=170)

    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {"slope_vs_bt_deviance": v.get("slope_vs_bt_deviance"),
                           "gap_at_max_alpha": v.get("gap_at_max_alpha")})
                      for k, v in analysis.items()}, indent=2))


if __name__ == "__main__":
    main()
