#!/usr/bin/env python3
"""Regenerate every paper table fragment from the stored artifacts.

Each fragment is a **tabular body only** -- rows between \\midrule and
\\bottomrule -- so it is \\input inside the manuscript's existing table
environment. Captions and labels stay in the manuscript, and no fragment can
create a nested table or a duplicate label.

One command regenerates all of them:

    python -m scripts.experiments.iclr2027_table1_v2.export_paper_tables

Every fragment is accompanied by an entry in ``result_manifest.json`` recording
which artifact, run, seeds and aggregation produced it, and whether the cell is
completed, partial or pending.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

OUT = Path("nbpo_iclr/generated")
RES = Path("results/iclr2027_table1_v2")
EXP = Path("experiments/iclr2027_table1_v2")
MANIFEST: dict = {}


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16] if Path(p).exists() else None


# Each fragment is a COMPLETE tabular environment -- column spec, rules, header
# and body -- and the manuscript's table/table* environment supplies only the
# caption and label. Emitting a bare body instead and \input-ing it between
# \midrule and \bottomrule breaks TeX's alignment scanner ("Misplaced \noalign",
# "Misplaced \omit"), and emitting the whole table environment would duplicate
# captions and labels. This is the fixed format.
HEADERS: dict = {}


def emit(name: str, rows: list, sources: list, status: str, note: str = "",
         seeds=None, aggregation: str = ""):
    OUT.mkdir(parents=True, exist_ok=True)
    colspec, header = HEADERS[name]
    body = ["\\begin{tabular}{" + colspec + "}", "\\toprule", header, "\\midrule",
            *rows, "\\bottomrule", "\\end{tabular}"]
    (OUT / f"{name}.tex").write_text("\n".join(body) + "\n")
    MANIFEST[name] = {
        "fragment": str(OUT / f"{name}.tex"),
        "status": status, "note": note,
        "seeds": seeds, "aggregation": aggregation,
        "sources": [{"path": str(s), "sha256_16": sha(Path(s))} for s in sources],
    }
    print(f"  {name:34s} {status:10s} {len(rows)} rows")


HEADERS.update({
 "controlled_compact": ("crrrrrrrr",
   r"$\alpha$ & BT dev. & \NBPO{} & Game-KS & Game-util. & Fixed-ref. Nash & "
   r"BT-RM--Nash & BT-RM--util. & legacy $R$-step$^\dagger$\\"),
 "controlled_full": ("lrrrrrr",
   r"method & min surplus & sd & $\min s/\rho^\star$ & sd & exploitability & inner resid.\\"),
 "solver_audit": ("lrrrr",
   r"solution & min surplus & $/\rho^\star$ & inner residual & TV to exact prox.\\"),
 "controlled_robustness": ("llrrrrr",
   r"cycle family & $\beta$ & seeds & \NBPO{}$-$FixedRef ($t_{95}$) & "
   r"\NBPO{}$-$BT-RM ($t_{95}$) & min $\rho^\star$ & converged\\"),
 "data_audit": ("lrrrrrll",
   r"dataset & rows & prompts & responses & obs.\ triangles & human conflict & "
   r"supervision & role\\"),
 "gpm_bt": ("llrrrrrrrrr",
   r"model & objective & \multicolumn{3}{c}{per-seed (mean of 3)} & $T$ & "
   r"\multicolumn{4}{c}{calibrated 3-seed ensemble} \\" "\n"
   r"\cmidrule(lr){3-5}\cmidrule(lr){7-10}" "\n"
   r" & & NLL & bal.\ acc & AUC & & NLL & bal.\ acc & AUC & ECE & ens.\ sd\\"),
 "pool_pilot": ("lrrrrrrrrrrr",
   r"geom. & dup.\ rate & entropy & ens.\ sd & GPM/BT sign & model conflict & "
   r"pred.\ cycles & split-half $r$ & $\rho$ & sign agr. & dual evals & solve (s)\\"),
 "solver_scaling": ("rrrrrrrrrr",
   r"prompts & pool & workers & solve (s) & dual evals & s/eval & proj.\ KKT & "
   r"inner resid. & peak RSS (MB) & artifact (s)\\"),
 "neural_realization": ("lrrrrrrrrrr",
   r"row & target RMS & $p_{10}$ & $p_{90}$ & $\|\lambda\|_1$ & proj.\ KKT & "
   r"inner resid. & identity & norm.\ MSE & sign agr. & Pearson\\"),
})


def f(x, d=4, signed=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    return f"${x:+.{d}f}$" if signed else f"${x:.{d}f}$"


def sci(x):
    if x is None:
        return "--"
    x = float(x)
    if x == 0:
        return "$0$"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    return f"${m:.0f}\\times10^{{{e}}}$" if abs(m - 1) > 0.3 else f"$10^{{{e}}}$"


def load_raw(tag=""):
    sfx = f"_{tag}" if tag else ""
    p = RES / "controlled_v2" / f"controlled_v2_raw{sfx}.json"
    raw = json.loads(p.read_text())
    cells = {}
    for r in raw["rows"]:
        cells.setdefault((r["alpha"], r["method"]), []).append(r)
    return raw, cells, p


def t_interval(d):
    n = len(d)
    if n < 2:
        return None, None, None
    m, s = statistics.fmean(d), statistics.stdev(d)
    tq = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
    h = tq.get(n - 1, 1.96) * s / math.sqrt(n)
    return m, m - h, m + h


LABEL = {"nbpo_direct": r"\NBPO{} direct", "game_ks": "Game-KS",
         "game_utilitarian": "Game-util.", "fixed_reference_nash": "Fixed-ref.\\ Nash",
         "bt_rm_nash": "BT-RM--Nash", "bt_rm_utilitarian": "BT-RM--util.",
         "nbpo_rstep_legacy": r"legacy $R$-step$^\dagger$",
         "exact_global_nash": "Exact global Nash",
         "exact_proximal_nash": "Exact proximal Nash", "reference": r"Reference $\piref$"}
ORDER = ["nbpo_direct", "game_ks", "game_utilitarian", "fixed_reference_nash",
         "bt_rm_nash", "bt_rm_utilitarian", "nbpo_rstep_legacy"]


# --------------------------------------------------------------------------
def controlled_compact():
    _, cells, p = load_raw()
    alphas = sorted({a for a, _ in cells})
    rows = []
    for a in alphas:
        dev = statistics.fmean([r["bt_deviance_per_edge"] for r in cells[(a, "nbpo_direct")]])
        vals = []
        for m in ORDER:
            v = [r.get("normalized_min_surplus") for r in cells[(a, m)]
                 if r.get("normalized_min_surplus") is not None]
            vals.append(f"${statistics.fmean(v):.3f}$" if v else "--")
        rows.append(f"${a:.2f}$ & ${dev:.4f}$ & " + " & ".join(vals) + r"\\")
    emit("controlled_compact", rows, [p], "completed",
         "mean over 5 independent controlled seeds of min_k s_k / rho*",
         seeds=[0, 1, 2, 3, 4], aggregation="mean over seeds")


def controlled_full():
    _, cells, p = load_raw()
    alphas = sorted({a for a, _ in cells})
    rows = []
    for a in alphas:
        dev = statistics.fmean([r["bt_deviance_per_edge"] for r in cells[(a, "nbpo_direct")]])
        rho = statistics.fmean([r["rho_star"] for r in cells[(a, "nbpo_direct")]])
        rows.append(r"\multicolumn{7}{l}{\textit{$\alpha=%.2f$, $\rho^\star=%.4f$, "
                    r"BT dev.\ $=%.4f$}}\\" % (a, rho, dev))
        for m in ORDER:
            rs = [r for r in cells.get((a, m), []) if r.get("status") == "ok"]
            if not rs:
                continue
            g = lambda k: [r[k] for r in rs if r.get(k) is not None]
            ms, nm, ex = g("min_surplus"), g("normalized_min_surplus"), g("exploitability")
            res = g("extra_map_residual")
            rows.append(" & ".join([
                r"\quad " + LABEL[m],
                f"${statistics.fmean(ms):+.4f}$" if ms else "--",
                (f"${statistics.fmean(ms):+.4f}$" if False else
                 f"${statistics.stdev(ms):.4f}$" if len(ms) > 1 else "--"),
                f"${statistics.fmean(nm):.3f}$" if nm else "--",
                f"${statistics.stdev(nm):.3f}$" if len(nm) > 1 else "--",
                f"${statistics.fmean(ex):.4f}$" if ex else "--",
                sci(max(res)) if res else "--"]) + r"\\")
        rows.append(r"\addlinespace")
    emit("controlled_full", rows, [p], "completed",
         "every method at every alpha; mean and sd over the 5 controlled seeds",
         seeds=[0, 1, 2, 3, 4], aggregation="mean +/- sd over seeds")


def solver_audit():
    """The ORIGINAL v1 family's solver decomposition -- a different experiment
    from controlled-v2, and labelled as such."""
    p = RES / "nontransitivity_audit" / "solver_comparison_v1.csv"
    rows_in = [r for r in csv.DictReader(open(p)) if float(r["alpha"]) == 1.0]
    order = ["A_exact_global_nash", "B_exact_proximal_nash", "C2_exact_inner_solve",
             "C_practical", "fixed_reference_nash", "bt_rm_nash"]
    lab = {"A_exact_global_nash": "Exact global Nash",
           "B_exact_proximal_nash": "Exact proximal Nash",
           "C2_exact_inner_solve": r"\NBPO{} direct inner solve",
           "C_practical": r"Legacy $R$-step map",
           "fixed_reference_nash": "Fixed-reference Nash",
           "bt_rm_nash": "BT-RM--Nash"}
    by = {r["method"]: r for r in rows_in}
    rows = []
    for m in order:
        r = by.get(m)
        if r is None:
            continue
        g = lambda k: (float(r[k]) if r.get(k) not in (None, "", "NA") else None)
        rows.append(" & ".join([
            lab[m], f(g("min_surplus_mean"), 4, True),
            f(g("min_surplus_over_rho_star_mean"), 3, True),
            sci(g("fixed_point_residual_max")),
            f(g("tv_to_exact_B_mean"), 3)]) + r"\\")
    emit("solver_audit", rows, [p], "completed",
         "ORIGINAL v1 controlled family at alpha=1, 5 seeds -- a different "
         "construction from controlled-v2; TV is to that family's exact proximal "
         "solution, not to a global Nash point",
         seeds=[0, 1, 2, 3, 4], aggregation="mean over seeds")


def controlled_robustness():
    rows = []
    src = []
    for tag, family, beta in (("", "Circulant (primary)", 0.25),
                              ("projected", "Projected skew-symmetric", 0.25),
                              ("projected_beta0.1", "Projected skew-symmetric", 0.10),
                              ("projected_beta0.5", "Projected skew-symmetric", 0.50)):
        try:
            _, cells, p = load_raw(tag)
        except FileNotFoundError:
            continue
        src.append(p)
        alphas = sorted({a for a, _ in cells})
        n_seeds = len(cells[(alphas[0], "nbpo_direct")])
        gaps = {}
        for ctrl in ("fixed_reference_nash", "bt_rm_nash"):
            A = {r["seed"]: r.get("normalized_min_surplus")
                 for r in cells[(alphas[-1], "nbpo_direct")]}
            B = {r["seed"]: r.get("normalized_min_surplus")
                 for r in cells[(alphas[-1], ctrl)]}
            d = [A[s] - B[s] for s in sorted(set(A) & set(B))
                 if A[s] is not None and B[s] is not None]
            gaps[ctrl] = t_interval(d)
        rho = [statistics.fmean([r["rho_star"] for r in cells[(a, "nbpo_direct")]])
               for a in alphas]
        conv = sum(1 for a in alphas for m in ORDER[:6] for r in cells[(a, m)]
                   if r.get("converged"))
        tot = sum(1 for a in alphas for m in ORDER[:6] for _ in cells[(a, m)])
        fx, bt = gaps["fixed_reference_nash"], gaps["bt_rm_nash"]
        rows.append(" & ".join([
            family, f"${beta:.2f}$", f"{n_seeds}",
            f"${fx[0]:+.4f}$ $[{fx[1]:+.4f},{fx[2]:+.4f}]$",
            f"${bt[0]:+.4f}$ $[{bt[1]:+.4f},{bt[2]:+.4f}]$",
            f"${min(rho):.4f}$", f"${conv}/{tot}$"]) + r"\\")
    emit("controlled_robustness", rows, src, "completed",
         "NBPO minus control at alpha=1, paired per seed, exact Student-t 95% "
         "interval (NOT a bootstrap interval). Convergence counts exclude the "
         "legacy R-step ablation.",
         aggregation="paired per-seed gap at alpha=1, Student-t interval")


def data_audit():
    m = json.loads((EXP / "saferlhf_splits" / "split_manifest.json").read_text())
    tri = sum(v["graph"]["per_objective"][o]["three_cycles"]
              for v in m["splits"].values() for o in ("helpfulness", "harmlessness"))
    conf = m["splits"]["test"]["label_balance"]["helpfulness_harmlessness_conflict_rate"]
    rows = [
        " & ".join([r"\textsc{SafeRLHF}", f"${m['total_rows']:,}$".replace(",", "{,}"),
                    f"${m['total_prompts']:,}$".replace(",", "{,}"), "$143{,}668$",
                    f"${tri}$", f"${100*conf:.2f}\\%$",
                    "direct pairwise helpful/safe", "natural objective conflict"]) + r"\\",
        " & ".join([r"\textsc{UltraFeedback}", "$63{,}967$", "$63{,}967$", "$4$/prompt",
                    "n/a (score-induced)", "n/a",
                    "four ordinal attributes", "transitive aggregation control"]) + r"\\",
    ]
    emit("data_audit", rows, [EXP / "saferlhf_splits" / "split_manifest.json"],
         "completed",
         "observed HUMAN triangles across all three splits and both objectives; "
         "conflict is the released human-label disagreement rate on the test split",
         aggregation="counts from the split manifest")


def gpm_bt():
    p = RES / "saferlhf_ensemble_ckpt" / "saferlhf_ensemble.json"
    r = json.loads(p.read_text())
    rows = []
    for model, lab in (("gpm", "Anti-symmetric GPM"), ("bt", "Scalar BT")):
        for obj in ("helpfulness", "harmlessness"):
            per = [r["per_seed"][model][str(s)][obj]["test_calibrated"]
                   for s in (41, 42, 43)]
            e = r["ensemble"][model][obj]
            temps = [r["calibration"][model][str(s)][obj]["temperature"]
                     for s in (41, 42, 43)]
            rows.append(" & ".join([
                lab, obj.replace("harmlessness", "harmless."),
                f"${statistics.fmean([x['nll'] for x in per]):.4f}$",
                f"${statistics.fmean([x['balanced_accuracy'] for x in per]):.4f}$",
                f"${statistics.fmean([x['roc_auc'] for x in per]):.4f}$",
                f"${statistics.fmean(temps):.3f}$",
                f"${e['nll']:.4f}$", f"${e['balanced_accuracy']:.4f}$",
                f"${e['roc_auc']:.4f}$", f"${e['ece']:.4f}$",
                f"${e['ensemble_disagreement_std_mean']:.4f}$"]) + r"\\")
    emit("gpm_bt", rows, [p], "completed",
         "per-seed columns are the mean over the three CHECKPOINTED seeds 41/42/43 "
         "(the run whose weights are on disk and used downstream); ensemble columns "
         "are the calibrated three-seed mean",
         seeds=[41, 42, 43], aggregation="per-seed mean; calibrated ensemble mean")


def pool_pilot():
    p = RES / "pool_pilot" / "pool_geometry.json"
    r = json.loads(p.read_text())
    rows = []
    for key in ("4+4", "8+8"):
        g = r["geometries"][key]
        sh, o, s = g["target_split_half"], g["oracle_gpm"], g["solve"]
        cyc = g["predicted_cycles_gpm"]["helpfulness"]
        rows.append(" & ".join([
            key, f"${g['pool']['exact_duplicate_pair_rate']:.4f}$",
            f"${o['predictive_entropy']:.4f}$",
            f"${o['ensemble_sd_mean']:.4f}$",
            f"${g['gpm_bt_agreement']['sign_agreement']:.4f}$",
            f"${100*g['objective_conflict_rate']:.2f}\\%$",
            f"${cyc['predicted_cycles']}/{cyc['triples']}$",
            f"${sh['pearson_mean']:.4f}$", f"${sh['spearman_mean']:.4f}$",
            f"${sh['sign_agreement_mean']:.4f}$",
            f"${s['dual_evaluations']}$",
            f"${g['total_solve_seconds']:.1f}$"]) + r"\\")
    emit("pool_pilot", rows, [p], "completed",
         "200 SafeRLHF validation prompts; conflict is the PREFERENCE-MODEL "
         "disagreement rate on GENERATED responses, not the human-label rate; "
         "cycles are model-predicted on the comparator tournament",
         seeds=[41, 42, 43], aggregation="calibrated ensemble mean over oracle seeds")


def solver_scaling():
    p = RES / "solver_scaling" / "direct_solver_scaling.csv"
    rows_in = list(csv.DictReader(open(p)))
    best = {}
    for r in rows_in:
        k = (int(r["prompts"]), int(r["responses"]))
        if k not in best or float(r["total_solve_seconds"]) < float(best[k]["total_solve_seconds"]):
            best[k] = r
    rows = []
    for (X, I), r in sorted(best.items()):
        rows.append(" & ".join([
            f"${X}$", f"${I}{{+}}{I}$", f"${r['workers']}$",
            f"${float(r['total_solve_seconds']):.1f}$",
            f"${r['dual_evaluations']}$",
            f"${float(r['seconds_per_dual_evaluation']):.3f}$",
            sci(float(r["projected_kkt_residual"])),
            sci(float(r["inner_extra_map_residual"])),
            f"${float(r['peak_rss_mb']):.0f}$",
            f"${float(r['artifact_write_seconds']):.2f}$"]) + r"\\")
    emit("solver_scaling", rows, [p], "completed",
         "measured, not extrapolated; best worker count at each size; all 24 "
         "configurations converged",
         aggregation="single measurement per configuration")


def neural_realization():
    """Target statistics only until training completes."""
    p = RES / "smoke" / "train" / "solutions" / "smoke_solutions.json"
    if not p.exists():
        emit("neural_realization", [r"\multicolumn{8}{l}{\textit{pending: "
                                    r"solver targets not yet built}}\\"], [],
             "pending", "no solver artifact")
        return
    d = json.loads(p.read_text())["rows"]
    rows = []
    for name in ("nbpo", "fixed_reference_nash", "bt_rm_nash", "game_utilitarian",
                 "game_ks"):
        r = d.get(name)
        if not r or r.get("status") != "ok":
            continue
        lab = {"nbpo": r"\NBPO{}", "fixed_reference_nash": "Fixed-ref.\\ Nash",
               "bt_rm_nash": "BT-RM--Nash", "game_utilitarian": "Game-util.",
               "game_ks": "Game-KS"}[name]
        rows.append(" & ".join([
            lab, f"${r['target_rms']:.3f}$",
            f"${r['target_p10']:+.3f}$", f"${r['target_p90']:+.3f}$",
            f"${r['weight_l1']:.2f}$",
            sci(r["projected_kkt_residual"]) if r.get("projected_kkt_residual") is not None else "--",
            sci(r["extra_map_residual"]), sci(r["target_identity_residual"]),
            r"\pending", r"\pending", r"\pending"]) + r"\\")
    emit("neural_realization", rows, [p], "partial",
         "solver-side columns are measured on the 1000-prompt smoke pool; the "
         "regression columns are pending until policy training completes",
         seeds=[11], aggregation="single smoke instance (not a 3-seed mean)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=RES / "result_manifest.json")
    ap.parse_args()
    print("regenerating paper table fragments")
    for fn in (controlled_compact, controlled_full, solver_audit,
               controlled_robustness, data_audit, gpm_bt, pool_pilot,
               solver_scaling, neural_realization):
        try:
            fn()
        except Exception as exc:
            print(f"  {fn.__name__:34s} FAILED     {type(exc).__name__}: {exc}")
            MANIFEST[fn.__name__] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "result_manifest.json").write_text(json.dumps(
        {"generated_by": "scripts/experiments/iclr2027_table1_v2/export_paper_tables.py",
         "fragments": MANIFEST}, indent=2))
    print(f"\nwrote {RES}/result_manifest.json")


if __name__ == "__main__":
    main()
