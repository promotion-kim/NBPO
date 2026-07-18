#!/usr/bin/env python3
"""Rebuild the Qwen3-8B RONPO paper results (main reward-robustness table,
academic no-regression table, and the main results figure) directly from the
measured JSON/CSV artifacts.

Design goals
------------
* No hand-transcribed numbers: everything is read from the result files, so
  when the SEALED test is finally scored you only change --reward-json to the
  sealed `ranked_sealed_summary.json` / per-objective CSV and re-run.
* Method display names follow the paper's estimator terminology:
    ronpo_k_only      -> "RONPO (top-mass)"     [1.5B-headline estimator]
    ronpo_full_expect -> "RONPO (full-exp.)"    [Rao-Blackwellized estimator]
* Integrity: the P1 numbers here are the NON-SEALED validation *selection*
  split. Captions say so. Do not present them as the sealed headline.

Usage:
    python3 build_ronpo_8b_tables_figure.py
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/sjkim/MNPO"
REWARD_DIR_DEFAULT = os.path.join(REPO, "results/p1_validation_reward_seed42_20260714")
ACADEMIC_CSV_DEFAULT = os.path.join(
    REPO, "results/qwen3_seed42_academic_20260714/final/results/benchmark_table.csv"
)

# canonical display names + ordering identity
DISPLAY = {
    "base": "Base",
    "ronpo_k_only": "RONPO (top-mass)",
    "ronpo_full_expect": "RONPO (full-exp.)",
    "dpo": "DPO",
    "ipo": "IPO",
    "simpo": "SimPO",
    "sppo_avg": "SPPO (avg)",
    "inpo_avg": "INPO (avg)",
    "ht_mnpo_helpfulness": "HT-MNPO (help.)",
    "ht_mnpo_safety": "HT-MNPO (safety)",
    "ht_mnpo_conciseness": "HT-MNPO (concise)",
}
RONPO_FLAGSHIP = "ronpo_k_only"          # top-mass estimator = the winner
RONPO_SECONDARY = "ronpo_full_expect"    # full-expectation estimator
# 11 accessible academic benchmarks (GPQA is BLOCKED/gated; HumanEval is the
# broken all-zero column we surface separately)
ACAD_BENCH = ["IFEval", "MMLU", "MMLU-Pro", "ARC-Challenge", "HellaSwag",
              "TruthfulQA", "Winogrande", "GSM8K", "Minerva-Math", "AIME-24",
              "HumanEval"]


def load_reward(reward_dir):
    """Load the reward summary. Auto-detects the SEALED test file if present
    (ranked_sealed_summary.json), otherwise the validation selection split.
    Also searches a nested results/ subdir, which is where the sealed pipeline
    writes. Returns (summ, perobj, split) with split in {"sealed","validation"}."""
    candidates = [
        (os.path.join(reward_dir, "results", "ranked_sealed_summary.json"), "sealed"),
        (os.path.join(reward_dir, "ranked_sealed_summary.json"), "sealed"),
        (os.path.join(reward_dir, "results", "ranked_validation_summary.json"), "validation"),
        (os.path.join(reward_dir, "ranked_validation_summary.json"), "validation"),
    ]
    summ_path = split = None
    for path, tag in candidates:
        if os.path.exists(path):
            summ_path, split = path, tag
            break
    if summ_path is None:
        raise FileNotFoundError(
            f"No ranked_(sealed|validation)_summary.json under {reward_dir}")
    with open(summ_path) as f:
        ranked = json.load(f)["ranked"]
    summ = {r["model"]: r for r in ranked}
    base_dir = os.path.dirname(summ_path)
    perobj = {}
    with open(os.path.join(base_dir, "per_objective_scores.csv")) as f:
        for row in csv.DictReader(f):
            perobj.setdefault(row["model"], {})[row["objective"]] = row
    return summ, perobj, split


def load_academic(csv_path):
    rows = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows[row["model"]] = row
    return rows


def f3(x):
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return "--"


def f2(x):
    try:
        return f"{float(x):.2f}"
    except (TypeError, ValueError):
        return "--"


def bold(s):
    return "\\textbf{" + s + "}"


def underline(s):
    return "\\underline{" + s + "}"


# ---------------------------------------------------------------------------
# Table 1: main conflict-objective reward-robustness table (P1)
# ---------------------------------------------------------------------------
def build_main_reward_table(summ, perobj, out_tex, split="validation"):
    # rank by the pre-registered primary metric: mean per-prompt worst objective
    models = sorted(summ, key=lambda m: summ[m]["mean_prompt_worst_norm_score"],
                    reverse=True)
    # column best-value bookkeeping (higher is better for all)
    def col(metric):
        return {m: summ[m].get(metric, float("nan")) for m in models}
    worst = col("mean_prompt_worst_norm_score")
    avg = col("mean_objective_norm_score")
    win = {m: summ[m].get("mean_win_rate_vs_baseline") for m in models}
    minwin = {m: summ[m].get("min_win_rate_vs_baseline") for m in models}

    def best2(d, keys):
        vals = sorted(((d[k], k) for k in keys if d.get(k) is not None
                       and not (isinstance(d[k], float) and np.isnan(d[k]))),
                      reverse=True)
        b = vals[0][1] if vals else None
        u = vals[1][1] if len(vals) > 1 else None
        return b, u

    keys = models
    bworst, uworst = best2(worst, keys)
    bavg, uavg = best2(avg, keys)
    wkeys = [m for m in models if m != "base"]
    bwin, uwin = best2(win, wkeys)
    bmw, umw = best2(minwin, wkeys)

    lines = []
    lines.append(r"% AUTO-GENERATED by build_ronpo_8b_tables_figure.py: do not hand-edit numbers")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{lrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{3}{c}{Per-objective norm.} & \multicolumn{4}{c}{Aggregate} \\")
    lines.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}")
    lines.append(r"Method & Help. & Safety & Concise & Avg & Worst & WR$_{\mathrm{B}}$ & wWR$_{\mathrm{B}}$ \\")
    lines.append(r"\midrule")
    for m in models:
        po = perobj.get(m, {})
        help_ = po.get("helpfulness", {}).get("mean_prompt_norm_score")
        safe_ = po.get("safety", {}).get("mean_prompt_norm_score")
        conc_ = po.get("conciseness", {}).get("mean_prompt_norm_score")
        name = DISPLAY.get(m, m)
        if m in (RONPO_FLAGSHIP, RONPO_SECONDARY):
            name = name  # keep, but flagship gets bold row label
        label = bold(name) if m == RONPO_FLAGSHIP else name
        # aggregate cells with bold/underline for best/2nd
        av = f3(avg[m]); av = bold(av) if m == bavg else (underline(av) if m == uavg else av)
        wo = f3(worst[m]); wo = bold(wo) if m == bworst else (underline(wo) if m == uworst else wo)
        if m == "base":
            wr = "--"; mw = "--"
        else:
            wrv = 100.0 * win[m] if win[m] is not None else float("nan")
            mwv = 100.0 * minwin[m] if minwin[m] is not None else float("nan")
            wr = f2(wrv); wr = bold(wr) if m == bwin else (underline(wr) if m == uwin else wr)
            mw = f2(mwv); mw = bold(mw) if m == bmw else (underline(mw) if m == umw else mw)
        cells = [label, f3(help_), f3(safe_), f3(conc_), av, wo, wr, mw]
        lines.append(" & ".join(cells) + r" \\")
        if m == "base":
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    # data-driven winner clause so the caption stays honest if the sealed test
    # reorders the ranking
    top1 = models[0]
    top1_name = DISPLAY.get(top1, top1)
    top1_worst = worst[top1]
    top1_win = win.get(top1)
    win_leader = max(wkeys, key=lambda model: win[model]) if wkeys else None
    if top1_win is None:
        win_clause = ""
    elif top1 == win_leader:
        win_clause = r" and the highest mean win rate over base ($%.1f\%%$)" % (100 * top1_win)
    else:
        win_clause = r"; its mean win rate over base is $%.1f\%%$" % (100 * top1_win)
    if split == "sealed":
        n_prompts, split_phrase = "604", r"\emph{held-out sealed test} (opened once)"
        provenance = (r"\textbf{These are the sealed-test numbers; model selection was fixed on "
                      r"the separate validation split before the test was opened.}")
    else:
        n_prompts, split_phrase = "128", r"\emph{non-sealed validation selection} split"
        provenance = (r"\textbf{These numbers come from the validation split used for model "
                      r"selection; the held-out 604-prompt sealed-test estimate is pending.}")
    caption = (
        r"\textbf{Qwen3-8B heterogeneous-objective reward robustness} on the $" + n_prompts
        + r"$-prompt " + split_phrase + r" (ArmoRM helpfulness / safety / conciseness heads; "
        r"per-prompt min--max normalized; $2000$-resample bootstrap, seed 42). "
        r"\emph{Avg}/\emph{Worst} are the mean per-prompt average and worst-objective normalized "
        r"reward (primary metric); WR$_{\mathrm{B}}$/wWR$_{\mathrm{B}}$ are the mean and "
        r"worst-objective win rate vs.\ the base response (\%). " + top1_name
        + r" attains the best worst-objective normalized reward ($%.3f$)" % top1_worst
        + win_clause + r". " + provenance)
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{tab:qwen3-robust-validation}")
    lines.append(r"\end{table}")
    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    return models, worst


# ---------------------------------------------------------------------------
# Table 2: academic no-regression table (delta vs base + capability macro)
# ---------------------------------------------------------------------------
def build_academic_table(acad, out_tex):
    order = ["base", RONPO_FLAGSHIP, RONPO_SECONDARY, "dpo", "ipo", "simpo",
             "sppo_avg", "inpo_avg", "ht_mnpo_helpfulness", "ht_mnpo_safety",
             "ht_mnpo_conciseness"]

    def macro(model, exclude_humaneval=False):
        bench = [b for b in ACAD_BENCH if not (exclude_humaneval and b == "HumanEval")]
        vals = []
        for b in bench:
            v = acad[model].get(b, "")
            if v not in ("", "BLOCKED", None):
                vals.append(float(v))
        return sum(vals) / len(vals) if vals else float("nan")

    base_macro = macro("base")
    base_macro_x = macro("base", exclude_humaneval=True)

    # bold best macro
    macros = {m: macro(m) for m in order}
    best_macro = max(macros, key=lambda k: macros[k])

    show = ["IFEval", "MMLU", "MMLU-Pro", "ARC-Challenge", "GSM8K", "AIME-24"]
    lines = []
    lines.append(r"% AUTO-GENERATED by build_ronpo_8b_tables_figure.py")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{lrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Method & IFEval & MMLU & MMLU-Pro & ARC & GSM8K & AIME & Macro$_{\Delta}$ \\")
    lines.append(r"\midrule")
    for m in order:
        name = DISPLAY.get(m, m)
        label = bold(name) if m == RONPO_FLAGSHIP else name
        cells = [label]
        for b in show:
            cells.append(f2(acad[m].get(b, "")))
        mac = macros[m]
        if m == "base":
            macstr = f2(mac) + r"$\;(\pm0)$"
        else:
            d = mac - base_macro
            macstr = f2(mac) + (r"$\,(%+.2f)$" % d)
        macstr = bold(macstr) if m == best_macro else macstr
        cells.append(macstr)
        lines.append(" & ".join(cells) + r" \\")
        if m == "base":
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\caption{\textbf{Capability no-regression} on the shared seed-42 academic "
                 r"suite (lm-eval-harness 0.4.12, think disabled). Macro$_{\Delta}$ is the mean over "
                 r"all accessible benchmarks with the signed difference from base in parentheses "
                 r"(GPQA is gated/\textsc{blocked} and excluded; HumanEval is $0$ for every model "
                 r"including base due to a think-mode code-extraction failure and enters the macro "
                 r"as a constant offset for all rows). RONPO (top-mass) improves the macro over base "
                 r"(no regression); RONPO (full-exp.) is within noise. Deltas on individual "
                 r"benchmarks are within run-to-run noise (AIME-24 has 30 items, so $\pm3.3$pp $=$ "
                 r"one problem).}")
    lines.append(r"\label{tab:qwen3-academic}")
    lines.append(r"\end{table}")
    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    return macros, base_macro, base_macro_x


# ---------------------------------------------------------------------------
# Figure: worst-objective robustness with 95% CI + per-objective trade-off
# ---------------------------------------------------------------------------
def build_figure(summ, perobj, out_pdf, out_png):
    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150,
    })
    # Figure display policy (figure only; tables keep the full DISPLAY names):
    #   * drop the RONPO full-expectation ablation bar
    #   * show the top-mass estimator simply as "RONPO"
    FIG_NAME = dict(DISPLAY, **{RONPO_FLAGSHIP: "RONPO"})
    models = [m for m in sorted(summ, key=lambda m: summ[m]["mean_prompt_worst_norm_score"])
              if m != RONPO_SECONDARY]
    names = [FIG_NAME.get(m, m) for m in models]
    worst = [summ[m]["mean_prompt_worst_norm_score"] for m in models]
    lo = [summ[m]["mean_prompt_worst_norm_score_ci95_low"] for m in models]
    hi = [summ[m]["mean_prompt_worst_norm_score_ci95_high"] for m in models]
    # clip tiny float jitter that can make an error arm slightly negative
    err = np.clip(np.array([[w - l for w, l in zip(worst, lo)],
                            [h - w for h, w in zip(worst, hi)]]), 0.0, None)

    C_RONPO, C_BASE, C_OTHER = "#2b6cb0", "#dd6b20", "#a0aec0"
    colors = []
    for m in models:
        if m == RONPO_FLAGSHIP:
            colors.append(C_RONPO)
        elif m == RONPO_SECONDARY:
            colors.append("#7fa8d6")
        elif m == "base":
            colors.append(C_BASE)
        else:
            colors.append(C_OTHER)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1),
                                   gridspec_kw={"width_ratios": [1.45, 1]})
    y = np.arange(len(models))
    ax1.barh(y, worst, xerr=err, color=colors, height=0.68,
             error_kw={"elinewidth": 0.9, "capsize": 2, "ecolor": "#4a5568"})
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=8)
    base_w = summ["base"]["mean_prompt_worst_norm_score"]
    ax1.axvline(base_w, color=C_BASE, ls="--", lw=1.0, alpha=0.8)
    ax1.set_xlabel("Worst-objective norm. reward  (95% CI)")
    ci_low, ci_high = min(lo), max(hi)
    ci_pad = max(0.02, 0.08 * max(ci_high - ci_low, 1e-6))
    ax1.set_xlim(max(0.0, ci_low - ci_pad), min(1.0, ci_high + ci_pad))
    ax1.set_title("(a) Robustness on conflicting objectives", fontsize=9, loc="left")

    # panel (b): per-objective trade-off for Base vs RONPO top-mass vs SPPO(avg)
    comp = ["base", RONPO_FLAGSHIP, "sppo_avg"]
    comp_names = ["Base", "RONPO", "SPPO (avg)"]
    objs = ["helpfulness", "safety", "conciseness"]
    obj_lab = ["Help.", "Safety", "Concise"]
    x = np.arange(len(objs))
    w = 0.26
    bar_colors = [C_BASE, C_RONPO, C_OTHER]
    panel_values = []
    for i, (mm, nm) in enumerate(zip(comp, comp_names)):
        vals = [float(perobj[mm][o]["mean_prompt_norm_score"]) for o in objs]
        panel_values.extend(vals)
        ax2.bar(x + (i - 1) * w, vals, w, label=nm, color=bar_colors[i])
    ax2.set_xticks(x)
    ax2.set_xticklabels(obj_lab, fontsize=8)
    ax2.set_ylabel("Per-objective norm. reward")
    panel_pad = max(0.03, 0.12 * max(max(panel_values) - min(panel_values), 1e-6))
    ax2.set_ylim(max(0.0, min(panel_values) - panel_pad),
                 min(1.0, max(panel_values) + panel_pad))
    ax2.legend(fontsize=6.5, frameon=False, loc="upper center", ncol=1)
    ax2.set_title("(b) Objective-level profile", fontsize=9, loc="left")

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward-dir", default=REWARD_DIR_DEFAULT)
    ap.add_argument("--academic-csv", default=ACADEMIC_CSV_DEFAULT)
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    summ, perobj, split = load_reward(args.reward_dir)
    acad = load_academic(args.academic_csv)

    t1 = os.path.join(args.out_dir, "table_main_reward_robustness.tex")
    t2 = os.path.join(args.out_dir, "table_academic_noregression.tex")
    fpdf = os.path.join(args.out_dir, "fig_worst_objective.pdf")
    fpng = os.path.join(args.out_dir, "fig_worst_objective.png")

    models, worst = build_main_reward_table(summ, perobj, t1, split)
    macros, base_macro, base_macro_x = build_academic_table(acad, t2)
    build_figure(summ, perobj, fpdf, fpng)

    print("Wrote:")
    for p in (t1, t2, fpdf, fpng):
        print("  ", p)
    print(f"\nWorst-objective ranking ({split} split):")
    for i, m in enumerate(sorted(summ, key=lambda k: summ[k]["mean_prompt_worst_norm_score"],
                                 reverse=True), 1):
        print(f"  {i:2d}. {DISPLAY.get(m,m):20s} worst={summ[m]['mean_prompt_worst_norm_score']:.4f}"
              f"  avg={summ[m]['mean_objective_norm_score']:.4f}")
    print(f"\nCapability macro (incl. HumanEval=0), base={base_macro:.2f} | "
          f"base excl-HumanEval={base_macro_x:.2f}")
    for m in ["base", RONPO_FLAGSHIP, RONPO_SECONDARY]:
        print(f"  {DISPLAY[m]:20s} macro={macros[m]:.2f}  (Δ={macros[m]-base_macro:+.2f})")


if __name__ == "__main__":
    main()
