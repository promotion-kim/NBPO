#!/usr/bin/env python3
"""Regenerate the three paper fragments from measured JSON only."""

import argparse
import json
from pathlib import Path

LABELS = {
    "base": "Base",
    "ht_skywork": "HT-MNPO (Sky.)",
    "ht_athene": "HT-MNPO (Ath.)",
    "ht_armo": "HT-MNPO (Armo)",
    "ronpo_os": "RONPO-OS",
    "ronpo_topmass": "RONPO-topmass",
    "ronpo_konly": "RONPO-$k$-only",
    "ronpo_aonly": "RONPO-$a$-only",
    "sppo_avg_s2": "SPPO-avg-s2",
    "inpo_avg_s2": "INPO-avg-s2",
    "maxmin_rlhf": "MaxMin-RLHF",
    "ronpo_lam4": "RONPO-OS ($\\lambda=4$)",
    "ronpo_lam16": "RONPO-OS ($\\lambda=16$)",
    "ronpo_os_s43": "RONPO-OS",
    "ronpo_konly_s43": "RONPO-$k$-only",
}


def decorate(values: dict[str, float], digits: int, scale: float = 1.0) -> dict[str, str]:
    order = sorted(values, key=lambda name: values[name], reverse=True)
    out = {name: f"{values[name] * scale:.{digits}f}" for name in values}
    if order:
        out[order[0]] = f"\\textbf{{{out[order[0]]}}}"
    if len(order) > 1:
        out[order[1]] = f"\\underline{{{out[order[1]]}}}"
    return out


def gate_map(path: Path) -> dict[str, bool]:
    out = {}
    for file in path.glob("*.json"):
        out[file.stem] = bool(json.loads(file.read_text()).get("passed"))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paired", required=True)
    p.add_argument("--floor", required=True)
    p.add_argument("--gates", required=True)
    p.add_argument("--ifeval", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    paired = json.loads(Path(args.paired).read_text())
    floor = json.loads(Path(args.floor).read_text())
    gates = gate_map(Path(args.gates))
    ifeval_rows = json.loads(Path(args.ifeval).read_text()) if Path(args.ifeval).exists() else []
    ifeval = {row["model"]: row for row in ifeval_rows}
    scores = paired["per_policy"]
    out = Path(args.output)

    main_names = [
        "base", "ht_skywork", "ht_athene", "ht_armo", "ronpo_os", "ronpo_topmass",
        "ronpo_konly", "ronpo_aonly", "sppo_avg_s2", "inpo_avg_s2", "maxmin_rlhf",
    ]
    ranked = sorted((n for n in main_names if n != "base"),
                    key=lambda n: scores[n]["mean_prompt_worst_norm_score"], reverse=True) + ["base"]
    avg = decorate({n: scores[n]["mean_objective_norm_score"] for n in main_names}, 4)
    worst = decorate({n: scores[n]["mean_prompt_worst_norm_score"] for n in main_names}, 4)
    wr = decorate({n: scores[n]["mean_win_rate_vs_base"] for n in main_names if n != "base"}, 2, 100)
    wwr = decorate({n: scores[n]["min_win_rate_vs_base"] for n in main_names if n != "base"}, 2, 100)
    strict_values = {n: float(ifeval[n]["mean_prompt_level_strict"]) for n in main_names if n in ifeval}
    strict = decorate(strict_values, 4)
    tokens = {n: str(round(float(ifeval[n]["mean_output_tokens"]))) for n in main_names
              if n in ifeval and ifeval[n].get("mean_output_tokens") is not None}
    rows = []
    for name in ranked:
        label = LABELS[name] + ("\\textsuperscript{$\\dagger$}" if not gates.get(name, False) else "")
        rows.append(" & ".join([
            label, avg[name], worst[name], "--" if name == "base" else wr[name],
            "--" if name == "base" else wwr[name], strict.get(name, "--"), tokens.get(name, "--"),
        ]) + " \\\\")
    main_tex = """\\begin{table}[t]
\\centering
\\scriptsize
\\setlength{\\tabcolsep}{3pt}
\\resizebox{\\columnwidth}{!}{%
\\begin{tabular}{lrrrrrr}
\\toprule
& \\multicolumn{4}{c}{Local RM} & \\multicolumn{2}{c}{IFEval} \\\\
\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}
Method & Avg & Worst & WR$_{\\mathrm{B}}$ & wWR$_{\\mathrm{B}}$ & Strict & Tok. \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}%
}
\\caption{Frozen joint stage-2 evaluation on 647 held-out prompts. Local RM scores are
per-prompt min--max normalized over the preregistered 15-policy context; Worst is
$\\mathbb{E}_x[\\min_k \\widetilde r_k(x)]$. IFEval is newly measured only for the
seven preregistered missing arms; dashes denote unavailable exact-provenance values.
$\\dagger$ denotes a failure of the unchanged reward-blind generation stability gate.
All policies remain shown, and paired confidence intervals are reported separately.}
\\label{tab:stage2-local-ifeval}
\\end{table}
"""
    (out / "frag_stage2_main_table.tex").write_text(main_tex)

    factor_rows = [
        ("ronpo_os", 42), ("ronpo_os_s43", 43), ("ronpo_konly", 42),
        ("ronpo_konly_s43", 43), ("ronpo_aonly", 42), (None, 43),
        ("ronpo_topmass", 42),
    ]
    lines = []
    for name, seed in factor_rows:
        if name is None:
            lines.append("RONPO-$a$-only & 43 & -- & -- & -- & -- \\\\")
            continue
        v = scores[name]
        label = LABELS[name] + ("\\textsuperscript{$\\dagger$}" if not gates.get(name, False) else "")
        lines.append(
            f"{label} & {seed} & {v['pairwise_floor']:.5f} & {v['soft_pairwise_floor']:.5f} & "
            f"{v['mean_objective_norm_score']:.4f} & {v['mean_prompt_worst_norm_score']:.4f} \\\\" )
    factor_tex = """\\begin{table}[t]
\\centering
\\scriptsize
\\begin{tabular}{lrrrrr}
\\toprule
Estimator & Seed & Pair floor & Soft floor & Avg & Worst \\\\
\\midrule
""" + "\n".join(lines) + """
\\bottomrule
\\end{tabular}
\\caption{Factorized-adversary ablation in one frozen scoring context. Seed 43 is a
descriptive replicate and is not averaged with seed 42. The missing $a$-only seed-43
row had no completed final checkpoint. $\\dagger$ marks a stability-gate failure.}
\\label{tab:factorized-adversary-ablation}
\\end{table}
"""
    (out / "frag_factorized_ablation.tex").write_text(factor_tex)

    floor_lines = []
    for name in paired["normalization_policies"]:
        v = floor["per_policy"][name]
        label = LABELS[name] + ("\\textsuperscript{$\\dagger$}" if not gates.get(name, False) else "")
        hard_ci = v["pairwise_floor_ci95"]
        soft_ci = v["soft_pairwise_floor_ci95"]
        floor_lines.append(
            f"{label} & {v['pairwise_floor']:.5f} [{hard_ci[0]:.5f}, {hard_ci[1]:.5f}] & "
            f"{v['soft_pairwise_floor']:.5f} [{soft_ci[0]:.5f}, {soft_ci[1]:.5f}] & "
            f"{v['finite_set_duality_gap_proxy']:.5f} \\\\" )
    floor_tex = """\\begin{table*}[t]
\\centering
\\scriptsize
\\begin{tabular}{lccc}
\\toprule
Policy & Hard pairwise floor [95\\% CI] & Soft floor [95\\% CI] & Frozen-set gap \\\\
\\midrule
""" + "\n".join(floor_lines) + """
\\bottomrule
\\end{tabular}
\\caption{Theory-aligned finite-pool endpoints on 647 prompts. The hard endpoint is
$\\mathbb{E}_x[\\min_{k,a}\\sigma(8(r_k(y)-r_k(a)))]$; the soft endpoint uses
$\\kappa=0.05$. The gap is relative to the best soft floor in this frozen policy set
and is not an unrestricted-game duality gap. $\\dagger$ marks a stability-gate failure.}
\\label{tab:stage2-pairwise-floor}
\\end{table*}
"""
    (out / "frag_floor_table.tex").write_text(floor_tex)


if __name__ == "__main__":
    main()
