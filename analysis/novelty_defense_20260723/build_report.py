#!/usr/bin/env python3
"""Regenerate the novelty-defense report from frozen evaluation artifacts."""

import argparse
import json
from pathlib import Path


LABELS = {
    "base": "Base",
    "ht_skywork": "HT-MNPO (Skywork)",
    "ht_athene": "HT-MNPO (Athene)",
    "ht_armo": "HT-MNPO (ArmoRM)",
    "ronpo_os": "RONPO-OS",
    "ronpo_topmass": "RONPO-topmass",
    "ronpo_konly": "RONPO-k-only",
    "ronpo_aonly": "RONPO-a-only",
    "sppo_avg_s2": "SPPO-avg-s2",
    "inpo_avg_s2": "INPO-avg-s2",
    "maxmin_rlhf": "MaxMin-RLHF",
    "ronpo_lam4": "RONPO-OS (lambda=4)",
    "ronpo_lam16": "RONPO-OS (lambda=16)",
    "ronpo_os_s43": "RONPO-OS (seed 43)",
    "ronpo_konly_s43": "RONPO-k-only (seed 43)",
}


def fmt_ci(ci: list[float], digits: int = 5) -> str:
    return f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def verdict(delta: float, ci: list[float]) -> str:
    if ci[0] > 0:
        return "OS is significantly higher."
    if ci[1] < 0:
        return "OS is significantly lower."
    return "No statistically separable difference."


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root)
    paired = json.loads((root / "paired_summary.json").read_text())
    raw = json.loads((root / "per_policy_scores/raw_reward_summary.json").read_text())
    ifeval_rows = json.loads((root / "ifeval/summary.json").read_text())
    ifeval = {row["model"]: row for row in ifeval_rows}
    gates = {
        f.stem: json.loads(f.read_text())
        for f in (root / "stability_gates").glob("*.json")
        if not f.stem.endswith("_reuse") and f.stem != "topmass_reuse"
    }
    scores = paired["per_policy"]
    core = [
        "base", "ht_skywork", "ht_athene", "ht_armo", "ronpo_os",
        "ronpo_topmass", "ronpo_konly", "ronpo_aonly", "sppo_avg_s2",
        "inpo_avg_s2", "maxmin_rlhf",
    ]
    context = paired["normalization_policies"]

    lines = [
        "# Qwen2.5-1.5B stage-2 novelty-defense evaluation",
        "",
        "## Bottom line",
        "",
        "The frozen joint evaluation does not support the claimed benefit of the fully "
        "factorized RONPO-OS adversary. On the preregistered theory-aligned pairwise floor, "
        "RONPO-OS is significantly below both k-only and a-only, and it is also below "
        "MaxMin-RLHF. MaxMin-RLHF has the best normalized Avg and Worst among the 11 core "
        "policies and passes the reward-blind stability gate. INPO-avg-s2 has the best hard "
        "and soft pairwise floors, but fails the repetition gate. Consequently, the reward "
        "ordering is reported in full but cannot be presented as a clean stability-valid "
        "method comparison for the failed policies.",
        "",
        "## Frozen protocol",
        "",
        f"- Prompts: {paired['num_prompts']} held-out UltraFeedback prompts.",
        "- Decode: vLLM seed 42, temperature 0.7, top-p 0.9, maximum 2,048 new tokens.",
        "- Scoring: Skywork, Athene, and ArmoRM in one shared BF16 batch context; no scores "
        "  from split passes were merged.",
        "- Normalization: per prompt over the preregistered 15-policy context, including "
        "  the two completed seed-43 replicates and lambda ablations.",
        "- Bootstrap: paired prompt resampling, 2,000 resamples, seed 42.",
        "- Pair floor: E_x[min_(k,a) sigmoid(8(r_k(y)-r_k(a)))]. Soft floor uses kappa=0.05.",
        "- Frozen response pool: stored stage-2 test opponents plus every compared policy's "
        "  own evaluation response.",
        "",
        "All normalized values differ from earlier tables because the frozen policy set "
        "changed. They must not be combined with values normalized over an older set.",
        "",
        "## Preregistered primary contrasts",
        "",
        "| Contrast | Pair-floor delta | 95% CI | Verdict |",
        "|---|---:|---:|---|",
    ]
    for key, label in [
        ("ronpo_os_minus_ronpo_konly", "OS minus k-only"),
        ("ronpo_os_minus_ronpo_aonly", "OS minus a-only"),
        ("ronpo_os_minus_maxmin_rlhf", "OS minus MaxMin-RLHF"),
    ]:
        row = paired["primary_contrasts"][key]
        lines.append(
            f"| {label} | {row['pairwise_floor_delta']:+.5f} | "
            f"{fmt_ci(row['pairwise_floor_delta_ci95'])} | "
            f"{verdict(row['pairwise_floor_delta'], row['pairwise_floor_delta_ci95'])} |"
        )
    lines += [
        "",
        "The normalized Worst and Avg contrasts for OS versus k-only and a-only have "
        "confidence intervals that include zero. OS is significantly below MaxMin-RLHF on "
        "both: Worst delta "
        f"{paired['primary_contrasts']['ronpo_os_minus_maxmin_rlhf']['worst_norm_delta']:+.4f} "
        f"{fmt_ci(paired['primary_contrasts']['ronpo_os_minus_maxmin_rlhf']['worst_norm_delta_ci95'], 4)} "
        "and Avg delta "
        f"{paired['primary_contrasts']['ronpo_os_minus_maxmin_rlhf']['avg_norm_delta']:+.4f} "
        f"{fmt_ci(paired['primary_contrasts']['ronpo_os_minus_maxmin_rlhf']['avg_norm_delta_ci95'], 4)}.",
        "",
        "## Core-policy results",
        "",
        "| Policy | Gate | Avg | Worst [95% CI] | Hard floor | Soft floor | WR_B | wWR_B |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    ranked = sorted((n for n in core if n != "base"),
                    key=lambda n: scores[n]["mean_prompt_worst_norm_score"], reverse=True) + ["base"]
    for name in ranked:
        s = scores[name]
        gate = "PASS" if gates[name]["passed"] else "FAIL"
        wr = "--" if name == "base" else f"{100*s['mean_win_rate_vs_base']:.2f}"
        wwr = "--" if name == "base" else f"{100*s['min_win_rate_vs_base']:.2f}"
        lines.append(
            f"| {LABELS[name]} | {gate} | {s['mean_objective_norm_score']:.4f} | "
            f"{s['mean_prompt_worst_norm_score']:.4f} {fmt_ci(s['worst_norm_ci95'], 4)} | "
            f"{s['pairwise_floor']:.5f} | {s['soft_pairwise_floor']:.5f} | {wr} | {wwr} |"
        )
    lines += [
        "",
        "## Factorized-adversary ablation",
        "",
        "| Estimator | Seed | Gate | Hard floor | Soft floor | Avg | Worst |",
        "|---|---:|:---:|---:|---:|---:|---:|",
    ]
    factor = ["ronpo_os", "ronpo_os_s43", "ronpo_konly", "ronpo_konly_s43", "ronpo_aonly", "ronpo_topmass"]
    seed = {"ronpo_os_s43": 43, "ronpo_konly_s43": 43}
    for name in factor:
        s = scores[name]
        lines.append(
            f"| {LABELS[name]} | {seed.get(name, 42)} | "
            f"{'PASS' if gates[name]['passed'] else 'FAIL'} | {s['pairwise_floor']:.5f} | "
            f"{s['soft_pairwise_floor']:.5f} | {s['mean_objective_norm_score']:.4f} | "
            f"{s['mean_prompt_worst_norm_score']:.4f} |"
        )
    lines += [
        "| RONPO-a-only | 43 | MISSING | -- | -- | -- | -- |",
        "",
        "Seed 43 is descriptive and is not averaged with seed 42. The a-only seed-43 run "
        "had no completed final checkpoint. The seed-43 ordering is not stable: OS is above "
        "k-only there, whereas OS is below k-only on the primary seed.",
        "",
        "## Raw reward means",
        "",
        "| Policy | Skywork | Athene | ArmoRM | Mean tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    raw_rows = raw.get("per_model", raw.get("policies", raw.get("per_policy", raw)))
    for name in context:
        r = raw_rows[name]
        s = scores[name]
        values = r.get("raw_objectives", r)
        values = {
            objective: value.get("raw_mean", value) if isinstance(value, dict) else value
            for objective, value in values.items()
        }
        lines.append(
            f"| {LABELS[name]} | {values['skywork']:.4f} | {values['athene']:.4f} | "
            f"{values['armo']:.4f} | {s['mean_tokens']:.1f} |"
        )
    lines += [
        "",
        "## Stability gate",
        "",
        "| Policy | Result | Records | Empty | Think leaks | Length ratio | Max repeat |",
        "|---|:---:|---:|---:|---:|---:|---:|",
    ]
    for name in context:
        g = gates[name]
        c = g["candidate"]
        lines.append(
            f"| {LABELS[name]} | {'PASS' if g['passed'] else 'FAIL'} | {c['records']} | "
            f"{c['empty_count']} | {c['think_leak_count']} | "
            f"{g['candidate_base_mean_word_ratio']:.3f} | {c['max_repeat_run']} |"
        )
    lines += [
        "",
        "Only MaxMin-RLHF and HT-MNPO (ArmoRM) pass. Base itself fails because prompt "
        "index 209 contains a 482-token repeated-list artifact. Most Base-derived policies "
        "repeat the same artifact. Other failures include HT-MNPO (Skywork), 37; "
        "HT-MNPO (Athene), 980; and INPO-avg-s2, 644 maximum consecutive repeated tokens. "
        "No generation was patched, removed, or resampled.",
        "",
        "## IFEval for newly recovered arms",
        "",
        "| Policy | Evaluated prompts | Prompt strict | Instruction strict | Mean output tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in sorted(ifeval, key=lambda n: ifeval[n]["mean_prompt_level_strict"], reverse=True):
        r = ifeval[name]
        lines.append(
            f"| {LABELS[name]} | {r['num']} | {r['mean_prompt_level_strict']:.4f} | "
            f"{r['mean_inst_level_strict']:.4f} | {r['mean_output_tokens']:.1f} |"
        )
    lines += [
        "",
        "The official value for any duplicated infrastructure run is the chronologically "
        "first completed report, fixed without reference to its score. Exact-provenance "
        "reports for Base, HT-MNPO, OS, and topmass were not recoverable and remain dashes "
        "in the main fragment.",
        "",
        "## Deliverables and exclusions",
        "",
        "- `frag_stage2_main_table.tex`: full 11-policy replacement table.",
        "- `frag_factorized_ablation.tex`: factorized-adversary ablation and replicates.",
        "- `frag_floor_table.tex`: hard floor, soft floor, and frozen-set gap for all 15 policies.",
        "- `paired_summary.json`, `floor_summary.json`, and `per_policy_scores/`: machine-readable results.",
        "- GPT-5.5 pairwise judgments were excluded as preregistered. No API call was attempted.",
        "- No paper source was edited and no new model was trained.",
        "- Original recovered B200 checkpoints were not deleted. The task-owned topmass transfer copy was "
        "  pruned only after its two shard hashes were reverified against the retained local source.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
