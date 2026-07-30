#!/usr/bin/env python3
"""Aggregate the joint pool and build the publication decision and stage curve."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "ronpo_os": "RONPO (OS)",
    "inpo_avg": "INPO (avg)",
    "sppo_avg": "SPPO (avg)",
    "simpo": "SimPO",
    "ipo": "IPO",
    "dpo": "DPO",
    "ht_mnpo_harmless": "HT-MNPO (harmless)",
    "ht_mnpo_helpfulness": "HT-MNPO (helpful)",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    audit = json.loads((out / "pool_audit.json").read_text(encoding="utf-8"))
    count = len(audit["eligible_models"])
    merge = args.project / "analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py"
    for objective in ("helpfulness", "harmlessness"):
        inputs = [out / "score_shards" / f"{objective}_{i}.jsonl" for i in range(6)]
        subprocess.run([
            sys.executable, str(merge), "merge", "--inputs", *map(str, inputs),
            "--output", str(out / "scores" / f"{objective}.jsonl"),
            "--audit", str(out / "scores" / f"{objective}_audit.json"),
            "--expected-records", "1000", "--expected-scores-per-row", str(count), "--strip-responses",
        ], check=True)
    display = {"base": "Base"}
    for method, label in LABELS.items():
        for stage in range(1, 5):
            display[f"{method}_stage{stage}"] = f"{label}, Stage {stage}"
    display_path = out / "display_names.json"
    display_path.write_text(json.dumps(display, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    aggregate = args.project / "analysis/p5_8b_robust_stage1_stage2_20260717/aggregate_stage1_comparison.py"
    subprocess.run([
        sys.executable, str(aggregate), "--helpfulness", str(out / "scores/helpfulness.jsonl"),
        "--harmlessness", str(out / "scores/harmlessness.jsonl"), "--pool-audit", str(out / "pool_audit.json"),
        "--output-dir", str(out / "results"), "--bootstrap", "2000", "--seed", "42",
        "--scope", "same previously-opened P8 1,000-prompt SafeRLHF panel; joint normalization across every eligible Stage-1--4 method point and base",
        "--report-title", "SafeRLHF Stage-1--4 joint-pool trajectory",
        "--ronpo-arm", "ronpo_os_stage4", "--comparison-label", "best eligible non-RONPO method-stage point",
        "--display-name-map", str(display_path),
    ], check=True)

    per_model: dict[str, dict[str, float]] = {}
    prompt_order: list[str] = []
    per_prompt: dict[str, dict[str, float]] = {}
    with (out / "results/per_prompt_metrics.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            prompt_id = row["prompt_id"]
            model = row["model"]
            if model == "base":
                prompt_order.append(prompt_id)
            per_prompt.setdefault(model, {})[prompt_id] = float(row["prompt_worst_norm"])
    with (out / "results/per_objective_scores.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            per_model[row["model"]] = row
    prompt_order = sorted(set(prompt_order))
    if len(prompt_order) != 1000:
        raise RuntimeError(f"expected 1000 prompts, got {len(prompt_order)}")
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(prompt_order), size=(2000, len(prompt_order)))
    curve_rows = []
    arrays: dict[str, np.ndarray] = {}
    for method in LABELS:
        for stage in range(1, 5):
            key = f"{method}_stage{stage}"
            if key not in per_prompt:
                curve_rows.append({"method": method, "display_name": LABELS[method], "stage": stage, "status": "failed_or_missing", "worst": "", "ci_low": "", "ci_high": ""})
                continue
            values = np.asarray([per_prompt[key][prompt_id] for prompt_id in prompt_order], dtype=np.float64)
            arrays[key] = values
            ci = interval(values[indices].mean(axis=1))
            curve_rows.append({"method": method, "display_name": LABELS[method], "stage": stage, "status": "eligible", "worst": float(values.mean()), "ci_low": ci[0], "ci_high": ci[1]})
    base_values = np.asarray([per_prompt["base"][prompt_id] for prompt_id in prompt_order], dtype=np.float64)
    base_ci = interval(base_values[indices].mean(axis=1))
    eligible_stage4 = [method for method in LABELS if method != "ronpo_os" and f"{method}_stage4" in arrays]
    if not eligible_stage4 or "ronpo_os_stage4" not in arrays or "ronpo_os_stage1" not in arrays:
        raise RuntimeError("RONPO or Stage-4 baseline comparison is unavailable")
    comparator = max(eligible_stage4, key=lambda method: (arrays[f"{method}_stage4"].mean(), method))
    stage4_delta = arrays["ronpo_os_stage4"] - arrays[f"{comparator}_stage4"]
    stage4_delta_ci = interval(stage4_delta[indices].mean(axis=1))
    comparator_stage1_available = f"{comparator}_stage1" in arrays
    if comparator_stage1_available:
        did = (arrays["ronpo_os_stage4"] - arrays["ronpo_os_stage1"]) - (arrays[f"{comparator}_stage4"] - arrays[f"{comparator}_stage1"])
        did_mean: float | None = float(did.mean())
        did_ci: list[float] | None = interval(did[indices].mean(axis=1))
        trajectory_pass = did_ci[0] > 0
    else:
        did_mean = None
        did_ci = None
        trajectory_pass = False
    decision = {
        "status": "complete",
        "comparison_baseline": comparator,
        "stage4_ronpo_minus_best_baseline": float(stage4_delta.mean()),
        "stage4_ronpo_minus_best_baseline_ci95": stage4_delta_ci,
        "stage1_to_4_gain_difference": did_mean,
        "stage1_to_4_gain_difference_ci95": did_ci,
        "stage4_lead_pass": stage4_delta_ci[0] > 0,
        "trajectory_comparator_stage1_available": comparator_stage1_available,
        "trajectory_advantage_pass": trajectory_pass,
        "paper_body_eligible": stage4_delta_ci[0] > 0 and trajectory_pass,
        "panel_is_previously_opened": True,
        "claim_scope": "descriptive same-panel trajectory; not an independent confirmation",
        "spent_604_prompt_sealed_split_touched": False,
    }
    (out / "paper_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    with (out / "stage_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "legend.fontsize": 7})
    fig, ax = plt.subplots(figsize=(6.6, 3.55))
    baseline_colors = plt.cm.Greys(np.linspace(0.35, 0.78, len(LABELS) - 1))
    color_index = 0
    for method, label in LABELS.items():
        rows = {int(row["stage"]): row for row in curve_rows if row["method"] == method}
        stages = [1, 2, 3, 4]
        values = [float(rows[stage]["worst"]) if rows[stage]["status"] == "eligible" else np.nan for stage in stages]
        if method == "ronpo_os":
            ax.plot(stages, values, color="#c62828", marker="o", linewidth=2.4, markersize=5.2, label=label, zorder=5)
            lows = [float(rows[stage]["ci_low"]) if rows[stage]["status"] == "eligible" else np.nan for stage in stages]
            highs = [float(rows[stage]["ci_high"]) if rows[stage]["status"] == "eligible" else np.nan for stage in stages]
            ax.fill_between(stages, lows, highs, color="#c62828", alpha=0.14, linewidth=0)
        else:
            ax.plot(stages, values, color=baseline_colors[color_index], marker="o", linewidth=1.05, markersize=3.1, label=label, alpha=0.9)
            color_index += 1
    ax.axhline(float(base_values.mean()), color="#1565c0", linestyle="--", linewidth=1.2, label="Base")
    ax.fill_between([1, 4], [base_ci[0], base_ci[0]], [base_ci[1], base_ci[1]], color="#1565c0", alpha=0.08, linewidth=0)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Self-improvement stage")
    ax.set_ylabel("Mean prompt-level worst normalized reward")
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)
    ax.set_xlim(0.85, 4.15)
    ax.legend(ncol=3, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out / "saferlhf_stage_worst.pdf", bbox_inches="tight")
    fig.savefig(out / "saferlhf_stage_worst.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    def tex_number(value: float) -> str:
        return f"{value:.4f}"

    macro_lines = [
        "% Generated from paper_decision.json by finalize_stage_curve.py.",
        rf"\newcommand{{\SafeStageFourDelta}}{{{tex_number(float(decision['stage4_ronpo_minus_best_baseline']))}}}",
        rf"\newcommand{{\SafeStageFourDeltaLow}}{{{tex_number(float(stage4_delta_ci[0]))}}}",
        rf"\newcommand{{\SafeStageFourDeltaHigh}}{{{tex_number(float(stage4_delta_ci[1]))}}}",
    ]
    if did_mean is not None and did_ci is not None:
        macro_lines += [
            rf"\newcommand{{\SafeStageGainDelta}}{{{tex_number(did_mean)}}}",
            rf"\newcommand{{\SafeStageGainDeltaLow}}{{{tex_number(float(did_ci[0]))}}}",
            rf"\newcommand{{\SafeStageGainDeltaHigh}}{{{tex_number(float(did_ci[1]))}}}",
        ]
    (out / "saferlhf_stage_curve_macros.tex").write_text("\n".join(macro_lines) + "\n", encoding="utf-8")
    figure_fragment = r"""\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/saferlhf_stage_worst.pdf}
\caption{\textbf{Worst-objective reward across SafeRLHF self-improvement stages.}
Every point uses the same held-out 1{,}000-prompt panel and one joint per-prompt
min--max normalization pool containing all gate-passing method-stage
checkpoints and the base policy. Lines connect consecutive stable checkpoints;
a missing point denotes a fail-closed stability outcome. The panel was used
previously for the Stage~4 comparison, so this trajectory is descriptive rather
than an independent confirmation.}
\label{fig:saferlhf-stage-worst}
\end{figure}
"""
    (out / "saferlhf_stage_curve_figure.tex").write_text(figure_fragment, encoding="utf-8")

    table = {method: {} for method in LABELS}
    for row in curve_rows:
        table[row["method"]][int(row["stage"])] = row
    report = [
        "# SafeRLHF Stage-1--4 worst-reward trajectory", "",
        "All points use the same previously opened 1,000-prompt panel and one joint min-max normalization pool.", "",
        "| Method | Stage 1 | Stage 2 | Stage 3 | Stage 4 |", "|---|---:|---:|---:|---:|",
    ]
    for method, label in LABELS.items():
        cells = []
        for stage in range(1, 5):
            row = table[method][stage]
            cells.append("FAILED" if row["status"] != "eligible" else f"{float(row['worst']):.4f} [{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}]")
        report.append(f"| {label} | " + " | ".join(cells) + " |")
    report += [
        "", "## Pre-registered paper decision", "",
        f"Comparator: {LABELS[comparator]}.",
        f"Stage-4 RONPO difference: {decision['stage4_ronpo_minus_best_baseline']:.4f} [{stage4_delta_ci[0]:.4f}, {stage4_delta_ci[1]:.4f}].",
        (
            f"Stage-1-to-4 gain difference: {did_mean:.4f} [{did_ci[0]:.4f}, {did_ci[1]:.4f}]."
            if did_ci is not None and did_mean is not None
            else "Stage-1-to-4 gain difference: unavailable because the Stage-4 comparator failed or is missing at Stage 1."
        ),
        f"Paper-body eligible: **{decision['paper_body_eligible']}**.", "",
        "This is a descriptive trajectory because the panel had already been opened for the Stage-4 experiment.", "",
        "## Provenance", "",
        f"- Run lock SHA-256: `{sha(out / 'run_lock.json')}`",
        f"- Joint pool audit SHA-256: `{sha(out / 'pool_audit.json')}`",
        f"- Helpfulness scores SHA-256: `{sha(out / 'scores/helpfulness.jsonl')}`",
        f"- Harmlessness scores SHA-256: `{sha(out / 'scores/harmlessness.jsonl')}`",
    ]
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
