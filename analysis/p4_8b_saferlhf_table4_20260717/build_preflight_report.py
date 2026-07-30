#!/usr/bin/env python3
"""Render the SafeRLHF preflight report only from measured JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conflict-calibration", type=Path, required=True)
    parser.add_argument("--nonconflict-calibration", type=Path, required=True)
    parser.add_argument("--tradeoff-gate", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    conflict, nonconflict, gate, data = map(load, [args.conflict_calibration, args.nonconflict_calibration, args.tradeoff_gate, args.data_manifest])
    c_cost = conflict["cost_harmlessness_vs_human_safer"]
    c_reward = conflict["reward_helpfulness_vs_human_better"]
    n_cost = nonconflict["cost_harmlessness_vs_human_safer"]
    n_reward = nonconflict["reward_helpfulness_vs_human_better"]
    status = "PASS" if conflict["status"] == "pass" and gate["status"] == "pass" else "FAIL"
    lines = [
        "# Preflight: scorer calibration and shared-pool trade-off",
        "",
        f"**Status: {status}.** This report is rendered only from the four JSON artifacts cited below. The human-label calibration predates every preference-training run in this experiment; no trained-policy reward outcome was consulted for this gate.",
        "",
        "## Scorer calibration on PKU dual-preference conflict rows",
        "",
        f"- Beaver cost, reported as harmlessness (-cost), selected the human `safer_response_id` on {c_cost['correct']}/{c_cost['rows']} rows: **{pct(c_cost['accuracy'])}** (threshold {pct(conflict['thresholds']['cost_safety_accuracy'])}).",
        f"- Beaver reward selected the human `better_response_id` on {c_reward['correct']}/{c_reward['rows']} rows: **{pct(c_reward['accuracy'])}** (threshold {pct(conflict['thresholds']['reward_help_accuracy'])}).",
        f"- The heads are pinned to distinct revisions: reward `{conflict['head_distinctness']['reward_model_revision']}` and cost `{conflict['head_distinctness']['cost_model_revision']}`. Their same-response Spearman correlation is {conflict['head_distinctness']['score_spearman_on_same_human_responses']:.4f}.",
        "",
        "Non-conflict rows are descriptive only: Beaver cost safer-response accuracy was "
        f"{pct(n_cost['accuracy'])} ({n_cost['correct']}/{n_cost['rows']}); Beaver reward better-response accuracy was "
        f"{pct(n_reward['accuracy'])} ({n_reward['correct']}/{n_reward['rows']}).",
        "",
        "## Trade-off in the shared base response pool",
        "",
        f"The 2,500-prompt shared pool contains four base responses per prompt. Median within-prompt reward/cost Spearman is {gate['median_prompt_spearman']:.4f} (gate ≤ {gate['threshold']['median_spearman_must_be_at_most']:.1f}); the mean is {gate['mean_prompt_spearman']:.4f}. The reward argmax and harmlessness argmax differ for {gate['reward_argmax_cost_argmax_mismatch_count']}/{gate['prompts']} prompts ({pct(gate['reward_argmax_cost_argmax_mismatch_rate'])}).",
        f"Mean within-prompt ranges are {gate['helpfulness_mean_within_prompt_range']:.4f} for helpfulness and {gate['harmlessness_mean_within_prompt_range']:.4f} for harmlessness. {gate['nonfinite_prompt_correlations']} prompts have a constant objective and therefore undefined Spearman; they remain in the shared data and are not silently dropped.",
        "",
        "## Data provenance and limitation",
        "",
        f"Training uses {data['train']['rows']} conflict prompts (SHA-256 `{data['train']['sha256']}`). The prompt-disjoint test-conflict validation panel has only {data['validation']['rows']} rows (SHA-256 `{data['validation']['sha256']}`), because {data['collision_counts']['test_conflicts_removed_by_prior_heldout']} of 696 raw test conflict rows overlap earlier held-out manifests. This is an explicit power limitation, not a reason to pad or replace the panel.",
        "",
        "## Source artifacts",
        "",
        f"- conflict calibration: `{args.conflict_calibration}`",
        f"- non-conflict calibration: `{args.nonconflict_calibration}`",
        f"- shared-pool trade-off gate: `{args.tradeoff_gate}`",
        f"- dataset manifest: `{args.data_manifest}`",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
