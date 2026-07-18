#!/usr/bin/env python3
"""Regenerate the P4 pre-flight report from measured JSON artifacts."""

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
    parser.add_argument("--conflict", type=Path, required=True)
    parser.add_argument("--nonconflict", type=Path, required=True)
    parser.add_argument("--toy", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    conflict, nonconflict, toy, data = map(load, (args.conflict, args.nonconflict, args.toy, args.data_manifest))
    cost = conflict["cost_harmlessness_vs_human_safer"]
    reward = conflict["reward_helpfulness_vs_human_better"]
    nc_cost = nonconflict["cost_harmlessness_vs_human_safer"]
    nc_reward = nonconflict["reward_helpfulness_vs_human_better"]
    distinct = conflict["head_distinctness"]
    passed = conflict["status"] == "pass" and toy["status"] == "pass" and distinct["distinct_checkpoint_revisions"]
    decision = "PASS: all preregistered pre-flight gates passed; shared pool construction is authorized." if passed else "FAIL: at least one preregistered pre-flight gate failed; do not construct or train the shared pool."
    lines = [
        "# P4 SafeRLHF pre-flight report",
        "",
        "This report is generated from the human-label calibration JSON, the deterministic analytic toy JSON, and the pinned data manifest. No trained-policy, validation, or fresh response is used.",
        "",
        "## Gate decision",
        "",
        decision,
        "",
        "## Human-label calibration on the 696 PKU conflict rows",
        "",
        "| Signal | Human label | Accuracy | Threshold | Mean |score gap| | Ties | Result |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| Beaver cost as harmlessness (-cost) | safer response | {pct(cost['accuracy'])} | {pct(conflict['thresholds']['cost_safety_accuracy'])} | {cost['mean_abs_pair_gap']:.4f} | {cost['ties']} | {'PASS' if cost['accuracy'] >= conflict['thresholds']['cost_safety_accuracy'] else 'FAIL'} |",
        f"| Beaver reward as helpfulness | better response | {pct(reward['accuracy'])} | {pct(conflict['thresholds']['reward_help_accuracy'])} | {reward['mean_abs_pair_gap']:.4f} | {reward['ties']} | {'PASS' if reward['accuracy'] >= conflict['thresholds']['reward_help_accuracy'] else 'FAIL'} |",
        "",
        f"The frozen heads are distinct revisions ({distinct['cost_model_revision']} and {distinct['reward_model_revision']}); their Spearman correlation over the same human responses is {distinct['score_spearman_on_same_human_responses']:.4f}.",
        "",
        "## Non-conflict rows, descriptive only",
        "",
        f"On {nc_cost['rows']} rows with matching human labels, cost-versus-safer agreement is {pct(nc_cost['accuracy'])} and reward-versus-better agreement is {pct(nc_reward['accuracy'])}. These values are not gate-tested because the preregistered resolution thresholds apply only to conflict rows.",
        "",
        "## Analytic imbalance direction check",
        "",
        "The toy is explicitly adaptive-adversary only. It does not establish that the repository's one-shot static target implements OMD; that static-versus-adaptive distinction remains an instrument result to be reported after target construction.",
        "",
        "| Benign fraction rho | Uniform-control worst | Adaptive robust worst | Robust minus uniform |",
        "|---:|---:|---:|---:|",
    ]
    for row in toy["rows"]:
        lines.append(f"| {row['rho_benign']:.2f} | {row['uniform_worst']:.6f} | {row['adaptive_robust_worst']:.6f} | {row['robust_minus_uniform_worst']:.6f} |")
    lines.extend([
        "",
        f"The toy criterion is `{toy['criterion']}` and its measured status is `{toy['status']}`.",
        "",
        "## Frozen data state",
        "",
        f"The matched train count is {data['matched_prompt_count_per_rho']} prompts per rho; the shared union has {data['union_prompt_count']} prompts. Validation has {data['files']['validation.jsonl']['rows']} prompts. The new fresh manifest has {data['files']['fresh_unopened.jsonl']['rows']} prompts and remains `{data['fresh_status']}`. The spent P1 sealed split was touched: `{data['spent_sealed_split_touched']}`.",
        "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "pass" if passed else "fail", "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
