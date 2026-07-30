#!/usr/bin/env python3
"""Merge P6 score shards and evaluate the pre-registered Stage-2 goal."""

from __future__ import annotations

import argparse
import csv
import json
import numpy as np
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = 1000
    audit = json.loads((args.output / "pool_audit.json").read_text())
    count = len(audit["eligible_models"])
    merge = args.project / "analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py"
    for objective in ["helpfulness", "harmlessness"]:
        inputs = [args.output / "score_shards" / f"{objective}_{index}.jsonl" for index in range(6)]
        subprocess.run([sys.executable, str(merge), "merge", "--inputs", *map(str, inputs), "--output", str(args.output / "scores" / f"{objective}.jsonl"), "--audit", str(args.output / "scores" / f"{objective}_audit.json"), "--expected-records", str(expected), "--expected-scores-per-row", str(count), "--strip-responses"], check=True)
    aggregate = args.project / "analysis/p5_8b_robust_stage1_stage2_20260717/aggregate_stage1_comparison.py"
    subprocess.run([sys.executable, str(aggregate), "--helpfulness", str(args.output / "scores/helpfulness.jsonl"), "--harmlessness", str(args.output / "scores/harmlessness.jsonl"), "--pool-audit", str(args.output / "pool_audit.json"), "--output-dir", str(args.output / "results"), "--bootstrap", "2000", "--seed", "42", "--scope", "fresh 1,000-prompt PKU-SafeRLHF test, prompt-disjoint from every listed earlier training/evaluation panel", "--report-title", "P6 Stage-2 fresh SafeRLHF test comparison", "--ronpo-arm", "ronpo_os_stage2", "--ronpo-arm", "ronpo_topmass_stage2", "--comparison-label", "best eligible non-RONPO trained Stage-2 arm"], check=True)
    summary = json.loads((args.output / "results/ranked_validation_summary.json").read_text())
    rows = {row["model"]: row for row in summary["ranking"]}
    per_prompt: dict[str, dict[str, dict]] = {}
    with (args.output / "results/per_prompt_metrics.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            per_prompt.setdefault(row["model"], {})[row["prompt_id"]] = row
    rng = np.random.default_rng(42)
    prompt_ids = sorted(per_prompt["base"])
    indices = rng.integers(0, len(prompt_ids), size=(2000, len(prompt_ids)))
    goals = {}
    for arm in ["ronpo_os_stage2", "ronpo_topmass_stage2"]:
        comp = summary["new_arm_paired_comparisons"].get(arm)
        if not comp:
            goals[arm] = {"eligible": False, "goal_met": False, "reason": "failed stability gate or absent from eligible scoring pool"}
            continue
        worst_other = comp["worst_comparator"]
        avg_other = max((row for key, row in rows.items() if key != "base" and not key.startswith("ronpo_")), key=lambda row: (row["mean_objective_norm_score"], row["model"]))["model"]
        delta_avg = np.asarray([float(per_prompt[arm][pid]["prompt_avg_norm"]) - float(per_prompt[avg_other][pid]["prompt_avg_norm"]) for pid in prompt_ids])
        avg_ci = [float(np.quantile(delta_avg[indices].mean(axis=1), 0.025)), float(np.quantile(delta_avg[indices].mean(axis=1), 0.975))]
        worst_ci = comp["paired_prompt_worst_difference_ci95"]
        worst_pass = worst_ci[0] > 0.0
        avg_pass = avg_ci[0] > -0.02
        goals[arm] = {"eligible": True, "worst_comparator": worst_other, "worst_difference": comp["paired_prompt_worst_difference"], "worst_difference_ci95": worst_ci, "worst_pass": worst_pass, "avg_comparator": avg_other, "avg_difference": float(delta_avg.mean()), "avg_difference_ci95": avg_ci, "avg_floor_pass": avg_pass, "goal_met": worst_pass and avg_pass}
    payload = {"status": "complete", "goal_rule": json.loads((args.output.parent / "stage2_eval_lock.json").read_text())["goal_rule"], "eligible_models": summary["eligible_models"], "failed_models": audit["failed_models"], "ronpo_goal_assessments": goals, "spent_sealed_split_touched": False}
    (args.output / "goal_assessment.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
