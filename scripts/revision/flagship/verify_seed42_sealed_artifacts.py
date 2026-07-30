#!/usr/bin/env python3
"""Fail-closed acceptance checks for the completed P1 sealed artifact set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--sealed-prompts", type=Path, required=True)
    parser.add_argument("--sealed-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lock = read_json(args.work / "selection_lock.json")
    opened = read_json(args.work / "sealed_opened.json")
    status = read_json(args.work / "status.json")
    summary = read_json(args.work / "results/ranked_sealed_summary.json")
    gates = read_json(args.work / "stability_gates/summary.json")
    wandb = read_json(args.work / "wandb_run.json")
    rows = summary["ranked"]
    digest = hashlib.sha256(args.sealed_prompts.read_bytes()).hexdigest()
    model_names = [row["model"] for row in rows]
    expected_models = {
        "base", "ronpo_full_expect", "ronpo_k_only", "dpo", "ipo", "simpo",
        "sppo_avg", "inpo_avg", "ht_mnpo_helpfulness", "ht_mnpo_safety",
        "ht_mnpo_conciseness",
    }
    selected_is_extra = opened.get("selected_model_row") == "ronpo_selected"
    if selected_is_extra:
        expected_models.add("ronpo_selected")
    finite_fields = (
        "mean_primary_prompt_worst_norm_score",
        "mean_primary_prompt_worst_norm_score_ci95_low",
        "mean_primary_prompt_worst_norm_score_ci95_high",
        "mean_primary_prompt_avg_norm_score",
    )
    finite = all(math.isfinite(float(row[field])) for row in rows for field in finite_fields)
    ci_ordered = all(
        float(row["mean_primary_prompt_worst_norm_score_ci95_low"])
        <= float(row["mean_primary_prompt_worst_norm_score"])
        <= float(row["mean_primary_prompt_worst_norm_score_ci95_high"])
        for row in rows
    )
    with (args.work / "results/per_objective_scores.csv").open(newline="", encoding="utf-8") as handle:
        per_objective = list(csv.DictReader(handle))
    checks = {
        "selection_locked": lock.get("status") == "locked",
        "selection_used_no_sealed_data": lock.get("sealed_data_consulted_for_selection") is False,
        "lock_precedes_open": datetime.fromisoformat(lock["locked_at"]) < datetime.fromisoformat(opened["opened_at"]),
        "single_open_marker": opened.get("single_open_policy") is True,
        "sealed_hash_matches": digest == args.sealed_sha256 == summary.get("sealed_sha256"),
        "status_completed": status.get("status") == "completed" and status.get("p1_sealed_test_opened") is True,
        "exact_model_pool": set(model_names) == expected_models and len(model_names) == len(expected_models),
        "all_summary_values_finite": finite,
        "worst_ci_ordered": ci_ordered,
        "all_prompt_counts_604": all(int(row.get("num_prompts", 0)) == 604 for row in rows),
        "per_objective_rows_complete": len(per_objective) == 3 * len(expected_models),
        "all_stability_gates_passed": gates.get("all_passed") is True,
        "wandb_completed": wandb.get("status") == "completed" and bool(wandb.get("wandb_run_id")),
        "sealed_report_present": (args.work / "results/SEALED_REPORT.md").is_file(),
    }
    payload = {
        "status": "passed" if all(checks.values()) else "failed",
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "checks": checks,
        "model_count": len(model_names),
        "models": model_names,
        "selected_ronpo_variant": lock.get("selected_ronpo_variant"),
        "wandb_run_id": wandb.get("wandb_run_id"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["status"] == "passed" else 4)


if __name__ == "__main__":
    main()
