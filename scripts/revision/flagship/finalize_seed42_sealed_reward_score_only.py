#!/usr/bin/env python3
"""Finalize sealed artifacts from completed score and aggregation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.revision.flagship.run_seed42_sealed_reward_score_only import (
    ALL_METHODS,
    ELIGIBLE_METHODS,
    atomic_json,
    load_models_tsv,
    report_markdown,
    validate_preflight,
    validate_results,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--models-tsv", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sealed-sha256", required=True)
    parser.add_argument("--armo-revision", required=True)
    args = parser.parse_args()
    result_dir = args.work / "results"
    audit, gates = validate_preflight(args.work)
    models = load_models_tsv(args.models_tsv)
    rows = json.loads((result_dir / "model_summary.json").read_text())
    validate_results(result_dir, rows)
    score_metadata = json.loads((args.work / "scores/score_metadata.json").read_text())
    if score_metadata.get("num_prompts") != 604 or score_metadata.get("response_model_names") != list(ELIGIBLE_METHODS):
        raise RuntimeError("completed ArmoRM score metadata mismatch")
    score_metadata["revision"] = args.armo_revision
    atomic_json(args.work / "scores/score_metadata.json", score_metadata)

    ranked = sorted(
        rows,
        key=lambda row: (-float(row["mean_primary_prompt_worst_norm_score"]), row["model"]),
    )
    values = [float(row["mean_primary_prompt_worst_norm_score"]) for row in ranked]
    for row, value in zip(ranked, values):
        row["worst_objective_rank"] = 1 + sum(other > value + 1e-12 for other in values)
    with (result_dir / "headline_table.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "worst_objective_rank", "model", "mean_primary_prompt_worst_norm_score",
            "mean_primary_prompt_worst_norm_score_ci95_low",
            "mean_primary_prompt_worst_norm_score_ci95_high",
            "mean_primary_prompt_avg_norm_score", "num_prompts",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranked)
    ranked_payload = {
        "selection_split": "held-out sealed test",
        "metric": "mean_primary_prompt_worst_norm_score",
        "ranked": ranked,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 42,
        "sealed_sha256": args.sealed_sha256,
        "final_selection": json.loads(args.selection_lock.read_text()),
        "p1_sealed_test_opened": True,
        "decode_invoked_during_resume": False,
        "stability_gates": {
            method: gates["models"][method]["status"] for method in ALL_METHODS
        },
        "unscored_models": {
            "dpo": {
                "reason": "genuine stability failure",
                "record_index": 252,
                "max_repeat_run": 1163,
                "threshold": 20,
            }
        },
        "reward_model": {
            "repo_id": "RLHFlow/ArmoRM-Llama3-8B-v0.1",
            "revision": args.armo_revision,
            "objectives": {
                "helpfulness": {"head": "ultrafeedback-helpfulness", "transform": "identity"},
                "safety": {"head": "beavertails-is_safe", "transform": "identity"},
                "conciseness": {"head": "helpsteer-verbosity", "transform": "negate"},
            },
        },
        "gate_correction": "../gate_correction.json",
    }
    atomic_json(result_dir / "ranked_sealed_summary.json", ranked_payload)

    wandb_env = os.environ.copy()
    wandb_env.update({"WANDB_MODE": "online", "WANDB_ENTITY": "promotion-kim", "WANDB_PROJECT": "mnpo"})
    subprocess.run([
        args.python,
        str(args.project / "scripts/revision/flagship/log_reward_results_wandb.py"),
        "--summary", str(result_dir / "ranked_sealed_summary.json"),
        "--stage", "p1-sealed-reward",
        "--output", str(args.work / "wandb_run.json"),
    ], cwd=args.project, env=wandb_env, check=True)
    wandb = json.loads((args.work / "wandb_run.json").read_text())
    report_markdown(
        result_dir / "SEALED_REPORT.md",
        ranked,
        result_dir / "per_objective_scores.csv",
        gates,
        models,
        wandb,
        args.sealed_sha256,
        args.armo_revision,
        audit["generation_artifacts"],
    )
    ronpo = next(row for row in ranked if row["model"] == "ronpo_k_only")
    atomic_json(args.work / "status.json", {
        "status": "completed",
        "stage": "measured_sealed_results",
        "selected_ronpo_variant": "top-mass",
        "selected_model_row": "ronpo_k_only",
        "ronpo_worst_objective_rank": ronpo["worst_objective_rank"],
        "ronpo_worst_objective_score": ronpo["mean_primary_prompt_worst_norm_score"],
        "stability_gates": {
            method: gates["models"][method]["status"] for method in ALL_METHODS
        },
        "decode_invoked_during_resume": False,
        "sealed_test_opened": True,
        "p1_sealed_test_opened": True,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })


if __name__ == "__main__":
    main()
