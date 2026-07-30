#!/usr/bin/env python3
"""Score all existing confirmatory outputs under the locked post-hoc amendment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path

from run_seed42_sealed_reward_eval import METHODS, atomic_json, validate_measured_results


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(result_dir: Path, ranked: list[dict], amendment: dict,
                 gates: dict, wandb: dict) -> None:
    objectives: dict[str, dict[str, dict]] = {}
    with (result_dir / "per_objective_scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            objectives.setdefault(row["model"], {})[row["objective"]] = row
    lines = [
        "# Post-hoc all-output confirmatory reward sensitivity", "",
        "This is a measured post-hoc sensitivity analysis, not the preregistered fair flagship table. "
        "RONPO passed the strict stability gate; baseline failures remain visible but their unmodified raw "
        "outputs were scored at the user's request.", "",
        "| Rank | Model | Worst (95% CI) | Avg | Win vs base | Help. | Safety | Concise. | S3 | Max repeat |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in ranked:
        model = row["model"]
        obj = objectives[model]
        win = row.get("mean_win_rate_vs_baseline")
        win_text = "--" if win is None or not math.isfinite(float(win)) else f"{100*float(win):.2f}%"
        gate = gates["models"][model]
        lines.append(
            f"| {row['worst_objective_rank']} | {model} | "
            f"{float(row['mean_primary_prompt_worst_norm_score']):.4f} "
            f"[{float(row['mean_primary_prompt_worst_norm_score_ci95_low']):.4f}, "
            f"{float(row['mean_primary_prompt_worst_norm_score_ci95_high']):.4f}] | "
            f"{float(row['mean_primary_prompt_avg_norm_score']):.4f} | {win_text} | "
            f"{float(obj['helpfulness']['mean_prompt_norm_score']):.4f} | "
            f"{float(obj['safety']['mean_prompt_norm_score']):.4f} | "
            f"{float(obj['conciseness']['mean_prompt_norm_score']):.4f} | "
            f"{gate['status']} | {int(gate['candidate']['max_repeat_run'])} |"
        )
    lines.extend([
        "", "## Provenance and limitation", "",
        f"- Prompts: {amendment['prompt_count']}; file SHA-256 `{amendment['prompt_file_sha256']}`.",
        "- No response was regenerated, cleaned, truncated post hoc, or removed.",
        "- All 11 raw-output files were scored by the same three ArmoRM heads and normalized together.",
        "- Paired prompt bootstrap: 2,000 resamples, seed 42.",
        "- Stability eligibility is asymmetric and was changed after the original fail-closed result; "
        "therefore these ranks are exploratory and not preregistered-headline eligible.",
        f"- W&B: `{wandb.get('wandb_run_id', 'unknown')}` ({wandb.get('wandb_url', 'unknown')}).", "",
    ])
    (result_dir / "POSTHOC_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    args = parser.parse_args()
    work = args.eval_root / "confirmatory"
    result_dir = work / "results_posthoc_all_outputs"
    result_dir.mkdir(exist_ok=True)
    logs = work / "logs_posthoc_all_outputs"
    logs.mkdir(exist_ok=True)
    status_path = work / "posthoc_scoring_status.json"
    amendment = json.loads(args.amendment.read_text())
    if amendment.get("status") != "locked_posthoc_scoring_only":
        raise RuntimeError("post-hoc scoring amendment is not locked")
    gates = json.loads((work / "stability_gates/summary.json").read_text())
    for ronpo in ("ronpo_full_expect", "ronpo_k_only"):
        if gates["models"][ronpo].get("passed") is not True:
            raise RuntimeError(f"strict RONPO gate no longer passes: {ronpo}")
    for method in METHODS:
        expected = amendment["generations"][method]
        path = Path(expected["path"])
        rows = json.loads(path.read_text())
        if len(rows) != expected["records"] or sha256_file(path) != expected["sha256"]:
            raise RuntimeError(f"raw generation changed after amendment lock: {method}")

    merged = work / "merged_generations_posthoc_all_outputs.json"
    atomic_json(status_path, {"status": "running", "stage": "merge_existing_raw_outputs",
                              "generation_started": False, "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
    command = [args.python, "-m", "mnpo_scripts.merge_model_generations", "--generations"]
    command.extend([f"{method}={work / 'generations' / method / 'output_42.json'}" for method in METHODS])
    command.extend(["--output_file", str(merged)])
    with (logs / "merge.log").open("a") as handle:
        subprocess.run(command, cwd=args.project, stdout=handle, stderr=subprocess.STDOUT, check=True)

    scores = work / "scores_posthoc_all_outputs"
    atomic_json(status_path, {"status": "running", "stage": "armo_scoring_all_raw_outputs",
                              "generation_started": False, "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
    with (logs / "score.log").open("a") as handle:
        subprocess.run([
            args.python, str(args.project / "scripts/revision/flagship/score_armo_primary_heads_sharded.py"),
            "--python", args.python, "--input-file", str(merged), "--output-dir", str(scores),
            "--cache-dir", str(args.root / "cache/huggingface/hub"),
            "--gpu-ids", "0", "1", "2", "3", "--batch-size", "8",
            "--sample-batch-size", "4", "--local-files-only",
        ], cwd=args.project, stdout=handle, stderr=subprocess.STDOUT, check=True)

    atomic_json(status_path, {"status": "running", "stage": "posthoc_aggregation_bootstrap",
                              "generation_started": False, "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
    with (logs / "evaluate.log").open("a") as handle:
        subprocess.run([
            args.python, "-m", "mnpo_scripts.evaluate_multi_objective_models",
            "--scored_files", f"helpfulness={scores / 'helpfulness.jsonl'}",
            f"safety={scores / 'safety.jsonl'}", f"conciseness={scores / 'conciseness.jsonl'}",
            "--output_dir", str(result_dir), "--baseline_model", "base",
            "--primary_objectives", "helpfulness", "safety", "conciseness",
            "--bootstrap_samples", "2000", "--bootstrap_seed", "42",
        ], cwd=args.project, stdout=handle, stderr=subprocess.STDOUT, check=True)
    rows = json.loads((result_dir / "model_summary.json").read_text())
    validate_measured_results(result_dir, rows, list(METHODS), amendment["prompt_count"])
    ranked = sorted(rows, key=lambda row: (-float(row["mean_primary_prompt_worst_norm_score"]), row["model"]))
    values = [float(row["mean_primary_prompt_worst_norm_score"]) for row in ranked]
    for row, value in zip(ranked, values):
        row["worst_objective_rank"] = 1 + sum(other > value + 1e-12 for other in values)
        row["stability_status"] = gates["models"][row["model"]]["status"]
        row["max_repeat_run"] = gates["models"][row["model"]]["candidate"]["max_repeat_run"]
    summary_path = result_dir / "ranked_posthoc_summary.json"
    atomic_json(summary_path, {
        "analysis_label": "posthoc_all_output_sensitivity",
        "paper_eligible_as_preregistered_flagship": False,
        "metric": "mean_primary_prompt_worst_norm_score",
        "prompt_count": amendment["prompt_count"],
        "ranked": ranked,
        "stability_gates": {method: gates["models"][method]["status"] for method in METHODS},
        "fairness_note": amendment["fairness_note"],
        "no_generation_or_output_cleaning": True,
        "bootstrap_resamples": 2000,
    })
    wandb_env = os.environ.copy()
    wandb_env.update({"WANDB_MODE": "online", "WANDB_ENTITY": "promotion-kim", "WANDB_PROJECT": "mnpo"})
    subprocess.run([
        args.python, str(args.project / "scripts/revision/flagship/log_reward_results_wandb.py"),
        "--summary", str(summary_path), "--stage", "p1-posthoc-all-output-reward",
        "--output", str(work / "wandb_posthoc_all_outputs.json"),
    ], cwd=args.project, env=wandb_env, check=True)
    wandb = json.loads((work / "wandb_posthoc_all_outputs.json").read_text())
    write_report(result_dir, ranked, amendment, gates, wandb)
    ronpo = next(row for row in ranked if row["model"] == "ronpo_k_only")
    atomic_json(status_path, {
        "status": "completed", "stage": "measured_posthoc_all_output_results",
        "generation_started": False, "paper_eligible_as_preregistered_flagship": False,
        "ronpo_k_only_rank": ronpo["worst_objective_rank"],
        "ronpo_k_only_score": ronpo["mean_primary_prompt_worst_norm_score"],
        "ronpo_k_only_ci95": [ronpo["mean_primary_prompt_worst_norm_score_ci95_low"],
                               ronpo["mean_primary_prompt_worst_norm_score_ci95_high"]],
        "wandb_run_id": wandb["wandb_run_id"],
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })


if __name__ == "__main__":
    main()
