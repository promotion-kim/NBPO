#!/usr/bin/env python3
"""Score the already-generated, gate-eligible sealed responses without decoding."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ALL_METHODS = (
    "base", "ronpo_full_expect", "ronpo_k_only", "dpo", "ipo", "simpo",
    "sppo_avg", "inpo_avg", "ht_mnpo_helpfulness", "ht_mnpo_safety",
    "ht_mnpo_conciseness",
)
ELIGIBLE_METHODS = tuple(method for method in ALL_METHODS if method != "dpo")
OBJECTIVES = ("helpfulness", "safety", "conciseness")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_models_tsv(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    models = {
        row["name"]: {
            "repo_id": row["model"],
            "revision": row["revision"],
            "seed": int(row["seed"]) if row.get("seed") else None,
        }
        for row in rows if row.get("name") in ALL_METHODS
    }
    if set(models) != set(ALL_METHODS):
        raise RuntimeError(f"models.tsv mismatch: {sorted(set(ALL_METHODS) - set(models))}")
    return models


def validate_preflight(work: Path) -> tuple[dict, dict]:
    audit = json.loads((work / "gate_correction.json").read_text())
    gates = json.loads((work / "stability_gates_corrected/summary.json").read_text())
    if audit.get("go_signal_for_reward_scoring") is not True:
        raise RuntimeError("gate_correction.json does not authorize reward scoring")
    measured_eligible = tuple(gates.get("eligible_models", []))
    if set(measured_eligible) != set(ELIGIBLE_METHODS):
        raise RuntimeError(f"corrected eligible set mismatch: {measured_eligible}")
    if gates["models"]["dpo"].get("passed") is not False:
        raise RuntimeError("DPO must remain fail-closed")
    for method in ALL_METHODS:
        generation = work / "generations" / method / "output_42.json"
        records = json.loads(generation.read_text())
        expected = audit["generation_artifacts"][method]
        if len(records) != 604 or expected["records"] != 604:
            raise RuntimeError(f"wrong sealed record count for {method}")
        if sha256(generation) != expected["sha256"]:
            raise RuntimeError(f"generation SHA-256 changed after gate correction: {method}")
    return audit, gates


def validate_results(result_dir: Path, rows: list[dict]) -> None:
    if len(rows) != len(ELIGIBLE_METHODS) or {row.get("model") for row in rows} != set(ELIGIBLE_METHODS):
        raise RuntimeError("model_summary does not contain exactly the corrected eligible pool")
    required = (
        "mean_primary_prompt_worst_norm_score",
        "mean_primary_prompt_worst_norm_score_ci95_low",
        "mean_primary_prompt_worst_norm_score_ci95_high",
        "mean_primary_prompt_avg_norm_score",
        "mean_primary_prompt_avg_norm_score_ci95_low",
        "mean_primary_prompt_avg_norm_score_ci95_high",
    )
    for row in rows:
        if int(row.get("num_prompts", 0)) != 604:
            raise RuntimeError(f"wrong prompt count for {row.get('model')}")
        for field in required:
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f"non-finite {field} for {row.get('model')}")
    per_objective = list(csv.DictReader(
        (result_dir / "per_objective_scores.csv").open(newline="", encoding="utf-8")
    ))
    expected = {(method, objective) for method in ELIGIBLE_METHODS for objective in OBJECTIVES}
    actual = {(row.get("model"), row.get("objective")) for row in per_objective}
    if len(per_objective) != len(expected) or actual != expected:
        raise RuntimeError("per-objective score table is incomplete or duplicated")
    for row in per_objective:
        for field in (
            "mean_prompt_norm_score", "mean_prompt_norm_score_ci95_low",
            "mean_prompt_norm_score_ci95_high", "mean_raw_score",
            "mean_raw_score_ci95_low", "mean_raw_score_ci95_high",
        ):
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f"non-finite {field} for {row.get('model')}")


def report_markdown(
    path: Path,
    ranked: list[dict],
    per_objective_path: Path,
    gates: dict,
    models: dict,
    wandb: dict,
    sealed_sha256: str,
    armo_revision: str,
    generation_hashes: dict,
) -> None:
    per_objective: dict[str, dict[str, dict]] = {}
    with per_objective_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            per_objective.setdefault(row["model"], {})[row["objective"]] = row
    lines = [
        "# P1 sealed reward report",
        "",
        "Selected RONPO variant (locked before sealed access): `ronpo_k_only` (`top-mass`).",
        "",
        "The corrected stability rule was finalized before reward scoring. DPO was excluded "
        "because record 252 has a genuine 1,163-token repeat run.",
        "",
        "| Rank | Model | Worst (95% CI) | Avg | Win vs base | Helpfulness | Safety | Conciseness | Stability |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in ranked:
        model = row["model"]
        objectives = per_objective[model]
        low = float(row["mean_primary_prompt_worst_norm_score_ci95_low"])
        high = float(row["mean_primary_prompt_worst_norm_score_ci95_high"])
        win = row.get("mean_win_rate_vs_baseline")
        win_text = "--" if win is None else f"{100 * float(win):.2f}%"
        values = [float(objectives[name]["mean_prompt_norm_score"]) for name in OBJECTIVES]
        lines.append(
            f"| {row['worst_objective_rank']} | {model} | "
            f"{float(row['mean_primary_prompt_worst_norm_score']):.4f} [{low:.4f}, {high:.4f}] | "
            f"{float(row['mean_primary_prompt_avg_norm_score']):.4f} | {win_text} | "
            f"{values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | passed |"
        )
    lines.extend([
        "| -- | dpo | -- | -- | -- | -- | -- | -- | failed: repeat run 1,163 at index 252 |",
        "",
        "## Provenance",
        "",
        "- Prompt count: 604",
        f"- Sealed prompt SHA-256: `{sealed_sha256}`",
        "- No sealed decoding was run during this resume. The preserved generations listed below were reused.",
        "- Decode: vLLM; seed 42; temperature 0.7; top-p 0.9; max_new_tokens 2048; "
        "chat template; enable_thinking=false; bfloat16.",
        f"- Reward model: `RLHFlow/ArmoRM-Llama3-8B-v0.1@{armo_revision}`.",
        "- Heads: `ultrafeedback-helpfulness`, `beavertails-is_safe`, and negated `helpsteer-verbosity`.",
        "- Normalization: per-prompt min-max over the ten corrected-gate-eligible sealed models.",
        "- Intervals: 2,000-resample paired prompt bootstrap, seed 42.",
        "- Gate audit: `../gate_correction.json`; original failed gate JSON files are preserved.",
        f"- W&B run ID: `{wandb.get('wandb_run_id', 'unknown')}` "
        f"({wandb.get('wandb_url', 'unknown')})",
        "- Exact model revisions and generation SHA-256 values:",
    ])
    for method in ALL_METHODS:
        lines.append(
            f"  - `{method}`: `{models[method]['repo_id']}@{models[method]['revision']}`; "
            f"generation `{generation_hashes[method]['sha256']}`"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--models-tsv", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sealed-sha256", required=True)
    parser.add_argument("--armo-revision", required=True)
    args = parser.parse_args()
    result_dir = args.work / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    logs = args.work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = args.work / "status.json"

    audit, gates = validate_preflight(args.work)
    models = load_models_tsv(args.models_tsv)
    atomic_json(status_path, {
        "status": "running",
        "stage": "sealed_score_only_merge",
        "decode_invoked": False,
        "eligible_models": list(ELIGIBLE_METHODS),
        "failed_models": ["dpo"],
        "p1_sealed_test_opened": True,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })

    merged = args.work / "merged_generations.json"
    merge_command = [args.python, "-m", "mnpo_scripts.merge_model_generations", "--generations"]
    merge_command.extend([
        f"{method}={args.work / 'generations' / method / 'output_42.json'}"
        for method in ELIGIBLE_METHODS
    ])
    merge_command.extend(["--output_file", str(merged)])
    with (logs / "merge_score_only.log").open("a") as handle:
        subprocess.run(merge_command, cwd=args.project, stdout=handle, stderr=subprocess.STDOUT, check=True)
    merged_rows = json.loads(merged.read_text())
    if len(merged_rows) != 604 or any(row.get("response_model_names") != list(ELIGIBLE_METHODS) for row in merged_rows):
        raise RuntimeError("merged generation input failed count or model-order validation")

    atomic_json(status_path, {
        "status": "running",
        "stage": "sealed_score_only_armo",
        "decode_invoked": False,
        "eligible_models": list(ELIGIBLE_METHODS),
        "failed_models": ["dpo"],
        "p1_sealed_test_opened": True,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    score_dir = args.work / "scores"
    with (logs / "score_only.log").open("a") as handle:
        subprocess.run([
            args.python,
            str(args.project / "scripts/revision/flagship/score_armo_primary_heads_sharded.py"),
            "--python", args.python,
            "--input-file", str(merged),
            "--output-dir", str(score_dir),
            "--cache-dir", str(args.root / "cache/huggingface/hub"),
            "--gpu-ids", "0", "1", "2", "3",
            "--batch-size", "8",
            "--sample-batch-size", "4",
            "--local-files-only",
        ], cwd=args.project, stdout=handle, stderr=subprocess.STDOUT, check=True)
    score_metadata = json.loads((score_dir / "score_metadata.json").read_text())
    if score_metadata.get("num_prompts") != 604 or score_metadata.get("response_model_names") != list(ELIGIBLE_METHODS):
        raise RuntimeError("ArmoRM score metadata mismatch")
    score_metadata["revision"] = args.armo_revision
    atomic_json(score_dir / "score_metadata.json", score_metadata)

    atomic_json(status_path, {
        "status": "running",
        "stage": "sealed_score_only_aggregation",
        "decode_invoked": False,
        "p1_sealed_test_opened": True,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    with (logs / "evaluate_score_only.log").open("a") as handle:
        subprocess.run([
            args.python, "-m", "mnpo_scripts.evaluate_multi_objective_models",
            "--scored_files",
            f"helpfulness={score_dir / 'helpfulness.jsonl'}",
            f"safety={score_dir / 'safety.jsonl'}",
            f"conciseness={score_dir / 'conciseness.jsonl'}",
            "--output_dir", str(result_dir),
            "--baseline_model", "base",
            "--primary_objectives", "helpfulness", "safety", "conciseness",
            "--bootstrap_samples", "2000",
            "--bootstrap_seed", "42",
        ], cwd=args.project, stdout=handle, stderr=subprocess.STDOUT, check=True)
    rows = json.loads((result_dir / "model_summary.json").read_text())
    validate_results(result_dir, rows)
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
        "final_selection": json.loads((args.work / "selection_lock.json").read_text()),
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
    atomic_json(status_path, {
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
    try:
        main()
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise
