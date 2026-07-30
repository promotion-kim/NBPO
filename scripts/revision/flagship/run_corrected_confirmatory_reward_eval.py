#!/usr/bin/env python3
"""One-shot confirmatory reward evaluation after decode-guard validation.

This is intentionally separate from the consumed source-test run.  It uses all
previously unused prompts remaining in the original validation partition and
keeps the already-locked RONPO model selection unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

from run_seed42_sealed_reward_eval import (
    METHODS,
    atomic_json,
    complete,
    frozen_models_tsv,
    run_stability_gates,
    validate_measured_results,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol_lock(path: Path, holdout: Path, expected_prompts: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status": "frozen_after_protocol_validation",
        "confirmatory_holdout_opened": False,
        "model_selection_changed_after_source_test": False,
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise RuntimeError(f"invalid corrected protocol lock field {key}: {payload.get(key)!r}")
    if payload.get("confirmatory_prompt_count") != expected_prompts:
        raise RuntimeError("corrected protocol lock prompt count mismatch")
    digest = sha256_file(holdout)
    if payload.get("confirmatory_file_sha256") != digest:
        raise RuntimeError("confirmatory holdout SHA-256 mismatch")
    if payload.get("decode") != {
        "backend": "vllm", "version": "0.24.0", "seed": 42,
        "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 2048,
        "dtype": "bfloat16", "enable_thinking": False,
        "bad_words": ["<think>", "</think>"],
        "repetition_detection": {"max_pattern_size": 4, "min_pattern_size": 1, "min_count": 20},
    }:
        raise RuntimeError("corrected protocol lock decode does not match runner constants")
    return payload


def write_report(result_dir: Path, ranked: list[dict], gates: dict[str, dict],
                 protocol: dict, models: dict[str, dict], wandb: dict) -> None:
    per_objective: dict[str, dict[str, dict]] = {}
    with (result_dir / "per_objective_scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            per_objective.setdefault(row["model"], {})[row["objective"]] = row
    lines = [
        "# Corrected confirmatory P1 reward report", "",
        "This report is not the consumed source-test run. It evaluates every previously unused prompt",
        "remaining in the original non-training validation partition under a newly frozen common decode guard.",
        "Model selection remains the pre-source-test lock: `ronpo_k_only` (top-mass).", "",
        "| Rank | Model | Worst (95% CI) | Avg | Win vs base | Helpfulness | Safety | Conciseness | Stability |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in ranked:
        model = row["model"]
        objective = per_objective[model]
        win = row.get("mean_win_rate_vs_baseline")
        win_text = "--" if win is None or not math.isfinite(float(win)) else f"{100*float(win):.2f}%"
        lines.append(
            f"| {row['worst_objective_rank']} | {model} | "
            f"{float(row['mean_primary_prompt_worst_norm_score']):.4f} "
            f"[{float(row['mean_primary_prompt_worst_norm_score_ci95_low']):.4f}, "
            f"{float(row['mean_primary_prompt_worst_norm_score_ci95_high']):.4f}] | "
            f"{float(row['mean_primary_prompt_avg_norm_score']):.4f} | {win_text} | "
            f"{float(objective['helpfulness']['mean_prompt_norm_score']):.4f} | "
            f"{float(objective['safety']['mean_prompt_norm_score']):.4f} | "
            f"{float(objective['conciseness']['mean_prompt_norm_score']):.4f} | "
            f"{gates[model].get('status', 'failed')} |"
        )
    lines.extend([
        "", "## Provenance", "",
        f"- Prompt count: {protocol['confirmatory_prompt_count']} (all unused validation-remainder prompts).",
        f"- Prompt-file SHA-256: `{protocol['confirmatory_file_sha256']}`.",
        "- Decode: vLLM 0.24.0; seed 42; temperature 0.7; top-p 0.9; max_new_tokens 2048; "
        "bfloat16; enable_thinking=false; bad_words `<think>`, `</think>`; token repetition detection "
        "pattern sizes 1--4, minimum count 20.",
        "- Reward model: `RLHFlow/ArmoRM-Llama3-8B-v0.1`; heads helpfulness, safety, and negated verbosity.",
        "- Normalization: per-prompt min-max across this fixed 11-model pool.",
        "- Intervals: paired prompt bootstrap, 2,000 resamples, seed 42.",
        f"- W&B: `{wandb.get('wandb_run_id', 'unknown')}` ({wandb.get('wandb_url', 'unknown')}).",
        "- Limitation: this holdout is from the source-train validation partition, not the consumed source-test partition.",
        "- Exact model revisions:",
    ])
    for method in METHODS:
        lines.append(f"  - `{method}`: `{models[method]['repo_id']}@{models[method]['revision']}`")
    lines.append("")
    (result_dir / "CONFIRMATORY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--decode-python", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--models-tsv", type=Path, required=True)
    parser.add_argument("--expected-prompts", type=int, default=1736)
    parser.add_argument("--base-revision", required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / "logs").mkdir(exist_ok=True)
    status = args.work / "status.json"
    protocol = load_protocol_lock(args.protocol_lock, args.prompts, args.expected_prompts)
    models = frozen_models_tsv(args.models_tsv, args.base_revision)
    missing = [method for method in METHODS if method not in models]
    if missing:
        raise RuntimeError(f"frozen model manifest is incomplete: {missing}")

    opened = args.work / "confirmatory_opened.json"
    opened_payload = {
        "opened_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "confirmatory_file_sha256": sha256_file(args.prompts),
        "prompt_count": args.expected_prompts,
        "protocol_lock": protocol,
        "models": models,
        "single_open_policy": True,
        "prior_source_test_status": "consumed_and_failed_closed_before_reward_scoring",
    }
    if opened.exists():
        previous = json.loads(opened.read_text())
        for key in ("confirmatory_file_sha256", "prompt_count", "protocol_lock", "models"):
            if previous.get(key) != opened_payload.get(key):
                raise RuntimeError("confirmatory holdout was already opened with a different manifest")
    else:
        atomic_json(opened, opened_payload)
    atomic_json(status, {"status": "running", "stage": "confirmatory_decode",
                         "confirmatory_holdout_opened": True})

    pending = list(METHODS)
    running: dict[int, tuple[str, subprocess.Popen, object]] = {}
    attempts: dict[str, int] = {}
    failures: list[dict] = []
    while pending or running:
        for gpu, (method, process, handle) in list(running.items()):
            rc = process.poll()
            if rc is None:
                continue
            handle.close()
            output = args.work / "generations" / method / "output_42.json"
            if rc == 0 and complete(output, args.expected_prompts):
                pass
            elif attempts[method] < 2:
                pending.insert(0, method)
            else:
                failures.append({"method": method, "attempt": attempts[method], "returncode": rc})
            del running[gpu]
        for gpu in (0, 1, 2, 3):
            if gpu in running or not pending:
                continue
            method = pending.pop(0)
            output_dir = args.work / "generations" / method
            output = output_dir / "output_42.json"
            if complete(output, args.expected_prompts):
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            attempts[method] = attempts.get(method, 0) + 1
            handle = (args.work / "logs" / f"decode_{method}_a{attempts[method]}.log").open("a")
            model = models[method]
            command = [
                args.decode_python,
                str(args.project / "scripts/revision/flagship/decode_vllm_non_thinking.py"),
                "--data-dir", str(args.prompts), "--model", model["repo_id"],
                "--revision", model["revision"], "--output-dir", str(output_dir),
                "--seed", "42", "--temperature", "0.7", "--top-p", "0.9",
                "--max-new-tokens", "2048", "--forbid-thinking-tags",
                "--repetition-detection-max-pattern-size", "4",
                "--repetition-detection-min-pattern-size", "1",
                "--repetition-detection-min-count", "20",
            ]
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(gpu), "TORCH_CUDNN_SDPA_ENABLED": "0",
                "TOKENIZERS_PARALLELISM": "false", "HF_HOME": str(args.root / "cache/huggingface"),
                "HF_HUB_CACHE": str(args.root / "cache/huggingface/hub"),
            })
            process = subprocess.Popen(command, cwd=args.project, env=env,
                                       stdout=handle, stderr=subprocess.STDOUT)
            running[gpu] = (method, process, handle)
        atomic_json(status, {
            "status": "running", "stage": "confirmatory_decode", "pending": pending,
            "running": [{"gpu": gpu, "method": item[0], "pid": item[1].pid}
                        for gpu, item in running.items()],
            "failures": failures, "confirmatory_holdout_opened": True,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        })
        time.sleep(15)
    if failures:
        atomic_json(status, {"status": "failed", "stage": "confirmatory_decode",
                             "failures": failures, "confirmatory_holdout_opened": True})
        return

    atomic_json(status, {"status": "running", "stage": "confirmatory_stability_gates",
                         "confirmatory_holdout_opened": True})
    args.expected_prompts = int(args.expected_prompts)
    gates = run_stability_gates(args, list(METHODS))
    failed_gates = [method for method, gate in gates.items() if gate.get("passed") is not True]
    if failed_gates:
        atomic_json(status, {"status": "failed", "stage": "confirmatory_stability_gates",
                             "failed_models": failed_gates, "stability_gates": gates,
                             "confirmatory_holdout_opened": True})
        return

    merged = args.work / "merged_generations.json"
    command = [args.python, "-m", "mnpo_scripts.merge_model_generations", "--generations"]
    command.extend([f"{method}={args.work / 'generations' / method / 'output_42.json'}"
                    for method in METHODS])
    command.extend(["--output_file", str(merged)])
    with (args.work / "logs/merge.log").open("a") as handle:
        subprocess.run(command, cwd=args.project, stdout=handle, stderr=subprocess.STDOUT, check=True)

    score_dir = args.work / "scores"
    atomic_json(status, {"status": "running", "stage": "confirmatory_armo_scoring",
                         "confirmatory_holdout_opened": True})
    with (args.work / "logs/score.log").open("a") as handle:
        subprocess.run([
            args.python, str(args.project / "scripts/revision/flagship/score_armo_primary_heads_sharded.py"),
            "--python", args.python, "--input-file", str(merged), "--output-dir", str(score_dir),
            "--cache-dir", str(args.root / "cache/huggingface/hub"),
            "--gpu-ids", "0", "1", "2", "3", "--batch-size", "8",
            "--sample-batch-size", "4", "--local-files-only",
        ], cwd=args.project, stdout=handle, stderr=subprocess.STDOUT, check=True)

    result_dir = args.work / "results"
    atomic_json(status, {"status": "running", "stage": "confirmatory_aggregation",
                         "confirmatory_holdout_opened": True})
    with (args.work / "logs/evaluate.log").open("a") as handle:
        subprocess.run([
            args.python, "-m", "mnpo_scripts.evaluate_multi_objective_models",
            "--scored_files", f"helpfulness={score_dir / 'helpfulness.jsonl'}",
            f"safety={score_dir / 'safety.jsonl'}", f"conciseness={score_dir / 'conciseness.jsonl'}",
            "--output_dir", str(result_dir), "--baseline_model", "base",
            "--primary_objectives", "helpfulness", "safety", "conciseness",
            "--bootstrap_samples", "2000", "--bootstrap_seed", "42",
        ], cwd=args.project, stdout=handle, stderr=subprocess.STDOUT, check=True)
    rows = json.loads((result_dir / "model_summary.json").read_text())
    validate_measured_results(result_dir, rows, list(METHODS), args.expected_prompts)
    ranked = sorted(rows, key=lambda row: (-float(row["mean_primary_prompt_worst_norm_score"]), row["model"]))
    values = [float(row["mean_primary_prompt_worst_norm_score"]) for row in ranked]
    for row, value in zip(ranked, values):
        row["worst_objective_rank"] = 1 + sum(other > value + 1e-12 for other in values)
    atomic_json(result_dir / "ranked_confirmatory_summary.json", {
        "evaluation_split": "all unused prompts in original non-training validation remainder",
        "metric": "mean_primary_prompt_worst_norm_score", "ranked": ranked,
        "bootstrap_resamples": 2000, "prompt_count": args.expected_prompts,
        "prompt_file_sha256": protocol["confirmatory_file_sha256"],
        "selected_ronpo_variant": "top-mass", "selected_model_name": "ronpo_k_only",
        "stability_gates": {method: gate.get("status") for method, gate in gates.items()},
        "prior_source_test": "consumed_and_failed_closed_before_reward_scoring",
    })
    with (result_dir / "headline_table.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["worst_objective_rank", "model", "mean_primary_prompt_worst_norm_score",
                  "mean_primary_prompt_worst_norm_score_ci95_low",
                  "mean_primary_prompt_worst_norm_score_ci95_high",
                  "mean_primary_prompt_avg_norm_score", "num_prompts"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranked)

    wandb_env = os.environ.copy()
    wandb_env.update({"WANDB_MODE": "online", "WANDB_ENTITY": "promotion-kim", "WANDB_PROJECT": "mnpo"})
    subprocess.run([
        args.python, str(args.project / "scripts/revision/flagship/log_reward_results_wandb.py"),
        "--summary", str(result_dir / "ranked_confirmatory_summary.json"),
        "--stage", "p1-corrected-confirmatory-reward", "--output", str(args.work / "wandb_run.json"),
    ], cwd=args.project, env=wandb_env, check=True)
    wandb = json.loads((args.work / "wandb_run.json").read_text())
    write_report(result_dir, ranked, gates, protocol, models, wandb)
    ronpo = next(row for row in ranked if row["model"] == "ronpo_k_only")
    atomic_json(status, {
        "status": "completed", "stage": "measured_corrected_confirmatory_results",
        "ronpo_worst_objective_rank": ronpo["worst_objective_rank"],
        "ronpo_worst_objective_score": ronpo["mean_primary_prompt_worst_norm_score"],
        "ronpo_worst_objective_ci95": [ronpo["mean_primary_prompt_worst_norm_score_ci95_low"],
                                       ronpo["mean_primary_prompt_worst_norm_score_ci95_high"]],
        "stability_gates": {method: gate.get("status") for method, gate in gates.items()},
        "confirmatory_holdout_opened": True,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })


if __name__ == "__main__":
    main()
