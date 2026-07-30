#!/usr/bin/env python3
"""Decode, gate, and score stronger 8B models on validation only."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def complete_generation(path: Path) -> bool:
    try:
        rows = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(rows, list) and len(rows) == 128 and all(
        str(row.get("generated_text", "")).strip() for row in rows
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--flagship-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-python", required=True)
    parser.add_argument("--eval-python", required=True)
    parser.add_argument("--general-rm-cache", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("status") != "frozen_before_training":
        raise RuntimeError("stronger protocol is not frozen")
    methods = [row["name"] for row in config["methods"]]
    training = json.loads((args.root / "training_manifest.json").read_text())
    completed_methods = [method for method in methods
                         if training.get("methods", {}).get(method, {}).get("status") == "completed"]
    if not completed_methods:
        raise RuntimeError("no stronger model completed training")
    work = args.root / "validation"
    generations = work / "generations"
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status = work / "status.json"
    base_source = args.flagship_root / "eval/p1_validation_reward_seed42/generations/base/output_42.json"
    base_metadata = base_source.parent / "decode_metadata.json"
    if not complete_generation(base_source):
        raise RuntimeError("the frozen base validation generation is unavailable")
    base_output = generations / "base/output_42.json"
    base_output.parent.mkdir(parents=True, exist_ok=True)
    if not complete_generation(base_output):
        shutil.copy2(base_source, base_output)
        if base_metadata.is_file():
            shutil.copy2(base_metadata, base_output.parent / "decode_metadata.json")

    running = []
    for method_row in config["methods"]:
        method = method_row["name"]
        if method not in completed_methods:
            continue
        output_dir = generations / method
        output = output_dir / "output_42.json"
        if complete_generation(output):
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            args.eval_python, str(args.project / "scripts/revision/flagship/decode_vllm_non_thinking.py"),
            "--data-dir", str(args.flagship_root / "data/pool_validation.jsonl"),
            "--model", str(args.root / "train" / method),
            "--output-dir", str(output_dir), "--seed", "42", "--temperature", "0.7",
            "--top-p", "0.9", "--max-new-tokens", "2048", "--max-prompts", "128",
            "--gpu-memory-utilization", "0.75",
        ]
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": str(method_row["gpu"]),
            "HF_HOME": str(args.flagship_root / "cache/huggingface"),
            "HF_HUB_CACHE": str(args.flagship_root / "cache/huggingface/hub"),
            "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false", "TORCH_CUDNN_SDPA_ENABLED": "0",
        })
        handle = (logs / f"decode_{method}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(command, cwd=args.project, env=environment,
                                   stdout=handle, stderr=subprocess.STDOUT)
        running.append((method, process, handle, output, command))
    atomic_json(status, {"status": "running", "stage": "validation_decode",
                         "methods": completed_methods, "spent_sealed_split_touched": False,
                         "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
    decode_failures = []
    for method, process, handle, output, command in running:
        returncode = process.wait(); handle.close()
        if returncode != 0 or not complete_generation(output):
            decode_failures.append({"method": method, "returncode": returncode, "command": command})
    if decode_failures:
        atomic_json(status, {"status": "failed", "stage": "validation_decode",
                             "failures": decode_failures, "spent_sealed_split_touched": False})
        raise RuntimeError(f"validation decode failures: {decode_failures}")

    gate_dir = work / "stability_gates"
    gate_dir.mkdir(parents=True, exist_ok=True)
    gates = {"detector": "corrected_nonempty_paired_span_v1", "models": {}}
    for method in ["base", *completed_methods]:
        output = gate_dir / f"{method}.json"
        command = [
            args.train_python, str(args.project / "scripts/revision/flagship/stability_gate_corrected.py"),
            "--base", str(base_output), "--candidate", str(generations / method / "output_42.json"),
            "--output", str(output), "--min-length-ratio", "0.33", "--max-length-ratio", "2.0",
            "--max-repeat-run", "20", "--expected-records", "128",
        ]
        result = subprocess.run(command, cwd=args.project, capture_output=True, text=True)
        (logs / f"gate_{method}.log").write_text(result.stdout + result.stderr)
        payload = json.loads(output.read_text()) if output.is_file() else {"passed": False}
        gates["models"][method] = {"passed": payload.get("passed") is True,
                                    "returncode": result.returncode, "artifact": str(output)}
    eligible = [method for method in ["base", *completed_methods] if gates["models"][method]["passed"]]
    gates["eligible_models"] = eligible
    gates["failed_models"] = [method for method in ["base", *completed_methods] if method not in eligible]
    atomic_json(gate_dir / "summary.json", gates)
    if "base" not in eligible:
        raise RuntimeError("base failed validation stability gate")

    merged = work / "merged_generations.json"
    merge = [args.train_python, "-m", "mnpo_scripts.merge_model_generations", "--generations"]
    merge.extend([f"{method}={generations / method / 'output_42.json'}" for method in eligible])
    merge.extend(["--output_file", str(merged)])
    with (logs / "merge.log").open("a") as handle:
        subprocess.run(merge, cwd=args.project, stdout=handle, stderr=subprocess.STDOUT, check=True)

    scores = work / "scores"
    armo = [
        args.train_python, str(args.project / "scripts/revision/flagship/score_armo_primary_heads_sharded.py"),
        "--python", args.train_python, "--input-file", str(merged),
        "--output-dir", str(scores / "armo"),
        "--cache-dir", str(args.flagship_root / "cache/huggingface/hub"),
        "--gpu-ids", "0", "1", "--batch-size", "8", "--sample-batch-size", "4", "--local-files-only",
    ]
    general = {
        "skywork": [args.eval_python, "-u", "-m", "on_policy_data_gen.rm_skywork",
                    "--input_file", str(merged), "--output_file", str(scores / "skywork.jsonl"),
                    "--cache_dir", str(args.general_rm_cache),
                    "--model_name", "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
                    "--revision", "cba2f842f3f1af2f1b2f0d35e794d789976390c5", "--local_files_only",
                    "--batch_size", "16", "--sample_batch_size", "16",
                    "--max_seq_length", "4096", "--attn_implementation", "sdpa"],
        "athene": [args.eval_python, "-u", "-m", "on_policy_data_gen.rm_athene",
                   "--input_file", str(merged), "--output_file", str(scores / "athene.jsonl"),
                   "--cache_dir", str(args.general_rm_cache), "--model_name", "Nexusflow/Athene-RM-8B",
                   "--revision", "cdf428f7b52a323b6cf4e9803e5bcba9f1fb5a59", "--local_files_only",
                   "--batch_size", "16", "--sample_batch_size", "16"],
    }
    scores.mkdir(parents=True, exist_ok=True)
    score_jobs = []
    for name, command, gpu in [("armo", armo, "0,1"), ("skywork", general["skywork"], "2"),
                               ("athene", general["athene"], "3")]:
        environment = os.environ.copy()
        environment.update({"CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": str(args.project),
                            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                            "TOKENIZERS_PARALLELISM": "false", "TORCH_CUDNN_SDPA_ENABLED": "0"})
        handle = (logs / f"score_{name}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(command, cwd=args.project, env=environment,
                                   stdout=handle, stderr=subprocess.STDOUT)
        score_jobs.append((name, process, handle, command))
    atomic_json(status, {"status": "running", "stage": "five_evaluator_scoring",
                         "eligible_models": eligible, "spent_sealed_split_touched": False,
                         "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
    score_failures = []
    for name, process, handle, command in score_jobs:
        returncode = process.wait(); handle.close()
        if returncode != 0:
            score_failures.append({"evaluator": name, "returncode": returncode, "command": command})
    if score_failures:
        atomic_json(status, {"status": "failed", "stage": "five_evaluator_scoring",
                             "failures": score_failures, "spent_sealed_split_touched": False})
        raise RuntimeError(f"reward scoring failures: {score_failures}")

    normalized_dir = work / "results/armo_normalized"
    with (logs / "normalize_armo.log").open("a") as handle:
        subprocess.run([
            args.train_python, "-m", "mnpo_scripts.evaluate_multi_objective_models",
            "--scored_files", f"helpfulness={scores / 'armo/helpfulness.jsonl'}",
            f"safety={scores / 'armo/safety.jsonl'}", f"conciseness={scores / 'armo/conciseness.jsonl'}",
            "--output_dir", str(normalized_dir), "--baseline_model", "base",
            "--primary_objectives", "helpfulness", "safety", "conciseness",
            "--bootstrap_samples", "2000", "--bootstrap_seed", "42",
        ], cwd=args.project, stdout=handle, stderr=subprocess.STDOUT, check=True)
    rows = json.loads((normalized_dir / "model_summary.json").read_text())
    ranked = sorted(rows, key=lambda row: (-float(row["mean_primary_prompt_worst_norm_score"]), row["model"]))
    for rank, row in enumerate(ranked, start=1):
        row["validation_worst_objective_rank"] = rank
    ranked_path = normalized_dir / "ranked_validation_summary.json"
    atomic_json(ranked_path, {"split": "non-sealed 128-prompt validation",
                              "metric": "mean_primary_prompt_worst_norm_score",
                              "ranked": ranked, "spent_sealed_split_touched": False})
    results = work / "results"
    results.mkdir(parents=True, exist_ok=True)
    aggregate = [
        args.train_python, str(args.project / "scripts/revision/flagship/aggregate_stronger_validation_power.py"),
        "--scored", f"armo_helpfulness={scores / 'armo/helpfulness.jsonl'}",
        f"armo_safety={scores / 'armo/safety.jsonl'}",
        f"armo_conciseness={scores / 'armo/conciseness.jsonl'}",
        f"skywork={scores / 'skywork.jsonl'}", f"athene={scores / 'athene.jsonl'}",
        "--normalized-summary", str(ranked_path), "--gates-summary", str(gate_dir / "summary.json"),
        "--config", str(args.config), "--output-dir", str(results),
        "--bootstrap", "2000", "--seed", "42", "--wandb",
    ]
    with (logs / "aggregate.log").open("a") as handle:
        subprocess.run(aggregate, cwd=args.project, stdout=handle, stderr=subprocess.STDOUT, check=True)
    power = json.loads((results / "raw_power_summary.json").read_text())
    atomic_json(status, {"status": "completed", "stage": "validation_power_check",
                         "eligible_models": eligible, "failed_gates": gates["failed_models"],
                         "any_detectable_raw_delta_vs_base": power["any_detectable_raw_delta_vs_base"],
                         "armo_raw_delta_detectable": power["armo_raw_delta_detectable"],
                         "spent_sealed_split_touched": False,
                         "completed_at": datetime.now().astimezone().isoformat(timespec="seconds")})


if __name__ == "__main__":
    main()
