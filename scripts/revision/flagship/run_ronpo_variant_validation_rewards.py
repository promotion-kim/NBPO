#!/usr/bin/env python3
"""Score validation checkpoints with the locked RM metric and select without panel access."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(bool(line.strip()) for line in path.open(encoding="utf-8"))


def run(command: list[str], cwd: Path, log: Path, env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def raw_paired_summary(path: Path, models: list[str], bootstrap: int = 2000) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 128 or any(row["response_model_names"] != models for row in rows):
        raise RuntimeError(f"unaligned independent RM output {path}")
    values = np.asarray([row["all_rm_scores"] for row in rows], dtype=float)
    if values.shape != (128, len(models)) or not np.isfinite(values).all():
        raise RuntimeError(f"invalid independent RM matrix {path}")
    base = models.index("base")
    rng = np.random.default_rng(42)
    indices = rng.integers(0, 128, size=(bootstrap, 128), dtype=np.int32)
    output = []
    for index, model in enumerate(models):
        delta = values[:, index] - values[:, base]
        boot = delta[indices].mean(axis=1)
        output.append({
            "model": model, "mean_raw_score": float(values[:, index].mean()),
            "mean_paired_delta_vs_base": float(delta.mean()),
            "paired_delta_vs_base_ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--flagship-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluator-lock", type=Path, required=True)
    parser.add_argument("--general-rm-cache", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    evaluator = json.loads(args.evaluator_lock.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_validation_decode_and_ranking":
        raise RuntimeError("checkpoint manifest is not frozen")
    if evaluator.get("status") != "LOCKED_BEFORE_ANY_NEW_METHOD_RANKING":
        raise RuntimeError("prospective evaluator is not locked")
    gates = json.loads((args.work / "stability_gates/summary.json").read_text(encoding="utf-8"))
    eligible = gates["eligible_models"]
    if "base" not in eligible or len(eligible) < 2:
        raise RuntimeError("no eligible checkpoint")
    generations = args.work / "generations"
    logs = args.work / "logs"
    merged = args.work / "merged_generations.json"
    merge = [args.python, "-m", "mnpo_scripts.merge_model_generations", "--generations"]
    merge.extend([f"{model}={generations / model / 'output_42.json'}" for model in eligible])
    merge.extend(["--output_file", str(merged)])
    run(merge, args.project, logs / "merge.log")
    merged_rows = json.loads(merged.read_text(encoding="utf-8"))
    if len(merged_rows) != 128:
        raise RuntimeError("validation merged prompt count is not 128")
    model_order = merged_rows[0]["response_model_names"]
    if model_order != eligible:
        raise RuntimeError("merged model order differs from gate order")

    scores = args.work / "scores"; scores.mkdir(parents=True, exist_ok=True)
    armo_outputs = [scores / "armo/helpfulness.jsonl", scores / "armo/safety.jsonl",
                    scores / "armo/conciseness.jsonl"]
    jobs = []
    common_env = os.environ.copy()
    common_env.update({"PYTHONPATH": str(args.project), "HF_HUB_OFFLINE": "1",
                       "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                       "TORCH_CUDNN_SDPA_ENABLED": "0"})
    if not all(count_jsonl(path) == 128 for path in armo_outputs):
        command = [
            args.python, str(args.project / "scripts/revision/flagship/score_armo_primary_heads_sharded.py"),
            "--python", args.python, "--input-file", str(merged), "--output-dir", str(scores / "armo"),
            "--cache-dir", str(args.flagship_root / "cache/huggingface/hub"),
            "--gpu-ids", "0", "1", "--batch-size", "8", "--sample-batch-size", "4",
            "--local-files-only",
        ]
        handle = (logs / "score_armo.log").open("a", encoding="utf-8")
        process = subprocess.Popen(command, cwd=args.project, env=common_env,
                                   stdout=handle, stderr=subprocess.STDOUT)
        jobs.append(("armo", process, handle, command))
    for name, gpu, module, model, revision, extra in [
        ("skywork", "2", "on_policy_data_gen.rm_skywork",
         "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
         "cba2f842f3f1af2f1b2f0d35e794d789976390c5",
         ["--max_seq_length", "4096", "--attn_implementation", "sdpa"]),
        ("athene", "3", "on_policy_data_gen.rm_athene",
         "Nexusflow/Athene-RM-8B", "cdf428f7b52a323b6cf4e9803e5bcba9f1fb5a59", []),
    ]:
        output = scores / f"{name}.jsonl"
        if count_jsonl(output) == 128:
            continue
        command = [args.python, "-u", "-m", module, "--input_file", str(merged),
                   "--output_file", str(output), "--cache_dir", str(args.general_rm_cache),
                   "--model_name", model, "--revision", revision, "--local_files_only",
                   "--batch_size", "16", "--sample_batch_size", "8", *extra]
        environment = common_env.copy(); environment["CUDA_VISIBLE_DEVICES"] = gpu
        handle = (logs / f"score_{name}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(command, cwd=args.project, env=environment,
                                   stdout=handle, stderr=subprocess.STDOUT)
        jobs.append((name, process, handle, command))
    atomic_json(args.work / "scoring_status.json", {
        "status": "running", "stage": "locked_validation_reward_scoring",
        "eligible_models": eligible, "panel_judgments_accessed": False,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "spent_sealed_split_touched": False,
    })
    failures = []
    for name, process, handle, command in jobs:
        returncode = process.wait(); handle.close()
        if returncode:
            failures.append({"evaluator": name, "returncode": returncode, "command": command})
    if failures:
        atomic_json(args.work / "scoring_status.json", {"status": "failed",
                    "stage": "locked_validation_reward_scoring", "failures": failures,
                    "spent_sealed_split_touched": False})
        raise RuntimeError(json.dumps(failures, indent=2))
    required = {"skywork": scores / "skywork.jsonl", "athene": scores / "athene.jsonl",
                "armo_helpfulness": scores / "armo/helpfulness.jsonl",
                "armo_safety": scores / "armo/safety.jsonl",
                "armo_conciseness": scores / "armo/conciseness.jsonl"}
    if any(count_jsonl(path) != 128 for path in required.values()):
        raise RuntimeError("one or more reward score files is incomplete")

    locked_dir = args.work / "results/locked_selection_metric"
    run([args.python, str(args.project / "scripts/revision/flagship/aggregate_fair_demo_rewards.py"),
         "--merged", str(merged), "--evaluator-lock", str(args.evaluator_lock),
         "--output-dir", str(locked_dir),
         "--score", f"skywork={required['skywork']}",
         "--score", f"armo_safety={required['armo_safety']}"],
        args.project, logs / "aggregate_locked_metric.log")
    armo_dir = args.work / "results/armo_table4_validation"
    run([args.python, "-m", "mnpo_scripts.evaluate_multi_objective_models",
         "--scored_files", f"helpfulness={required['armo_helpfulness']}",
         f"safety={required['armo_safety']}", f"conciseness={required['armo_conciseness']}",
         "--output_dir", str(armo_dir), "--baseline_model", "base",
         "--primary_objectives", "helpfulness", "safety", "conciseness",
         "--bootstrap_samples", "2000", "--bootstrap_seed", "42"],
        args.project, logs / "aggregate_armo_table4.log")

    locked = json.loads((locked_dir / "reward_summary.json").read_text(encoding="utf-8"))
    metric_by_model = {row["model"]: row for row in locked["ranked_secondary"]}
    independent = raw_paired_summary(required["athene"], model_order)
    independent_by_model = {row["model"]: row for row in independent}
    armo_rows = json.loads((armo_dir / "model_summary.json").read_text(encoding="utf-8"))
    armo_by_model = {row["model"]: row for row in armo_rows}
    manifest_by_model = {row["model_id"]: row for row in manifest["models"]}
    candidate_ids = sorted({row["candidate_id"] for row in manifest["models"]})
    selected = []
    all_rows = []
    for candidate_id in candidate_ids:
        rows = []
        for model_id in eligible:
            record = manifest_by_model.get(model_id)
            if record is None or record["candidate_id"] != candidate_id:
                continue
            metric = metric_by_model[model_id]
            row = {**record,
                   "s3_pass": True,
                   "selection_metric": metric["mean_prompt_worst_standardized_delta"],
                   "selection_metric_ci95": metric["mean_prompt_worst_standardized_delta_ci95"],
                   "athene": independent_by_model[model_id],
                   "armo_table4_validation": armo_by_model[model_id]}
            rows.append(row); all_rows.append(row)
        if not rows:
            selected.append({"candidate_id": candidate_id, "status": "terminal_failed_all_checkpoints"})
            continue
        best = sorted(rows, key=lambda row: (-float(row["selection_metric"]), int(row["step"])))[0]
        selected.append({**best, "status": "selected_on_validation_locked_rm_metric"})
    eligible_selected = [row for row in selected if row.get("status") == "selected_on_validation_locked_rm_metric"]
    overall = (sorted(eligible_selected,
                      key=lambda row: (-float(row["selection_metric"]), row["candidate_id"]))[0]
               if eligible_selected else None)
    selection = {
        "status": "locked_after_validation_rm_scoring_before_any_variant_panel_judgment",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_split": "existing prompt-disjoint 128-prompt validation",
        "selection_metric": evaluator["selection"]["checkpoint_metric"],
        "panel_judgments_used_for_selection": False,
        "per_candidate_selected_checkpoint": selected,
        "overall_validation_incumbent": overall,
        "all_eligible_checkpoint_rows": all_rows,
        "failed_gate_models": gates["failed_models"],
        "bootstrap": {"resamples": 2000, "seed": 42, "unit": "prompt", "paired": True},
        "spent_sealed_split_touched": False,
    }
    output = args.work / "results/selection_lock.json"
    atomic_json(output, selection)
    fields = ["candidate_id", "model_id", "round", "step", "selection_metric",
              "selection_metric_ci95", "s3_pass", "wandb_run_id"]
    with (args.work / "results/all_checkpoint_selection_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(all_rows)
    atomic_json(args.work / "results/athene_secondary.json", {
        "status": "completed", "rows": independent, "bootstrap_resamples": 2000,
        "bootstrap_seed": 42, "spent_sealed_split_touched": False})
    atomic_json(args.work / "scoring_status.json", {
        "status": "completed", "stage": "validation_rm_selection_locked",
        "selection_lock": str(output), "overall_validation_incumbent": overall,
        "panel_judgments_accessed": False,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "spent_sealed_split_touched": False,
    })
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
