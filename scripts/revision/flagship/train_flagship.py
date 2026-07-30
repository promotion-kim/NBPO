#!/usr/bin/env python3
"""GPU-aware matched-budget P1 training orchestrator with idempotent status files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SEEDS = (42, 43, 44)
RONPO = ("ronpo_full_expect", "ronpo_k_only")
BASELINES = (
    "dpo", "ipo", "simpo", "kto", "sppo_avg", "inpo_avg",
    "ht_mnpo_helpfulness", "ht_mnpo_safety", "ht_mnpo_conciseness",
)
METHODS = RONPO + BASELINES


@dataclass
class Running:
    method: str
    seed: int
    gpu: int
    gpus: tuple[int, ...]
    process: subprocess.Popen[Any]
    log_handle: Any
    log_path: Path
    output_dir: Path
    wandb_id: str
    started: float


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def run_id(stage: str, method: str, seed: int, attempt: int) -> str:
    return hashlib.sha256(f"aaai27-resource-v9|{stage}|{method}|{seed}|{attempt}".encode()).hexdigest()[:12]


def unified_config(
    model: str, dataset: str, output: Path, stage: str, method: str, seed: int, attempt: int,
) -> dict[str, Any]:
    loss = {
        "ronpo_full_expect": "ronpo", "ronpo_k_only": "ronpo", "dpo": "dpo", "ipo": "ipo",
        "simpo": "simpo", "sppo_avg": "sppo", "inpo_avg": "inpo",
        "ht_mnpo_helpfulness": "ht_mnpo", "ht_mnpo_safety": "ht_mnpo",
        "ht_mnpo_conciseness": "ht_mnpo",
    }[method]
    # Smoke attempt 2 is a resource-only fallback after the measured
    # microbatch-2 OOM. Full training starts with that validated resource
    # profile, while its attempt number continues to mean method stability.
    stability_attempt = max(1, attempt - 1) if stage == "smoke" else attempt
    # The 2026-07-13 continuation freezes every full-training attempt at the
    # validated low-memory profile.  Retry number changes only the registered
    # stabilization recipe (LR/anchors), never the resource profile.
    safe_resource_profile = stage == "full" or attempt > 1
    lr = 1.0e-7 if method in RONPO else 5.0e-7
    if stability_attempt > 1:
        lr /= 2 ** (stability_attempt - 1)
    config: dict[str, Any] = {
        "model_name_or_path": model,
        "torch_dtype": None,
        "attn_implementation": "sdpa",
        "dataset_mixer": {dataset: 1.0},
        "dataset_splits": ["train", "test"],
        "preprocessing_num_workers": 4,
        "bf16": True,
        "loss_type": loss,
        "eta": 0.0075,
        "ratio": 0.3333,
        "max_history_t": 1,
        "history_weights": [1.0],
        "dpo_beta": 0.05,
        "simpo_beta": 2.0,
        "simpo_gamma": 0.6,
        "ronpo_alpha": 0.5,
        "ronpo_tau": 0.05,
        "ronpo_target_column": "ronpo_target",
        "reference_anchor_weight": (
            0.05 if method in RONPO or stability_attempt >= 3
            else 0.02 if stability_attempt == 2 else 0.0
        ),
        "preference_sft_weight": (
            0.005 if method in RONPO or stability_attempt >= 3
            else 0.002 if stability_attempt == 2 else 0.0
        ),
        "ht_target_column": "ht_target",
        "ht_target_scale": 1.0,
        "beta": 10.0,
        "learning_rate": lr,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "optim": "adamw_torch",
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "seed": seed,
        # Full training is frozen at microbatch 1 / accumulation 16 with
        # gradient checkpointing.  Smoke attempt 1 retains the historical
        # resource probe; smoke retries use the same safe profile as full.
        "gradient_accumulation_steps": 16 if safe_resource_profile else 8,
        "gradient_checkpointing": safe_resource_profile,
        "num_train_epochs": 1,
        "max_steps": 20 if stage == "smoke" else 900,
        "per_device_train_batch_size": 1 if safe_resource_profile else 2,
        "per_device_eval_batch_size": 1 if safe_resource_profile else 2,
        "max_length": 2048,
        "max_prompt_length": 1800,
        "do_eval": False,
        "eval_strategy": "no",
        "logging_steps": 1 if stage == "smoke" else 5,
        "log_level": "info",
        "generate_during_eval": False,
        "load_best_model_at_end": False,
        "save_strategy": "no" if stage == "smoke" else "steps",
        "save_steps": 900,
        "save_total_limit": 1,
        "save_only_model": True,
        "save_safetensors": True,
        "push_to_hub": False,
        "report_to": ["wandb"],
        "output_dir": str(output),
        "run_name": f"aaai27-flagship-{stage}-{method}-s{seed}-a{attempt}",
    }
    return config


def dataset_for(root: Path, method: str) -> Path:
    if method in RONPO or method.startswith("ht_mnpo_"):
        return root / "precomputed" / method
    return root / "precomputed" / "avg"


def command_for(
    args: argparse.Namespace, stage: str, method: str, seed: int, attempt: int, output: Path,
) -> tuple[list[str], Path]:
    if method != "kto":
        config = unified_config(
            args.model, str(dataset_for(args.root, method)), output, stage, method, seed, attempt
        )
        config_path = output / "config.yaml"
        output.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        command = [
            args.python, "-m", "accelerate.commands.launch",
            "--config_file", str(args.project / "accelerate_configs/single_gpu.yaml"),
            "--num_processes=1", "-m", "mnpo_scripts.run_mnpo", str(config_path),
        ]
        return command, config_path

    output.mkdir(parents=True, exist_ok=True)
    stability_attempt = max(1, attempt - 1) if stage == "smoke" else attempt
    command = [
        args.python, "-m", "accelerate.commands.launch",
        "--config_file", str(args.project / "accelerate_configs/single_gpu.yaml"),
        "--num_processes=1", str(args.project / "scripts/revision/run_kto_full.py"),
        "--model", args.model,
        "--train-file", str(args.root / "kto_pointwise/train_kto.jsonl"),
        "--eval-file", str(args.root / "kto_pointwise/test_kto.jsonl"),
        "--output-dir", str(output), "--beta", "0.05", "--seed", str(seed),
        "--run-name", f"aaai27-flagship-{stage}-kto-s{seed}-a{attempt}",
        "--max-steps", "20" if stage == "smoke" else "900",
        # TRL KTO rejects an actual batch of one. `run_kto_full.py` disables
        # AdamW foreach temporaries so batch two fits without changing AdamW.
        "--per-device-train-batch-size", "2",
        "--per-device-eval-batch-size", "2",
        "--gradient-accumulation-steps", "8",
        "--learning-rate", str(5.0e-7 / (2 ** (stability_attempt - 1))),
        "--logging-steps", "1" if stage == "smoke" else "5", "--save-steps", "900",
        "--save-total-limit", "1", "--report-to", "wandb",
        "--reference-device", "cuda:1",
    ]
    if stage == "smoke":
        command.extend(["--smoke", "--skip-final-save"])
    config_path = output / "command.json"
    config_path.write_text(json.dumps(command, indent=2) + "\n")
    return command, config_path


def metrics_finite(output: Path) -> tuple[bool, str]:
    path = output / "train_results.json"
    if not path.exists():
        return False, "missing_train_results"
    data = json.loads(path.read_text())
    numeric = [float(value) for value in data.values() if isinstance(value, (int, float))]
    if not numeric or not all(math.isfinite(value) for value in numeric):
        return False, "nonfinite_train_metrics"
    return True, "finite"


def model_complete(output: Path) -> bool:
    return (output / "model.safetensors").is_file() or (output / "model.safetensors.index.json").is_file()


def launch(
    args: argparse.Namespace, stage: str, method: str, seed: int, gpus: tuple[int, ...], attempt: int,
) -> Running:
    gpu = gpus[0]
    output = args.root / stage / method / f"seed{seed}" / f"attempt{attempt}"
    status = output / "job_status.json"
    wandb_id = run_id(stage, method, seed, attempt)
    command, config_path = command_for(args, stage, method, seed, attempt, output)
    log_path = args.root / "logs" / f"{stage}_{method}_s{seed}_a{attempt}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    (args.root / "wandb" / method / f"seed{seed}").mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": ",".join(str(value) for value in gpus),
        "PYTHONPATH": str(args.project),
        "HF_HOME": str(args.root / "cache/huggingface"),
        "HF_HUB_CACHE": str(args.root / "cache/huggingface/hub"),
        "HUGGINGFACE_HUB_CACHE": str(args.root / "cache/huggingface/hub"),
        "WANDB_MODE": "online", "WANDB_ENTITY": "promotion-kim", "WANDB_PROJECT": "mnpo",
        "WANDB_RUN_GROUP": "ronpo-aaai27-flagship-p1", "WANDB_RUN_ID": wandb_id,
        "WANDB_NAME": f"aaai27-{stage}-{method}-s{seed}-a{attempt}",
        "WANDB_RESUME": "allow", "WANDB_DIR": str(args.root / "wandb" / method / f"seed{seed}"),
        "MNPO_DISABLE_CUDNN_SDPA": "1", "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    if stage == "smoke" and method != "kto":
        env["MNPO_SKIP_FINAL_SAVE"] = "1"
    atomic_json(status, {
        "status": "running", "stage": stage, "method": method, "seed": seed, "attempt": attempt,
        "gpu": gpu, "gpus": list(gpus), "optimizer_steps": 20 if stage == "smoke" else 900,
        "effective_batch_size": 16, "wandb_run_id": wandb_id,
        "wandb_url": f"https://wandb.ai/promotion-kim/mnpo/runs/{wandb_id}",
        "config_or_command": str(config_path), "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    process = subprocess.Popen(command, cwd=args.project, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
    return Running(method, seed, gpu, gpus, process, log_handle, log_path, output, wandb_id, time.time())


def run_stability_gate(args: argparse.Namespace, job: Running) -> tuple[bool, str]:
    generation_dir = job.output_dir / "stability" / "generations"
    generation = generation_dir / "output_42.json"
    log_path = job.output_dir / "stability" / "decode.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
    command = [
        args.python, str(args.project / "scripts/revision/decode_transformers_non_thinking.py"),
        "--data_dir", str(args.root / "data/pool_validation.jsonl"),
        "--model", str(job.output_dir), "--output_dir", str(generation_dir),
        "--cache_dir", str(args.root / "cache/huggingface/hub"), "--seed", "42",
        "--temperature", "0.7", "--top_p", "0.9", "--max_new_tokens", "2048",
        "--batch_size", "8", "--attn_implementation", "eager", "--local_files_only",
        "--max_prompts", "128",
    ]
    with log_path.open("a", encoding="utf-8") as handle:
        decode = subprocess.run(command, cwd=args.project, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if decode.returncode != 0 or not generation.exists():
        return False, "stability_decode_failed"
    gate_path = job.output_dir / "stability" / "gate.json"
    gate_command = [
        args.python, str(args.project / "scripts/revision/flagship/stability_gate.py"),
        "--base", str(args.root / "stability/base/generations/output_42.json"),
        "--candidate", str(generation), "--output", str(gate_path),
        "--min-length-ratio", "0.33", "--max-length-ratio", "2.0",
        "--max-repeat-run", "20", "--expected-records", "128",
    ]
    gate = subprocess.run(gate_command, cwd=args.project, capture_output=True, text=True)
    (job.output_dir / "stability" / "gate.log").write_text(gate.stdout + gate.stderr)
    return gate.returncode == 0, "passed" if gate.returncode == 0 else "stability_gate_failed"


def finish(args: argparse.Namespace, job: Running, stage: str, attempt: int) -> bool:
    rc = job.process.wait()
    job.log_handle.close()
    finite, reason = metrics_finite(job.output_dir)
    log_tail = job.log_path.read_text(errors="replace")[-20000:] if job.log_path.exists() else ""
    oom = "OutOfMemory" in log_tail or "CUDA out of memory" in log_tail
    traceback = "Traceback (most recent call last)" in log_tail
    complete = rc == 0 and finite and not oom and not traceback
    if stage == "full":
        complete = complete and model_complete(job.output_dir)
        if not model_complete(job.output_dir):
            reason = "missing_final_model"
        if complete:
            complete, reason = run_stability_gate(args, job)
    status = {
        "status": "completed" if complete else "failed", "stage": stage, "method": job.method,
        "seed": job.seed, "attempt": attempt, "gpu": job.gpu, "gpus": list(job.gpus), "returncode": rc,
        "finite_metrics": finite, "oom": oom, "traceback": traceback, "reason": reason,
        "optimizer_steps": 20 if stage == "smoke" else 900, "effective_batch_size": 16,
        "wandb_run_id": job.wandb_id,
        "wandb_url": f"https://wandb.ai/promotion-kim/mnpo/runs/{job.wandb_id}",
        "output_dir": str(job.output_dir), "log": str(job.log_path),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(job.output_dir / "job_status.json", status)
    return complete


def run_queue(
    args: argparse.Namespace, stage: str, jobs: list[tuple[str, int]], max_attempts: int,
    gpu_ids: list[int] | None = None, start_attempt: int = 1,
) -> list[dict[str, int | str]]:
    if not 1 <= start_attempt <= max_attempts:
        raise ValueError(f"invalid start attempt {start_attempt} for max_attempts={max_attempts}")
    pending = [(method, seed, start_attempt) for method, seed in jobs]
    available = list(range(8)) if gpu_ids is None else sorted(gpu_ids)
    running: dict[int, Running] = {}
    failures = []
    while pending or running:
        while pending and available:
            method, seed, attempt = pending.pop(0)
            output = args.root / stage / method / f"seed{seed}" / f"attempt{attempt}"
            status_path = output / "job_status.json"
            if status_path.exists():
                prior = json.loads(status_path.read_text()).get("status")
                if prior == "completed":
                    continue
                if prior == "failed":
                    if attempt < max_attempts:
                        pending.insert(0, (method, seed, attempt + 1))
                    else:
                        failures.append({"method": method, "seed": seed, "attempt": attempt})
                    continue
            required = 2 if method == "kto" else 1
            if len(available) < required:
                pending.insert(0, (method, seed, attempt))
                break
            allocated = tuple(available.pop(0) for _ in range(required))
            gpu = allocated[0]
            running[gpu] = launch(args, stage, method, seed, allocated, attempt)
        if not running:
            continue
        time.sleep(5)
        for gpu, job in list(running.items()):
            if job.process.poll() is None:
                continue
            attempt = int(job.output_dir.name.removeprefix("attempt"))
            success = finish(args, job, stage, attempt)
            del running[gpu]
            available.extend(job.gpus)
            available.sort()
            if not success:
                if attempt < max_attempts:
                    pending.insert(0, (job.method, job.seed, attempt + 1))
                else:
                    failures.append({"method": job.method, "seed": job.seed, "attempt": attempt})
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--early-smoke-methods", nargs="*", choices=list(METHODS), default=None,
        help="Run only the listed seed-42 smoke gates; full training remains in the main queue.",
    )
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=None)
    args = parser.parse_args()
    protocol = json.loads((args.project / "results/ronpo_flagship_20260712/objective_protocol.json").read_text())
    if protocol["optimizer_steps"] != 900 or protocol["effective_batch_size"] != 16:
        raise RuntimeError("protocol budget mismatch")

    status_root = args.root / "status"
    if args.early_smoke_methods:
        if not args.gpu_ids:
            raise RuntimeError("--early-smoke-methods requires explicit --gpu-ids")
        invalid = sorted(set(args.gpu_ids) - set(range(8)))
        if invalid:
            raise RuntimeError(f"invalid GPU IDs: {invalid}")
        atomic_json(status_root / "early_baseline_smoke.json", {
            "status": "running", "methods": args.early_smoke_methods, "gpu_ids": args.gpu_ids,
        })
        failures = run_queue(
            args, "smoke", [(method, 42) for method in args.early_smoke_methods],
            max_attempts=3, gpu_ids=args.gpu_ids,
        )
        atomic_json(status_root / "early_baseline_smoke.json", {
            "status": "completed" if not failures else "completed_with_failures",
            "methods": args.early_smoke_methods, "gpu_ids": args.gpu_ids,
            "failures": failures,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        return
    external_smoke_status = status_root / "early_baseline_smoke.json"
    for _ in range(1440):
        if not external_smoke_status.exists():
            break
        external = json.loads(external_smoke_status.read_text())
        if external.get("status") != "running" or external.get("methods") != ["kto"]:
            break
        time.sleep(5)
    atomic_json(status_root / "s2.json", {"status": "running", "stage": "ronpo_smoke"})
    smoke_failures = run_queue(args, "smoke", [(method, 42) for method in RONPO], max_attempts=3)
    atomic_json(status_root / "s2.json", {"status": "running", "stage": "baseline_smoke"})
    smoke_failures.extend(run_queue(args, "smoke", [(method, 42) for method in BASELINES], max_attempts=3))

    base_generation = args.root / "stability/base/generations/output_42.json"
    if not base_generation.exists():
        base_generation.parent.mkdir(parents=True, exist_ok=True)
        base_log = args.root / "stability/base/decode.log"
        base_env = os.environ.copy()
        base_env["CUDA_VISIBLE_DEVICES"] = "0"
        base_command = [
            args.python, str(args.project / "scripts/revision/decode_transformers_non_thinking.py"),
            "--data_dir", str(args.root / "data/pool_validation.jsonl"), "--model", args.model,
            "--output_dir", str(base_generation.parent),
            "--cache_dir", str(args.root / "cache/huggingface/hub"), "--seed", "42",
            "--temperature", "0.7", "--top_p", "0.9", "--max_new_tokens", "2048",
            "--batch_size", "8", "--attn_implementation", "eager", "--local_files_only",
            "--max_prompts", "128",
        ]
        with base_log.open("a", encoding="utf-8") as handle:
            base_rc = subprocess.run(base_command, cwd=args.project, env=base_env, stdout=handle, stderr=subprocess.STDOUT)
        if base_rc.returncode != 0 or not base_generation.exists():
            raise RuntimeError("base stability decode failed")

    atomic_json(status_root / "s2.json", {"status": "running", "stage": "matched_full_training"})
    smoke_failed_methods = {str(row["method"]) for row in smoke_failures}
    eligible = [method for method in METHODS if method not in smoke_failed_methods]
    jobs = [(method, seed) for method in eligible for seed in SEEDS]
    full_failures = run_queue(args, "full", jobs, max_attempts=3)
    final_status = "completed" if not smoke_failures and not full_failures else "completed_with_failures"
    atomic_json(status_root / "s2.json", {
        "status": final_status, "stage": "matched_full_training", "methods": list(METHODS),
        "eligible_methods": eligible, "smoke_failures": smoke_failures, "full_failures": full_failures,
        "seeds": list(SEEDS), "optimizer_steps": 900, "effective_batch_size": 16,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    atomic_json(status_root / "s3.json", {
        "status": final_status, "stage": "pre_eval_stability_gates",
        "smoke_failures": smoke_failures, "full_or_stability_failures": full_failures,
        "validation_prompts": 128, "think_leakage_required": 0,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })


if __name__ == "__main__":
    main()
