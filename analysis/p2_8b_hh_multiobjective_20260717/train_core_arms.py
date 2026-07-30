#!/usr/bin/env python3
"""Run the four preregistered trainable core arms concurrently on four GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import yaml


ARMS = {
    "ronpo_os": {"loss_type": "ronpo", "ronpo_target_column": "target_os_k0p01"},
    "ronpo_topmass": {"loss_type": "ronpo", "ronpo_target_column": "target_topmass_k0p01"},
    "inpo_avg": {"loss_type": "inpo", "ronpo_target_column": "target_os_k0p01"},
    "ht_mnpo_harmless": {"loss_type": "ht_mnpo", "ronpo_target_column": "target_os_k0p01"},
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wandb_id(stage: str, arm: str) -> str:
    return hashlib.sha256(f"p2-hh-llama31-s42|{stage}|{arm}".encode()).hexdigest()[:12]


def load_env(path: Path) -> dict[str, str]:
    values = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"").strip("'")
    return values


def config(model: str, dataset: str, output: Path, arm: str, stage: str) -> dict:
    smoke = stage == "smoke"
    return {
        "model_name_or_path": model,
        "torch_dtype": None,
        "attn_implementation": "sdpa",
        "dataset_mixer": {dataset: 1.0},
        "dataset_splits": ["train", "test"],
        "preprocessing_num_workers": 4,
        "bf16": True,
        "loss_type": ARMS[arm]["loss_type"],
        "eta": 0.0075,
        "ratio": 0.3333,
        "max_history_t": 1,
        "history_weights": [1.0],
        "dpo_beta": 0.05,
        "simpo_beta": 2.0,
        "simpo_gamma": 0.6,
        "ronpo_alpha": 0.5,
        "ronpo_tau": 0.05,
        "ronpo_target_column": ARMS[arm]["ronpo_target_column"],
        "ht_target_column": "ht_target",
        "ht_target_scale": 1.0,
        "reference_anchor_weight": 0.05,
        "preference_sft_weight": 0.005,
        "beta": 10.0,
        "learning_rate": 1.0e-7,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "optim": "adamw_torch",
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "seed": 42,
        "gradient_accumulation_steps": 16,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "num_train_epochs": 100,
        "max_steps": 20 if smoke else 900,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "max_length": 2048,
        "max_prompt_length": 1024,
        "do_eval": False,
        "eval_strategy": "no",
        "logging_steps": 1 if smoke else 5,
        "log_level": "info",
        "generate_during_eval": False,
        "load_best_model_at_end": False,
        "save_strategy": "no" if smoke else "steps",
        "save_steps": 900,
        "save_total_limit": 1,
        "save_only_model": True,
        "save_safetensors": True,
        "push_to_hub": False,
        "report_to": ["none" if smoke else "wandb"],
        "output_dir": str(output),
        "run_name": f"p2-hh-llama31-{stage}-{arm}-s42",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stage", choices=["smoke", "full"], required=True)
    parser.add_argument("--wandb-env", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]
    if len(gpus) != len(ARMS):
        raise RuntimeError("exactly four GPUs are required for the four trainable core arms")
    python = args.venv / "bin/python"
    accel = args.project / "accelerate_configs/single_gpu.yaml"
    stage_root = args.root / "train" / args.stage
    logs = args.root / "logs" / "train"
    logs.mkdir(parents=True, exist_ok=True)
    common_env = os.environ.copy()
    common_env.update(load_env(args.wandb_env))
    common_env.update({
        "PYTHONPATH": str(args.project),
        "WANDB_MODE": "online",
        "WANDB_ENTITY": "promotion-kim",
        "WANDB_PROJECT": "mnpo",
        "WANDB_RUN_GROUP": "p2-llama31-hh-multiobjective-s42",
        "WANDB_DIR": str(args.root / "wandb"),
        "MNPO_DISABLE_CUDNN_SDPA": "1",
        "TORCH_CUDNN_SDPA_ENABLED": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    if args.stage == "smoke":
        common_env["MNPO_SKIP_FINAL_SAVE"] = "1"
    jobs = []
    for gpu, arm in zip(gpus, ARMS):
        output = stage_root / arm
        output.mkdir(parents=True, exist_ok=True)
        cfg = output / "config.yaml"
        cfg.write_text(yaml.safe_dump(config(args.model, args.dataset, output, arm, args.stage), sort_keys=False), encoding="utf-8")
        run_id = wandb_id(args.stage, arm)
        env = common_env.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "WANDB_RUN_ID": run_id,
            "WANDB_NAME": f"p2-hh-llama31-{args.stage}-{arm}-s42",
            "WANDB_RESUME": "allow",
        })
        log = logs / f"{args.stage}_{arm}.log"
        handle = log.open("a", encoding="utf-8")
        command = [
            str(python), "-m", "accelerate.commands.launch",
            "--config_file", str(accel), "--num_processes=1",
            "-m", "mnpo_scripts.run_mnpo", str(cfg),
        ]
        started = time.time()
        process = subprocess.Popen(command, cwd=args.project, env=env, stdout=handle, stderr=subprocess.STDOUT)
        jobs.append((arm, gpu, output, cfg, run_id, log, handle, process, started, command))
        (output / "job_status.json").write_text(json.dumps({
            "status": "running", "stage": args.stage, "arm": arm, "gpu": gpu,
            "seed": 42, "max_steps": 20 if args.stage == "smoke" else 900,
            "effective_batch": 16, "wandb_run_id": run_id,
            "config_sha256": sha(cfg), "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "command": command,
        }, indent=2) + "\n", encoding="utf-8")
    failures = []
    for arm, gpu, output, cfg, run_id, log, handle, process, started, command in jobs:
        rc = process.wait()
        handle.close()
        metrics_path = output / "train_results.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
        numeric = [float(value) for value in metrics.values() if isinstance(value, (int, float))]
        finite = bool(numeric) and all(math.isfinite(value) for value in numeric)
        checkpoint = output / "checkpoint-900"
        model_present = args.stage == "smoke" or (checkpoint / "config.json").is_file()
        text = log.read_text(errors="replace")[-30000:] if log.is_file() else ""
        clean = rc == 0 and finite and model_present and "Traceback (most recent call last)" not in text and "out of memory" not in text.lower()
        status = {
            "status": "completed" if clean else "failed", "stage": args.stage, "arm": arm,
            "gpu": gpu, "returncode": rc, "finite_metrics": finite, "model_present": model_present,
            "max_steps": 20 if args.stage == "smoke" else 900, "effective_batch": 16,
            "wandb_run_id": run_id, "wandb_url": f"https://wandb.ai/promotion-kim/mnpo/runs/{run_id}",
            "config": str(cfg), "config_sha256": sha(cfg), "log": str(log),
            "checkpoint": str(checkpoint) if args.stage == "full" else None,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (output / "job_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        if not clean:
            failures.append(arm)
    summary = {"stage": args.stage, "arms": list(ARMS), "failed": failures, "status": "complete" if not failures else "failed"}
    (stage_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
