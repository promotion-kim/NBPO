#!/usr/bin/env python3
"""Launch matched single-GPU Stage-1 RONPO arms with online W&B logging.

Credentials are intentionally *not* read from files.  The parent launcher must
pass WANDB_API_KEY only as an inherited process environment variable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import yaml


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(model: str, dataset: str, output: Path, arm: dict, steps: int) -> dict:
    smoke = steps <= 20
    return {
        "model_name_or_path": model,
        "torch_dtype": None,
        "attn_implementation": "sdpa",
        "dataset_mixer": {dataset: 1.0},
        "dataset_splits": ["train", "test"],
        "preprocessing_num_workers": 4,
        "bf16": True,
        "loss_type": "ronpo",
        "eta": 0.0075,
        "ratio": 0.3333,
        "max_history_t": 1,
        "history_weights": [1.0],
        "dpo_beta": 0.05,
        "simpo_beta": 2.0,
        "simpo_gamma": 0.6,
        "ronpo_alpha": 1.0,
        "ronpo_tau": 0.05,
        "ronpo_target_column": arm["target_column"],
        "ht_target_column": "ht_target",
        "ht_target_scale": 1.0,
        "reference_anchor_weight": 0.05,
        "preference_sft_weight": 0.005,
        "beta": 10.0,
        "learning_rate": 5e-7,
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
        "max_steps": steps,
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
        "save_strategy": "no",
        "save_total_limit": 1,
        "save_only_model": True,
        "save_safetensors": True,
        "push_to_hub": False,
        "report_to": ["wandb"],
        "output_dir": str(output),
        "run_name": arm["run_name"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--arms-json", type=Path, required=True)
    parser.add_argument("--stage", choices=["smoke", "full"], required=True)
    parser.add_argument("--gpus", required=True)
    args = parser.parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY must be supplied ephemerally by the parent process")
    arms = json.loads(args.arms_json.read_text(encoding="utf-8"))["arms"]
    gpus = [int(value) for value in args.gpus.split(",")]
    if len(gpus) < len(arms):
        raise ValueError("fewer GPUs than requested Stage-1 arms")
    steps = 20 if args.stage == "smoke" else 900
    root = args.experiment / "stage1" / args.stage
    logs = args.experiment / "logs" / "stage1"
    root.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    base_env = os.environ.copy()
    base_env.update({
        "PYTHONPATH": str(args.project),
        "WANDB_MODE": "online",
        "WANDB_ENTITY": "promotion-kim",
        "WANDB_PROJECT": "mnpo",
        "WANDB_RUN_GROUP": "p5-llama31-saferlhf-stage1-s42",
        "WANDB_DIR": str(args.experiment / "wandb"),
        "MNPO_DISABLE_CUDNN_SDPA": "1",
        "TORCH_CUDNN_SDPA_ENABLED": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    if args.stage == "smoke":
        base_env["MNPO_SKIP_FINAL_SAVE"] = "1"
    jobs = []
    for gpu, arm in zip(gpus, arms):
        output = root / arm["name"]
        output.mkdir(parents=True, exist_ok=True)
        run_name = f"p5-llama31-saferlhf-{args.stage}-{arm['name']}-s42"
        payload = dict(arm)
        payload["run_name"] = run_name
        config_path = output / "config.yaml"
        config_path.write_text(yaml.safe_dump(_config(args.model, args.dataset, output, payload, steps), sort_keys=False), encoding="utf-8")
        run_id = hashlib.sha256(run_name.encode("utf-8")).hexdigest()[:12]
        env = base_env.copy()
        env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "WANDB_RUN_ID": run_id, "WANDB_NAME": run_name, "WANDB_RESUME": "allow"})
        log_path = logs / f"{args.stage}_{arm['name']}.log"
        handle = log_path.open("a", encoding="utf-8")
        command = [
            str(args.venv / "bin/python"), "-m", "accelerate.commands.launch",
            "--config_file", str(args.project / "accelerate_configs/single_gpu.yaml"), "--num_processes=1",
            "-m", "mnpo_scripts.run_mnpo", str(config_path),
        ]
        started = time.time()
        process = subprocess.Popen(command, cwd=args.project, env=env, stdout=handle, stderr=subprocess.STDOUT)
        jobs.append((arm, gpu, output, config_path, run_id, log_path, handle, process, started))
        (output / "job_status.json").write_text(json.dumps({"status": "running", "stage": args.stage, "arm": arm["name"], "gpu": gpu, "steps": steps, "effective_batch": 16, "seed": 42, "wandb_run_id": run_id, "wandb_url": f"https://wandb.ai/promotion-kim/mnpo/runs/{run_id}", "config": str(config_path), "config_sha256": _sha(config_path)}, indent=2) + "\n", encoding="utf-8")
    records, failed = [], []
    for arm, gpu, output, config_path, run_id, log_path, handle, process, started in jobs:
        rc = process.wait()
        handle.close()
        metrics_path = output / "train_results.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        finite = bool(metrics) and all(math.isfinite(float(value)) for value in metrics.values() if isinstance(value, (float, int)))
        log_text = log_path.read_text(encoding="utf-8", errors="replace")[-50000:] if log_path.is_file() else ""
        clean = rc == 0 and finite and "Traceback (most recent call last)" not in log_text and "out of memory" not in log_text.lower()
        record = {"status": "completed" if clean else "failed", "stage": args.stage, "arm": arm["name"], "gpu": gpu, "steps": steps, "effective_batch": 16, "seed": 42, "returncode": rc, "finite_metrics": finite, "wandb_run_id": run_id, "wandb_url": f"https://wandb.ai/promotion-kim/mnpo/runs/{run_id}", "config": str(config_path), "config_sha256": _sha(config_path), "log": str(log_path), "model_dir": str(output) if args.stage == "full" else None, "elapsed_seconds": time.time() - started}
        (output / "job_status.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        records.append(record)
        if not clean:
            failed.append(arm["name"])
    summary = {"status": "complete" if not failed else "failed", "stage": args.stage, "arms": records, "failed": failed}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
