#!/usr/bin/env python3
"""Run one stage-matched Stage-2 arm after its refreshed pool is prepared."""

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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config(model: str, dataset: str, output: Path, arm: str, loss_type: str, target_column: str | None, steps: int) -> dict:
    smoke = steps <= 20
    payload = {
        "model_name_or_path": model, "torch_dtype": None, "attn_implementation": "sdpa",
        "dataset_mixer": {dataset: 1.0}, "dataset_splits": ["train", "test"], "preprocessing_num_workers": 4,
        "bf16": True, "loss_type": loss_type, "eta": 0.0075, "ratio": 0.3333,
        "max_history_t": 1, "history_weights": [1.0], "dpo_beta": 0.05,
        "simpo_beta": 2.0, "simpo_gamma": 0.6, "ronpo_alpha": 1.0, "ronpo_tau": 0.05,
        "ronpo_target_column": target_column or "target_os_k0p1", "ht_target_column": target_column or "ht_target",
        "ht_target_scale": 1.0, "reference_anchor_weight": 0.05, "preference_sft_weight": 0.005,
        "beta": 10.0, "learning_rate": 5e-7, "lr_scheduler_type": "cosine", "warmup_ratio": 0.1,
        "optim": "adamw_torch", "weight_decay": 0.0, "max_grad_norm": 1.0, "seed": 42,
        "gradient_accumulation_steps": 16, "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False}, "num_train_epochs": 100,
        "max_steps": steps, "per_device_train_batch_size": 1, "per_device_eval_batch_size": 1,
        "max_length": 2048, "max_prompt_length": 1024, "do_eval": False, "eval_strategy": "no",
        "logging_steps": 1 if smoke else 5, "log_level": "info", "generate_during_eval": False,
        "load_best_model_at_end": False, "save_strategy": "no", "save_total_limit": 1,
        "save_only_model": True, "save_safetensors": True, "push_to_hub": False,
        "report_to": ["wandb"], "output_dir": str(output),
        "run_name": f"p5-llama31-saferlhf-stage2-{arm}-s42-{steps}steps",
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--parent-model", required=True)
    parser.add_argument("--loss-type", choices=["ronpo", "inpo", "sppo", "mnpo", "ht_mnpo", "dpo", "ipo", "simpo"], required=True)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--stage", choices=["smoke", "full"], required=True)
    args = parser.parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY must be supplied ephemerally by the parent process")
    steps = 20 if args.stage == "smoke" else 900
    dataset = args.experiment / "stage2" / args.arm / "pool" / "precompute" / "targets"
    if not (dataset / "dataset_dict.json").is_file():
        raise RuntimeError(f"missing prepared Stage-2 dataset: {dataset}")
    output = args.experiment / "stage2" / args.arm / "train" / args.stage
    output.mkdir(parents=True, exist_ok=True)
    cfg = output / "config.yaml"
    cfg.write_text(yaml.safe_dump(config(args.parent_model, str(dataset), output, args.arm, args.loss_type, args.target_column, steps), sort_keys=False), encoding="utf-8")
    run_name = f"p5-llama31-saferlhf-stage2-{args.arm}-s42-{steps}steps"
    run_id = hashlib.sha256(run_name.encode()).hexdigest()[:12]
    log = args.experiment / "logs" / "stage2" / f"{args.arm}_{args.stage}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(args.project), "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "WANDB_MODE": "online", "WANDB_ENTITY": "promotion-kim", "WANDB_PROJECT": "mnpo",
        "WANDB_RUN_GROUP": "p5-llama31-saferlhf-stage2-s42", "WANDB_DIR": str(args.experiment / "wandb"),
        "WANDB_RUN_ID": run_id, "WANDB_NAME": run_name, "WANDB_RESUME": "allow",
        "MNPO_DISABLE_CUDNN_SDPA": "1", "TORCH_CUDNN_SDPA_ENABLED": "0", "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    if args.stage == "smoke":
        env["MNPO_SKIP_FINAL_SAVE"] = "1"
    command = [str(args.venv / "bin/python"), "-m", "accelerate.commands.launch", "--config_file", str(args.project / "accelerate_configs/single_gpu.yaml"), "--num_processes=1", "-m", "mnpo_scripts.run_mnpo", str(cfg)]
    started = time.time()
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=args.project, env=env, stdout=handle, stderr=subprocess.STDOUT)
        rc = process.wait()
    metrics_path = output / "train_results.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    finite = bool(metrics) and all(math.isfinite(float(value)) for value in metrics.values() if isinstance(value, (int, float)))
    text = log.read_text(encoding="utf-8", errors="replace")[-50000:] if log.is_file() else ""
    clean = rc == 0 and finite and "Traceback (most recent call last)" not in text and "out of memory" not in text.lower()
    status = {"status": "completed" if clean else "failed", "stage": args.stage, "arm": args.arm, "loss_type": args.loss_type, "target_column": args.target_column, "gpu": args.gpu, "steps": steps, "effective_batch": 16, "seed": 42, "returncode": rc, "finite_metrics": finite, "parent_model": args.parent_model, "dataset": str(dataset), "config": str(cfg), "config_sha256": sha(cfg), "log": str(log), "model_dir": str(output) if args.stage == "full" else None, "wandb_run_id": run_id, "wandb_url": f"https://wandb.ai/promotion-kim/mnpo/runs/{run_id}", "elapsed_seconds": time.time() - started}
    (output / "job_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))
    if not clean:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
