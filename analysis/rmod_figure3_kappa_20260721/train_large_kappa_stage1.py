#!/usr/bin/env python3
"""Train one preregistered large-kappa SafeRLHF RONPO-OS Stage-1 arm."""

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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--venv", type=Path, required=True)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--label", choices=["k1", "k2"], required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--steps", type=int, choices=[20, 900], required=True)
    a = p.parse_args()
    if not (Path.home() / ".netrc").is_file():
        raise RuntimeError("W&B authentication is unavailable")
    phase = "smoke" if a.steps == 20 else "full"
    out = a.experiment / a.label / "stage1" / phase
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "job_status.json"
    if status_path.is_file() and json.loads(status_path.read_text()).get("status") == "completed":
        print(status_path.read_text(), end="")
        return
    target = f"target_os_{a.label}"
    run_name = f"fig3-{a.label}-llama31-saferlhf-stage1-s42-{a.steps}steps"
    run_id = hashlib.sha256(run_name.encode()).hexdigest()[:12]
    cfg = {
        "model_name_or_path": a.model, "torch_dtype": None, "attn_implementation": "sdpa",
        "dataset_mixer": {a.dataset: 1.0}, "dataset_splits": ["train", "test"],
        "preprocessing_num_workers": 4, "bf16": True, "loss_type": "ronpo", "eta": 0.0075,
        "ratio": 0.3333, "max_history_t": 1, "history_weights": [1.0], "dpo_beta": 0.05,
        "simpo_beta": 2.0, "simpo_gamma": 0.6, "ronpo_alpha": 1.0, "ronpo_tau": 0.05,
        "ronpo_target_column": target, "ht_target_column": "ht_target", "ht_target_scale": 1.0,
        "reference_anchor_weight": 0.05, "preference_sft_weight": 0.005, "beta": 10.0,
        "learning_rate": 5e-7, "lr_scheduler_type": "cosine", "warmup_ratio": 0.1,
        "optim": "adamw_torch", "weight_decay": 0.0, "max_grad_norm": 1.0, "seed": 42,
        "gradient_accumulation_steps": 16, "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False}, "num_train_epochs": 100,
        "max_steps": a.steps, "per_device_train_batch_size": 1, "per_device_eval_batch_size": 1,
        "max_length": 2048, "max_prompt_length": 1024, "do_eval": False, "eval_strategy": "no",
        "logging_steps": 1 if a.steps == 20 else 5, "generate_during_eval": False,
        "load_best_model_at_end": False, "save_strategy": "no", "save_total_limit": 1,
        "save_only_model": True, "save_safetensors": True, "push_to_hub": False,
        "report_to": ["wandb"], "output_dir": str(out), "run_name": run_name,
    }
    config = out / "config.yaml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    log = out / "train.log"
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(a.project), "CUDA_VISIBLE_DEVICES": str(a.gpu), "WANDB_MODE": "online",
        "WANDB_ENTITY": "promotion-kim", "WANDB_PROJECT": "mnpo", "WANDB_RUN_ID": run_id,
        "WANDB_NAME": run_name, "WANDB_RESUME": "allow", "WANDB_DIR": str(a.experiment / "wandb"),
        "MNPO_DISABLE_CUDNN_SDPA": "1", "TORCH_CUDNN_SDPA_ENABLED": "0",
        "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    if a.steps == 20:
        env["MNPO_SKIP_FINAL_SAVE"] = "1"
    command = [str(a.venv / "bin/python"), "-m", "accelerate.commands.launch", "--config_file",
               str(a.project / "accelerate_configs/single_gpu.yaml"), "--num_processes=1", "-m",
               "mnpo_scripts.run_mnpo", str(config)]
    started = time.time()
    with log.open("w") as handle:
        rc = subprocess.Popen(command, cwd=a.project, env=env, stdout=handle, stderr=subprocess.STDOUT).wait()
    metrics_path = out / "train_results.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    finite = bool(metrics) and all(math.isfinite(float(v)) for v in metrics.values() if isinstance(v, (int, float)))
    tail = log.read_text(errors="replace")[-50000:]
    clean = rc == 0 and finite and "Traceback (most recent call last)" not in tail and "out of memory" not in tail.lower()
    status = {
        "status": "completed" if clean else "failed", "label": a.label, "stage": 1, "phase": phase,
        "target_column": target, "gpu": a.gpu, "steps": a.steps, "seed": 42, "returncode": rc,
        "finite_metrics": finite, "model_dir": str(out) if a.steps == 900 else None,
        "config": str(config), "wandb_run_id": run_id,
        "wandb_url": f"https://wandb.ai/promotion-kim/mnpo/runs/{run_id}",
        "elapsed_seconds": time.time() - started,
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    if not clean:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
