#!/usr/bin/env python3
"""Continue a stability-selected repaired baseline with mandatory W&B logging."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--arm", choices=["inpo_avg", "ipo", "sppo_avg"], required=True)
    parser.add_argument("--stage", type=int, choices=[2, 3, 4], required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY must be supplied ephemerally")
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    spec = lock["candidates"][args.candidate]
    if spec["arm"] != args.arm:
        raise RuntimeError("candidate/arm mismatch")
    loss_type = {"inpo_avg": "inpo", "ipo": "ipo", "sppo_avg": "sppo"}[args.arm]
    parent = args.root / f"seeds/s42/stage{args.stage - 1}/{args.arm}/train/full"
    dataset = args.root / f"seeds/s42/stage{args.stage}/{args.arm}/pool/precompute/targets"
    if not (parent / "config.json").is_file() or not (dataset / "dataset_dict.json").is_file():
        raise RuntimeError(f"missing parent or dataset: {parent}; {dataset}")
    output_root = args.root / f"seeds/s42/stage{args.stage}/{args.arm}/train"
    for phase, steps in (("smoke", 20), ("full", 900)):
        output = output_root / phase
        status_path = output / "job_status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "completed" and status.get("finite_metrics") is True:
                continue
            raise RuntimeError(f"terminal status exists: {status_path}")
        output.mkdir(parents=True, exist_ok=False)
        run_name = f"p14-repair-q25-s42-stage{args.stage}-{args.arm}-{args.candidate}-{steps}steps"
        run_id = hashlib.sha256(run_name.encode()).hexdigest()[:12]
        config = {
            "model_name_or_path": str(parent), "attn_implementation": "sdpa",
            "dataset_mixer": {str(dataset): 1.0}, "dataset_splits": ["train", "test"],
            "preprocessing_num_workers": 4, "bf16": True, "loss_type": loss_type,
            "eta": spec["eta"], "ratio": 0.3333, "max_history_t": 1, "history_weights": [1.0],
            "dpo_beta": spec["dpo_beta"], "simpo_beta": 2.0, "simpo_gamma": 0.6,
            "ronpo_alpha": 1.0, "ronpo_tau": 0.05, "ronpo_target_column": "target_os_k0p05",
            "reference_anchor_weight": spec["reference_anchor_weight"],
            "preference_sft_weight": spec["preference_sft_weight"], "beta": 10.0,
            "learning_rate": spec["learning_rate"], "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1, "optim": "adamw_torch", "weight_decay": 0.0,
            "max_grad_norm": 1.0, "seed": 42, "gradient_accumulation_steps": 16,
            "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "num_train_epochs": 100, "max_steps": steps, "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1, "max_length": 2048, "max_prompt_length": 1024,
            "do_eval": False, "eval_strategy": "no", "logging_steps": 1 if steps == 20 else 5,
            "generate_during_eval": False, "load_best_model_at_end": False, "save_strategy": "no",
            "save_total_limit": 1, "save_only_model": True, "save_safetensors": True,
            "push_to_hub": False, "report_to": ["wandb"], "output_dir": str(output), "run_name": run_name,
        }
        config_path = output / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        log = args.root / f"seeds/s42/logs/stage{args.stage}_{args.arm}_{phase}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "PYTHONPATH": str(args.project), "HF_HOME": str(args.cache),
            "CUDA_VISIBLE_DEVICES": str(args.gpu), "WANDB_MODE": "online",
            "WANDB_ENTITY": "promotion-kim", "WANDB_PROJECT": "mnpo",
            "WANDB_RUN_GROUP": "p14-qwen25-s42-baseline-repair-continuation",
            "WANDB_RUN_ID": run_id, "WANDB_NAME": run_name, "WANDB_RESUME": "allow",
            "WANDB_DIR": str(args.root / "wandb"), "MNPO_DISABLE_CUDNN_SDPA": "1",
            "MNPO_DISABLE_APEX": "1", "TORCH_CUDNN_SDPA_ENABLED": "0",
            "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        })
        if phase == "smoke":
            env["MNPO_SKIP_FINAL_SAVE"] = "1"
        command = [str(args.python), "-m", "accelerate.commands.launch", "--config_file",
                   str(args.project / "accelerate_configs/single_gpu.yaml"), "--num_processes=1",
                   "-m", "mnpo_scripts.run_mnpo", str(config_path)]
        started = time.time()
        with log.open("w", encoding="utf-8") as handle:
            returncode = subprocess.run(command, cwd=args.project, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
        metrics_path = output / "train_results.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        finite = bool(metrics) and all(math.isfinite(float(v)) for v in metrics.values() if isinstance(v, (int, float)))
        tail = log.read_text(encoding="utf-8", errors="replace")[-50000:]
        clean = returncode == 0 and finite and "Traceback (most recent call last)" not in tail and "out of memory" not in tail.lower()
        status_path.write_text(json.dumps({
            "status": "completed" if clean else "failed", "phase": phase, "arm": args.arm,
            "stage": args.stage, "candidate_config": args.candidate, "steps": steps,
            "returncode": returncode, "finite_metrics": finite, "parent_model": str(parent),
            "dataset": str(dataset), "config": str(config_path), "wandb_run_id": run_id,
            "wandb_url": f"https://wandb.ai/promotion-kim/mnpo/runs/{run_id}",
            "elapsed_seconds": time.time() - started,
        }, indent=2) + "\n", encoding="utf-8")
        if not clean:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
