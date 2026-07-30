#!/usr/bin/env python3
"""Train one preregistered baseline-repair candidate with mandatory W&B."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()

    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY must be supplied ephemerally")
    lock_path = args.repair_root / "baseline_repair_lock.json"
    if not lock_path.is_file():
        raise RuntimeError(f"missing preregistration lock: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    spec = lock["candidates"][args.candidate]
    arm, stage = spec["arm"], int(spec["stage"])
    loss_type = {"inpo_avg": "inpo", "ipo": "ipo", "sppo_avg": "sppo"}[arm]
    base = args.cache / "models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    if stage == 1:
        parent = base
        dataset = args.source_root / "shared/stage1_pool/precompute/targets"
    else:
        parent = args.repair_root / "inputs/sppo_stage2_parent"
        dataset = args.repair_root / "inputs/sppo_stage3_targets"
    if not (parent / "config.json").is_file() or not (dataset / "dataset_dict.json").is_file():
        raise RuntimeError(f"missing parent or dataset: parent={parent}; dataset={dataset}")

    candidate_root = args.repair_root / "candidates" / args.candidate
    for phase, steps in (("smoke", 20), ("full", 900)):
        output = candidate_root / "train" / phase
        status_path = output / "job_status.json"
        if status_path.is_file():
            prior = json.loads(status_path.read_text(encoding="utf-8"))
            if prior.get("status") == "completed" and prior.get("finite_metrics") is True:
                continue
            raise RuntimeError(f"fail-closed terminal status already exists: {status_path}")
        output.mkdir(parents=True, exist_ok=False)
        run_name = f"p14-repair-q25-s42-{args.candidate}-{steps}steps"
        run_id = hashlib.sha256(run_name.encode()).hexdigest()[:12]
        config = {
            "model_name_or_path": str(parent),
            "attn_implementation": "sdpa",
            "dataset_mixer": {str(dataset): 1.0},
            "dataset_splits": ["train", "test"],
            "preprocessing_num_workers": 4,
            "bf16": True,
            "loss_type": loss_type,
            "eta": spec["eta"],
            "ratio": 0.3333,
            "max_history_t": 1,
            "history_weights": [1.0],
            "dpo_beta": spec["dpo_beta"],
            "simpo_beta": 2.0,
            "simpo_gamma": 0.6,
            "ronpo_alpha": 1.0,
            "ronpo_tau": 0.05,
            "ronpo_target_column": "target_os_k0p05",
            "reference_anchor_weight": spec["reference_anchor_weight"],
            "preference_sft_weight": spec["preference_sft_weight"],
            "beta": 10.0,
            "learning_rate": spec["learning_rate"],
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
            "logging_steps": 1 if steps == 20 else 5,
            "generate_during_eval": False,
            "load_best_model_at_end": False,
            "save_strategy": "no",
            "save_total_limit": 1,
            "save_only_model": True,
            "save_safetensors": True,
            "push_to_hub": False,
            "report_to": ["wandb"],
            "output_dir": str(output),
            "run_name": run_name,
        }
        config_path = output / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        log = candidate_root / "logs" / f"{phase}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "PYTHONPATH": str(args.project),
            "HF_HOME": str(args.cache),
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "WANDB_MODE": "online",
            "WANDB_ENTITY": "promotion-kim",
            "WANDB_PROJECT": "mnpo",
            "WANDB_RUN_GROUP": "p14-qwen25-s42-baseline-repair",
            "WANDB_RUN_ID": run_id,
            "WANDB_NAME": run_name,
            "WANDB_RESUME": "allow",
            "WANDB_DIR": str(args.repair_root / "wandb"),
            "MNPO_DISABLE_CUDNN_SDPA": "1",
            "MNPO_DISABLE_APEX": "1",
            "TORCH_CUDNN_SDPA_ENABLED": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        })
        if phase == "smoke":
            env["MNPO_SKIP_FINAL_SAVE"] = "1"
        command = [
            str(args.python), "-m", "accelerate.commands.launch",
            "--config_file", str(args.project / "accelerate_configs/single_gpu.yaml"),
            "--num_processes=1", "-m", "mnpo_scripts.run_mnpo", str(config_path),
        ]
        started = time.time()
        with log.open("w", encoding="utf-8") as handle:
            returncode = subprocess.run(command, cwd=args.project, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
        metrics_path = output / "train_results.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        finite = bool(metrics) and all(
            math.isfinite(float(value)) for value in metrics.values() if isinstance(value, (int, float))
        )
        tail = log.read_text(encoding="utf-8", errors="replace")[-50000:]
        clean = returncode == 0 and finite and "Traceback (most recent call last)" not in tail and "out of memory" not in tail.lower()
        status = {
            "status": "completed" if clean else "failed",
            "candidate": args.candidate,
            "arm": arm,
            "stage": stage,
            "phase": phase,
            "steps": steps,
            "gpu": args.gpu,
            "returncode": returncode,
            "finite_metrics": finite,
            "parent_model": str(parent),
            "dataset": str(dataset),
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "wandb_run_id": run_id,
            "wandb_url": f"https://wandb.ai/promotion-kim/mnpo/runs/{run_id}",
            "elapsed_seconds": time.time() - started,
        }
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        if not clean:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
