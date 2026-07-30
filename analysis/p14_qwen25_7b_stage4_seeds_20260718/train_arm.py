#!/usr/bin/env python3
"""Train one locked Qwen2.5 SafeRLHF arm/stage with mandatory W&B online."""

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

from common import ARMS, model_dir, seed_root, stage_root, tag, sha256


def complete(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("status") == "completed" and data.get("finite_metrics") is True
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--train-python", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[42, 43, 44], required=True)
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY must be supplied ephemerally")
    if not (args.root / "run_lock.json").is_file() or not (args.root / "kappa_lock.json").is_file():
        raise RuntimeError("pre-training locks are missing")
    loss_type, target_family = ARMS[args.arm]
    kappa = float(json.loads((args.root / "kappa_lock.json").read_text(encoding="utf-8"))["confirmatory_os_kappa"])
    target_column = None
    if target_family == "target_os":
        target_column = f"target_os_k{tag(kappa)}"
    elif target_family == "target_topmass":
        target_column = f"target_topmass_k{tag(kappa)}"
    elif target_family:
        target_column = target_family
    base = args.cache / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots" / "a09a35458c702b33eeacc393d103063234e8bc28"
    parent = base if args.stage == 1 else model_dir(args.root, args.seed, args.stage - 1, args.arm)
    dataset = (args.root / "shared/stage1_pool/precompute/targets" if args.stage == 1 else stage_root(args.root, args.seed, args.stage) / args.arm / "pool/precompute/targets")
    if not (parent / "config.json").is_file() or not (dataset / "dataset_dict.json").is_file():
        raise RuntimeError(f"missing parent or dataset: parent={parent} dataset={dataset}")
    output_root = stage_root(args.root, args.seed, args.stage) / args.arm / "train"
    for phase, steps in (("smoke", 20), ("full", 900)):
        output = output_root / phase
        status_path = output / "job_status.json"
        if complete(status_path):
            continue
        if status_path.exists():
            raise RuntimeError(f"fail-closed terminal status exists: {status_path}")
        output.mkdir(parents=True, exist_ok=False)
        repair_path = args.root / "stability_repair_amendment.json"
        repair = json.loads(repair_path.read_text(encoding="utf-8")) if repair_path.is_file() else {}
        repair_spec = repair.get("arm_overrides", {}).get(args.arm)
        run_suffix = "-stability-repair-v1" if repair_spec else ""
        run_name = f"p14-qwen25-saferlhf-s{args.seed}-stage{args.stage}-{args.arm}{run_suffix}-{steps}steps"
        run_id = hashlib.sha256(run_name.encode()).hexdigest()[:12]
        config = {
            "model_name_or_path": str(parent), "torch_dtype": None, "attn_implementation": "sdpa",
            "tokenizer_name_or_path": str(base),
            "dataset_mixer": {str(dataset): 1.0}, "dataset_splits": ["train", "test"], "preprocessing_num_workers": 4,
            "bf16": True, "loss_type": loss_type, "eta": 0.0075, "ratio": 0.3333, "max_history_t": 1,
            "history_weights": [1.0], "dpo_beta": 0.05, "simpo_beta": 2.0, "simpo_gamma": 0.6,
            "ronpo_alpha": 1.0, "ronpo_tau": 0.05, "ronpo_target_column": target_column or f"target_os_k{tag(kappa)}",
            "ht_target_column": target_column or "ht_target", "ht_target_scale": 1.0,
            "reference_anchor_weight": 0.05, "preference_sft_weight": 0.005, "beta": 10.0,
            "learning_rate": 5e-7, "lr_scheduler_type": "cosine", "warmup_ratio": 0.1, "optim": "adamw_torch",
            "weight_decay": 0.0, "max_grad_norm": 1.0, "seed": args.seed, "gradient_accumulation_steps": 16,
            "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "num_train_epochs": 100, "max_steps": steps, "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1, "max_length": 2048, "max_prompt_length": 1024,
            "do_eval": False, "eval_strategy": "no", "logging_steps": 1 if steps == 20 else 5,
            "log_level": "info", "generate_during_eval": False, "load_best_model_at_end": False,
            "save_strategy": "no", "save_total_limit": 1, "save_only_model": True, "save_safetensors": True,
            "push_to_hub": False, "report_to": ["wandb"], "output_dir": str(output), "run_name": run_name,
        }
        if repair_spec:
            config.update({key: repair_spec[key] for key in (
                "eta", "dpo_beta", "learning_rate", "reference_anchor_weight", "preference_sft_weight"
            )})
        config_path = output / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        log = seed_root(args.root, args.seed) / "logs" / f"stage{args.stage}_{args.arm}_{phase}_{time.time_ns()}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "PYTHONPATH": str(args.project), "HF_HOME": str(args.cache), "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "WANDB_MODE": "online", "WANDB_ENTITY": "promotion-kim", "WANDB_PROJECT": "mnpo",
            "WANDB_RUN_GROUP": f"p14-qwen25-saferlhf-s{args.seed}", "WANDB_DIR": str(seed_root(args.root, args.seed) / "wandb"),
            "WANDB_RUN_ID": run_id, "WANDB_NAME": run_name, "WANDB_RESUME": "allow",
            "MNPO_DISABLE_CUDNN_SDPA": "1", "MNPO_DISABLE_APEX": "1", "TORCH_CUDNN_SDPA_ENABLED": "0",
            "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        })
        if phase == "smoke":
            env["MNPO_SKIP_FINAL_SAVE"] = "1"
        command = [str(args.train_python), "-m", "accelerate.commands.launch", "--config_file", str(args.project / "accelerate_configs/single_gpu.yaml"), "--num_processes=1", "-m", "mnpo_scripts.run_mnpo", str(config_path)]
        started = time.time()
        with log.open("w", encoding="utf-8") as handle:
            returncode = subprocess.run(command, cwd=args.project, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
        metrics_path = output / "train_results.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        finite = bool(metrics) and all(math.isfinite(float(value)) for value in metrics.values() if isinstance(value, (int, float)))
        tail = log.read_text(encoding="utf-8", errors="replace")[-50000:]
        clean = returncode == 0 and finite and "Traceback (most recent call last)" not in tail and "out of memory" not in tail.lower()
        status = {
            "status": "completed" if clean else "failed", "phase": phase, "seed": args.seed, "stage": args.stage,
            "arm": args.arm, "loss_type": loss_type, "target_column": target_column, "gpu": args.gpu,
            "steps": steps, "effective_batch": 16, "returncode": returncode, "finite_metrics": finite,
            "parent_model": str(parent), "dataset": str(dataset), "config": str(config_path), "config_sha256": sha256(config_path),
            "log": str(log), "wandb_run_id": run_id, "wandb_url": f"https://wandb.ai/promotion-kim/mnpo/runs/{run_id}",
            "elapsed_seconds": time.time() - started,
        }
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        if not clean:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
