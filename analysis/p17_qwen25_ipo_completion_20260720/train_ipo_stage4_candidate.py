#!/usr/bin/env python3
"""Train one preregistered IPO Stage-4 stability candidate with W&B."""

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
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--gpu", type=int, required=True)
    a = p.parse_args()
    spec = json.loads(a.lock.read_text())["candidates"][a.candidate]
    base = a.cache / "models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    if not (a.parent / "config.json").is_file() or not (a.dataset / "dataset_dict.json").is_file():
        raise RuntimeError("parent or dataset is missing")
    for phase, steps in (("smoke", 20), ("full", 900)):
        out = a.output_root / "candidate_runs" / a.candidate / "s43/stage4/train" / phase
        status = out / "job_status.json"
        if status.is_file() and json.loads(status.read_text()).get("status") == "completed":
            continue
        out.mkdir(parents=True, exist_ok=False)
        name = f"p17-q25-ipo-s43-stage4-{a.candidate}-{steps}steps"
        run_id = hashlib.sha256(name.encode()).hexdigest()[:12]
        cfg = {
            "model_name_or_path": str(a.parent), "tokenizer_name_or_path": str(base),
            "attn_implementation": "sdpa", "dataset_mixer": {str(a.dataset): 1.0},
            "dataset_splits": ["train", "test"], "preprocessing_num_workers": 4,
            "bf16": True, "loss_type": "ipo", "eta": 0.0075, "ratio": 0.3333,
            "max_history_t": 1, "history_weights": [1.0], "dpo_beta": spec["dpo_beta"],
            "simpo_beta": 2.0, "simpo_gamma": 0.6, "ronpo_alpha": 1.0, "ronpo_tau": 0.05,
            "ronpo_target_column": "target_os_k0p1", "reference_anchor_weight": spec["reference_anchor_weight"],
            "preference_sft_weight": spec["preference_sft_weight"], "beta": 10.0,
            "learning_rate": spec["learning_rate"], "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1, "optim": "adamw_torch", "weight_decay": 0.0,
            "max_grad_norm": 1.0, "seed": 43, "gradient_accumulation_steps": 16,
            "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "num_train_epochs": 100, "max_steps": steps, "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1, "max_length": 2048, "max_prompt_length": 1024,
            "do_eval": False, "eval_strategy": "no", "logging_steps": 1 if steps == 20 else 5,
            "generate_during_eval": False, "load_best_model_at_end": False, "save_strategy": "no",
            "save_total_limit": 1, "save_only_model": True, "save_safetensors": True,
            "push_to_hub": False, "report_to": ["wandb"], "output_dir": str(out), "run_name": name,
        }
        config = out / "config.yaml"
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        log = out / "train.log"
        env = os.environ.copy()
        env.update({
            "PYTHONPATH": str(a.project), "HF_HOME": str(a.cache), "CUDA_VISIBLE_DEVICES": str(a.gpu),
            "WANDB_MODE": "online", "WANDB_ENTITY": "promotion-kim", "WANDB_PROJECT": "mnpo",
            "WANDB_RUN_GROUP": "p17-q25-ipo-stage4-completion", "WANDB_RUN_ID": run_id,
            "WANDB_NAME": name, "WANDB_RESUME": "allow", "WANDB_DIR": str(a.output_root / "wandb"),
            "MNPO_DISABLE_CUDNN_SDPA": "1", "MNPO_DISABLE_APEX": "1",
            "TORCH_CUDNN_SDPA_ENABLED": "0", "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        })
        if phase == "smoke":
            env["MNPO_SKIP_FINAL_SAVE"] = "1"
        cmd = [str(a.python), "-m", "accelerate.commands.launch", "--config_file",
               str(a.project / "accelerate_configs/single_gpu.yaml"), "--num_processes=1",
               "-m", "mnpo_scripts.run_mnpo", str(config)]
        started = time.time()
        with log.open("w") as h:
            rc = subprocess.run(cmd, cwd=a.project, env=env, stdout=h, stderr=subprocess.STDOUT).returncode
        metrics_file = out / "train_results.json"
        metrics = json.loads(metrics_file.read_text()) if metrics_file.is_file() else {}
        finite = bool(metrics) and all(math.isfinite(float(v)) for v in metrics.values() if isinstance(v, (int, float)))
        clean = rc == 0 and finite and "Traceback (most recent call last)" not in log.read_text(errors="replace")[-50000:]
        status.write_text(json.dumps({
            "status": "completed" if clean else "failed", "seed": 43, "candidate": a.candidate,
            "phase": phase, "steps": steps, "returncode": rc, "finite_metrics": finite,
            "parent": str(a.parent), "dataset": str(a.dataset), "wandb_run_id": run_id,
            "wandb_url": f"https://wandb.ai/promotion-kim/mnpo/runs/{run_id}",
            "elapsed_seconds": time.time() - started,
        }, indent=2) + "\n")
        if not clean:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

