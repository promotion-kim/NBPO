#!/usr/bin/env python3
"""Launch the frozen stronger Qwen3-8B four-method training comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from scripts.revision.flagship.train_flagship import unified_config


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_id(method: str) -> str:
    return hashlib.sha256(f"qwen3-8b-stronger-power-v1|{method}|42".encode()).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text())
    if protocol.get("status") != "frozen_before_training":
        raise RuntimeError("stronger-training config was not frozen")
    methods = protocol["methods"]
    if {row["gpu"] for row in methods} != {0, 1, 2, 3}:
        raise RuntimeError("the frozen run must use exactly GPUs 0,1,2,3")
    shared = protocol["stronger_training"]
    train_root = args.root / "train"
    logs = args.root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    statuses = {}
    for method_row in methods:
        method = method_row["name"]
        gpu = int(method_row["gpu"])
        output = train_root / method
        output.mkdir(parents=True, exist_ok=True)
        dataset = args.root.parent / "flagship_20260712" / "precomputed" / method_row["dataset_variant"]
        if not (dataset / "dataset_dict.json").is_file():
            raise RuntimeError(f"missing frozen precomputed dataset: {dataset}")
        config = unified_config(
            protocol["base_model"]["remote_snapshot"], str(dataset), output,
            "full", method, int(protocol["seed"]), 1,
        )
        config.update({
            "learning_rate": float(shared["learning_rate"]),
            "max_steps": int(shared["optimizer_steps"]),
            "reference_anchor_weight": float(shared["reference_anchor_weight"]),
            "preference_sft_weight": float(shared["preference_sft_weight"]),
            "eta": float(shared["eta"]),
            "ronpo_alpha": float(shared["ronpo_alpha"]),
            "ronpo_tau": float(shared["ronpo_tau"]),
            "save_steps": int(shared["save_steps"]),
            "run_name": f"qwen3-8b-stronger-{method}-s42",
            "output_dir": str(output),
        })
        yaml_path = output / "config.yaml"
        yaml_path.write_text(yaml.safe_dump(config, sort_keys=False))
        command = [
            args.python, "-m", "accelerate.commands.launch",
            "--config_file", str(args.project / "accelerate_configs/single_gpu.yaml"),
            "--num_processes=1", "-m", "mnpo_scripts.run_mnpo", str(yaml_path),
        ]
        wandb_id = run_id(method)
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONPATH": str(args.project),
            "HF_HOME": str(args.root.parent / "flagship_20260712/cache/huggingface"),
            "HF_HUB_CACHE": str(args.root.parent / "flagship_20260712/cache/huggingface/hub"),
            "HUGGINGFACE_HUB_CACHE": str(args.root.parent / "flagship_20260712/cache/huggingface/hub"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "online", "WANDB_ENTITY": "promotion-kim", "WANDB_PROJECT": "mnpo",
            "WANDB_RUN_GROUP": "qwen3-8b-stronger-power-20260715",
            "WANDB_RUN_ID": wandb_id, "WANDB_NAME": f"qwen3-8b-stronger-{method}-s42",
            "WANDB_RESUME": "allow", "WANDB_DIR": str(args.root / "wandb" / method),
            "MNPO_DISABLE_CUDNN_SDPA": "1", "TORCH_CUDNN_SDPA_ENABLED": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        })
        (args.root / "wandb" / method).mkdir(parents=True, exist_ok=True)
        log_path = logs / f"train_{method}.log"
        handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(command, cwd=args.project, env=environment,
                                   stdout=handle, stderr=subprocess.STDOUT)
        status = {
            "status": "running", "method": method, "seed": 42, "gpu": gpu,
            "pid": process.pid, "optimizer_steps": shared["optimizer_steps"],
            "effective_batch_size": shared["effective_batch_size"],
            "learning_rate": shared["learning_rate"],
            "reference_anchor_weight": shared["reference_anchor_weight"],
            "preference_sft_weight": shared["preference_sft_weight"],
            "wandb_run_id": wandb_id,
            "wandb_url": f"https://wandb.ai/promotion-kim/mnpo/runs/{wandb_id}",
            "output_dir": str(output), "config": str(yaml_path), "log": str(log_path),
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        atomic_json(output / "training_status.json", status)
        statuses[method] = status
        processes.append((method, process, handle, output, log_path, status))
    atomic_json(args.root / "training_manifest.json", {
        "status": "running", "config": str(args.config),
        "methods": statuses, "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })

    failures = []
    for method, process, handle, output, log_path, status in processes:
        returncode = process.wait()
        handle.close()
        result_path = output / "train_results.json"
        model_exists = (output / "model.safetensors").is_file() or (output / "model.safetensors.index.json").is_file()
        finite = False
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            numeric = [float(value) for value in result.values() if isinstance(value, (int, float))]
            finite = bool(numeric) and all(math.isfinite(value) for value in numeric)
        log_tail = log_path.read_text(errors="replace")[-30000:]
        failed = returncode != 0 or not model_exists or not finite or "Traceback (most recent call last)" in log_tail
        status.update({
            "status": "failed" if failed else "completed",
            "returncode": returncode, "model_exists": model_exists,
            "finite_metrics": finite,
            "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        })
        atomic_json(output / "training_status.json", status)
        if failed:
            failures.append(method)
    atomic_json(args.root / "training_manifest.json", {
        "status": "failed" if failures else "completed", "config": str(args.config),
        "failed_methods": failures,
        "methods": {method: json.loads((train_root / method / "training_status.json").read_text())
                    for method in [row["name"] for row in methods]},
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    if failures:
        raise RuntimeError(f"stronger training failures: {failures}")


if __name__ == "__main__":
    main()
