#!/usr/bin/env python3
"""Run one frozen fair-demo candidate on an explicitly verified idle GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.revision.flagship.run_fair_demo_symmetric_sweep import METHOD_KEYS, dataset_path, run_id
from scripts.revision.flagship.train_flagship import unified_config


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def model_complete(path: Path) -> bool:
    return (path / "model.safetensors").is_file() or (path / "model.safetensors.index.json").is_file()


def finite_results(path: Path) -> bool:
    result = path / "train_results.json"
    if not result.is_file():
        return False
    values = json.loads(result.read_text())
    numeric = [float(value) for value in values.values() if isinstance(value, (int, float))]
    return bool(numeric) and all(math.isfinite(value) for value in numeric)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--prereg-lock", type=Path, required=True)
    parser.add_argument("--flagship-root", type=Path, required=True)
    parser.add_argument("--fair-root", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--wandb-run-id")
    args = parser.parse_args()
    grid = json.loads(args.grid.read_text())
    lock = json.loads(args.prereg_lock.read_text())
    if lock["files"]["sweep/grid.json"] != sha256(args.grid):
        raise RuntimeError("grid hash differs from preregistration")
    matches = [row for row in grid["candidates"] if row["id"] == args.candidate_id]
    if len(matches) != 1:
        raise RuntimeError("candidate is not uniquely present in frozen grid")
    candidate = matches[0]
    output = args.work / "candidates" / args.candidate_id
    output.mkdir(parents=True, exist_ok=True)
    if model_complete(output) and finite_results(output):
        return
    dataset = dataset_path(args.flagship_root, args.fair_root, candidate["dataset"])
    common = grid["common"]
    config = unified_config(args.base_model, str(dataset), output, "full", candidate["method"], 42, 1)
    config.update({key: candidate[key] for key in METHOD_KEYS if key in candidate})
    config.update({"max_steps": int(common["optimizer_steps"]),
                   "save_steps": int(common["optimizer_steps"]),
                   "per_device_train_batch_size": int(common["per_device_train_batch_size"]),
                   "gradient_accumulation_steps": int(common["gradient_accumulation_steps"]),
                   "gradient_checkpointing": bool(common["gradient_checkpointing"]),
                   "run_name": f"qwen3-8b-fair-demo-{args.candidate_id}", "output_dir": str(output)})
    config_path = output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = [args.python, "-m", "accelerate.commands.launch",
               "--config_file", str(args.project / "accelerate_configs/single_gpu.yaml"),
               "--num_processes=1", "-m", "mnpo_scripts.run_mnpo", str(config_path)]
    wandb_id = args.wandb_run_id or run_id(args.candidate_id)
    wandb_dir = args.work / "wandb" / (
        args.candidate_id if args.wandb_run_id is None else f"{args.candidate_id}_{wandb_id}"
    )
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONPATH": str(args.project),
                        "HF_HOME": str(args.flagship_root / "cache/huggingface"),
                        "HF_HUB_CACHE": str(args.flagship_root / "cache/huggingface/hub"),
                        "HUGGINGFACE_HUB_CACHE": str(args.flagship_root / "cache/huggingface/hub"),
                        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "WANDB_MODE": "online", "WANDB_ENTITY": common["wandb_entity"],
                        "WANDB_PROJECT": common["wandb_project"], "WANDB_RUN_GROUP": common["wandb_group"],
                        "WANDB_RUN_ID": wandb_id, "WANDB_NAME": f"qwen3-8b-fair-demo-{args.candidate_id}",
                        "WANDB_RESUME": "allow", "WANDB_DIR": str(wandb_dir),
                        "WANDB_INIT_TIMEOUT": "300",
                        "MNPO_DISABLE_CUDNN_SDPA": "1", "TORCH_CUDNN_SDPA_ENABLED": "0",
                        "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    wandb_dir.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "candidate_id": args.candidate_id, "method": candidate["method"],
              "seed": 42, "gpu": args.gpu, "config": str(config_path), "dataset": str(dataset),
              "optimizer_steps": common["optimizer_steps"], "effective_batch_size": common["effective_batch_size"],
              "wandb_run_id": wandb_id,
              "wandb_url": f"https://wandb.ai/{common['wandb_entity']}/{common['wandb_project']}/runs/{wandb_id}",
              "launcher": "supplemental_one_candidate", "started_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    atomic_json(output / "training_status.json", status)
    log = args.work / "logs" / f"train_{args.candidate_id}.log"
    with log.open("a", encoding="utf-8") as handle:
        returncode = subprocess.run(command, cwd=args.project, env=environment,
                                    stdout=handle, stderr=subprocess.STDOUT).returncode
    failed = returncode != 0 or not model_complete(output) or not finite_results(output)
    status.update({"status": "failed" if failed else "completed", "returncode": returncode,
                   "model_complete": model_complete(output), "finite_results": finite_results(output),
                   "completed_at": datetime.now().astimezone().isoformat(timespec="seconds")})
    atomic_json(output / "training_status.json", status)
    if failed:
        raise RuntimeError(f"supplemental candidate failed: {args.candidate_id}")


if __name__ == "__main__":
    main()
