#!/usr/bin/env python3
"""Run one frozen four-candidate RONPO variant-search round on GPUs 0--3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml

from scripts.revision.flagship.train_flagship import unified_config


CONFIG_KEYS = {
    "learning_rate", "warmup_ratio", "ronpo_alpha", "ronpo_tau", "eta",
    "reference_anchor_weight", "preference_sft_weight", "ronpo_target_column",
    "ronpo_target_schedule_columns", "ronpo_target_schedule_boundaries",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def model_complete(path: Path) -> bool:
    return (path / "model.safetensors").is_file() or (path / "model.safetensors.index.json").is_file()


def finite_results(path: Path) -> bool:
    result = path / "train_results.json"
    if not result.is_file():
        return False
    data = json.loads(result.read_text(encoding="utf-8"))
    values = [float(value) for value in data.values() if isinstance(value, (int, float))]
    return bool(values) and all(math.isfinite(value) for value in values)


def gpu_snapshot() -> dict:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    return {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "gpus": gpu, "compute_processes": [row for row in processes if row.strip()]}


def current_step(output: Path) -> int:
    candidates = [output / "trainer_state.json", *sorted(output.glob("checkpoint-*/trainer_state.json"))]
    best = 0
    for path in candidates:
        if path.is_file():
            try:
                best = max(best, int(json.loads(path.read_text()).get("global_step", 0)))
            except (OSError, ValueError, json.JSONDecodeError):
                pass
    return best


def run_id(candidate: str) -> str:
    return hashlib.sha256(f"ronpo-8b-variant-search-v1|{candidate}|42".encode()).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--grid-lock", type=Path, required=True)
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    args = parser.parse_args()
    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    lock = json.loads(args.grid_lock.read_text(encoding="utf-8"))
    if grid.get("status") != "frozen_before_round_launch_and_ranking":
        raise RuntimeError("round grid is not frozen")
    if lock.get("grid_sha256") != sha256(args.grid):
        raise RuntimeError("round grid hash mismatch")
    if lock.get("spent_sealed_split_touched") is not False:
        raise RuntimeError("integrity lock missing")
    candidates = grid["candidates"]
    if not 1 <= len(candidates) <= 4:
        raise RuntimeError("a round must contain one to four candidates")
    if not (args.target_dataset / "dataset_dict.json").is_file():
        raise RuntimeError("variant target dataset is missing")

    samples = []
    for index in range(3):
        sample = gpu_snapshot(); samples.append(sample)
        if sample["compute_processes"]:
            raise RuntimeError(f"authorized GPU process detected before launch: {sample['compute_processes']}")
        if index < 2:
            time.sleep(2)
    atomic_json(args.work / "prelaunch_gpu_samples.json", {
        "status": "verified_idle", "samples": samples, "authorized_gpu_ids": [0, 1, 2, 3],
        "spent_sealed_split_touched": False,
    })

    common = grid["common"]
    pending = []
    completed = []
    for candidate in candidates:
        output = args.work / "candidates" / candidate["id"]
        if model_complete(output) and finite_results(output):
            completed.append(candidate["id"])
        else:
            pending.append(candidate)
    running = {}
    failures = []
    while pending or running:
        for gpu in range(4):
            if not pending or gpu in running:
                continue
            candidate = pending.pop(0)
            identifier = candidate["id"]
            output = args.work / "candidates" / identifier
            output.mkdir(parents=True, exist_ok=True)
            config = unified_config(args.base_model, str(args.target_dataset), output, "full", "ronpo_full_expect", 42, 1)
            config.update({key: candidate[key] for key in CONFIG_KEYS if key in candidate})
            config.update({
                "max_steps": int(common["optimizer_steps"]),
                "save_strategy": "steps", "save_steps": int(common["save_steps"]),
                "save_total_limit": int(common["save_total_limit"]), "save_only_model": True,
                "per_device_train_batch_size": 1, "per_device_eval_batch_size": 1,
                "gradient_accumulation_steps": 16, "gradient_checkpointing": True,
                "run_name": f"ronpo-8b-variant-search-{identifier}", "output_dir": str(output),
            })
            config_path = output / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            command = [args.python, "-m", "accelerate.commands.launch",
                       "--config_file", str(args.project / "accelerate_configs/single_gpu.yaml"),
                       "--num_processes=1", "-m", "mnpo_scripts.run_mnpo", str(config_path)]
            wandb_id = run_id(identifier)
            environment = os.environ.copy()
            environment.update({
                "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(args.project),
                "HF_HOME": str(Path(args.base_model).parents[3] if "snapshots" in args.base_model else args.work / "cache"),
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "WANDB_MODE": "online", "WANDB_ENTITY": "promotion-kim", "WANDB_PROJECT": "mnpo",
                "WANDB_RUN_GROUP": "qwen3-8b-ronpo-variant-search-20260715",
                "WANDB_RUN_ID": wandb_id, "WANDB_NAME": f"ronpo-8b-variant-search-{identifier}",
                "WANDB_RESUME": "allow", "WANDB_DIR": str(args.work / "wandb" / identifier),
                "WANDB_INIT_TIMEOUT": "300", "MNPO_DISABLE_CUDNN_SDPA": "1",
                "TORCH_CUDNN_SDPA_ENABLED": "0", "TOKENIZERS_PARALLELISM": "false",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            })
            (args.work / "wandb" / identifier).mkdir(parents=True, exist_ok=True)
            log_path = args.work / "logs" / f"train_{identifier}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            process = subprocess.Popen(command, cwd=args.project, env=environment, stdout=handle, stderr=subprocess.STDOUT)
            status = {
                "status": "running", "candidate_id": identifier, "gpu": gpu, "pid": process.pid,
                "config": str(config_path), "target_dataset": str(args.target_dataset),
                "optimizer_steps": common["optimizer_steps"], "effective_batch_size": 16,
                "wandb_run_id": wandb_id,
                "wandb_url": f"https://wandb.ai/promotion-kim/mnpo/runs/{wandb_id}",
                "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "spent_sealed_split_touched": False,
            }
            atomic_json(output / "training_status.json", status)
            running[gpu] = (candidate, process, handle, log_path, output, status)
        atomic_json(args.work / "training_manifest.json", {
            "status": "running", "pending": [row["id"] for row in pending],
            "running": [{"gpu": gpu, "candidate": row[0]["id"], "pid": row[1].pid,
                         "measured_step": current_step(row[4])} for gpu, row in running.items()],
            "completed": completed, "failures": failures,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "spent_sealed_split_touched": False,
        })
        if not running:
            continue
        time.sleep(20)
        for gpu, (candidate, process, handle, log_path, output, status) in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            tail = log_path.read_text(errors="replace")[-50000:]
            failed = (returncode != 0 or not model_complete(output) or not finite_results(output)
                      or "Traceback (most recent call last)" in tail or "OutOfMemory" in tail
                      or "CUDA out of memory" in tail)
            status.update({
                "status": "failed" if failed else "completed", "returncode": returncode,
                "model_complete": model_complete(output), "finite_results": finite_results(output),
                "measured_step": current_step(output),
                "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            })
            atomic_json(output / "training_status.json", status)
            if failed:
                failures.append({"candidate_id": candidate["id"], "returncode": returncode, "log": str(log_path)})
            else:
                completed.append(candidate["id"])
            del running[gpu]
    final = {
        "status": "failed" if failures else "completed", "completed": completed,
        "failures": failures, "grid_sha256": sha256(args.grid),
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "spent_sealed_split_touched": False,
    }
    atomic_json(args.work / "training_manifest.json", final)
    if failures:
        raise RuntimeError(json.dumps(failures, indent=2))


if __name__ == "__main__":
    main()
