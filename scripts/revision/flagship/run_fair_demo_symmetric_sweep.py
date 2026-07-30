#!/usr/bin/env python3
"""Idempotent four-GPU scheduler for the frozen fair-demo validation sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.revision.flagship.train_flagship import unified_config


METHOD_KEYS = {
    "learning_rate", "warmup_ratio", "ronpo_alpha", "ronpo_tau",
    "reference_anchor_weight", "preference_sft_weight", "dpo_beta",
    "simpo_beta", "simpo_gamma", "eta", "ratio",
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
    values = json.loads(result.read_text(encoding="utf-8"))
    numeric = [float(value) for value in values.values() if isinstance(value, (int, float))]
    return bool(numeric) and all(math.isfinite(value) for value in numeric)


def dataset_path(flagship_root: Path, fair_root: Path, name: str) -> Path:
    if name == "ronpo_full_expect_kall":
        return flagship_root / "precomputed/ronpo_full_expect"
    if name in {"ronpo_full_expect_k6", "ronpo_k_only_k1", "ronpo_k_only_k2"}:
        return fair_root / "precomputed" / name
    return flagship_root / "precomputed" / name


def run_id(candidate_id: str) -> str:
    return hashlib.sha256(f"qwen3-8b-fair-demo-v1|{candidate_id}|42".encode()).hexdigest()[:12]


def current_step(output: Path) -> int:
    state = output / "trainer_state.json"
    if not state.is_file():
        return 0
    try:
        return int(json.loads(state.read_text(encoding="utf-8")).get("global_step", 0))
    except (OSError, ValueError, json.JSONDecodeError):
        return 0


def idle_authorized_gpus(authorized: list[int]) -> list[int]:
    """Return authorized GPUs with no compute process, without mutating any process."""
    gpu_rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    uuid_to_index = {}
    for row in gpu_rows:
        index, uuid = [part.strip() for part in row.split(",", 1)]
        uuid_to_index[uuid] = int(index)
    process_rows = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    busy = set()
    for row in process_rows:
        if not row.strip():
            continue
        uuid = row.split(",", 1)[0].strip()
        if uuid in uuid_to_index:
            busy.add(uuid_to_index[uuid])
    return [gpu for gpu in authorized if gpu not in busy]


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
    args = parser.parse_args()

    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    lock = json.loads(args.prereg_lock.read_text(encoding="utf-8"))
    if grid.get("status") != "frozen_before_training_and_validation_ranking":
        raise RuntimeError("sweep grid is not frozen")
    if lock.get("files", {}).get("sweep/grid.json") != sha256(args.grid):
        raise RuntimeError("sweep grid hash differs from preregistration lock")
    candidates = grid["candidates"]
    if len(candidates) != 20:
        raise RuntimeError(f"expected 20 frozen candidates, found {len(candidates)}")
    counts = {}
    for candidate in candidates:
        counts[candidate["method"]] = counts.get(candidate["method"], 0) + 1
    if set(counts.values()) != {2}:
        raise RuntimeError(f"asymmetric tuning counts: {counts}")
    for candidate in candidates:
        dataset = dataset_path(args.flagship_root, args.fair_root, candidate["dataset"])
        if not (dataset / "dataset_dict.json").is_file():
            raise RuntimeError(f"missing frozen dataset for {candidate['id']}: {dataset}")

    args.work.mkdir(parents=True, exist_ok=True)
    logs = args.work / "logs"
    logs.mkdir(exist_ok=True)
    common = grid["common"]
    pending = []
    completed = []
    failures = []
    for candidate in candidates:
        output = args.work / "candidates" / candidate["id"]
        status_path = output / "training_status.json"
        if status_path.is_file() and model_complete(output) and finite_results(output):
            completed.append(candidate["id"])
        else:
            pending.append(candidate)

    authorized = [0, 1, 2, 3]
    available = []
    running = {}
    while pending or running:
        occupied_by_scheduler = set(running)
        for gpu in idle_authorized_gpus(authorized):
            if gpu not in occupied_by_scheduler and gpu not in available:
                available.append(gpu)
        available.sort()
        while pending and available:
            gpu = available.pop(0)
            candidate = pending.pop(0)
            identifier = candidate["id"]
            method = candidate["method"]
            output = args.work / "candidates" / identifier
            output.mkdir(parents=True, exist_ok=True)
            dataset = dataset_path(args.flagship_root, args.fair_root, candidate["dataset"])
            config = unified_config(args.base_model, str(dataset), output, "full", method, 42, 1)
            config.update({key: candidate[key] for key in METHOD_KEYS if key in candidate})
            config.update({
                "max_steps": int(common["optimizer_steps"]), "save_steps": int(common["optimizer_steps"]),
                "per_device_train_batch_size": int(common["per_device_train_batch_size"]),
                "gradient_accumulation_steps": int(common["gradient_accumulation_steps"]),
                "gradient_checkpointing": bool(common["gradient_checkpointing"]),
                "run_name": f"qwen3-8b-fair-demo-{identifier}", "output_dir": str(output),
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
                "HF_HOME": str(args.flagship_root / "cache/huggingface"),
                "HF_HUB_CACHE": str(args.flagship_root / "cache/huggingface/hub"),
                "HUGGINGFACE_HUB_CACHE": str(args.flagship_root / "cache/huggingface/hub"),
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "WANDB_MODE": "online", "WANDB_ENTITY": common["wandb_entity"],
                "WANDB_PROJECT": common["wandb_project"], "WANDB_RUN_GROUP": common["wandb_group"],
                "WANDB_RUN_ID": wandb_id, "WANDB_NAME": f"qwen3-8b-fair-demo-{identifier}",
                "WANDB_RESUME": "allow", "WANDB_DIR": str(args.work / "wandb" / identifier),
                "WANDB_INIT_TIMEOUT": "300",
                "MNPO_DISABLE_CUDNN_SDPA": "1", "TORCH_CUDNN_SDPA_ENABLED": "0",
                "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            })
            (args.work / "wandb" / identifier).mkdir(parents=True, exist_ok=True)
            log_path = logs / f"train_{identifier}.log"
            handle = log_path.open("a", encoding="utf-8")
            process = subprocess.Popen(command, cwd=args.project, env=environment,
                                       stdout=handle, stderr=subprocess.STDOUT)
            status = {
                "status": "running", "candidate_id": identifier, "method": method,
                "seed": 42, "gpu": gpu, "pid": process.pid, "config": str(config_path),
                "dataset": str(dataset), "optimizer_steps": common["optimizer_steps"],
                "effective_batch_size": common["effective_batch_size"],
                "wandb_run_id": wandb_id,
                "wandb_url": f"https://wandb.ai/{common['wandb_entity']}/{common['wandb_project']}/runs/{wandb_id}",
                "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            atomic_json(output / "training_status.json", status)
            running[gpu] = (candidate, process, handle, log_path, output, status)

        manifest = {
            "status": "running", "pending": [row["id"] for row in pending],
            "running": [{"gpu": gpu, "candidate_id": value[0]["id"],
                         "pid": value[1].pid, "measured_step": current_step(value[4])}
                        for gpu, value in running.items()],
            "completed": completed, "failures": failures,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "spent_sealed_split_touched": False,
        }
        atomic_json(args.work / "training_manifest.json", manifest)
        if not running:
            continue
        time.sleep(20)
        for gpu, (candidate, process, handle, log_path, output, status) in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            tail = log_path.read_text(errors="replace")[-30000:]
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
                failures.append({"candidate_id": candidate["id"], "gpu": gpu,
                                 "returncode": returncode, "log": str(log_path)})
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
