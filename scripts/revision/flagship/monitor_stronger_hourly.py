#!/usr/bin/env python3
"""Write rank-free measured snapshots for the stronger Qwen3-8B run."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


STEP_RE = re.compile(r"(\d+)/1800")
ERROR_PATTERNS = ("Traceback (most recent call last)", "OutOfMemory", "CUDA out of memory", "FloatingPointError")


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def gpu_snapshot() -> list[dict]:
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, check=True)
    rows = []
    for line in result.stdout.splitlines():
        index, memory, utilization, temperature = [int(value.strip()) for value in line.split(",")]
        rows.append({"gpu": index, "memory_used_mib": memory,
                     "utilization_percent": utilization, "temperature_c": temperature})
    return rows


def progress(root: Path, method: str) -> dict:
    log = root / "logs" / f"train_{method}.log"
    text = log.read_text(errors="replace") if log.is_file() else ""
    steps = [int(value) for value in STEP_RE.findall(text.replace("\r", "\n"))]
    errors = [pattern for pattern in ERROR_PATTERNS if pattern in text]
    status = read_json(root / "train" / method / "training_status.json")
    return {"step": max(steps, default=0), "target_step": 1800,
            "errors": errors, "status": status.get("status") if status else "unknown",
            "wandb_run_id": status.get("wandb_run_id") if status else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=3600)
    parser.add_argument("--stop-at", required=True)
    args = parser.parse_args()
    stop = datetime.fromisoformat(args.stop_at).timestamp()
    methods = ("ronpo_full_expect", "ronpo_k_only", "inpo_avg", "sppo_avg")
    hourly = args.root / "hourly"
    hourly.mkdir(parents=True, exist_ok=True)
    while time.time() < stop:
        now = datetime.now().astimezone()
        validation = read_json(args.root / "validation/status.json")
        training = {method: progress(args.root, method) for method in methods}
        stage = validation.get("stage") if validation else "stronger_training"
        payload = {
            "measured_at": now.isoformat(timespec="seconds"), "stage": stage,
            "spent_sealed_split_touched": False, "gpus": gpu_snapshot(),
            "training": training, "training_manifest": read_json(args.root / "training_manifest.json"),
            "validation_status": validation,
            "errors_seen": {method: row["errors"] for method, row in training.items() if row["errors"]},
        }
        name = now.strftime("%Y%m%dT%H%M%S%z") + ".json"
        (hourly / name).write_text(json.dumps(payload, indent=2) + "\n")
        if validation and validation.get("status") in {"completed", "failed"}:
            break
        remaining = stop - time.time()
        if remaining <= 0:
            break
        time.sleep(min(args.interval_seconds, remaining))


if __name__ == "__main__":
    main()
