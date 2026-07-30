#!/usr/bin/env python3
"""Write one measured, rank-free OS-only progress snapshot."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    now = datetime.now().astimezone()
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    gpus = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    candidates = []
    for status_path in sorted((args.train_root / "candidates").glob("*/training_status.json")):
        status = read_json(status_path)
        log = args.train_root / "logs" / f"train_{status_path.parent.name}.log"
        text = log.read_text(errors="replace") if log.is_file() else ""
        progress = [int(value) for value in re.findall(r"(?<!\d)(\d{1,3})/900", text)]
        errors = sorted(set(re.findall(r"Traceback \(most recent call last\)|OutOfMemory|CUDA out of memory|\bnan\b|\bNaN\b", text)))
        candidates.append({
            "candidate_id": status_path.parent.name, "status": status.get("status"),
            "gpu": status.get("gpu"), "pid": status.get("pid"),
            "measured_progress_step": max(progress, default=status.get("measured_step", 0) or 0),
            "wandb_run_id": status.get("wandb_run_id"), "wandb_url": status.get("wandb_url"),
            "error_tokens": errors,
        })
    payload = {
        "timestamp": now.isoformat(timespec="seconds"), "stage": args.stage,
        "gpu_snapshot": gpus, "compute_processes": [row for row in processes if row.strip()],
        "training_manifest": read_json(args.train_root / "training_manifest.json"),
        "candidates": candidates, "rank_computed": False,
        "spent_sealed_split_touched": False,
    }
    output = args.result_root / "hourly" / f"{now.strftime('%Y%m%dT%H%M%S%z')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
