#!/usr/bin/env python3
"""Write one measured, rank-free hourly snapshot for the RONPO variant search."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path


def current_step(output: Path) -> int:
    best = 0
    for path in [output / "trainer_state.json", *output.glob("checkpoint-*/trainer_state.json")]:
        try:
            best = max(best, int(json.loads(path.read_text()).get("global_step", 0)))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    log = output.parents[1] / "logs" / f"train_{output.name}.log"
    if log.is_file():
        matches = re.findall(r"(\d+)/900", log.read_text(errors="replace"))
        if matches:
            best = max(best, max(map(int, matches)))
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    now = datetime.now().astimezone()
    gpus = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu,power.draw,temperature.gpu",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    candidates = []
    errors = []
    for output in sorted(args.weights_root.glob("round*/candidates/*")):
        status_path = output / "training_status.json"
        status = json.loads(status_path.read_text()) if status_path.is_file() else {}
        log = output.parents[1] / "logs" / f"train_{output.name}.log"
        text = log.read_text(errors="replace") if log.is_file() else ""
        found = sorted(set(re.findall(r"Traceback \(most recent call last\)|OutOfMemory|CUDA out of memory|NaN", text)))
        candidates.append({"candidate_id": output.name, "status": status.get("status", "unknown"),
                           "measured_step": current_step(output), "wandb_run_id": status.get("wandb_run_id"),
                           "errors": found})
        if found:
            errors.append({"candidate_id": output.name, "errors": found})
    snapshot = {
        "timestamp": now.isoformat(timespec="seconds"), "stage": args.stage,
        "gpu_snapshot": gpus, "compute_processes": [line for line in processes if line.strip()],
        "candidates": candidates, "errors": errors, "rank_claim": None,
        "spent_sealed_split_touched": False,
    }
    path = args.run_dir / "hourly" / now.strftime("%Y%m%dT%H%M%S%z.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
