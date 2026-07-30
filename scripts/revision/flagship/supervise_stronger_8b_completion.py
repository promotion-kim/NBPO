#!/usr/bin/env python3
"""Outcome-blind handoff from stronger training to validation.

The supervisor never stops a process.  It waits for the frozen training
manifest, takes three read-only GPU samples, and starts validation only when
all four authorized devices have no compute process.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


def now() -> datetime:
    return datetime.now().astimezone()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def gpu_sample() -> dict:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    compute = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    return {
        "measured_at": now().isoformat(timespec="seconds"),
        "gpus": [line.strip() for line in gpu],
        "compute_apps": [line.strip() for line in compute if line.strip()],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--flagship-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-python", required=True)
    parser.add_argument("--eval-python", required=True)
    parser.add_argument("--general-rm-cache", type=Path, required=True)
    parser.add_argument("--latest-validation-start", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.latest_validation_start)
    state_path = args.root / "supervisor_status.json"
    validation_status = args.root / "validation/status.json"
    while True:
        if now() >= deadline:
            atomic_json(state_path, {
                "status": "stopped_before_validation", "reason": "latest start passed",
                "spent_sealed_split_touched": False, "updated_at": now().isoformat(timespec="seconds"),
            })
            return
        existing = read_json(validation_status)
        if existing and existing.get("status") in {"running", "completed"}:
            atomic_json(state_path, {
                "status": "validation_already_started", "validation_status": existing,
                "spent_sealed_split_touched": False, "updated_at": now().isoformat(timespec="seconds"),
            })
            return
        training = read_json(args.root / "training_manifest.json")
        if training and training.get("status") in {"completed", "failed"}:
            completed = [name for name, value in training.get("methods", {}).items()
                         if value.get("status") == "completed"]
            if not completed:
                atomic_json(state_path, {
                    "status": "stopped_no_completed_training", "training_manifest": training,
                    "spent_sealed_split_touched": False,
                    "updated_at": now().isoformat(timespec="seconds"),
                })
                return
            break
        atomic_json(state_path, {
            "status": "waiting_for_training", "training_status": (training or {}).get("status"),
            "spent_sealed_split_touched": False, "updated_at": now().isoformat(timespec="seconds"),
        })
        time.sleep(args.poll_seconds)

    samples = []
    while now() < deadline:
        samples = []
        for _ in range(3):
            samples.append(gpu_sample())
            time.sleep(2)
        if all(not sample["compute_apps"] for sample in samples):
            break
        atomic_json(state_path, {
            "status": "waiting_for_idle_gpus", "gpu_samples": samples,
            "policy": "No process is stopped or modified.", "spent_sealed_split_touched": False,
            "updated_at": now().isoformat(timespec="seconds"),
        })
        time.sleep(args.poll_seconds)
    else:
        return

    command = [
        args.train_python, str(args.project / "scripts/revision/flagship/run_stronger_8b_validation.py"),
        "--project", str(args.project), "--flagship-root", str(args.flagship_root),
        "--root", str(args.root), "--config", str(args.config),
        "--train-python", args.train_python, "--eval-python", args.eval_python,
        "--general-rm-cache", str(args.general_rm_cache),
    ]
    atomic_json(state_path, {
        "status": "launching_validation", "gpu_samples": samples, "command": command,
        "spent_sealed_split_touched": False, "updated_at": now().isoformat(timespec="seconds"),
    })
    log_path = args.root / "validation_driver.log"
    with log_path.open("a", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=args.project, stdout=handle, stderr=subprocess.STDOUT)
    validation = read_json(validation_status)
    if result.returncode != 0 or not validation or validation.get("status") != "completed":
        atomic_json(state_path, {
            "status": "validation_failed", "returncode": result.returncode,
            "validation_status": validation, "log": str(log_path),
            "spent_sealed_split_touched": False, "updated_at": now().isoformat(timespec="seconds"),
        })
        return

    power_path = args.root / "validation/results/raw_power_summary.json"
    power = read_json(power_path) or {}
    preregistration = "not_authorized_by_power_rule"
    if power.get("requires_fresh_sealed_preregistration") is True:
        prereg_command = [
            args.train_python,
            str(args.project / "scripts/revision/flagship/prepare_fresh_stronger_sealed_manifest.py"),
            "--source-pool", str(args.flagship_root / "data/pool_validation.jsonl"),
            "--spent-sealed", str(args.flagship_root / "data/sealed_test_prompts.jsonl"),
            "--power-summary", str(power_path),
            "--output-dir", str(args.root / "fresh_sealed_preregistration"), "--count", "604",
        ]
        with (args.root / "fresh_sealed_preregistration.log").open("a", encoding="utf-8") as handle:
            prereg = subprocess.run(prereg_command, cwd=args.project,
                                    stdout=handle, stderr=subprocess.STDOUT)
        preregistration = "completed_unopened" if prereg.returncode == 0 else "failed"
    atomic_json(state_path, {
        "status": "completed", "validation_status": validation,
        "fresh_sealed_preregistration": preregistration,
        "spent_sealed_split_touched": False, "updated_at": now().isoformat(timespec="seconds"),
    })


if __name__ == "__main__":
    main()
