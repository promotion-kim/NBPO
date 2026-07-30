#!/usr/bin/env python3
"""Fill newly free B200s with the remaining frozen-profile flagship jobs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def worker_reserved_gpus(root: Path) -> set[int]:
    reserved: set[int] = set()
    status_dir = root / "status/selected_workers"
    for path in status_dir.glob("*.json") if status_dir.exists() else ():
        try:
            status = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if status.get("status") == "running":
            reserved.update(int(gpu) for gpu in status.get("gpu_ids", []))
    return reserved


def free_gpus(root: Path, reserved: set[int]) -> list[int]:
    command = [
        "nvidia-smi", "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    free = []
    worker_reserved = worker_reserved_gpus(root)
    for line in output.splitlines():
        index_text, memory_text = (part.strip() for part in line.split(",", 1))
        index, memory = int(index_text), int(memory_text)
        if index not in reserved and index not in worker_reserved and memory < 1000:
            free.append(index)
    return free


def job_status(root: Path, method: str, seed: int) -> str | None:
    for attempt in (3, 2, 1):
        path = root / "full" / method / f"seed{seed}" / f"attempt{attempt}/job_status.json"
        if path.exists():
            return json.loads(path.read_text()).get("status")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stop-launch-at", required=True, help="ISO-8601 timestamp with UTC offset")
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()
    stop_at = datetime.fromisoformat(args.stop_launch_at).timestamp()

    queue = [
        ("inpo_avg", 43), ("inpo_avg", 44),
        ("ht_mnpo_helpfulness", 42), ("ht_mnpo_helpfulness", 43),
        ("ht_mnpo_helpfulness", 44), ("ht_mnpo_safety", 42),
        ("ht_mnpo_safety", 43), ("ht_mnpo_safety", 44),
        ("ht_mnpo_conciseness", 42), ("ht_mnpo_conciseness", 43),
        ("ht_mnpo_conciseness", 44),
    ]
    running: dict[tuple[str, int], tuple[subprocess.Popen, int, object]] = {}
    finished: list[dict] = []
    status_path = args.root / "status/recovery_dispatch.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.project)

    while queue or running:
        for job, (process, gpu, handle) in list(running.items()):
            rc = process.poll()
            if rc is None:
                continue
            handle.close()
            method, seed = job
            finished.append({
                "method": method, "seed": seed, "gpu": gpu, "returncode": rc,
                "final_status": job_status(args.root, method, seed),
            })
            del running[job]

        now = time.time()
        reserved = {gpu for _, gpu, _ in running.values()}
        if now < stop_at:
            available = free_gpus(args.root, reserved)
            while queue and available:
                method, seed = queue.pop(0)
                prior = job_status(args.root, method, seed)
                if prior in {"completed", "failed", "running"}:
                    finished.append({
                        "method": method, "seed": seed, "gpu": None,
                        "returncode": None, "final_status": prior,
                        "note": "pre-existing status; not relaunched",
                    })
                    continue
                gpu = available.pop(0)
                log_path = args.root / "logs" / f"recovery_{method}_s{seed}.log"
                handle = log_path.open("a", encoding="utf-8")
                command = [
                    args.python, "scripts/revision/flagship/run_selected_full.py",
                    "--root", str(args.root), "--project", str(args.project),
                    "--python", args.python, "--model", args.model,
                    "--method", method, "--seed", str(seed), "--gpu-ids", str(gpu),
                ]
                process = subprocess.Popen(
                    command, cwd=args.project, env=env, stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                running[(method, seed)] = (process, gpu, handle)

        atomic_json(status_path, {
            "status": (
                "completed" if not queue and not running
                else "launch_deadline_reached" if now >= stop_at
                else "running"
            ),
            "stop_launch_at": args.stop_launch_at,
            "queue": [{"method": method, "seed": seed} for method, seed in queue],
            "running": [
                {"method": method, "seed": seed, "gpu": gpu, "pid": process.pid}
                for (method, seed), (process, gpu, _) in running.items()
            ],
            "finished": finished,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        })
        if now >= stop_at and not running:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
