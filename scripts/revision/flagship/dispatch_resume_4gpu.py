#!/usr/bin/env python3
"""Idempotently resume the remaining frozen P1 jobs on four authorized B200s."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


QUEUE = [
    ("inpo_avg", 43, 1, 1),
    ("inpo_avg", 44, 1, 1),
    ("ht_mnpo_helpfulness", 42, 1, 1),
    ("ht_mnpo_helpfulness", 43, 1, 1),
    ("ht_mnpo_helpfulness", 44, 1, 1),
    ("ht_mnpo_safety", 42, 1, 1),
    ("ht_mnpo_safety", 43, 1, 1),
    ("ht_mnpo_safety", 44, 1, 1),
    ("ht_mnpo_conciseness", 42, 1, 1),
    ("ht_mnpo_conciseness", 43, 1, 1),
    ("ht_mnpo_conciseness", 44, 1, 1),
]


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def selected_status(root: Path, method: str, seed: int) -> str | None:
    path = root / "status/selected_workers" / f"{method}_s{seed}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("status")
    except (OSError, json.JSONDecodeError):
        return None


def free_authorized_gpus(reserved: set[int]) -> list[int]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ], text=True)
    free = []
    for line in output.splitlines():
        index_text, memory_text = (part.strip() for part in line.split(",", 1))
        index, memory = int(index_text), int(memory_text)
        if index in range(4) and index not in reserved and memory < 1000:
            free.append(index)
    return free


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stop-launch-at", required=True)
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()
    stop_at = datetime.fromisoformat(args.stop_launch_at).timestamp()
    pending = list(QUEUE)
    running: dict[tuple[str, int], tuple[subprocess.Popen, tuple[int, ...], object]] = {}
    finished: list[dict] = []
    status_path = args.root / "status/recovery_dispatch_4gpu.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.project)

    while pending or running:
        for job, (process, gpus, handle) in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            method, seed = job
            finished.append({
                "method": method, "seed": seed, "gpu_ids": list(gpus),
                "returncode": returncode,
                "final_status": selected_status(args.root, method, seed),
            })
            del running[job]

        for item in list(pending):
            method, seed, start_attempt, required = item
            status = selected_status(args.root, method, seed)
            if status in {"completed", "terminal_failed"}:
                finished.append({
                    "method": method, "seed": seed, "gpu_ids": [],
                    "returncode": None, "final_status": status,
                    "note": "pre-existing terminal status; not relaunched",
                })
                pending.remove(item)

        now = time.time()
        if now < stop_at:
            launched = True
            while launched:
                launched = False
                reserved = {gpu for _, gpus, _ in running.values() for gpu in gpus}
                free = free_authorized_gpus(reserved)
                for item in list(pending):
                    method, seed, start_attempt, required = item
                    if len(free) < required:
                        continue
                    gpus = tuple(free[:required])
                    log_path = args.root / "logs" / f"resume_{method}_s{seed}.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    handle = log_path.open("a", encoding="utf-8")
                    command = [
                        args.python, "scripts/revision/flagship/run_selected_full.py",
                        "--root", str(args.root), "--project", str(args.project),
                        "--python", args.python, "--model", args.model,
                        "--method", method, "--seed", str(seed),
                        "--start-attempt", str(start_attempt), "--gpu-ids",
                        *(str(gpu) for gpu in gpus),
                    ]
                    process = subprocess.Popen(
                        command, cwd=args.project, env=env, stdout=handle,
                        stderr=subprocess.STDOUT,
                    )
                    running[(method, seed)] = (process, gpus, handle)
                    pending.remove(item)
                    launched = True
                    break

        atomic_json(status_path, {
            "status": (
                "completed" if not pending and not running
                else "launch_deadline_reached" if now >= stop_at
                else "running"
            ),
            "authorized_gpu_ids": [0, 1, 2, 3],
            "stop_launch_at": args.stop_launch_at,
            "pending": [
                {"method": m, "seed": s, "start_attempt": a, "gpus": n}
                for m, s, a, n in pending
            ],
            "running": [
                {"method": m, "seed": s, "gpu_ids": list(gpus), "pid": process.pid}
                for (m, s), (process, gpus, _) in running.items()
            ],
            "finished": finished,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        })
        if now >= stop_at and not running:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
