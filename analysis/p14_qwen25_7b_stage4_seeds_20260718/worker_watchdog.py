#!/usr/bin/env python3
"""Keep exactly one scheduler worker available on each authorized idle GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from common import ARMS, SEEDS


def processes() -> list[tuple[int, str]]:
    output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    rows = []
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2:
            rows.append((int(fields[0]), fields[1]))
    return rows


def gpu_busy(gpu: int) -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader", "-i", str(gpu)],
        text=True,
        capture_output=True,
        check=False,
    )
    return any(line.strip().isdigit() for line in result.stdout.splitlines())


def terminal_count(root: Path) -> int:
    scheduler = root / "scheduler"
    return sum(
        any((scheduler / f"s{seed}__stage{stage}__{arm}.{suffix}.json").exists()
            for suffix in ("DONE", "FAILED", "BLOCKED"))
        for stage in range(1, 5)
        for arm in ARMS
        for seed in SEEDS
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--train-python", type=Path, required=True)
    parser.add_argument("--infer-python", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY must be supplied ephemerally")

    total = 4 * len(ARMS) * len(SEEDS)
    log_dir = args.root / "watchdog"
    log_dir.mkdir(parents=True, exist_ok=True)
    events = log_dir / "events.jsonl"
    worker_script = args.project / "analysis/p14_qwen25_7b_stage4_seeds_20260718/worker.py"

    while True:
        terminal = terminal_count(args.root)
        if terminal == total:
            with events.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"time": time.time(), "event": "complete", "terminal": terminal}) + "\n")
            return

        rows = processes()
        for gpu in args.gpus:
            signature = f"{worker_script}"
            matches = [pid for pid, command in rows if signature in command and f"--root {args.root}" in command and f"--gpu {gpu}" in command]
            pending = [pid for pid, command in rows if "wait_then_start_worker.sh" in command and f" {args.root} " in command and f" {gpu} " in command]
            event = {"time": time.time(), "gpu": gpu, "terminal": terminal, "workers": matches, "pending": pending}
            if len(matches) > 1:
                event["event"] = "duplicate_workers_alert"
            elif matches or pending:
                event["event"] = "managed"
            elif gpu_busy(gpu):
                event["event"] = "busy_without_worker_no_action"
            else:
                log_path = log_dir / f"worker_g{gpu}.log"
                command = [
                    str(args.train_python), str(worker_script),
                    "--project", str(args.project),
                    "--train-python", str(args.train_python),
                    "--infer-python", str(args.infer_python),
                    "--root", str(args.root),
                    "--cache", str(args.cache),
                    "--gpu", str(gpu),
                ]
                with log_path.open("a", encoding="utf-8") as handle:
                    process = subprocess.Popen(
                        command,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                event.update({"event": "worker_started", "pid": process.pid, "log": str(log_path)})
            with events.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
