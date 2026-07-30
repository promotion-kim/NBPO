#!/usr/bin/env python3
"""Stop only explicitly selected own workers and freeze the queue for evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path


STOP = {("ht_mnpo_safety", 43), ("ht_mnpo_safety", 44)}
KEEP = {("ht_mnpo_safety", 42), ("ht_mnpo_conciseness", 42)}
CANCEL = {("ht_mnpo_conciseness", 43), ("ht_mnpo_conciseness", 44)}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def process_table() -> dict[int, tuple[int, str]]:
    output = subprocess.check_output(
        ["ps", "-u", str(os.getuid()), "-o", "pid=,ppid=,cmd="], text=True
    )
    result = {}
    for line in output.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) == 3:
            result[int(fields[0])] = (int(fields[1]), fields[2])
    return result


def descendants(table: dict[int, tuple[int, str]], parent: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for pid, (ppid, _) in table.items():
        children.setdefault(ppid, []).append(pid)
    result = []

    def visit(pid: int) -> None:
        for child in children.get(pid, []):
            visit(child)
            result.append(child)

    visit(parent)
    return result


def terminate_exact_tree(parent: int, method: str, seed: int) -> list[int]:
    table = process_table()
    command = table.get(parent, (0, ""))[1]
    expected = f"--method {method} --seed {seed}"
    if "run_selected_full.py" not in command or expected not in command:
        raise RuntimeError(f"refusing PID {parent}: command does not match {expected!r}: {command}")
    targets = descendants(table, parent) + [parent]
    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 20
    while time.time() < deadline:
        alive = [pid for pid in targets if Path(f"/proc/{pid}").exists()]
        if not alive:
            return targets
        time.sleep(1)
    for pid in targets:
        if Path(f"/proc/{pid}").exists():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return targets


def latest_progress(root: Path, method: str, seed: int) -> int | None:
    logs = sorted((root / "logs").glob(f"full_{method}_s{seed}_a*.log"))
    if not logs:
        return None
    tail = logs[-1].read_bytes()[-500_000:].decode("utf-8", errors="replace").replace("\r", "\n")
    values = [int(value) for value in re.findall(r"(?:^|\s)(\d{1,3})/900(?:\s|$)", tail)]
    return max(values) if values else None


def pause_dispatcher(root: Path) -> list[int]:
    table = process_table()
    targets = [
        pid for pid, (_, command) in table.items()
        if Path(command.split()[0]).name.startswith("python")
        and "dispatch_resume_4gpu.py" in command
        and f"--root {root}" in command
    ]
    for pid in targets:
        try:
            # SIGSTOP keeps the tmux pane and already-launched seed-42 workers
            # alive while preventing any queued seed-43/44 launch.
            os.kill(pid, signal.SIGSTOP)
        except ProcessLookupError:
            pass
    return targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    dispatch_path = args.root / "status/recovery_dispatch_4gpu.json"
    dispatch = read_json(dispatch_path)
    running = {
        (str(row["method"]), int(row["seed"])): row
        for row in dispatch.get("running", [])
    }
    missing = sorted(STOP - set(running))
    if missing:
        raise RuntimeError(f"requested stop workers are not running: {missing}")

    stopped = []
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    for method, seed in sorted(STOP):
        row = running[(method, seed)]
        pids = terminate_exact_tree(int(row["pid"]), method, seed)
        progress = latest_progress(args.root, method, seed)
        worker_status = {
            "status": "stopped_by_user", "method": method, "seed": seed,
            "gpu_ids": row["gpu_ids"], "partial_optimizer_steps": progress,
            "reason": "single_seed_cutover_for_evaluation",
            "resume_required": False, "stopped_at": timestamp,
            "terminated_exact_pids": pids,
        }
        atomic_json(
            args.root / "status/selected_workers" / f"{method}_s{seed}.json",
            worker_status,
        )
        statuses = sorted((args.root / "full" / method / f"seed{seed}").glob("attempt*/job_status.json"))
        if statuses:
            prior = read_json(statuses[-1])
            atomic_json(statuses[-1], prior | worker_status)
        stopped.append(worker_status)

    dispatcher_pids = pause_dispatcher(args.root)
    kept = [row for key, row in running.items() if key in KEEP]
    atomic_json(dispatch_path, {
        "status": "stopped_by_user_for_evaluation",
        "policy": "single training seed 42",
        "pending": [],
        "cancelled_before_launch": [
            {"method": method, "seed": seed} for method, seed in sorted(CANCEL)
        ],
        "running_seed42_workers": kept,
        "stopped_workers": stopped,
        "prior_finished": dispatch.get("finished", []),
        "dispatcher_pids_sigstopped": dispatcher_pids,
        "updated_at": timestamp,
    })
    atomic_json(args.root / "status/single_seed_cutover.json", {
        "status": "completed", "kept_running": kept,
        "stopped": stopped,
        "cancelled_before_launch": [
            {"method": method, "seed": seed} for method, seed in sorted(CANCEL)
        ],
        "sealed_test_opened": False, "completed_at": timestamp,
    })


if __name__ == "__main__":
    main()
