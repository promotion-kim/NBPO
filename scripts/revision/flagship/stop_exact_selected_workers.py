#!/usr/bin/env python3
"""Stop exact own selected-worker process trees after a dispatcher race."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from scripts.revision.flagship.apply_single_seed_cutover import (
    atomic_json,
    latest_progress,
    process_table,
    terminate_exact_tree,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job", action="append", required=True, help="METHOD:SEED")
    args = parser.parse_args()
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    stopped = []
    table = process_table()
    for spec in args.job:
        method, seed_text = spec.rsplit(":", 1)
        seed = int(seed_text)
        needle = f"--method {method} --seed {seed}"
        candidates = [
            pid for pid, (_, command) in table.items()
            if "run_selected_full.py" in command and needle in command
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"expected one worker for {spec}, found {candidates}")
        pid = candidates[0]
        pids = terminate_exact_tree(pid, method, seed)
        progress = latest_progress(args.root, method, seed)
        row = {
            "status": "stopped_by_user", "method": method, "seed": seed,
            "partial_optimizer_steps": progress,
            "reason": "single_seed_cutover_dispatcher_race",
            "resume_required": False, "stopped_at": timestamp,
            "terminated_exact_pids": pids,
        }
        atomic_json(
            args.root / "status/selected_workers" / f"{method}_s{seed}.json", row
        )
        statuses = sorted((args.root / "full" / method / f"seed{seed}").glob("attempt*/job_status.json"))
        if statuses:
            prior = json.loads(statuses[-1].read_text())
            atomic_json(statuses[-1], prior | row)
        stopped.append(row)
    atomic_json(args.root / "status/single_seed_cutover_race_cleanup.json", {
        "status": "completed", "stopped": stopped,
        "sealed_test_opened": False, "completed_at": timestamp,
    })


if __name__ == "__main__":
    main()
