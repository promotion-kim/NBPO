#!/usr/bin/env python3
"""Run one selected flagship full-training job on explicit GPUs.

This small recovery entry point is intentionally separate from the all-method
orchestrator: it lets a resumed interactive container fill GPUs without
relaunching already-running KTO processes adopted from the previous scheduler.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

from scripts.revision.flagship import train_flagship as tf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=tf.METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=tf.SEEDS, required=True)
    parser.add_argument("--gpu-ids", nargs="+", type=int, required=True)
    parser.add_argument(
        "--start-attempt", type=int, choices=(1, 2, 3), default=1,
        help="Resume at an already-declared stability attempt without recreating earlier attempts.",
    )
    args = parser.parse_args()

    required = 2 if args.method == "kto" else 1
    if len(args.gpu_ids) != required or len(set(args.gpu_ids)) != required:
        raise SystemExit(f"{args.method} requires exactly {required} distinct GPU(s)")
    if any(gpu not in range(8) for gpu in args.gpu_ids):
        raise SystemExit("GPU IDs must be in [0, 7]")

    protocol = json.loads(
        (args.project / "results/ronpo_flagship_20260712/objective_protocol.json").read_text()
    )
    if protocol["optimizer_steps"] != 900 or protocol["effective_batch_size"] != 16:
        raise RuntimeError("protocol budget mismatch")

    namespace = SimpleNamespace(
        root=args.root, project=args.project, python=args.python, model=args.model
    )
    status_path = args.root / "status/selected_workers" / f"{args.method}_s{args.seed}.json"
    tf.atomic_json(status_path, {
        "status": "running", "method": args.method, "seed": args.seed,
        "gpu_ids": args.gpu_ids, "max_attempts": 3,
        "start_attempt": args.start_attempt,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    failures = tf.run_queue(
        namespace, "full", [(args.method, args.seed)], max_attempts=3,
        gpu_ids=args.gpu_ids, start_attempt=args.start_attempt,
    )
    tf.atomic_json(status_path, {
        "status": "completed" if not failures else "terminal_failed",
        "method": args.method, "seed": args.seed, "gpu_ids": args.gpu_ids,
        "max_attempts": 3, "start_attempt": args.start_attempt, "failures": failures,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })


if __name__ == "__main__":
    main()
