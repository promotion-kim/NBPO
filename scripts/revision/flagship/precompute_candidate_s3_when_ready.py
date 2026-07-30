#!/usr/bin/env python3
"""Precompute one sweep candidate's frozen S3 gate on its now-free GPU."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from scripts.revision.flagship import train_flagship as tf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--stop-at", required=True)
    args = parser.parse_args()
    stop_at = time.mktime(time.strptime(args.stop_at, "%Y-%m-%dT%H:%M:%S%z"))
    while time.time() < stop_at and Path(f"/proc/{args.training_pid}").exists():
        time.sleep(10)
    if Path(f"/proc/{args.training_pid}").exists():
        raise TimeoutError(f"training PID still alive: {args.training_pid}")
    finite, reason = tf.metrics_finite(args.output)
    if not finite or not tf.model_complete(args.output):
        raise RuntimeError(f"candidate training artifact incomplete: {reason}")
    namespace = SimpleNamespace(root=args.root, project=args.project, python=args.python)
    job = SimpleNamespace(gpu=args.gpu, output_dir=args.output)
    passed, gate_reason = tf.run_stability_gate(namespace, job)
    evidence = {
        "status": "completed" if passed else "failed",
        "output": str(args.output), "gpu": args.gpu,
        "waited_for_training_pid": args.training_pid,
        "reason": gate_reason,
        "gate": str(args.output / "stability/gate.json"),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = args.output / "stability/precompute_status.json"
    path.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
