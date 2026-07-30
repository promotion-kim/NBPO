#!/usr/bin/env python3
"""Run frozen S3 and finalize status for an already-saved full model."""

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
    parser.add_argument("--method", choices=tf.METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=tf.SEEDS, required=True)
    parser.add_argument("--attempt", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--gpu", type=int, choices=range(8), required=True)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    args = parser.parse_args()

    output = args.root / "full" / args.method / f"seed{args.seed}" / f"attempt{args.attempt}"
    status_path = output / "job_status.json"
    prior = json.loads(status_path.read_text()) if status_path.exists() else {}
    finite, finite_reason = tf.metrics_finite(output)
    complete_model = tf.model_complete(output)
    gate_ok = False
    reason = finite_reason
    if finite and complete_model:
        namespace = SimpleNamespace(root=args.root, project=args.project, python=args.python)
        job = SimpleNamespace(gpu=args.gpu, output_dir=output)
        gate_ok, reason = tf.run_stability_gate(namespace, job)
    elif not complete_model:
        reason = "missing_final_model"

    status = {
        **prior,
        "status": "completed" if gate_ok else "failed",
        "stage": "full", "method": args.method, "seed": args.seed,
        "attempt": args.attempt, "gpu": args.gpu, "gpus": args.gpus,
        "finite_metrics": finite, "reason": reason,
        "optimizer_steps": 900, "effective_batch_size": 16,
        "output_dir": str(output),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tf.atomic_json(status_path, status)
    print(json.dumps(status, indent=2))
    raise SystemExit(0 if gate_ok else 1)


if __name__ == "__main__":
    main()
