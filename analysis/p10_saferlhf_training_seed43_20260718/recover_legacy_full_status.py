#!/usr/bin/env python3
"""Audit a legacy append-log full run without overwriting its original status."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    root = args.experiment / "stage1" / args.arm / "train" / "full"
    legacy = root / "job_status.json"
    log = args.experiment / "logs" / "stage1" / f"{args.arm}_full.log"
    metrics_file = root / "train_results.json"
    if not (legacy.is_file() and log.is_file() and metrics_file.is_file()):
        raise RuntimeError("legacy status, full log, or train results missing")
    text = log.read_text(encoding="utf-8", errors="replace")
    marker = "***** Running training *****"
    index = text.rfind(marker)
    if index < 0:
        raise RuntimeError("could not isolate the last full-training attempt")
    attempt = text[index:]
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    finite = bool(metrics) and all(math.isfinite(float(v)) for v in metrics.values() if isinstance(v, (int, float)))
    clean = finite and "Training complete" in attempt and "Traceback (most recent call last)" not in attempt and "out of memory" not in attempt.lower()
    old = json.loads(legacy.read_text(encoding="utf-8"))
    payload = {
        "status": "completed" if clean else "failed",
        "repaired_from_legacy_append_log": True,
        "original_status_path": str(legacy),
        "original_status": old.get("status"),
        "criterion": "Only the suffix beginning at the last '***** Running training *****' marker was inspected. Earlier tracebacks remain preserved in the original append-only log.",
        "latest_attempt_has_training_complete": "Training complete" in attempt,
        "latest_attempt_has_traceback": "Traceback (most recent call last)" in attempt,
        "latest_attempt_has_oom": "out of memory" in attempt.lower(),
        "finite_metrics": finite,
        "metrics": metrics,
        "model_dir": str(root),
        "full_log": str(log),
        "wandb_run_id": old.get("wandb_run_id"),
    }
    target = root / "job_status_repaired.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
