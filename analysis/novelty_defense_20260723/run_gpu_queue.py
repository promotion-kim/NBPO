#!/usr/bin/env python3
"""Backfill independent single-GPU decode jobs from a frozen JSON queue."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--queue", required=True)
    p.add_argument("--gpus", required=True)
    p.add_argument("--log-dir", required=True)
    args = p.parse_args()
    jobs = json.loads(Path(args.queue).read_text())
    gpus = args.gpus.split(",")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    active = {}
    pending = iter(jobs)

    def launch(gpu: str, job: dict) -> None:
        log = open(log_dir / f"{job['name']}.log", "a", buffering=1)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        process = subprocess.Popen(job["command"], env=env, stdout=log, stderr=subprocess.STDOUT)
        active[gpu] = (process, job, log)

    exhausted = False
    while active or not exhausted:
        for gpu in gpus:
            if gpu not in active and not exhausted:
                try:
                    launch(gpu, next(pending))
                except StopIteration:
                    exhausted = True
        time.sleep(5)
        for gpu, (process, job, log) in list(active.items()):
            rc = process.poll()
            if rc is not None:
                log.write(f"\nQUEUE_EXIT name={job['name']} rc={rc}\n")
                log.close()
                del active[gpu]
    failures = []
    for log in log_dir.glob("*.log"):
        if "QUEUE_EXIT" not in log.read_text(errors="replace") or "rc=0" not in log.read_text(errors="replace"):
            failures.append(log.stem)
    if failures:
        raise SystemExit(f"failed jobs: {failures}")


if __name__ == "__main__":
    main()
