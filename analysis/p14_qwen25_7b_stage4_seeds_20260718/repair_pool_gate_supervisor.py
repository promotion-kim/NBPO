#!/usr/bin/env python3
"""Gate each repaired stage and serialize continuation-pool construction."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path


def passed(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("passed") is True and data.get("status") == "passed"
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def run_locked(lock_path: Path, command: list[str], log: Path, env: dict[str, str]) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock, log.open("a", encoding="utf-8") as handle:
        fcntl.flock(lock, fcntl.LOCK_EX)
        handle.write(json.dumps({"started": time.time(), "command": command}) + "\n")
        handle.flush()
        if subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode:
            raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--train-python", type=Path, required=True)
    p.add_argument("--infer-python", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--arm", choices=["inpo_avg", "ipo"], required=True)
    p.add_argument("--gpu", type=int, required=True)
    args = p.parse_args()
    env = os.environ.copy()
    lock = args.root / "scheduler/repair_pool_gpu.lock"
    log = args.root / f"seeds/s42/logs/{args.arm}_pool_gate_supervisor.log"
    gate_script = args.project / "analysis/p14_qwen25_7b_stage4_seeds_20260718/decode_and_gate.sh"
    pool_script = args.project / "analysis/p14_qwen25_7b_stage4_seeds_20260718/prepare_continuation_pool.sh"
    for stage in (2, 3, 4):
        status = args.root / f"seeds/s42/stage{stage}/{args.arm}/train/full/job_status.json"
        while True:
            try:
                data = json.loads(status.read_text(encoding="utf-8"))
                if data.get("status") == "completed" and data.get("finite_metrics") is True:
                    break
                if data.get("status") == "failed":
                    raise SystemExit(1)
            except FileNotFoundError:
                pass
            time.sleep(30)
        gate = args.root / f"seeds/s42/stage{stage}/gates/{args.arm}.json"
        if not passed(gate):
            command = ["bash", str(gate_script), str(args.project), str(args.train_python), str(args.infer_python),
                       str(args.root), "42", str(stage), args.arm, str(args.gpu)]
            run_locked(lock, command, log, env)
        if not passed(gate):
            raise SystemExit(1)
        if stage < 4:
            prepared = args.root / f"seeds/s42/stage{stage + 1}/{args.arm}/pool/PREPARED"
            if not prepared.is_file():
                command = ["bash", str(pool_script), str(args.project), str(args.train_python), str(args.infer_python),
                           str(args.root), str(args.cache), "42", str(stage + 1), args.arm, str(args.gpu)]
                run_locked(lock, command, log, env)


if __name__ == "__main__":
    main()
