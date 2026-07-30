#!/usr/bin/env python3
"""Select the first passing preregistered SPPO repair and finish Stage 4."""

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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--train-python", type=Path, required=True)
    p.add_argument("--infer-python", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    args = p.parse_args()
    ext = args.root / "sppo_extension"
    selected = None
    for candidate in ("sppo_strong_a", "sppo_strong_b"):
        gate = ext / f"candidates/{candidate}/gate.json"
        status = ext / f"candidates/{candidate}/gate_supervisor_status.json"
        while not gate.is_file() and not status.is_file():
            time.sleep(30)
        if passed(gate):
            selected = candidate
            break
    selection = args.root / "seeds/s42/stage3/sppo_avg/repair_selection.json"
    selection.parent.mkdir(parents=True, exist_ok=True)
    selection.write_text(json.dumps({"status": "selected" if selected else "failed", "candidate": selected,
                                     "rule": "A if full gate passes, otherwise B", "selected_at": time.time()}, indent=2) + "\n")
    if selected is None:
        raise SystemExit(1)
    parent = args.root / "seeds/s42/stage3/sppo_avg/train/full"
    parent.parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        parent.unlink()
    if not parent.exists():
        parent.symlink_to(ext / f"candidates/{selected}/train/full", target_is_directory=True)
    out = args.root / "seeds/s42/stage3"
    (out / "generations/sppo_avg").mkdir(parents=True, exist_ok=True)
    (out / "gates").mkdir(parents=True, exist_ok=True)
    for source, target in ((ext / f"candidates/{selected}/generation/output_42.json", out / "generations/sppo_avg/output_42.json"),
                           (ext / f"candidates/{selected}/gate.json", out / "gates/sppo_avg.json")):
        if target.is_symlink():
            target.unlink()
        if not target.exists():
            target.symlink_to(source)
    lock_path = args.root / "scheduler/repair_pool_gpu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pool = ["bash", str(args.project / "analysis/p14_qwen25_7b_stage4_seeds_20260718/prepare_continuation_pool.sh"),
                str(args.project), str(args.train_python), str(args.infer_python), str(args.root), str(args.cache),
                "42", "4", "sppo_avg", str(args.gpu)]
        if subprocess.run(pool, env=os.environ.copy()).returncode:
            raise SystemExit(1)
    status = args.root / "seeds/s42/stage4/sppo_avg/train/full/job_status.json"
    while not status.is_file():
        time.sleep(30)
    data = json.loads(status.read_text())
    if data.get("status") != "completed" or data.get("finite_metrics") is not True:
        raise SystemExit(1)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        gate = ["bash", str(args.project / "analysis/p14_qwen25_7b_stage4_seeds_20260718/decode_and_gate.sh"),
                str(args.project), str(args.train_python), str(args.infer_python), str(args.root), "42", "4", "sppo_avg", str(args.gpu)]
        raise SystemExit(subprocess.run(gate, env=os.environ.copy()).returncode)


if __name__ == "__main__":
    main()
