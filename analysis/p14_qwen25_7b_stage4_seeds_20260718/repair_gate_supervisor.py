#!/usr/bin/env python3
"""Run the locked full-panel gate as soon as a repair training job finishes."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--train-python", type=Path, required=True)
    parser.add_argument("--infer-python", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--stage", type=int, required=True)
    args = parser.parse_args()
    candidate_root = args.repair_root / "candidates" / args.candidate
    status_out = candidate_root / "gate_supervisor_status.json"
    train_status = candidate_root / "train/full/job_status.json"
    deadline = time.time() + 3 * 3600
    while time.time() < deadline:
        if train_status.is_file():
            status = json.loads(train_status.read_text(encoding="utf-8"))
            if status.get("status") != "completed" or status.get("finite_metrics") is not True:
                write(status_out, {"status": "blocked", "reason": "training_failed", "training": status})
                return
            break
        time.sleep(30)
    else:
        write(status_out, {"status": "blocked", "reason": "training_timeout"})
        return
    gate_script = args.project / "analysis/p14_qwen25_7b_stage4_seeds_20260718/decode_gate_repair.sh"
    command = [
        str(gate_script), str(args.project), str(args.train_python), str(args.infer_python),
        str(args.repair_root), str(args.source_root), str(args.cache), args.candidate,
        str(args.gpu), str(args.stage),
    ]
    started = time.time()
    returncode = subprocess.run(command).returncode
    gate_path = candidate_root / "gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else None
    write(status_out, {
        "status": "completed" if returncode == 0 and gate is not None else "failed",
        "returncode": returncode,
        "gate": gate,
        "elapsed_seconds": time.time() - started,
    })


if __name__ == "__main__":
    main()
