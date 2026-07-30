#!/usr/bin/env python3
"""Decode a validation-locked model set on the pre-registered fixed test."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    lock = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_AFTER_VALIDATION_BEFORE_FIXED647_DECODE":
        raise RuntimeError("selection is not locked")
    models = [lock["base"], *lock["selected"]]
    gpu_queue = [value.strip() for value in args.gpus.split(",") if value.strip()]
    queued = []
    for row in models:
        output_dir = args.work / "generations" / row["candidate_id"]
        output = output_dir / "output_42.json"
        if output.is_file():
            value = json.loads(output.read_text(encoding="utf-8"))
            if isinstance(value, list) and len(value) == 647:
                continue
        queued.append((row, output_dir))
    env_base = os.environ.copy()
    env_base.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                     "TOKENIZERS_PARALLELISM": "false", "TORCH_CUDNN_SDPA_ENABLED": "0",
                     "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "PYTHONPATH": str(args.project)})
    logs = args.work / "logs/decode"; logs.mkdir(parents=True, exist_ok=True)
    running = []; completed = []
    while queued or running:
        while queued and len(running) < len(gpu_queue):
            row, output_dir = queued.pop(0)
            gpu = next(value for value in gpu_queue if value not in {item[1] for item in running})
            output_dir.mkdir(parents=True, exist_ok=True)
            command = [args.python, "-u", str(args.project / "scripts/revision/flagship/decode_vllm_non_thinking.py"),
                       "--data-dir", lock["fixed_test"]["path"], "--model", row["model_path"],
                       "--output-dir", str(output_dir), "--seed", "42", "--temperature", "0.7",
                       "--top-p", "0.9", "--max-new-tokens", "2048", "--max-model-len", "8192",
                       "--gpu-memory-utilization", "0.88"]
            env = env_base.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu
            handle = (logs / f"{row['candidate_id']}.log").open("a", encoding="utf-8")
            process = subprocess.Popen(command, cwd=args.project, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT)
            running.append((row["candidate_id"], gpu, process, handle, command))
        remainder = []
        for candidate_id, gpu, process, handle, command in running:
            rc = process.poll()
            if rc is None:
                remainder.append((candidate_id, gpu, process, handle, command)); continue
            handle.close(); completed.append({"candidate_id": candidate_id, "gpu": gpu,
                                               "returncode": rc, "command": command})
            if rc:
                atomic_json(args.work / "decode_status.json", {"status": "failed", "completed": completed,
                            "spent_sealed_split_touched": False})
                raise RuntimeError(f"fixed test decode failed: {candidate_id}, rc={rc}")
        running = remainder
        atomic_json(args.work / "decode_status.json", {
            "status": "running" if queued or running else "completed",
            "updated_at": datetime.now().astimezone().isoformat(), "completed": completed,
            "running": [{"candidate_id": row[0], "gpu": row[1]} for row in running],
            "queued": [row[0]["candidate_id"] for row in queued],
            "selection_lock": str(args.selection_lock), "spent_sealed_split_touched": False})
        if running:
            time.sleep(2)
    print(json.dumps({"status": "completed", "model_count": len(models)}, indent=2))


if __name__ == "__main__":
    main()
