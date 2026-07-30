#!/usr/bin/env python3
"""Decode the pre-locked frozen public Stage-1 candidates on validation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    if grid.get("status") != "LOCKED_BEFORE_VALIDATION_LOCAL_RM_RANKING":
        raise RuntimeError("candidate grid is not locked")
    if sum(bool(line.strip()) for line in args.prompts.open(encoding="utf-8")) != 128:
        raise RuntimeError("validation prompt file is not 128 records")
    jobs = []
    for candidate in grid["candidates"]:
        if candidate["source"] != "public_flagship_frozen_checkpoint":
            continue
        output = Path(candidate["validation_generation"])
        if output.is_file() and len(json.loads(output.read_text(encoding="utf-8"))) == 128:
            continue
        repo_cache_name = "models--" + candidate["hf_repo"].replace("/", "--")
        snapshot = args.hf_cache / repo_cache_name / "snapshots" / candidate["hf_revision"]
        if not snapshot.is_dir():
            raise RuntimeError(f"exact frozen snapshot is unavailable: {snapshot}")
        jobs.append((candidate, output, snapshot))
    gpu_queue = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpu_queue:
        raise RuntimeError("no GPUs supplied")
    logs = args.work / "validation/logs/frozen_decode"; logs.mkdir(parents=True, exist_ok=True)
    running: list[tuple[str, str, subprocess.Popen, object, list[str]]] = []
    completed = []
    env_base = os.environ.copy()
    env_base.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                     "TOKENIZERS_PARALLELISM": "false", "TORCH_CUDNN_SDPA_ENABLED": "0",
                     "PYTHONPATH": str(args.project)})

    while jobs or running:
        while jobs and len(running) < len(gpu_queue):
            candidate, output, snapshot = jobs.pop(0)
            gpu = next(value for value in gpu_queue if value not in {row[1] for row in running})
            output.parent.mkdir(parents=True, exist_ok=True)
            command = [args.python, "-u",
                       str(args.project / "scripts/revision/flagship/decode_vllm_non_thinking.py"),
                       "--data-dir", str(args.prompts), "--model", str(snapshot),
                       "--temperature", "0.7", "--top-p", "0.9", "--max-new-tokens", "2048",
                       "--max-model-len", "8192", "--output-dir", str(output.parent),
                       "--seed", "42", "--gpu-memory-utilization", "0.88"]
            env = env_base.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu
            handle = (logs / f"{candidate['id']}.log").open("a", encoding="utf-8")
            process = subprocess.Popen(command, cwd=args.project, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT)
            running.append((candidate["id"], gpu, process, handle, command))
        next_running = []
        for candidate_id, gpu, process, handle, command in running:
            returncode = process.poll()
            if returncode is None:
                next_running.append((candidate_id, gpu, process, handle, command)); continue
            handle.close()
            completed.append({"id": candidate_id, "gpu": gpu, "returncode": returncode,
                              "command": command})
            if returncode:
                atomic_json(args.work / "validation/frozen_decode_status.json", {
                    "status": "failed", "completed": completed, "spent_sealed_split_touched": False})
                raise RuntimeError(f"frozen validation decode failed: {candidate_id}, rc={returncode}")
        running = next_running
        if running:
            import time
            time.sleep(2)
        atomic_json(args.work / "validation/frozen_decode_status.json", {
            "status": "running" if jobs or running else "completed",
            "updated_at": datetime.now().astimezone().isoformat(), "completed": completed,
            "running": [{"id": row[0], "gpu": row[1]} for row in running],
            "queued": [row[0]["id"] for row in jobs], "spent_sealed_split_touched": False})
    print(json.dumps({"status": "completed", "models": [row["id"] for row in completed]}, indent=2))


if __name__ == "__main__":
    main()
