#!/usr/bin/env python3
"""Schedule exact-revision open-benchmark generation over four authorized GPUs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def models(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 11 or any(not row.get("revision") for row in rows):
        raise RuntimeError("models.tsv must contain exactly 11 exact revisions")
    return rows


def complete(path: Path, count: int) -> bool:
    response = path / "responses.jsonl"
    metadata = path / "decode_metadata.json"
    if not response.exists() or not metadata.exists():
        return False
    return sum(1 for line in response.open(encoding="utf-8") if line.strip()) == count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--models-tsv", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / "logs").mkdir(exist_ok=True)
    count = sum(1 for line in args.prompts.open(encoding="utf-8") if line.strip())
    if count != 1385:
        raise RuntimeError(f"expected 1385 benchmark items, got {count}")

    pending = [row for row in models(args.models_tsv) if not complete(args.work / "responses" / row["name"], count)]
    running: dict[int, tuple[dict, subprocess.Popen, object]] = {}
    completed: list[str] = []
    failed: list[dict] = []
    gpus = [int(value) for value in args.gpus.split(",")]
    status_path = args.work / "generation_status.json"

    while pending or running:
        for gpu, (model, process, handle) in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            target = args.work / "responses" / model["name"]
            if returncode == 0 and complete(target, count):
                completed.append(model["name"])
            else:
                failed.append({"model": model["name"], "returncode": returncode})
            del running[gpu]

        for gpu in [value for value in gpus if value not in running]:
            if not pending:
                break
            model = pending.pop(0)
            target = args.work / "responses" / model["name"]
            target.mkdir(parents=True, exist_ok=True)
            log_path = args.work / "logs" / f"decode_{model['name']}.log"
            handle = log_path.open("a", encoding="utf-8")
            command = [
                args.python,
                str(args.project / "scripts/revision/flagship/decode_openbench_vllm.py"),
                "--prompts", str(args.prompts),
                "--model-name", model["name"],
                "--model", model["model"],
                "--revision", model["revision"],
                "--output-dir", str(target),
                "--seed", "42", "--temperature", "0", "--top-p", "1",
                "--max-new-tokens", "2048", "--max-model-len", "32768",
                "--gpu-memory-utilization", "0.88",
            ]
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "TOKENIZERS_PARALLELISM": "false",
                "TORCH_CUDNN_SDPA_ENABLED": "0",
                "VLLM_HOST_IP": "127.0.0.1",
                "VLLM_PORT": str(62000 + gpu * 20),
                "VLLM_DP_MASTER_PORT": str(62001 + gpu * 20),
                "MASTER_PORT": str(62002 + gpu * 20),
            })
            process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
            running[gpu] = (model, process, handle)

        atomic_json(status_path, {
            "status": "running" if pending or running else "completed",
            "pending": [row["name"] for row in pending],
            "running": [
                {"gpu": gpu, "model": row["name"], "pid": process.pid}
                for gpu, (row, process, _) in running.items()
            ],
            "completed": sorted(set(completed)),
            "failed": failed,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        })
        if running:
            time.sleep(15)
        elif pending:
            time.sleep(2)

    if failed:
        raise RuntimeError(f"generation failures: {failed}")


if __name__ == "__main__":
    main()
