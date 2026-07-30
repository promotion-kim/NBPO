#!/usr/bin/env python3
"""Decode a frozen RONPO checkpoint manifest on validation and gate fail-closed."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def complete_generation(path: Path, expected: int = 128) -> bool:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(rows, list) and len(rows) == expected and all(
        str(row.get("prompt", "")).strip() and str(row.get("generated_text", "")).strip()
        for row in rows
    )


def gpu_snapshot() -> dict:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    return {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "gpus": gpu, "compute_processes": [row for row in processes if row.strip()]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--flagship-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_validation_decode_and_ranking":
        raise RuntimeError("checkpoint manifest is not frozen")
    models = manifest["models"]
    if not models:
        raise RuntimeError("empty checkpoint manifest")

    samples = []
    for index in range(3):
        sample = gpu_snapshot(); samples.append(sample)
        if sample["compute_processes"]:
            raise RuntimeError(f"GPU process detected before validation launch: {sample['compute_processes']}")
        if index < 2:
            time.sleep(2)
    atomic_json(args.work / "prelaunch_gpu_samples.json", {
        "status": "verified_idle", "samples": samples, "authorized_gpu_ids": [0, 1, 2, 3],
        "spent_sealed_split_touched": False,
    })
    generations = args.work / "generations"
    logs = args.work / "logs"; logs.mkdir(parents=True, exist_ok=True)
    base_source = args.flagship_root / "eval/p1_validation_reward_seed42/generations/base/output_42.json"
    if not complete_generation(base_source):
        raise RuntimeError("frozen base validation generation unavailable")
    base_output = generations / "base/output_42.json"
    base_output.parent.mkdir(parents=True, exist_ok=True)
    if not complete_generation(base_output):
        shutil.copy2(base_source, base_output)
        metadata = base_source.parent / "decode_metadata.json"
        if metadata.is_file():
            shutil.copy2(metadata, base_output.parent / "decode_metadata.json")

    pending = [row for row in models if not complete_generation(
        generations / row["model_id"] / "output_42.json")]
    running: dict[int, tuple] = {}
    available = [0, 1, 2, 3]
    failures = []
    while pending or running:
        while pending and available:
            gpu = available.pop(0); row = pending.pop(0)
            model_id = row["model_id"]
            output_dir = generations / model_id; output_dir.mkdir(parents=True, exist_ok=True)
            command = [
                args.python, str(args.project / "scripts/revision/flagship/decode_vllm_non_thinking.py"),
                "--data-dir", str(args.flagship_root / "data/pool_validation.jsonl"),
                "--model", row["model_path"], "--output-dir", str(output_dir),
                "--seed", "42", "--temperature", "0.7", "--top-p", "0.9",
                "--max-new-tokens", "2048", "--max-prompts", "128",
                "--gpu-memory-utilization", "0.80",
            ]
            environment = os.environ.copy()
            environment.update({
                "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(args.project),
                "HF_HOME": str(args.flagship_root / "cache/huggingface"),
                "HF_HUB_CACHE": str(args.flagship_root / "cache/huggingface/hub"),
                "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false", "TORCH_CUDNN_SDPA_ENABLED": "0",
            })
            handle = (logs / f"decode_{model_id}.log").open("a", encoding="utf-8")
            process = subprocess.Popen(command, cwd=args.project, env=environment,
                                       stdout=handle, stderr=subprocess.STDOUT)
            running[gpu] = (row, process, handle, command)
        atomic_json(args.work / "status.json", {
            "status": "running", "stage": "validation_decode",
            "pending": [row["model_id"] for row in pending],
            "running": [{"gpu": gpu, "model_id": value[0]["model_id"], "pid": value[1].pid}
                        for gpu, value in running.items()],
            "failures": failures, "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "spent_sealed_split_touched": False,
        })
        if not running:
            continue
        time.sleep(10)
        for gpu, (row, process, handle, command) in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            output = generations / row["model_id"] / "output_42.json"
            if returncode or not complete_generation(output):
                failures.append({"model_id": row["model_id"], "returncode": returncode,
                                 "command": command})
            del running[gpu]; available.append(gpu); available.sort()
    if failures:
        atomic_json(args.work / "status.json", {"status": "failed", "stage": "validation_decode",
                    "failures": failures, "spent_sealed_split_touched": False})
        raise RuntimeError(json.dumps(failures, indent=2))

    gate_dir = args.work / "stability_gates"; gate_dir.mkdir(parents=True, exist_ok=True)
    all_ids = ["base", *[row["model_id"] for row in models]]
    gates = {"detector": "corrected_nonempty_paired_span_v1", "models": {}}
    for model_id in all_ids:
        output = gate_dir / f"{model_id}.json"
        command = [
            args.python, str(args.project / "scripts/revision/flagship/stability_gate_corrected.py"),
            "--base", str(base_output), "--candidate", str(generations / model_id / "output_42.json"),
            "--output", str(output), "--min-length-ratio", "0.33", "--max-length-ratio", "2.0",
            "--max-repeat-run", "20", "--expected-records", "128",
        ]
        result = subprocess.run(command, cwd=args.project, capture_output=True, text=True)
        (logs / f"gate_{model_id}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        payload = json.loads(output.read_text()) if output.is_file() else {"passed": False}
        gates["models"][model_id] = {"passed": payload.get("passed") is True,
                                      "returncode": result.returncode, "artifact": str(output)}
    gates["eligible_models"] = [model for model in all_ids if gates["models"][model]["passed"]]
    gates["failed_models"] = [model for model in all_ids if not gates["models"][model]["passed"]]
    gates["spent_sealed_split_touched"] = False
    atomic_json(gate_dir / "summary.json", gates)
    if "base" not in gates["eligible_models"]:
        raise RuntimeError("base failed validation stability gate")
    atomic_json(args.work / "status.json", {
        "status": "completed", "stage": "validation_decode_and_stability_gate",
        "eligible_models": gates["eligible_models"], "failed_models": gates["failed_models"],
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "spent_sealed_split_touched": False,
    })


if __name__ == "__main__":
    main()
