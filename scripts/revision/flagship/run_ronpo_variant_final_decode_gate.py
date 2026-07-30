#!/usr/bin/env python3
"""Decode the locked finalist on the existing fresh test and apply S3 fail-closed."""

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


def complete(path: Path, expected: int = 1024) -> bool:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(rows, list) and len(rows) == expected and all(
        str(row.get("prompt", "")).strip() and str(row.get("generated_text", "")).strip() for row in rows)


def gpu_snapshot() -> dict:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu",
                          "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout.splitlines()
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                                "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout.splitlines()
    return {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"), "gpus": gpu,
            "compute_processes": [row for row in processes if row.strip()]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--flagship-root", type=Path, required=True)
    parser.add_argument("--fair-run", type=Path, required=True)
    parser.add_argument("--final-lock", type=Path, required=True)
    parser.add_argument("--fresh-prompts", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.final_lock.read_text(encoding="utf-8"))
    if lock.get("status") != "FINAL_VARIANT_SET_LOCKED_BEFORE_FRESH_PANEL":
        raise RuntimeError("final variant set is not locked")
    if lock.get("variant_panel_judgments_seen_before_lock") is not False:
        raise RuntimeError("variant panel leakage before final lock")
    variant = lock["selected_variant"]
    model_id = variant["model_id"]
    samples = []
    for index in range(3):
        sample = gpu_snapshot(); samples.append(sample)
        if sample["compute_processes"]:
            raise RuntimeError(f"GPU process detected before final decode: {sample['compute_processes']}")
        if index < 2: time.sleep(2)
    atomic_json(args.work / "prelaunch_gpu_samples.json", {"status": "verified_idle", "samples": samples,
                "authorized_gpu_ids": [0, 1, 2, 3], "spent_sealed_split_touched": False})
    generations = args.work / "generations"; logs = args.work / "logs"; logs.mkdir(parents=True, exist_ok=True)
    old_base = args.fair_run / "fresh_test/generations/base/output_42.json"
    if not complete(old_base):
        raise RuntimeError("frozen fresh-test base generation is unavailable")
    base = generations / "base/output_42.json"; base.parent.mkdir(parents=True, exist_ok=True)
    if not complete(base):
        shutil.copy2(old_base, base)
        metadata = old_base.parent / "decode_metadata.json"
        if metadata.is_file(): shutil.copy2(metadata, base.parent / "decode_metadata.json")
    candidate = generations / model_id / "output_42.json"
    if not complete(candidate):
        output_dir = candidate.parent; output_dir.mkdir(parents=True, exist_ok=True)
        command = [args.python, str(args.project / "scripts/revision/flagship/decode_vllm_non_thinking.py"),
                   "--data-dir", str(args.fresh_prompts), "--model", variant["model_path"],
                   "--output-dir", str(output_dir), "--seed", "42", "--temperature", "0.7",
                   "--top-p", "0.9", "--max-new-tokens", "2048", "--max-prompts", "1024",
                   "--gpu-memory-utilization", "0.80"]
        environment = os.environ.copy(); environment.update({
            "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(args.project),
            "HF_HOME": str(args.flagship_root / "cache/huggingface"),
            "HF_HUB_CACHE": str(args.flagship_root / "cache/huggingface/hub"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false", "TORCH_CUDNN_SDPA_ENABLED": "0"})
        with (logs / f"decode_{model_id}.log").open("a", encoding="utf-8") as handle:
            subprocess.run(command, cwd=args.project, env=environment, stdout=handle,
                           stderr=subprocess.STDOUT, check=True)
    if not complete(candidate):
        raise RuntimeError("locked finalist generation is incomplete")
    gate_dir = args.work / "stability_gates"; gate_dir.mkdir(parents=True, exist_ok=True)
    gates = {"detector": "corrected_nonempty_paired_span_v1", "models": {}}
    for name, path in [("base", base), (model_id, candidate)]:
        output = gate_dir / f"{name}.json"
        command = [args.python, str(args.project / "scripts/revision/flagship/stability_gate_corrected.py"),
                   "--base", str(base), "--candidate", str(path), "--output", str(output),
                   "--min-length-ratio", "0.33", "--max-length-ratio", "2.0",
                   "--max-repeat-run", "20", "--expected-records", "1024"]
        result = subprocess.run(command, cwd=args.project, capture_output=True, text=True)
        (logs / f"gate_{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        payload = json.loads(output.read_text()) if output.is_file() else {"passed": False}
        gates["models"][name] = {"passed": payload.get("passed") is True,
                                  "returncode": result.returncode, "artifact": str(output)}
    gates["eligible_models"] = [name for name, value in gates["models"].items() if value["passed"]]
    gates["failed_models"] = [name for name, value in gates["models"].items() if not value["passed"]]
    gates["spent_sealed_split_touched"] = False
    atomic_json(gate_dir / "summary.json", gates)
    atomic_json(args.work / "status.json", {"status": "completed" if model_id in gates["eligible_models"] else "failed",
                "stage": "finalist_fresh_decode_and_gate", "selected_model_id": model_id,
                "eligible_models": gates["eligible_models"], "failed_models": gates["failed_models"],
                "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "spent_sealed_split_touched": False})
    if model_id not in gates["eligible_models"]:
        raise RuntimeError("locked finalist failed the fresh-test S3 gate")


if __name__ == "__main__":
    main()
