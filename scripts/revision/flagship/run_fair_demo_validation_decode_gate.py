#!/usr/bin/env python3
"""Decode all frozen sweep candidates on validation and apply the unchanged stability gate."""

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
        str(row.get("prompt", "")).strip() and str(row.get("generated_text", "")).strip() for row in rows
    )


def model_complete(path: Path) -> bool:
    return (path / "model.safetensors").is_file() or (path / "model.safetensors.index.json").is_file()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--flagship-root", type=Path, required=True)
    parser.add_argument("--fair-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    args = parser.parse_args()
    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    frozen_candidates = [row["id"] for row in grid["candidates"]]
    candidate_root = args.fair_root / "sweep/candidates"
    terminal_path = args.run_dir / "training_terminal.json"
    if not terminal_path.is_file():
        raise RuntimeError("terminal training audit is missing")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal_failed = set(terminal.get("terminal_failed_candidates", []))
    if not terminal_failed.issubset(frozen_candidates):
        raise RuntimeError("terminal training audit names an unknown candidate")
    candidates = []
    unresolved = []
    for candidate in frozen_candidates:
        model = candidate_root / candidate
        status_path = model / "training_status.json"
        status = json.loads(status_path.read_text()) if status_path.is_file() else {}
        if status.get("status") == "completed" and model_complete(model):
            candidates.append(candidate)
        elif candidate not in terminal_failed:
            unresolved.append(candidate)
    if unresolved:
        raise RuntimeError(f"validation cannot start with unresolved frozen candidates: {unresolved}")
    if not candidates:
        raise RuntimeError("no frozen training candidate completed")
    work = args.run_dir / "validation"
    generations = work / "generations"
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = work / "status.json"
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

    pending = [candidate for candidate in candidates
               if not complete_generation(generations / candidate / "output_42.json")]
    running = {}
    available = [0, 1, 2, 3]
    failures = []
    while pending or running:
        while pending and available:
            gpu = available.pop(0)
            candidate = pending.pop(0)
            output_dir = generations / candidate
            output_dir.mkdir(parents=True, exist_ok=True)
            command = [
                args.python, str(args.project / "scripts/revision/flagship/decode_vllm_non_thinking.py"),
                "--data-dir", str(args.flagship_root / "data/pool_validation.jsonl"),
                "--model", str(candidate_root / candidate), "--output-dir", str(output_dir),
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
            handle = (logs / f"decode_{candidate}.log").open("a", encoding="utf-8")
            process = subprocess.Popen(command, cwd=args.project, env=environment,
                                       stdout=handle, stderr=subprocess.STDOUT)
            running[gpu] = (candidate, process, handle, command)
        atomic_json(status_path, {
            "status": "running", "stage": "validation_decode", "pending": pending,
            "running": [{"gpu": gpu, "candidate": value[0], "pid": value[1].pid}
                        for gpu, value in running.items()],
            "failures": failures, "spent_sealed_split_touched": False,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        })
        if not running:
            continue
        time.sleep(10)
        for gpu, (candidate, process, handle, command) in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            if returncode or not complete_generation(generations / candidate / "output_42.json"):
                failures.append({"candidate": candidate, "returncode": returncode, "command": command})
            del running[gpu]
            available.append(gpu); available.sort()
    if failures:
        atomic_json(status_path, {"status": "failed", "stage": "validation_decode",
                                  "failures": failures, "spent_sealed_split_touched": False})
        raise RuntimeError(json.dumps(failures, indent=2))

    gate_dir = work / "stability_gates"
    gate_dir.mkdir(parents=True, exist_ok=True)
    gates = {"detector": "corrected_nonempty_paired_span_v1", "models": {}}
    for model in ["base", *candidates]:
        output = gate_dir / f"{model}.json"
        command = [
            args.python, str(args.project / "scripts/revision/flagship/stability_gate_corrected.py"),
            "--base", str(base_output), "--candidate", str(generations / model / "output_42.json"),
            "--output", str(output), "--min-length-ratio", "0.33", "--max-length-ratio", "2.0",
            "--max-repeat-run", "20", "--expected-records", "128",
        ]
        result = subprocess.run(command, cwd=args.project, capture_output=True, text=True)
        (logs / f"gate_{model}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        payload = json.loads(output.read_text()) if output.is_file() else {"passed": False}
        gates["models"][model] = {"passed": payload.get("passed") is True,
                                     "returncode": result.returncode, "artifact": str(output)}
    gates["eligible_models"] = [model for model in ["base", *candidates] if gates["models"][model]["passed"]]
    gates["failed_models"] = [model for model in ["base", *candidates] if not gates["models"][model]["passed"]]
    gates["spent_sealed_split_touched"] = False
    atomic_json(gate_dir / "summary.json", gates)
    if "base" not in gates["eligible_models"]:
        raise RuntimeError("base failed validation stability gate")
    atomic_json(status_path, {
        "status": "completed", "stage": "validation_decode_and_stability_gate",
        "eligible_models": gates["eligible_models"], "failed_models": gates["failed_models"],
        "terminal_training_failures": sorted(terminal_failed),
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "spent_sealed_split_touched": False,
    })


if __name__ == "__main__":
    main()
