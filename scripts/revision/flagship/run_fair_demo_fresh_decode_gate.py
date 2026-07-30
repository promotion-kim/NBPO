#!/usr/bin/env python3
"""Open the fresh prompt-disjoint test once, decode locked models, and gate fail-closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def complete(path: Path, expected: int) -> bool:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(rows, list) and len(rows) == expected and all(
        str(row.get("prompt", "")).strip() and str(row.get("generated_text", "")).strip() for row in rows
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--flagship-root", type=Path, required=True)
    parser.add_argument("--fair-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path)
    parser.add_argument("--fresh-manifest", type=Path, required=True)
    parser.add_argument("--fresh-prompts", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    manifest = json.loads(args.fresh_manifest.read_text(encoding="utf-8"))
    if selection.get("status") != "VALIDATION_SELECTION_LOCKED_BEFORE_FRESH_TEST":
        raise RuntimeError("selection not locked")
    if manifest.get("status") != "FRESH_TEST_PROMPTS_LOCKED_UNOPENED" or manifest.get("fresh_test_opened") is not False:
        raise RuntimeError("fresh manifest not locked and unopened")
    if manifest.get("selection_lock_sha256") is not None:
        if manifest["selection_lock_sha256"] != sha256(args.selection_lock):
            raise RuntimeError("selection lock hash mismatch")
    else:
        if args.execution_lock is None:
            raise RuntimeError("prospectively locked manifest requires --execution-lock")
        execution = json.loads(args.execution_lock.read_text(encoding="utf-8"))
        if execution.get("status") != "FRESH_EXECUTION_LOCKED_BEFORE_OPENING":
            raise RuntimeError("fresh execution is not locked")
        if execution.get("fresh_manifest_sha256") != sha256(args.fresh_manifest):
            raise RuntimeError("execution lock fresh manifest hash mismatch")
        if execution.get("selection_lock_sha256") != sha256(args.selection_lock):
            raise RuntimeError("execution lock selection hash mismatch")
    if manifest["prompt_file_sha256"] != sha256(args.fresh_prompts):
        raise RuntimeError("fresh prompt file hash mismatch")
    count = int(manifest["prompt_count"])
    selected = {method: row["candidate_id"] for method, row in selection["selected_by_method"].items()}
    models = {"base": args.base_model}
    for candidate in selected.values():
        models[candidate] = args.fair_root / "sweep/candidates" / candidate
    work = args.run_dir / "fresh_test"
    opened_path = work / "fresh_test_opened.json"
    if not opened_path.is_file():
        atomic_json(opened_path, {
            "status": "OPENED_ONCE_FOR_LOCKED_SELECTION", "opened_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "fresh_manifest_sha256": sha256(args.fresh_manifest), "selection_lock_sha256": sha256(args.selection_lock),
            "execution_lock_sha256": None if args.execution_lock is None else sha256(args.execution_lock),
            "models": sorted(models), "decode": {"seed": 42, "temperature": 0.7, "top_p": 0.9,
                                                     "max_new_tokens": 2048, "bf16": True, "enable_thinking": False},
            "spent_sealed_split_touched": False,
        })
    else:
        opened = json.loads(opened_path.read_text(encoding="utf-8"))
        if opened.get("fresh_manifest_sha256") != sha256(args.fresh_manifest) or sorted(opened.get("models", [])) != sorted(models):
            raise RuntimeError("existing fresh opening differs from locked inputs")
    generations = work / "generations"
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = work / "status.json"
    pending = [model for model in models if not complete(generations / model / "output_42.json", count)]
    available = [0, 1, 2, 3]
    running = {}
    failures = []
    while pending or running:
        while pending and available:
            gpu = available.pop(0)
            model = pending.pop(0)
            output_dir = generations / model
            output_dir.mkdir(parents=True, exist_ok=True)
            command = [
                args.python, str(args.project / "scripts/revision/flagship/decode_vllm_non_thinking.py"),
                "--data-dir", str(args.fresh_prompts), "--model", str(models[model]),
                "--output-dir", str(output_dir), "--seed", "42", "--temperature", "0.7",
                "--top-p", "0.9", "--max-new-tokens", "2048", "--max-prompts", str(count),
                "--gpu-memory-utilization", "0.80",
            ]
            environment = os.environ.copy()
            environment.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(args.project),
                                "HF_HOME": str(args.flagship_root / "cache/huggingface"),
                                "HF_HUB_CACHE": str(args.flagship_root / "cache/huggingface/hub"),
                                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                                "TOKENIZERS_PARALLELISM": "false", "TORCH_CUDNN_SDPA_ENABLED": "0"})
            handle = (logs / f"decode_{model}.log").open("a", encoding="utf-8")
            process = subprocess.Popen(command, cwd=args.project, env=environment,
                                       stdout=handle, stderr=subprocess.STDOUT)
            running[gpu] = (model, process, handle, command)
        atomic_json(status_path, {"status": "running", "stage": "fresh_decode", "pending": pending,
                                  "running": [{"gpu": gpu, "model": row[0], "pid": row[1].pid}
                                              for gpu, row in running.items()],
                                  "failures": failures, "spent_sealed_split_touched": False,
                                  "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
        if not running:
            continue
        time.sleep(10)
        for gpu, (model, process, handle, command) in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            if returncode or not complete(generations / model / "output_42.json", count):
                failures.append({"model": model, "returncode": returncode, "command": command})
            del running[gpu]; available.append(gpu); available.sort()
    if failures:
        atomic_json(status_path, {"status": "failed", "stage": "fresh_decode", "failures": failures,
                                  "spent_sealed_split_touched": False})
        raise RuntimeError(json.dumps(failures, indent=2))
    gate_dir = work / "stability_gates"
    gate_dir.mkdir(parents=True, exist_ok=True)
    gates = {"detector": "corrected_nonempty_paired_span_v1", "models": {}}
    base_output = generations / "base/output_42.json"
    for model in models:
        output = gate_dir / f"{model}.json"
        command = [args.python, str(args.project / "scripts/revision/flagship/stability_gate_corrected.py"),
                   "--base", str(base_output), "--candidate", str(generations / model / "output_42.json"),
                   "--output", str(output), "--min-length-ratio", "0.33", "--max-length-ratio", "2.0",
                   "--max-repeat-run", "20", "--expected-records", str(count)]
        result = subprocess.run(command, cwd=args.project, capture_output=True, text=True)
        (logs / f"gate_{model}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        payload = json.loads(output.read_text()) if output.is_file() else {"passed": False}
        gates["models"][model] = {"passed": payload.get("passed") is True,
                                     "returncode": result.returncode, "artifact": str(output)}
    gates["eligible_models"] = [model for model in models if gates["models"][model]["passed"]]
    gates["failed_models"] = [model for model in models if not gates["models"][model]["passed"]]
    gates["spent_sealed_split_touched"] = False
    atomic_json(gate_dir / "summary.json", gates)
    if "base" not in gates["eligible_models"]:
        raise RuntimeError("base failed fresh stability gate")
    atomic_json(status_path, {"status": "completed", "stage": "fresh_decode_and_stability_gate",
                              "eligible_models": gates["eligible_models"], "failed_models": gates["failed_models"],
                              "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                              "spent_sealed_split_touched": False})


if __name__ == "__main__":
    main()
