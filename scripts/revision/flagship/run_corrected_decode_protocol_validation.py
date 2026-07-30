#!/usr/bin/env python3
"""Validate the post-failure common decode guard before opening a new holdout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

from run_seed42_sealed_reward_eval import (
    METHODS,
    atomic_json,
    complete,
    frozen_models_tsv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--decode-python", required=True)
    parser.add_argument("--gate-python", required=True)
    parser.add_argument("--models-tsv", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--expected-prompts", type=int, required=True)
    parser.add_argument("--base-revision", required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / "logs").mkdir(exist_ok=True)
    models = frozen_models_tsv(args.models_tsv, args.base_revision)
    missing = [method for method in METHODS if method not in models]
    if missing:
        raise RuntimeError(f"frozen model manifest is incomplete: {missing}")

    atomic_json(args.work / "protocol_candidate.json", {
        "schema_version": 1,
        "status": "validation_only_not_yet_frozen_for_holdout",
        "decode": {
            "backend": "vllm", "version": "0.24.0", "seed": 42,
            "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 2048,
            "dtype": "bfloat16", "enable_thinking": False,
            "bad_words": ["<think>", "</think>"],
            "repetition_detection": {"max_pattern_size": 4, "min_pattern_size": 1, "min_count": 20},
        },
        "stability_gate": {"records": args.expected_prompts, "empty": 0, "think_leak": 0,
                           "length_ratio": [0.33, 2.0], "max_repeat_run": 20},
        "models": models,
        "created_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
    })

    pending = list(METHODS)
    running: dict[int, tuple[str, subprocess.Popen, object]] = {}
    attempts: dict[str, int] = {}
    failures: list[dict] = []
    while pending or running:
        for gpu, (method, process, handle) in list(running.items()):
            rc = process.poll()
            if rc is None:
                continue
            handle.close()
            output = args.work / "generations" / method / "output_42.json"
            if rc == 0 and complete(output, args.expected_prompts):
                pass
            elif attempts[method] < 2:
                pending.insert(0, method)
            else:
                failures.append({"method": method, "attempt": attempts[method], "returncode": rc})
            del running[gpu]
        for gpu in (0, 1, 2, 3):
            if gpu in running or not pending:
                continue
            method = pending.pop(0)
            output_dir = args.work / "generations" / method
            output = output_dir / "output_42.json"
            if complete(output, args.expected_prompts):
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            attempts[method] = attempts.get(method, 0) + 1
            handle = (args.work / "logs" / f"decode_{method}_a{attempts[method]}.log").open("a")
            model = models[method]
            command = [
                args.decode_python,
                str(args.project / "scripts/revision/flagship/decode_vllm_non_thinking.py"),
                "--data-dir", str(args.data), "--model", model["repo_id"],
                "--revision", model["revision"], "--output-dir", str(output_dir),
                "--seed", "42", "--temperature", "0.7", "--top-p", "0.9",
                "--max-new-tokens", "2048", "--forbid-thinking-tags",
                "--repetition-detection-max-pattern-size", "4",
                "--repetition-detection-min-pattern-size", "1",
                "--repetition-detection-min-count", "20",
            ]
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(gpu), "TORCH_CUDNN_SDPA_ENABLED": "0",
                "TOKENIZERS_PARALLELISM": "false", "HF_HOME": str(args.root / "cache/huggingface"),
                "HF_HUB_CACHE": str(args.root / "cache/huggingface/hub"),
            })
            process = subprocess.Popen(command, cwd=args.project, env=env,
                                       stdout=handle, stderr=subprocess.STDOUT)
            running[gpu] = (method, process, handle)
        atomic_json(args.work / "status.json", {
            "status": "running", "stage": "corrected_decode_protocol_validation",
            "pending": pending,
            "running": [{"gpu": gpu, "method": item[0], "pid": item[1].pid}
                        for gpu, item in running.items()],
            "failures": failures,
            "confirmatory_holdout_opened": False,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        })
        time.sleep(10)
    if failures:
        atomic_json(args.work / "status.json", {
            "status": "failed", "stage": "corrected_decode_protocol_validation",
            "failures": failures, "confirmatory_holdout_opened": False,
        })
        raise SystemExit(2)

    base = args.work / "generations/base/output_42.json"
    gates: dict[str, dict] = {}
    for method in METHODS:
        output = args.work / "stability_gates" / f"{method}.json"
        output.parent.mkdir(exist_ok=True)
        command = [
            args.gate_python, str(args.project / "scripts/revision/flagship/stability_gate.py"),
            "--base", str(base), "--candidate",
            str(args.work / "generations" / method / "output_42.json"),
            "--output", str(output), "--expected-records", str(args.expected_prompts),
            "--min-length-ratio", "0.33", "--max-length-ratio", "2.0",
            "--max-repeat-run", "20",
        ]
        completed = subprocess.run(command, cwd=args.project, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.STDOUT, check=False)
        gate = json.loads(output.read_text()) if output.exists() else {
            "passed": False, "status": "failed", "returncode": completed.returncode,
        }
        gates[method] = gate
    all_passed = all(gate.get("passed") is True for gate in gates.values())
    atomic_json(args.work / "stability_gates/summary.json", {
        "fail_closed": True, "all_passed": all_passed, "models": gates,
    })
    atomic_json(args.work / "status.json", {
        "status": "completed" if all_passed else "failed",
        "stage": "corrected_decode_protocol_validation_gates",
        "all_passed": all_passed,
        "failed_models": [method for method, gate in gates.items() if gate.get("passed") is not True],
        "confirmatory_holdout_opened": False,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    raise SystemExit(0 if all_passed else 4)


if __name__ == "__main__":
    main()
