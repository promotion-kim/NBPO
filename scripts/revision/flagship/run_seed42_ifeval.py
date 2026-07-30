#!/usr/bin/env python3
"""Run the frozen seed-42 IFEval comparison on two explicitly free GPUs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


METHODS = (
    "base", "ronpo_full_expect", "ronpo_k_only", "dpo", "ipo", "simpo", "sppo_avg",
    "inpo_avg", "ht_mnpo_helpfulness", "ht_mnpo_safety",
    "ht_mnpo_conciseness",
)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def ledger_models(root: Path, base_revision: str) -> dict[str, dict]:
    selected = {
        "base": {
            "method": "base", "seed": None, "repo_id": "Qwen/Qwen3-8B",
            "revision": base_revision,
        }
    }
    ledger = root / "hf_uploads.jsonl"
    if not ledger.exists():
        return selected
    for line in ledger.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        method = str(row.get("method"))
        if method not in METHODS or str(row.get("seed")) != "42" or row.get("verified") is not True:
            continue
        commit_url = str(row.get("upload_commit", ""))
        revision = commit_url.rsplit("/", 1)[-1] if "/commit/" in commit_url else None
        selected[method] = {
            "method": method, "seed": 42, "repo_id": row["repo_id"],
            "revision": revision, "upload_commit": commit_url,
        }
    return selected


def result_complete(output: Path) -> bool:
    for path in output.rglob("*.json") if output.exists() else ():
        if path.name.endswith("status.json"):
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        entry = payload.get("results", {}).get("ifeval") if isinstance(payload, dict) else None
        if not isinstance(entry, dict):
            continue
        values = [
            value for key, value in entry.items()
            if key.split(",", 1)[0] == "prompt_level_strict_acc"
            and isinstance(value, (int, float))
        ]
        if len(values) == 1:
            return True
    return False


def run_id(method: str) -> str:
    return hashlib.sha256(f"aaai27-p2-ifeval-seed42-v1|{method}".encode()).hexdigest()[:12]


def command(args: argparse.Namespace, model: dict, output: Path) -> list[str]:
    model_args = [
        f"pretrained={model['repo_id']}", "dtype=bfloat16",
        "gpu_memory_utilization=0.88", "max_model_len=32768",
        "enable_thinking=False",
    ]
    if model.get("revision"):
        model_args.append(f"revision={model['revision']}")
    method = model["method"]
    identifier = run_id(method)
    return [
        args.python, "-m", "lm_eval", "run", "--model", "vllm",
        "--model_args", ",".join(model_args), "--tasks", "ifeval",
        "--num_fewshot", "0", "--batch_size", "auto", "--max_batch_size", "256",
        "--apply_chat_template", "--seed", "42,42,42,42",
        "--cache_requests", "true", "--show_config", "--output_path", str(output),
        "--wandb_args", "entity=promotion-kim", "project=mnpo", f"id={identifier}",
        f"name=aaai27-p2-ifeval-{method}-s42", "group=flagship_p2_seed42",
        "job_type=lm_eval", "resume=allow",
        "--wandb_config_args", "flagship_stage=P2_early_IFEval",
        f"model_name={method}", "task_group=ifeval", "num_fewshot=0",
        "enable_thinking=False",
    ]


def write_provenance(args: argparse.Namespace, models: dict[str, dict]) -> None:
    task_root = Path(importlib.metadata.distribution("lm_eval").locate_file("lm_eval/tasks/ifeval"))
    task_files = []
    for path in sorted(task_root.rglob("*")) if task_root.exists() else ():
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}:
            task_files.append({
                "path": str(path.relative_to(task_root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    atomic_json(args.work / "provenance.json", {
        "selection": "seed 42 only; KTO excluded; RONPO k-only included by the 2026-07-14 user scope amendment",
        "models": [models[key] for key in METHODS if key in models],
        "missing": [key for key in METHODS if key not in models],
        "lm_eval_version": importlib.metadata.version("lm_eval"),
        "vllm_version": importlib.metadata.version("vllm"),
        "task": "ifeval", "num_fewshot": 0, "apply_chat_template": True,
        "enable_thinking": False, "seed": [42, 42, 42, 42],
        "task_source_hashes": task_files,
        "p1_sealed_test_opened": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--stop-at", required=True)
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / "logs").mkdir(exist_ok=True)
    stop_at = datetime.fromisoformat(args.stop_at).timestamp()
    pending = list(METHODS)
    running: dict[int, tuple[str, subprocess.Popen, object, int]] = {}
    completed = []
    failed = []
    attempts: dict[str, int] = {}
    status_path = args.work / "status.json"

    while (pending or running) and time.time() < stop_at:
        for gpu, (method, process, handle, attempt) in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            output = args.work / "raw" / method
            if returncode == 0 and result_complete(output):
                completed.append({"method": method, "attempt": attempt})
            elif attempt < 3:
                pending.insert(0, method)
            else:
                failed.append({"method": method, "attempt": attempt, "returncode": returncode})
            del running[gpu]

        models = ledger_models(args.root, args.base_revision)
        write_provenance(args, models)
        free = [gpu for gpu in (1, 3) if gpu not in running]
        for gpu in list(free):
            ready = next((method for method in pending if method in models), None)
            if ready is None:
                break
            pending.remove(ready)
            output = args.work / "raw" / ready
            if result_complete(output):
                completed.append({"method": ready, "attempt": attempts.get(ready, 0), "note": "pre-existing"})
                continue
            attempts[ready] = attempts.get(ready, 0) + 1
            attempt = attempts[ready]
            log = args.work / "logs" / f"{ready}_a{attempt}.log"
            handle = log.open("a", encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "HF_HOME": str(args.root / "cache/huggingface"),
                "HF_HUB_CACHE": str(args.root / "cache/huggingface/hub"),
                "HF_DATASETS_CACHE": str(args.root / "cache/huggingface/datasets"),
                "WANDB_MODE": "online", "WANDB_ENTITY": "promotion-kim",
                "WANDB_PROJECT": "mnpo", "TOKENIZERS_PARALLELISM": "false",
                "VLLM_HOST_IP": "127.0.0.1", "VLLM_PORT": str(65000 + gpu * 20),
                "VLLM_DP_MASTER_PORT": str(65001 + gpu * 20),
                "MASTER_PORT": str(65002 + gpu * 20),
                "TORCH_CUDNN_SDPA_ENABLED": "0",
            })
            process = subprocess.Popen(
                command(args, models[ready], output), env=env,
                stdout=handle, stderr=subprocess.STDOUT,
            )
            running[gpu] = (ready, process, handle, attempt)

        atomic_json(status_path, {
            "status": "running", "pending": pending,
            "running": [
                {"gpu": gpu, "method": method, "pid": process.pid, "attempt": attempt}
                for gpu, (method, process, _, attempt) in running.items()
            ],
            "completed": completed, "failed": failed,
            "models_available": [method for method in METHODS if method in models],
            "models_waiting_for_s3_hf": [method for method in METHODS if method not in models],
            "p1_sealed_test_opened": False,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        })
        time.sleep(args.poll_seconds)

    final = "completed" if not pending and not running and not failed else "completed_with_failures" if not pending and not running else "deadline_reached"
    current = json.loads(status_path.read_text()) if status_path.exists() else {}
    atomic_json(status_path, current | {
        "status": final,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })


if __name__ == "__main__":
    main()
