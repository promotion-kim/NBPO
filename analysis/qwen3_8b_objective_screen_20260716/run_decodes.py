#!/usr/bin/env python3
"""Run the locked inference-only decode queue over four authorized GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
from pathlib import Path


PROBES = {"weak_small", "verbose", "terse", "less_aligned"}
TRAINED = [
    "ronpo_full_expect", "ronpo_k_only", "ipo", "simpo", "sppo_avg", "inpo_avg",
    "ht_mnpo_helpfulness", "ht_mnpo_safety", "ht_mnpo_conciseness",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_task(task: dict, gpu: int, repo: Path, root: Path, python: str) -> dict:
    name, split, model = task["name"], task["split"], task["model"]
    raw_dir = root / "generations_raw" / split / name
    restored = root / "generations" / split / name / "output_42.json"
    log = root / "logs" / f"decode_{split}_{name}.log"
    raw_dir.mkdir(parents=True, exist_ok=True)
    restored.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    probe_input = root / "probe_inputs" / split / f"{name if name in PROBES else 'base'}.jsonl"
    data_input = probe_input if name in PROBES else root / "prompt_manifests" / f"{split}.jsonl"
    command = [
        python, str(repo / "scripts/revision/flagship/decode_vllm_non_thinking.py"),
        "--data-dir", str(data_input), "--model", model,
        "--output-dir", str(raw_dir), "--seed", "42", "--temperature", "0.7",
        "--top-p", "0.9", "--max-new-tokens", "2048", "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.80",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["VLLM_USE_V1"] = "0"
    env["TORCH_CUDNN_SDPA_ENABLED"] = "0"
    with log.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND=" + json.dumps(command) + "\n")
        subprocess.run(command, cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        restore_input = probe_input if name in PROBES else root / "probe_inputs" / split / "base.jsonl"
        restore = [
            python, str(repo / "analysis/qwen3_8b_objective_screen_20260716/restore_probe_prompts.py"),
            "--probe-input", str(restore_input), "--decode-output", str(raw_dir / "output_42.json"),
            "--output", str(restored),
        ]
        subprocess.run(restore, cwd=repo, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return {"name": name, "split": split, "gpu": gpu, "output": str(restored), "sha256": sha(restored), "status": "completed"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--base-snapshot", required=True)
    parser.add_argument("--weak-snapshot", required=True)
    args = parser.parse_args()
    lock = json.loads((args.root / "screen_lock.json").read_text(encoding="utf-8"))
    snapshots = {row["name"]: row["snapshot"] for row in lock["policies"]["trained_pool"]}
    snapshots["base"] = args.base_snapshot
    snapshots["weak_small"] = args.weak_snapshot
    snapshots["verbose"] = args.base_snapshot
    snapshots["terse"] = args.base_snapshot
    snapshots["less_aligned"] = args.base_snapshot
    tasks = []
    for name in ["weak_small", "verbose", "terse", "less_aligned"]:
        tasks.append({"name": name, "split": "general", "model": snapshots[name]})
    for name in ["base", "weak_small", "verbose", "terse", "less_aligned"] + TRAINED:
        tasks.append({"name": name, "split": "conflict_curated", "model": snapshots[name]})

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        future_to_gpu = {}
        pending = iter(tasks)
        for gpu in range(4):
            task = next(pending, None)
            if task:
                future_to_gpu[pool.submit(run_task, task, gpu, args.repo, args.root, args.python)] = gpu
        while future_to_gpu:
            done, _ = concurrent.futures.wait(future_to_gpu, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                gpu = future_to_gpu.pop(future)
                results.append(future.result())
                task = next(pending, None)
                if task:
                    future_to_gpu[pool.submit(run_task, task, gpu, args.repo, args.root, args.python)] = gpu
                (args.root / "decode_status.json").write_text(json.dumps({"completed": results, "total": len(tasks)}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed": len(results), "total": len(tasks)}, indent=2))


if __name__ == "__main__":
    main()
