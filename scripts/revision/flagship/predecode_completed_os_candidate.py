#!/usr/bin/env python3
"""Use one newly idle authorized GPU to predecode a completed OS candidate for the frozen gate."""

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


def complete(path: Path) -> bool:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(rows, list) and len(rows) == 647 and all(
        str(row.get("prompt", "")).strip() and str(row.get("generated_text", "")).strip() for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fixed647", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    args = parser.parse_args()
    if args.gpu not in {0, 1, 2, 3}:
        raise RuntimeError("unauthorized GPU")
    status = json.loads((args.candidate_root / "training_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "completed" or status.get("measured_step") != 900:
        raise RuntimeError("candidate training is not complete")
    samples = []
    for index in range(3):
        sample = subprocess.run(["nvidia-smi", "-i", str(args.gpu),
                                 "--query-compute-apps=pid,process_name,used_memory",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
        rows = [row for row in sample.stdout.splitlines() if row.strip()]
        samples.append({"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "gpu": args.gpu, "compute_processes": rows})
        if rows:
            raise RuntimeError(f"target GPU is not idle: {rows}")
        if index < 2:
            time.sleep(2)
    work = args.output_root
    atomic_json(work / f"predecode_gpu{args.gpu}_{args.candidate_id}_samples.json",
                {"status": "verified_idle", "samples": samples, "spent_sealed_split_touched": False})
    entries = []
    for step in range(100, 901, 100):
        model_path = args.candidate_root if step == 900 else args.candidate_root / f"checkpoint-{step}"
        if not model_path.is_dir():
            raise RuntimeError(f"missing checkpoint {step}")
        entries.append((f"{args.candidate_id}__s{step}", model_path))
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONPATH": str(args.project),
                "HF_HOME": str(args.hf_cache.parent), "HF_HUB_CACHE": str(args.hf_cache),
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false", "TORCH_CUDNN_SDPA_ENABLED": "0",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
    completed = []
    for model_id, model_path in entries:
        output = work / "generations_4096" / model_id
        output.mkdir(parents=True, exist_ok=True)
        if complete(output / "output_42.json"):
            completed.append(model_id); continue
        command = [args.python, "-u", str(args.project / "scripts/revision/flagship/decode_vllm_non_thinking.py"),
                   "--data-dir", str(args.fixed647), "--model", str(model_path), "--output-dir", str(output),
                   "--seed", "42", "--temperature", "0.7", "--top-p", "0.9",
                   "--max-new-tokens", "4096", "--max-model-len", "8192",
                   "--gpu-memory-utilization", "0.88"]
        log = work / "logs" / f"decode_{model_id}.log"; log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            subprocess.run(command, cwd=args.project, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        if not complete(output / "output_42.json"):
            raise RuntimeError(f"incomplete predecode: {model_id}")
        completed.append(model_id)
        atomic_json(work / f"predecode_gpu{args.gpu}_{args.candidate_id}_status.json",
                    {"status": "running", "completed": completed, "total": len(entries),
                     "reward_scores_consulted": False, "spent_sealed_split_touched": False})
    atomic_json(work / f"predecode_gpu{args.gpu}_{args.candidate_id}_status.json",
                {"status": "completed", "completed": completed, "total": len(entries),
                 "reward_scores_consulted": False, "spent_sealed_split_touched": False})


if __name__ == "__main__":
    main()
