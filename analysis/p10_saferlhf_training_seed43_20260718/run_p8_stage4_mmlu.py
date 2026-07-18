#!/usr/bin/env python3
"""Run a pre-locked 5-shot MMLU cohort for all displayed Stage-4 models."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from run_p8_stage4_arc_challenge import MODELS, idle


def lock(output: Path) -> None:
    path = output / "capability_lock.json"
    if path.exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "locked_before_evaluation",
        "scope": "Stage-4 appendix cohort capability measurement; not a selection criterion.",
        "task": "mmlu", "lm_eval_version": "0.4.12", "num_fewshot": 5,
        "apply_chat_template": True, "enable_thinking": False, "seed": 42,
        "requested_models": list(MODELS), "model_paths": MODELS,
        "spent_sealed_split_touched": False,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output / "capability_lock.sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "  capability_lock.json\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--stage4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--write-lock-only", action="store_true")
    args = parser.parse_args()
    lock(args.output)
    locked = json.loads((args.output / "capability_lock.json").read_text(encoding="utf-8"))["requested_models"]
    if any(name not in locked for name in args.models):
        raise SystemExit("worker model is absent from the immutable cohort lock")
    if args.write_lock_only:
        return
    lm_python = "/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1/flagship_20260712/venv_lm_eval/bin/python"
    clean_site = "/NHNHOME/AIPR/sjkim/venv_clean/lib/python3.12/site-packages"
    for name in args.models:
        work = args.output / name
        result = work / "result.json"
        if result.exists():
            continue
        if not idle(args.gpu):
            raise SystemExit(f"GPU {args.gpu} was not idle in three samples")
        model = MODELS[name]
        if not model.startswith("/"):
            model = str(args.stage4 / model)
        work.mkdir(parents=True, exist_ok=True)
        log = args.output / "logs" / f"{name}_gpu{args.gpu}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": args.gpu, "PYTHONPATH": clean_site + ":" + str(args.project),
                    "TOKENIZERS_PARALLELISM": "false", "VLLM_PORT": str(59000 + int(args.gpu) * 10)})
        command = [lm_python, "-m", "lm_eval", "run", "--model", "vllm",
                   "--model_args", f"pretrained={model},dtype=bfloat16,gpu_memory_utilization=0.55,max_model_len=4096,enable_thinking=False",
                   "--tasks", "mmlu", "--num_fewshot", "5", "--batch_size", "auto", "--apply_chat_template",
                   "--seed", "42,42,42,42", "--output_path", str(work)]
        with log.open("a", encoding="utf-8") as handle:
            handle.write(" ".join(command) + "\n"); handle.flush()
            subprocess.run(command, cwd=args.project, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        candidates = sorted(work.glob("**/results*.json"))
        if not candidates:
            raise RuntimeError(f"{name}: no lm-eval result JSON")
        source = candidates[-1]
        group = json.loads(source.read_text(encoding="utf-8")).get("groups", {}).get("mmlu", {})
        value = group.get("acc,none")
        if not isinstance(value, (float, int)):
            raise RuntimeError(f"{name}: no MMLU aggregate accuracy")
        result.write_text(json.dumps({"status": "completed", "model": name, "score": float(value),
                                      "metric": "mmlu acc,none", "source": str(source)}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
