#!/usr/bin/env python3
"""Run the preregistered P8 Stage-4 IFEval cohort on idle B200 GPUs.

This is an evaluation-only, public benchmark measurement.  It contains every
method displayed in the Stage-4 appendix table, so it cannot select a method
after observing IFEval.  One worker uses exactly one local GPU at a time.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


MODELS = [
    ("base", "/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1/base_objective_screen/hf_ipv4/llama31"),
    ("ronpo_os", "stage4/ronpo_os_stage4/train/full"),
    ("ronpo_os_s43", "stage4/ronpo_os_stage4_s43/train/full"),
    ("ipo_s43", "stage4/ipo_stage4_s43/train/full"),
    ("ipo", "stage4/ipo_stage4/train/full"),
    ("dpo", "stage4/dpo_stage4/train/full"),
    ("simpo", "stage4/simpo_stage4/train/full"),
    ("inpo_avg", "stage4/inpo_avg_stage4/train/full"),
    ("ht_mnpo_helpfulness", "stage4/ht_mnpo_helpfulness_stage4/train/full"),
    ("ht_mnpo_harmless", "stage4/ht_mnpo_harmless_stage4/train/full"),
    ("sppo_avg", "stage4/sppo_avg_stage4/train/full"),
]


def idle(gpu: str) -> bool:
    for _ in range(3):
        text = subprocess.check_output(
            ["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if text:
            return False
        time.sleep(3)
    return True


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--stage4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--write-lock-only", action="store_true")
    args = parser.parse_args()
    mapping = dict(MODELS)
    unknown = [name for name in args.models if name not in mapping]
    if unknown:
        raise SystemExit(f"unknown locked model(s): {unknown}")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / "ifeval_lock.json"
    if not lock.exists():
        write_json(lock, {
            "status": "locked_before_run",
            "task": "ifeval",
            "lm_eval_version": "0.4.12",
            "num_fewshot": 0,
            "seed": 42,
            "apply_chat_template": True,
            "enable_thinking": False,
            "all_stage4_appendix_models": [name for name, _ in MODELS],
            "requested_models": args.models,
            "scope": "Evaluation-only capability measurement for all Stage-4 appendix methods; no model selection.",
            "spent_sealed_split_touched": False,
        })
    if args.write_lock_only:
        return
    lm_python = "/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1/flagship_20260712/venv_lm_eval/bin/python"
    clean_site = "/NHNHOME/AIPR/sjkim/venv_clean/lib/python3.12/site-packages"
    for name in args.models:
        model = mapping[name]
        if not model.startswith("/"):
            model = str(args.stage4 / model)
        work = args.output / name
        result = work / "results.json"
        if result.exists():
            continue
        if not idle(args.gpu):
            raise SystemExit(f"GPU {args.gpu} was not idle in all three pre-launch samples")
        work.mkdir(parents=True, exist_ok=True)
        log = args.output / "logs" / f"{name}_gpu{args.gpu}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "PYTHONPATH": clean_site + ":" + str(args.project),
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_PORT": str(57000 + int(args.gpu) * 10),
        })
        command = [
            lm_python, "-m", "lm_eval", "run", "--model", "vllm",
            "--model_args", f"pretrained={model},dtype=bfloat16,gpu_memory_utilization=0.55,max_model_len=4096,enable_thinking=False",
            "--tasks", "ifeval", "--num_fewshot", "0", "--batch_size", "auto",
            "--apply_chat_template", "--seed", "42,42,42,42", "--output_path", str(work),
        ]
        with log.open("a", encoding="utf-8") as handle:
            handle.write(" ".join(command) + "\n")
            handle.flush()
            subprocess.run(command, cwd=args.project, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        candidates = sorted(work.glob("**/results*.json"))
        if not candidates:
            raise RuntimeError(f"{name}: lm-eval returned success but wrote no results JSON")
        source = candidates[-1]
        payload = json.loads(source.read_text(encoding="utf-8"))
        entry = payload.get("results", {}).get("ifeval", {})
        values = [
            float(value) for key, value in entry.items()
            if key.split(",", 1)[0] == "prompt_level_strict_acc"
            and isinstance(value, (int, float))
        ]
        if len(values) != 1:
            raise RuntimeError(f"{name}: missing or ambiguous IFEval prompt_level_strict_acc in {source}")
        score = values[0]
        write_json(result, {"model": name, "score": float(score), "source": str(source), "status": "completed"})
    write_json(args.output / f"worker_gpu{args.gpu}_completed.json", {"status": "completed", "gpu": args.gpu, "models": args.models})


if __name__ == "__main__":
    main()
