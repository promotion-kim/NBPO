#!/usr/bin/env python3
"""Run a locked ARC-Challenge capability cohort for the Stage-4 table.

The cohort is fixed in ``capability_lock.json`` before the first model is
evaluated.  Workers may receive disjoint subsets of that immutable cohort so
that one B200 per worker can run independently.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


MODELS = {
    "base": "/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1/base_objective_screen/hf_ipv4/llama31",
    "ronpo_os": "stage4/ronpo_os_stage4/train/full",
    "ipo": "stage4/ipo_stage4/train/full",
    "dpo": "stage4/dpo_stage4/train/full",
    "simpo": "stage4/simpo_stage4/train/full",
    "inpo_avg": "stage4/inpo_avg_stage4/train/full",
    "ht_mnpo_helpfulness": "stage4/ht_mnpo_helpfulness_stage4/train/full",
    "ht_mnpo_harmless": "stage4/ht_mnpo_harmless_stage4/train/full",
    "sppo_avg": "stage4/sppo_avg_stage4/train/full",
}


def idle(gpu: str) -> bool:
    for _ in range(3):
        pids = subprocess.check_output(
            ["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if pids:
            return False
        time.sleep(3)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--stage4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    lock_path = args.output / "capability_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    locked = lock["requested_models"]
    unknown = [name for name in args.models if name not in locked or name not in MODELS]
    if unknown:
        raise SystemExit(f"model(s) absent from the pre-existing lock: {unknown}")

    lm_python = "/NHNHOME/AIPR/sjkim/revision_qwen3_8b/full_iter1/flagship_20260712/venv_lm_eval/bin/python"
    clean_site = "/NHNHOME/AIPR/sjkim/venv_clean/lib/python3.12/site-packages"
    for name in args.models:
        work = args.output / name
        result = work / "result.json"
        if result.exists():
            continue
        if not idle(args.gpu):
            raise SystemExit(f"GPU {args.gpu} was not idle in all three pre-launch samples")
        model = MODELS[name]
        if not model.startswith("/"):
            model = str(args.stage4 / model)
        work.mkdir(parents=True, exist_ok=True)
        log = args.output / "logs" / f"{name}_gpu{args.gpu}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "PYTHONPATH": clean_site + ":" + str(args.project),
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_PORT": str(58000 + int(args.gpu) * 10),
        })
        command = [
            lm_python, "-m", "lm_eval", "run", "--model", "vllm",
            "--model_args", f"pretrained={model},dtype=bfloat16,gpu_memory_utilization=0.55,max_model_len=4096,enable_thinking=False",
            "--tasks", "arc_challenge", "--num_fewshot", "25", "--batch_size", "auto",
            "--apply_chat_template", "--seed", "42,42,42,42", "--output_path", str(work),
        ]
        with log.open("a", encoding="utf-8") as handle:
            handle.write(" ".join(command) + "\n")
            handle.flush()
            subprocess.run(command, cwd=args.project, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        candidates = sorted(work.glob("**/results*.json"))
        if not candidates:
            raise RuntimeError(f"{name}: lm-eval returned success without a results JSON")
        source = candidates[-1]
        payload = json.loads(source.read_text(encoding="utf-8"))
        entry = payload.get("results", {}).get("arc_challenge", {})
        value = entry.get("acc_norm,none", entry.get("acc,none"))
        if not isinstance(value, (float, int)):
            raise RuntimeError(f"{name}: ARC-Challenge accuracy is missing in {source}")
        result.write_text(json.dumps({
            "status": "completed", "model": name, "score": float(value), "metric": "acc_norm,none",
            "source": str(source),
        }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
