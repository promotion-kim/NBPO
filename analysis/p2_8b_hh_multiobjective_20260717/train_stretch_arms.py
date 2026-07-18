#!/usr/bin/env python3
"""Train preregistered stretch arms with the frozen core-arm budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import yaml

from analysis.p2_8b_hh_multiobjective_20260717 import train_core_arms as core


STRETCH = {
    "ht_mnpo_helpful": {"loss_type": "ht_mnpo", "ronpo_target_column": "target_os_k0p01", "ht_target_column": "ht_target_helpfulness"},
    "sppo_avg": {"loss_type": "sppo", "ronpo_target_column": "target_os_k0p01"},
    "simpo": {"loss_type": "simpo", "ronpo_target_column": "target_os_k0p01"},
    "ipo": {"loss_type": "ipo", "ronpo_target_column": "target_os_k0p01"},
    "ronpo_full_expect": {"loss_type": "ronpo", "ronpo_target_column": "target_fullexp_k0p01"},
}
core.ARMS.update({name: {key: value for key, value in settings.items() if key in {"loss_type", "ronpo_target_column"}} for name, settings in STRETCH.items()})


def run_id(stage: str, arm: str) -> str:
    return hashlib.sha256(f"p2-hh-llama31-stretch-s42|{stage}|{arm}".encode()).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stage", choices=["smoke", "full"], required=True)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--wandb-env", type=Path, required=True)
    parser.add_argument("--gpus", required=True)
    args = parser.parse_args()
    arms = [name for name in args.arms.split(",") if name]
    gpus = [int(value) for value in args.gpus.split(",") if value]
    if not arms or len(arms) != len(gpus) or any(name not in STRETCH for name in arms):
        raise RuntimeError("requested stretch arms/GPU mapping is invalid")
    python = args.venv / "bin/python"
    accel = args.project / "accelerate_configs/single_gpu.yaml"
    stage_root = args.root / "train" / "stretch" / args.stage
    logs = args.root / "logs" / "train"
    logs.mkdir(parents=True, exist_ok=True)
    common_env = os.environ.copy()
    common_env.update(core.load_env(args.wandb_env))
    common_env.update({
        "PYTHONPATH": str(args.project), "WANDB_MODE": "online", "WANDB_ENTITY": "promotion-kim",
        "WANDB_PROJECT": "mnpo", "WANDB_RUN_GROUP": "p2-llama31-hh-multiobjective-s42",
        "WANDB_DIR": str(args.root / "wandb"), "MNPO_DISABLE_CUDNN_SDPA": "1",
        "TORCH_CUDNN_SDPA_ENABLED": "0", "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    if args.stage == "smoke":
        common_env["MNPO_SKIP_FINAL_SAVE"] = "1"
    jobs = []
    for gpu, arm in zip(gpus, arms):
        output = stage_root / arm
        output.mkdir(parents=True, exist_ok=True)
        cfg = output / "config.yaml"
        values = core.config(args.model, args.dataset, output, arm, args.stage)
        values["ht_target_column"] = STRETCH[arm].get("ht_target_column", "ht_target")
        values["run_name"] = f"p2-hh-llama31-stretch-{args.stage}-{arm}-s42"
        cfg.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
        env = common_env.copy()
        ident = run_id(args.stage, arm)
        env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "WANDB_RUN_ID": ident,
                    "WANDB_NAME": values["run_name"], "WANDB_RESUME": "allow"})
        log = logs / f"stretch_{args.stage}_{arm}.log"
        handle = log.open("a", encoding="utf-8")
        command = [str(python), "-m", "accelerate.commands.launch", "--config_file", str(accel),
                   "--num_processes=1", "-m", "mnpo_scripts.run_mnpo", str(cfg)]
        started = time.time()
        proc = subprocess.Popen(command, cwd=args.project, env=env, stdout=handle, stderr=subprocess.STDOUT)
        (output / "job_status.json").write_text(json.dumps({
            "status": "running", "phase": "stretch", "stage": args.stage, "arm": arm, "gpu": gpu,
            "seed": 42, "max_steps": 20 if args.stage == "smoke" else 900, "effective_batch": 16,
            "wandb_run_id": ident, "config_sha256": core.sha(cfg),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)), "command": command,
        }, indent=2) + "\n", encoding="utf-8")
        jobs.append((arm, gpu, output, cfg, ident, log, handle, proc))
    failures = []
    for arm, gpu, output, cfg, ident, log, handle, proc in jobs:
        rc = proc.wait(); handle.close()
        metrics_path = output / "train_results.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
        numeric = [float(value) for value in metrics.values() if isinstance(value, (int, float))]
        text = log.read_text(errors="replace")[-30000:] if log.is_file() else ""
        checkpoint = output / "checkpoint-900"
        clean = (rc == 0 and bool(numeric) and all(math.isfinite(value) for value in numeric)
                 and (args.stage == "smoke" or (checkpoint / "config.json").is_file())
                 and "Traceback (most recent call last)" not in text and "out of memory" not in text.lower())
        status = {"status": "completed" if clean else "failed", "phase": "stretch", "stage": args.stage,
                  "arm": arm, "gpu": gpu, "returncode": rc, "finite_metrics": bool(numeric) and all(math.isfinite(value) for value in numeric),
                  "model_present": args.stage == "smoke" or (checkpoint / "config.json").is_file(),
                  "max_steps": 20 if args.stage == "smoke" else 900, "effective_batch": 16,
                  "wandb_run_id": ident, "wandb_url": f"https://wandb.ai/promotion-kim/mnpo/runs/{ident}",
                  "config": str(cfg), "config_sha256": core.sha(cfg), "log": str(log),
                  "checkpoint": str(checkpoint) if args.stage == "full" else None,
                  "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (output / "job_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        if not clean:
            failures.append(arm)
    summary = {"phase": "stretch", "stage": args.stage, "arms": arms, "failed": failures,
               "status": "complete" if not failures else "failed"}
    (stage_root / f"summary_{hashlib.sha256(','.join(arms).encode()).hexdigest()[:8]}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
