#!/usr/bin/env python3
"""Train a frozen seed-43 Stage-3/4 continuation arm with W&B online."""

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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_pretraining_lock(args: argparse.Namespace) -> None:
    """Accept the global lock, or a strictly arm-local early Stage-4 lock.

    The early lock is valid only for a Stage-3 parent that already completed
    and passed the frozen gate.  This permits safe backfilling of idle GPUs
    without pretending that the still-running arms are ready for Stage 4.
    """
    global_lock = args.experiment / "continuation_lock.json"
    if global_lock.is_file():
        return
    if args.continuation_stage != "stage4":
        raise RuntimeError("missing pre-training continuation lock")
    early_lock = args.experiment / "early_locks" / f"{args.arm}.json"
    if not early_lock.is_file():
        raise RuntimeError("missing global or arm-local pre-training continuation lock")
    lock = json.loads(early_lock.read_text(encoding="utf-8"))
    expected_target = args.target_column or None
    expected = {
        "status": "locked_before_stage4_pool_or_training",
        "stage": "stage4",
        "seed": args.seed,
        "arm": args.arm,
        "loss_type": args.loss_type,
        "target_column": expected_target,
        "parent": str(Path(args.parent_model)),
    }
    for key, value in expected.items():
        if lock.get(key) != value:
            raise RuntimeError(f"arm-local lock mismatch for {key}: {lock.get(key)!r} != {value!r}")
    for field in ("parent_status", "required_parent_gate"):
        evidence = lock.get(field, {})
        path = Path(evidence.get("path", ""))
        if not path.is_file() or evidence.get("sha256") != sha(path):
            raise RuntimeError(f"arm-local lock evidence changed or is missing: {field}")
    gate_path = Path(lock["required_parent_gate"]["path"])
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "passed" or gate.get("passed") is not True:
        raise RuntimeError("arm-local lock parent gate is not passed")


def build_config(args: argparse.Namespace, dataset: Path, output: Path, steps: int) -> dict:
    return {
        "model_name_or_path": args.parent_model, "torch_dtype": None, "attn_implementation": "sdpa",
        "dataset_mixer": {str(dataset): 1.0}, "dataset_splits": ["train", "test"], "preprocessing_num_workers": 4,
        "bf16": True, "loss_type": args.loss_type, "eta": 0.0075, "ratio": 0.3333,
        "max_history_t": 1, "history_weights": [1.0], "dpo_beta": 0.05, "simpo_beta": 2.0, "simpo_gamma": 0.6,
        "ronpo_alpha": 1.0, "ronpo_tau": 0.05, "ronpo_target_column": args.target_column or "target_os_k0p1",
        "ht_target_column": args.target_column or "ht_target", "ht_target_scale": 1.0,
        "reference_anchor_weight": 0.05, "preference_sft_weight": 0.005, "beta": 10.0,
        "learning_rate": 5e-7, "lr_scheduler_type": "cosine", "warmup_ratio": 0.1, "optim": "adamw_torch",
        "weight_decay": 0.0, "max_grad_norm": 1.0, "seed": args.seed, "gradient_accumulation_steps": 16,
        "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "num_train_epochs": 100, "max_steps": steps, "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1, "max_length": 2048, "max_prompt_length": 1024,
        "do_eval": False, "eval_strategy": "no", "logging_steps": 1 if steps <= 20 else 5,
        "log_level": "info", "generate_during_eval": False, "load_best_model_at_end": False,
        "save_strategy": "no", "save_total_limit": 1, "save_only_model": True, "save_safetensors": True,
        "push_to_hub": False, "report_to": ["wandb"], "output_dir": str(output),
        "run_name": f"{args.run_prefix}-llama31-saferlhf-{args.continuation_stage}-{args.arm}-s{args.seed}-{steps}steps",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--continuation-stage", choices=["stage3", "stage4"], required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--parent-model", required=True)
    parser.add_argument("--loss-type", choices=["ronpo", "inpo", "sppo", "mnpo", "ht_mnpo", "dpo", "ipo", "simpo"], required=True)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--run-stage", choices=["smoke", "full"], required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--run-prefix", default="p10")
    args = parser.parse_args()
    verify_pretraining_lock(args)
    parent_status = Path(args.parent_model) / "job_status.json"
    if not parent_status.is_file() or json.loads(parent_status.read_text(encoding="utf-8")).get("status") != "completed":
        raise RuntimeError(f"parent is not a completed full run: {args.parent_model}")
    steps = 20 if args.run_stage == "smoke" else 900
    dataset = args.experiment / args.continuation_stage / args.arm / "pool" / "precompute" / "targets"
    if not (dataset / "dataset_dict.json").is_file():
        raise RuntimeError(f"missing prepared dataset: {dataset}")
    output = args.experiment / args.continuation_stage / args.arm / "train" / args.run_stage
    if (output / "job_status.json").is_file():
        raise RuntimeError(f"refusing to overwrite terminal run: {output}")
    # An infrastructure interruption can leave a directory without a terminal
    # job status. Reuse that directory for a clean restart; a terminal status
    # remains immutable and is rejected above.
    output.mkdir(parents=True, exist_ok=True)
    config = output / "config.yaml"
    config.write_text(yaml.safe_dump(build_config(args, dataset, output, steps), sort_keys=False), encoding="utf-8")
    run_name = f"{args.run_prefix}-llama31-saferlhf-{args.continuation_stage}-{args.arm}-s{args.seed}-{steps}steps"
    run_id = hashlib.sha256(run_name.encode()).hexdigest()[:12]
    attempt_dir = args.experiment / "logs" / args.continuation_stage / "attempts"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    log = attempt_dir / f"{args.arm}_{args.run_stage}_{time.time_ns()}.log"
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(args.project), "CUDA_VISIBLE_DEVICES": str(args.gpu), "WANDB_MODE": "online",
        "WANDB_ENTITY": "promotion-kim", "WANDB_PROJECT": "mnpo",
        "WANDB_RUN_GROUP": f"{args.run_prefix}-llama31-saferlhf-{args.continuation_stage}-s{args.seed}",
        "WANDB_DIR": str(args.experiment / "wandb"), "WANDB_RUN_ID": run_id, "WANDB_NAME": run_name,
        "WANDB_RESUME": "allow", "MNPO_DISABLE_CUDNN_SDPA": "1", "TORCH_CUDNN_SDPA_ENABLED": "0",
        "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    if args.run_stage == "smoke":
        env["MNPO_SKIP_FINAL_SAVE"] = "1"
    command = [str(args.venv / "bin/python"), "-m", "accelerate.commands.launch", "--config_file",
               str(args.project / "accelerate_configs" / "single_gpu.yaml"), "--num_processes=1", "-m",
               "mnpo_scripts.run_mnpo", str(config)]
    started = time.time()
    with log.open("w", encoding="utf-8") as handle:
        rc = subprocess.Popen(command, cwd=args.project, env=env, stdout=handle, stderr=subprocess.STDOUT).wait()
    metrics_path = output / "train_results.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    finite = bool(metrics) and all(math.isfinite(float(v)) for v in metrics.values() if isinstance(v, (int, float)))
    tail = log.read_text(encoding="utf-8", errors="replace")[-50000:]
    clean = rc == 0 and finite and "Traceback (most recent call last)" not in tail and "out of memory" not in tail.lower()
    status = {
        "status": "completed" if clean else "failed", "continuation_stage": args.continuation_stage,
        "run_stage": args.run_stage, "arm": args.arm, "loss_type": args.loss_type,
        "target_column": args.target_column, "gpu": args.gpu, "steps": steps, "effective_batch": 16,
        "seed": args.seed, "returncode": rc, "finite_metrics": finite, "parent_model": args.parent_model,
        "dataset": str(dataset), "config": str(config), "config_sha256": sha(config), "log": str(log),
        "model_dir": str(output) if args.run_stage == "full" else None, "wandb_run_id": run_id,
        "wandb_url": f"https://wandb.ai/promotion-kim/mnpo/runs/{run_id}", "elapsed_seconds": time.time() - started,
    }
    (output / "job_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))
    if not clean:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
