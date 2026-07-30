#!/usr/bin/env python3
"""Queue matched-budget SafeRLHF Table-4 arms across the authorized GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import yaml


def target_tag(kappa: float) -> str:
    return f"{kappa:g}".replace(".", "p")


def arm_specs(lock: dict, wave: str, best_other: str | None) -> list[tuple[str, dict]]:
    choices = {float(item["entropy_target"]): float(item["selected_kappa"]) for item in lock["selected"]}
    def os_arm(target: float, name: str, lr: float = 5e-7) -> tuple[str, dict]:
        return name, {"loss_type": "ronpo", "ronpo_target_column": f"target_os_k{target_tag(choices[target])}", "learning_rate": lr, "diagnostic": target != 0.55 or lr != 5e-7}
    w1 = [
        os_arm(0.55, "ronpo_os_confirmatory"),
        ("inpo_avg", {"loss_type": "inpo"}),
        ("sppo_avg", {"loss_type": "sppo"}),
        ("simpo", {"loss_type": "simpo"}),
        ("ipo", {"loss_type": "ipo"}),
        ("dpo", {"loss_type": "dpo"}),
        ("ht_mnpo_harmless", {"loss_type": "ht_mnpo", "ht_target_column": "ht_target"}),
        ("ht_mnpo_helpfulness", {"loss_type": "ht_mnpo", "ht_target_column": "ht_target_helpfulness"}),
    ]
    if wave == "w1":
        return w1
    if wave != "w2":
        raise ValueError(wave)
    specs = [
        ("ronpo_topmass", {"loss_type": "ronpo", "ronpo_target_column": f"target_topmass_k{target_tag(choices[0.55])}", "diagnostic": True}),
        ("ronpo_uniform", {"loss_type": "ronpo", "ronpo_target_column": "target_uniform", "diagnostic": True}),
        os_arm(0.15, "ronpo_os_entropy_0p15_diagnostic"),
        os_arm(0.85, "ronpo_os_entropy_0p85_diagnostic"),
        ("mnpo", {"loss_type": "mnpo", "diagnostic": True}),
        os_arm(0.55, "ronpo_os_lr_1e-6_diagnostic", lr=1e-6),
        ("inpo_avg_lr_1e-6_fairness", {"loss_type": "inpo", "learning_rate": 1e-6, "diagnostic": True}),
    ]
    if best_other:
        if best_other not in {name for name, _ in w1 if name != "ronpo_os_confirmatory"}:
            raise ValueError(f"invalid best_other={best_other}")
        base = next(config for name, config in w1 if name == best_other).copy()
        base.update({"learning_rate": 1e-6, "diagnostic": True})
        specs.append((f"{best_other}_lr_1e-6_fairness", base))
    return specs


def config(model: str, dataset: str, output: Path, arm: str, spec: dict, steps: int) -> dict:
    return {
        "model_name_or_path": model, "torch_dtype": None, "attn_implementation": "sdpa",
        "dataset_mixer": {dataset: 1.0}, "dataset_splits": ["train", "test"],
        "preprocessing_num_workers": 4, "bf16": True, "loss_type": spec["loss_type"],
        "eta": 0.0075, "ratio": 0.3333, "max_history_t": 1, "history_weights": [1.0],
        "dpo_beta": 0.05, "simpo_beta": 2.0, "simpo_gamma": 0.6,
        "ronpo_alpha": 1.0, "ronpo_tau": 0.05,
        "ronpo_target_column": spec.get("ronpo_target_column", "target_uniform"),
        "ht_target_column": spec.get("ht_target_column", "ht_target"), "ht_target_scale": 1.0,
        "reference_anchor_weight": 0.05, "preference_sft_weight": 0.005,
        "beta": 10.0, "learning_rate": spec.get("learning_rate", 5e-7),
        "lr_scheduler_type": "cosine", "warmup_ratio": 0.1, "optim": "adamw_torch",
        "weight_decay": 0.0, "max_grad_norm": 1.0, "seed": 42,
        "gradient_accumulation_steps": 16, "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False}, "num_train_epochs": 100,
        "max_steps": steps, "per_device_train_batch_size": 1, "per_device_eval_batch_size": 1,
        "max_length": 2048, "max_prompt_length": 1024, "do_eval": False, "eval_strategy": "no",
        "logging_steps": 1 if steps <= 20 else 5, "log_level": "info", "generate_during_eval": False,
        "load_best_model_at_end": False, "save_strategy": "no" if steps <= 20 else "steps",
        "save_steps": steps, "save_total_limit": 1, "save_only_model": True, "save_safetensors": True,
        "push_to_hub": False, "report_to": ["none"], "output_dir": str(output),
        "run_name": f"p4-saferlhf-table4-{arm}-s42-{steps}steps",
    }


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--kappa-lock", type=Path, required=True)
    parser.add_argument("--wave", choices=["w1", "w2"], required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--best-other", default=None)
    args = parser.parse_args()
    lock = json.loads(args.kappa_lock.read_text(encoding="utf-8"))
    arms = arm_specs(lock, args.wave, args.best_other)
    gpus = [int(value) for value in args.gpus.split(",")]
    if not gpus:
        raise ValueError("no GPUs")
    root = args.experiment / "train" / f"{args.wave}_{args.steps}steps"
    logs = args.experiment / "logs" / "train"
    root.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    env_base = os.environ.copy()
    env_base.update({"PYTHONPATH": str(args.project), "TORCH_CUDNN_SDPA_ENABLED": "0", "MNPO_DISABLE_CUDNN_SDPA": "1", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    active: list[tuple[subprocess.Popen, object, str, int, Path, Path, dict, float]] = []
    completed = []

    def collect(block: bool) -> None:
        nonlocal active
        keep = []
        for process, handle, arm, gpu, output, cfg, spec, started in active:
            rc = process.wait() if block else process.poll()
            if rc is None:
                keep.append((process, handle, arm, gpu, output, cfg, spec, started)); continue
            handle.close()
            text = (logs / f"{args.wave}_{args.steps}_{arm}.log").read_text(errors="replace")[-30000:]
            metrics_path = output / "train_results.json"
            metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
            finite = bool(metrics) and all(math.isfinite(float(v)) for v in metrics.values() if isinstance(v, (float, int)))
            checkpoint = output / f"checkpoint-{args.steps}"
            model_ok = args.steps <= 20 or (checkpoint / "config.json").is_file()
            status = "completed" if rc == 0 and finite and model_ok and "Traceback (most recent call last)" not in text and "out of memory" not in text.lower() else "failed"
            grad_matches = re.findall(r"['\"]grad_norm['\"]\s*:\s*['\"]?([-+0-9.eE]+)", text)
            payload = {"status": status, "returncode": rc, "arm": arm, "gpu": gpu, "steps": args.steps, "effective_batch": 16, "learning_rate": spec.get("learning_rate", 5e-7), "loss_type": spec["loss_type"], "diagnostic": bool(spec.get("diagnostic", False)), "config": str(cfg), "config_sha256": sha(cfg), "log": str(logs / f"{args.wave}_{args.steps}_{arm}.log"), "checkpoint": str(checkpoint), "elapsed_seconds": time.time() - started, "first_logged_grad_norm": float(grad_matches[0]) if grad_matches else None}
            (output / "job_status.json").write_text(json.dumps(payload, indent=2) + "\n")
            completed.append(payload)
        active = keep

    for index, (arm, spec) in enumerate(arms):
        while len(active) >= len(gpus):
            time.sleep(3); collect(block=False)
        gpu = gpus[index % len(gpus)] if len(active) < len(gpus) else gpus[0]
        busy = {entry[3] for entry in active}
        gpu = next(candidate for candidate in gpus if candidate not in busy)
        output = root / arm; output.mkdir(parents=True, exist_ok=True)
        cfg = output / "config.yaml"; cfg.write_text(yaml.safe_dump(config(args.model, args.dataset, output, arm, spec, args.steps), sort_keys=False))
        log = logs / f"{args.wave}_{args.steps}_{arm}.log"
        env = env_base.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [str(args.venv / "bin/python"), "-m", "accelerate.commands.launch", "--config_file", str(args.project / "accelerate_configs/single_gpu.yaml"), "--num_processes=1", "-m", "mnpo_scripts.run_mnpo", str(cfg)]
        handle = log.open("a", encoding="utf-8")
        started = time.time(); process = subprocess.Popen(command, cwd=args.project, env=env, stdout=handle, stderr=subprocess.STDOUT)
        (output / "job_status.json").write_text(json.dumps({"status": "running", "arm": arm, "gpu": gpu, "steps": args.steps, "config": str(cfg), "config_sha256": sha(cfg), "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))}, indent=2) + "\n")
        active.append((process, handle, arm, gpu, output, cfg, spec, started))
    while active:
        time.sleep(3); collect(block=False)
    summary = {"wave": args.wave, "steps": args.steps, "arms": completed, "status": "complete" if all(row["status"] == "completed" for row in completed) else "failed"}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
