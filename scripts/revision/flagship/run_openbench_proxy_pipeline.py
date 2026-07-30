#!/usr/bin/env python3
"""Run generation, four-GPU local judging, aggregation, and W&B logging."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--models-tsv", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    logs = args.work / "logs"
    logs.mkdir(exist_ok=True)
    status_path = args.work / "pipeline_status.json"
    common_env = os.environ.copy()
    common_env.update({
        "HF_HOME": str(args.hf_cache),
        "HF_HUB_CACHE": str(args.hf_cache / "hub"),
        "TOKENIZERS_PARALLELISM": "false",
        "TORCH_CUDNN_SDPA_ENABLED": "0",
        "WANDB_MODE": "online",
        "WANDB_ENTITY": "promotion-kim",
        "WANDB_PROJECT": "mnpo",
    })

    atomic_json(status_path, {
        "status": "running", "stage": "model_generation",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    generation_command = [
        args.python,
        str(args.project / "scripts/revision/flagship/run_openbench_proxy_generations.py"),
        "--project", str(args.project), "--python", args.python,
        "--models-tsv", str(args.models_tsv), "--prompts", str(args.prompts),
        "--work", str(args.work), "--gpus", "0,1,2,3",
    ]
    with (logs / "generation_scheduler.log").open("a", encoding="utf-8") as handle:
        subprocess.run(generation_command, env=common_env, stdout=handle, stderr=subprocess.STDOUT, check=True)

    atomic_json(status_path, {
        "status": "running", "stage": "qwen3_32b_pairwise_judge",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    judge_dir = args.work / "judgments"
    judge_dir.mkdir(exist_ok=True)
    running = []
    for gpu in range(4):
        output = judge_dir / f"shard_{gpu}.jsonl"
        log_handle = (logs / f"judge_shard_{gpu}.log").open("a", encoding="utf-8")
        command = [
            args.python,
            str(args.project / "scripts/revision/flagship/judge_openbench_pairwise.py"),
            "--responses-root", str(args.work / "responses"),
            "--protocol-lock", str(args.protocol_lock),
            "--output", str(output), "--shard-index", str(gpu), "--num-shards", "4",
            "--gpu-memory-utilization", "0.90",
        ]
        env = common_env.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "VLLM_HOST_IP": "127.0.0.1",
            "VLLM_PORT": str(64000 + gpu * 20),
            "VLLM_DP_MASTER_PORT": str(64001 + gpu * 20),
            "MASTER_PORT": str(64002 + gpu * 20),
        })
        process = subprocess.Popen(command, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
        running.append((gpu, process, log_handle))
    failures = []
    for gpu, process, handle in running:
        returncode = process.wait()
        handle.close()
        if returncode != 0:
            failures.append({"gpu": gpu, "returncode": returncode})
    if failures:
        atomic_json(status_path, {"status": "failed", "stage": "judge", "failures": failures})
        raise RuntimeError(f"judge failures: {failures}")

    atomic_json(status_path, {
        "status": "running", "stage": "aggregation",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    result_dir = args.work / "results"
    subprocess.run([
        args.python,
        str(args.project / "scripts/revision/flagship/aggregate_openbench_proxy.py"),
        "--judge-dir", str(judge_dir), "--protocol-lock", str(args.protocol_lock),
        "--output-dir", str(result_dir),
    ], env=common_env, check=True)

    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    run_id = hashlib.sha256(b"aaai27-qwen3-8b-openbench-proxy-v1").hexdigest()[:12]
    import wandb
    run = wandb.init(
        entity="promotion-kim", project="mnpo", id=run_id, resume="allow",
        name="aaai27-qwen3-8b-openbench-local-judge-proxy",
        group="flagship_openbench_proxy", job_type="evaluation",
        config=json.loads(args.protocol_lock.read_text(encoding="utf-8")),
    )
    metrics = {}
    for row in summary["comparisons"]:
        key = f"proxy/{row['benchmark']}/ronpo_vs_{row['opponent']}"
        metrics[key] = row["proxy_win_rate"]
        metrics[key + "_ci_low"] = row["ci95_low"]
        metrics[key + "_ci_high"] = row["ci95_high"]
    run.log(metrics)
    artifact = wandb.Artifact("qwen3-8b-openbench-proxy-results", type="evaluation")
    artifact.add_dir(str(result_dir))
    run.log_artifact(artifact)
    run_url = run.url
    run.finish()
    atomic_json(status_path, {
        "status": "completed", "stage": "done", "wandb_run_id": run_id,
        "wandb_url": run_url,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })


if __name__ == "__main__":
    main()
