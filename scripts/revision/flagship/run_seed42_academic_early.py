#!/usr/bin/env python3
"""Use only free GPUs 1/3 for cost-free P2 tasks while two trainers run.

The existing IFEval dispatcher is paused before this worker is launched.  This
worker drains when both remaining verified HF models become available, resumes
that dispatcher, and leaves all completed JSONs in the final academic work tree
so the full idempotent runner will skip them later.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

from run_seed42_academic_suite import (
    GROUPS,
    command,
    models_from_ledger,
    result_complete,
    write_status,
)


EARLY_METHODS = (
    "base", "ronpo_full_expect", "ronpo_k_only", "dpo", "ipo", "simpo", "sppo_avg",
    "inpo_avg", "ht_mnpo_helpfulness",
)
WAITING_METHODS = ("ht_mnpo_safety", "ht_mnpo_conciseness")


def safe_resume_ifeval(pid: int) -> bool:
    try:
        command_line = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return False
    if "run_seed42_ifeval.py" not in command_line:
        return False
    os.kill(pid, signal.SIGCONT)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--stop-at", required=True)
    parser.add_argument("--resume-ifeval-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    status_path = args.work / "early_status.json"
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / "logs").mkdir(exist_ok=True)
    stop_at = datetime.fromisoformat(args.stop_at).timestamp()

    models = models_from_ledger(args.root, args.base_revision)
    missing = [method for method in EARLY_METHODS if method not in models]
    if missing:
        write_status(status_path, "blocked", reason="verified HF models missing", missing=missing)
        safe_resume_ifeval(args.resume_ifeval_pid)
        return

    pending = []
    completed = []
    for method in EARLY_METHODS:
        for group, shots, tasks in GROUPS:
            job = (method, group, shots, tasks)
            output = args.work / "raw" / method / group
            item = {"method": method, "group": group, "tasks": list(tasks)}
            if result_complete(output, tasks):
                completed.append(item | {"note": "pre-existing"})
            else:
                pending.append(job)

    running: dict[int, tuple[tuple, subprocess.Popen, object, int]] = {}
    attempts: dict[tuple[str, str], int] = {}
    failed = []
    drain_reason = "all_early_jobs_finished"
    try:
        while (pending or running) and time.time() < stop_at:
            available = models_from_ledger(args.root, args.base_revision)
            drain = all(method in available for method in WAITING_METHODS)
            if drain:
                drain_reason = "remaining_models_verified; yield GPUs to final IFEval"

            for gpu, (job, process, handle, attempt) in list(running.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                handle.close()
                method, group, shots, tasks = job
                output = args.work / "raw" / method / group
                item = {"method": method, "group": group, "tasks": list(tasks)}
                if returncode == 0 and result_complete(output, tasks):
                    completed.append(item | {"attempt": attempt})
                elif attempt < 3 and not drain:
                    pending.insert(0, job)
                else:
                    failed.append(item | {"attempt": attempt, "returncode": returncode})
                del running[gpu]

            if not drain:
                for gpu in (1, 3):
                    if gpu in running or not pending:
                        continue
                    job = pending.pop(0)
                    method, group, shots, tasks = job
                    key = (method, group)
                    attempts[key] = attempts.get(key, 0) + 1
                    attempt = attempts[key]
                    output = args.work / "raw" / method / group
                    output.mkdir(parents=True, exist_ok=True)
                    log_path = args.work / "logs" / f"{method}_{group}_a{attempt}.log"
                    handle = log_path.open("a", encoding="utf-8")
                    env = os.environ.copy()
                    env.update({
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "HF_HOME": str(args.root / "cache/huggingface"),
                        "HF_HUB_CACHE": str(args.root / "cache/huggingface/hub"),
                        "HF_DATASETS_CACHE": str(args.root / "cache/huggingface/datasets"),
                        "WANDB_MODE": "online", "WANDB_ENTITY": "promotion-kim",
                        "WANDB_PROJECT": "mnpo", "HF_ALLOW_CODE_EVAL": "1",
                        "TOKENIZERS_PARALLELISM": "false", "VLLM_HOST_IP": "127.0.0.1",
                        "VLLM_PORT": str(65000 + gpu * 20),
                        "VLLM_DP_MASTER_PORT": str(65001 + gpu * 20),
                        "MASTER_PORT": str(65002 + gpu * 20),
                        "TORCH_CUDNN_SDPA_ENABLED": "0",
                    })
                    process = subprocess.Popen(
                        command(args, models[method], group, shots, tasks, output),
                        cwd=args.project, env=env, stdout=handle, stderr=subprocess.STDOUT,
                    )
                    running[gpu] = (job, process, handle, attempt)

            write_status(
                status_path, "draining" if drain else "running",
                gpu_ids=[1, 3], pending_count=len(pending),
                running=[{
                    "gpu": gpu, "method": job[0], "group": job[1],
                    "pid": process.pid, "attempt": attempt,
                } for gpu, (job, process, _, attempt) in running.items()],
                completed=completed, failed=failed, drain_reason=drain_reason if drain else None,
            )
            if drain and not running:
                break
            time.sleep(args.poll_seconds)
    finally:
        resumed = safe_resume_ifeval(args.resume_ifeval_pid)

    state = "yielded_to_final_ifeval" if pending else "completed_with_failures" if failed else "completed"
    write_status(
        status_path, state, gpu_ids=[1, 3], completed=completed, failed=failed,
        remaining_jobs=[{"method": m, "group": g} for m, g, _, _ in pending],
        ifeval_dispatcher_resumed=resumed, reason=drain_reason,
    )


if __name__ == "__main__":
    main()
