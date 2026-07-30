#!/usr/bin/env python3
"""Fail-closed supervisor for the preregistered fair 8B demonstration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def model_complete(path: Path) -> bool:
    return (path / "model.safetensors").is_file() or (path / "model.safetensors.index.json").is_file()


def run(command: list[str], *, cwd: Path, log: Path, env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def gpu_snapshot() -> list[dict]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,power.draw",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        index, memory, utilization, power = [value.strip() for value in line.split(",")]
        rows.append({"gpu": int(index), "memory_used_mib": int(memory),
                     "utilization_percent": int(utilization), "power_w": float(power)})
    return rows


def idle_gpus() -> list[int]:
    """Return GPUs with no compute process; utilization alone is not an ownership test."""
    inventory = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    uuid_to_index = {}
    for line in inventory.stdout.splitlines():
        index, uuid = [value.strip() for value in line.split(",", maxsplit=1)]
        uuid_to_index[uuid] = int(index)
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    busy = set()
    for line in processes.stdout.splitlines():
        if not line.strip():
            continue
        uuid = line.split(",", maxsplit=1)[0].strip()
        if uuid in uuid_to_index:
            busy.add(uuid_to_index[uuid])
    return sorted(set(uuid_to_index.values()) - busy)


def process_alive(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    try:
        # A zombie still answers kill(pid, 0), but cannot make progress and must
        # not keep the idempotent supervisor waiting forever.
        if stat.read_text(encoding="utf-8").split()[2] == "Z":
            return False
    except (OSError, IndexError):
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def measured_log_step(path: Path, expected: int = 900) -> int:
    """Read optimizer progress from the append-only Trainer log for audit only."""
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 262144))
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return 0
    matches = re.findall(rf"\b(\d+)/{expected}\b", tail)
    return max((int(value) for value in matches), default=0)


def supplemental_wandb_run_id(candidate_id: str) -> str:
    payload = f"qwen3-8b-fair-demo-v1|{candidate_id}|42|supplemental-recovery-v1"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--flagship-root", type=Path, required=True)
    parser.add_argument("--fair-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--general-rm-cache", type=Path, required=True)
    parser.add_argument("--qwen-judge", type=Path, required=True)
    parser.add_argument("--gpt-judge", type=Path, required=True)
    args = parser.parse_args()
    grid = args.run_dir / "sweep/grid.json"
    prereg = args.run_dir / "prereg_lock.json"
    candidates = [row["id"] for row in json.loads(grid.read_text())["candidates"]]
    logs = args.run_dir / "pipeline_logs"; logs.mkdir(parents=True, exist_ok=True)
    status_path = args.run_dir / "pipeline_status.json"

    sweep_command = [
        args.python, str(args.project / "scripts/revision/flagship/run_fair_demo_symmetric_sweep.py"),
        "--project", str(args.project), "--python", args.python, "--grid", str(grid),
        "--prereg-lock", str(prereg), "--flagship-root", str(args.flagship_root),
        "--fair-root", str(args.fair_root), "--work", str(args.fair_root / "sweep"),
        "--base-model", str(args.base_model),
    ]
    prior_status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    retry_count = (int(prior_status.get("retry_count", 0))
                   if prior_status.get("stage") == "symmetric_training" else 0)
    last_hour = None
    supplemental_registry = args.run_dir / "supplemental_training.json"
    terminal_training = args.run_dir / "training_terminal.json"
    while True:
        complete = []
        for candidate in candidates:
            root = args.fair_root / "sweep/candidates" / candidate
            state_path = root / "training_status.json"
            state = json.loads(state_path.read_text()) if state_path.is_file() else {}
            if state.get("status") == "completed" and model_complete(root):
                complete.append(candidate)
        if len(complete) == len(candidates):
            atomic_json(terminal_training, {
                "status": "completed", "completed_candidates": sorted(complete),
                "terminal_failed_candidates": [], "pre_ranking_same_config_retries": retry_count,
                "spent_sealed_split_touched": False,
                "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            })
            break
        manifest_path = args.fair_root / "sweep/training_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        updated = manifest.get("updated_at")
        fresh_running = False
        if manifest.get("status") == "running" and updated:
            age = datetime.now().astimezone().timestamp() - datetime.fromisoformat(updated).timestamp()
            fresh_running = age < 180
        now = datetime.now().astimezone()
        hour_key = now.strftime("%Y%m%dT%H00%z")
        if hour_key != last_hour:
            log_steps = {
                candidate: measured_log_step(args.fair_root / "sweep/logs" / f"train_{candidate}.log")
                for candidate in candidates if candidate not in complete
            }
            atomic_json(args.run_dir / "hourly" / f"{now.strftime('%Y%m%dT%H%M%S%z')}.json", {
                "timestamp": now.isoformat(timespec="seconds"), "stage": "symmetric_training",
                "completed_candidates": complete, "remaining_candidates": sorted(set(candidates) - set(complete)),
                "training_manifest": manifest, "measured_log_steps": log_steps,
                "gpu_snapshot": gpu_snapshot(),
                "spent_sealed_split_touched": False,
            })
            last_hour = hour_key
        atomic_json(status_path, {"status": "running", "stage": "symmetric_training",
                                  "completed": complete, "remaining": sorted(set(candidates) - set(complete)),
                                  "retry_count": retry_count, "spent_sealed_split_touched": False,
                                  "updated_at": now.isoformat(timespec="seconds")})
        # A launch-time W&B authentication failure left one frozen candidate outside
        # the scheduler queue before its first optimizer update.  When a later wave
        # has an actually idle GPU, fill that slot with the exact preregistered job.
        # The registry makes this idempotent across supervisor restarts.
        accounted = set(manifest.get("pending", []))
        accounted.update(row.get("candidate_id") for row in manifest.get("running", []))
        orphan_missing = sorted((set(candidates) - set(complete)) - accounted)
        supplemental = (json.loads(supplemental_registry.read_text())
                        if supplemental_registry.is_file() else {})
        supplemental_active = bool(supplemental.get("pid") and process_alive(int(supplemental["pid"])))
        supplemental_retryable = (not supplemental or supplemental.get("status") in {
            "aborted_resource_reservation_race", "failed_wandb_init_timeout_before_step0",
        })
        if fresh_running and orphan_missing and supplemental_retryable and not supplemental_active:
            # A newly launched trainer may reserve a GPU in the scheduler
            # manifest before it creates a CUDA context visible to nvidia-smi.
            # Exclude both sources to avoid a launch-transition race.
            reserved = {int(row["gpu"]) for row in manifest.get("running", [])
                        if row.get("gpu") is not None}
            available = [gpu for gpu in idle_gpus() if gpu not in reserved]
            if available:
                candidate_id = orphan_missing[0]
                gpu = available[0]
                supplemental_command = [
                    args.python,
                    str(args.project / "scripts/revision/flagship/run_fair_demo_one_candidate.py"),
                    "--project", str(args.project), "--python", args.python,
                    "--grid", str(grid), "--prereg-lock", str(prereg),
                    "--flagship-root", str(args.flagship_root), "--fair-root", str(args.fair_root),
                    "--work", str(args.fair_root / "sweep"), "--base-model", str(args.base_model),
                    "--candidate-id", candidate_id, "--gpu", str(gpu),
                    "--wandb-run-id", supplemental_wandb_run_id(candidate_id),
                ]
                log_path = logs / f"supplemental_{candidate_id}.log"
                handle = log_path.open("a", encoding="utf-8")
                process = subprocess.Popen(supplemental_command, cwd=args.project,
                                           stdout=handle, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                handle.close()
                atomic_json(supplemental_registry, {
                    "status": "running",
                    "candidate_id": candidate_id, "gpu": gpu, "pid": process.pid,
                    "wandb_run_id": supplemental_wandb_run_id(candidate_id),
                    "command": supplemental_command, "log": str(log_path),
                    "launched_at": now.isoformat(timespec="seconds"),
                    "reason": "fill a verified idle GPU with the preregistered launch-time failure",
                })
        if fresh_running or supplemental_active:
            time.sleep(60); continue
        if retry_count >= 2:
            failed = sorted(set(candidates) - set(complete))
            atomic_json(terminal_training, {
                "status": "completed_with_terminal_failures",
                "completed_candidates": sorted(complete), "terminal_failed_candidates": failed,
                "pre_ranking_same_config_retries": retry_count,
                "reason": "Frozen candidates remained incomplete after two same-config pre-ranking retries; they are fail-closed and ineligible.",
                "spent_sealed_split_touched": False,
                "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            })
            break
        retry_count += 1
        retry_log = logs / f"sweep_retry_{retry_count}.log"
        with retry_log.open("a", encoding="utf-8") as handle:
            subprocess.run(sweep_command, cwd=args.project, stdout=handle,
                           stderr=subprocess.STDOUT, check=False)

    atomic_json(status_path, {"status": "running", "stage": "gpt_oss_evaluator_diagnostic",
                              "spent_sealed_split_touched": False,
                              "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
    judge_input = args.run_dir / "diagnostics/inputs/resolution_controls.json"
    judge_lock = args.run_dir / "diagnostics/judge_diagnostic_lock_v2.json"
    judge_jobs = []
    for gpu in range(4):
        output = args.run_dir / f"diagnostics/judges/gpt_oss_120b_max512/shard_{gpu}.jsonl"
        command = [
            args.python, str(args.project / "scripts/revision/flagship/judge_fair_demo_resolution.py"),
            "--input", str(judge_input), "--lock", str(judge_lock), "--judge-id", "gpt_oss_120b",
            "--model-path", str(args.gpt_judge), "--output", str(output),
            "--shard-index", str(gpu), "--num-shards", "4", "--batch-size", "128",
        ]
        environment = os.environ.copy()
        environment.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(args.project),
                            "HF_HOME": str(args.flagship_root / "cache/huggingface"),
                            "HF_HUB_CACHE": str(args.flagship_root / "cache/huggingface/hub"),
                            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                            "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "TOKENIZERS_PARALLELISM": "false"})
        handle = (logs / f"gpt_oss_diagnostic_v2_shard{gpu}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(command, cwd=args.project, env=environment,
                                   stdout=handle, stderr=subprocess.STDOUT)
        judge_jobs.append((gpu, process, handle, command))
    failures = []
    for gpu, process, handle, command in judge_jobs:
        returncode = process.wait(); handle.close()
        if returncode:
            failures.append({"gpu": gpu, "returncode": returncode, "command": command})
    if failures:
        raise RuntimeError(f"gpt-oss diagnostic failed closed: {json.dumps(failures, indent=2)}")
    atomic_json(status_path, {"status": "running", "stage": "qwen_evaluator_diagnostic_v2",
                              "spent_sealed_split_touched": False,
                              "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
    judge_jobs = []
    for gpu in range(4):
        output = args.run_dir / f"diagnostics/judges/qwen3_32b_max512/shard_{gpu}.jsonl"
        command = [
            args.python, str(args.project / "scripts/revision/flagship/judge_fair_demo_resolution.py"),
            "--input", str(judge_input), "--lock", str(judge_lock), "--judge-id", "qwen3_32b",
            "--model-path", str(args.qwen_judge), "--output", str(output),
            "--shard-index", str(gpu), "--num-shards", "4", "--batch-size", "128",
        ]
        environment = os.environ.copy()
        environment.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(args.project),
                            "HF_HOME": str(args.flagship_root / "cache/huggingface"),
                            "HF_HUB_CACHE": str(args.flagship_root / "cache/huggingface/hub"),
                            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                            "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "TOKENIZERS_PARALLELISM": "false"})
        handle = (logs / f"qwen_diagnostic_v2_shard{gpu}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(command, cwd=args.project, env=environment,
                                   stdout=handle, stderr=subprocess.STDOUT)
        judge_jobs.append((gpu, process, handle, command))
    failures = []
    for gpu, process, handle, command in judge_jobs:
        returncode = process.wait(); handle.close()
        if returncode:
            failures.append({"gpu": gpu, "returncode": returncode, "command": command})
    if failures:
        raise RuntimeError(f"Qwen diagnostic failed closed: {json.dumps(failures, indent=2)}")
    run([args.python, str(args.project / "scripts/revision/flagship/finalize_fair_demo_evaluator.py"),
         "--run-dir", str(args.run_dir)], cwd=args.project, log=logs / "finalize_evaluator.log")
    evaluator_lock = args.run_dir / "evaluator_lock.json"

    atomic_json(status_path, {"status": "running", "stage": "validation_decode_gate",
                              "spent_sealed_split_touched": False,
                              "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
    run([args.python, str(args.project / "scripts/revision/flagship/run_fair_demo_validation_decode_gate.py"),
         "--project", str(args.project), "--python", args.python, "--flagship-root", str(args.flagship_root),
         "--fair-root", str(args.fair_root), "--run-dir", str(args.run_dir), "--grid", str(grid)],
        cwd=args.project, log=logs / "validation_decode_gate.log")
    atomic_json(status_path, {"status": "running", "stage": "validation_locked_scoring",
                              "spent_sealed_split_touched": False,
                              "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
    scoring_common = [
        args.python, str(args.project / "scripts/revision/flagship/run_fair_demo_validation_scoring.py"),
        "--project", str(args.project), "--python", args.python, "--flagship-root", str(args.flagship_root),
        "--fair-root", str(args.fair_root), "--run-dir", str(args.run_dir), "--grid", str(grid),
        "--evaluator-lock", str(evaluator_lock), "--general-rm-cache", str(args.general_rm_cache),
        "--qwen-judge", str(args.qwen_judge), "--gpt-judge", str(args.gpt_judge),
    ]
    run(scoring_common, cwd=args.project, log=logs / "validation_scoring.log")
    selection_lock = args.run_dir / "validation/results/panel/selection_lock.json"

    atomic_json(status_path, {"status": "running", "stage": "fresh_test_preregistration",
                              "spent_sealed_split_touched": False,
                              "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
    fresh_prereg = args.run_dir / "fresh_test_preregistration"
    run([args.python, str(args.project / "scripts/revision/flagship/prepare_fair_demo_fresh_test.py"),
         "--selection-lock", str(selection_lock), "--evaluator-lock", str(evaluator_lock),
         "--avg-precomputed", str(args.flagship_root / "precomputed/avg"),
         "--validation-prompts", str(args.flagship_root / "data/pool_validation.jsonl"),
         "--base-model", str(args.base_model), "--output-dir", str(fresh_prereg),
         "--cache-dir", str(args.flagship_root / "cache/huggingface")],
        cwd=args.project, log=logs / "prepare_fresh_test.log")
    fresh_manifest = fresh_prereg / "fresh_test_manifest.json"
    fresh_prompts = fresh_prereg / "fresh_test_prompts.jsonl"

    atomic_json(status_path, {"status": "running", "stage": "fresh_test_decode_gate",
                              "spent_sealed_split_touched": False,
                              "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
    run([args.python, str(args.project / "scripts/revision/flagship/run_fair_demo_fresh_decode_gate.py"),
         "--project", str(args.project), "--python", args.python, "--flagship-root", str(args.flagship_root),
         "--fair-root", str(args.fair_root), "--run-dir", str(args.run_dir),
         "--selection-lock", str(selection_lock), "--fresh-manifest", str(fresh_manifest),
         "--fresh-prompts", str(fresh_prompts), "--base-model", str(args.base_model)],
        cwd=args.project, log=logs / "fresh_decode_gate.log")
    atomic_json(status_path, {"status": "running", "stage": "fresh_test_locked_scoring",
                              "spent_sealed_split_touched": False,
                              "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
    run([*scoring_common, "--split", "fresh_test"], cwd=args.project, log=logs / "fresh_scoring.log")
    run([args.python, str(args.project / "scripts/revision/flagship/build_fair_demo_report.py"),
         "--run-dir", str(args.run_dir), "--fair-root", str(args.fair_root)],
        cwd=args.project, log=logs / "build_report.log")
    run([args.python, str(args.project / "scripts/revision/flagship/log_fair_demo_results_wandb.py"),
         "--run-dir", str(args.run_dir)], cwd=args.project, log=logs / "wandb_final_eval.log")
    atomic_json(status_path, {"status": "running", "stage": "verified_public_hf_upload",
                              "spent_sealed_split_touched": False,
                              "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
    run([args.python, str(args.project / "scripts/revision/flagship/upload_fair_demo_selected_to_hf.py"),
         "--selection-lock", str(selection_lock), "--fair-root", str(args.fair_root),
         "--run-dir", str(args.run_dir), "--namespace", "promotion"],
        cwd=args.project, log=logs / "hf_upload.log")
    atomic_json(status_path, {"status": "completed", "stage": "report_built",
                              "report": str(args.run_dir / "REPORT.md"),
                              "spent_sealed_split_touched": False,
                              "completed_at": datetime.now().astimezone().isoformat(timespec="seconds")})


if __name__ == "__main__":
    main()
