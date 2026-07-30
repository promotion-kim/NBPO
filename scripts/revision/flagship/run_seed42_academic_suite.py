#!/usr/bin/env python3
"""Run the amended, KTO-free seed-42 academic suite on four B200 GPUs.

The queue waits for the separate deterministic IFEval queue to finish, copies its
measured JSON artifacts, and then evaluates every exact HF commit under the
frozen task/n-shot protocol. P1 sealed prompts are never read by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


METHODS = (
    "base", "ronpo_full_expect", "ronpo_k_only", "dpo", "ipo", "simpo", "sppo_avg",
    "inpo_avg", "ht_mnpo_helpfulness", "ht_mnpo_safety",
    "ht_mnpo_conciseness",
)

GROUPS = (
    # GPQA is intentionally absent from the runnable queue: the official
    # Idavidrein/gpqa dataset is gated and the current HF account has no access.
    # Keep that benchmark BLOCKED in provenance instead of silently using a
    # mirror or letting it prevent the other independent 0-shot evaluations.
    ("n0_truthfulqa", 0, ("truthfulqa_mc2",)),
    ("n0_aime24", 0, ("aime24",)),
    ("n0_humaneval", 0, ("humaneval",)),
    ("n5", 5, ("mmlu", "mmlu_pro", "winogrande")),
    ("n10", 10, ("hellaswag",)),
    ("n25", 25, ("arc_challenge",)),
    ("n8", 8, ("gsm8k_cot",)),
    ("n4", 4, ("minerva_math",)),
)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def models_from_ledger(root: Path, base_revision: str) -> dict[str, dict]:
    models = {
        "base": {
            "name": "base", "method": "base", "seed": "",
            "repo_id": "Qwen/Qwen3-8B", "revision": base_revision,
        }
    }
    ledger = root / "hf_uploads.jsonl"
    if not ledger.exists():
        return models
    for line in ledger.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        method = str(row.get("method"))
        if method not in METHODS or str(row.get("seed")) != "42" or row.get("verified") is not True:
            continue
        commit_url = str(row.get("upload_commit", ""))
        revision = commit_url.rsplit("/", 1)[-1] if "/commit/" in commit_url else ""
        if not revision:
            continue
        models[method] = {
            "name": method, "method": method, "seed": "42",
            "repo_id": row["repo_id"], "revision": revision,
            "upload_commit": commit_url,
        }
    return models


def write_manifest(work: Path, models: dict[str, dict]) -> None:
    path = work / "models.tsv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("name", "model", "method", "seed", "revision", "upload_commit"),
            delimiter="\t",
        )
        writer.writeheader()
        for method in METHODS:
            if method not in models:
                continue
            row = models[method]
            writer.writerow({
                "name": method, "model": row["repo_id"], "method": method,
                "seed": row["seed"], "revision": row["revision"],
                "upload_commit": row.get("upload_commit", ""),
            })


def payload_has_tasks(payload: dict, tasks: tuple[str, ...]) -> bool:
    results = payload.get("results", {})
    groups = payload.get("groups", {})
    return all(task in results or task in groups for task in tasks)


def result_complete(output: Path, tasks: tuple[str, ...]) -> bool:
    if not output.exists():
        return False
    for path in output.rglob("results_*.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload_has_tasks(payload, tasks):
            return True
    return False


def ifeval_result_complete(output: Path) -> bool:
    if not output.exists():
        return False
    for path in output.rglob("results_*.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload.get("results", {}).get("ifeval"), dict):
            return True
    return False


def copy_ifeval_results(ifeval_work: Path, academic_work: Path) -> dict[str, str]:
    sources = {}
    for method in METHODS:
        candidates = []
        for path in (ifeval_work / "raw" / method).rglob("results_*.json"):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if "ifeval" in payload.get("results", {}):
                candidates.append(path)
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one IFEval JSON for {method}, found {len(candidates)}")
        source = candidates[0]
        target_dir = academic_work / "raw" / method / "ifeval_reused"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        shutil.copy2(source, target)
        sources[method] = str(source)
    atomic_json(academic_work / "ifeval_source_map.json", {
        "policy": "Reused exact deterministic IFEval JSON; no re-evaluation or number re-roll.",
        "sources": sources,
        "copied_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    return sources


def run_identifier(method: str, group: str) -> str:
    # lm-eval parses an all-digit W&B id as an integer; W&B requires a string.
    # A fixed alphabetic prefix keeps every deterministic id string-typed.
    return "r" + hashlib.sha256(
        f"aaai27-p2-seed42-v2|{method}|{group}".encode()
    ).hexdigest()[:11]


def command(args: argparse.Namespace, model: dict, group: str, shots: int,
            tasks: tuple[str, ...], output: Path) -> list[str]:
    model_args = ",".join((
        f"pretrained={model['repo_id']}", "dtype=bfloat16",
        "gpu_memory_utilization=0.88", "max_model_len=32768",
        "enable_thinking=False", f"revision={model['revision']}",
    ))
    identifier = run_identifier(model["method"], group)
    return [
        args.python, "-m", "lm_eval", "run", "--model", "vllm",
        "--model_args", model_args, "--tasks", *tasks,
        "--num_fewshot", str(shots), "--batch_size", "auto",
        "--max_batch_size", "256", "--apply_chat_template",
        "--seed", "42,42,42,42", "--cache_requests", "true",
        "--show_config", "--output_path", str(output),
        "--confirm_run_unsafe_code",
        "--wandb_args", "entity=promotion-kim", "project=mnpo",
        f"id={identifier}", f"name=aaai27-p2-{model['method']}-{group}-s42",
        "group=flagship_p2_seed42", "job_type=lm_eval", "resume=allow",
        "--wandb_config_args", "flagship_stage=P2_academic_seed42",
        f"model_name={model['method']}", f"task_group={group}",
        f"num_fewshot={shots}", "enable_thinking=False",
    ]


def write_status(path: Path, state: str, **fields: object) -> None:
    atomic_json(path, {
        "status": state,
        **fields,
        "p1_sealed_test_opened": False,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--ifeval-work", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--stop-at", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / "logs").mkdir(exist_ok=True)
    protocol_payload = json.loads(args.protocol.read_text())
    inherited = protocol_payload.get("inherits_frozen_task_protocol")
    if inherited:
        frozen_protocol = args.project / inherited
        if not frozen_protocol.is_file():
            raise FileNotFoundError(f"inherited frozen protocol not found: {frozen_protocol}")
        shutil.copy2(frozen_protocol, args.work / "p2_protocol.json")
        shutil.copy2(args.protocol, args.work / "p2_scope_amendment.json")
    else:
        shutil.copy2(args.protocol, args.work / "p2_protocol.json")
    status_path = args.work / "status.json"
    stop_at = datetime.fromisoformat(args.stop_at).timestamp()

    while time.time() < stop_at:
        ifeval_status_path = args.ifeval_work / "status.json"
        if ifeval_status_path.exists():
            ifeval_status = json.loads(ifeval_status_path.read_text())
            if ifeval_status.get("status") == "completed":
                missing_ifeval = [
                    method for method in METHODS
                    if not ifeval_result_complete(args.ifeval_work / "raw" / method)
                ]
                if not missing_ifeval:
                    break
                write_status(
                    status_path, "waiting", stage="ifeval_scope_completion",
                    missing_ifeval=missing_ifeval,
                )
                time.sleep(args.poll_seconds)
                continue
            if ifeval_status.get("status") in {"completed_with_failures", "deadline_reached"}:
                write_status(status_path, "blocked", reason="IFEval queue did not complete", ifeval_status=ifeval_status)
                return
        write_status(status_path, "waiting", stage="ifeval_completion")
        time.sleep(args.poll_seconds)
    else:
        write_status(status_path, "deadline_reached", stage="ifeval_completion")
        return

    while time.time() < stop_at:
        models = models_from_ledger(args.root, args.base_revision)
        missing_models = [method for method in METHODS if method not in models]
        write_manifest(args.work, models)
        if not missing_models:
            break
        write_status(status_path, "waiting", stage="verified_hf_models", missing_models=missing_models)
        time.sleep(args.poll_seconds)
    else:
        write_status(status_path, "deadline_reached", stage="verified_hf_models")
        return

    validation_status_path = args.root / "eval/p1_validation_reward_seed42/status.json"
    while time.time() < stop_at:
        if validation_status_path.exists():
            validation_status = json.loads(validation_status_path.read_text())
            if validation_status.get("status") == "completed":
                break
            if validation_status.get("status") in {"failed", "deadline_reached"}:
                write_status(
                    status_path, "blocked", stage="validation_model_selection",
                    validation_status=validation_status,
                )
                return
        # The complete measured IFEval table can independently make the P3
        # sweep mandatory.  In that case the academic baselines may safely
        # start on GPUs 1/3 while reward validation and the sweep reserve 0/2;
        # no sealed data or model-selection result is needed for this branch.
        ifeval_table = args.ifeval_work / "results/ifeval_seed42.json"
        if ifeval_table.exists():
            try:
                payload = json.loads(ifeval_table.read_text())
                rows = {row["method"]: float(row["ifeval_prompt_strict_percent"])
                        for row in payload.get("rows", [])}
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                payload, rows = {}, {}
            if payload.get("complete") is True and all(method in rows for method in METHODS):
                ronpo_value = rows["ronpo_full_expect"]
                ronpo_rank = 1 + sum(value > ronpo_value + 1e-12 for value in rows.values())
                if ronpo_rank > 2 or ronpo_value + 1e-12 < rows["base"]:
                    validation_status = {
                        "status": "measured_ifeval_trigger",
                        "needs_seed42_sweep": True,
                        "ronpo_ifeval_rank": ronpo_rank,
                        "ronpo_ifeval_percent": ronpo_value,
                        "base_ifeval_percent": rows["base"],
                    }
                    break
        write_status(status_path, "waiting", stage="validation_model_selection")
        time.sleep(args.poll_seconds)
    else:
        write_status(status_path, "deadline_reached", stage="validation_model_selection")
        return

    # If validation calls for a seed-42 RONPO sweep, reserve GPUs 0/2 for that
    # P1-critical work and keep the cost-free academic queue progressing on 1/3.
    # Otherwise expand the academic queue to all four approved GPUs.
    gpu_ids = (1, 3) if validation_status.get("needs_seed42_sweep") else (0, 1, 2, 3)

    copy_ifeval_results(args.ifeval_work, args.work)
    jobs = [(method, group, shots, tasks) for method in METHODS for group, shots, tasks in GROUPS]
    pending = []
    completed = []
    for method, group, shots, tasks in jobs:
        output = args.work / "raw" / method / group
        item = {"method": method, "group": group, "shots": shots, "tasks": list(tasks)}
        if result_complete(output, tasks):
            completed.append(item | {"note": "pre-existing"})
        else:
            pending.append((method, group, shots, tasks))

    running: dict[int, tuple[tuple, subprocess.Popen, object, int]] = {}
    attempts: dict[tuple[str, str], int] = {}
    failed = []
    final_override_applied = False
    while (pending or running) and time.time() < stop_at:
        final_selection_path = args.root / "eval/p1_validation_reward_seed42/final_model_selection.json"
        sealed_status_path = args.root / "eval/p1_sealed_reward_seed42/status.json"
        sealed_completed = False
        if sealed_status_path.exists():
            sealed_completed = json.loads(sealed_status_path.read_text()).get("status") == "completed"
        drain_for_sealed = final_selection_path.exists() and not sealed_completed
        correction_status_path = args.root / "eval/p2_final_selected_correction/status.json"
        correction = json.loads(correction_status_path.read_text()) if correction_status_path.exists() else {}
        drain_for_correction = (
            sealed_completed and final_selection_path.exists()
            and correction.get("status") != "completed"
        )
        if sealed_completed:
            gpu_ids = (0, 1, 2, 3)
        if sealed_completed and correction.get("status") == "completed" and not final_override_applied:
            override_path = args.root / "eval/p2_final_model_override.json"
            if not override_path.exists():
                raise RuntimeError("P2 final model override manifest missing")
            override = json.loads(override_path.read_text())
            models["ronpo_full_expect"] = override["ronpo_full_expect"]
            write_manifest(args.work, models)
            running_jobs = {(value[0][0], value[0][1]) for value in running.values()}
            pending_jobs = {(value[0], value[1]) for value in pending}
            for group, shots, tasks in GROUPS:
                key = ("ronpo_full_expect", group)
                output = args.work / "raw/ronpo_full_expect" / group
                if not result_complete(output, tasks) and key not in running_jobs and key not in pending_jobs:
                    pending.append(("ronpo_full_expect", group, shots, tasks))
            final_override_applied = True
        for gpu, (job, process, handle, attempt) in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            method, group, shots, tasks = job
            output = args.work / "raw" / method / group
            item = {"method": method, "group": group, "shots": shots, "tasks": list(tasks)}
            if returncode == 0 and result_complete(output, tasks):
                completed.append(item | {"attempt": attempt})
            elif attempt < 3:
                pending.insert(0, job)
            else:
                failed.append(item | {"attempt": attempt, "returncode": returncode})
            del running[gpu]

        if not drain_for_sealed and not drain_for_correction:
            for gpu in gpu_ids:
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
            status_path, "waiting" if (drain_for_sealed or drain_for_correction) and not running else "running",
            stage=("sealed_p1_completion" if drain_for_sealed else
                   "final_selected_p2_correction" if drain_for_correction else "matched_academic_suite"),
            gpu_ids=list(gpu_ids),
            pending=[{"method": m, "group": g} for m, g, _, _ in pending],
            running=[{
                "gpu": gpu, "method": job[0], "group": job[1],
                "pid": process.pid, "attempt": attempt,
            } for gpu, (job, process, _, attempt) in running.items()],
            completed=completed, failed=failed,
        )
        time.sleep(args.poll_seconds)

    if running or pending:
        write_status(status_path, "deadline_reached", pending_count=len(pending), running_count=len(running), failed=failed)
        return

    aggregate_log = args.work / "logs" / "aggregate.log"
    with aggregate_log.open("a", encoding="utf-8") as handle:
        completed_process = subprocess.run([
            args.python, str(args.project / "scripts/revision/flagship/aggregate_p2_results.py"),
            "--root", str(args.work), "--protocol", str(args.work / "p2_protocol.json"),
            "--output-dir", str(args.work / "results"),
            "--blockers", str(args.project / "results/ronpo_flagship_20260712/p2_access_blockers_20260713.json"),
        ], cwd=args.project, stdout=handle, stderr=subprocess.STDOUT, check=False)
    final = "completed" if completed_process.returncode == 0 and not failed else "completed_with_failures"
    write_status(status_path, final, stage="measured_tables", completed=completed, failed=failed,
                 aggregate_returncode=completed_process.returncode)


if __name__ == "__main__":
    main()
