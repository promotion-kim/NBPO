#!/usr/bin/env python3
"""Decode and gate every frozen OS checkpoint on the full 647 prompts, reward-blind."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def complete(path: Path, expected: int = 647) -> bool:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(rows, list) and len(rows) == expected and all(
        str(row.get("prompt", "")).strip() and str(row.get("generated_text", "")).strip()
        for row in rows
    )


def gpu_snapshot() -> dict:
    gpus = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    return {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "gpus": gpus, "compute_processes": [row for row in processes if row.strip()]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed647", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--hf-cache", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_validation_decode_and_ranking":
        raise RuntimeError("checkpoint manifest is not frozen")
    models = manifest["models"]
    if len(models) != 36 or sorted({int(row["step"]) for row in models}) != list(range(100, 901, 100)):
        raise RuntimeError("expected four OS candidates with all nine saved checkpoints")
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if gpu_ids != ["0", "1", "2", "3"]:
        raise RuntimeError("the preregistered authorized B200 set is GPUs 0,1,2,3")
    # Completed-candidate predecode jobs launched by this run may still be
    # draining while the slowest training recipe finishes. Waiting is safe;
    # attaching to or stopping any process is not. The required three idle
    # samples are taken only after the whole authorized pool is clear.
    while True:
        waiting = gpu_snapshot()
        if not waiting["compute_processes"]:
            break
        atomic_json(args.work / "waiting_for_idle.json", {
            "status": "waiting_without_modifying_processes", "snapshot": waiting,
            "spent_sealed_split_touched": False,
        })
        time.sleep(20)
    samples = []
    for index in range(3):
        sample = gpu_snapshot(); samples.append(sample)
        if sample["compute_processes"]:
            raise RuntimeError(f"GPU became busy during idle verification: {sample['compute_processes']}")
        if index < 2:
            time.sleep(2)
    atomic_json(args.work / "prelaunch_gpu_samples.json", {
        "status": "verified_idle", "samples": samples, "authorized_gpu_ids": [0, 1, 2, 3],
        "spent_sealed_split_touched": False,
    })

    generations = args.work / "generations_4096"
    logs = args.work / "logs"; logs.mkdir(parents=True, exist_ok=True)
    entries = [{"model_id": "base", "model_path": args.base_model, "candidate_id": "base", "step": 0}, *models]
    queued = [row for row in entries if not complete(generations / row["model_id"] / "output_42.json")]
    running: dict[str, tuple] = {}
    env_base = os.environ.copy()
    env_base.update({
        "PYTHONPATH": str(args.project), "HF_HOME": str(args.hf_cache.parent),
        "HF_HUB_CACHE": str(args.hf_cache), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "TORCH_CUDNN_SDPA_ENABLED": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    })
    failures = []
    while queued or running:
        for gpu in gpu_ids:
            if not queued or gpu in running:
                continue
            row = queued.pop(0)
            out = generations / row["model_id"]; out.mkdir(parents=True, exist_ok=True)
            command = [
                args.python, "-u", str(args.project / "scripts/revision/flagship/decode_vllm_non_thinking.py"),
                "--data-dir", str(args.fixed647), "--model", row["model_path"],
                "--output-dir", str(out), "--seed", "42", "--temperature", "0.7",
                "--top-p", "0.9", "--max-new-tokens", "4096", "--max-model-len", "8192",
                "--gpu-memory-utilization", "0.88",
            ]
            env = env_base.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu
            handle = (logs / f"decode_{row['model_id']}.log").open("a", encoding="utf-8")
            process = subprocess.Popen(command, cwd=args.project, env=env,
                                       stdout=handle, stderr=subprocess.STDOUT)
            running[gpu] = (row, process, handle, command)
        atomic_json(args.work / "decode_status.json", {
            "status": "running", "queued": [row["model_id"] for row in queued],
            "running": [{"gpu": gpu, "model_id": value[0]["model_id"], "pid": value[1].pid}
                        for gpu, value in running.items()],
            "failures": failures, "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "spent_sealed_split_touched": False,
        })
        if not running:
            continue
        time.sleep(5)
        for gpu, (row, process, handle, command) in list(running.items()):
            rc = process.poll()
            if rc is None:
                continue
            handle.close()
            output = generations / row["model_id"] / "output_42.json"
            if rc or not complete(output):
                failures.append({"model_id": row["model_id"], "returncode": rc, "command": command})
            del running[gpu]
        if failures:
            atomic_json(args.work / "decode_status.json", {"status": "failed", "failures": failures,
                        "spent_sealed_split_touched": False})
            raise RuntimeError(json.dumps(failures, indent=2))

    base_file = generations / "base/output_42.json"
    gate_dir = args.work / "gates"; gate_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in models:
        output = gate_dir / f"{row['model_id']}.json"
        command = [
            args.python, str(args.project / "scripts/revision/flagship/stability_gate_corrected.py"),
            "--base", str(base_file), "--candidate", str(generations / row["model_id"] / "output_42.json"),
            "--output", str(output), "--expected-records", "647", "--min-length-ratio", "0.33",
            "--max-length-ratio", "2.0", "--max-repeat-run", "20",
        ]
        result = subprocess.run(command, cwd=args.project, capture_output=True, text=True)
        (logs / f"gate_{row['model_id']}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode not in {0, 4} or not output.is_file():
            raise RuntimeError(f"gate detector failed for {row['model_id']}: rc={result.returncode}")
        payload = json.loads(output.read_text(encoding="utf-8"))
        rows.append({"model_id": row["model_id"], "candidate_id": row["candidate_id"],
                     "step": row["step"], "passed": payload.get("passed") is True,
                     "status": payload.get("status"), "checks": payload.get("checks"),
                     "candidate": payload.get("candidate"), "model_path": row["model_path"],
                     "candidate_base_mean_word_ratio": payload.get("candidate_base_mean_word_ratio"),
                     "wandb_run_id": row.get("wandb_run_id")})
    by_candidate = {}
    for row in rows:
        by_candidate.setdefault(row["candidate_id"], []).append(row)
    profiles = []
    robust_ids = []
    for candidate_id, values in sorted(by_candidate.items()):
        passed_steps = {int(row["step"]) for row in values if row["passed"]}
        for row in values:
            step = int(row["step"])
            neighbors = [value for value in (step - 100, step + 100) if 100 <= value <= 900]
            row["neighbor_steps"] = neighbors
            row["robust_neighbor_pass"] = bool(neighbors) and all(value in passed_steps for value in neighbors)
            if row["passed"] and row["robust_neighbor_pass"]:
                robust_ids.append(row["model_id"])
        profiles.append({"candidate_id": candidate_id, "passed_steps": sorted(passed_steps),
                         "failed_steps": sorted(set(range(100, 901, 100)) - passed_steps),
                         "robust_pass_steps": sorted(int(row["step"]) for row in values
                                                     if row["passed"] and row["robust_neighbor_pass"])})
    summary = {
        "status": "completed_fail_closed_before_reward_scoring",
        "detector": "corrected_nonempty_paired_span_v1", "prompt_count": 647,
        "decode": {"seed": 42, "temperature": 0.7, "top_p": 0.9,
                   "max_new_tokens": 4096, "enable_thinking": False},
        "thresholds": {"records": 647, "empty": 0, "think_leaks": 0,
                       "mean_word_ratio": [0.33, 2.0], "max_repeat_run": 20},
        "rows": rows, "profiles": profiles,
        "eligible_model_ids": [row["model_id"] for row in rows if row["passed"]],
        "robust_neighbor_model_ids": robust_ids,
        "failed_model_ids": [row["model_id"] for row in rows if not row["passed"]],
        "reward_scores_consulted": False, "spent_sealed_split_touched": False,
    }
    atomic_json(gate_dir / "summary.json", summary)
    atomic_json(args.work / "decode_status.json", {
        "status": "completed", "model_count": len(entries), "candidate_checkpoint_count": len(models),
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "spent_sealed_split_touched": False,
    })
    print(json.dumps({"status": summary["status"], "profiles": profiles,
                      "eligible_count": len(summary["eligible_model_ids"]),
                      "robust_count": len(robust_ids)}, indent=2))


if __name__ == "__main__":
    main()
