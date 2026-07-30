#!/usr/bin/env python3
"""Score preserved generations with the pre-committed Skywork/Athene set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


EXPECTED_MODELS = [
    "base", "ronpo_k_only", "ronpo_full_expect", "ipo", "simpo",
    "sppo_avg", "inpo_avg", "ht_mnpo_helpfulness", "ht_mnpo_safety",
    "ht_mnpo_conciseness",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text())
    if lock.get("status") != "locked_before_scoring":
        raise RuntimeError("reward-model set is not locked")
    if lock.get("models") != EXPECTED_MODELS:
        raise RuntimeError("locked model order differs from the score runner")
    records = json.loads(args.input_file.read_text())
    if len(records) != 604:
        raise RuntimeError(f"expected 604 preserved prompts, found {len(records)}")
    for row in records:
        if row.get("response_model_names") != EXPECTED_MODELS:
            raise RuntimeError("preserved response order differs from locked model order")
        if len(row.get("all_generated_responses", [])) != len(EXPECTED_MODELS):
            raise RuntimeError("preserved response count mismatch")

    committed = {row["name"]: row for row in lock["reward_models"]}
    jobs = [
        ("skywork", 0, 0, committed["Skywork/Skywork-Reward-V2-Llama-3.1-8B"]),
        ("skywork", 1, 1, committed["Skywork/Skywork-Reward-V2-Llama-3.1-8B"]),
        ("athene", 0, 2, committed["Nexusflow/Athene-RM-8B"]),
        ("athene", 1, 3, committed["Nexusflow/Athene-RM-8B"]),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = args.output_dir / "logs"
    logs.mkdir(exist_ok=True)
    status = args.output_dir / "score_status.json"
    processes = []
    manifest_jobs = []
    for reward_name, shard, gpu, rm in jobs:
        output = args.output_dir / "scores" / reward_name / f"shard_{shard}.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        if count_lines(output) == 302:
            manifest_jobs.append({"reward_model": reward_name, "shard": shard, "gpu": gpu,
                                  "status": "reused_complete", "output": str(output)})
            continue
        module = f"on_policy_data_gen.rm_{reward_name}"
        command = [
            args.python, "-u", "-m", module,
            "--input_file", str(args.input_file),
            "--output_file", str(output),
            "--cache_dir", str(args.cache_dir),
            "--model_name", rm["name"],
            "--revision", rm["revision"],
            "--local_files_only",
            "--batch_size", "16",
            "--sample_batch_size", "16",
            "--num_shards", "2",
            "--shard_index", str(shard),
        ]
        if reward_name == "skywork":
            command.extend(["--max_seq_length", "4096", "--attn_implementation", "sdpa"])
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONPATH": str(args.project),
            "HF_HOME": str(args.cache_dir),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TORCH_CUDNN_SDPA_ENABLED": "0",
        })
        log_handle = (logs / f"{reward_name}_shard{shard}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(command, cwd=args.project, env=environment,
                                   stdout=log_handle, stderr=subprocess.STDOUT)
        processes.append((reward_name, shard, gpu, process, log_handle, output, command))
        manifest_jobs.append({"reward_model": reward_name, "shard": shard, "gpu": gpu,
                              "status": "running", "pid": process.pid,
                              "output": str(output), "command": command})

    manifest = {
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "lock": str(args.lock),
        "lock_sha256": sha256(args.lock),
        "input_file": str(args.input_file),
        "input_sha256": sha256(args.input_file),
        "num_prompts": len(records),
        "new_decode_invoked": False,
        "jobs": manifest_jobs,
    }
    (args.output_dir / "score_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    status.write_text(json.dumps({"status": "running", "stage": "committed_rm_scoring",
                                  "updated_at": manifest["started_at"]}, indent=2) + "\n")

    failures = []
    for reward_name, shard, gpu, process, handle, output, command in processes:
        returncode = process.wait()
        handle.close()
        measured = count_lines(output)
        if returncode != 0 or measured != 302:
            failures.append({"reward_model": reward_name, "shard": shard, "gpu": gpu,
                             "returncode": returncode, "records": measured,
                             "output": str(output), "command": command})
    completed = datetime.now().astimezone().isoformat(timespec="seconds")
    if failures:
        status.write_text(json.dumps({"status": "failed", "stage": "committed_rm_scoring",
                                      "failures": failures, "updated_at": completed}, indent=2) + "\n")
        raise RuntimeError(f"committed reward scoring failed: {failures}")
    for reward_name in ("skywork", "athene"):
        for shard in (0, 1):
            output = args.output_dir / "scores" / reward_name / f"shard_{shard}.jsonl"
            if count_lines(output) != 302:
                raise RuntimeError(f"incomplete output after scoring: {output}")
    status.write_text(json.dumps({"status": "completed", "stage": "committed_rm_scoring",
                                  "new_decode_invoked": False, "updated_at": completed}, indent=2) + "\n")


if __name__ == "__main__":
    main()
