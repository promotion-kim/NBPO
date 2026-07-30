#!/usr/bin/env python3
"""Score a locked merged panel with three revision-pinned local reward models."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


REVISIONS = {
    "skywork": "cba2f842f3f1af2f1b2f0d35e794d789976390c5",
    "athene": "cdf428f7b52a323b6cf4e9803e5bcba9f1fb5a59",
    "armo": "eb2676d20da2f2d41082289d23c59b9f7427f955",
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def jsonl_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def merge_shards(merged_input: Path, shard_paths: list[Path], output: Path) -> None:
    source = json.loads(merged_input.read_text(encoding="utf-8"))
    by_prompt = {}
    for shard in shard_paths:
        for row in jsonl_rows(shard):
            key = str(row.get("prompt_id") or row["prompt"])
            if key in by_prompt:
                raise RuntimeError(f"duplicate score prompt across shards: {key}")
            by_prompt[key] = row
    ordered = []
    for row in source:
        key = str(row.get("prompt_id") or row["prompt"])
        if key not in by_prompt:
            raise RuntimeError(f"missing score prompt after shard merge: {key}")
        ordered.append(by_prompt[key])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
                      encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--general-rm-cache", type=Path, required=True)
    parser.add_argument("--armo-cache", type=Path, required=True)
    parser.add_argument("--expected-prompts", type=int, required=True)
    args = parser.parse_args()
    merged = json.loads(args.merged.read_text(encoding="utf-8"))
    if len(merged) != args.expected_prompts:
        raise RuntimeError("merged prompt count differs from the locked split")
    names = merged[0]["response_model_names"]
    if any(row["response_model_names"] != names for row in merged):
        raise RuntimeError("merged model order is inconsistent")
    scores = args.work / "scores"; logs = args.work / "logs"
    scores.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    env_base = os.environ.copy()
    env_base.update({"PYTHONPATH": str(args.project), "HF_HUB_OFFLINE": "1",
                     "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                     "TORCH_CUDNN_SDPA_ENABLED": "0"})
    jobs = []
    commands = [
        ("skywork_s0", "0", [args.python, "-u", "-m", "on_policy_data_gen.rm_skywork",
          "--input_file", str(args.merged), "--output_file", str(scores / "skywork.shard0.jsonl"),
          "--cache_dir", str(args.general_rm_cache), "--model_name", "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
          "--revision", REVISIONS["skywork"], "--local_files_only", "--max_seq_length", "4096",
          "--attn_implementation", "sdpa", "--batch_size", "16", "--sample_batch_size", "4",
          "--num_shards", "2", "--shard_index", "0"]),
        ("athene", "1", [args.python, "-u", "-m", "on_policy_data_gen.rm_athene",
          "--input_file", str(args.merged), "--output_file", str(scores / "athene.jsonl"),
          "--cache_dir", str(args.general_rm_cache), "--model_name", "Nexusflow/Athene-RM-8B",
          "--revision", REVISIONS["athene"], "--local_files_only", "--batch_size", "16",
          "--sample_batch_size", "4"]),
        ("armo", "2", [args.python, "-u", "-m", "on_policy_data_gen.rm_armo",
          "--input_file", str(args.merged), "--output_file", str(scores / "armo.jsonl"),
          "--cache_dir", str(args.armo_cache), "--revision", REVISIONS["armo"],
          "--local_files_only", "--batch_size", "16", "--sample_batch_size", "4",
          "--max_seq_length", "4096"]),
        ("skywork_s1", "3", [args.python, "-u", "-m", "on_policy_data_gen.rm_skywork",
          "--input_file", str(args.merged), "--output_file", str(scores / "skywork.shard1.jsonl"),
          "--cache_dir", str(args.general_rm_cache), "--model_name", "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
          "--revision", REVISIONS["skywork"], "--local_files_only", "--max_seq_length", "4096",
          "--attn_implementation", "sdpa", "--batch_size", "16", "--sample_batch_size", "4",
          "--num_shards", "2", "--shard_index", "1"]),
    ]
    for name, gpu, command in commands:
        expected = args.expected_prompts if name in {"athene", "armo"} else (args.expected_prompts + 1) // 2 if name.endswith("s0") else args.expected_prompts // 2
        output_token = command[command.index("--output_file") + 1]
        if len(jsonl_rows(Path(output_token))) == expected:
            continue
        env = env_base.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu
        handle = (logs / f"{name}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(command, cwd=args.project, env=env, stdout=handle,
                                   stderr=subprocess.STDOUT)
        jobs.append((name, gpu, process, handle, command))
    atomic_json(args.work / "scoring_status.json", {
        "status": "running", "started_at": datetime.now().astimezone().isoformat(),
        "models": names, "jobs": [{"name": row[0], "gpu": row[1], "command": row[4]} for row in jobs],
        "revisions": REVISIONS, "spent_sealed_split_touched": False})
    failures = []
    for name, gpu, process, handle, command in jobs:
        returncode = process.wait(); handle.close()
        if returncode:
            failures.append({"name": name, "gpu": gpu, "returncode": returncode, "command": command})
    if failures:
        atomic_json(args.work / "scoring_status.json", {"status": "failed", "failures": failures,
                    "revisions": REVISIONS, "spent_sealed_split_touched": False})
        raise RuntimeError(json.dumps(failures, indent=2))
    merge_shards(args.merged, [scores / "skywork.shard0.jsonl", scores / "skywork.shard1.jsonl"],
                 scores / "skywork.jsonl")
    for name in ("skywork", "athene", "armo"):
        rows = jsonl_rows(scores / f"{name}.jsonl")
        if len(rows) != args.expected_prompts or any(row["response_model_names"] != names for row in rows):
            raise RuntimeError(f"final {name} output is incomplete or misaligned")
    atomic_json(args.work / "scoring_status.json", {
        "status": "completed", "completed_at": datetime.now().astimezone().isoformat(),
        "models": names, "prompt_count": args.expected_prompts, "revisions": REVISIONS,
        "outputs": {name: str(scores / f"{name}.jsonl") for name in ("skywork", "athene", "armo")},
        "spent_sealed_split_touched": False})
    print(json.dumps({"status": "completed", "prompt_count": args.expected_prompts, "models": names}, indent=2))


if __name__ == "__main__":
    main()
