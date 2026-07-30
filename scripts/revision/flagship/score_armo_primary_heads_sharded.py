#!/usr/bin/env python3
"""Run frozen ArmoRM primary-head scoring over deterministic prompt shards."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


OBJECTIVES = ("helpfulness", "safety", "conciseness")


def complete(output: Path, expected: int) -> bool:
    try:
        return all(sum(1 for line in (output / f"{name}.jsonl").open() if line.strip()) == expected
                   for name in OBJECTIVES)
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--gpu-ids", nargs="+", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    records = json.loads(args.input_file.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("input must be a non-empty JSON list")
    if complete(args.output_dir, len(records)):
        print(f"skip complete sharded Armo scores: {args.output_dir}", flush=True)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_root = args.output_dir / "shards"
    shard_root.mkdir(exist_ok=True)
    shard_size = (len(records) + len(args.gpu_ids) - 1) // len(args.gpu_ids)
    processes = []
    active = []
    scorer = Path(__file__).with_name("score_armo_primary_heads.py")
    for shard_index, gpu in enumerate(args.gpu_ids):
        start = shard_index * shard_size
        end = min(len(records), start + shard_size)
        if start >= end:
            continue
        shard_input = shard_root / f"input_{shard_index}.json"
        shard_output = shard_root / f"output_{shard_index}"
        shard_input.write_text(json.dumps(records[start:end], ensure_ascii=False) + "\n")
        command = [
            args.python, str(scorer), "--input-file", str(shard_input),
            "--output-dir", str(shard_output), "--cache-dir", args.cache_dir,
            "--batch-size", str(args.batch_size),
            "--sample-batch-size", str(args.sample_batch_size),
            "--max-seq-length", str(args.max_seq_length),
        ]
        if args.local_files_only:
            command.append("--local-files-only")
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "TORCH_CUDNN_SDPA_ENABLED": "0",
                    "TOKENIZERS_PARALLELISM": "false"})
        log = (shard_root / f"score_{shard_index}.log").open("a")
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append((shard_index, process, log, shard_output, end - start))
        active.append({"shard": shard_index, "gpu": gpu, "start": start, "end": end,
                       "pid": process.pid})
    (args.output_dir / "shard_manifest.json").write_text(json.dumps({
        "input_file": str(args.input_file), "num_records": len(records),
        "gpu_ids": args.gpu_ids, "shards": active,
    }, indent=2) + "\n")

    failures = []
    for shard_index, process, log, shard_output, expected in processes:
        rc = process.wait()
        log.close()
        if rc != 0 or not complete(shard_output, expected):
            failures.append({"shard": shard_index, "returncode": rc,
                             "output": str(shard_output), "expected": expected})
    if failures:
        raise RuntimeError(f"Armo scoring shard failures: {failures}")

    for objective in OBJECTIVES:
        with (args.output_dir / f"{objective}.jsonl").open("w", encoding="utf-8") as target:
            for shard_index, _, _, shard_output, _ in processes:
                with (shard_output / f"{objective}.jsonl").open("r", encoding="utf-8") as source:
                    for line in source:
                        if line.strip():
                            target.write(line)
    first_metadata = json.loads((processes[0][3] / "score_metadata.json").read_text())
    first_metadata.update({
        "num_prompts": len(records), "sharded": True,
        "num_shards": len(processes), "gpu_ids": args.gpu_ids,
        "shard_manifest": str(args.output_dir / "shard_manifest.json"),
    })
    (args.output_dir / "score_metadata.json").write_text(
        json.dumps(first_metadata, indent=2) + "\n"
    )
    if not complete(args.output_dir, len(records)):
        raise RuntimeError("merged Armo score count mismatch")
    print(f"merged {len(records)} prompt scores from {len(processes)} shards", flush=True)


if __name__ == "__main__":
    main()
