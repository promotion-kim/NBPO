#!/usr/bin/env python3
"""Merge disjoint RMOD prompt shards and regenerate their summary."""

import argparse
import json
import statistics as st
from pathlib import Path


def read(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--expected-n", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()

    kinds = ("fmt", "helpfulness", "harmlessness")
    merged = {kind: [] for kind in kinds}
    summaries = []
    for index in range(args.shards):
        stem = f"k{args.k}_shard{index}of{args.shards}"
        summaries.append(json.loads((args.root / f"{stem}_summary.json").read_text()))
        for kind in kinds:
            merged[kind].extend(read(args.root / f"{stem}_{kind}.jsonl"))
    if any(len(rows) != args.expected_n for rows in merged.values()):
        raise ValueError({kind: len(rows) for kind, rows in merged.items()})
    prompts = [row["prompt"] for row in merged["fmt"]]
    if len(set(prompts)) != args.expected_n:
        raise ValueError("prompt shards overlap or omit prompts")
    for kind, rows in merged.items():
        with (args.root / f"k{args.k}_{kind}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    helpful = [row["all_rm_scores"][0] for row in merged["helpfulness"]]
    harmless = [row["all_rm_scores"][0] for row in merged["harmlessness"]]
    checkpoints = {row["value_function_checkpoint"] for row in summaries}
    if len(checkpoints) != 1:
        raise ValueError(f"checkpoint mismatch: {checkpoints}")
    record = {
        "k": args.k, "block_size": args.block_size, "helpful": st.mean(helpful),
        "harmless": st.mean(harmless), "n": args.expected_n, "chat_template": True,
        "shard_count": args.shards, "value_function_checkpoint": checkpoints.pop(),
    }
    (args.root / f"k{args.k}_summary.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
