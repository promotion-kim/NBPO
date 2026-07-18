#!/usr/bin/env python3
"""Deterministically shard score JSONL by prompt id and merge scored shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split(args: argparse.Namespace) -> None:
    rows = sorted(read_jsonl(args.input), key=lambda row: str(row["prompt_id"]))
    if len(rows) != args.expected_records:
        raise RuntimeError(f"expected {args.expected_records}, got {len(rows)}")
    for shard in range(args.num_shards):
        selected = [row for index, row in enumerate(rows) if index % args.num_shards == shard]
        path = args.output_dir / f"shard_{shard}.jsonl"
        write_jsonl(path, selected)
        print(json.dumps({"path": str(path), "rows": len(selected), "sha256": sha(path)}))


def merge(args: argparse.Namespace) -> None:
    rows = []
    for path in args.inputs:
        rows.extend(read_jsonl(path))
    rows.sort(key=lambda row: str(row["prompt_id"]))
    prompt_ids = [str(row["prompt_id"]) for row in rows]
    if len(rows) != args.expected_records or len(set(prompt_ids)) != len(rows):
        raise RuntimeError(f"invalid merge: rows={len(rows)} unique={len(set(prompt_ids))}")
    for row in rows:
        scores = [float(value) for value in row["all_rm_scores"]]
        if len(scores) != args.expected_scores_per_row or not all(math.isfinite(value) for value in scores):
            raise RuntimeError(f"invalid scores for {row['prompt_id']}")
        if args.strip_responses:
            row.pop("all_generated_responses", None)
    write_jsonl(args.output, rows)
    payload = {
        "status": "complete", "records": len(rows),
        "scores_per_record": args.expected_scores_per_row,
        "responses_stripped": args.strip_responses,
        "sha256": sha(args.output), "inputs": [str(path) for path in args.inputs],
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)
    split_parser = subs.add_parser("split")
    split_parser.add_argument("--input", type=Path, required=True)
    split_parser.add_argument("--output-dir", type=Path, required=True)
    split_parser.add_argument("--num-shards", type=int, default=2)
    split_parser.add_argument("--expected-records", type=int, default=770)
    split_parser.set_defaults(func=split)
    merge_parser = subs.add_parser("merge")
    merge_parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--audit", type=Path, required=True)
    merge_parser.add_argument("--expected-records", type=int, default=770)
    merge_parser.add_argument("--expected-scores-per-row", type=int, default=4)
    merge_parser.add_argument("--strip-responses", action="store_true")
    merge_parser.set_defaults(func=merge)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
