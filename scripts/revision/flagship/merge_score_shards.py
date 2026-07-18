#!/usr/bin/env python3
"""Merge modulo-sharded reward files, validate coverage, and apply sign transforms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def row_key(row: dict[str, Any]) -> str:
    return str(row.get("prompt_id") or row.get("prompt"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-prompts", type=int, required=True)
    parser.add_argument("--negate", action="store_true")
    parser.add_argument("--objective-name", required=True)
    args = parser.parse_args()

    rows: dict[str, dict[str, Any]] = {}
    for input_name in args.inputs:
        with Path(input_name).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row_key(row)
                if key in rows:
                    raise RuntimeError(f"duplicate prompt across shards: {key}")
                scores = row.get("all_rm_scores")
                responses = row.get("all_generated_responses")
                if not isinstance(scores, list) or not isinstance(responses, list) or len(scores) != len(responses):
                    raise RuntimeError(f"score/response mismatch: {key}")
                if args.negate:
                    row["all_rm_scores"] = [-float(value) for value in scores]
                row["flagship_objective_name"] = args.objective_name
                row["flagship_score_transform"] = "negate" if args.negate else "identity"
                rows[key] = row
    if len(rows) != args.expected_prompts:
        raise RuntimeError(f"expected {args.expected_prompts} prompts, found {len(rows)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for key in sorted(rows):
            handle.write(json.dumps(rows[key], ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "prompts": len(rows), "objective": args.objective_name,
                      "transform": "negate" if args.negate else "identity"}, indent=2))


if __name__ == "__main__":
    main()
