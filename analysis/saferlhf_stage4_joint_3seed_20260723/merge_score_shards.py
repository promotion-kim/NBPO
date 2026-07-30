#!/usr/bin/env python3
"""Merge score shards into the frozen prompt order and validate completeness."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pool = [json.loads(line) for line in args.pool.read_text(encoding="utf-8").splitlines() if line.strip()]
    order = [row["prompt_id"] for row in pool]
    names = pool[0]["response_model_names"]
    rows = []
    for path in args.inputs:
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    by_id = {row["prompt_id"]: row for row in rows}
    if len(rows) != len(order) or len(by_id) != len(order) or set(by_id) != set(order):
        raise RuntimeError("score shard prompt identity/count mismatch")
    merged = [by_id[prompt_id] for prompt_id in order]
    for row in merged:
        values = row["all_rm_scores"]
        if row["response_model_names"] != names or len(values) != len(names):
            raise RuntimeError("score policy order/width mismatch")
        if not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError("non-finite score")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
