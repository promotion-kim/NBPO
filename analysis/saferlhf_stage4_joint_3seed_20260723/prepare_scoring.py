#!/usr/bin/env python3
"""Validate the lock and split the frozen joint pool by prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if sha256(args.pool) != lock["response_pool_sha256"]:
        raise RuntimeError("response-pool hash differs from run_lock.json")
    rows = [json.loads(line) for line in args.pool.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != lock["records"]:
        raise RuntimeError("response-pool row count mismatch")
    if any(row["response_model_names"] != lock["policy_order"] for row in rows):
        raise RuntimeError("response-pool policy order mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    handles = [
        (args.output / f"shard_{index}.jsonl").open("w", encoding="utf-8")
        for index in range(args.num_shards)
    ]
    try:
        for index, row in enumerate(rows):
            handles[index % args.num_shards].write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        for handle in handles:
            handle.close()
    audit = {
        "pool_sha256": sha256(args.pool),
        "rows": len(rows),
        "responses_per_prompt": len(lock["policy_order"]),
        "shards": {
            path.name: {"rows": sum(1 for _ in path.open()), "sha256": sha256(path)}
            for path in sorted(args.output.glob("shard_*.jsonl"))
        },
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
