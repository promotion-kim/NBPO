#!/usr/bin/env python3
"""Fail-closed structural audit for a completed P10 Stage-2 pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--expected-prompts", type=int, default=2500)
    args = parser.parse_args()
    pool = args.pool
    required = {
        "prepared": pool / "PREPARED",
        "response_pool": pool / "response_pool.jsonl",
        "helpfulness": pool / "scores" / "helpfulness.jsonl",
        "harmlessness": pool / "scores" / "harmlessness.jsonl",
        "pairs_train": pool / "pairs_train.jsonl",
        "pairs_test": pool / "pairs_test.jsonl",
        "targets": pool / "precompute" / "targets" / "dataset_dict.json",
        "pool_audit": pool / "pool_audit.json",
    }
    missing = [name for name, path in required.items() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"missing or empty pool artifacts: {missing}")
    # A marker from an earlier pool attempt must not authorize a training run
    # while a newer attempt is still rewriting any required input.
    marker_mtime_ns = required["prepared"].stat().st_mtime_ns
    newer_than_marker = [
        name
        for name, path in required.items()
        if name != "prepared" and path.stat().st_mtime_ns > marker_mtime_ns
    ]
    if newer_than_marker:
        raise RuntimeError(
            "PREPARED is older than required pool artifacts: "
            + ", ".join(newer_than_marker)
        )
    counts = {name: count_jsonl(required[name]) for name in ("response_pool", "helpfulness", "harmlessness")}
    if any(count != args.expected_prompts for count in counts.values()):
        raise RuntimeError(f"unexpected prompt counts: {counts}")
    audit = json.loads(required["pool_audit"].read_text(encoding="utf-8"))
    expected = audit.get("expected_records", audit.get("expected_prompts"))
    if expected is not None and int(expected) != args.expected_prompts:
        raise RuntimeError(f"pool audit expected {expected}, not {args.expected_prompts}")
    payload = {
        "status": "completed",
        "expected_prompts": args.expected_prompts,
        "counts": counts,
        "prepared_mtime_ns": marker_mtime_ns,
        "artifact_mtime_ns": {name: path.stat().st_mtime_ns for name, path in required.items()},
        "artifacts": {name: {"path": str(path), "sha256": sha256(path)} for name, path in required.items()},
    }
    target = pool / "POOL_AUDITED.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
