#!/usr/bin/env python3
"""Make a scorer-compatible one-policy JSONL from a locked decode artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, default=1000)
    args = parser.parse_args()
    records = json.loads(args.generation.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != args.expected_records:
        raise RuntimeError(f"expected {args.expected_records} generation records")
    seen: set[str] = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in records:
            prompt_id = str(row["prompt_id"])
            response = str(row.get("generated_text", ""))
            if prompt_id in seen or not response.strip():
                raise RuntimeError(f"invalid record {prompt_id}")
            seen.add(prompt_id)
            handle.write(json.dumps({
                "prompt_id": prompt_id,
                "prompt": str(row["prompt"]),
                "all_generated_responses": [response],
                "response_model_names": [args.model],
            }, ensure_ascii=False) + "\n")
    print(json.dumps({"records": len(seen), "model": args.model, "output": str(args.output), "sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
