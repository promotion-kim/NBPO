#!/usr/bin/env python3
"""Make a four-response Stage-2 pool from two base and two parent samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


THINK_SPAN = re.compile(r"<think>\s*\S[\s\S]*?</think>", re.IGNORECASE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, expected: int) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != expected:
        raise RuntimeError(f"{path}: expected {expected} records, got {len(rows)}")
    mapping = {str(row["prompt_id"]): row for row in rows}
    if len(mapping) != len(rows):
        raise RuntimeError(f"{path}: duplicate prompt ids")
    for index, row in enumerate(rows):
        text = str(row.get("generated_text_raw", ""))
        if not text.strip() or THINK_SPAN.search(text):
            raise RuntimeError(f"{path}: generation integrity failure at row {index}")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-42", type=Path, required=True)
    parser.add_argument("--base-43", type=Path, required=True)
    parser.add_argument("--parent-42", type=Path, required=True)
    parser.add_argument("--parent-43", type=Path, required=True)
    parser.add_argument("--parent-name", required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    source_paths = [args.base_42, args.base_43, args.parent_42, args.parent_43]
    sources = [load(path, args.expected_records) for path in source_paths]
    ids = sorted(sources[0])
    if any(set(source) != set(ids) for source in sources[1:]):
        raise RuntimeError("source prompt-id sets differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for prompt_id in ids:
            first = sources[0][prompt_id]
            handle.write(json.dumps({
                "prompt_id": prompt_id,
                "prompt": first["prompt"],
                "source": first.get("source"),
                "slice": first.get("slice"),
                "behavior_label": first.get("behavior_label"),
                "response_model_names": ["base_seed42", "base_seed43", f"{args.parent_name}_seed42", f"{args.parent_name}_seed43"],
                "all_generated_responses": [
                    str(sources[0][prompt_id]["generated_text_raw"]),
                    str(sources[1][prompt_id]["generated_text_raw"]),
                    str(sources[2][prompt_id]["generated_text_raw"]),
                    str(sources[3][prompt_id]["generated_text_raw"]),
                ],
            }, ensure_ascii=False) + "\n")
    audit = {
        "status": "complete",
        "records": len(ids),
        "parent_name": args.parent_name,
        "source_paths": [str(path) for path in source_paths],
        "source_sha256": [sha(path) for path in source_paths],
        "response_composition": "two frozen base responses plus two fresh Stage-1-parent responses per prompt",
        "output": str(args.output),
        "output_sha256": sha(args.output),
        "spent_sealed_split_touched": False,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
