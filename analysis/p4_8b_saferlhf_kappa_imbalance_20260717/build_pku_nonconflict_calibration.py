#!/usr/bin/env python3
"""Convert PKU non-conflict human response pairs to the common RM JSONL schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    if sha256(args.input) != args.expected_sha256:
        raise RuntimeError("pinned PKU test checksum mismatch")
    rows = []
    with args.input.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            source = json.loads(line)
            if int(source["better_response_id"]) != int(source["safer_response_id"]):
                continue
            prompt = str(source["prompt"])
            digest = hashlib.sha256(f"test|{index}|{prompt}".encode()).hexdigest()[:16]
            rows.append({
                "prompt_id": f"pku-test-{index}-{digest}",
                "prompt": prompt,
                "all_generated_responses": [str(source["response_0"]), str(source["response_1"])],
                "better_response_id": int(source["better_response_id"]),
                "safer_response_id": int(source["safer_response_id"]),
                "behavior_label": "benign" if bool(source["is_response_0_safe"]) and bool(source["is_response_1_safe"]) else "safety_active",
                "pku_split": "test",
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
