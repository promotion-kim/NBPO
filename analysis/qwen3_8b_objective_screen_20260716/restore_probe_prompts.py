#!/usr/bin/env python3
"""Restore evaluation prompts after a deterministic decode-only probe instruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("expected JSON list")
        return value
    # Iterate on physical newlines only. str.splitlines() also splits on U+2028
    # and related characters that are legal inside JSON strings.
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-input", type=Path, required=True)
    parser.add_argument("--decode-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mapping = {
        str(row["prompt"]): {
            "original_prompt": str(row.get("original_prompt", row["prompt"])),
            "prompt_id": row.get("prompt_id"),
            "probe_policy": row.get("probe_policy"),
            "probe_instruction": row.get("probe_instruction"),
        }
        for row in read_rows(args.probe_input)
    }
    restored = []
    for row in read_rows(args.decode_output):
        decoded_prompt = str(row["prompt"])
        if decoded_prompt not in mapping:
            raise KeyError(f"decode prompt absent from probe input: {decoded_prompt[:100]}")
        meta = mapping[decoded_prompt]
        restored.append({
            **row,
            "decode_prompt": decoded_prompt,
            "prompt": meta["original_prompt"],
            "original_prompt": meta["original_prompt"],
            "prompt_id": meta["prompt_id"],
            "probe_policy": meta["probe_policy"],
            "probe_instruction": meta["probe_instruction"],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(restored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(restored)}, indent=2))


if __name__ == "__main__":
    main()
