#!/usr/bin/env python3
"""Assemble one compact, aligned response pool without modifying response text."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_rows(path: Path) -> dict[str, dict]:
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    out = {}
    for row in rows:
        key = str(row.get("original_prompt") or row.get("prompt"))
        out[key] = row
    return out


def generated_text(row: dict) -> str:
    value = row.get("generated_text_raw", row.get("generated_text", ""))
    return str(value)


def parse_mapping(value: str) -> tuple[str, Path]:
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy", action="append", required=True, help="name=output.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_rows(args.manifest)
    mappings = [parse_mapping(value) for value in args.policy]
    by_policy = {name: load_rows(path) for name, path in mappings}
    names = [name for name, _ in mappings]
    if len(names) != len(set(names)):
        raise ValueError("duplicate policy name")

    rows = []
    for prompt in sorted(manifest):
        responses = []
        hashes = []
        for name in names:
            if prompt not in by_policy[name]:
                raise KeyError(f"missing prompt for {name}: {prompt[:120]}")
            text = generated_text(by_policy[name][prompt])
            if not text.strip():
                raise ValueError(f"empty response for {name}: {prompt[:120]}")
            responses.append(text)
            hashes.append(hashlib.sha256(text.encode()).hexdigest())
        source = manifest[prompt]
        rows.append({
            "prompt_id": source.get("prompt_id"),
            "prompt": prompt,
            "split": source.get("split"),
            "category": source.get("category", source.get("source", "unknown")),
            "response_model_names": names,
            "response_sha256": hashes,
            "all_generated_responses": responses,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "prompts": len(rows), "policies": names}, indent=2))


if __name__ == "__main__":
    main()
