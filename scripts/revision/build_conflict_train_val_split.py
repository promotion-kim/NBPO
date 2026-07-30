#!/usr/bin/env python3
"""Create a prompt-disjoint, deterministic validation split for conflict training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def prompt_of(row: dict) -> str:
    value = row.get("prompt")
    if not isinstance(value, str) or not value:
        raise ValueError("row has no non-empty prompt")
    return value


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                prompt_of(row)
                rows.append(row)
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def stable_key(prompt: str) -> tuple[str, str]:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest(), prompt


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--avg-pairs", required=True)
    parser.add_argument("--ronpo-pairs", required=True)
    parser.add_argument("--test-pairs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-prompts", type=int, default=500)
    args = parser.parse_args()

    avg_rows = read_rows(Path(args.avg_pairs))
    ronpo_rows = read_rows(Path(args.ronpo_pairs))
    test_rows = read_rows(Path(args.test_pairs))
    avg_prompts = {prompt_of(row) for row in avg_rows}
    ronpo_prompts = {prompt_of(row) for row in ronpo_rows}
    test_prompts = {prompt_of(row) for row in test_rows}
    common_prompts = avg_prompts & ronpo_prompts
    if not common_prompts:
        raise ValueError("avg/RONPO prompt intersection is empty")
    if common_prompts & test_prompts:
        raise ValueError(f"train/test prompt overlap: {len(common_prompts & test_prompts)}")
    if not 1 <= args.validation_prompts < len(common_prompts):
        raise ValueError("validation-prompts must be between 1 and number of train prompts - 1")

    val_prompts = set(sorted(common_prompts, key=stable_key)[: args.validation_prompts])
    split = {}
    for name, rows in (("avg", avg_rows), ("ronpo", ronpo_rows)):
        train = [row for row in rows if prompt_of(row) in common_prompts - val_prompts]
        val = [row for row in rows if prompt_of(row) in val_prompts]
        write_rows(Path(args.output_dir) / f"{name}_train.jsonl", train)
        write_rows(Path(args.output_dir) / f"{name}_validation.jsonl", val)
        split[name] = {
            "train_rows": len(train),
            "validation_rows": len(val),
            "train_prompts": len({prompt_of(row) for row in train}),
            "validation_prompts": len({prompt_of(row) for row in val}),
        }

    manifest = {
        "selection_rule": "500 prompts with smallest SHA256(prompt), fixed before training",
        "validation_prompts": len(val_prompts),
        "test_prompts": len(test_prompts),
        "test_is_prompt_disjoint": True,
        "common_train_prompts": len(common_prompts),
        "dropped_noncommon_prompts": {
            "avg_only": len(avg_prompts - ronpo_prompts),
            "ronpo_only": len(ronpo_prompts - avg_prompts),
        },
        "inputs": {
            "avg_pairs": str(Path(args.avg_pairs).resolve()),
            "ronpo_pairs": str(Path(args.ronpo_pairs).resolve()),
            "test_pairs": str(Path(args.test_pairs).resolve()),
        },
        "splits": split,
        "validation_prompt_sha256": sorted(
            hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in val_prompts
        ),
    }
    output = Path(args.output_dir) / "split_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
