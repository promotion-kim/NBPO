#!/usr/bin/env python3
"""Extract a score-free Qwen3 response pool and freeze prompt-disjoint splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from datasets import load_from_disk


USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)


def prompt_id(row: dict[str, Any]) -> str:
    value = row.get("prompt_id")
    if value:
        return str(value)
    return hashlib.sha256(str(row.get("prompt", "")).encode()).hexdigest()


def user_prompt(value: Any) -> str:
    if isinstance(value, list):
        for message in value:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", "")).strip()
    text = str(value)
    match = USER_RE.search(text)
    if match:
        return match.group(1).strip()
    marker = "<|im_start|>assistant"
    return text.split(marker, 1)[0].strip() if marker in text else text.strip()


def digest_ids(values: list[str]) -> str:
    payload = "\n".join(sorted(values)).encode()
    return hashlib.sha256(payload).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_pool(dataset: Any, source_split: str) -> list[dict[str, Any]]:
    rows = []
    for source in dataset:
        responses = source.get("all_generated_responses")
        if not isinstance(responses, list) or len(responses) < 2:
            continue
        rows.append(
            {
                "prompt_id": prompt_id(source),
                "prompt": user_prompt(source.get("prompt")),
                "all_generated_responses": [str(value) for value in responses],
                "source_split": source_split,
                "pool_source": "qwen3_8b_existing_response_pool_scores_removed",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation fraction must lie in (0, 1)")

    source = Path(args.source_dir)
    output = Path(args.output_dir)
    train_source = clean_pool(load_from_disk(str(source / "train")), "source_train")
    sealed_source = clean_pool(load_from_disk(str(source / "test")), "source_sealed_test")

    ranked = sorted(
        train_source,
        key=lambda row: hashlib.sha256(("flagship-v1||" + row["prompt_id"]).encode()).hexdigest(),
    )
    validation_count = round(len(ranked) * args.validation_fraction)
    validation = ranked[:validation_count]
    train = ranked[validation_count:]

    sets = {
        "train": {row["prompt_id"] for row in train},
        "validation": {row["prompt_id"] for row in validation},
        "sealed_test": {row["prompt_id"] for row in sealed_source},
    }
    overlaps = {
        "train_validation": len(sets["train"] & sets["validation"]),
        "train_sealed_test": len(sets["train"] & sets["sealed_test"]),
        "validation_sealed_test": len(sets["validation"] & sets["sealed_test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"prompt leakage across splits: {overlaps}")
    for name, rows in (("train", train), ("validation", validation)):
        write_jsonl(output / f"pool_{name}.jsonl", rows)
    write_jsonl(output / "pool_gate.jsonl", train + validation)
    # Only prompts are exposed here; the old test responses and scores are not
    # used by objective selection or pair construction.
    write_jsonl(
        output / "sealed_test_prompts.jsonl",
        [{"prompt_id": row["prompt_id"], "prompt": row["prompt"]} for row in sealed_source],
    )

    manifest = {
        "schema_version": 1,
        "split_rule": "sort source-train prompts by sha256('flagship-v1||'+prompt_id); first round(10%) validation",
        "validation_fraction": args.validation_fraction,
        "source_dir": str(source),
        "base_model": "Qwen/Qwen3-8B",
        "counts": {name: len(values) for name, values in sets.items()},
        "prompt_id_sha256": {name: digest_ids(list(values)) for name, values in sets.items()},
        "overlaps": overlaps,
        "sealed_test_policy": "prompts only until S4a; no objective scores or model generations used in S0-S3",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
