#!/usr/bin/env python3
"""Build the locked 40/20/20/20 training split from unused screen prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.qwen3_8b_base_objective_screen_20260716.build_dataset_manifest import (
    REVISIONS,
    load_beaver,
    load_or,
    load_pku,
    load_xstest,
)


TARGETS = {
    "pku_saferlhf": 308,
    "or_bench_hard": 154,
    "xstest_safe": 77,
    "xstest_unsafe": 77,
    "beavertails": 154,
}
SOURCE_ORDER = ["pku_saferlhf", "or_bench_hard", "xstest_safe", "xstest_unsafe", "beavertails"]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--screen-manifest-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "pku_saferlhf": args.source_root / "pku_saferlhf/data/Alpaca3-8B/test.jsonl",
        "or_bench_hard": args.source_root / "or_bench/or-bench-hard-1k.csv",
        "xstest": args.source_root / "xstest/xstest_prompts.csv",
        "beavertails": args.source_root / "beavertails/round0/30k/test.jsonl.gz",
    }
    xstest = load_xstest(paths["xstest"])
    pools = {
        "pku_saferlhf": load_pku(paths["pku_saferlhf"]),
        "or_bench_hard": load_or(paths["or_bench_hard"]),
        "xstest_safe": xstest["xstest_safe"],
        "xstest_unsafe": xstest["xstest_unsafe"],
        "beavertails": load_beaver(paths["beavertails"]),
    }

    # Match the screen's cross-source precedence before excluding both held-out
    # manifests, so no prompt identity can enter more than one source bucket.
    cross_seen: set[str] = set()
    for key in SOURCE_ORDER:
        unique = []
        for row in pools[key]:
            marker = norm(row["prompt"])
            if marker not in cross_seen:
                unique.append(row)
                cross_seen.add(marker)
        pools[key] = unique

    validation_path = args.screen_manifest_root / "validation.jsonl"
    fresh_path = args.screen_manifest_root / "fresh_confirmation.jsonl"
    validation = {norm(row["prompt"]) for row in read_jsonl(validation_path)}
    fresh = {norm(row["prompt"]) for row in read_jsonl(fresh_path)}
    if validation & fresh:
        raise RuntimeError("locked validation/fresh overlap")
    excluded = validation | fresh

    rows: list[dict] = []
    availability = {}
    for key in SOURCE_ORDER:
        available = [row for row in pools[key] if norm(row["prompt"]) not in excluded]
        availability[key] = len(available)
        requested = TARGETS[key]
        if len(available) < requested:
            raise RuntimeError(f"{key}: only {len(available)} unused rows, need {requested}")
        for row in available[:requested]:
            prompt_hash = sha_bytes(norm(row["prompt"]).encode())
            rows.append({**row, "split": "train", "prompt_id": f"{key}_{prompt_hash[:20]}"})
    rows.sort(key=lambda row: str(row["prompt_id"]))

    if len(rows) != 770 or len({norm(row["prompt"]) for row in rows}) != len(rows):
        raise RuntimeError("train count or uniqueness failure")
    if any(norm(row["prompt"]) in excluded for row in rows):
        raise RuntimeError("train overlaps a locked held-out manifest")

    output = args.output_root / "train.jsonl"
    write_jsonl(output, rows)
    payload = {
        "status": "locked_before_training",
        "construction": "exact 40/20/20/20 source mix using all but one available PKU conflict row to preserve the mixture under the deadline",
        "train": {"path": str(output), "count": len(rows), "sha256": sha_file(output), "counts": TARGETS},
        "validation": {"path": str(validation_path), "count": 640, "sha256": sha_file(validation_path)},
        "fresh_confirmation": {"path": str(fresh_path), "count": 320, "sha256": sha_file(fresh_path), "status": "manifest_hash_verified_not_decoded_or_scored"},
        "source_files": {
            key: {"path": str(path), "sha256": sha_file(path), "revision": REVISIONS[key]}
            for key, path in paths.items()
        },
        "available_after_validation_fresh_exclusion": availability,
        "dedup": "normalized whitespace/lowercase; screen cross-source precedence; excluded exact normalized prompts from both locked held-out manifests",
        "builder_sha256": sha_file(Path(__file__)),
        "spent_sealed_split_touched": False,
    }
    manifest = args.output_root / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
