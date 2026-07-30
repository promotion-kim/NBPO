#!/usr/bin/env python3
"""Lock a fresh panel from the official multi-configuration PKU test split.

This script combines the three released test JSONL files that form the
Hugging Face ``default`` configuration at a pinned dataset revision.  It
selects by a fixed hash of normalized prompt text and never reads generated
responses or reward scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", type=Path, nargs="+", required=True)
    parser.add_argument("--exclude", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--salt", default="p6b-stage2-default-test-1000-seed42")
    args = parser.parse_args()

    exclusions: set[str] = set()
    exclusion_counts: dict[str, int] = {}
    for path in args.exclude:
        prompts = {normalize(str(row["prompt"])) for row in read_jsonl(path) if "prompt" in row}
        exclusions.update(prompts)
        exclusion_counts[str(path)] = len(prompts)

    sources = []
    unique: dict[str, tuple[str, int, dict]] = {}
    raw_total = 0
    for path in args.tests:
        rows = read_jsonl(path)
        config = path.parent.name
        raw_total += len(rows)
        sources.append({"path": str(path), "config": config, "rows": len(rows), "sha256": sha(path)})
        for index, row in enumerate(rows):
            marker = normalize(str(row["prompt"]))
            unique.setdefault(marker, (config, index, row))
    candidates = [(marker, config, index, row) for marker, (config, index, row) in unique.items() if marker not in exclusions]
    candidates.sort(key=lambda item: hashlib.sha256(f"{args.salt}\0{item[0]}".encode()).hexdigest())
    if len(candidates) < args.sample_size:
        raise RuntimeError(f"only {len(candidates)} prompt-disjoint default-test prompts; need {args.sample_size}")

    selected = candidates[:args.sample_size]
    records = []
    for rank, (marker, config, index, row) in enumerate(selected):
        digest = hashlib.sha256(f"{config}|{index}|{row['prompt']}".encode()).hexdigest()[:16]
        records.append({
            "prompt_id": f"p6b-{config}-{rank:04d}-{digest}",
            "prompt": str(row["prompt"]),
            "prompt_normalized": marker,
            "source": "pku_saferlhf",
            "source_config": config,
            "source_revision": args.dataset_revision,
            "slice": "default_test_fresh_prompt_disjoint",
            "behavior_label": "dual_preference_conflict" if int(row["better_response_id"]) != int(row["safer_response_id"]) else "nonconflict",
            "pku_split": "test",
            "better_response_id": int(row["better_response_id"]),
            "safer_response_id": int(row["safer_response_id"]),
        })

    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = args.output_dir / "fresh_default_test_1000.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = {
        "status": "locked_before_decode",
        "scope": "One 1,000-prompt fresh panel from the official PKU-SafeRLHF default test configuration. Selection is prompt-only, score-blind, and deterministic.",
        "dataset_revision": args.dataset_revision,
        "source_tests": sources,
        "raw_test_rows": raw_total,
        "unique_test_prompts": len(unique),
        "excluded_prompt_union": len(exclusions),
        "exclude_files_unique_prompt_counts": exclusion_counts,
        "eligible_fresh_prompts": len(candidates),
        "sample_size": args.sample_size,
        "selection_salt": args.salt,
        "conflict_rows_selected": sum(row["behavior_label"] == "dual_preference_conflict" for row in records),
        "nonconflict_rows_selected": sum(row["behavior_label"] == "nonconflict" for row in records),
        "manifest": str(manifest),
        "manifest_sha256": sha(manifest),
        "spent_sealed_split_touched": False,
    }
    (args.output_dir / "data_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
