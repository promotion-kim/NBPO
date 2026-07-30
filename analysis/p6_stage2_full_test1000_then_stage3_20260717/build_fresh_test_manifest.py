#!/usr/bin/env python3
"""Lock a fresh, prompt-disjoint PKU-SafeRLHF test panel before decoding.

The selection uses only source rows, normalized prompt identity, and a fixed
SHA-256 ordering.  It neither reads model generations nor reward scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path


EXPECTED_TEST_SHA256 = "7f7ee8812fbeb52e1568a2b91d1d90bf6d0064b88ba8362198a7234d30007781"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--salt", default="p6-stage2-fresh-saferlhf-test-1000-seed42")
    args = parser.parse_args()

    if sha(args.test) != EXPECTED_TEST_SHA256:
        raise RuntimeError("pinned SafeRLHF test SHA-256 mismatch")
    excluded: set[str] = set()
    exclusions: dict[str, int] = {}
    for path in args.exclude:
        rows = read_jsonl(path)
        prompts = {normalize(str(row["prompt"])) for row in rows if "prompt" in row}
        excluded.update(prompts)
        exclusions[str(path)] = len(prompts)

    unique: dict[str, tuple[int, dict]] = {}
    raw = read_jsonl(args.test)
    for index, row in enumerate(raw):
        marker = normalize(str(row["prompt"]))
        unique.setdefault(marker, (index, row))
    candidates = [(marker, index, row) for marker, (index, row) in unique.items() if marker not in excluded]
    candidates.sort(key=lambda item: hashlib.sha256(f"{args.salt}\0{item[0]}".encode()).hexdigest())
    if len(candidates) < args.sample_size:
        raise RuntimeError(f"only {len(candidates)} fresh prompts; need {args.sample_size}")
    selected = candidates[:args.sample_size]
    records = []
    for rank, (marker, index, row) in enumerate(selected):
        digest = hashlib.sha256(f"test|{index}|{row['prompt']}".encode()).hexdigest()[:16]
        records.append({
            "prompt_id": f"p6-test-{rank:04d}-{digest}",
            "prompt": str(row["prompt"]),
            "prompt_normalized": marker,
            "source": "pku_saferlhf",
            "slice": "full_test_fresh_prompt_disjoint",
            "behavior_label": "dual_preference_conflict" if int(row["better_response_id"]) != int(row["safer_response_id"]) else "nonconflict",
            "pku_split": "test",
            "better_response_id": int(row["better_response_id"]),
            "safer_response_id": int(row["safer_response_id"]),
        })

    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = args.output_dir / "fresh_test_1000.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = {
        "status": "locked_before_decode",
        "scope": "One 1,000-prompt fresh PKU-SafeRLHF test evaluation. Prompt selection is score-blind and deterministic.",
        "source_test": str(args.test),
        "source_test_sha256": sha(args.test),
        "raw_test_rows": len(raw),
        "unique_test_prompts": len(unique),
        "excluded_prompt_union": len(excluded),
        "exclude_files_unique_prompt_counts": exclusions,
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
