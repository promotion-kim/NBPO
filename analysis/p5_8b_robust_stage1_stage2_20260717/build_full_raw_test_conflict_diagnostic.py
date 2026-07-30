#!/usr/bin/env python3
"""Build a retrospective full PKU-SafeRLHF conflict-test diagnostic manifest.

This is intentionally not a fresh evaluation: it preserves and reports prompt
overlap with earlier held-out panels rather than filtering it away.  Duplicate
normalized prompts are emitted once so prompt-level bootstrap units remain
independent.
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


def heldout_prompts(paths: list[Path]) -> tuple[set[str], dict[str, int]]:
    union: set[str] = set()
    per_file = {}
    for path in paths:
        rows = read_jsonl(path)
        prompts = {normalize(str(row["prompt"])) for row in rows if "prompt" in row}
        union.update(prompts)
        per_file[str(path)] = len(prompts)
    return union, per_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--prior-heldout", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.test) != EXPECTED_TEST_SHA256:
        raise RuntimeError("pinned SafeRLHF test SHA-256 mismatch")
    previous, previous_counts = heldout_prompts(args.prior_heldout)
    raw = read_jsonl(args.test)
    conflicts = [
        (index, row)
        for index, row in enumerate(raw)
        if int(row["better_response_id"]) != int(row["safer_response_id"])
    ]
    seen: set[str] = set()
    rows = []
    for index, row in conflicts:
        prompt = str(row["prompt"])
        marker = normalize(prompt)
        if marker in seen:
            continue
        seen.add(marker)
        digest = hashlib.sha256(f"test|{index}|{prompt}".encode("utf-8")).hexdigest()[:16]
        rows.append({
            "prompt_id": f"pku-full-test-{index}-{digest}",
            "prompt": prompt,
            "prompt_normalized": marker,
            "source": "pku_saferlhf",
            "slice": "dual_preference_conflict_full_test_retrospective",
            "behavior_label": "dual_preference_conflict",
            "pku_split": "test",
            "better_response_id": int(row["better_response_id"]),
            "safer_response_id": int(row["safer_response_id"]),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "full_test_conflict_unique.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = {
        "status": "retrospective_diagnostic_manifest_locked_before_decode",
        "scope": "Full raw PKU-SafeRLHF test conflict diagnostic. It intentionally contains prompts overlapping earlier held-out panels and is ineligible for model selection, paper ranking, or a fresh-generalization claim.",
        "source_test": str(args.test),
        "source_test_sha256": sha(args.test),
        "raw_test_rows": len(raw),
        "raw_conflict_rows": len(conflicts),
        "unique_conflict_prompts": len(rows),
        "duplicate_conflict_rows_removed": len(conflicts) - len(rows),
        "prior_heldout_counts": previous_counts,
        "prior_heldout_union": len(previous),
        "unique_conflicts_overlapping_prior_heldout": sum(row["prompt_normalized"] in previous for row in rows),
        "raw_conflicts_overlapping_prior_heldout": sum(normalize(str(row["prompt"])) in previous for _, row in conflicts),
        "manifest": str(manifest),
        "manifest_sha256": sha(manifest),
        "spent_sealed_split_touched": False,
    }
    (args.output_dir / "data_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
