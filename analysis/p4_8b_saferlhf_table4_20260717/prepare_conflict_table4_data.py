#!/usr/bin/env python3
"""Freeze conflict-only SafeRLHF train and validation manifests for Table 4."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value)).casefold()).strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pku(source: dict, split: str, index: int) -> dict:
    prompt = str(source["prompt"])
    digest = hashlib.sha256(f"{split}|{index}|{prompt}".encode()).hexdigest()[:16]
    return {
        "prompt_id": f"pku-{split}-{index}-{digest}",
        "prompt": prompt,
        "prompt_normalized": norm(prompt),
        "source": "pku_saferlhf",
        "slice": "dual_preference_conflict",
        "behavior_label": "dual_preference_conflict",
        "pku_split": split,
        "better_response_id": int(source["better_response_id"]),
        "safer_response_id": int(source["safer_response_id"]),
    }


def deterministic(rows: list[dict], namespace: str) -> list[dict]:
    return sorted(rows, key=lambda row: hashlib.sha256(f"{namespace}|{row['prompt_id']}".encode()).hexdigest())


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def heldout(paths: list[Path]) -> tuple[set[str], dict[str, int]]:
    union, counts = set(), {}
    for path in paths:
        rows = read(path)
        current = {norm(row["prompt"]) for row in rows if "prompt" in row}
        union.update(current)
        counts[str(path)] = len(current)
    return union, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--expected-test-sha256", required=True)
    parser.add_argument("--heldout", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-cap", type=int, default=2500)
    args = parser.parse_args()
    if sha(args.test) != args.expected_test_sha256:
        raise RuntimeError("pinned test checksum mismatch")
    prior, prior_counts = heldout(args.heldout)
    train_raw, test_raw = read(args.train), read(args.test)
    test_norms = {norm(row["prompt"]) for row in test_raw}
    train_conflicts = [pku(row, "train", index) for index, row in enumerate(train_raw) if int(row["better_response_id"]) != int(row["safer_response_id"])]
    test_conflicts = [pku(row, "test", index) for index, row in enumerate(test_raw) if int(row["better_response_id"]) != int(row["safer_response_id"])]
    selected_train = []
    seen = set()
    for row in deterministic(train_conflicts, "table4-conflict-train"):
        marker = row["prompt_normalized"]
        if marker in seen or marker in test_norms or marker in prior:
            continue
        seen.add(marker)
        selected_train.append(row)
        if len(selected_train) == args.prompt_cap:
            break
    if not selected_train:
        raise RuntimeError("no deduplicated conflict train prompts available")
    train_norms = {row["prompt_normalized"] for row in selected_train}
    validation, val_seen = [], set()
    for row in deterministic(test_conflicts, "table4-conflict-validation"):
        marker = row["prompt_normalized"]
        if marker in prior or marker in train_norms or marker in val_seen:
            continue
        val_seen.add(marker)
        validation.append(row)
    if not validation:
        raise RuntimeError("no held-out conflict validation prompts available")
    out = args.output_dir
    write(out / "train_conflict.jsonl", selected_train)
    write(out / "validation_conflict.jsonl", validation)
    payload = {
        "status": "locked_before_training",
        "train_raw_rows": len(train_raw),
        "test_raw_rows": len(test_raw),
        "train_conflict_rows_raw": len(train_conflicts),
        "test_conflict_rows_raw": len(test_conflicts),
        "heldout_prompt_counts": prior_counts,
        "heldout_prompt_union": len(prior),
        "collision_counts": {
            "train_conflicts_removed_by_test_overlap": sum(row["prompt_normalized"] in test_norms for row in train_conflicts),
            "train_conflicts_removed_by_prior_heldout": sum(row["prompt_normalized"] in prior for row in train_conflicts),
            "test_conflicts_removed_by_prior_heldout": sum(row["prompt_normalized"] in prior for row in test_conflicts),
        },
        "train": {"rows": len(selected_train), "sha256": sha(out / "train_conflict.jsonl")},
        "validation": {"rows": len(validation), "sha256": sha(out / "validation_conflict.jsonl")},
        "test_sha256": sha(args.test),
        "spent_sealed_split_touched": False,
    }
    (out / "data_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
