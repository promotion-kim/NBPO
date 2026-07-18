#!/usr/bin/env python3
"""Prepare preregistered SafeRLHF imbalance and held-out manifests.

This script uses only PKU human annotations and prompt text.  It never loads a
policy, reward model, response score, or experimental checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable


def normalized_prompt(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value)).casefold()).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def deterministic(rows: list[dict], namespace: str) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{namespace}|{row['prompt_id']}".encode("utf-8")
        ).hexdigest(),
    )


def pku_row(row: dict, split: str, index: int) -> dict:
    prompt = str(row["prompt"])
    response0 = str(row["response_0"])
    response1 = str(row["response_1"])
    benign = bool(row["is_response_0_safe"]) and bool(row["is_response_1_safe"])
    digest = hashlib.sha256(f"{split}|{index}|{prompt}".encode("utf-8")).hexdigest()
    return {
        "prompt_id": f"pku-{split}-{index}-{digest[:16]}",
        "prompt": prompt,
        "prompt_normalized": normalized_prompt(prompt),
        "pku_split": split,
        "pku_index": index,
        "behavior_label": "benign" if benign else "safety_active",
        "is_response_0_safe": bool(row["is_response_0_safe"]),
        "is_response_1_safe": bool(row["is_response_1_safe"]),
        "better_response_id": int(row["better_response_id"]),
        "safer_response_id": int(row["safer_response_id"]),
        "response_0": response0,
        "response_1": response1,
        "prompt_source": row.get("prompt_source"),
    }


def heldout_prompts(paths: list[Path]) -> tuple[set[str], dict[str, int]]:
    prompts: set[str] = set()
    counts: dict[str, int] = {}
    for path in paths:
        rows = read_jsonl(path)
        current = {
            normalized_prompt(row["prompt"])
            for row in rows
            if isinstance(row, dict) and "prompt" in row
        }
        prompts.update(current)
        counts[str(path)] = len(current)
    return prompts, counts


def take_balanced(
    rows: list[dict],
    per_label: int,
    namespace: str,
) -> list[dict]:
    selected: list[dict] = []
    for label in ("benign", "safety_active"):
        eligible = [row for row in rows if row["behavior_label"] == label]
        ordered = deterministic(eligible, f"{namespace}|{label}")
        if len(ordered) < per_label:
            raise RuntimeError(
                f"{namespace}: need {per_label} {label} prompts, found {len(ordered)}"
            )
        selected.extend(ordered[:per_label])
    return deterministic(selected, f"{namespace}|final")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-test-sha256", required=True)
    parser.add_argument("--prompt-cap", type=int, default=5000)
    parser.add_argument("--validation-per-label", type=int, default=128)
    parser.add_argument("--fresh-per-label", type=int, default=64)
    args = parser.parse_args()

    if sha256(args.test) != args.expected_test_sha256:
        raise RuntimeError("Pinned test.jsonl checksum mismatch")
    if args.prompt_cap < 10:
        raise ValueError("prompt cap must be at least ten")

    train = [pku_row(row, "train", index) for index, row in enumerate(read_jsonl(args.train))]
    test = [pku_row(row, "test", index) for index, row in enumerate(read_jsonl(args.test))]
    heldout, heldout_counts = heldout_prompts(args.heldout)
    all_test = {row["prompt_normalized"] for row in test}

    panel_candidates = [row for row in test if row["prompt_normalized"] not in heldout]
    validation = take_balanced(panel_candidates, args.validation_per_label, "p4-validation")
    validation_norm = {row["prompt_normalized"] for row in validation}
    fresh_candidates = [
        row for row in panel_candidates if row["prompt_normalized"] not in validation_norm
    ]
    fresh = take_balanced(fresh_candidates, args.fresh_per_label, "p4-fresh")
    fresh_norm = {row["prompt_normalized"] for row in fresh}

    # Entire dataset test is held out from training, not just the two P4 panels.
    train_candidates = [
        row for row in train
        if row["prompt_normalized"] not in all_test
        and row["prompt_normalized"] not in heldout
        and row["prompt_normalized"] not in validation_norm
        and row["prompt_normalized"] not in fresh_norm
    ]
    # Keep one representative per normalized prompt before label counting.
    deduped_train: list[dict] = []
    seen: set[str] = set()
    for row in deterministic(train_candidates, "p4-train-dedup"):
        if row["prompt_normalized"] not in seen:
            seen.add(row["prompt_normalized"])
            deduped_train.append(row)

    available = Counter(row["behavior_label"] for row in deduped_train)
    # Need rho=.9 benign/.1 active and rho=.5 benign/.5 active.  Round down to
    # a multiple of ten so each selected mixture has integral counts.
    max_common = min(
        args.prompt_cap,
        math.floor(available["benign"] / 0.9),
        math.floor(available["safety_active"] / 0.5),
    )
    matched_n = 10 * (max_common // 10)
    if matched_n < 3000:
        raise RuntimeError(
            f"Only {matched_n} matched prompts are constructible; preregistered minimum is 3000"
        )

    mixtures: dict[str, list[dict]] = {}
    for rho in (0.5, 0.9):
        benign_count = int(round(rho * matched_n))
        active_count = matched_n - benign_count
        selected: list[dict] = []
        for label, count in (("benign", benign_count), ("safety_active", active_count)):
            rows = [row for row in deduped_train if row["behavior_label"] == label]
            selected.extend(deterministic(rows, f"p4-rho-{rho:g}-{label}")[:count])
        if len(selected) != matched_n:
            raise RuntimeError(f"rho={rho}: insufficient prompt count")
        mixtures[f"rho_{rho:g}"] = deterministic(selected, f"p4-rho-{rho:g}-final")

    union: dict[str, dict] = {}
    for rows in mixtures.values():
        for row in rows:
            union[row["prompt_id"]] = row
    calibration = [
        {
            "prompt_id": row["prompt_id"],
            "prompt": row["prompt"],
            "all_generated_responses": [row["response_0"], row["response_1"]],
            "better_response_id": row["better_response_id"],
            "safer_response_id": row["safer_response_id"],
            "behavior_label": row["behavior_label"],
            "pku_split": "test",
        }
        for row in test
        if row["better_response_id"] != row["safer_response_id"]
    ]

    out = args.output_dir
    write_jsonl(out / "validation.jsonl", validation)
    write_jsonl(out / "fresh_unopened.jsonl", fresh)
    write_jsonl(out / "rho_0p5_prompts.jsonl", mixtures["rho_0.5"])
    write_jsonl(out / "rho_0p9_prompts.jsonl", mixtures["rho_0.9"])
    write_jsonl(out / "union_train_prompts.jsonl", deterministic(list(union.values()), "p4-union"))
    write_jsonl(out / "calibration_conflict.jsonl", calibration)

    files = [
        "validation.jsonl", "fresh_unopened.jsonl", "rho_0p5_prompts.jsonl",
        "rho_0p9_prompts.jsonl", "union_train_prompts.jsonl", "calibration_conflict.jsonl",
    ]
    payload = {
        "status": "complete",
        "dataset": "PKU-Alignment/PKU-SafeRLHF@9421ffafec3fa40a1f1a7d567b4d525079477ecb",
        "test_sha256": sha256(args.test),
        "train_rows_raw": len(train),
        "test_rows_raw": len(test),
        "heldout_prompt_counts": heldout_counts,
        "heldout_prompt_union": len(heldout),
        "panel_candidates_after_heldout_dedup": len(panel_candidates),
        "train_available_after_full_test_and_heldout_dedup": dict(available),
        "matched_prompt_count_per_rho": matched_n,
        "union_prompt_count": len(union),
        "calibration_conflict_rows": len(calibration),
        "files": {
            name: {"rows": sum(1 for _ in open(out / name, encoding="utf-8")), "sha256": sha256(out / name)}
            for name in files
        },
        "fresh_status": "unopened",
        "spent_sealed_split_touched": False,
    }
    (out / "data_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
