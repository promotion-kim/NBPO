#!/usr/bin/env python3
"""Fail-closed stability gate for the 1.5B averaged-baseline repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def normalized_word(word: str) -> str:
    return re.sub(r"[^\w]+", "", word.lower())


def max_word_run(words: list[str]) -> int:
    best = current = 0
    previous = None
    for word in words:
        normalized = normalized_word(word)
        if normalized and normalized == previous:
            current += 1
        else:
            current = 1
            previous = normalized
        best = max(best, current)
    return best


def load(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"expected a JSON list: {path}")
    return data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def diagnostics(records: list[dict]) -> dict:
    word_counts: list[int] = []
    runs: list[int] = []
    empty = think = 0
    for record in records:
        response = str(record.get("generated_text", ""))
        words = response.split()
        word_counts.append(len(words))
        runs.append(max_word_run(words))
        empty += int(not response.strip())
        lowered = response.lower()
        think += int("<think>" in lowered or "</think>" in lowered)
    return {
        "records": len(records),
        "mean_words": sum(word_counts) / len(word_counts),
        "empty_responses": empty,
        "think_tag_leakage": think,
        "max_consecutive_identical_word_run": max(runs, default=0),
        "records_max_run_gt_20": sum(value > 20 for value in runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-records", type=int, default=128)
    parser.add_argument("--candidate-name", required=True)
    args = parser.parse_args()

    base_all = load(args.base)
    candidate_all = load(args.candidate)
    if len(base_all) != len(candidate_all):
        raise ValueError(f"record-count mismatch: base={len(base_all)} candidate={len(candidate_all)}")
    if len(base_all) < args.num_records:
        raise ValueError(f"only {len(base_all)} records; gate needs {args.num_records}")
    for index, (base, candidate) in enumerate(zip(base_all, candidate_all)):
        if base.get("prompt") != candidate.get("prompt"):
            raise ValueError(f"prompt mismatch at record {index}")

    base_diag = diagnostics(base_all[: args.num_records])
    candidate_diag = diagnostics(candidate_all[: args.num_records])
    ratio = candidate_diag["mean_words"] / base_diag["mean_words"]
    checks = {
        "record_count_exact": candidate_diag["records"] == args.num_records,
        "empty_responses_zero": candidate_diag["empty_responses"] == 0,
        "think_tag_leakage_zero": candidate_diag["think_tag_leakage"] == 0,
        "mean_word_ratio_in_range": math.isfinite(ratio) and 0.33 <= ratio <= 2.0,
        "max_consecutive_identical_word_run_le_20": (
            candidate_diag["max_consecutive_identical_word_run"] <= 20
        ),
    }
    result = {
        "measured_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "candidate": args.candidate_name,
        "passed": all(checks.values()),
        "fail_closed": True,
        "fixed_subset": "first 128 records in the common sorted 647-prompt decode order",
        "thresholds": {
            "records": args.num_records,
            "empty_responses": 0,
            "think_tag_leakage": 0,
            "mean_word_ratio_vs_base": [0.33, 2.0],
            "max_consecutive_identical_word_run": 20,
        },
        "checks": checks,
        "base": base_diag,
        "candidate_diagnostics": candidate_diag,
        "mean_word_ratio_vs_base": ratio,
        "provenance": {
            "base_path": str(args.base),
            "base_sha256": sha256(args.base),
            "candidate_path": str(args.candidate),
            "candidate_sha256": sha256(args.candidate),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 3)


if __name__ == "__main__":
    main()
