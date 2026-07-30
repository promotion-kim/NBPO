#!/usr/bin/env python3
"""Fail closed on thinking leakage, length collapse, empties, or repetition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def records(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("generation file must contain a JSON list")
    return [
        {
            "clean": str(row.get("generated_text") or row.get("text") or row.get("response") or ""),
            "raw": str(row.get("generated_text_raw") or row.get("generated_text") or row.get("text") or row.get("response") or ""),
        }
        for row in data
    ]


def max_repeat_run(text: str) -> int:
    words = text.split()
    if not words:
        return 0
    best = run = 1
    for previous, current in zip(words, words[1:]):
        run = run + 1 if current == previous else 1
        best = max(best, run)
    return best


def summary(values: list[dict[str, str]]) -> dict[str, Any]:
    word_counts = [len(value["clean"].split()) for value in values]
    leaks = [
        index for index, value in enumerate(values)
        if "<think>" in value["raw"].lower() or "</think>" in value["raw"].lower()
    ]
    repeat_runs = [max_repeat_run(value["clean"]) for value in values]
    return {
        "records": len(values),
        "empty_count": sum(not value["clean"].strip() for value in values),
        "think_leak_count": len(leaks),
        "think_leak_indices": leaks[:20],
        "mean_words": sum(word_counts) / len(word_counts),
        "max_words": max(word_counts),
        "max_repeat_run": max(repeat_runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-length-ratio", type=float, default=0.33)
    parser.add_argument("--max-length-ratio", type=float, default=2.0)
    parser.add_argument("--max-repeat-run", type=int, default=20)
    parser.add_argument("--expected-records", type=int, default=128)
    args = parser.parse_args()
    base = summary(records(Path(args.base)))
    candidate = summary(records(Path(args.candidate)))
    ratio = candidate["mean_words"] / max(base["mean_words"], 1e-12)
    checks = {
        "complete_records": candidate["records"] == args.expected_records,
        "zero_empty": candidate["empty_count"] == 0,
        "zero_think_leakage": candidate["think_leak_count"] == 0,
        "length_near_base": args.min_length_ratio <= ratio <= args.max_length_ratio,
        "repeat_run_below_threshold": candidate["max_repeat_run"] <= args.max_repeat_run,
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "base": base,
        "candidate": candidate,
        "candidate_base_mean_word_ratio": ratio,
        "thresholds": {
            "min_length_ratio": args.min_length_ratio,
            "max_length_ratio": args.max_length_ratio,
            "max_repeat_run": args.max_repeat_run,
            "expected_records": args.expected_records,
            "think_leakage": 0,
            "empty_count": 0,
        },
        "checks": checks,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 4)


if __name__ == "__main__":
    main()
