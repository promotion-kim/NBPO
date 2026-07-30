#!/usr/bin/env python3
"""Measure the frozen generation-stability diagnostics for the 1.5B repair run."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def _normalized_word(word: str) -> str:
    return re.sub(r"[^\w]+", "", word.lower())


def _max_word_run(words: list[str]) -> int:
    best = current = 0
    previous = None
    for word in words:
        normalized = _normalized_word(word)
        if normalized and normalized == previous:
            current += 1
        else:
            current = 1
            previous = normalized
        best = max(best, current)
    return best


def _max_repeated_ngram(words: list[str], n: int = 4) -> int:
    if len(words) < n:
        return 0
    counts = Counter(tuple(words[index : index + n]) for index in range(len(words) - n + 1))
    return max(counts.values(), default=0)


def diagnose(path: Path) -> dict[str, float | int | str]:
    records = json.loads(path.read_text(encoding="utf-8"))
    word_counts: list[int] = []
    max_runs: list[int] = []
    repeated_fourgrams: list[int] = []
    empty = think_leak = 0
    for record in records:
        response = str(record.get("generated_text", ""))
        words = response.split()
        word_counts.append(len(words))
        max_runs.append(_max_word_run(words))
        repeated_fourgrams.append(_max_repeated_ngram(words))
        empty += int(not response.strip())
        lowered = response.lower()
        think_leak += int("<think>" in lowered or "</think>" in lowered)
    return {
        "path": str(path),
        "records": len(records),
        "mean_words": sum(word_counts) / len(word_counts),
        "median_words": statistics.median(word_counts),
        "max_words": max(word_counts),
        "empty_responses": empty,
        "think_tag_leakage": think_leak,
        "max_consecutive_identical_word_run": max(max_runs),
        "records_max_run_gt_20": sum(value > 20 for value in max_runs),
        "max_repeated_fourgram_count": max(repeated_fourgrams),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    diagnostics = {
        name: diagnose(args.generation_root / name / "output_42.json")
        for name in ("baseline", "sppo", "inpo", "ronpo")
    }
    base_mean = float(diagnostics["baseline"]["mean_words"])
    for values in diagnostics.values():
        values["mean_word_ratio_vs_base"] = float(values["mean_words"]) / base_mean

    output = {
        "measured_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "seed": 42,
        "decode": {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 2048},
        "diagnostics": diagnostics,
        "collapse_hypothesis": {
            "sppo": "length drift/max-token saturation",
            "inpo": "severe consecutive-token repetition and length drift",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
