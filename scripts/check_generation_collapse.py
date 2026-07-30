#!/usr/bin/env python3
"""Summarize eval generations for simple model-collapse monitoring."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[idx])


def max_repeated_ngram(words: list[str], n: int = 4) -> int:
    if len(words) < n:
        return 0
    counts = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return max(counts.values(), default=0)


def token_run(words: list[str]) -> int:
    best = cur = 0
    prev = None
    for word in words:
        if word == prev:
            cur += 1
        else:
            cur = 1
            prev = word
        best = max(best, cur)
    return best


def load_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path, help="policy_generations.jsonl path")
    parser.add_argument("--json", action="store_true", help="emit JSON summary")
    args = parser.parse_args()

    records = list(load_records(args.jsonl))
    response_chars: list[int] = []
    words_per_response: list[int] = []
    unique_char_ratios: list[float] = []
    max_runs: list[int] = []
    rep4_counts: list[int] = []
    top_word_fracs: list[float] = []

    for record in records:
        response = str(record.get("response", ""))
        words = response.split()
        response_chars.append(len(response))
        words_per_response.append(len(words))
        unique_char_ratios.append(len(set(response)) / max(len(response), 1))
        max_runs.append(max(int(record.get("max_token_run", 0) or 0), token_run(words)))
        rep4_counts.append(max_repeated_ngram(words, 4))
        counts = Counter(words)
        top_word_fracs.append(max(counts.values(), default=0) / max(len(words), 1))

    collapse_like = [
        i
        for i, (chars, run, rep4, uniq, top_frac, words) in enumerate(
            zip(response_chars, max_runs, rep4_counts, unique_char_ratios, top_word_fracs, words_per_response)
        )
        if run >= 50
        or rep4 >= 20
        or (words >= 80 and top_frac >= 0.5)
        or (chars >= 9000 and (rep4 >= 10 or top_frac >= 0.2 or uniq <= 0.01))
    ]

    summary = {
        "path": str(args.jsonl),
        "n": len(records),
        "chars_median": percentile(response_chars, 0.50),
        "chars_p90": percentile(response_chars, 0.90),
        "chars_max": max(response_chars, default=0),
        "words_median": percentile(words_per_response, 0.50),
        "max_token_run_max": max(max_runs, default=0),
        "rep4_p90": percentile(rep4_counts, 0.90),
        "rep4_max": max(rep4_counts, default=0),
        "unique_char_ratio_min": min(unique_char_ratios, default=0.0),
        "top_word_fraction_max": max(top_word_fracs, default=0.0),
        "collapse_like_count": len(collapse_like),
        "collapse_like_indices": collapse_like[:20],
        "status": "collapse_suspect" if collapse_like else "ok",
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(
            "status={status} n={n} chars_med={chars_median:.0f} chars_p90={chars_p90:.0f} "
            "chars_max={chars_max} rep4_p90={rep4_p90:.0f} rep4_max={rep4_max} "
            "run_max={max_token_run_max} collapse_like={collapse_like_count}".format(**summary)
        )
        if collapse_like:
            print(f"collapse_like_indices={summary['collapse_like_indices']}")


if __name__ == "__main__":
    main()
