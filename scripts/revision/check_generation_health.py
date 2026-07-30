#!/usr/bin/env python3
"""Fail a checkpoint gate on length drift or obvious mixed-script degeneration."""

from __future__ import annotations

import argparse
import json
import math
import unicodedata
from pathlib import Path


def load_texts(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("expected a JSON list")
    return [str(row.get("generated_text") or row.get("text") or "") for row in data]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def latin_letter_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 1.0
    latin = sum("LATIN" in unicodedata.name(char, "") for char in letters)
    return latin / len(letters)


def long_token_char_ratio(text: str) -> float:
    tokens = text.split()
    total = sum(len(token) for token in tokens)
    if total == 0:
        return 0.0
    return sum(len(token) for token in tokens if len(token) > 40) / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--max-mean-words", type=float, default=650.0)
    parser.add_argument("--max-p95-words", type=float, default=1600.0)
    parser.add_argument("--min-mean-latin-ratio", type=float, default=0.70)
    parser.add_argument("--max-low-latin-fraction", type=float, default=0.15)
    parser.add_argument("--max-long-token-fraction", type=float, default=0.10)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    texts = load_texts(Path(args.path))
    if not texts or any(not text.strip() for text in texts):
        raise SystemExit("empty generation detected")

    words = [float(len(text.split())) for text in texts]
    latin = [latin_letter_ratio(text) for text in texts]
    long_tokens = [long_token_char_ratio(text) for text in texts]
    summary = {
        "records": len(texts),
        "mean_words": sum(words) / len(words),
        "p95_words": percentile(words, 0.95),
        "max_words": max(words),
        "mean_latin_letter_ratio": sum(latin) / len(latin),
        "low_latin_fraction": sum(value < 0.50 for value in latin) / len(latin),
        "mean_long_token_char_ratio": sum(long_tokens) / len(long_tokens),
        "high_long_token_fraction": sum(value > 0.10 for value in long_tokens) / len(long_tokens),
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")

    failures = []
    if summary["mean_words"] > args.max_mean_words:
        failures.append("mean_words")
    if summary["p95_words"] > args.max_p95_words:
        failures.append("p95_words")
    if summary["mean_latin_letter_ratio"] < args.min_mean_latin_ratio:
        failures.append("mean_latin_letter_ratio")
    if summary["low_latin_fraction"] > args.max_low_latin_fraction:
        failures.append("low_latin_fraction")
    if summary["high_long_token_fraction"] > args.max_long_token_fraction:
        failures.append("high_long_token_fraction")
    if failures:
        raise SystemExit("generation health gate failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
