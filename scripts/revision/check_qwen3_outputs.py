#!/usr/bin/env python3
"""Check revision smoke-test generations for Qwen3 thinking leakage and length drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def iter_records(path: Path):
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        yield from data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--max-mean-words", type=float, default=450.0)
    parser.add_argument("--max-any-words", type=int, default=1200)
    args = parser.parse_args()

    records = list(iter_records(Path(args.path)))
    if not records:
        raise SystemExit("no records")

    leaks = []
    word_counts = []
    samples = []
    for idx, record in enumerate(records):
        if "all_generated_responses" in record:
            texts = list(record.get("all_generated_responses") or [])
        else:
            texts = [record.get("generated_text") or record.get("text") or record.get("response") or ""]
        raw_values = record.get("all_generated_responses_raw")
        raw_texts = list(raw_values) if isinstance(raw_values, list) and raw_values else list(texts)
        if "all_generated_responses" not in record:
            raw_texts = [record.get("generated_text_raw") or texts[0]]

        record_words = 0
        record_leaked = False
        for text, raw_text in zip(texts, raw_texts):
            lowered = raw_text.lower()
            if "<think>" in lowered or "</think>" in lowered:
                record_leaked = True
            record_words += len(text.split())
        if record_leaked:
            leaks.append(idx)
        word_counts.append(record_words / max(1, len(texts)))
        if len(samples) < 5:
            samples.append({"idx": idx, "words": word_counts[-1], "text": texts[0][:1000] if texts else ""})

    mean_words = sum(word_counts) / len(word_counts)
    max_words = max(word_counts)
    summary = {
        "n": len(records),
        "think_leak_count": len(leaks),
        "think_leak_indices": leaks[:20],
        "mean_words": mean_words,
        "max_words": max_words,
        "samples": samples,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if leaks:
        raise SystemExit("thinking-token leakage detected")
    if mean_words > args.max_mean_words:
        raise SystemExit(f"mean length too high: {mean_words:.1f} > {args.max_mean_words}")
    if max_words > args.max_any_words:
        raise SystemExit(f"max length too high: {max_words} > {args.max_any_words}")


if __name__ == "__main__":
    main()
