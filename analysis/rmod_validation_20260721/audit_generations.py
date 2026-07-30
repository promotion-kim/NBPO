#!/usr/bin/env python3
"""Reward-blind integrity checks for RMOD generation files."""

import argparse
import json
import re
from pathlib import Path


THINK = re.compile(r"<think>\s*\S[\s\S]*?</think>", re.IGNORECASE)
WORD = re.compile(r"\S+")


def load(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def responses(row):
    values = row.get("all_generated_responses")
    if values is not None:
        return values
    for key in ("generated_text_raw", "generated_text", "response"):
        if key in row:
            return [row[key]]
    return []


def repeat_run(words):
    best = run = 0
    previous = None
    for word in words:
        key = word.casefold()
        run = run + 1 if key == previous else 1
        previous = key
        best = max(best, run)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="label=path")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = {"reference": args.reference, "methods": {}}
    for item in args.inputs:
        label, raw_path = item.split("=", 1)
        rows = load(Path(raw_path))
        texts = [str(value or "") for row in rows for value in responses(row)]
        word_lists = [WORD.findall(text) for text in texts]
        report["methods"][label] = {
            "prompt_records": len(rows),
            "responses": len(texts),
            "empty_responses": sum(not text.strip() for text in texts),
            "nonempty_think_spans": sum(bool(THINK.search(text)) for text in texts),
            "mean_words": sum(map(len, word_lists)) / len(word_lists),
            "max_consecutive_identical_word_run": max(map(repeat_run, word_lists)),
        }
    ref_words = report["methods"][args.reference]["mean_words"]
    for record in report["methods"].values():
        record["mean_word_ratio_vs_reference"] = record["mean_words"] / ref_words
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
