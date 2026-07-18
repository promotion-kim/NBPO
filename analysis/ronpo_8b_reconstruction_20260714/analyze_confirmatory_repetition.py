#!/usr/bin/env python3
"""Measure per-response consecutive identical-word repetition from raw outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def repeat_diagnostic(text: str) -> tuple[int, str, int]:
    words = text.split()
    if not words:
        return 0, "", 0
    best = run = 1
    best_word = words[0]
    for previous, current in zip(words, words[1:]):
        run = run + 1 if current == previous else 1
        if run > best:
            best = run
            best_word = current
    return best, best_word, len(words)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    violations = []
    for model_dir in sorted(path for path in args.generation_root.iterdir() if path.is_dir()):
        path = model_dir / "output_42.json"
        if not path.is_file():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        measured = []
        for index, row in enumerate(rows):
            text = str(row.get("generated_text_raw") or row.get("generated_text") or "")
            run, word, word_count = repeat_diagnostic(text)
            item = {
                "model": model_dir.name,
                "response_index": index,
                "prompt_sha256": hashlib.sha256(str(row.get("prompt", "")).encode()).hexdigest(),
                "max_repeat_run": run,
                "repeated_word": word,
                "response_word_count": word_count,
            }
            measured.append(item)
            if run > args.threshold:
                violations.append(item)
        if not measured:
            continue
        ordered = sorted(measured, key=lambda item: (item["max_repeat_run"], item["response_index"]))
        worst = ordered[-1]
        count = sum(item["max_repeat_run"] > args.threshold for item in measured)
        summaries.append({
            "model": model_dir.name,
            "responses": len(measured),
            "responses_over_threshold": count,
            "percent_over_threshold": 100.0 * count / len(measured),
            "max_repeat_run": worst["max_repeat_run"],
            "worst_response_index": worst["response_index"],
            "worst_repeated_word": worst["repeated_word"],
            "worst_response_word_count": worst["response_word_count"],
            "gate": "failed" if count else "passed",
        })

    summaries.sort(key=lambda row: row["model"])
    violations.sort(key=lambda row: (row["model"], -row["max_repeat_run"], row["response_index"]))
    with (args.output_dir / "repetition_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    payload = {
        "source_generation_root": str(args.generation_root),
        "definition": "maximum consecutive run of identical whitespace-delimited words in each raw response",
        "gate_threshold": args.threshold,
        "violation_rule": f"max_repeat_run > {args.threshold}",
        "summaries": summaries,
        "violations": violations,
    }
    (args.output_dir / "repetition_diagnostics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
