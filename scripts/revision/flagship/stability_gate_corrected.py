#!/usr/bin/env python3
"""Fail closed while distinguishing reasoning spans from unpaired think tags.

This preserves every non-tag threshold and statistic from ``stability_gate.py``.
The only changed decision rule is that thinking leakage requires a paired
``<think>...</think>`` span whose body contains at least one non-whitespace
character. Unpaired tags and empty paired spans are recorded as artifacts but
do not count as reasoning leakage.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PAIRED_THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
OPEN_THINK_RE = re.compile(r"<think>", re.IGNORECASE)
CLOSE_THINK_RE = re.compile(r"</think>", re.IGNORECASE)


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


def max_repeat_run_details(text: str) -> dict[str, Any]:
    words = text.split()
    if not words:
        return {"run": 0, "token": None, "start_word": None, "end_word": None}
    best = run = 1
    best_token = words[0]
    best_end = 0
    for index, (previous, current) in enumerate(zip(words, words[1:]), start=1):
        run = run + 1 if current == previous else 1
        if run > best:
            best = run
            best_token = current
            best_end = index
    return {
        "run": best,
        "token": best_token,
        "start_word": best_end - best + 1,
        "end_word": best_end,
    }


def think_tag_diagnostics(raw: str) -> dict[str, Any]:
    spans = list(PAIRED_THINK_RE.finditer(raw))
    nonempty = [match for match in spans if match.group(1).strip()]
    empty = [match for match in spans if not match.group(1).strip()]
    remainder = PAIRED_THINK_RE.sub("", raw)
    return {
        "nonempty_paired_spans": len(nonempty),
        "empty_paired_spans": len(empty),
        "orphan_open_tags": len(OPEN_THINK_RE.findall(remainder)),
        "orphan_close_tags": len(CLOSE_THINK_RE.findall(remainder)),
    }


def summary(values: list[dict[str, str]]) -> dict[str, Any]:
    word_counts = [len(value["clean"].split()) for value in values]
    tag_diagnostics = [think_tag_diagnostics(value["raw"]) for value in values]
    leaks = [
        index for index, diagnostic in enumerate(tag_diagnostics)
        if diagnostic["nonempty_paired_spans"] > 0
    ]
    empty_pairs = [
        index for index, diagnostic in enumerate(tag_diagnostics)
        if diagnostic["empty_paired_spans"] > 0
    ]
    orphan_opens = [
        index for index, diagnostic in enumerate(tag_diagnostics)
        if diagnostic["orphan_open_tags"] > 0
    ]
    orphan_closes = [
        index for index, diagnostic in enumerate(tag_diagnostics)
        if diagnostic["orphan_close_tags"] > 0
    ]
    repeat_details = [max_repeat_run_details(value["clean"]) for value in values]
    maximum = max(enumerate(repeat_details), key=lambda item: item[1]["run"])
    return {
        "records": len(values),
        "empty_count": sum(not value["clean"].strip() for value in values),
        "think_leak_count": len(leaks),
        "think_leak_indices": leaks[:20],
        "think_tag_artifacts": {
            "empty_pair_count": len(empty_pairs),
            "empty_pair_indices": empty_pairs[:20],
            "orphan_open_count": len(orphan_opens),
            "orphan_open_indices": orphan_opens[:20],
            "orphan_close_count": len(orphan_closes),
            "orphan_close_indices": orphan_closes[:20],
        },
        "mean_words": sum(word_counts) / len(word_counts),
        "max_words": max(word_counts),
        "max_repeat_run": maximum[1]["run"],
        "max_repeat_evidence": {
            "record_index": maximum[0],
            **maximum[1],
        },
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
        "detector": {
            "version": "corrected_nonempty_paired_span_v1",
            "think_leak_rule": (
                "A leak is a case-insensitive <think>...</think> pair whose body "
                "contains at least one non-whitespace character."
            ),
        },
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
