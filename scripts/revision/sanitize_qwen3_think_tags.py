#!/usr/bin/env python3
"""Remove literal Qwen3 thinking tags from generated evaluation outputs.

Qwen3 non-thinking mode can still emit a literal closing tag on prompts that
explicitly request chain-of-thought. For evaluation we keep only the visible
answer after the final </think> marker and record every edit.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
OPEN_RE = re.compile(r"<think>", flags=re.IGNORECASE)
CLOSE_RE = re.compile(r"</think>", flags=re.IGNORECASE)


def has_tag(text: str) -> bool:
    lowered = text.lower()
    return "<think>" in lowered or "</think>" in lowered


def sanitize_text(text: str) -> tuple[str, str | None]:
    if not has_tag(text):
        return text, None

    close_matches = list(CLOSE_RE.finditer(text))
    if close_matches:
        suffix = text[close_matches[-1].end() :].lstrip()
        if suffix:
            return suffix, "kept_suffix_after_final_closing_tag"

    without_blocks = THINK_BLOCK_RE.sub("", text)
    without_tags = CLOSE_RE.sub("", OPEN_RE.sub("", without_blocks)).strip()
    return without_tags, "removed_literal_think_tags"


def iter_text_fields(record: dict[str, Any]) -> list[str]:
    if "generated_text" in record:
        return ["generated_text"]
    if "text" in record:
        return ["text"]
    if "response" in record:
        return ["response"]
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--summary-file", default=None)
    args = parser.parse_args()

    path = Path(args.path)
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        jsonl = True
    else:
        records = json.loads(path.read_text(encoding="utf-8"))
        jsonl = False

    edits: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        for field in iter_text_fields(record):
            original = str(record.get(field) or "")
            sanitized, reason = sanitize_text(original)
            if reason is None:
                continue
            record[field] = sanitized
            record.setdefault("think_sanitize", []).append(
                {
                    "field": field,
                    "reason": reason,
                    "original_words": len(original.split()),
                    "sanitized_words": len(sanitized.split()),
                }
            )
            edits.append(
                {
                    "index": idx,
                    "field": field,
                    "reason": reason,
                    "original_words": len(original.split()),
                    "sanitized_words": len(sanitized.split()),
                }
            )

    remaining = 0
    for record in records:
        if isinstance(record, dict):
            remaining += sum(1 for field in iter_text_fields(record) if has_tag(str(record.get(field) or "")))

    if jsonl:
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
    else:
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "path": str(path),
        "edit_count": len(edits),
        "remaining_think_tag_count": remaining,
        "edits": edits[:100],
    }
    summary_path = Path(args.summary_file) if args.summary_file else path.with_name(path.stem + "_think_sanitize.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if remaining:
        raise SystemExit(f"remaining think tags after sanitization: {remaining}")


if __name__ == "__main__":
    main()
