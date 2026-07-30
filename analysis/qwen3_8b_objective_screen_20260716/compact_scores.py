#!/usr/bin/env python3
"""Remove redundant response text from completed score JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


KEEP = {
    "prompt_id", "prompt", "split", "category", "response_model_names", "response_sha256",
    "all_rm_scores", "armo_score_type", "armo_reward_attribute_name", "armo_reward_attribute_index",
    "armo_reward_attribute_names", "qwen3guard_labels", "qwen3guard_categories",
    "qwen3guard_refusals",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(path: Path) -> dict:
    tmp = path.with_suffix(path.suffix + ".compact.tmp")
    count = 0
    with path.open("r", encoding="utf-8") as source, tmp.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            target.write(json.dumps({key: row[key] for key in KEEP if key in row}, ensure_ascii=False) + "\n")
            count += 1
    os.replace(tmp, path)
    return {"path": str(path), "rows": count, "sha256": sha(path), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = [compact(path) for path in sorted((args.root / "scores").glob("*/*.jsonl"))]
    payload = {"files": rows, "response_text_removed": True}
    (args.root / "compact_score_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(rows), "bytes": sum(row["bytes"] for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
