#!/usr/bin/env python3
"""Remove duplicated response text from completed score JSONL while preserving every score."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


KEEP = {
    "prompt_id", "prompt", "source", "slice", "response_model_names", "all_rm_scores",
    "score_definition", "model_revision", "guard_family", "armo_score_source",
    "armo_reward_attribute_name", "armo_reward_attribute_index", "armo_reward_attribute_names",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    for path in sorted((args.root / "scores").glob("*.jsonl")):
        temp = path.with_suffix(".compact.tmp")
        count = 0
        with path.open("r", encoding="utf-8") as source, temp.open("w", encoding="utf-8") as target:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                target.write(json.dumps({key: row[key] for key in row if key in KEEP}, ensure_ascii=False) + "\n")
                count += 1
        if count != 768:
            raise RuntimeError(f"{path}: expected 768 rows, got {count}")
        os.replace(temp, path)
        print(path)


if __name__ == "__main__":
    main()
