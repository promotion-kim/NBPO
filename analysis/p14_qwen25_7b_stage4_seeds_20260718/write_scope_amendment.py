#!/usr/bin/env python3
"""Record the user-directed single-seed scope change without rewriting the original lock."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    lock = args.root / "run_lock.json"
    if not lock.is_file():
        raise RuntimeError("original run lock is missing")
    if list(args.root.glob("scheduler/*.DONE.json")):
        raise RuntimeError("scope amendment must precede completed training tasks")
    payload = {
        "status": "user_scope_amendment_before_completed_training",
        "original_run_lock_sha256": sha256(lock),
        "effective_seeds": [42],
        "effective_arms": [
            "ronpo_os", "inpo_avg", "sppo_avg", "simpo", "ipo", "dpo",
            "ht_mnpo_harmless", "ht_mnpo_helpfulness",
        ],
        "excluded_by_user": {
            "seeds": [43, 44],
            "arms": ["ronpo_topmass"],
        },
        "stages": [1, 2, 3, 4],
        "evaluation_prompts": 1000,
        "paper_table_format": "Table 4 two-objective columns",
        "other_locked_fields_unchanged": True,
        "spent_sealed_split_touched": False,
    }
    output = args.root / "scope_amendment.json"
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text() != text:
        raise RuntimeError("existing scope amendment differs")
    output.write_text(text)
    (args.root / "scope_amendment.json.sha256").write_text(f"{sha256(output)}  scope_amendment.json\n")
    print(text, end="")


if __name__ == "__main__":
    main()
