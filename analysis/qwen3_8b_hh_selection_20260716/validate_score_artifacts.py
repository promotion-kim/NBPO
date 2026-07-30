#!/usr/bin/env python3
"""Outcome-blind structural validation for locked HH reward score artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


OBJECTIVES = [
    "skywork_llama", "skywork_qwen3", "athene", "armo_helpfulness",
    "beaver_v1", "beaver_v2", "llama_guard3", "shieldgemma", "qwen3guard8",
]
POLICIES = [
    "base", "weak_small", "over_refusing", "terse", "answer_anything",
    "ronpo_full_expect", "ronpo_k_only", "ipo", "simpo", "sppo_avg", "inpo_avg",
    "ht_mnpo_helpfulness", "ht_mnpo_safety", "ht_mnpo_conciseness",
]


def read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    reference = None
    outputs = {}
    for objective in OBJECTIVES:
        path = args.root / "scores" / f"{objective}.jsonl"
        rows = read(path)
        if len(rows) != 768:
            raise RuntimeError(f"{objective}: expected 768 rows, got {len(rows)}")
        keys = []
        for row in rows:
            key = (str(row["prompt_id"]), str(row["source"]), str(row["slice"]), str(row["prompt"]))
            keys.append(key)
            if row.get("response_model_names") != POLICIES:
                raise RuntimeError(f"{objective}/{key[0]}: model order mismatch")
            values = row.get("all_rm_scores")
            if not isinstance(values, list) or len(values) != len(POLICIES):
                raise RuntimeError(f"{objective}/{key[0]}: score count mismatch")
            if not all(math.isfinite(float(value)) for value in values):
                raise RuntimeError(f"{objective}/{key[0]}: non-finite score")
        if len(set(keys)) != 768:
            raise RuntimeError(f"{objective}: duplicate prompt records")
        keys = sorted(keys)
        if reference is None:
            reference = keys
        elif keys != reference:
            raise RuntimeError(f"{objective}: prompt/source/slice set mismatch")
        outputs[objective] = {
            "records": len(rows), "scores_per_record": len(POLICIES),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    payload = {
        "status": "PASS", "objectives": outputs,
        "prompt_records": 768, "policies_per_prompt": POLICIES,
        "finite_score_count": 768 * len(POLICIES) * len(OBJECTIVES),
        "fresh_confirmation_opened": False, "spent_sealed_split_touched": False,
    }
    output = args.root / "score_validation.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
