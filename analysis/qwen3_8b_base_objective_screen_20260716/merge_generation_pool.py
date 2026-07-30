#!/usr/bin/env python3
"""Merge the preregistered policy responses into the common RM input format."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


THINK_SPAN = re.compile(r"<think>\s*\S[\s\S]*?</think>", re.IGNORECASE)
POLICY_ORDER = [
    "llama31", "qwen25", "mistral7",
    "llama31_over_refusing", "qwen25_over_refusing", "mistral7_over_refusing",
    "llama31_terse", "qwen25_terse", "mistral7_terse",
    "unsafe_zephyr", "weak_small",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    args = parser.parse_args()

    by_policy: dict[str, dict[str, dict]] = {}
    diagnostics: dict[str, dict] = {}
    for policy in POLICY_ORDER:
        path = args.generation_root / policy / "output_42.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        if len(rows) != 640:
            raise RuntimeError(f"{policy}: expected 640 rows, got {len(rows)}")
        mapping = {str(row["prompt_id"]): row for row in rows}
        if len(mapping) != len(rows):
            raise RuntimeError(f"{policy}: duplicate prompt ids")
        texts = [str(row["generated_text_raw"]) for row in rows]
        diagnostics[policy] = {
            "records": len(rows),
            "empty": sum(not value.strip() for value in texts),
            "nonempty_think_span": sum(bool(THINK_SPAN.search(value)) for value in texts),
            "mean_words": sum(len(value.split()) for value in texts) / len(texts),
            "generation_sha256": sha256(path),
        }
        if diagnostics[policy]["empty"] or diagnostics[policy]["nonempty_think_span"]:
            raise RuntimeError(f"{policy}: generation integrity failure {diagnostics[policy]}")
        by_policy[policy] = mapping

    prompt_ids = sorted(by_policy[POLICY_ORDER[0]])
    if any(set(mapping) != set(prompt_ids) for mapping in by_policy.values()):
        raise RuntimeError("policy prompt sets differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for prompt_id in prompt_ids:
            first = by_policy[POLICY_ORDER[0]][prompt_id]
            handle.write(json.dumps({
                "prompt_id": prompt_id,
                "prompt": first["prompt"],
                "source": first["source"],
                "slice": first["slice"],
                "behavior_label": first["behavior_label"],
                "response_model_names": POLICY_ORDER,
                "all_generated_responses": [
                    by_policy[policy][prompt_id]["generated_text_raw"] for policy in POLICY_ORDER
                ],
            }, ensure_ascii=False) + "\n")

    payload = {
        "status": "complete",
        "policy_order": POLICY_ORDER,
        "response_pool_sha256": sha256(args.output),
        "per_policy": diagnostics,
        "fresh_confirmation_opened": False,
        "spent_sealed_split_touched": False,
    }
    args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
