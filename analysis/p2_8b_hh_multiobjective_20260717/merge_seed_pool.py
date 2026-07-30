#!/usr/bin/env python3
"""Merge four preregistered base-policy seeds into one shared response pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


THINK_SPAN = re.compile(r"<think>\s*\S[\s\S]*?</think>", re.IGNORECASE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--seeds", default="42,43,44,45")
    parser.add_argument("--expected-records", type=int, default=770)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    names = [f"seed{seed}" for seed in seeds]
    mappings: dict[str, dict[str, dict]] = {}
    diagnostics = {}
    for name, seed in zip(names, seeds):
        path = args.generation_root / name / f"output_{seed}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        if len(rows) != args.expected_records:
            raise RuntimeError(f"{name}: {len(rows)} != {args.expected_records}")
        mapping = {str(row["prompt_id"]): row for row in rows}
        if len(mapping) != len(rows):
            raise RuntimeError(f"{name}: duplicate prompt ids")
        texts = [str(row["generated_text_raw"]) for row in rows]
        diagnostics[name] = {
            "records": len(rows),
            "empty": sum(not text.strip() for text in texts),
            "nonempty_think_span": sum(bool(THINK_SPAN.search(text)) for text in texts),
            "mean_words": sum(len(text.split()) for text in texts) / len(texts),
            "sha256": sha(path),
        }
        if diagnostics[name]["empty"] or diagnostics[name]["nonempty_think_span"]:
            raise RuntimeError(f"{name}: generation integrity failure")
        mappings[name] = mapping
    prompt_ids = sorted(mappings[names[0]])
    if any(set(mapping) != set(prompt_ids) for mapping in mappings.values()):
        raise RuntimeError("seed prompt sets differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for prompt_id in prompt_ids:
            first = mappings[names[0]][prompt_id]
            handle.write(json.dumps({
                "prompt_id": prompt_id,
                "prompt": first["prompt"],
                "source": first["source"],
                "slice": first["slice"],
                "behavior_label": first["behavior_label"],
                "response_model_names": names,
                "all_generated_responses": [mappings[name][prompt_id]["generated_text_raw"] for name in names],
            }, ensure_ascii=False) + "\n")
    payload = {
        "status": "complete",
        "seeds": seeds,
        "response_names": names,
        "records": len(prompt_ids),
        "response_pool_sha256": sha(args.output),
        "per_seed": diagnostics,
        "spent_sealed_split_touched": False,
        "fresh_confirmation_opened": False,
    }
    args.diagnostics.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
