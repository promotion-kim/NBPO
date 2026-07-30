#!/usr/bin/env python3
"""Merge one response per eligible model into the shared evaluation-score pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_generation(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    mapping = {str(row.get("prompt_id", row["prompt"])): row for row in rows}
    if len(mapping) != len(rows):
        raise RuntimeError(f"duplicate prompt ids in {path}")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--models", required=True, help="comma-separated locked model order, base first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    requested = [value for value in args.models.split(",") if value]
    eligible, failures = [], {}
    for model in requested:
        gate = json.loads((args.gate_root / f"{model}.json").read_text(encoding="utf-8"))
        if gate.get("passed"):
            eligible.append(model)
        else:
            failures[model] = gate
    if not eligible or eligible[0] != "base":
        raise RuntimeError("base must pass and remain first")
    generations = {
        model: read_generation(args.generation_root / model / f"output_{args.seed}.json")
        for model in eligible
    }
    first_ids = list(generations["base"])
    if len(first_ids) != args.expected_records:
        raise RuntimeError(f"base count {len(first_ids)} != {args.expected_records}")
    if any(set(mapping) != set(first_ids) for mapping in generations.values()):
        raise RuntimeError("eligible model prompt sets differ")
    rows = []
    for prompt_id in sorted(first_ids):
        first = generations["base"][prompt_id]
        rows.append({
            "prompt_id": prompt_id,
            "prompt": first["prompt"],
            "source": first.get("source"),
            "slice": first.get("slice"),
            "behavior_label": first.get("behavior_label"),
            "response_model_names": eligible,
            "all_generated_responses": [generations[model][prompt_id]["generated_text_raw"] for model in eligible],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = {
        "status": "complete", "requested_models": requested, "eligible_models": eligible,
        "failed_models": sorted(failures), "records": len(rows), "response_pool_sha256": sha(args.output),
        "spent_sealed_split_touched": False,
    }
    args.audit.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
