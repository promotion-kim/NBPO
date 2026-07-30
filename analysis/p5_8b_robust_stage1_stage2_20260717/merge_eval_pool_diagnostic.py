#!/usr/bin/env python3
"""Merge fixed-panel generations for a clearly labelled diagnostic evaluation.

Unlike the paper evaluator, this tool deliberately includes pre-existing
generation files whose frozen stability gate failed.  It is only for tracing
the relationship between length drift and reward scores.  The audit records
the gate outcome for every model so callers cannot mistake it for an eligible
result table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_generation(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    # Two legacy Stage-2 decoders serialized different prompt_id metadata for
    # the same prompt text.  The prompt strings were verified byte-identical
    # by diagnose_generation_prompt_sets.py, so canonicalize *diagnostically*
    # on a hash of prompt text.  This avoids aligning scores by incompatible
    # arbitrary identifiers and never changes any generation or gate result.
    mapping = {
        hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(): row
        for row in rows
    }
    if len(mapping) != len(rows):
        raise RuntimeError(f"duplicate prompt identifiers in {path}")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    models = [value for value in args.models.split(",") if value]
    if not models or models[0] != "base":
        raise ValueError("base must be first")
    gates = {model: json.loads((args.gate_root / f"{model}.json").read_text(encoding="utf-8")) for model in models}
    generations = {
        model: read_generation(args.generation_root / model / f"output_{args.seed}.json")
        for model in models
    }
    prompt_ids = sorted(generations["base"])
    if len(prompt_ids) != args.expected_records:
        raise RuntimeError(f"base records {len(prompt_ids)} != {args.expected_records}")
    for model, values in generations.items():
        if set(values) != set(prompt_ids):
            raise RuntimeError(f"prompt set mismatch for {model}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for prompt_hash in prompt_ids:
            base = generations["base"][prompt_hash]
            handle.write(json.dumps({
                "prompt_id": prompt_hash,
                "prompt": base["prompt"],
                "source": base.get("source"),
                "slice": base.get("slice"),
                "behavior_label": base.get("behavior_label"),
                "response_model_names": models,
                "all_generated_responses": [generations[model][prompt_hash]["generated_text_raw"] for model in models],
            }, ensure_ascii=False) + "\n")
    payload = {
        "status": "complete",
        "scope": "diagnostic_only_includes_stability_failed_models; not eligible for paper ranking or model selection",
        "alignment_key": "sha256 of byte-identical prompt text; legacy prompt_id metadata differed for SimPO and IPO",
        "models": models,
        "records": len(prompt_ids),
        "gate_status": {
            model: {
                "passed": bool(gates[model].get("passed")),
                "status": gates[model].get("status"),
                "length_ratio": gates[model].get("candidate_base_mean_word_ratio"),
                "gate_json": str(args.gate_root / f"{model}.json"),
            }
            for model in models
        },
        "response_pool_sha256": digest(args.output),
        "spent_sealed_split_touched": False,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
