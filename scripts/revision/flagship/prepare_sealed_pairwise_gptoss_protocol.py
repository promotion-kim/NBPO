#!/usr/bin/env python3
"""Freeze the RM-independent sealed pairwise judge protocol before judging."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path


MODELS = (
    "base", "ronpo_full_expect", "ronpo_k_only", "ipo", "simpo",
    "sppo_avg", "inpo_avg", "ht_mnpo_helpfulness", "ht_mnpo_safety",
    "ht_mnpo_conciseness",
)
SYSTEM = (
    "Act as an impartial judge comparing Assistant A and Assistant B on the user prompt. "
    "Evaluate correctness, task fulfillment, helpfulness, relevance, clarity, conciseness, "
    "and creativity when needed. Do not generate your own alternative answer. Return exactly "
    "one final label and nothing else: [[A>>B]], [[A>B]], [[A=B]], [[B>A]], or [[B>>A]]."
)
USER_TEMPLATE = """<|User Prompt|>
{prompt}

<|The Start of Assistant A's Answer|>
{answer_a}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{answer_b}
<|The End of Assistant B's Answer|>

Return exactly one of these strings now: [[A>>B]], [[A>B]], [[A=B]], [[B>A]], or [[B>>A]]. Do not return a bare A or B."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gates = json.loads((args.work / "stability_gates_corrected/summary.json").read_text())
    audit = json.loads((args.work / "gate_correction.json").read_text())
    if tuple(gates.get("eligible_models", [])) != MODELS:
        raise RuntimeError("eligible model order does not match the corrected gate")
    if audit.get("go_signal_for_reward_scoring") is not True:
        raise RuntimeError("gate correction audit is not finalized")
    generation_files = {}
    prompts = None
    for model in MODELS:
        path = args.work / "generations" / model / "output_42.json"
        rows = json.loads(path.read_text())
        if len(rows) != 604:
            raise RuntimeError(f"wrong generation count for {model}")
        current_prompts = [str(row["prompt"]) for row in rows]
        if prompts is None:
            prompts = current_prompts
        elif current_prompts != prompts:
            raise RuntimeError(f"prompt ordering mismatch for {model}")
        expected_hash = audit["generation_artifacts"][model]["sha256"]
        if sha256(path) != expected_hash:
            raise RuntimeError(f"generation hash mismatch for {model}")
        generation_files[model] = {"path": str(path), "sha256": expected_hash, "records": len(rows)}
    prompt_digest = hashlib.sha256(
        json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_JUDGING",
        "locked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "RM-independent all-pairwise judgment of already-generated sealed responses",
        "selection_impact": "none; this evaluation cannot change the locked model selection",
        "sealed_prompt_count": 604,
        "sealed_prompt_sha256": "52b4028bd3ce095524e3ae66f49bf495d1236fea4635248b4263f9db1920df69",
        "ordered_prompt_content_sha256": prompt_digest,
        "models": list(MODELS),
        "pairs_per_prompt": len(list(combinations(MODELS, 2))),
        "expected_judgments": 604 * len(list(combinations(MODELS, 2))),
        "generation_files": generation_files,
        "excluded_models": {
            "dpo": {"reason": "genuine sealed stability failure", "record_index": 252, "max_repeat_run": 1163}
        },
        "judge": {
            "model": "openai/gpt-oss-120b",
            "revision": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
            "license": "Apache-2.0",
            "dtype": "auto",
            "max_model_len": 16384,
            "max_new_tokens": 2048,
            "reasoning_effort": "low",
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.86,
            "trust_remote_code": False,
        },
        "rubric": {
            "system": SYSTEM,
            "user_template": USER_TEMPLATE,
            "labels": ["A>>B", "A>B", "A=B", "B>A", "B>>A"],
            "scoring": "winner receives 1, loser 0, tie 0.5; slight and significant labels have equal win value",
        },
        "randomization": {
            "order": "SHA-256 parity of task_id plus seed 42 assigns pair order deterministically",
            "shard": "first 16 SHA-256 hex digits of task_id modulo 4",
            "num_shards": 4,
        },
        "aggregation": {
            "primary_metric": "mean per-prompt all-opponent pairwise score",
            "bootstrap_unit": "prompt",
            "bootstrap_resamples": 2000,
            "bootstrap_seed": 42,
            "interval": "percentile_95",
        },
        "provenance": {
            "gate_correction_sha256": sha256(args.work / "gate_correction.json"),
            "ranked_sealed_summary_sha256": sha256(args.work / "results/ranked_sealed_summary.json"),
            "no_new_decode": True,
            "paid_api": False,
        },
    }
    configuration = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["configuration_sha256"] = hashlib.sha256(configuration).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
