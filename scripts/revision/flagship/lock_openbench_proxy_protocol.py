#!/usr/bin/env python3
"""Write an immutable protocol lock before any benchmark model generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-tsv", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--official-prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.models_tsv.open(encoding="utf-8") as handle:
        models = list(csv.DictReader(handle, delimiter="\t"))
    if len(models) != 11 or any(not row.get("revision") for row in models):
        raise RuntimeError("expected 11 exact model revisions")
    manifest = json.loads(args.prompt_manifest.read_text(encoding="utf-8"))
    if manifest.get("counts") != {"alpaca_eval_2": 805, "arena_hard_v0.1": 500, "mt_bench": 80}:
        raise RuntimeError("prompt manifest counts are not frozen official counts")
    value = {
        "protocol": "aaai27-qwen3-8b-zero-cost-openbench-proxy-v1",
        "locked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "Test whether validation-selected RONPO top-mass beats every flagship baseline without paid APIs.",
        "score_label": "open-weight local-judge pairwise proxy; not official benchmark score",
        "official_score_reproduction": False,
        "paid_api_calls": False,
        "prompt_manifest": manifest,
        "official_prompts_sha256": sha256(args.official_prompts),
        "models_tsv_sha256": sha256(args.models_tsv),
        "models": models,
        "selected_ronpo": {
            "name": "ronpo_k_only",
            "selection_source": "results/p1_validation_reward_seed42_20260714/ranked_validation_summary.json",
            "selection_metric": "mean_prompt_worst_norm_score",
            "selection_value": 0.2853,
            "note": "Selection was frozen before this open-benchmark proxy; no proxy result is used for selection.",
        },
        "generation": {
            "backend": "vllm 0.24.0",
            "seed": 42,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 2048,
            "max_model_len": 32768,
            "dtype": "bfloat16",
            "apply_chat_template": True,
            "enable_thinking": False,
            "bad_words": ["<think>", "</think>"],
        },
        "judge": {
            "model": "Qwen/Qwen3-32B",
            "revision": "9216db5781bf21249d130ec9da846c4624c16137",
            "license": "Apache-2.0",
            "backend": "vllm 0.24.0",
            "seed": 42,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 192,
            "max_model_len": 32768,
            "enable_thinking": False,
            "position_swap": True,
            "rubric": ["task fulfillment", "correctness", "relevance", "clarity", "safety", "multi-turn consistency"],
            "explicit_no_length_preference": True,
        },
        "comparisons": {
            "candidate": "ronpo_k_only",
            "opponents": [row["name"] for row in models if row["name"] != "ronpo_k_only"],
            "orders_per_item": 2,
            "expected_judgments": 1385 * 10 * 2,
        },
        "statistics": {
            "pair_score": "mean of the two position-swapped judgments; win=1, tie=0.5, loss=0",
            "bootstrap_unit": "benchmark item",
            "bootstrap_resamples": 2000,
            "bootstrap_seed": 42,
            "confidence_interval": 0.95,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        prior.pop("locked_at", None)
        compare = dict(value)
        compare.pop("locked_at", None)
        if prior != compare:
            raise RuntimeError("existing protocol lock differs; refusing to overwrite")
        print(f"protocol already locked: {args.output}")
        return
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
