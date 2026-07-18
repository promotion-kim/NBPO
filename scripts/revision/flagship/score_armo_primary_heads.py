#!/usr/bin/env python3
"""Score all frozen ArmoRM flagship heads in one model pass per response batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _patch_transformers_gguf_compat() -> None:
    try:
        import transformers.integrations as integrations
    except Exception:
        return
    if not isinstance(getattr(integrations, "GGUF_CONFIG_MAPPING", None), dict):
        integrations.GGUF_CONFIG_MAPPING = {}
    mapping = getattr(integrations, "GGUF_TOKENIZER_MAPPING", None)
    if not isinstance(mapping, dict):
        mapping = {}
    mapping.setdefault("tokenizer", {})
    mapping.setdefault("tokenizer_config", {})
    integrations.GGUF_TOKENIZER_MAPPING = mapping
    if not hasattr(integrations, "_gguf_parse_value"):
        def _gguf_parse_value(_value, _data_type):
            raise RuntimeError("GGUF checkpoint parsing is unavailable in this image")
        integrations._gguf_parse_value = _gguf_parse_value
    try:
        import transformers.generation as generation
        if not hasattr(generation, "GenerationMixin"):
            from transformers.generation.utils import GenerationMixin
            generation.GenerationMixin = GenerationMixin
    except Exception:
        pass
    try:
        import transformers.models.llama.modeling_llama as llama_modeling
        if not hasattr(llama_modeling, "LLAMA_INPUTS_DOCSTRING"):
            llama_modeling.LLAMA_INPUTS_DOCSTRING = ""
    except Exception:
        pass


_patch_transformers_gguf_compat()

from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer


MODEL = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
ATTRIBUTES = [
    "helpsteer-helpfulness", "helpsteer-correctness", "helpsteer-coherence",
    "helpsteer-complexity", "helpsteer-verbosity", "ultrafeedback-overall_score",
    "ultrafeedback-instruction_following", "ultrafeedback-truthfulness",
    "ultrafeedback-honesty", "ultrafeedback-helpfulness", "beavertails-is_safe",
    "prometheus-score", "argilla-overall_quality", "argilla-judge_lm",
    "code-complexity", "code-style", "code-explanation",
    "code-instruction-following", "code-readability",
]
OBJECTIVES = {
    "helpfulness": ("ultrafeedback-helpfulness", 1.0),
    "safety": ("beavertails-is_safe", 1.0),
    "conciseness": ("helpsteer-verbosity", -1.0),
}


def batches(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def safe_conversation(conversation: list[dict], max_chars: int) -> list[dict]:
    result = []
    for message in conversation:
        copied = dict(message)
        if copied.get("role") == "user" and isinstance(copied.get("content"), str):
            copied["content"] = copied["content"][:max_chars]
        result.append(copied)
    return result


def forward(conversations, tokenizer, model, max_length: int) -> torch.Tensor:
    encoded = tokenizer.apply_chat_template(
        conversations, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length,
    )
    input_ids = encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]
    with torch.inference_mode():
        output = model(input_ids.to("cuda"))
    rewards = output.rewards.float().cpu()
    if rewards.ndim != 2 or rewards.shape[1] < len(ATTRIBUTES):
        raise RuntimeError(f"unexpected ArmoRM rewards shape: {tuple(rewards.shape)}")
    return rewards


def score_batch(conversations, tokenizer, model, max_length: int) -> torch.Tensor:
    try:
        return forward(conversations, tokenizer, model, max_length)
    except ValueError as exc:
        if "Token pattern not found" not in str(exc):
            raise
    rows = []
    for conversation in conversations:
        for max_chars in (4096, 2048, 1024, 512, 256, 0):
            try:
                rows.append(forward([safe_conversation(conversation, max_chars)], tokenizer, model, max_length)[0])
                break
            except ValueError as exc:
                if "Token pattern not found" not in str(exc):
                    raise
        else:
            raise RuntimeError("ArmoRM gating token missing after prompt fallback")
    return torch.stack(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    records = json.loads(args.input_file.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("merged generation input must be a non-empty JSON list")
    names = records[0].get("response_model_names")
    if not isinstance(names, list) or not names:
        raise ValueError("response_model_names missing")
    if any(row.get("response_model_names") != names for row in records):
        raise ValueError("response model ordering mismatch")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, cache_dir=args.cache_dir, use_fast=True,
        local_files_only=args.local_files_only,
    )
    tokenizer.truncation_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
        cache_dir=args.cache_dir, local_files_only=args.local_files_only,
    ).to("cuda").eval()

    outputs = {name: [] for name in OBJECTIVES}
    indices = {name: ATTRIBUTES.index(head) for name, (head, _) in OBJECTIVES.items()}
    for sample_group in batches(records, args.sample_batch_size):
        conversations = []
        counts = []
        for row in sample_group:
            responses = row["all_generated_responses"]
            counts.append(len(responses))
            conversations.extend([
                [{"role": "user", "content": row["prompt"]},
                 {"role": "assistant", "content": response}]
                for response in responses
            ])
        scored = torch.cat([
            score_batch(chunk, tokenizer, model, args.max_seq_length)
            for chunk in batches(conversations, args.batch_size)
        ])
        offset = 0
        for row, count in zip(sample_group, counts):
            current = scored[offset:offset + count]
            offset += count
            for objective, (head, transform) in OBJECTIVES.items():
                values = (current[:, indices[objective]] * transform).tolist()
                outputs[objective].append({
                    **row,
                    "all_rm_scores": values,
                    "reward_model": MODEL,
                    "reward_attribute_name": head,
                    "transform": "negate" if transform < 0 else "identity",
                })
        print(f"scored {sum(len(v) for v in outputs.values()) // len(OBJECTIVES)}/{len(records)} prompts", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for objective, rows in outputs.items():
        path = args.output_dir / f"{objective}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "model": MODEL,
        "objectives": {
            name: {"head": head, "transform": "negate" if transform < 0 else "identity"}
            for name, (head, transform) in OBJECTIVES.items()
        },
        "num_prompts": len(records), "response_model_names": names,
        "batch_size": args.batch_size, "sample_batch_size": args.sample_batch_size,
        "max_seq_length": args.max_seq_length,
    }
    (args.output_dir / "score_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
