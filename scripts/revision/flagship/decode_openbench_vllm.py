#!/usr/bin/env python3
"""Generate deterministic non-thinking answers for the three open benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
from pathlib import Path

from vllm import LLM, SamplingParams


def chat_prompt(tokenizer, messages: list[dict]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    signature = inspect.signature(tokenizer.apply_chat_template)
    if "enable_thinking" in signature.parameters or any(
        value.kind == inspect.Parameter.VAR_KEYWORD for value in signature.parameters.values()
    ):
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def generate(llm, tokenizer, messages: list[list[dict]], params: SamplingParams) -> list[str]:
    formatted = [chat_prompt(tokenizer, value) for value in messages]
    outputs = llm.generate(formatted, params, use_tqdm=True)
    if len(outputs) != len(messages):
        raise RuntimeError(f"generation count mismatch: {len(outputs)} != {len(messages)}")
    return [value.outputs[0].text for value in outputs]


def max_identical_word_run(text: str) -> int:
    words = re.findall(r"\S+", text.lower())
    best = current = 0
    previous = None
    for word in words:
        current = current + 1 if word == previous else 1
        best = max(best, current)
        previous = word
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "responses.jsonl"
    metadata_path = args.output_dir / "decode_metadata.json"
    rows = read_rows(args.prompts)
    if output_path.exists() and metadata_path.exists():
        prior = read_rows(output_path)
        if len(prior) == len(rows) and all(len(x.get("responses", [])) == len(x["turns"]) for x in prior):
            print(f"skip complete {output_path}", flush=True)
            return

    llm = LLM(
        model=args.model,
        revision=args.revision,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        trust_remote_code=False,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        bad_words=["<think>", "</think>"],
    )

    first_messages = [[{"role": "user", "content": row["turns"][0]}] for row in rows]
    first_answers = generate(llm, tokenizer, first_messages, params)
    second_indices = [index for index, row in enumerate(rows) if len(row["turns"]) == 2]
    second_messages = [
        [
            {"role": "user", "content": rows[index]["turns"][0]},
            {"role": "assistant", "content": first_answers[index]},
            {"role": "user", "content": rows[index]["turns"][1]},
        ]
        for index in second_indices
    ]
    second_answers = generate(llm, tokenizer, second_messages, params) if second_messages else []
    second_by_index = dict(zip(second_indices, second_answers))

    completed = []
    for index, (source, first) in enumerate(zip(rows, first_answers)):
        answers = [first]
        if index in second_by_index:
            answers.append(second_by_index[index])
        completed.append(source | {
            "model_name": args.model_name,
            "model": args.model,
            "revision": args.revision,
            "responses": answers,
        })
    with output_path.open("w", encoding="utf-8") as handle:
        for row in completed:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    flat = [response for row in completed for response in row["responses"]]
    metadata = {
        "model_name": args.model_name,
        "model": args.model,
        "revision": args.revision,
        "backend": "vllm",
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "dtype": "bfloat16",
        "apply_chat_template": True,
        "enable_thinking": False,
        "bad_words": ["<think>", "</think>"],
        "num_items": len(completed),
        "num_turn_responses": len(flat),
        "empty_responses": sum(not value.strip() for value in flat),
        "think_tag_leaks": sum(bool(re.search(r"</?think>", value, re.I)) for value in flat),
        "max_identical_word_run": max((max_identical_word_run(value) for value in flat), default=0),
        "mean_words": sum(len(value.split()) for value in flat) / max(1, len(flat)),
        "responses_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
