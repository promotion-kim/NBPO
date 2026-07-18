#!/usr/bin/env python3
"""Deterministic vLLM decode with Qwen thinking explicitly disabled."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.sampling_params import RepetitionDetectionParams


def prompts(path: Path) -> list[str]:
    values = set()
    # Iterate by physical newline only.  str.splitlines() also splits on
    # Unicode line-separator/control characters that can legally occur inside
    # JSON strings in this response pool and would corrupt an otherwise valid
    # JSONL record.
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                values.add(json.loads(line)["prompt"])
    return sorted(values)


def format_prompt(tokenizer, prompt: str) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    signature = inspect.signature(tokenizer.apply_chat_template)
    if "enable_thinking" in signature.parameters or any(
        value.kind == inspect.Parameter.VAR_KEYWORD for value in signature.parameters.values()
    ):
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--forbid-thinking-tags", action="store_true")
    parser.add_argument("--repetition-detection-max-pattern-size", type=int, default=0)
    parser.add_argument("--repetition-detection-min-pattern-size", type=int, default=0)
    parser.add_argument("--repetition-detection-min-count", type=int, default=0)
    args = parser.parse_args()
    values = prompts(args.data_dir)
    if args.max_prompts:
        values = values[:args.max_prompts]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"output_{args.seed}.json"
    if output.exists():
        try:
            prior = json.loads(output.read_text())
        except json.JSONDecodeError:
            prior = None
        if isinstance(prior, list) and len(prior) == len(values):
            print(f"skip complete {output}", flush=True)
            return

    llm = LLM(
        model=args.model, revision=args.revision, dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed, trust_remote_code=False,
    )
    tokenizer = llm.get_tokenizer()
    formatted = [format_prompt(tokenizer, prompt) for prompt in values]
    sampling_kwargs = {
        "n": 1,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    if args.forbid_thinking_tags:
        sampling_kwargs["bad_words"] = ["<think>", "</think>"]
    if args.repetition_detection_max_pattern_size:
        sampling_kwargs["repetition_detection"] = RepetitionDetectionParams(
            max_pattern_size=args.repetition_detection_max_pattern_size,
            min_pattern_size=args.repetition_detection_min_pattern_size,
            min_count=args.repetition_detection_min_count,
        )
    params = SamplingParams(**sampling_kwargs)
    generated = llm.generate(formatted, params, use_tqdm=True)
    if len(generated) != len(values):
        raise RuntimeError(f"expected {len(values)} outputs, got {len(generated)}")
    rows = []
    for prompt, formatted_prompt, request_output in zip(values, formatted, generated):
        text = request_output.outputs[0].text
        rows.append({
            "prompt": prompt, "format_prompt": formatted_prompt,
            "generated_text": text, "generated_text_raw": text,
        })
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    metadata = {
        "backend": "vllm", "model": args.model, "revision": args.revision,
        "seed": args.seed, "temperature": args.temperature, "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens, "enable_thinking": False,
        "forbid_thinking_tags": args.forbid_thinking_tags,
        "bad_words": ["<think>", "</think>"] if args.forbid_thinking_tags else [],
        "repetition_detection": {
            "max_pattern_size": args.repetition_detection_max_pattern_size,
            "min_pattern_size": args.repetition_detection_min_pattern_size,
            "min_count": args.repetition_detection_min_count,
        },
        "num_prompts": len(rows), "prompt_sha256": hashlib.sha256(
            "\n".join(values).encode()
        ).hexdigest(),
    }
    (args.output_dir / "decode_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
