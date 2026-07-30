#!/usr/bin/env python3
"""Decode held-out prompts with Transformers, forcing Qwen3 non-thinking mode."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_prompts(path: Path) -> List[str]:
    prompts = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompts.add(record["prompt"])
    return sorted(prompts)


def load_partial(path: Path, prompts: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            expected_idx = len(rows)
            if expected_idx >= len(prompts) or record.get("prompt") != prompts[expected_idx]:
                raise ValueError(
                    f"Partial output prompt mismatch at {path}:{line_number}; "
                    "remove the partial file or keep the same input ordering."
                )
            rows.append(record)
    return rows


def format_generation_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    signature = inspect.signature(tokenizer.apply_chat_template)
    supports_extra_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if "enable_thinking" in signature.parameters or supports_extra_kwargs:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_prompts", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"output_{args.seed}.json"
    partial_path = out_dir / f"output_{args.seed}.partial.jsonl"

    prompts = read_prompts(Path(args.data_dir))
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        if isinstance(existing, list) and len(existing) == len(prompts):
            print(f"Skipping seed {args.seed}; found complete {output_path}", flush=True)
            return

    print(
        f"[decode-transformers] model={args.model} prompts={len(prompts)} "
        f"seed={args.seed} temperature={args.temperature} top_p={args.top_p} "
        f"max_new_tokens={args.max_new_tokens} batch_size={args.batch_size}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        trust_remote_code=False,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=False,
        local_files_only=args.local_files_only,
    ).to("cuda")
    model.eval()

    output_rows = load_partial(partial_path, prompts)
    start_idx = len(output_rows)
    start_time = time.time()
    completed_this_run = 0

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    for batch_start in range(start_idx, len(prompts), args.batch_size):
        batch_prompts = prompts[batch_start : batch_start + args.batch_size]
        formatted = [format_generation_prompt(tokenizer, prompt) for prompt in batch_prompts]
        inputs = tokenizer(formatted, return_tensors="pt", padding=True).to("cuda")
        prompt_width = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            outputs = model.generate(**inputs, **generation_kwargs)
        generated = outputs[:, prompt_width:]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        # Preserve the undecorated token stream for the hard thinking-leakage
        # gate. Qwen tokenizers may classify thinking delimiters as special
        # tokens, in which case skip_special_tokens=True would hide a leak.
        raw_texts = tokenizer.batch_decode(generated, skip_special_tokens=False)

        chunk_rows = []
        for prompt, fmt, text, raw_text in zip(batch_prompts, formatted, texts, raw_texts):
            chunk_rows.append(
                {
                    "prompt": prompt,
                    "format_prompt": fmt,
                    "generated_text": text,
                    "generated_text_raw": raw_text,
                }
            )

        with partial_path.open("a", encoding="utf-8") as f:
            for row in chunk_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        output_rows.extend(chunk_rows)
        completed_this_run += len(chunk_rows)
        elapsed = time.time() - start_time
        rate = completed_this_run / elapsed if elapsed > 0 else 0.0
        remaining = len(prompts) - len(output_rows)
        eta = remaining / rate if rate > 0 else float("inf")
        print(
            f"[decode-transformers] prompts={len(output_rows)}/{len(prompts)} "
            f"rate={rate:.3f} prompt/s elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
            flush=True,
        )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_rows, f, ensure_ascii=False, indent=2)
    print(f"[decode-transformers] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
