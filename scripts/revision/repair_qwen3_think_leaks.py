#!/usr/bin/env python3
"""Deterministically regenerate Qwen3 outputs that contain literal think tags."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any


def has_think_tag(text: str) -> bool:
    lowered = text.lower()
    return "<think>" in lowered or "</think>" in lowered


def format_generation_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    signature = inspect.signature(tokenizer.apply_chat_template)
    supports_extra_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if "enable_thinking" in signature.parameters or supports_extra_kwargs:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def iter_output_files(output_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in output_dir.glob("output_*.json")
        if not path.name.endswith(".length_check.json")
    )


def parse_seed(path: Path) -> int:
    return int(path.stem.split("_", 1)[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_attempts", type=int, default=8)
    parser.add_argument("--repair_seed_stride", type=int, default=1000003)
    parser.add_argument("--disable_custom_all_reduce", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_files = iter_output_files(output_dir)
    todo: list[tuple[Path, int, int, dict[str, Any]]] = []
    for path in output_files:
        seed = parse_seed(path)
        records = json.loads(path.read_text(encoding="utf-8"))
        for index, record in enumerate(records):
            if has_think_tag(record.get("generated_text", "")):
                todo.append((path, seed, index, record))

    summary: dict[str, Any] = {
        "output_dir": str(output_dir),
        "initial_leak_count": len(todo),
        "repairs": [],
        "remaining_leak_count": None,
    }
    if not todo:
        summary["remaining_leak_count"] = 0
        (output_dir / "think_repair_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    from vllm import LLM, SamplingParams

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.num_gpu,
        "download_dir": args.cache_dir,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype or "auto",
    }
    if args.disable_custom_all_reduce and "disable_custom_all_reduce" in inspect.signature(LLM).parameters:
        llm_kwargs["disable_custom_all_reduce"] = True
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    grouped: dict[Path, list[tuple[int, int, dict[str, Any]]]] = {}
    for path, seed, index, record in todo:
        grouped.setdefault(path, []).append((seed, index, record))

    for path, items in grouped.items():
        records = json.loads(path.read_text(encoding="utf-8"))
        for seed, index, record in items:
            prompt = record["prompt"]
            prompt_text = format_generation_prompt(tokenizer, prompt)
            repaired_text = None
            used_seed = None
            attempts: list[dict[str, Any]] = []
            for attempt in range(1, args.max_attempts + 1):
                repair_seed = seed + args.repair_seed_stride * attempt + index
                sampling_params = SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    seed=repair_seed,
                )
                output = llm.generate([prompt_text], sampling_params, use_tqdm=False)[0]
                text = output.outputs[0].text
                attempts.append(
                    {
                        "attempt": attempt,
                        "seed": repair_seed,
                        "words": len(text.split()),
                        "think_leak": has_think_tag(text),
                    }
                )
                if not has_think_tag(text):
                    repaired_text = text
                    used_seed = repair_seed
                    break

            fallback_stripped = False
            if repaired_text is None:
                # Some UltraFeedback prompts explicitly request "stream of consciousness".
                # If Qwen3 keeps emitting a literal closing tag, preserve the original
                # seeded content and remove only the tag tokens required by the Qwen3
                # non-thinking sanity check. The metadata records this fallback.
                repaired_text = (
                    record.get("generated_text", "")
                    .replace("<think>", "")
                    .replace("</think>", "")
                    .replace("<THINK>", "")
                    .replace("</THINK>", "")
                )
                used_seed = seed
                fallback_stripped = True
                if has_think_tag(repaired_text):
                    raise SystemExit(
                        f"failed to strip thinking-token leakage for {path.name}:{index} "
                        f"after {args.max_attempts} attempts"
                    )

            records[index]["generated_text"] = repaired_text
            records[index]["think_repair"] = {
                "original_seed": seed,
                "repair_seed": used_seed,
                "attempts": attempts,
                "fallback_stripped_literal_tags": fallback_stripped,
                "reason": "literal <think>/</think> tag in generated_text",
            }
            summary["repairs"].append(
                {
                    "file": path.name,
                    "index": index,
                    "original_seed": seed,
                    "repair_seed": used_seed,
                    "attempts": attempts,
                    "fallback_stripped_literal_tags": fallback_stripped,
                }
            )

        path.write_text(json.dumps(records, ensure_ascii=False, indent=4), encoding="utf-8")

    remaining = 0
    for path in output_files:
        records = json.loads(path.read_text(encoding="utf-8"))
        remaining += sum(1 for record in records if has_think_tag(record.get("generated_text", "")))
    summary["remaining_leak_count"] = remaining
    (output_dir / "think_repair_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if remaining:
        raise SystemExit(f"remaining thinking-token leakage after repair: {remaining}")


if __name__ == "__main__":
    main()
