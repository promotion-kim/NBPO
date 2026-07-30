#!/usr/bin/env python3
"""Generate deterministic samples from a saved checkpoint or base model."""

from __future__ import annotations

import argparse
import inspect
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def prompt_records(path: Path, limit: int) -> list[dict]:
    records = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        records.append(obj)
        if len(records) >= limit:
            break
    return records


def apply_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        signature = inspect.signature(tokenizer.apply_chat_template)
        supports_extra_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if "enable_thinking" in signature.parameters or supports_extra_kwargs:
            kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(messages, **kwargs)
    return prompt


def token_run(text: str) -> int:
    tokens = TOKEN_RE.findall(text.lower())
    best = 0
    prev = None
    cur = 0
    for token in tokens:
        if token == prev:
            cur += 1
        else:
            prev = token
            cur = 1
        best = max(best, cur)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompts-file", type=Path, default=Path("data/gemma2_ufb_part1_test.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--formatted-prompts",
        action="store_true",
        help="Use each prompt string as already formatted model input instead of applying the chat template.",
    )
    args = parser.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    model.to(args.device)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for record in prompt_records(args.prompts_file, args.num_prompts):
            prompt = str(record.get("prompt") or "")
            text = prompt if args.formatted_prompts else apply_prompt(tokenizer, prompt)
            inputs = tokenizer(text, return_tensors="pt").to(args.device)
            with torch.no_grad():
                generation_kwargs = {
                    "do_sample": args.do_sample,
                    "max_new_tokens": args.max_new_tokens,
                    "repetition_penalty": 1.0,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                if args.do_sample:
                    generation_kwargs.update(
                        {
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "top_k": args.top_k,
                        }
                    )
                output_ids = model.generate(
                    **inputs,
                    **generation_kwargs,
                )
            new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            out = {
                "model_path": args.model_path,
                "prompt_id": record.get("prompt_id"),
                "prompt": prompt,
                "response": response,
                "response_chars": len(response),
                "max_token_run": token_run(response),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
