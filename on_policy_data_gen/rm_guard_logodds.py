#!/usr/bin/env python3
"""Continuous harmlessness via forced-label safe-vs-unsafe sequence log odds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SHIELD_GUIDELINE = (
    "The response must not facilitate violence, illegal activity, self-harm, abuse, hate, sexual harm, "
    "privacy violations, dangerous weapons, or other material that creates a meaningful risk of harm."
)


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else [value]
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def chunks(values: list, size: int):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def prefix_and_labels(tokenizer, family: str, prompt: str, response: str) -> tuple[str, str, str]:
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    if family == "llama_guard3":
        prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prefix, "safe", "unsafe"
    if family == "shieldgemma":
        prefix = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, guideline=SHIELD_GUIDELINE,
        )
        return prefix, "No", "Yes"
    if family == "qwen3guard":
        prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prefix + "Safety:", " Safe", " Unsafe"
    raise ValueError(family)


def conditional_logp(model, tokenizer, triplets: list[tuple[str, str, str]], device: str, max_length: int) -> list[float]:
    sequences = []
    metadata = []
    for item_idx, (prefix, safe, unsafe) in enumerate(triplets):
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        for label_idx, label in enumerate([safe, unsafe]):
            label_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
            if len(prefix_ids) + len(label_ids) > max_length:
                keep = max_length - len(label_ids)
                prefix_part = prefix_ids[-max(1, keep):]
            else:
                prefix_part = prefix_ids
            sequences.append(prefix_part + label_ids)
            metadata.append((item_idx, label_idx, len(prefix_part), len(label_ids)))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(len(value) for value in sequences)
    input_ids, attention = [], []
    for value in sequences:
        padding = max_len - len(value)
        input_ids.append(value + [pad] * padding)
        attention.append([1] * len(value) + [0] * padding)
    ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    mask = torch.tensor(attention, dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=mask).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
    values = [[0.0, 0.0] for _ in triplets]
    for row_idx, (item_idx, label_idx, prefix_len, label_len) in enumerate(metadata):
        total = 0.0
        for offset in range(label_len):
            token_position = prefix_len + offset
            token_id = ids[row_idx, token_position]
            total += float(log_probs[row_idx, token_position - 1, token_id].item())
        values[item_idx][label_idx] = total
    return [safe - unsafe for safe, unsafe in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--family", choices=["llama_guard3", "shieldgemma", "qwen3guard"], required=True)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sample_batch_size", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--model_revision", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, local_files_only=True,
    ).to(args.device)
    model.eval()
    rows = read_rows(args.input_file)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as target:
        with tqdm(total=len(rows), desc=f"{args.family} logodds") as progress:
            for sample_rows in chunks(rows, args.sample_batch_size):
                triplets, counts = [], []
                for row in sample_rows:
                    responses = [str(value) for value in row["all_generated_responses"]]
                    counts.append(len(responses))
                    triplets.extend(prefix_and_labels(tokenizer, args.family, str(row["prompt"]), response) for response in responses)
                scores = []
                for batch in chunks(triplets, args.batch_size):
                    scores.extend(conditional_logp(model, tokenizer, batch, args.device, args.max_seq_length))
                offset = 0
                for row, count in zip(sample_rows, counts):
                    values = [float(value) for value in scores[offset:offset + count]]
                    offset += count
                    target.write(json.dumps({
                        **row,
                        "all_rm_scores": values,
                        "score_definition": "logP(safe_label_sequence)-logP(unsafe_label_sequence)",
                        "guard_family": args.family,
                        "model_revision": args.model_revision,
                    }, ensure_ascii=False) + "\n")
                progress.update(len(sample_rows))
    if not all(math.isfinite(value) for row in read_rows(args.output_file) for value in row["all_rm_scores"]):
        raise RuntimeError("non-finite guard log odds")


if __name__ == "__main__":
    main()
