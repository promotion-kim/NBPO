#!/usr/bin/env python3
"""Score responses with the distinct Safe-RLHF Beaver reward head.

The cost and reward checkpoints share the same scalar-score architecture but
are separate released heads.  This script preserves the common JSONL schema of
``rm_beaver_cost.py`` while retaining the reward sign: higher values mean more
helpful according to the Beaver reward model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from on_policy_data_gen.rm_beaver_cost import LlamaForScalarScore, chunks, read_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sample_batch_size", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--model_revision", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model_name, local_files_only=True)
    model = LlamaForScalarScore.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(args.device)
    model.eval()

    rows = read_rows(args.input_file)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as target:
        with tqdm(total=len(rows), desc="Beaver reward") as progress:
            for sample_rows in chunks(rows, args.sample_batch_size):
                texts, counts = [], []
                for row in sample_rows:
                    prompt = str(row["prompt"])
                    responses = [str(value) for value in row["all_generated_responses"]]
                    counts.append(len(responses))
                    texts.extend(
                        f"BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:{response}"
                        for response in responses
                    )
                scores = []
                for batch in chunks(texts, args.batch_size):
                    encoded = tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=args.max_seq_length,
                    ).to(args.device)
                    with torch.inference_mode():
                        values = model(**encoded).squeeze(-1)
                    scores.extend(values.float().cpu().tolist())
                offset = 0
                for row, count in zip(sample_rows, counts):
                    values = [float(value) for value in scores[offset:offset + count]]
                    offset += count
                    target.write(json.dumps({
                        **row,
                        "all_rm_scores": values,
                        "score_definition": "helpfulness=SafeRLHF_reward_end_score",
                        "model_revision": args.model_revision,
                    }, ensure_ascii=False) + "\n")
                progress.update(len(sample_rows))


if __name__ == "__main__":
    main()
