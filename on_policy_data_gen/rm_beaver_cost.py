#!/usr/bin/env python3
"""Score responses with a Safe-RLHF Beaver cost model; output -cost as harmlessness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaModel, LlamaPreTrainedModel


class LlamaForScalarScore(LlamaPreTrainedModel):
    """Minimal inference-only equivalent of Safe-RLHF's LlamaForScore."""

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.score_head = nn.Linear(
            config.hidden_size,
            int(getattr(config, "score_dim", 1)),
            bias=bool(getattr(config, "score_bias", True)),
        )
        self.post_init()

    def forward(self, input_ids, attention_mask):
        hidden = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        token_scores = self.score_head(hidden).float()
        end_index = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
        batch = torch.arange(token_scores.shape[0], device=token_scores.device)
        return token_scores[batch, end_index]


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
        with tqdm(total=len(rows), desc="Beaver cost") as progress:
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
                        batch, return_tensors="pt", padding=True, truncation=True,
                        max_length=args.max_seq_length,
                    ).to(args.device)
                    with torch.inference_mode():
                        costs = model(**encoded).squeeze(-1)
                    scores.extend((-costs.float()).cpu().tolist())
                offset = 0
                for row, count in zip(sample_rows, counts):
                    values = [float(value) for value in scores[offset:offset + count]]
                    offset += count
                    target.write(json.dumps({
                        **row,
                        "all_rm_scores": values,
                        "score_definition": "harmlessness=-SafeRLHF_cost_end_score",
                        "model_revision": args.model_revision,
                    }, ensure_ascii=False) + "\n")
                progress.update(len(sample_rows))


if __name__ == "__main__":
    main()
