import argparse
import json
import os
from typing import Dict, List, Iterable, Any

import torch
from torch import nn
from tqdm import tqdm


def _patch_transformers_gguf_compat():
    try:
        import transformers.integrations as integrations
    except Exception:
        return

    config_mapping = getattr(integrations, "GGUF_CONFIG_MAPPING", None)
    if not isinstance(config_mapping, dict):
        config_mapping = {}
    integrations.GGUF_CONFIG_MAPPING = config_mapping

    tokenizer_mapping = getattr(integrations, "GGUF_TOKENIZER_MAPPING", None)
    if not isinstance(tokenizer_mapping, dict):
        tokenizer_mapping = {}
    tokenizer_mapping.setdefault("tokenizer", {})
    tokenizer_mapping.setdefault("tokenizer_config", {})
    integrations.GGUF_TOKENIZER_MAPPING = tokenizer_mapping

    if not hasattr(integrations, "_gguf_parse_value"):
        def _gguf_parse_value(_value, data_type):
            raise RuntimeError("GGUF checkpoint parsing is unavailable in this image.")

        integrations._gguf_parse_value = _gguf_parse_value

    try:
        import transformers.generation as generation
        if not hasattr(generation, "GenerationMixin"):
            try:
                from transformers.generation.utils import GenerationMixin
            except Exception:
                class GenerationMixin:
                    pass
            generation.GenerationMixin = GenerationMixin
    except Exception:
        return


_patch_transformers_gguf_compat()
del _patch_transformers_gguf_compat

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaModel, LlamaPreTrainedModel
from transformers.pipelines import pipeline
from transformers.pipelines.text_classification import TextClassificationPipeline


# ======================
# 1. Athene model
# ======================
class AtheneForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.CLS_ID = 128003
        self.post_init()

    def get_device(self):
        return self.model.device

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        **kwargs,
    ):
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = transformer_outputs.hidden_states[-1]
        rewards = self.v_head(hidden_states).squeeze(-1)

        bs = int(input_ids.shape[0])
        scores = []

        for i in range(bs):
            c_inds = (input_ids[i] == self.CLS_ID).nonzero()
            if c_inds.numel() == 0:
                c_ind = -1
            else:
                c_ind = c_inds[-1].item()
            scores.append(rewards[i, c_ind])

        scores = torch.stack(scores)
        return {"scores": scores}


# ======================
# 2. Pipeline
# ======================
class AtheneRewardPipeline(TextClassificationPipeline):
    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, torch.Tensor]:
        """
        inputs:  messages (list[{"role": ..., "content": ...}])
        """
        return_tensors = getattr(self, "framework", "pt") or "pt"

        formatted = self.tokenizer.apply_chat_template(
            inputs,
            tokenize=False,
        )

        formatted = formatted + self.tokenizer.cls_token

        return self.tokenizer(
            formatted,
            return_tensors=return_tensors,
            max_length=4096,
            padding="longest",
            truncation=True,
            add_special_tokens=False,
        )

    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True):
        return float(model_outputs["scores"].cpu().float().item())


# ======================
# 3. IO ：json / jsonl
# ======================
def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_json(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2 structures：list 或 {"data": [...]} / {"train": [...]}
    if isinstance(data, list):
        for item in data:
            yield item
    elif isinstance(data, dict):

        for v in data.values():
            if isinstance(v, list):
                for item in v:
                    yield item
                break
    else:
        raise ValueError("Unsupported json structure")


def iter_dataset(path: str) -> Iterable[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        return read_jsonl(path)
    elif ext == ".json":
        return read_json(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}, only .json / .jsonl are supported.")


# ======================
# 4. scoring
# ======================
def build_messages(prompt: str, answer: str) -> List[Dict[str, str]]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]


def chunks(items, chunk_size):
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input json/jsonl file (gemma2_ufb_part1_split1.jsonl)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to output jsonl file")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HF cache dir for model/tokenizer")
    parser.add_argument("--model_name", type=str, default="Nexusflow/Athene-RM-8B",
                        help="Hugging Face model name or local path")
    parser.add_argument("--revision", type=str, default=None,
                        help="Optional exact Hugging Face revision to load")
    parser.add_argument("--local_files_only", action="store_true",
                        help="Require the model and tokenizer to be present in the local cache")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for Athene pipeline")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for reward scoring")
    parser.add_argument("--sample_batch_size", type=int, default=32,
                        help="Number of prompts to group before flattening responses for scoring")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional cap on the number of prompt samples to score")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Number of input shards for data-parallel scoring")
    parser.add_argument("--shard_index", type=int, default=0,
                        help="Zero-based shard index for data-parallel scoring")
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    # ======================
    # 4.1 load model and tokenizer
    # ======================
    print("Loading Athene-RM-8B ...")
    model = AtheneForSequenceClassification.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )

    if tokenizer.cls_token is None:
        raise ValueError("Tokenizer has no cls_token, but Athene RM expects one.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    rm_pipe = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        pipeline_class=AtheneRewardPipeline,
        device=0 if device.startswith("cuda") else -1,
    )

    # ======================
    # 4.2 traverse data
    # ======================
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    fout = open(args.output_file, "w", encoding="utf-8")

    data = list(iter_dataset(args.input_file))
    if args.max_samples is not None:
        data = data[:args.max_samples]
    data = [row for idx, row in enumerate(data) if idx % args.num_shards == args.shard_index]
    print(f"Scoring {len(data)} samples")
    if args.max_samples is not None:
        print(f"max_samples applied: {args.max_samples}")
    print(f"batch_size={args.batch_size}, sample_batch_size={args.sample_batch_size}")
    print(f"shard={args.shard_index}/{args.num_shards}")

    import numpy as np

    with tqdm(total=len(data), desc="Scoring") as pbar:
        for sample_batch in chunks(data, args.sample_batch_size):
            all_messages = []
            response_counts = []

            for sample in sample_batch:
                prompt = sample["prompt"]
                responses: List[str] = sample["all_generated_responses"]
                response_counts.append(len(responses))
                all_messages.extend(build_messages(prompt, resp) for resp in responses)

            flat_scores: List[float] = rm_pipe(
                all_messages,
                batch_size=args.batch_size,
            ) if all_messages else []

            offset = 0
            for sample, count in zip(sample_batch, response_counts):
                prompt = sample["prompt"]
                responses: List[str] = sample["all_generated_responses"]
                scores = flat_scores[offset:offset + count]
                offset += count

                out_sample = dict(sample)
                if not responses:
                    out_sample["all_rm_scores"] = []
                    out_sample["chosen"] = []
                    out_sample["rejected"] = []
                    fout.write(json.dumps(out_sample, ensure_ascii=False) + "\n")
                    continue

                scores_array = np.array(scores, dtype=float)
                best_idx = int(scores_array.argmax())
                worst_idx = int(scores_array.argmin())

                best_resp = responses[best_idx]
                worst_resp = responses[worst_idx]

                out_sample["all_rm_scores"] = [float(s) for s in scores]
                out_sample["chosen"] = build_messages(prompt, best_resp)
                out_sample["rejected"] = build_messages(prompt, worst_resp)

                fout.write(json.dumps(out_sample, ensure_ascii=False) + "\n")

            pbar.update(len(sample_batch))

    fout.close()
    print("Done! Saved to", args.output_file)


if __name__ == "__main__":
    main()
