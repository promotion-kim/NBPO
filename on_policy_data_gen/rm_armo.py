import torch


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

    try:
        import transformers.models.llama.modeling_llama as llama_modeling
        if not hasattr(llama_modeling, "LLAMA_INPUTS_DOCSTRING"):
            llama_modeling.LLAMA_INPUTS_DOCSTRING = ""
    except Exception:
        return


_patch_transformers_gguf_compat()
del _patch_transformers_gguf_compat

from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
import json
import os
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from transformers.hf_argparser import HfArgumentParser


@dataclass
class ScriptArgs:
    cache_dir: str = field(metadata={"help": "cache directory"})
    input_file: str = field(metadata={"help": "input json/jsonl"})
    output_file: str = field(metadata={"help": "output jsonl"})
    device: str = field(default="cuda", metadata={"help": "device for reward scoring"})
    batch_size: int = field(default=16, metadata={"help": "number of prompt-response pairs per forward pass"})
    sample_batch_size: int = field(default=32, metadata={"help": "number of prompts to group before scoring"})
    max_samples: Optional[int] = field(default=None, metadata={"help": "optional cap on prompt samples to score"})
    max_seq_length: int = field(default=4096, metadata={"help": "maximum sequence length"})
    local_files_only: bool = field(default=False, metadata={"help": "only load model files from local cache"})
    revision: Optional[str] = field(default=None, metadata={"help": "exact Hugging Face revision"})
    num_shards: int = field(default=1, metadata={"help": "number of input shards for data parallel scoring"})
    shard_index: int = field(default=0, metadata={"help": "0-based shard index for data parallel scoring"})
    reward_attribute_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "If set, score with this ArmoRM multi-objective head from output.rewards "
                "instead of the gated preference score. Example: beavertails-is_safe"
            )
        },
    )
    reward_attribute_names: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional comma-separated ArmoRM reward-head names. If omitted, the script "
                "tries model/config attributes and then falls back to the public ArmoRM v0.1 list."
            )
        },
    )

parser = HfArgumentParser(ScriptArgs)
args = parser.parse_args_into_dataclasses()[0]

# parameters configuration
MODEL_NAME = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
CACHE_DIR = args.cache_dir
INPUT_FILE = args.input_file
OUTPUT_FILE = args.output_file
DEVICE = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

MAX_SEQ_LENGTH = args.max_seq_length
ARMO_GATE_TOKEN_PATTERN = [128009, 128006, 78191, 128007, 271]

ARMO_V01_ATTRIBUTES = [
    "helpsteer-helpfulness",
    "helpsteer-correctness",
    "helpsteer-coherence",
    "helpsteer-complexity",
    "helpsteer-verbosity",
    "ultrafeedback-overall_score",
    "ultrafeedback-instruction_following",
    "ultrafeedback-truthfulness",
    "ultrafeedback-honesty",
    "ultrafeedback-helpfulness",
    "beavertails-is_safe",
    "prometheus-score",
    "argilla-overall_quality",
    "argilla-judge_lm",
    "code-complexity",
    "code-style",
    "code-explanation",
    "code-instruction-following",
    "code-readability",
]


def parse_attribute_names(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise ValueError("--reward_attribute_names was provided but no names were parsed")
    return names


def resolve_attribute_names(model, explicit_names: Optional[str]) -> Tuple[List[str], str]:
    parsed = parse_attribute_names(explicit_names)
    if parsed is not None:
        return parsed, "cli"

    candidates = [
        getattr(model, "reward_attributes", None),
        getattr(model, "reward_objectives", None),
        getattr(model, "attributes", None),
        getattr(model.config, "reward_attributes", None),
        getattr(model.config, "reward_objectives", None),
        getattr(model.config, "attributes", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, (list, tuple)) and all(isinstance(x, str) for x in candidate):
            return list(candidate), "model"

    return ARMO_V01_ATTRIBUTES[:], "armo_v0.1_default"


def load_data(file_path, max_samples=None):
    suffix = os.path.splitext(file_path)[1].lower()


    if suffix in [".jsonl", ".jsonlines", ".ljson"]:
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
                if max_samples is not None and len(records) >= max_samples:
                    break
        return records

    elif suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data[:max_samples] if max_samples is not None else data

        elif isinstance(data, dict):
            return [data]

        else:
            raise ValueError("JSON file has to be list or dict")

    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def chunks(items, chunk_size):
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def truncate_user_content(conversation, max_chars):
    if max_chars is None:
        return conversation
    out = []
    for message in conversation:
        copied = dict(message)
        if copied.get("role") == "user" and isinstance(copied.get("content"), str):
            content = copied["content"]
            if len(content) > max_chars:
                copied["content"] = content[:max_chars]
        out.append(copied)
    return out


def score_batch(conversations, tokenizer, model, device, reward_attribute_index):
    encoded = tokenizer.apply_chat_template(
        conversations,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )
    if isinstance(encoded, torch.Tensor):
        input_ids = encoded.to(device)
    elif isinstance(encoded, dict) or hasattr(encoded, "__getitem__"):
        input_ids = encoded["input_ids"].to(device)
    else:
        raise TypeError(f"Unsupported apply_chat_template return type: {type(encoded)!r}")

    with torch.inference_mode():
        output = model(input_ids)
        if reward_attribute_index is None:
            batch_scores = output.score
        else:
            if not hasattr(output, "rewards"):
                raise RuntimeError("ArmoRM output has no `rewards` field for multi-objective scoring")
            rewards = output.rewards
            if rewards.ndim != 2:
                raise RuntimeError(f"Expected output.rewards to be rank-2, got shape={tuple(rewards.shape)}")
            if reward_attribute_index >= rewards.shape[1]:
                raise RuntimeError(
                    f"Reward attribute index {reward_attribute_index} is outside rewards shape {tuple(rewards.shape)}"
                )
            batch_scores = rewards[:, reward_attribute_index]
        return batch_scores.float().view(-1).cpu().tolist()


def score_conversation_with_prompt_fallback(conversation, tokenizer, model, device, reward_attribute_index):
    for max_chars in (4096, 2048, 1024, 512, 256, 0):
        try:
            safe_conversation = truncate_user_content(conversation, max_chars)
            return score_batch([safe_conversation], tokenizer, model, device, reward_attribute_index)[0]
        except ValueError as exc:
            if "Token pattern not found" not in str(exc):
                raise
    raise ValueError("Token pattern not found even after prompt fallback")


def score_conversations(
    conversations,
    tokenizer,
    model,
    device,
    batch_size,
    reward_attribute_index: Optional[int],
):
    scores = []
    for batch in chunks(conversations, batch_size):
        try:
            scores.extend(score_batch(batch, tokenizer, model, device, reward_attribute_index))
        except ValueError as exc:
            if "Token pattern not found" not in str(exc):
                raise
            print(
                "[warn] ArmoRM gating token was truncated in a batch; "
                "retrying those samples individually with user-prompt truncation.",
                flush=True,
            )
            for conversation in batch:
                scores.append(
                    score_conversation_with_prompt_fallback(
                        conversation,
                        tokenizer,
                        model,
                        device,
                        reward_attribute_index,
                    )
                )

    return scores


def main():
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")

    print(f"Step 1: loading {MODEL_NAME}...")
    print(f"         - Cache Dir: {CACHE_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        revision=args.revision,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    tokenizer.truncation_side = "right"

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,  # must enable for armorm
        cache_dir=CACHE_DIR,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    model.to(DEVICE)
    model.eval()
    print("         - model loading completed")
    reward_attribute_index = None
    reward_attribute_names = None
    if args.reward_attribute_name:
        reward_attribute_names, source = resolve_attribute_names(model, args.reward_attribute_names)
        if args.reward_attribute_name not in reward_attribute_names:
            raise ValueError(
                f"Reward attribute {args.reward_attribute_name!r} not found in {source} attribute list: "
                f"{reward_attribute_names}"
            )
        reward_attribute_index = reward_attribute_names.index(args.reward_attribute_name)
        print(
            "         - scoring ArmoRM reward head "
            f"{args.reward_attribute_name!r} at index {reward_attribute_index} (attribute_source={source})"
        )
    else:
        print("         - scoring gated preference output.score")

    print(f"Step 2: processing {INPUT_FILE}...")
    dataset = [
        row
        for idx, row in enumerate(load_data(INPUT_FILE, args.max_samples))
        if idx % args.num_shards == args.shard_index
    ]
    total_samples = len(dataset)
    print(f"         - found {total_samples} samples")
    if args.max_samples is not None:
        print(f"         - max_samples applied: {args.max_samples}")
    print(f"         - shard={args.shard_index}/{args.num_shards}")
    print(f"         - batch_size={args.batch_size}, sample_batch_size={args.sample_batch_size}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:

        with tqdm(total=total_samples, desc="Scoring and creating pairs") as pbar:
            for sample_batch in chunks(dataset, args.sample_batch_size):
                conversations = []
                response_counts = []

                for sample in sample_batch:
                    prompt = sample['prompt']
                    responses = sample['all_generated_responses']
                    response_counts.append(len(responses))
                    for resp in responses:
                        conversations.append([
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": resp}
                        ])

                flat_scores = score_conversations(
                    conversations,
                    tokenizer,
                    model,
                    DEVICE,
                    args.batch_size,
                    reward_attribute_index,
                ) if conversations else []

                offset = 0
                for sample, count in zip(sample_batch, response_counts):
                    prompt = sample['prompt']
                    responses = sample['all_generated_responses']
                    scores = flat_scores[offset:offset + count]
                    offset += count

                    if not responses:
                        sample['all_rm_scores'] = []
                        sample['chosen'] = []
                        sample['rejected'] = []
                        f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
                        continue

                    max_idx = scores.index(max(scores))
                    min_idx = scores.index(min(scores))

                    if max_idx == min_idx:
                        if len(scores) == 1:
                            chosen_response = responses[0]
                            rejected_response = responses[0]
                        else:
                            chosen_response = responses[0]
                            rejected_response = responses[1]
                    else:
                        chosen_response = responses[max_idx]
                        rejected_response = responses[min_idx]

                    sample['all_rm_scores'] = scores
                    sample['armo_score_source'] = (
                        args.reward_attribute_name if args.reward_attribute_name else "gated_preference_score"
                    )
                    if args.reward_attribute_name:
                        sample['armo_reward_attribute_name'] = args.reward_attribute_name
                        sample['armo_reward_attribute_index'] = reward_attribute_index
                        sample['armo_reward_attribute_names'] = reward_attribute_names

                    sample['chosen'] = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": chosen_response}
                    ]

                    sample['rejected'] = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": rejected_response}
                    ]

                    f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')

                pbar.update(len(sample_batch))

    print(f"\nStep 3: all done!")
    print(f"         - ArmoRM formatted data saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
