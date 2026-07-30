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


_patch_transformers_gguf_compat()
del _patch_transformers_gguf_compat

from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
import json
import os
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Score responses with Skywork Reward Model")

    parser.add_argument(
        "--model_name",
        type=str,
        default="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
        help="HuggingFace model name or local path"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/cache",
        help="Cache directory for HF models"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional exact Hugging Face revision to load."
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Require the model and tokenizer to be present in the local cache."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input .json or .jsonl file (all_outputs.json)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output .jsonl file (scored outputs)"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Maximum sequence length for the tokenizer"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "flash_attention_2", "sdpa"],
        help="Attention implementation for the reward model. Use eager when flash-attn is unavailable."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for reward scoring when not using device_map."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of prompt-response pairs per reward-model forward pass."
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=32,
        help="Number of prompts to group before flattening responses for batched scoring."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on the number of prompt samples to score."
    )
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Number of input shards for data-parallel scoring.")
    parser.add_argument("--shard_index", type=int, default=0,
                        help="Zero-based shard index for data-parallel scoring.")

    return parser.parse_args()

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
            raise ValueError("JSON format has to be list or dict")

    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def chunks(items, chunk_size):
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def score_texts(texts, tokenizer, model, device, max_seq_length, batch_size):
    scores = []
    for batch in chunks(texts, batch_size):
        # The chat template already contains the model's BOS/control tokens.
        # Encoding with default special-token insertion would duplicate BOS.
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        ).to(device)

        with torch.inference_mode():
            logits = model(**inputs).logits
            scores.extend(logits.squeeze(-1).float().cpu().tolist())

    return scores




def main(args):
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")
    MODEL_NAME = args.model_name
    CACHE_DIR = args.cache_dir
    INPUT_FILE = args.input_file
    OUTPUT_FILE = args.output_file
    MAX_SEQ_LENGTH = args.max_seq_length
    DEVICE = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    print(f"Step 1: loading {MODEL_NAME}...")
    print(f"         - Cache Dir: {CACHE_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        num_labels=1,
        cache_dir=CACHE_DIR,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    model.to(DEVICE)
    model.eval()
    print("         - model loaded.")

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
    print(f"         - batch_size={args.batch_size}, sample_batch_size={args.sample_batch_size}")
    print(f"         - shard={args.shard_index}/{args.num_shards}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        with tqdm(total=total_samples, desc="Scoring responses") as pbar:
            for sample_batch in chunks(dataset, args.sample_batch_size):
                formatted_texts = []
                response_counts = []

                for sample in sample_batch:
                    prompt = sample['prompt']
                    responses = sample['all_generated_responses']
                    response_counts.append(len(responses))

                    conversations_batch = [
                        [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": resp}
                        ]
                        for resp in responses
                    ]
                    if not conversations_batch:
                        continue

                    formatted_batch = tokenizer.apply_chat_template(
                        conversations_batch,
                        tokenize=False,
                    )

                    formatted_texts.extend(formatted_batch)

                flat_scores = score_texts(
                    formatted_texts,
                    tokenizer,
                    model,
                    DEVICE,
                    MAX_SEQ_LENGTH,
                    args.batch_size,
                ) if formatted_texts else []

                offset = 0
                for sample, count in zip(sample_batch, response_counts):
                    prompt = sample['prompt']
                    responses = sample['all_generated_responses']
                    scores = flat_scores[offset:offset + count]
                    offset += count

                    if not responses:
                        sample['skywork_v2_scores'] = []
                        sample['all_rm_scores'] = []
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

    print(f"\nStep 3: all done！")
    print(f"         - results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
