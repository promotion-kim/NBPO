import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SAFETY_RE = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)
REFUSAL_RE = re.compile(r"Refusal:\s*(Yes|No)", re.IGNORECASE)
CATEGORY_RE = re.compile(
    r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|"
    r"Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|"
    r"Copyright Violation|Jailbreak|None)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score prompt-response candidates with Qwen3Guard response moderation."
    )
    parser.add_argument("--model_name", default="Qwen/Qwen3Guard-Gen-0.6B")
    parser.add_argument("--cache_dir", default="/ext_hdd/sjkim/huggingface")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=32,
        help="Number of prompts to flatten before batched guard generation.",
    )
    parser.add_argument("--max_input_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Require the model to already exist in the local cache.",
    )
    return parser.parse_args()


def read_records(path: str, max_samples: Optional[int]) -> Iterable[Tuple[int, Dict[str, Any]]]:
    suffix = os.path.splitext(path)[1].lower()
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        if suffix in {".jsonl", ".jsonlines"}:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                yield idx, json.loads(line)
                count += 1
                if max_samples is not None and count >= max_samples:
                    return
        elif suffix == ".json":
            data = json.load(f)
            records = data if isinstance(data, list) else [data]
            for idx, record in enumerate(records):
                yield idx, record
                count += 1
                if max_samples is not None and count >= max_samples:
                    return
        else:
            raise ValueError(f"Unsupported file suffix: {path}")


def chunks(items: List[Any], chunk_size: int) -> Iterable[List[Any]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def extract_label_categories_refusal(content: str) -> Tuple[str, List[str], Optional[str]]:
    safety_match = SAFETY_RE.search(content)
    refusal_match = REFUSAL_RE.search(content)
    if safety_match:
        safety = safety_match.group(1).capitalize()
    else:
        lowered = content.lower()
        if "unsafe" in lowered:
            safety = "Unsafe"
        elif "controversial" in lowered:
            safety = "Controversial"
        elif "safe" in lowered:
            safety = "Safe"
        else:
            safety = "Controversial"
    refusal = refusal_match.group(1).capitalize() if refusal_match else None
    categories = [match.group(1) for match in CATEGORY_RE.finditer(content)]
    return safety, categories, refusal


def safety_score(label: str) -> float:
    normalized = label.lower()
    if normalized == "safe":
        return 1.0
    if normalized == "controversial":
        return 0.5
    if normalized == "unsafe":
        return 0.0
    return 0.5


def response_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        last = value[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    if isinstance(value, dict):
        return str(value.get("content", ""))
    return str(value)


def format_messages(tokenizer: Any, prompt: str, responses: List[str]) -> List[str]:
    formatted = []
    for response in responses:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        formatted.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return formatted


def classify_batch(
    texts: List[str],
    tokenizer: Any,
    model: Any,
    device: str,
    max_input_length: int,
    max_new_tokens: int,
) -> List[Tuple[float, str, List[str], Optional[str], str]]:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    ).to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[:, input_len:]
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    rows = []
    for content in decoded:
        label, categories, refusal = extract_label_categories_refusal(content)
        rows.append((safety_score(label), label, categories, refusal, content))
    return rows


def row_for_scores(base: Dict[str, Any], scores: List[float]) -> Dict[str, Any]:
    responses = base["all_generated_responses"]
    best = max(range(len(scores)), key=lambda i: scores[i])
    worst = min(range(len(scores)), key=lambda i: scores[i])
    return {
        **base,
        "all_rm_scores": [float(score) for score in scores],
        "chosen": [
            {"role": "user", "content": base["prompt"]},
            {"role": "assistant", "content": responses[best]},
        ],
        "rejected": [
            {"role": "user", "content": base["prompt"]},
            {"role": "assistant", "content": responses[worst]},
        ],
    }


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"[load] {args.model_name} cache={args.cache_dir} device={device}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    # Decoder-only batched generation must left-pad so each sequence's final
    # prompt token aligns with the generation start position.
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        torch_dtype="auto",
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = [
        record
        for idx, record in read_records(args.input_file, args.max_samples)
        if idx % args.num_shards == args.shard_index
    ]
    print(
        f"[score] records={len(records)} shard={args.shard_index}/{args.num_shards} "
        f"batch_size={args.batch_size}"
    )

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as out:
        with tqdm(total=len(records), desc="Qwen3Guard") as pbar:
            for record_batch in chunks(records, args.sample_batch_size):
                formatted: List[str] = []
                response_counts: List[int] = []
                valid_records: List[Dict[str, Any]] = []
                for record in record_batch:
                    prompt = str(record.get("prompt", ""))
                    responses = [response_text(resp) for resp in record.get("all_generated_responses", [])]
                    if not responses:
                        continue
                    valid_records.append(record)
                    response_counts.append(len(responses))
                    formatted.extend(format_messages(tokenizer, prompt, responses))

                flat_rows: List[Tuple[float, str, List[str], Optional[str], str]] = []
                for batch in chunks(formatted, args.batch_size):
                    flat_rows.extend(
                        classify_batch(
                            batch,
                            tokenizer,
                            model,
                            device,
                            args.max_input_length,
                            args.max_new_tokens,
                        )
                    )

                offset = 0
                for record, count in zip(valid_records, response_counts):
                    rows = flat_rows[offset : offset + count]
                    offset += count
                    scores = [row[0] for row in rows]
                    labels = [row[1] for row in rows]
                    categories_by_response = [row[2] for row in rows]
                    refusals = [row[3] for row in rows]
                    raw_outputs = [row[4] for row in rows]

                    out_row = row_for_scores(record, scores)
                    out_row["qwen3guard_labels"] = labels
                    out_row["qwen3guard_categories"] = categories_by_response
                    out_row["qwen3guard_refusals"] = refusals
                    out_row["qwen3guard_raw_outputs"] = raw_outputs
                    out.write(json.dumps(out_row, ensure_ascii=False) + "\n")

                pbar.update(len(record_batch))

    print(f"[done] wrote {args.output_file}")


if __name__ == "__main__":
    main()
