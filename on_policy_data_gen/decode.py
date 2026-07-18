import os
import argparse
import json
import inspect
import math
import time
from tqdm import tqdm
from datasets import load_dataset

parser = argparse.ArgumentParser(description='Decode with vllm')
parser.add_argument('--data_dir', type=str, default="HuggingFaceH4/ultrafeedback_binarized",
                    help='Directory containing the data')
parser.add_argument('--model', type=str, default="google/gemma-2-9b-it",
                    help='Path to the LLM model')
parser.add_argument('--temperature', type=float, default=0.8,
                    help='Temperature for sampling')
parser.add_argument('--top_p', type=float, default=0.95,
                    help='Top-p probability for sampling')
parser.add_argument('--max_tokens', type=int, default=4096,
                    help='Maximum number of tokens to generate')
parser.add_argument('--output_dir', type=str, default="datasets/gemma2_ultrafeedback",
                    help='Output directory')
parser.add_argument('--num_gpu', type=int, default=4)
parser.add_argument('--sanity_check', action='store_true', help="Enable sanity check (only use 100 samples)")
parser.add_argument('--batch_size', type=int, default=512,
                    help='Number of prompts to submit to vLLM per generate() call')
parser.add_argument('--cache_dir', type=str, default=None,
                    help='Cache directory for model and dataset')
parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                    help='A list of random seeds to run')
parser.add_argument('--attention_backend', type=str, default=None,
                    help='Optional vLLM attention backend, e.g. FLASHINFER or XFORMERS. Defaults to vLLM auto.')
parser.add_argument('--dtype', type=str, default=None,
                    help='Optional vLLM dtype override, e.g. bfloat16, float16, or auto.')
parser.add_argument('--gpu_memory_utilization', type=float, default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
                    help='Fraction of GPU memory vLLM may reserve for model/KV cache.')
parser.add_argument('--enable_thinking', action='store_true',
                    help='Enable Qwen3 thinking mode when the tokenizer supports it. Defaults to disabled.')
parser.add_argument('--disable_custom_all_reduce', action='store_true',
                    help='Disable vLLM custom all-reduce kernels for hardware/runtime combinations where they fail.')
args = parser.parse_args()

if args.attention_backend:
    os.environ["VLLM_ATTENTION_BACKEND"] = args.attention_backend

from vllm import LLM, SamplingParams

print(args, flush=True)


def format_duration(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def load_partial_outputs(partial_file, prompts):
    output_data = []
    if not os.path.exists(partial_file):
        return output_data

    with open(partial_file, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            expected_idx = len(output_data)
            if expected_idx >= len(prompts) or record.get("prompt") != prompts[expected_idx]:
                raise ValueError(
                    f"Partial output prompt mismatch at {partial_file}:{line_number}; "
                    "remove the partial file or decode with the same input ordering."
                )
            output_data.append(record)
    return output_data

data_dir = args.data_dir
llm_kwargs = {
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

if os.path.exists(data_dir):
    # If the input is an existing local file path
    print("Detected local file path, loading local file...", flush=True)
    # Use the 'json' loader, which supports both .json and .jsonl files
    train_dataset = load_dataset("json", data_files=data_dir, split="train")
else:
    # If not a local file path, assume it is a dataset name on Hugging Face Hub
    print("No local file detected, trying to load from Hugging Face Hub...", flush=True)
    train_dataset = load_dataset(data_dir, split="train")

# If sanity check is enabled, only select a small number of samples
if args.sanity_check:
    print("Performing sanity check, using only 100 samples.", flush=True)
    train_dataset = train_dataset.select(range(min(len(train_dataset), 100)))

prompts = sorted(list(set(train_dataset['prompt'])))


def format_generation_prompt(prompt):
    messages = [{'role': 'user', 'content': prompt}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    signature = inspect.signature(tokenizer.apply_chat_template)
    supports_extra_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if "enable_thinking" in signature.parameters or supports_extra_kwargs:
        kwargs["enable_thinking"] = bool(args.enable_thinking)
    return tokenizer.apply_chat_template(messages, **kwargs)


conversations = [format_generation_prompt(prompt) for prompt in prompts]
os.makedirs(args.output_dir, exist_ok=True)
batch_size = max(1, args.batch_size)
generate_kwargs = {}
if "use_tqdm" in inspect.signature(llm.generate).parameters:
    generate_kwargs["use_tqdm"] = False

for seed in args.seeds:
    print(f"\n--- Processing for seed {seed} ---", flush=True)

    sampling_params = SamplingParams(temperature=args.temperature,
                                     top_p=args.top_p,
                                     max_tokens=args.max_tokens,
                                     seed=seed,)
    output_file = f'output_{seed}.json'
    output_path = os.path.join(args.output_dir, output_file)
    partial_path = os.path.join(args.output_dir, f'output_{seed}.partial.jsonl')

    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_output = json.load(f)
        if len(existing_output) == len(prompts):
            print(f"Skipping seed {seed}; found complete {output_path}", flush=True)
            continue

    output_data = load_partial_outputs(partial_path, prompts)
    resume_count = len(output_data)
    if resume_count:
        print(f"Resuming seed {seed} from {resume_count}/{len(conversations)} prompts in {partial_path}", flush=True)

    total_chunks = math.ceil((len(conversations) - resume_count) / batch_size)
    print(
        f"Submitting {len(conversations) - resume_count}/{len(conversations)} prompts to vLLM "
        f"in chunks of {batch_size} ({total_chunks} chunks).",
        flush=True,
    )

    seed_start = time.time()
    session_completed = 0

    try:
        with tqdm(total=len(conversations), initial=resume_count, desc=f"seed {seed}", unit="prompt") as progress:
            for chunk_number, chunk_start in enumerate(range(resume_count, len(conversations), batch_size), start=1):
                chunk_end = min(chunk_start + batch_size, len(conversations))
                chunk_outputs = llm.generate(
                    conversations[chunk_start:chunk_end],
                    sampling_params,
                    **generate_kwargs,
                )
                chunk_records = []
                for i, output in zip(range(chunk_start, chunk_end), chunk_outputs):
                    chunk_records.append({
                        'prompt': prompts[i],
                        "format_prompt": output.prompt,
                        'generated_text': output.outputs[0].text,
                    })

                with open(partial_path, 'a', encoding='utf-8') as f:
                    for record in chunk_records:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                output_data.extend(chunk_records)
                session_completed += len(chunk_records)
                progress.update(len(chunk_records))

                elapsed = time.time() - seed_start
                rate = session_completed / elapsed if elapsed > 0 else 0.0
                remaining = len(conversations) - len(output_data)
                eta = remaining / rate if rate > 0 else float("inf")
                print(
                    f"[decode] seed={seed} chunk={chunk_number}/{total_chunks} "
                    f"prompts={len(output_data)}/{len(conversations)} "
                    f"rate={rate:.2f} prompt/s elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
                    flush=True,
                )

        print("Generation complete. Writing final output...", flush=True)
    except Exception as e:
        print(f"Generation failed with error: {e}", flush=True)
        raise

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Outputs saved to {output_path}", flush=True)

print("\nAll seeds processed.", flush=True)
