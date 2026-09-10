"""Sample the shared UF-4 candidate pool: 8 learners and 8 comparators per prompt.

Every method and every seed reads this one pool, so it is generated once from the
original base under raw-policy sampling and never regenerated per arm. Each
occurrence keeps its own token event; duplicate text is kept as a separate
occurrence with its multiplicity intact, because the occurrence measure the
solver uses is uniform over draws, not over distinct strings.

Chunks are hash-verified and resumable, so the 200-prompt profile is literally
the first chunks of the full run: same prompts, same per-candidate seeds, same
settings file. Nothing is generated twice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
ROLES = ("learner", "comparator")
POOL = 8
SEED_NAMESPACE = "20260910-uf4-pool:"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def object_hash(obj):
    return digest(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


def read_jsonl(path):
    with open(path) as stream:
        for line in stream:
            line = line.strip()
            if line:
                yield json.loads(line)


def prompt_tokens(tokenizer, instruction, cap):
    """Chat-templated prompt, truncated from the FRONT if it exceeds the cap.

    keep_end matches the policy trainer's truncation_mode: when an instruction is
    too long the trailing text carries the actual request, so the head is what is
    dropped. Every truncation is recorded rather than silently applied.
    """
    ids = tokenizer.apply_chat_template([{"role": "user", "content": instruction}],
                                        tokenize=True, add_generation_prompt=True)
    if len(ids) <= cap:
        return ids, 0, instruction
    body = tokenizer(instruction, add_special_tokens=False)["input_ids"]
    dropped = 0
    text = instruction
    while len(ids) > cap and len(body) > 16:
        overflow = len(ids) - cap
        body = body[min(overflow + 8, len(body) - 16):]
        dropped += min(overflow + 8, max(len(body), 1))
        text = tokenizer.decode(body, skip_special_tokens=True)
        ids = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                            tokenize=True, add_generation_prompt=True)
    return ids, dropped, text


def make_event(prompt, role, index, seed, completion, tokenizer, terminal_ids, max_total):
    pid = prompt["prompt_id"]
    pids = prompt["prompt_token_ids"]
    rids = list(completion.token_ids)
    if not rids:
        raise ValueError(f"Empty sampled token event: {pid}/{role}/{index}")
    if completion.finish_reason not in ("stop", "length"):
        raise ValueError(f"Unexpected termination: {completion.finish_reason}")
    if completion.finish_reason == "stop" and rids[-1] not in terminal_ids:
        raise ValueError("Stop event does not retain its actually sampled terminal token")
    if len(pids) + len(rids) > max_total:
        raise ValueError("Sample would be truncated by training")
    text = tokenizer.decode(rids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    ids = pids + rids
    return {"prompt_id": pid, "split": prompt["split"], "source": prompt["source"],
            "prompt": prompt["prompt"], "role": role, "sample_index": index, "seed": seed,
            "candidate_id": f"{pid}:{role}:{index}", "response": text,
            "vllm_output_text": completion.text, "decode_text_equal": text == completion.text,
            "prompt_token_ids": pids, "response_token_ids": rids,
            "input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(pids) + rids,
            "token_event_sha256": object_hash({"input_ids": ids, "labels": [-100] * len(pids) + rids}),
            "response_sha256": digest(text), "n_tokens": len(rids),
            "finish_reason": completion.finish_reason,
            "capped_horizon": completion.finish_reason == "length",
            "terminal_token_id": rids[-1] if rids[-1] in terminal_ids else None,
            "artificial_eos_appended": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", default="v1")
    ap.add_argument("--split-names", nargs="+", default=["policy_train"])
    ap.add_argument("--model", default="/work/models/bases/Llama-3.1-8B-Instruct")
    ap.add_argument("--model-revision", default="0e9e39f249a16976918f6564b8830bc894c89659")
    ap.add_argument("--out-name", default="v1")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--prompts-per-chunk", type=int, default=25)
    ap.add_argument("--max-chunks", type=int,
                    help="Stop after this many chunks. The profile uses it; the full "
                         "run continues from the same chunk boundary with the same seeds.")
    ap.add_argument("--max-prompt-tokens", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    generation_config = json.loads((Path(args.model) / "generation_config.json").read_text())
    terminal = generation_config["eos_token_id"]
    terminal = terminal if isinstance(terminal, list) else [terminal]

    split_dir = ROOT / "splits" / args.splits
    rows = []
    truncated = []
    for split in args.split_names:
        for row in read_jsonl(split_dir / f"{split}.jsonl"):
            ids, dropped, text = prompt_tokens(tokenizer, row["instruction"], args.max_prompt_tokens)
            if dropped:
                truncated.append({"prompt_id": row["prompt_id"], "source": row["source"],
                                  "dropped_prompt_tokens": dropped, "kept_tokens": len(ids)})
            rows.append({"prompt_id": row["prompt_id"], "source": row["source"], "split": split,
                         "prompt": text, "original_instruction": row["instruction"],
                         "prompt_token_ids": ids})
    shard_rows = rows[args.shard::args.shards]

    plan = [(role, index) for role in ROLES for index in range(POOL)]
    seeds = {f"{row['prompt_id']}:{role}:{index}":
             int(digest(SEED_NAMESPACE + f"{row['prompt_id']}:{role}:{index}")[:16], 16) % (2**63 - 1)
             for row in rows for role, index in plan}
    if len(set(seeds.values())) != len(seeds):
        raise ValueError("Random stream seed collision")

    out = ROOT / "pools" / args.out_name / f"shard{args.shard}"
    out.mkdir(parents=True, exist_ok=True)
    settings = {"model": args.model, "model_revision": args.model_revision,
                "splits_dir": str(split_dir),
                "split_files_sha256": {s: file_hash(split_dir / f"{s}.jsonl") for s in args.split_names},
                "roles": list(ROLES), "pool_per_role": POOL,
                "reference_construction": "comparator_self_bank",
                "temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
                "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
                "max_tokens": args.max_tokens, "max_prompt_tokens": args.max_prompt_tokens,
                "max_model_len": args.max_model_len, "stop_token_ids": terminal,
                "generation_config": "vllm", "n": 1,
                "occurrence_mass": 1.0 / POOL, "duplicate_occurrences_retained": True,
                "seed_rule": f"SHA256('{SEED_NAMESPACE}' + candidate_id)[:16] mod (2**63-1)",
                "shard": args.shard, "shards": args.shards,
                "prompts_per_chunk": args.prompts_per_chunk,
                "chat_template_sha256": digest(tokenizer.chat_template or ""),
                "source_sha256": file_hash(__file__)}
    settings_path = out / "settings.json"
    if settings_path.exists():
        existing = json.loads(settings_path.read_text())
        drift = {k: (existing.get(k), settings[k]) for k in settings
                 if existing.get(k) != settings[k] and k != "source_sha256"}
        if drift:
            raise ValueError(f"Existing shard has different generation settings: {sorted(drift)}")
    else:
        write_json(settings_path, settings)
        write_json(out / "prompt_truncation.json",
                   {"n_truncated": len(truncated), "n_prompts": len(rows),
                    "max_prompt_tokens": args.max_prompt_tokens,
                    "rule": "chat-templated prompt truncated from the front (keep_end)",
                    "rows": truncated[:500]})

    llm = None
    total_events = total_tokens = 0
    chunk_reports = []
    start = time.monotonic()
    chunks = list(range(0, len(shard_rows), args.prompts_per_chunk))
    if args.max_chunks:
        chunks = chunks[:args.max_chunks]
    for chunk_index, lo in enumerate(chunks):
        destination = out / f"chunk{chunk_index:04d}.jsonl"
        metadata = out / f"chunk{chunk_index:04d}.manifest.json"
        prompts = shard_rows[lo:lo + args.prompts_per_chunk]
        if destination.exists():
            if not metadata.exists() or file_hash(destination) != json.loads(metadata.read_text())["sha256"]:
                raise ValueError(f"Partial or corrupt chunk {destination}; preserve and recover explicitly")
            continue
        if llm is None:
            llm = LLM(model=args.model, tokenizer=args.model, tensor_parallel_size=1,
                      dtype="bfloat16", max_model_len=args.max_model_len,
                      gpu_memory_utilization=args.gpu_memory_utilization,
                      max_num_seqs=256, max_num_batched_tokens=16384,
                      generation_config="vllm", seed=20260910, trust_remote_code=False)
        requests, params, mapping = [], [], []
        for prompt in prompts:
            for role, index in plan:
                candidate_id = f"{prompt['prompt_id']}:{role}:{index}"
                seed = seeds[candidate_id]
                requests.append({"prompt_token_ids": prompt["prompt_token_ids"]})
                params.append(SamplingParams(n=1, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0,
                                             repetition_penalty=1.0, presence_penalty=0.0,
                                             frequency_penalty=0.0, max_tokens=args.max_tokens,
                                             seed=seed, stop_token_ids=terminal, ignore_eos=False))
                mapping.append((prompt, role, index, seed))
        before = time.monotonic()
        generated = llm.generate(requests, params, use_tqdm=False)
        if len(generated) != len(requests) or any(len(r.outputs) != 1 for r in generated):
            raise ValueError("Generation lost requests or produced the wrong number of completions")
        events = [make_event(*item, result.outputs[0], tokenizer, terminal, args.max_model_len)
                  for item, result in zip(mapping, generated)]
        with destination.open("x") as sink:
            for event in events:
                sink.write(json.dumps(event, ensure_ascii=False) + "\n")
        elapsed = time.monotonic() - before
        n_tokens = sum(e["n_tokens"] for e in events)
        distinct = len({(e["prompt_id"], e["role"], e["response_sha256"]) for e in events})
        report = {"sha256": file_hash(destination), "n_prompts": len(prompts),
                  "n_events": len(events), "response_tokens": n_tokens,
                  "seconds": elapsed, "tokens_per_second": n_tokens / elapsed,
                  "finish_reasons": dict(Counter(e["finish_reason"] for e in events)),
                  "capped_fraction": sum(e["capped_horizon"] for e in events) / len(events),
                  "distinct_response_groups": distinct,
                  "duplicate_occurrences": len(events) - distinct,
                  "decode_text_mismatches": sum(not e["decode_text_equal"] for e in events),
                  "bytes": destination.stat().st_size}
        write_json(metadata, report)
        chunk_reports.append(report)
        total_events += len(events)
        total_tokens += n_tokens
        print(json.dumps({"chunk": chunk_index, "prompts": len(prompts), "events": len(events),
                          "response_tokens": n_tokens, "seconds": round(elapsed, 2),
                          "tok_per_s": round(n_tokens / elapsed, 1)}), flush=True)
    summary = {"shard": args.shard, "n_prompts_in_shard": len(shard_rows),
               "chunks_run_now": len(chunk_reports), "new_events": total_events,
               "new_response_tokens": total_tokens,
               "wall_seconds": time.monotonic() - start,
               "settings_sha256": file_hash(settings_path),
               "prompt_truncated": len(truncated)}
    write_json(out / f"complete_shard{args.shard}.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
