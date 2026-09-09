"""Raw-policy occurrence sampling with immutable token events and resumable shards."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from scripts.experiments.nbpo_repair_20260909.common import (
    digest, file_hash, object_hash, read_jsonl, write_json, write_jsonl,
)


def make_event(prompt, role, index, seed, completion, tokenizer, terminal_ids):
    pid = prompt["prompt_id"]
    pids = prompt["prompt_token_ids"]
    rids = list(completion.token_ids)
    if not rids:
        raise ValueError(f"Empty sampled token event: {pid}/{role}/{index}")
    if completion.finish_reason == "length" and len(rids) != 1024:
        raise ValueError("Unexpected capped horizon")
    if completion.finish_reason == "stop" and rids[-1] not in terminal_ids:
        raise ValueError("Stop event does not retain its actually sampled terminal token")
    if completion.finish_reason not in ("stop", "length"):
        raise ValueError(f"Unexpected termination: {completion.finish_reason}")
    if len(pids) + len(rids) > 2048:
        raise ValueError("Sample would be truncated by training")
    text = tokenizer.decode(rids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    ids = pids + rids
    labels = [-100] * len(pids) + rids
    return {"prompt_id": pid, "prompt": prompt["prompt"], "split": prompt["split"],
        "role": role, "sample_index": index, "seed": seed,
        "candidate_id": f"{pid}:{role}:{index}", "response": text,
        "vllm_output_text": completion.text, "decode_text_equal": text == completion.text,
        "prompt_token_ids": pids, "response_token_ids": rids,
        "input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels,
        "token_event_sha256": object_hash({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}),
        "response_sha256": digest(text), "n_tokens": len(rids),
        "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason,
        "terminal_token_id": rids[-1] if rids[-1] in terminal_ids else None,
        "capped_horizon": completion.finish_reason == "length", "artificial_eos_appended": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--model", default="/work/models/bases/Llama-3.1-8B-Instruct")
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--prompts-per-chunk", type=int, default=16)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = json.loads((Path(args.model) / "generation_config.json").read_text())
    terminal = config["eos_token_id"]
    terminal = terminal if isinstance(terminal, list) else [terminal]
    all_rows = [row for s in args.splits for row in read_jsonl(args.root / "splits" / f"{s}.jsonl")]
    rows = all_rows[args.shard::args.shards]
    plan = [(role, idx) for role in ("learner", "comparator", "reference_learner") for idx in range(8)]
    out = args.root / "pools" / f"shard{args.shard}"
    out.mkdir(parents=True, exist_ok=True)
    settings = {"model": args.model, "inventory_sha256": file_hash(args.root / "provenance/inventory.json"),
        "split_manifest_sha256": file_hash(args.root / "splits/manifest.json"),
        "temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
        "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
        "max_tokens": 1024, "max_prompt_tokens": 1024, "max_model_len": 2048,
        "stop_token_ids": terminal, "generation_config": "vllm", "n": 1,
        "occurrence_mass": 0.125, "duplicate_occurrences_retained": True,
        "reference_construction": "independent reference-as-learner8 versus actual comparator8; additional8 explicitly costed",
        "independent_seed_rule": "SHA256('20260909-repair-pool:' + candidate_id) first 16 hex mod (2**63-1), globally checked",
        "shard": args.shard, "shards": args.shards, "splits": args.splits,
        "prompts_per_chunk": args.prompts_per_chunk,
        "source_sha256": file_hash(__file__)}
    mp = out / "settings.json"
    if mp.exists():
        if json.loads(mp.read_text()) != settings:
            raise ValueError("Existing shard has different generation settings")
    else:
        write_json(mp, settings)
    llm = None
    seeds = [int(digest("20260909-repair-pool:" + f"{p['prompt_id']}:{r}:{i}")[:16], 16) % (2**63 - 1)
             for p in all_rows for r, i in plan]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Random stream seed collision")
    start = time.monotonic()
    all_count, token_count = 0, 0
    for chunk_index, lo in enumerate(range(0, len(rows), args.prompts_per_chunk)):
        destination = out / f"chunk{chunk_index:04d}.jsonl"
        metadata = out / f"chunk{chunk_index:04d}.manifest.json"
        prompts = rows[lo:lo + args.prompts_per_chunk]
        if destination.exists():
            if not metadata.exists() or file_hash(destination) != json.loads(metadata.read_text())["sha256"]:
                raise ValueError(f"Partial/corrupt completed chunk {destination}; preserve and recover explicitly")
            continue
        if llm is None:
            llm = LLM(model=args.model, tokenizer=args.model, tensor_parallel_size=1,
                dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.85,
                max_num_seqs=256, max_num_batched_tokens=16384,
                generation_config="vllm", seed=20260909, trust_remote_code=False)
        requests, params, mapping = [], [], []
        for prompt in prompts:
            actual = tok.apply_chat_template([{"role": "user", "content": prompt["prompt"]}], tokenize=True, add_generation_prompt=True)
            if actual != prompt["prompt_token_ids"]:
                raise ValueError("Frozen prompt token event changed")
            for role, index in plan:
                candidate_id = f"{prompt['prompt_id']}:{role}:{index}"
                seed = int(digest("20260909-repair-pool:" + candidate_id)[:16], 16) % (2**63 - 1)
                requests.append({"prompt_token_ids": actual})
                params.append(SamplingParams(n=1, temperature=1.0, top_p=1.0, top_k=-1,
                    min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0,
                    frequency_penalty=0.0, max_tokens=1024, seed=seed,
                    stop_token_ids=terminal, ignore_eos=False))
                mapping.append((prompt, role, index, seed))
        before = time.monotonic()
        generated = llm.generate(requests, params, use_tqdm=False)
        if len(generated) != len(requests) or any(len(result.outputs) != 1 for result in generated):
            raise ValueError("Generation lost requests or produced wrong number of completions")
        events = [make_event(*item, result.outputs[0], tok, terminal) for item, result in zip(mapping, generated)]
        write_jsonl(destination, events)
        elapsed = time.monotonic() - before
        ntok = sum(r["n_tokens"] for r in events)
        write_json(metadata, {"sha256": file_hash(destination), "n_prompts": len(prompts),
            "n_events": len(events), "response_tokens": ntok, "seconds": elapsed,
            "tokens_per_second": ntok / elapsed, "finish_reasons": dict(Counter(r["finish_reason"] for r in events)),
            "decoding_text_mismatches": sum(not r["decode_text_equal"] for r in events)})
        all_count += len(events)
        token_count += ntok
        print(json.dumps({"shard": args.shard, "chunk": chunk_index, "events": len(events),
            "response_tokens": ntok, "seconds": round(elapsed, 2),
            "tokens_per_second": round(ntok / elapsed, 1)}), flush=True)
    write_json(out / "complete.json", {"n_prompts": len(rows), "expected_events": len(rows) * 24,
        "new_events_this_invocation": all_count, "new_tokens": token_count,
        "wall_seconds": time.monotonic() - start, "settings_sha256": file_hash(mp)})


if __name__ == "__main__":
    main()
