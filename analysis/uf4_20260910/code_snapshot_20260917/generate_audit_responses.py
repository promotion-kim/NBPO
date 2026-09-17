"""Four reference responses per audit prompt, at Appendix C's own decoding.

T=0.8, top_p=0.9, max_new_tokens=2048 from the unchanged base. This is NOT the
training pool's distribution (T=1, top_p=1, 1024 tokens), and the two are kept
in separate directories with separate settings files so neither can be
substituted for the other.

Duplicate and length-capped responses are kept and flagged; the judge reads the
full recorded text.
"""
from __future__ import annotations

import argparse, hashlib, json, time
from collections import Counter
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
SEED_NAMESPACE = "20260911-uf4-audit:"


def digest(t): return hashlib.sha256(t.encode("utf-8")).hexdigest()


def file_hash(p):
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-name", default="v1")
    ap.add_argument("--split-file", required=True, choices=("audit_100", "pilot_20"))
    ap.add_argument("--model", default="/work/models/bases/Llama-3.1-8B-Instruct")
    ap.add_argument("--model-revision", default="0e9e39f249a16976918f6564b8830bc894c89659")
    ap.add_argument("--responses-per-prompt", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    gen_cfg = json.loads((Path(args.model) / "generation_config.json").read_text())
    terminal = gen_cfg["eos_token_id"]
    terminal = terminal if isinstance(terminal, list) else [terminal]

    base = ROOT / "audit" / args.audit_name
    rows = [json.loads(l) for l in (base / f"{args.split_file}.jsonl").open() if l.strip()]
    out = base / "responses" / args.split_file
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "responses.jsonl"
    if dest.exists():
        print(json.dumps({"skipped": "already generated", "path": str(dest)}), flush=True)
        return

    settings = {"model": args.model, "model_revision": args.model_revision,
                "temperature": args.temperature, "top_p": args.top_p, "top_k": -1,
                "repetition_penalty": 1.0, "max_tokens": args.max_tokens,
                "max_model_len": args.max_model_len,
                "responses_per_prompt": args.responses_per_prompt,
                "distinct_from_training_pool": ("training pool is T=1, top_p=1, 1024 tokens; "
                                                "this audit distribution is not interchangeable with it"),
                "seed_rule": f"SHA256('{SEED_NAMESPACE}' + prompt_id:index)[:16] mod (2**63-1)",
                "prompt_file_sha256": file_hash(base / f"{args.split_file}.jsonl"),
                "chat_template_sha256": digest(tok.chat_template or ""),
                "source_sha256": file_hash(__file__)}
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    llm = LLM(model=args.model, tokenizer=args.model, tensor_parallel_size=1, dtype="bfloat16",
              max_model_len=args.max_model_len, gpu_memory_utilization=0.85, max_num_seqs=128,
              generation_config="vllm", seed=20260911, trust_remote_code=False)
    requests, params, mapping = [], [], []
    for row in rows:
        ids = tok.apply_chat_template([{"role": "user", "content": row["instruction"]}],
                                      tokenize=True, add_generation_prompt=True)
        if len(ids) + args.max_tokens > args.max_model_len:
            ids = ids[-(args.max_model_len - args.max_tokens):]
        for i in range(args.responses_per_prompt):
            cid = f"{row['prompt_id']}:{i}"
            seed = int(digest(SEED_NAMESPACE + cid)[:16], 16) % (2**63 - 1)
            requests.append({"prompt_token_ids": ids})
            params.append(SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p,
                                         top_k=-1, repetition_penalty=1.0,
                                         max_tokens=args.max_tokens, seed=seed,
                                         stop_token_ids=terminal, ignore_eos=False))
            mapping.append((row, i, seed, ids))

    started = time.monotonic()
    generated = llm.generate(requests, params, use_tqdm=False)
    elapsed = time.monotonic() - started
    events = []
    for (row, index, seed, ids), result in zip(mapping, generated):
        o = result.outputs[0]
        text = tok.decode(list(o.token_ids), skip_special_tokens=True,
                          clean_up_tokenization_spaces=False)
        events.append({"prompt_id": row["prompt_id"], "source": row["source"],
                       "instruction": row["instruction"],
                       "response_id": f"{row['prompt_id']}:{index}", "response_index": index,
                       "seed": seed, "response": text, "n_tokens": len(o.token_ids),
                       "finish_reason": o.finish_reason,
                       "capped": o.finish_reason == "length",
                       "response_sha256": digest(text)})
    with dest.open("x") as stream:
        for e in events:
            stream.write(json.dumps(e, ensure_ascii=False) + "\n")

    per_prompt_dupes = 0
    by_prompt = {}
    for e in events:
        by_prompt.setdefault(e["prompt_id"], []).append(e["response_sha256"])
    for pid, hashes in by_prompt.items():
        per_prompt_dupes += len(hashes) - len(set(hashes))
    report = {"split_file": args.split_file, "n_prompts": len(rows), "n_responses": len(events),
              "seconds": elapsed, "response_tokens": sum(e["n_tokens"] for e in events),
              "capped_responses": sum(e["capped"] for e in events),
              "duplicate_responses_within_prompt": per_prompt_dupes,
              "finish_reasons": dict(Counter(e["finish_reason"] for e in events)),
              "median_tokens": sorted(e["n_tokens"] for e in events)[len(events)//2],
              "responses_sha256": file_hash(dest), "settings_sha256": file_hash(out / "settings.json")}
    (out / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
