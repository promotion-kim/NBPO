"""Eight base candidate occurrences per panel prompt, at the training-pool decoding.

Adapted from the UF-4 audit generator with two deliberate changes. The count is
eight, so a prompt's response graph has the 28 unordered pairs and 56 distinct
triples the screening contract needs. The decoding is the TRAINING pool's
(T=1, top_p=1, 1024 tokens), not the UF audit's (T=0.8, top_p=0.9, 2048), so
these occurrences can later serve as the learner pool Y of the policy panel
without a second distribution appearing in the same campaign.

The base model and its revision are the campaign's existing ones; nothing here
changes the backbone. Duplicate and length-capped responses are kept and
flagged, and the judge reads the recorded text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

ROOT = Path("/work/sub_20260914")
SEED_NAMESPACE = "20260914-sub-candidates:"


def digest(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def file_hash(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", required=True, help="file under panel/, e.g. pilot10.jsonl")
    ap.add_argument("--model", default="/work/models/bases/Llama-3.1-8B-Instruct")
    ap.add_argument("--model-revision", default="0e9e39f249a16976918f6564b8830bc894c89659")
    ap.add_argument("--responses-per-prompt", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--tag", default=None, help="output directory name; defaults to the panel stem")
    ap.add_argument("--seed-tag", default="pool",
                    help="enters the per-candidate seed namespace, so two draws from the "
                         "same model on the same prompt are independent rather than "
                         "identical -- without it the base's fresh response would be a "
                         "byte copy of reference occurrence 0")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    panel_path = ROOT / "panel" / args.panel
    if not panel_path.exists():
        panel_path = ROOT / "splits" / args.panel
    rows = [json.loads(l) for l in panel_path.open() if l.strip()]
    tag = args.tag or panel_path.stem
    out = ROOT / "responses" / tag
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "responses.jsonl"
    if dest.exists():
        print(json.dumps({"skipped": "already generated", "path": str(dest)}), flush=True)
        return 0

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    gen_cfg = json.loads((Path(args.model) / "generation_config.json").read_text())
    terminal = gen_cfg["eos_token_id"]
    terminal = terminal if isinstance(terminal, list) else [terminal]

    settings = {
        "model": args.model, "model_revision": args.model_revision,
        "temperature": args.temperature, "top_p": args.top_p, "top_k": -1,
        "repetition_penalty": 1.0, "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "responses_per_prompt": args.responses_per_prompt,
        "decoding": ("temperature %g, top_p %g, %d tokens"
                     % (args.temperature, args.top_p, args.max_tokens)),
        "matches_uf4_training_pool_decoding": bool(args.temperature == 1.0
                                                   and args.top_p == 1.0
                                                   and args.max_tokens == 1024),
        "seed_tag": args.seed_tag,
        "seed_rule": ("SHA256('%s' + seed_tag:prompt_id:index)[:16] mod (2**63-1)"
                      % SEED_NAMESPACE),
        "panel_file": str(panel_path), "panel_sha256": file_hash(panel_path),
        "chat_template_sha256": digest(tok.chat_template or ""),
        "source_sha256": file_hash(__file__)}
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    llm = LLM(model=args.model, tokenizer=args.model, tensor_parallel_size=1,
              dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=128,
              generation_config="vllm", seed=20260914, trust_remote_code=False)

    requests, params, mapping = [], [], []
    for row in rows:
        ids = tok.apply_chat_template([{"role": "user", "content": row["instruction"]}],
                                      tokenize=True, add_generation_prompt=True)
        if len(ids) + args.max_tokens > args.max_model_len:
            ids = ids[-(args.max_model_len - args.max_tokens):]
        for i in range(args.responses_per_prompt):
            seed = int(digest(SEED_NAMESPACE + "%s:%s:%d"
                               % (args.seed_tag, row["prompt_id"], i))[:16], 16) % (2**63 - 1)
            requests.append({"prompt_token_ids": ids})
            params.append(SamplingParams(n=1, temperature=args.temperature,
                                         top_p=args.top_p, top_k=-1,
                                         repetition_penalty=1.0, max_tokens=args.max_tokens,
                                         seed=seed, stop_token_ids=terminal,
                                         ignore_eos=False))
            mapping.append((row, i, seed))

    started = time.monotonic()
    generated = llm.generate(requests, params, use_tqdm=False)
    elapsed = time.monotonic() - started

    events = []
    for (row, index, seed), result in zip(mapping, generated):
        o = result.outputs[0]
        text = tok.decode(list(o.token_ids), skip_special_tokens=True,
                          clean_up_tokenization_spaces=False)
        events.append({"prompt_id": row["prompt_id"], "source": row["source"],
                       "instruction": row["instruction"],
                       "response_id": "%s:%d" % (row["prompt_id"], index),
                       "response_index": index, "seed": seed, "response": text,
                       "n_tokens": len(o.token_ids), "finish_reason": o.finish_reason,
                       "capped": o.finish_reason == "length",
                       "response_sha256": digest(text)})
    with dest.open("x") as stream:
        for e in events:
            stream.write(json.dumps(e, ensure_ascii=False) + "\n")

    by_prompt = {}
    for e in events:
        by_prompt.setdefault(e["prompt_id"], []).append(e["response_sha256"])
    dupes = sum(len(v) - len(set(v)) for v in by_prompt.values())
    report = {"panel": args.panel, "tag": tag, "n_prompts": len(rows),
              "n_responses": len(events), "seconds": elapsed,
              "responses_per_second": len(events) / max(elapsed, 1e-9),
              "response_tokens": sum(e["n_tokens"] for e in events),
              "capped_responses": sum(e["capped"] for e in events),
              "duplicate_responses_within_prompt": dupes,
              "prompts_with_a_duplicate": sum(1 for v in by_prompt.values()
                                              if len(v) != len(set(v))),
              "finish_reasons": dict(Counter(e["finish_reason"] for e in events)),
              "median_tokens": sorted(e["n_tokens"] for e in events)[len(events) // 2],
              "responses_sha256": file_hash(dest),
              "settings_sha256": file_hash(out / "settings.json")}
    (out / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
