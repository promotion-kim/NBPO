#!/usr/bin/env python3
"""Generate benchmark responses for one policy, API-free, under one fixed decoding contract.

Every benchmark in the panel is generated the same way -- greedy, temperature 0,
top_p 1, one response per prompt, the model's own chat template -- so that a
difference between methods cannot come from a difference in decoding. MT-Bench is
excluded here because it is two turns and needs its own runner.

The per-benchmark prompt construction is the only thing that varies, and it
follows each benchmark's own convention:

``gsm8k``      the official question plus the fixed CoT instruction the protocol
               pins, so the parsed answer format is well defined
``harmbench``  contextual behaviours keep their official context; copyright and
               standard behaviours are sent as the behaviour string
everything else is the instruction as published.

Generation statistics are recorded next to the responses -- token counts, how
often the length cap was hit, empty outputs -- because a method that answers
nothing must not be able to look safe or correct by omission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GSM8K_SUFFIX = ('\n\nSolve the problem step by step. '
                'End with "The answer is <number>".')


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).open() if l.strip()]


def build(bench, data_dir):
    """Return [(uid, user_message, meta)] for one benchmark."""
    d = Path(data_dir)
    if bench == "alpaca_eval":
        return [(f"alpaca:{r['id']}", r["instruction"], {"dataset": r.get("dataset")})
                for r in load_jsonl(d / "alpaca_eval.jsonl")]
    if bench == "arena_hard":
        out = []
        for r in load_jsonl(d / "arena_hard.jsonl"):
            p = r["prompt"]
            if isinstance(p, list):
                p = p[0] if p else ""
            out.append((f"arena:{r['uid']}", p,
                        {"category": r.get("category"), "subcategory": r.get("subcategory")}))
        return out
    if bench == "ifeval":
        return [(f"ifeval:{r['key']}", r["prompt"],
                 {"instruction_id_list": r.get("instruction_id_list")})
                for r in load_jsonl(d / "ifeval.jsonl")]
    if bench == "gsm8k":
        return [(f"gsm8k:{r['id']}", r["question"] + GSM8K_SUFFIX,
                 {"gold": r["answer"]}) for r in load_jsonl(d / "gsm8k_test.jsonl")]
    if bench == "harmbench":
        out = []
        for r in load_jsonl(d / "harmbench_behaviors.jsonl"):
            ctx = (r.get("ContextString") or "").strip()
            beh = r["Behavior"].strip()
            msg = f"{ctx}\n\n---\n\n{beh}" if ctx else beh
            out.append((f"harmbench:{r['BehaviorID']}", msg,
                        {"functional_category": r.get("FunctionalCategory"),
                         "semantic_category": r.get("SemanticCategory"),
                         "has_context": bool(ctx)}))
        return out
    raise SystemExit(f"unknown benchmark {bench}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tokenizer", default=None,
                    help="checkpoints saved without a tokenizer borrow the base one")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--benchmarks", nargs="+",
                    default=["alpaca_eval", "arena_hard", "ifeval", "gsm8k", "harmbench"])
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok_src = args.tokenizer or args.model
    tok = AutoTokenizer.from_pretrained(tok_src)
    llm = LLM(model=args.model, tokenizer=tok_src, dtype="bfloat16",
              max_model_len=args.max_model_len, seed=args.seed,
              gpu_memory_utilization=args.gpu_memory_utilization,
              enforce_eager=False)
    # greedy; the protocol fixes one response per prompt and no reranking
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens,
                        seed=args.seed, n=1)

    out = Path(args.out_dir) / args.label
    out.mkdir(parents=True, exist_ok=True)
    summary = {"label": args.label, "model": args.model, "tokenizer": tok_src,
               "decoding": {"temperature": 0.0, "top_p": 1.0,
                            "max_new_tokens": args.max_new_tokens, "seed": args.seed,
                            "n": 1, "greedy": True},
               "max_model_len": args.max_model_len, "benchmarks": {}}

    for bench in args.benchmarks:
        items = build(bench, args.data_dir)
        prompts, over = [], 0
        for uid, msg, meta in items:
            text = tok.apply_chat_template([{"role": "user", "content": msg}],
                                           tokenize=False, add_generation_prompt=True)
            n_in = len(tok(text, add_special_tokens=False)["input_ids"])
            if n_in + args.max_new_tokens > args.max_model_len:
                over += 1
            prompts.append(text)
        outs = llm.generate(prompts, sp)
        rows, ntok, hit_cap, empty = [], [], 0, 0
        for (uid, msg, meta), o in zip(items, outs):
            g = o.outputs[0]
            txt = g.text
            n = len(g.token_ids)
            ntok.append(n)
            if g.finish_reason == "length":
                hit_cap += 1
            if not txt.strip():
                empty += 1
            rows.append({"uid": uid, "prompt": msg, "output": txt,
                         "n_output_tokens": n, "finish_reason": g.finish_reason,
                         "meta": meta})
        (out / f"{bench}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        ntok.sort()
        summary["benchmarks"][bench] = {
            "n": len(rows), "mean_output_tokens": sum(ntok) / max(1, len(ntok)),
            "median_output_tokens": ntok[len(ntok) // 2] if ntok else 0,
            "hit_max_tokens": hit_cap, "empty_outputs": empty,
            "prompts_over_context_budget": over}
        print(f"[{args.label}] {bench}: n={len(rows)} "
              f"mean_tok={summary['benchmarks'][bench]['mean_output_tokens']:.1f} "
              f"cap_hit={hit_cap} empty={empty} over_ctx={over}", flush=True)

    (out / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[{args.label}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
