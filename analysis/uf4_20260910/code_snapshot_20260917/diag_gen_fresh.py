"""One fresh stochastic response per panel prompt, for one frozen checkpoint.

Decoding matches the campaign pool exactly -- temperature 1.0, top_p 1.0,
top_k -1, min_p 0.0, 1024 new tokens, the pool's terminal-token convention --
so a fresh row differs from a cached candidate only in the draw. The generation
seed namespace is separate from the policy seed and from the pool's namespace,
so these are new draws and not a re-read of training candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
SEED_NAMESPACE = "20260914-uf4-freshdiag:"
REFUSAL = ("i can't", "i cannot", "i won't", "i'm sorry", "i am sorry", "as an ai")


def digest(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--panel", default=str(ROOT / "analysis/diag_20260914/panel_dev200.json"))
    ap.add_argument("--out", default=str(ROOT / "analysis/diag_20260914/fresh"))
    ap.add_argument("--pool", default="dev_v1")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-prompt-tokens", type=int, default=1024)
    args = ap.parse_args()

    panel = json.loads(Path(args.panel).read_text())
    ids = panel["panel_prompt_ids"]

    # instructions come from the same cached pool the panel was frozen on
    import glob
    instructions = {}
    for path in sorted(glob.glob(str(ROOT / "pools" / args.pool / "shard*/chunk*.jsonl"))):
        with open(path) as stream:
            for line in stream:
                r = json.loads(line)
                if r["prompt_id"] in set(ids) and r["prompt_id"] not in instructions:
                    instructions[r["prompt_id"]] = r["prompt"]
    missing = [p for p in ids if p not in instructions]
    if missing:
        raise SystemExit("no instruction text for %d panel prompts" % len(missing))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / ("%s.jsonl" % args.arm)
    done = out_dir / ("complete_%s.json" % args.arm)
    if done.exists():
        print(json.dumps({"skipped": "already generated", "path": str(done)}))
        return 0

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    terminal = [t for t in {tok.eos_token_id,
                            tok.convert_tokens_to_ids("<|eot_id|>")} if isinstance(t, int) and t >= 0]
    llm = LLM(model=args.model, tokenizer=args.model, tensor_parallel_size=1, dtype="bfloat16",
              max_model_len=args.max_model_len, gpu_memory_utilization=0.85,
              generation_config="vllm", seed=20260914, trust_remote_code=False)

    requests, params, meta = [], [], []
    for pid in ids:
        text = instructions[pid]
        tids = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                       add_generation_prompt=True)
        truncated = len(tids) > args.max_prompt_tokens
        if truncated:
            tids = tids[:args.max_prompt_tokens]
        seed = int(digest(SEED_NAMESPACE + "%s:%s" % (args.arm, pid))[:16], 16) % (2**63 - 1)
        requests.append({"prompt_token_ids": tids})
        params.append(SamplingParams(n=1, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0,
                                     max_tokens=args.max_tokens, seed=seed,
                                     stop_token_ids=terminal, ignore_eos=False))
        meta.append({"prompt_id": pid, "seed": seed, "prompt_truncated": truncated})

    started = time.monotonic()
    generated = llm.generate(requests, params, use_tqdm=False)
    elapsed = time.monotonic() - started

    lengths, capped, refusals = [], 0, 0
    tmp = dest.with_suffix(".jsonl.tmp")
    with tmp.open("w") as stream:
        for m, result in zip(meta, generated):
            out = result.outputs[0]
            n = len(out.token_ids)
            lengths.append(n)
            if out.finish_reason == "length":
                capped += 1
            low = out.text[:400].lower()
            if any(k in low for k in REFUSAL):
                refusals += 1
            stream.write(json.dumps({**m, "arm": args.arm, "response": out.text,
                                     "response_sha256": digest(out.text),
                                     "n_tokens": n, "finish_reason": out.finish_reason},
                                    ensure_ascii=False) + "\n")
    tmp.replace(dest)
    lengths.sort()
    summary = {"arm": args.arm, "model": args.model, "prompts": len(ids),
               "decoding": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
                            "max_tokens": args.max_tokens, "terminal_ids": terminal},
               "seed_namespace": SEED_NAMESPACE,
               "median_response_tokens": lengths[len(lengths) // 2],
               "mean_response_tokens": round(sum(lengths) / len(lengths), 1),
               "max_length_hits": capped,
               "keyword_refusal_count_diagnostic_only": refusals,
               "seconds": round(elapsed, 1),
               "responses_sha256": digest(dest.read_text())}
    done.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
