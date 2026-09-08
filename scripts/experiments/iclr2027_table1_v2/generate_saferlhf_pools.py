#!/usr/bin/env python3
"""Sample the response pools for the SafeRLHF pool-geometry pilot.

All responses come from **one frozen reference policy** at identical decoding
settings; only the sampling seed differs. Learner and comparator draws use
disjoint seed blocks, so the two sides are exchangeable by construction and any
learner/comparator asymmetry in the pilot is a property of the estimator rather
than of the sampler.

The 4+4 geometry is the **first four of each side of the 8+8 draw**, not an
independent sample. That makes the comparison a strict nesting: whatever differs
between the two is the effect of pool size and not of a different random draw.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

LEARNER_SEED_BASE = 1000
COMPARATOR_SEED_BASE = 2000


def sha(t):
    return hashlib.sha256(str(t).encode("utf-8")).hexdigest()


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def select_prompts(rows, n, salt="pool-pilot-v1"):
    """Deterministic, outcome-blind: the lowest salted hashes among unique prompts."""
    seen, uniq = set(), []
    for r in rows:
        if r["prompt_sha256"] in seen:
            continue
        seen.add(r["prompt_sha256"])
        uniq.append({"prompt_id": r["prompt_sha256"], "prompt": r["prompt"]})
    uniq.sort(key=lambda p: sha(f"{salt}|{p['prompt_id']}"))
    return uniq[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--split", default="validation",
                    choices=("train", "validation", "test"),
                    help="which prompt-disjoint split to draw prompts from")
    ap.add_argument("--salt", default="pool-pilot-v1",
                    help="selection salt; change it to draw a DIFFERENT prompt set "
                         "from the same split without touching the split itself")
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--n-learner", type=int, default=8)
    ap.add_argument("--n-comparator", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--tensor-parallel-size", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = ap.parse_args()

    prompts = select_prompts(
        read_jsonl(args.splits_dir / f"saferlhf_{args.split}.jsonl"),
        args.n_prompts, salt=args.salt)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=args.max_model_len, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_memory_utilization,
              trust_remote_code=False)
    tok = llm.get_tokenizer()

    def chat(p):
        return tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)

    rendered = [chat(p["prompt"]) for p in prompts]
    rows, t0 = [], time.time()
    plan = ([("learner", i, LEARNER_SEED_BASE + i) for i in range(args.n_learner)]
            + [("comparator", i, COMPARATOR_SEED_BASE + i)
               for i in range(args.n_comparator)])
    for role, idx, seed in plan:
        sp = SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p,
                            max_tokens=args.max_tokens, seed=seed)
        out = llm.generate(rendered, sp, use_tqdm=False)
        for p, g in zip(prompts, out):
            rows.append({"prompt_id": p["prompt_id"], "prompt": p["prompt"],
                         "role": role, "sample_index": idx, "seed": seed,
                         "response": g.outputs[0].text.strip(),
                         "n_tokens": len(g.outputs[0].token_ids),
                         "finish_reason": getattr(g.outputs[0], "finish_reason", None)})
        print(f"  {role} sample {idx} (seed {seed}) done "
              f"[{time.time()-t0:.0f}s]", flush=True)

    (args.out_dir / "pool_responses.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    (args.out_dir / "pool_manifest.json").write_text(json.dumps({
        "model_path": args.model_path,
        "n_prompts": len(prompts),
        "n_learner": args.n_learner, "n_comparator": args.n_comparator,
        "learner_seeds": [LEARNER_SEED_BASE + i for i in range(args.n_learner)],
        "comparator_seeds": [COMPARATOR_SEED_BASE + i for i in range(args.n_comparator)],
        "seed_blocks_disjoint": True,
        "decoding": {"temperature": args.temperature, "top_p": args.top_p,
                     "max_tokens": args.max_tokens},
        "identical_decoding_across_roles": True,
        "nesting": ("the 4+4 geometry is the first four of each side of this draw, "
                    "so the pool-size comparison is a strict nesting"),
        "split": args.split, "selection_salt": args.salt,
        "prompt_selection": f"lowest sha256('{args.salt}|'+prompt_id) within the "
                            f"{args.split} split; outcome-blind",
        "prompt_ids": [p["prompt_id"] for p in prompts],
        "wall_clock_seconds": round(time.time() - t0, 1),
        "n_rows": len(rows),
    }, indent=2))
    print(f"\nwrote {len(rows)} responses to {args.out_dir}/pool_responses.jsonl "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
