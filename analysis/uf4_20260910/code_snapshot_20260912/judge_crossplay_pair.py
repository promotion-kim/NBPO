"""Independently judge one ordered-pair-free cross-play cell: policy A against policy B.

One job per unordered pair, rather than one job for the whole bank. Two reasons.
Adding the development-selected competitor then costs only the three pairs that
involve it instead of re-judging everything, and each job is about 4,000
judgments -- roughly six minutes on one card -- which is the shape that fits the
window where a single-GPU judge runs and the other cards are reserved for the
next 4-GPU training.

Everything about the evaluator is the final-eval contract unchanged: the same
local Qwen3-14B at the same revision, the same rubric file, the same template
and verdict grammar, thinking disabled, temperature 0, both presentation orders,
ties 0.5, and a judgment that does not parse stays missing rather than being
counted as a loss.

Thinking is disabled explicitly, and that line is the whole reason this file was
revised. Qwen3 chat templates leave the assistant turn open unless told not to,
the judge then spends its output budget on a reasoning block, and the verdict
never arrives: the first cross-play runs parsed 2.6-6.9% of judgments at the
declared 256-token limit while the byte-identical final-eval judge -- which does
pass the flag -- parses 99.4%. Raising the limit to 2048 tokens recovered 88-91%
but left a protocol that the declared evaluator contract does not describe, so
those runs are set aside rather than reported.
The only differences are that both sides are trained policies and that the
prompts are the held-out cross-play 500, which no training pool, dev selection
or final evaluation has touched.

Antisymmetry convention, recorded here because the manuscript requires it before
the matrix is filled: only the unordered pair is judged, and the reverse cell is
defined as W_k(B,A) = 1 - W_k(A,B). Order averaging already removes presentation
bias within the pair, so judging the reverse separately would add sampling noise
without adding information. Diagonal cells are never produced by this script.
"""
from __future__ import annotations

import argparse, hashlib, json, re, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/work/uf4_20260910")
CRITERIA = ("instruction_following", "truthfulness", "honesty", "helpfulness")
SEED_NAMESPACE = "20260912-uf4-crossplay:"
VERDICT = re.compile(r"\[\[(A|B|TIE)\]\]", re.I)

TEMPLATE = """You are comparing two responses to the same user instruction.

{rubric}

Ignore which response is longer unless the instruction asked for a particular length. Ignore the order in which they are shown.

### INSTRUCTION
{instruction}

### RESPONSE A
{response_a}

### RESPONSE B
{response_b}

Answer with one line of reasoning, then your verdict on its own line in exactly this form: [[A]] if A is better, [[B]] if B is better, or [[TIE]] if they are equally good."""


def digest(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def file_hash(p):
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_pool_learner0(pool_root, shards=1):
    """prompt_id -> the fixed learner:0 response, the same convention the final eval uses."""
    out = {}
    for shard in range(shards):
        d = Path(pool_root) / f"shard{shard}"
        for path in sorted(d.glob("chunk*.jsonl")):
            with path.open() as stream:
                for line in stream:
                    e = json.loads(line)
                    if e["role"] == "learner" and e["sample_index"] == 0:
                        out[e["prompt_id"]] = e
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy-a", required=True)
    ap.add_argument("--policy-b", required=True)
    ap.add_argument("--pool-a", required=True)
    ap.add_argument("--pool-b", required=True)
    ap.add_argument("--judge", default=str(ROOT / "assets/Qwen3-14B"))
    ap.add_argument("--judge-revision", default="40c069824f4251a91eefaf281ebe4c544efd3e18")
    ap.add_argument("--rubrics", default=str(ROOT / "audit/v1/rubrics.json"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    if args.policy_a == args.policy_b:
        raise SystemExit("diagonal cells are not produced by this script")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.judge, local_files_only=True)
    rubrics = json.loads(Path(args.rubrics).read_text())["rubrics"]

    a_pool = load_pool_learner0(args.pool_a)
    b_pool = load_pool_learner0(args.pool_b)
    shared = sorted(set(a_pool) & set(b_pool))
    if not shared:
        raise SystemExit("the two pools share no prompt")

    tag = "%s__vs__%s" % (args.policy_a, args.policy_b)
    out = ROOT / "evaluation/crossplay/pairs" / tag
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "judgments.jsonl"
    if dest.exists():
        print(json.dumps({"skipped": "already judged", "path": str(dest)}), flush=True)
        return

    llm = LLM(model=args.judge, tokenizer=args.judge, tensor_parallel_size=1, dtype="bfloat16",
              max_model_len=args.max_model_len, gpu_memory_utilization=0.90, max_num_seqs=64,
              generation_config="vllm", seed=20260912, trust_remote_code=False)

    requests, params, meta = [], [], []
    truncated = 0
    for pid in shared:
        instruction = a_pool[pid]["prompt"]
        for criterion in CRITERIA:
            for order in (0, 1):
                first, second = ((a_pool[pid], b_pool[pid]) if order == 0
                                 else (b_pool[pid], a_pool[pid]))
                text = TEMPLATE.format(rubric=rubrics[criterion], instruction=instruction,
                                       response_a=first["response"], response_b=second["response"])
                ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                              add_generation_prompt=True, tokenize=True,
                                              enable_thinking=False)
                budget = args.max_model_len - args.max_tokens
                if len(ids) > budget:
                    ids = ids[:budget]
                    truncated += 1
                seed = int(digest(SEED_NAMESPACE +
                                  f"{tag}:{pid}:{criterion}:{order}")[:16], 16) % (2 ** 63 - 1)
                requests.append({"prompt_token_ids": ids})
                params.append(SamplingParams(n=1, temperature=args.temperature,
                                             max_tokens=args.max_tokens, seed=seed))
                meta.append({"prompt_id": pid, "criterion": criterion, "order": order,
                             "a_first": order == 0, "policy_a": args.policy_a,
                             "policy_b": args.policy_b, "seed": seed})

    print(json.dumps({"phase": "start", "pair": tag, "prompts": len(shared),
                      "judgments": len(requests), "truncated": truncated}), flush=True)
    started = time.monotonic()
    generated = llm.generate(requests, params, use_tqdm=False)
    elapsed = time.monotonic() - started

    values, counts = defaultdict(dict), Counter()
    with dest.open("x") as stream:
        for m, result in zip(meta, generated):
            text = result.outputs[0].text
            found = VERDICT.findall(text)
            if len(found) != 1:
                v, status = None, ("no_verdict" if not found else "multiple_verdicts")
            else:
                verdict = found[0].upper()
                if verdict == "TIE":
                    v = 0.5
                elif m["a_first"]:
                    v = 1.0 if verdict == "A" else 0.0
                else:
                    v = 0.0 if verdict == "A" else 1.0
                status = "ok"
            counts[status] += 1
            if status == "ok":
                values[(m["criterion"], m["prompt_id"])][m["order"]] = v
            stream.write(json.dumps({**m, "value_for_a": v, "status": status,
                                     "raw": text[-300:]}, ensure_ascii=False) + "\n")

    rng = np.random.default_rng(20260912)
    summary, complete_ids = {}, {}
    for criterion in CRITERIA:
        per_prompt, pids = [], []
        for pid in shared:
            got = values.get((criterion, pid), {})
            if len(got) == 2:                      # both presentation orders required
                per_prompt.append(0.5 * (got[0] + got[1]))
                pids.append(pid)
        arr = np.array(per_prompt, dtype=float)
        boot = (np.array([rng.choice(arr, size=arr.size, replace=True).mean()
                          for _ in range(args.bootstrap)]) if arr.size else np.array([np.nan]))
        summary[criterion] = {
            "win_rate_a_over_b": float(arr.mean()) if arr.size else None,
            "win_rate_b_over_a": float(1.0 - arr.mean()) if arr.size else None,
            "ci95_a_over_b": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "n_prompts_complete": int(arr.size), "n_prompts_planned": len(shared),
            "tie_fraction": float(np.mean(arr == 0.5)) if arr.size else None}
        complete_ids[criterion] = pids

    report = {"pair": tag, "policy_a": args.policy_a, "policy_b": args.policy_b,
              "pool_a": args.pool_a, "pool_b": args.pool_b,
              "judge": args.judge, "judge_revision": args.judge_revision,
              "prompts": "held-out cross-play 500, disjoint from every assigned split",
              "decoding": {"temperature": args.temperature, "max_tokens": args.max_tokens,
                           "enable_thinking": False},
              "estimand": "order-averaged win rate of A over B, ties 0.5",
              "antisymmetry_convention": ("only the unordered pair is judged; the reverse cell "
                                          "is defined as 1 - W_k(A,B). Order averaging already "
                                          "removes presentation bias within the pair."),
              "diagonal": "not produced by this script",
              "n_judgments": len(requests), "seconds": elapsed,
              "status_counts": dict(counts), "truncated_judge_inputs": truncated,
              "criteria": summary, "judgments_sha256": file_hash(dest)}
    (out / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "complete_prompt_ids.json").write_text(json.dumps(complete_ids) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "criteria"}), flush=True)
    for c, v in summary.items():
        print(json.dumps({"criterion": c, "W_a_over_b": v["win_rate_a_over_b"],
                          "ci95": v["ci95_a_over_b"], "n": v["n_prompts_complete"]}), flush=True)


if __name__ == "__main__":
    main()
