"""Independent four-objective evaluation of one policy against the frozen base response.

Every method, seed and objective is compared with the SAME cached base response
per prompt -- learner index 0 of the final_base_v1 pool -- so the reference is
identical across arms and the comparison is not contaminated by re-sampling it.

Both presentation orders are judged and averaged, ties count 0.5, and the win
rate reported is the order-averaged estimand. A judgment that does not parse is
missing and keeps its place in the denominator.

The judge is the local Qwen3-14B, a different model family from the ModernBERT
teacher that produced the training targets. It is the evaluator here and is not
used to build any training tensor.
"""
from __future__ import annotations

import argparse, hashlib, json, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/work/uf4_20260910")
CRITERIA = ("instruction_following", "truthfulness", "honesty", "helpfulness")
SEED_NAMESPACE = "20260911-uf4-finaleval:"
import re
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


def digest(t): return hashlib.sha256(t.encode("utf-8")).hexdigest()


def file_hash(p):
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_pool_learner0(pool_root, shards=4):
    """prompt_id -> the fixed learner:0 response of that pool."""
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
    ap.add_argument("--arm", required=True)
    ap.add_argument("--policy-pool", required=True)
    ap.add_argument("--base-pool", default=str(ROOT / "pools/final_base_v1"))
    ap.add_argument("--judge", default=str(ROOT / "assets/Qwen3-14B"))
    ap.add_argument("--judge-revision", default="40c069824f4251a91eefaf281ebe4c544efd3e18")
    ap.add_argument("--rubrics", default=str(ROOT / "audit/v1/rubrics.json"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.judge, local_files_only=True)
    rubrics = json.loads(Path(args.rubrics).read_text())["rubrics"]

    policy = load_pool_learner0(args.policy_pool)
    base = load_pool_learner0(args.base_pool)
    shared = sorted(set(policy) & set(base))
    if len(shared) != len(base):
        print(json.dumps({"warning": "policy pool does not cover every base prompt",
                          "base": len(base), "policy": len(policy), "shared": len(shared)}),
              flush=True)

    out = ROOT / "evaluation/final_eval" / args.arm
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "judgments.jsonl"
    if dest.exists():
        print(json.dumps({"skipped": "already judged", "path": str(dest)}), flush=True)
        return

    llm = LLM(model=args.judge, tokenizer=args.judge, tensor_parallel_size=1, dtype="bfloat16",
              max_model_len=args.max_model_len, gpu_memory_utilization=0.90, max_num_seqs=64,
              generation_config="vllm", seed=20260911, trust_remote_code=False)

    requests, params, meta = [], [], []
    truncated = 0
    for pid in shared:
        instruction = base[pid]["prompt"]
        for criterion in CRITERIA:
            for order in (0, 1):
                a, b = ((policy[pid], base[pid]) if order == 0 else (base[pid], policy[pid]))
                text = TEMPLATE.format(rubric=rubrics[criterion], instruction=instruction,
                                       response_a=a["response"], response_b=b["response"])
                ids = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                              add_generation_prompt=True, enable_thinking=False)
                budget = args.max_model_len - args.max_tokens
                if len(ids) > budget:
                    ids = ids[:budget]
                    truncated += 1
                seed = int(digest(SEED_NAMESPACE + f"{args.arm}:{pid}:{criterion}:{order}")[:16], 16) % (2**63 - 1)
                requests.append({"prompt_token_ids": ids})
                params.append(SamplingParams(n=1, temperature=args.temperature,
                                             max_tokens=args.max_tokens, seed=seed))
                meta.append({"prompt_id": pid, "criterion": criterion, "order": order,
                             "policy_first": order == 0, "seed": seed})

    print(json.dumps({"phase": "start", "arm": args.arm, "prompts": len(shared),
                      "judgments": len(requests), "truncated": truncated}), flush=True)
    started = time.monotonic()
    generated = llm.generate(requests, params, use_tqdm=False)
    elapsed = time.monotonic() - started

    values = defaultdict(dict)          # (criterion, prompt) -> {order: value for policy}
    counts = Counter()
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
                elif m["policy_first"]:
                    v = 1.0 if verdict == "A" else 0.0
                else:
                    v = 0.0 if verdict == "A" else 1.0
                status = "ok"
            counts[status] += 1
            if status == "ok":
                values[(m["criterion"], m["prompt_id"])][m["order"]] = v
            stream.write(json.dumps({**m, "value_for_policy": v, "status": status,
                                     "raw": text[-300:]}, ensure_ascii=False) + "\n")

    summary = {}
    rng = np.random.default_rng(20260911)
    for criterion in CRITERIA:
        per_prompt = []
        for pid in shared:
            got = values.get((criterion, pid), {})
            if len(got) == 2:                      # both orders required
                per_prompt.append(0.5 * (got[0] + got[1]))
        arr = np.array(per_prompt, dtype=float)
        boot = np.array([rng.choice(arr, size=arr.size, replace=True).mean()
                         for _ in range(2000)]) if arr.size else np.array([np.nan])
        summary[criterion] = {
            "win_rate_vs_base": float(arr.mean()) if arr.size else None,
            "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "n_prompts_complete": int(arr.size), "n_prompts_planned": len(shared),
            "tie_fraction": float(np.mean(arr == 0.5)) if arr.size else None,
            "bootstrap": "2000 whole-prompt replicates; prompt variability only"}
    report = {"arm": args.arm, "judge": args.judge, "judge_revision": args.judge_revision,
              "reference": "learner:0 of final_base_v1, identical for every arm",
              "decoding": {"temperature": args.temperature, "max_tokens": args.max_tokens},
              "estimand": "order-averaged win rate against the fixed base response, ties 0.5",
              "n_judgments": len(requests), "seconds": elapsed,
              "status_counts": dict(counts), "truncated_judge_inputs": truncated,
              "criteria": summary,
              "worst_criterion_win_rate": min(
                  (v["win_rate_vs_base"] for v in summary.values() if v["win_rate_vs_base"] is not None),
                  default=None),
              "judgments_sha256": file_hash(dest)}
    (out / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "criteria"}), flush=True)
    for c, v in summary.items():
        print(json.dumps({"criterion": c, **{k: v[k] for k in ("win_rate_vs_base", "ci95", "n_prompts_complete")}}), flush=True)


if __name__ == "__main__":
    main()
