"""Decisive A/B: same judge, same process, cross-play inputs vs final-eval inputs.

The cross-play judge parses 3% of its verdicts where the final-eval judge parses
99.5%, with byte-identical template, regex and token budget, and statistically
identical prompt and response lengths. This runs both input families through one
LLM instance at two token budgets, so the cause and the fix are settled together
rather than argued about.
"""
import json, hashlib, re, sys
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
CRIT = "instruction_following"
VERDICT = re.compile(r"\[\[(A|B|TIE)\]\]", re.I)
TEMPLATE = open(ROOT / "code/judge_final_eval.py").read()
TEMPLATE = re.search(r'TEMPLATE = """(.*?)"""', TEMPLATE, re.S).group(1)
rubrics = json.loads((ROOT / "audit/v1/rubrics.json").read_text())["rubrics"]


def learner0(root, shards):
    out = {}
    for s in range(shards):
        for p in sorted((Path(root) / f"shard{s}").glob("chunk*.jsonl")):
            for line in open(p):
                r = json.loads(line)
                if r["role"] == "learner" and r["sample_index"] == 0:
                    out[r["prompt_id"]] = r
    return out


cp_a = learner0(ROOT / "pools/crossplay_base", 1)
cp_b = learner0(ROOT / "pools/crossplay_nbpo_mse_s42", 1)
fe_a = learner0(ROOT / "pools/final_base_v1", 4)
fe_b = learner0(ROOT / "pools/final_nbpo_mse_s42", 4)

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
judge = str(ROOT / "assets/Qwen3-14B")
tok = AutoTokenizer.from_pretrained(judge, local_files_only=True)
llm = LLM(model=judge, tokenizer=judge, tensor_parallel_size=1, dtype="bfloat16",
          max_model_len=8192, gpu_memory_utilization=0.90, max_num_seqs=64,
          generation_config="vllm", seed=20260912, trust_remote_code=False)

cases = []
for label, A, B in (("crossplay", cp_a, cp_b), ("finaleval", fe_a, fe_b)):
    pids = sorted(set(A) & set(B))[:25]
    for pid in pids:
        text = TEMPLATE.format(rubric=rubrics[CRIT], instruction=A[pid]["prompt"],
                               response_a=A[pid]["response"], response_b=B[pid]["response"])
        ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, tokenize=True)
        cases.append((label, pid, ids))

for budget in (256, 1024):
    reqs = [{"prompt_token_ids": ids} for _, _, ids in cases]
    pars = [SamplingParams(n=1, temperature=0.0, max_tokens=budget,
                           seed=int(hashlib.sha256((l + pid).encode()).hexdigest()[:16], 16)
                           % (2 ** 63 - 1))
            for l, pid, _ in cases]
    gen = llm.generate(reqs, pars, use_tqdm=False)
    tally = {}
    for (label, _, ids), g in zip(cases, gen):
        t = g.outputs[0].text
        ok = len(VERDICT.findall(t)) == 1
        d = tally.setdefault(label, {"ok": 0, "n": 0, "out_tokens": [], "in_tokens": []})
        d["ok"] += ok; d["n"] += 1
        d["out_tokens"].append(len(g.outputs[0].token_ids)); d["in_tokens"].append(len(ids))
    for label, d in sorted(tally.items()):
        ot = sorted(d["out_tokens"]); it = sorted(d["in_tokens"])
        print(json.dumps({"max_tokens": budget, "family": label,
                          "parse_rate": round(d["ok"] / d["n"], 3),
                          "median_output_tokens": ot[len(ot) // 2],
                          "max_output_tokens": ot[-1],
                          "median_input_tokens": it[len(it) // 2]}), flush=True)
