# Generated from judge_eval_batch.py by make_ahv2_judge.py -- do not edit by
# hand. source sha256 72dadb73533bd1d30bb7fd3c53e2c9b52adc0ab262cafc95e6564b8f0a063766
# change: load_baseline gains an arenahard_v2 branch (the released gpt-4.1
#         answers, joined by uid exactly as v0.1 is). Template, decoding,
#         order handling, tie convention, bootstrap and length recording are
#         unchanged, so a v2.0 number differs from a v0.1 number in the
#         benchmark and not in the measurement.
# Generated from judge_eval_pairwise.py by make_batch_judge.py -- do not edit.
# Same judging, one model load for every (arm, benchmark) pair.
"""Arena-Hard / AlpacaEval pairwise judging against the official static baselines.

No paid API is used. The questions and the baseline answers are the projects'
own released files -- Arena-Hard v0.1 question.jsonl with the gpt-4-0314
answers, AlpacaEval 2.0's 805 instructions with the gpt4_1106_preview outputs --
and the judge is this campaign's frozen local Qwen3-14B under the final-eval
contract: temperature 0, thinking disabled, both presentation orders, one draw
per order, ties count 0.5, an unparsed verdict stays missing and is never
retried.

Because the judge is local rather than GPT-4, these numbers are comparable
ACROSS THE ARMS OF THIS CAMPAIGN and are not the official leaderboard metric.
That is stated in the artifact as well as here.

The reported statistic is the order-averaged win rate against the baseline with
a 95% prompt bootstrap, and the per-arm response lengths are recorded next to
it because pairwise judges tend to prefer longer answers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SUB = Path("/work/sub_20260914")
UF = Path("/work/uf4_20260910")
EVALSETS = SUB / "prosper/evalsets"
VERDICT = re.compile(r"\[\[(A|B|TIE)\]\]", re.IGNORECASE)
TEMPLATE = """You are comparing two AI assistant responses to the same user question.

Judge overall response quality: correctness, helpfulness, depth, and how well the response follows the user's request. Do not reward length for its own sake.

User question:
{instruction}

Response A:
{response_a}

Response B:
{response_b}

Answer with one line of reasoning, then your verdict on its own line in exactly this form: [[A]] if A is better, [[B]] if B is better, or [[TIE]] if they are equally good."""


def digest(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def file_hash(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).replace("​", "")
    return re.sub(r"\s+", " ", t).strip().lower()


BASELINES = {
    "arenahard": ("ah_baseline_gpt4_0314.jsonl", "gpt-4-0314"),
    "arenahard_v2": ("ahv2_baseline_gpt41.jsonl", "gpt-4.1"),
}


def load_baseline(kind):
    """prompt-join-key -> baseline answer text."""
    if kind in BASELINES:
        name, label = BASELINES[kind]
        path = EVALSETS / name
        out = {}
        for line in path.open():
            if not line.strip():
                continue
            d = json.loads(line)
            msgs = d["messages"]
            text = None
            for m in msgs:
                content = m.get("content")
                if isinstance(content, dict):
                    content = content.get("answer")
                if m.get("role") == "assistant" and content:
                    text = content
            if text is None:
                raise SystemExit("no assistant answer for uid %s" % d.get("uid"))
            out[d["uid"]] = text
        return out, "uid", path, label
    path = EVALSETS / "ae_baseline_gpt4_1106.json"
    out = {norm(d["instruction"]): d["output"] for d in json.load(path.open())}
    return out, "instruction", path, "gpt4_1106_preview"


def judge_one(args, llm, tok):

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    panel_path = SUB / "panel" / args.panel
    panel = {}
    for line in panel_path.open():
        if line.strip():
            r = json.loads(line)
            panel[r["prompt_id"]] = r

    resp_path = SUB / "responses" / args.arm_responses / "responses.jsonl"
    arm = {}
    for line in resp_path.open():
        if not line.strip():
            continue
        e = json.loads(line)
        if int(e["response_index"]) != 0:
            continue
        arm[e["prompt_id"]] = e

    baseline, join, base_path, base_name = load_baseline(args.kind)

    pairs, missing_baseline = [], 0
    for pid, meta in panel.items():
        if pid not in arm:
            continue
        key = meta["uid"] if join == "uid" else norm(meta["instruction"])
        if key not in baseline:
            missing_baseline += 1
            continue
        pairs.append((pid, meta["instruction"], arm[pid]["response"], baseline[key]))
    if not pairs:
        raise SystemExit("no joinable prompts")

    out = SUB / "eval_pairwise" / args.out_tag
    out.mkdir(parents=True, exist_ok=True)
    settings = {
        "kind": args.kind, "arm": args.arm_name,
        "panel": str(panel_path), "panel_sha256": file_hash(panel_path),
        "arm_responses": str(resp_path), "arm_responses_sha256": file_hash(resp_path),
        "baseline": base_name, "baseline_file": str(base_path),
        "baseline_sha256": file_hash(base_path),
        "judge": args.judge, "judge_revision": args.judge_revision,
        "tensor_parallel_size": args.tensor_parallel_size,
        "judge_revision_is_a_declaration": True,
        "decoding": "temperature 0, thinking disabled (final-eval contract)",
        "orders": "both, verdict mapped back to the arm",
        "ties": 0.5, "unparsed": "left missing, never retried",
        "not_the_official_metric": (
            "Arena-Hard normally judges with gpt-4-1106 and AlpacaEval 2.0 with "
            "weighted_alpaca_eval_gpt4_turbo. This run judges with the local "
            + os.path.basename(args.judge.rstrip("/")) +
            " to keep the paid-API cost at zero. Questions and baseline "
            "answers are the official static files, so arms are comparable with "
            "each other but these are not leaderboard numbers."),
        "prompts_joined": len(pairs), "prompts_missing_baseline": missing_baseline,
        "template_sha256": digest(TEMPLATE), "source_sha256": file_hash(__file__)}
    (out / "settings.json").write_text(json.dumps(settings, indent=2,
                                                  ensure_ascii=False) + "\n")


    requests, params, meta_rows = [], [], []
    truncated = 0
    for pid, instruction, arm_text, base_text in pairs:
        for order in (0, 1):
            a, b = (arm_text, base_text) if order == 0 else (base_text, arm_text)
            text = TEMPLATE.format(instruction=instruction, response_a=a, response_b=b)
            ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                          tokenize=True, add_generation_prompt=True,
                                          enable_thinking=False)
            budget = args.max_model_len - args.max_tokens
            if len(ids) > budget:
                ids = ids[:budget]
                truncated += 1
            requests.append({"prompt_token_ids": ids})
            params.append(SamplingParams(n=1, temperature=0.0,
                                         max_tokens=args.max_tokens))
            meta_rows.append({"prompt_id": pid, "order": order})

    generated = llm.generate(requests, params, use_tqdm=False)
    counts = Counter()
    per_prompt = defaultdict(list)
    with (out / "verdicts.jsonl").open("w") as stream:
        for m, res in zip(meta_rows, generated):
            found = VERDICT.findall(res.outputs[0].text)
            if len(found) != 1:
                value, status = None, ("no_verdict" if not found else "multiple_verdicts")
            else:
                v = found[0].upper()
                if v == "TIE":
                    value = 0.5
                elif m["order"] == 0:
                    value = 1.0 if v == "A" else 0.0
                else:
                    value = 0.0 if v == "A" else 1.0
                status = "ok"
            counts[status] += 1
            if value is not None:
                per_prompt[m["prompt_id"]].append(value)
            stream.write(json.dumps({**m, "value_for_arm": value,
                                     "status": status}) + "\n")

    complete = {p: float(np.mean(v)) for p, v in per_prompt.items() if len(v) == 2}
    vals = np.array([complete[p] for p in sorted(complete)])
    rng = np.random.default_rng(args.seed)
    draws = np.array([vals[rng.integers(0, len(vals), len(vals))].mean()
                      for _ in range(args.bootstrap)]) if len(vals) else np.array([])
    arm_len = [len(tok(arm[p]["response"]).input_ids) for p in sorted(complete)]
    base_key = (lambda pid: panel[pid]["uid"] if join == "uid"
                else norm(panel[pid]["instruction"]))
    base_len = [len(tok(baseline[base_key(p)]).input_ids) for p in sorted(complete)]
    report = {
        **settings,
        "prompts_scored": len(vals),
        "WIN_RATE_VS_BASELINE": float(vals.mean()) if len(vals) else None,
        "win_rate_ci95": [float(np.percentile(draws, 2.5)),
                          float(np.percentile(draws, 97.5))] if len(draws) else None,
        "status_counts": dict(counts),
        "valid_fraction": counts["ok"] / max(sum(counts.values()), 1),
        "truncated_judge_inputs": truncated,
        "response_tokens": {"arm_median": float(np.median(arm_len)),
                            "arm_mean": float(np.mean(arm_len)),
                            "baseline_median": float(np.median(base_len)),
                            "baseline_mean": float(np.mean(base_len))},
        "bootstrap": {"replicates": args.bootstrap, "unit": "prompt"},
    }
    (out / "complete.json").write_text(json.dumps(report, indent=2,
                                                  ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("kind", "arm", "prompts_scored", "WIN_RATE_VS_BASELINE",
                       "win_rate_ci95", "valid_fraction", "response_tokens")}), flush=True)
    return 0




def main():
    import copy
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--benches", nargs="+", default=["arenahard", "alpacaeval"])
    ap.add_argument("--responses-prefix", default="eval_uw_",
                    help="responses tag is prefix + arm + '_' + bench")
    ap.add_argument("--base-responses-prefix", default="eval_base_",
                    help="the base row reuses existing responses")
    ap.add_argument("--out-prefix", default="uw_e72_")
    ap.add_argument("--judge", default="/work/models/bases/Qwen2.5-72B-Instruct")
    ap.add_argument("--judge-revision",
                    default="495f39366efef23836d0cfae4fbe635880d2be31")
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260915)
    top = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(top.judge, local_files_only=True)
    llm = LLM(model=top.judge, tokenizer=top.judge,
              tensor_parallel_size=top.tensor_parallel_size, dtype="bfloat16",
              max_model_len=top.max_model_len,
              gpu_memory_utilization=top.gpu_memory_utilization,
              max_num_seqs=top.max_num_seqs, generation_config="vllm",
              seed=20260915, trust_remote_code=False)
    done, failed = [], []
    for arm in top.arms:
        for bench in top.benches:
            prefix = top.base_responses_prefix if arm == "base" else top.responses_prefix
            responses = (prefix + bench if arm == "base"
                         else prefix + arm + "_" + bench)
            one = copy.copy(top)
            one.kind = bench
            one.panel = "eval_%s.jsonl" % bench
            one.arm_responses = responses
            one.arm_name = arm
            one.out_tag = top.out_prefix + arm + "_" + bench
            target = SUB / "eval_pairwise" / one.out_tag / "complete.json"
            if target.exists():
                print(json.dumps({"skipped": one.out_tag}), flush=True); continue
            try:
                judge_one(one, llm, tok)
                done.append(one.out_tag)
            except Exception as exc:
                failed.append({"tag": one.out_tag, "error": str(exc)[:300]})
                print(json.dumps({"failed": one.out_tag,
                                  "error": str(exc)[:300]}), flush=True)
    print(json.dumps({"judged": done, "failed": failed}), flush=True)
    if failed:
        raise SystemExit("%d of %d judgings failed" % (len(failed),
                                                       len(done) + len(failed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
