"""Cross-play judging: every unordered policy pair, every prompt, every item.

The contract asks for the frozen rubric prompt judged by the independent
evaluator in both orders, so this reuses the PSC five-point template of the
training judge -- byte-identical, imported rather than retyped -- and runs it
under the 72B evaluator. One response per policy per prompt is already sampled,
so a cell is a single comparison of two fixed texts under one item.

Both orders are always requested and a pair counts only when both parse:
P_k(a > b) = 1/2[v_ab/4 + 1 - v_ba/4]. An identical pair of texts is 0.5 by
convention and logged. Over-budget renderings are excluded, never truncated.
Nothing is aggregated here; the matrix, the minima and the bootstrap live in the
aggregator so that this stage can be re-read without re-judging.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from collections import Counter
from pathlib import Path

SUB = Path("/work/sub_20260914")
UF = Path("/work/uf4_20260910")
SEED_NS = "20260917-uw-crossplay:"


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
    ap.add_argument("--bank", nargs="+", required=True)
    ap.add_argument("--items", default=str(SUB / "panel/uw_crossplay_items.json"))
    ap.add_argument("--out-tag", default="uw_xplay")
    ap.add_argument("--judge", default="/work/models/bases/Qwen2.5-72B-Instruct")
    ap.add_argument("--judge-revision",
                    default="495f39366efef23836d0cfae4fbe635880d2be31")
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=4000)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    # the training judge's template and verdict grammar, imported so the two
    # stages cannot drift apart
    from judge_pool_prosper import TEMPLATE, parse_verdict, render

    items = json.loads(Path(args.items).read_text())
    responses = {}
    for arm in args.bank:
        path = SUB / "responses" / ("xplay_%s" % arm) / "responses.jsonl"
        table = {}
        for line in path.open():
            if line.strip():
                e = json.loads(line)
                table[e["prompt_id"]] = e
        responses[arm] = table
    prompts = sorted(set.intersection(*[set(v) for v in responses.values()]) & set(items))
    if not prompts:
        raise SystemExit("no prompt is shared by the bank and the item map")

    tok = AutoTokenizer.from_pretrained(args.judge, local_files_only=True)
    llm = LLM(model=args.judge, tokenizer=args.judge,
              tensor_parallel_size=args.tensor_parallel_size, dtype="bfloat16",
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_num_seqs=args.max_num_seqs, generation_config="vllm",
              seed=20260917, trust_remote_code=False)

    out = SUB / "crossplay" / args.out_tag
    out.mkdir(parents=True, exist_ok=True)
    pairs = list(itertools.combinations(args.bank, 2))
    planned = len(pairs) * len(prompts) * 4 * 2
    settings = {
        "bank": args.bank, "pairs": len(pairs), "prompts": len(prompts),
        "items_per_prompt": 4, "orders": 2, "planned_verdicts": planned,
        "judge": args.judge, "judge_revision": args.judge_revision,
        "template_sha256": digest(TEMPLATE),
        "template_source": ("the training judge's PSC five-point template, imported "
                            "from judge_pool_prosper rather than retyped"),
        "decoding": {"temperature": args.temperature, "max_tokens": args.max_tokens,
                     "max_model_len": args.max_model_len},
        "estimator": "P_k(a>b) = 1/2[v_ab/4 + 1 - v_ba/4], both orders required",
        "identical_text_convention": "0.5, logged",
        "over_budget": "excluded, never truncated",
        "seed_rule": "SHA256('%s' + a:b:prompt:item:order)[:16] mod (2**63-1)" % SEED_NS,
        "source_sha256": file_hash(__file__)}
    (out / "settings.json").write_text(json.dumps(settings, indent=1) + "\n")

    requests, meta = [], []
    excluded = 0
    for a, b in pairs:
        for pid in prompts:
            ra, rb = responses[a][pid], responses[b][pid]
            identical = ra["response_sha256"] == rb["response_sha256"]
            for item_index, item in enumerate(items[pid]):
                for order in (0, 1):
                    first, second = (ra, rb) if order == 0 else (rb, ra)
                    text = render(ra["instruction"], first["response"],
                                  second["response"], item["text"])
                    ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                                  tokenize=True,
                                                  add_generation_prompt=True,
                                                  enable_thinking=False)
                    if len(ids) + args.max_tokens > args.max_model_len:
                        excluded += 1
                        continue
                    seed = int(digest(SEED_NS + "%s:%s:%s:%d:%d"
                                      % (a, b, pid, item_index, order))[:16], 16) % (2**63 - 1)
                    requests.append({"prompt_token_ids": ids})
                    meta.append({"a": a, "b": b, "prompt_id": pid,
                                 "item": item_index, "order": order,
                                 "identical_text": identical, "seed": seed})

    sp = [SamplingParams(n=1, temperature=args.temperature, top_p=1.0, top_k=-1,
                         max_tokens=args.max_tokens, seed=m["seed"]) for m in meta]
    status = Counter()
    started = time.monotonic()
    with (out / "verdicts.jsonl").open("w") as stream:
        for begin in range(0, len(requests), args.chunk):
            part = slice(begin, begin + args.chunk)
            generated = llm.generate(requests[part], sp[part], use_tqdm=False)
            for m, g in zip(meta[part], generated):
                value, state = parse_verdict(g.outputs[0].text)
                if state == "ok" and m["order"] == 1:
                    value = 1.0 - value
                if m["identical_text"] and state == "ok":
                    value, state = 0.5, "identical_text"
                status[state] += 1
                stream.write(json.dumps({**{k: v for k, v in m.items() if k != "seed"},
                                         "value_for_a": value, "status": state}) + "\n")
            print(json.dumps({"done": min(begin + args.chunk, len(requests)),
                              "of": len(requests)}), flush=True)

    report = {**settings, "requested_verdicts": len(requests),
              "excluded_over_budget": excluded,
              "status_counts": dict(status),
              "parse_rate": round(status["ok"] / max(len(requests), 1), 6),
              "seconds": round(time.monotonic() - started, 1),
              "verdicts_sha256": file_hash(out / "verdicts.jsonl")}
    (out / "complete.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("prompts", "pairs", "requested_verdicts",
                       "excluded_over_budget", "parse_rate", "seconds")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
