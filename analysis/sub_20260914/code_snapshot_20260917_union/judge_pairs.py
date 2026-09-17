"""Direct pairwise judgments for the submission screening contract.

Adapted from the UF-4 audit judge, keeping its template, verdict grammar,
order handling and per-draw seed derivation, and changing only what the new
contract changes: two global rubrics instead of four, eight responses per
prompt (so 28 unordered pairs), and a panel/repeat structure that separates
the screening repetitions from the later confirmation repetitions.

    screening    --panel A --draws-per-order 2   28 x 2 orders x 2 x K verdicts
    confirmation --panel B --draws-per-order 5   on the preselected 50 prompts

Panel identity enters the seed, so panel B is an independent draw rather than a
re-read of panel A's stream, and no edge is ever re-judged selectively: the
prompt list comes from the frozen panel file.

A judgment that does not parse to exactly one [[A]] / [[B]] / [[TIE]] is
recorded as missing with its raw tail. Nothing is retried and nothing is
dropped, because the denominator has to keep it.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import time
from collections import Counter
from pathlib import Path

ROOT = Path("/work/sub_20260914")
SEED_NAMESPACE = "20260914-sub-judge:"
VERDICT = re.compile(r"\[\[(A|B|TIE)\]\]", re.IGNORECASE)
TEMPLATE = """You are comparing two responses to the same user message under one specific criterion.

Criterion: {rubric}

User message:
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--responses", required=True,
                    help="directory under responses/, e.g. screen200")
    ap.add_argument("--prompt-list", default=None,
                    help="frozen panel file restricting the prompts, e.g. pilot10.jsonl")
    ap.add_argument("--out-tag", required=True, help="directory under judgments/")
    ap.add_argument("--judge", default="/work/uf4_20260910/assets/Qwen3-14B")
    ap.add_argument("--judge-revision", default="40c069824f4251a91eefaf281ebe4c544efd3e18")
    ap.add_argument("--rubrics", default=str(ROOT / "contract/rubrics_safe.json"))
    ap.add_argument("--panel", default="A")
    ap.add_argument("--draws-per-order", type=int, default=2)
    ap.add_argument("--exclude-prompt-list", default=None,
                    help="panel file whose prompts are already judged in another shard")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rubric_path = Path(args.rubrics)
    rubrics = json.loads(rubric_path.read_text())["rubrics"]
    resp_path = ROOT / "responses" / args.responses / "responses.jsonl"
    events = [json.loads(l) for l in resp_path.open() if l.strip()]
    by_prompt = {}
    for e in events:
        by_prompt.setdefault(e["prompt_id"], {})[e["response_index"]] = e

    keep = set(by_prompt)
    if args.prompt_list:
        wanted = {json.loads(l)["prompt_id"]
                  for l in (ROOT / "panel" / args.prompt_list).open() if l.strip()}
        keep &= wanted
    if args.exclude_prompt_list:
        drop = {json.loads(l)["prompt_id"]
                for l in (ROOT / "panel" / args.exclude_prompt_list).open() if l.strip()}
        keep -= drop
    prompt_ids = sorted(keep)
    if args.shards > 1:
        prompt_ids = prompt_ids[args.shard::args.shards]
    if not prompt_ids:
        raise SystemExit("no prompts selected")

    out = ROOT / "judgments" / args.out_tag
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "judgments.jsonl"
    if dest.exists():
        print(json.dumps({"skipped": "already collected", "path": str(dest)}), flush=True)
        return 0

    tok = AutoTokenizer.from_pretrained(args.judge, local_files_only=True)
    settings = {
        "judge": args.judge, "judge_revision": args.judge_revision,
        "judge_revision_is_a_declaration": ("the local directory carries no upstream "
                                            "snapshot id, so this revision is our record, "
                                            "not an independently verified identifier"),
        "thinking": "disabled via the chat template's enable_thinking=False",
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
        "max_output_tokens": args.max_tokens, "input_budget_tokens": args.max_model_len,
        "panel": args.panel, "draws_per_order": args.draws_per_order,
        "orders": "both presentation orders, verdict mapped back to the lower index",
        "rubrics": sorted(rubrics), "rubrics_sha256": file_hash(rubric_path),
        "template_sha256": digest(TEMPLATE),
        "responses_sha256": file_hash(resp_path),
        "prompt_list": args.prompt_list, "exclude_prompt_list": args.exclude_prompt_list,
        "shard": args.shard, "shards": args.shards, "n_prompts": len(prompt_ids),
        "verdict_grammar": "[[A]] / [[B]] / [[TIE]]; anything else is missing",
        "no_selective_retries": True,
        "seed_rule": ("SHA256('%s' + panel:prompt:i:j:rubric:order:draw)[:16] "
                      "mod (2**63-1)" % SEED_NAMESPACE),
        "source_sha256": file_hash(__file__)}
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    llm = LLM(model=args.judge, tokenizer=args.judge, tensor_parallel_size=1,
              dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=64,
              generation_config="vllm", seed=20260914, trust_remote_code=False)

    requests, params, meta = [], [], []
    truncated = 0
    for pid in prompt_ids:
        responses = by_prompt[pid]
        instruction = responses[min(responses)]["instruction"]
        for i, j in itertools.combinations(sorted(responses), 2):
            for rubric_name, rubric_text in rubrics.items():
                for order in (0, 1):
                    first, second = (i, j) if order == 0 else (j, i)
                    text = TEMPLATE.format(rubric=rubric_text, instruction=instruction,
                                           response_a=responses[first]["response"],
                                           response_b=responses[second]["response"])
                    ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                                  tokenize=True, add_generation_prompt=True,
                                                  enable_thinking=False)
                    budget = args.max_model_len - args.max_tokens
                    if len(ids) > budget:
                        ids = ids[:budget]
                        truncated += 1
                    for draw in range(args.draws_per_order):
                        key = "%s:%s:%d:%d:%s:%d:%d" % (args.panel, pid, i, j,
                                                        rubric_name, order, draw)
                        seed = int(digest(SEED_NAMESPACE + key)[:16], 16) % (2**63 - 1)
                        requests.append({"prompt_token_ids": ids})
                        params.append(SamplingParams(
                            n=1, temperature=args.temperature, top_p=args.top_p,
                            top_k=args.top_k, max_tokens=args.max_tokens, seed=seed))
                        meta.append({"panel": args.panel, "prompt_id": pid, "i": i, "j": j,
                                     "rubric": rubric_name, "order": order, "draw": draw,
                                     "first_index": first, "second_index": second,
                                     "seed": seed, "prompt_tokens": len(ids)})

    print(json.dumps({"phase": "start", "prompts": len(prompt_ids),
                      "judgments": len(requests),
                      "truncated_judge_inputs": truncated}), flush=True)
    started = time.monotonic()
    generated = llm.generate(requests, params, use_tqdm=False)
    elapsed = time.monotonic() - started

    counts, out_tokens = Counter(), 0
    with dest.open("x") as stream:
        for m, result in zip(meta, generated):
            o = result.outputs[0]
            out_tokens += len(o.token_ids)
            found = VERDICT.findall(o.text)
            if len(found) != 1:
                verdict, value = None, None
                status = "no_verdict" if not found else "multiple_verdicts"
            else:
                verdict = found[0].upper()
                if verdict == "TIE":
                    value = 0.5
                elif m["order"] == 0:
                    value = 1.0 if verdict == "A" else 0.0
                else:
                    value = 0.0 if verdict == "A" else 1.0
                status = "ok"
            counts[status] += 1
            if status == "ok":
                counts["verdict_" + verdict] += 1
            stream.write(json.dumps({**m, "verdict": verdict, "value_for_i": value,
                                     "status": status, "output_tokens": len(o.token_ids),
                                     "finish_reason": o.finish_reason,
                                     "raw": o.text[-400:]}, ensure_ascii=False) + "\n")

    report = {"out_tag": args.out_tag, "panel": args.panel,
              "n_prompts": len(prompt_ids), "n_judgments": len(requests),
              "seconds": elapsed, "judgments_per_second": len(requests) / max(elapsed, 1e-9),
              "output_tokens": out_tokens,
              "mean_output_tokens": out_tokens / max(len(requests), 1),
              "status_counts": dict(counts),
              "valid_fraction": counts["ok"] / max(len(requests), 1),
              "truncated_judge_inputs": truncated,
              "judgments_sha256": file_hash(dest),
              "settings_sha256": file_hash(out / "settings.json")}
    (out / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
