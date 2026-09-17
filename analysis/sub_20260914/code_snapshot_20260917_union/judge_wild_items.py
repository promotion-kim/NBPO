"""Native item-level pairwise judging for the WildChecklists row of Table 37.

Unlike the Safe and UF panels, the criteria here are prompt-specific: each
prompt carries its own checklist, so the judged criterion comes from the panel
row rather than from a shared rubric file. That is the whole point of this row
-- it measures cycles WITHIN a native item, and the campaign never treats item
k of one prompt as the same objective as item k of another.

Per prompt, per item, both presentation orders, two independent repeats over
the 28 unordered pairs of its eight responses. The template, verdict grammar,
order handling and namespaced per-draw seeds are the same as the other panels,
so the three rows of Table 37 differ in their data and criteria and not in how
a comparison is asked.

Chunked by prompt so a long run is resumable, and an unparsed verdict stays
missing.
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

SUB = Path("/work/sub_20260914")
UF = Path("/work/uf4_20260910")
SEED_NAMESPACE = "20260915-sub-wild-judge:"
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
    ap.add_argument("--panel", default="wild200.jsonl")
    ap.add_argument("--responses", default="wild200")
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--judge", default=str(UF / "assets/Qwen3-14B"))
    ap.add_argument("--judge-revision", default="40c069824f4251a91eefaf281ebe4c544efd3e18")
    ap.add_argument("--panel-name", default="A")
    ap.add_argument("--draws-per-order", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--chunk-prompts", type=int, default=25)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    panel_path = SUB / "panel" / args.panel
    panel = {}
    for line in panel_path.open():
        if line.strip():
            row = json.loads(line)
            panel[row["prompt_id"]] = row

    resp_path = SUB / "responses" / args.responses / "responses.jsonl"
    by_prompt = {}
    for line in resp_path.open():
        if not line.strip():
            continue
        e = json.loads(line)
        by_prompt.setdefault(e["prompt_id"], {})[e["response_index"]] = e

    prompt_ids = sorted(set(panel) & set(by_prompt))[args.shard::args.shards]
    if not prompt_ids:
        raise SystemExit("no prompts selected")

    out = SUB / "wild_judgments" / args.out_tag
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.judge, local_files_only=True)
    settings = {
        "panel": str(panel_path), "panel_sha256": file_hash(panel_path),
        "responses_sha256": file_hash(resp_path),
        "judge": args.judge, "judge_revision": args.judge_revision,
        "judge_revision_is_a_declaration": True,
        "criteria_source": ("each prompt's own checklist items, from the panel row; "
                            "item k of one prompt is never treated as the same objective "
                            "as item k of another"),
        "thinking": "disabled via the chat template's enable_thinking=False",
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
        "max_output_tokens": args.max_tokens,
        "panel_name": args.panel_name, "draws_per_order": args.draws_per_order,
        "orders": "both, verdict mapped back to the lower-indexed response",
        "template_sha256": digest(TEMPLATE),
        "verdict_grammar": "[[A]] / [[B]] / [[TIE]]; anything else is missing",
        "no_selective_retries": True,
        "seed_rule": ("SHA256('%s' + panel:prompt:item:i:j:order:draw)[:16] mod (2**63-1)"
                      % SEED_NAMESPACE),
        "shard": args.shard, "shards": args.shards, "n_prompts": len(prompt_ids),
        "source_sha256": file_hash(__file__)}
    (out / "settings.json").write_text(json.dumps(settings, indent=2,
                                                  ensure_ascii=False) + "\n")

    llm = LLM(model=args.judge, tokenizer=args.judge, tensor_parallel_size=1,
              dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_num_seqs=args.max_num_seqs, generation_config="vllm",
              seed=20260915, trust_remote_code=False)

    chunks = [prompt_ids[i:i + args.chunk_prompts]
              for i in range(0, len(prompt_ids), args.chunk_prompts)]
    totals = Counter()
    started_all = time.monotonic()
    for index, chunk in enumerate(chunks):
        dest = out / ("chunk%04d.jsonl" % index)
        if dest.exists():
            print(json.dumps({"skipped_chunk": index}), flush=True)
            continue
        requests, params, meta = [], [], []
        truncated = 0
        for pid in chunk:
            responses = by_prompt[pid]
            instruction = responses[min(responses)]["instruction"]
            for item_index, item in enumerate(panel[pid]["items"]):
                for i, j in itertools.combinations(sorted(responses), 2):
                    for order in (0, 1):
                        first, second = (i, j) if order == 0 else (j, i)
                        text = TEMPLATE.format(rubric=item["text"],
                                               instruction=instruction,
                                               response_a=responses[first]["response"],
                                               response_b=responses[second]["response"])
                        ids = tok.apply_chat_template(
                            [{"role": "user", "content": text}], tokenize=True,
                            add_generation_prompt=True, enable_thinking=False)
                        budget = args.max_model_len - args.max_tokens
                        if len(ids) > budget:
                            ids = ids[:budget]
                            truncated += 1
                        for draw in range(args.draws_per_order):
                            key = "%s:%s:%d:%d:%d:%d:%d" % (args.panel_name, pid,
                                                            item_index, i, j, order, draw)
                            seed = int(digest(SEED_NAMESPACE + key)[:16], 16) % (2**63 - 1)
                            requests.append({"prompt_token_ids": ids})
                            params.append(SamplingParams(
                                n=1, temperature=args.temperature, top_p=args.top_p,
                                top_k=args.top_k, max_tokens=args.max_tokens, seed=seed))
                            meta.append({"prompt_id": pid, "item": item_index,
                                         "item_importance": item.get("importance"),
                                         "i": i, "j": j, "order": order, "draw": draw,
                                         "panel": args.panel_name, "seed": seed})
        t0 = time.monotonic()
        generated = llm.generate(requests, params, use_tqdm=False)
        elapsed = time.monotonic() - t0
        counts = Counter()
        tmp = dest.with_suffix(".jsonl.tmp")
        with tmp.open("w") as stream:
            for m, result in zip(meta, generated):
                o = result.outputs[0]
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
                stream.write(json.dumps({**m, "verdict": verdict, "value_for_i": value,
                                         "status": status}, ensure_ascii=False) + "\n")
        tmp.replace(dest)
        manifest = {"chunk": index, "prompts": len(chunk), "verdicts": len(requests),
                    "seconds": elapsed,
                    "verdicts_per_second": len(requests) / max(elapsed, 1e-9),
                    "status_counts": dict(counts),
                    "truncated_judge_inputs": truncated, "sha256": file_hash(dest)}
        (out / ("chunk%04d.manifest.json" % index)).write_text(
            json.dumps(manifest, indent=2) + "\n")
        totals.update(counts)
        totals["verdicts"] += len(requests)
        print(json.dumps({k: manifest[k] for k in
                          ("chunk", "prompts", "verdicts", "verdicts_per_second")}),
              flush=True)

    report = {"out_tag": args.out_tag, "shard": args.shard,
              "n_prompts": len(prompt_ids), "chunks": len(chunks),
              "verdicts": totals["verdicts"], "status_counts": dict(totals),
              "valid_fraction": totals["ok"] / max(totals["verdicts"], 1),
              "seconds": time.monotonic() - started_all}
    (out / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
