"""Collect the Appendix C within-rubric pairwise judgments with the local Qwen3-14B.

For every audit prompt, every unordered pair of its four responses, every rubric
and every presentation order, two independent panels A and B each draw five
judgments. Seeds are derived from a declared namespace so every draw is
reproducible and no two draws share a stream.

A judgment that does not parse to an explicit A / B / TIE verdict is recorded as
missing. It is never retried selectively and never silently dropped, because the
denominator has to keep it.

Verdicts are mapped back to the fixed response ids as 1 / 0 / 0.5 for the first
listed response, after undoing the presentation order. Nothing here derives a
preference from a score.
"""
from __future__ import annotations

import argparse, hashlib, itertools, json, re, time
from collections import Counter
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
SEED_NAMESPACE = "20260911-uf4-audit-judge:"
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-name", default="v1")
    ap.add_argument("--split-file", required=True, choices=("audit_100", "pilot_20"))
    ap.add_argument("--judge", default=str(ROOT / "assets/Qwen3-14B"))
    ap.add_argument("--judge-revision", default="40c069824f4251a91eefaf281ebe4c544efd3e18")
    ap.add_argument("--rubrics", default=str(ROOT / "audit/v1/rubrics.json"))
    ap.add_argument("--panels", nargs="+", default=["A", "B"])
    ap.add_argument("--draws-per-order", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--limit-prompts", type=int)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.judge, local_files_only=True)
    rubrics = json.loads(Path(args.rubrics).read_text())["rubrics"]

    base = ROOT / args.audit_name if (ROOT / args.audit_name).exists() else ROOT / "audit" / args.audit_name
    resp_path = base / "responses" / args.split_file / "responses.jsonl"
    events = [json.loads(l) for l in resp_path.open() if l.strip()]
    by_prompt = {}
    for e in events:
        by_prompt.setdefault(e["prompt_id"], {})[e["response_index"]] = e
    prompt_ids = sorted(by_prompt)
    if args.limit_prompts:
        prompt_ids = prompt_ids[:args.limit_prompts]

    out = base / "judgments" / args.split_file
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "judgments.jsonl"
    if dest.exists():
        print(json.dumps({"skipped": "already collected", "path": str(dest)}), flush=True)
        return

    settings = {
        "judge": args.judge, "judge_revision": args.judge_revision,
        "thinking": "disabled via the chat template's enable_thinking=False",
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
        "max_output_tokens": args.max_tokens, "input_budget_tokens": args.max_model_len,
        "panels": args.panels, "draws_per_order": args.draws_per_order,
        "rubrics_sha256": file_hash(args.rubrics),
        "template_sha256": digest(TEMPLATE),
        "responses_sha256": file_hash(resp_path),
        "verdict_grammar": "[[A]] / [[B]] / [[TIE]]; anything else is missing",
        "no_selective_retries": True,
        "seed_rule": f"SHA256('{SEED_NAMESPACE}' + panel:prompt:i:j:rubric:order:draw)[:16] mod (2**63-1)",
        "source_sha256": file_hash(__file__)}
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    llm = LLM(model=args.judge, tokenizer=args.judge, tensor_parallel_size=1, dtype="bfloat16",
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
              max_num_seqs=64, generation_config="vllm", seed=20260911, trust_remote_code=False)

    requests, params, meta = [], [], []
    truncated = 0
    for pid in prompt_ids:
        responses = by_prompt[pid]
        instruction = responses[0]["instruction"]
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
                    for panel in args.panels:
                        for draw in range(args.draws_per_order):
                            key = f"{panel}:{pid}:{i}:{j}:{rubric_name}:{order}:{draw}"
                            seed = int(digest(SEED_NAMESPACE + key)[:16], 16) % (2**63 - 1)
                            requests.append({"prompt_token_ids": ids})
                            params.append(SamplingParams(
                                n=1, temperature=args.temperature, top_p=args.top_p,
                                top_k=args.top_k, max_tokens=args.max_tokens, seed=seed))
                            meta.append({"panel": panel, "prompt_id": pid, "i": i, "j": j,
                                         "rubric": rubric_name, "order": order, "draw": draw,
                                         "first_index": first, "second_index": second,
                                         "seed": seed,
                                         "prompt_tokens": len(ids)})

    print(json.dumps({"phase": "start", "prompts": len(prompt_ids), "judgments": len(requests),
                      "truncated_judge_inputs": truncated}), flush=True)
    started = time.monotonic()
    generated = llm.generate(requests, params, use_tqdm=False)
    elapsed = time.monotonic() - started

    counts = Counter()
    out_tokens = 0
    with dest.open("x") as stream:
        for m, result in zip(meta, generated):
            o = result.outputs[0]
            text = o.text
            out_tokens += len(o.token_ids)
            found = VERDICT.findall(text)
            if len(found) != 1:
                verdict, value, status = None, None, ("no_verdict" if not found else "multiple_verdicts")
            else:
                verdict = found[0].upper()
                # Undo the presentation order: value is for the LOWER-indexed response i.
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
                                     "raw": text[-400:]}, ensure_ascii=False) + "\n")

    report = {"split_file": args.split_file, "n_prompts": len(prompt_ids),
              "n_judgments": len(requests), "seconds": elapsed,
              "judgments_per_second": len(requests) / elapsed,
              "output_tokens": out_tokens, "mean_output_tokens": out_tokens / max(len(requests), 1),
              "status_counts": dict(counts),
              "valid_fraction": counts["ok"] / max(len(requests), 1),
              "truncated_judge_inputs": truncated,
              "judgments_sha256": file_hash(dest), "settings_sha256": file_hash(out / "settings.json")}
    (out / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
