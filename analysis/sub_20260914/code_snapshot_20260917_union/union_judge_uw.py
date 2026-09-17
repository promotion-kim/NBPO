# Generated from judge_pool_prosper.py by make_union_judge.py -- do not edit.
# Adds the learner triangle so the graph matches the union contract
# (cross 8x8 + learner C(8,2) + reference C(8,2) = 120 slots per item).
"""8Y+8Z pool judging under PROSPER's own 5-point PSC prompt (their Figure 4).

Differences from this campaign's binary judge, all of them to match PROSPER
(arXiv 2602.19041, Appendix C):

  * the prompt is PROSPER's Figure 4 template, transcribed from the paper: a
    five-point preference score in {0,1,2,3,4} for the FIRST response over the
    second, with their few-shot examples, returned as
    {"explanation": "...", "verdict": "..."}. -1 is their "totally confused"
    escape and is treated as missing.
  * the per-prompt checklist item goes in the Question slot, one check per
    judgment, which is what makes this the single-check (PSC) judge.
  * sampling is Table 6: temperature 0.8, top_p 0.9, top_k 20, 256 new tokens.
  * a pair whose rendered sequence exceeds 4096 tokens is EXCLUDED, not
    truncated. PROSPER drops them; truncating would silently judge a clipped
    response.
  * the presentation order is permuted and the score reversed before averaging,
    so value_for_i is 1 - verdict/4 when j is shown first.

The score is mapped to a preference probability as verdict/4, so 2 is
indifference and the scale is symmetric about it. Everything else -- the 92-pair
set, the namespaced per-draw seeds, the resumable chunking, the no-retry policy
-- is the campaign's existing contract.
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

UF = Path("/work/uf4_20260910")
SUB = Path("/work/sub_20260914")
SEED_NAMESPACE = "20260915-prosper-psc-judge:"
MAX_SEQUENCE = 4096

TEMPLATE = """You are an expert in comparing the quality of responses and giving preference scores. Based on the provided input instruction and two responses from a worker, assess two responses based on the following criteria:
1. Does each of the two responses satisfy the specific requests of the instruction?
2. Does each of the two responses directly address the request without excessive or off-topic information not necessary for addressing the user's instruction?
3. Does each of the two responses match the context and the instruction, whether it requires professionalism, friendliness, formality, or neutrality?
Accordingly, assign a score (in {0,1,2,3,4}) to indicate the preference for the first response over the second assessing how well the first response addresses the instruction compared to the second response.
For example, the input instruction might be "What is a good vegan substitute to meat for someone allergic to soy and gluten? Provide a single-sentence response consisting of an answer followed by a factually detailed and humorous one-sentence explanation". Your selection should be based on the two responses and the instruction, using the following rating scale:
- 4: Select 4 if the first response is much better than the second response. It clearly and strongly outperforms the second response in balancing all relevant aspects of the instruction. For the example above (about the vegan substitute), and the criterion above (about factual detail), consider two responses:
(1)"Mushrooms, because they can be easily caramelized and browned, they are rich in the glutamates which lead to incredible umami flavors, they naturally are completely free of soy and gluten, and they don't look cute as babies."
(2)"Mushrooms are tasty."
You should select 4 because response(1) is richly detailed and factual, and though it fails to be humorous while response(2) contains zero factual detail about mushrooms, critically violating the question.
- 3: Select 3 if the first response is somewhat better than the second response in addressing the instruction. Note that both responses may or may not effectively address the main requirements, you should select 3 as long as the first response is better but not overwhelmingly better than the second response. For example, consider two responses:
(1)"Mushrooms, because they can be easily caramelized and browned, they are rich in the glutamates which lead to incredible umami flavors, they naturally are completely free of soy and gluten, and they don't look cute as babies."
(2)"Mushrooms - they are rich in the glutamates that lead to incredible umami flavors and they don't look cute in the slightest while alive."
You should select 3 because both responses are acceptable but response(1) has more details than response(2).
Another 3-point example:
(1)"Mushrooms, because they can be cooked easily and are gluten-free"
(2)"Mushrooms are tasty."
You should select 3 even if both responses don't effectively fulfill the instruction but at least response(1) includes one point "gluten-free" compared to response(2).
- 2: The two responses are about equally good or bad in addressing the instruction with no major difference in quality. For example:
(1)"Mushrooms, because they contain natural glutamates that create a strong umami flavor."
(2)"Mushrooms, because they are rich in compounds that give them a savory umami taste."
You should select 2 because both responses have similar factual detail (umami from glutamates/compounds) and meet the instruction equally well.
Another 2-point example:
(1)"Mushrooms are tasty."
(2)"Mushrooms, because they can be cooked easily."
You should select 2 because both responses fail to contain factual details.
- 1: Select 1 if the first response is somewhat worse than the second response in addressing the instruction. Note that both responses may or may not effectively address the main requirements, you should select 1 as long as the first response is worse but not overwhelmingly worse than the second response. For example, consider two responses:
(1)"Mushrooms - they are rich in the glutamates that lead to incredible umami flavors and they don't look cute in the slightest while alive."
(2)"Mushrooms, because they can be easily caramelized and browned, they are rich in the glutamates which lead to incredible umami flavors, they naturally are completely free of soy and gluten, and they don't look cute as babies."
You should select 1 because both responses are acceptable but response(1) has less details than response(2).
Another 1-point example:
(1)"Mushrooms are tasty."
(2)"Mushrooms, because they can be cooked easily and are gluten-free"
You should select 1 even if both responses don't effectively fulfill the instruction but response(1) has no factual details at all and response(2) at least includes one point "gluten-free".
- 0: Select 0 if the first response is much worse than the second response. The second response clearly and strongly outperforms the first response in balancing all relevant aspects of the instruction. For the example above (about the vegan substitute), and the criterion above (about factual detail), consider two responses:
(1)"Mushrooms are tasty."
(2)"Mushrooms, because they can be easily caramelized and browned, they are rich in the glutamates which lead to incredible umami flavors, they naturally are completely free of soy and gluten, and they don't look cute as babies."
You should select 0 because response(1) contains zero factual detail about mushrooms, critically violating the question while response(2) is richly detailed and factual, though it fails to be humorous.
Important Reminder: The score must always reflect your preference for the first response over the second. This is a relative judgment - do not evaluate each response in isolation. Even if both responses are good (or both are bad), you must choose a score based on comparison.
Your score can be any number in {0,1,2,3,4}. If you are totally confused, return -1 as a default. You should use your judgment to determine the most appropriate score. Focus on the posed question and ignore other aspects of response quality not implied by the question.
First provide a short 1-2 explanation of your choice and why you made it. Then provide your verdict as a single number in {0,1,2,3,4}. Use the format:
`{"explanation":"...", "verdict":"..."}`
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Input:
<<PROMPT>>
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Generated Text A:
<<RESPONSE_A>>
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Generated Text B:
<<RESPONSE_B>>
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Question:
<<CHECK>>
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Preference:
"""

VERDICT_JSON = re.compile(r'"verdict"\s*:\s*"?(-?\d+)"?')
VERDICT_BARE = re.compile(r"(-?\d+)\s*$")


def digest(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def file_hash(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def render(instruction, response_a, response_b, check):
    return (TEMPLATE
            .replace("<<PROMPT>>", instruction)
            .replace("<<RESPONSE_A>>", response_a)
            .replace("<<RESPONSE_B>>", response_b)
            .replace("<<CHECK>>", check))


def parse_verdict(text):
    """(value_for_first, status). None value means missing."""
    m = VERDICT_JSON.search(text)
    if m is None:
        m = VERDICT_BARE.search(text.strip())
        if m is None:
            return None, "no_verdict"
    try:
        score = int(m.group(1))
    except ValueError:
        return None, "unparsable_verdict"
    if score == -1:
        return None, "judge_declared_confused"
    if not 0 <= score <= 4:
        return None, "verdict_out_of_range"
    return score / 4.0, "ok"


def load_pool(name, shards):
    pool, settings = {}, None
    for shard in range(shards):
        directory = UF / "pools" / name / ("shard%d" % shard)
        settings = json.loads((directory / "settings.json").read_text())
        for path in sorted(directory.glob("chunk*.jsonl")):
            manifest = json.loads((directory / (path.stem + ".manifest.json")).read_text())
            if file_hash(path) != manifest["sha256"]:
                raise SystemExit("pool chunk hash mismatch: %s" % path)
            with path.open() as stream:
                for line in stream:
                    e = json.loads(line)
                    pool.setdefault(e["prompt_id"], {}).setdefault(
                        e["role"], {})[e["sample_index"]] = e
    return pool, settings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--pool-shards", type=int, default=4)
    ap.add_argument("--per-prompt-items", required=True)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--judge", default=str(UF / "assets/Qwen3-14B"))
    ap.add_argument("--judge-revision", default="40c069824f4251a91eefaf281ebe4c544efd3e18")
    ap.add_argument("--panel", default="A")
    ap.add_argument("--draws-per-order", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=MAX_SEQUENCE)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--chunk-prompts", type=int, default=10)
    ap.add_argument("--prompt-list", default=None,
                    help="JSON list of prompt ids to restrict to. Used by the "
                         "draws-per-pair ablation so the comparison is paired on the "
                         "same prompts and responses as the two-draw run.")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    items_path = Path(args.per_prompt_items)
    per_prompt_items = json.loads(items_path.read_text())
    pool, pool_settings = load_pool(args.pool, args.pool_shards)
    selected = [p for p in sorted(pool) if p in per_prompt_items]
    if args.prompt_list:
        keep = set(json.loads(Path(args.prompt_list).read_text()))
        missing = keep - set(selected)
        if missing:
            raise SystemExit("%d requested prompts are not in the pool" % len(missing))
        selected = [p for p in selected if p in keep]
    prompt_ids = selected[args.shard::args.shards]
    if not prompt_ids:
        raise SystemExit("no prompts selected")
    I = int(pool_settings["pool_per_role"])

    out = SUB / "pool_judgments" / args.out_tag / ("shard%d" % args.shard)
    out.mkdir(parents=True, exist_ok=True)
    n_items = max(len(v) for v in per_prompt_items.values())
    settings = {
        "pool": args.pool, "pool_sha256_settings": digest(json.dumps(pool_settings, sort_keys=True)),
        "judge": args.judge, "judge_revision": args.judge_revision,
        "judge_revision_is_a_declaration": True,
        "judge_prompt": ("PROSPER Figure 4, the single-check (PSC) five-point template, "
                         "transcribed from arXiv 2602.19041"),
        "scale": "verdict in {0,1,2,3,4} for the FIRST response; p = verdict/4; 2 is indifference",
        "confused_escape": "-1 is PROSPER's confused default and is recorded as missing",
        "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                     "top_k": args.top_k, "max_new_tokens": args.max_tokens},
        "sampling_source": "PROSPER Appendix C Table 6",
        "length_policy": ("a rendered sequence longer than %d tokens is EXCLUDED, not "
                          "truncated, which is PROSPER's protocol" % args.max_model_len),
        "orders": "both; the score is reversed before averaging when j is shown first",
        "draws_per_order": args.draws_per_order,
        "criteria_source": ("each prompt's own checklist items in the Question slot; "
                            "item k of two different prompts is never one objective"),
        "items_file": str(items_path), "items_sha256": file_hash(items_path),
        "pairs_per_prompt": {"learner_vs_comparator": I * I,
                             "comparator_vs_comparator": I * (I - 1) // 2},
        "verdicts_per_prompt": (I * I + I * (I - 1) // 2) * n_items * 2 * args.draws_per_order,
        "template_sha256": digest(TEMPLATE),
        "no_selective_retries": True,
        "seed_rule": ("SHA256('%s' + panel:prompt:role_i:i:role_j:j:item:order:draw)"
                      "[:16] mod (2**63-1)" % SEED_NAMESPACE),
        "shard": args.shard, "shards": args.shards, "n_prompts": len(prompt_ids),
        "source_sha256": file_hash(__file__)}
    (out / "settings.json").write_text(json.dumps(settings, indent=2,
                                                  ensure_ascii=False) + "\n")

    tok = AutoTokenizer.from_pretrained(args.judge, local_files_only=True)
    llm = LLM(model=args.judge, tokenizer=args.judge, tensor_parallel_size=1,
              dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_num_seqs=args.max_num_seqs, generation_config="vllm",
              seed=20260915, trust_remote_code=False)

    def pairs_for(prompt):
        """The union contract's graph: cross block, learner triangle, reference
        triangle -- 64 + 28 + 28 = 120 semantic slots per item, two orders each.
        The recorded PROSPER-setting judge omitted the learner triangle, which is
        the block the learner-side losses actually fit."""
        learners, comparators = prompt["learner"], prompt["comparator"]
        for i in sorted(learners):
            for j in sorted(comparators):
                yield ("learner", i, "comparator", j)
        for i, j in itertools.combinations(sorted(learners), 2):
            yield ("learner", i, "learner", j)
        for i, j in itertools.combinations(sorted(comparators), 2):
            yield ("comparator", i, "comparator", j)

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
        excluded = 0
        for pid in chunk:
            prompt = pool[pid]
            instruction = prompt["learner"][0]["prompt"]
            items = per_prompt_items[pid]
            for role_i, i, role_j, j in pairs_for(prompt):
                ev_i, ev_j = prompt[role_i][i], prompt[role_j][j]
                for item_index, item in enumerate(items):
                    for order in (0, 1):
                        a, b = (ev_i, ev_j) if order == 0 else (ev_j, ev_i)
                        text = render(instruction, a["response"], b["response"],
                                      item["text"])
                        ids = tok.apply_chat_template(
                            [{"role": "user", "content": text}], tokenize=True,
                            add_generation_prompt=True, enable_thinking=False)
                        if len(ids) + args.max_tokens > args.max_model_len:
                            excluded += 1
                            continue
                        for draw in range(args.draws_per_order):
                            key = "%s:%s:%s:%d:%s:%d:item%d:%d:%d" % (
                                args.panel, pid, role_i, i, role_j, j, item_index,
                                order, draw)
                            seed = int(digest(SEED_NAMESPACE + key)[:16], 16) % (2**63 - 1)
                            requests.append({"prompt_token_ids": ids})
                            params.append(SamplingParams(
                                n=1, temperature=args.temperature, top_p=args.top_p,
                                top_k=args.top_k, max_tokens=args.max_tokens, seed=seed))
                            meta.append({"prompt_id": pid, "role_i": role_i, "i": i,
                                         "role_j": role_j, "j": j,
                                         "rubric": "item%d" % item_index,
                                         "order": order, "draw": draw,
                                         "panel": args.panel, "seed": seed})
        t0 = time.monotonic()
        generated = llm.generate(requests, params, use_tqdm=False) if requests else []
        elapsed = time.monotonic() - t0
        counts = Counter()
        tmp = dest.with_suffix(".jsonl.tmp")
        with tmp.open("w") as stream:
            for m, result in zip(meta, generated):
                raw = result.outputs[0].text
                value_first, status = parse_verdict(raw)
                if value_first is None:
                    value = None
                else:
                    # the verdict scores the response shown FIRST; reverse it when
                    # j was shown first so every row is the preference for i
                    value = value_first if m["order"] == 0 else 1.0 - value_first
                counts[status] += 1
                stream.write(json.dumps({**m, "value_for_i": value,
                                         "status": status}, ensure_ascii=False) + "\n")
        tmp.replace(dest)
        manifest = {"chunk": index, "prompts": len(chunk), "verdicts": len(requests),
                    "excluded_over_length": excluded, "seconds": elapsed,
                    "verdicts_per_second": len(requests) / max(elapsed, 1e-9),
                    "status_counts": dict(counts), "sha256": file_hash(dest)}
        (out / ("chunk%04d.manifest.json" % index)).write_text(
            json.dumps(manifest, indent=2) + "\n")
        totals.update(counts)
        totals["verdicts"] += len(requests)
        totals["excluded_over_length"] += excluded
        print(json.dumps({k: manifest[k] for k in
                          ("chunk", "prompts", "verdicts", "excluded_over_length",
                           "verdicts_per_second")}), flush=True)

    report = {"out_tag": args.out_tag, "shard": args.shard,
              "n_prompts": len(prompt_ids), "chunks": len(chunks),
              "verdicts": totals["verdicts"],
              "excluded_over_length": totals["excluded_over_length"],
              "status_counts": {k: v for k, v in totals.items()
                                if k not in ("verdicts", "excluded_over_length")},
              "valid_fraction": totals["ok"] / max(totals["verdicts"], 1),
              "seconds": time.monotonic() - started_all}
    (out / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
