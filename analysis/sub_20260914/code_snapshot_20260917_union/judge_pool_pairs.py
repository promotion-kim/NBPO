"""Direct pairwise judging of the 8Y+8Z pool: 64 cross pairs and 28 reference pairs.

This is the feedback the policy panel trains on, and it is collected by the
local judge on raw text. It deliberately does NOT reuse the UF-4 scoring stage,
whose teacher is a ModernBERT score model fitted to UltraFeedback attribute
ratings: reusing a score-induced teacher and calling the result new conflict
data is exactly the confound this campaign exists to avoid.

Per prompt, per objective, both presentation orders, two independent repeats:

    64  learner_i  vs comparator_j     -> A_policy
    28  comparator_i vs comparator_j   -> A_ref (the reference bank's own matrix)
    ---
    92 pairs x 2 objectives x 2 orders x 2 repeats = 736 verdicts

Learner-learner pairs are NOT judged: they are outside the declared 92-pair
budget, and adding them for one method only would give that method extra
information.

Verdicts are written per shard as raw JSONL, with the same grammar, seed rule
and order handling as the screening judge. An unparsed verdict stays missing.
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
SEED_NAMESPACE = "20260914-sub-pool-judge:"
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


def load_pool(pool_name, shards):
    """prompt_id -> {role: {index: event}}, hash-verified like the solver does."""
    pool, settings = {}, None
    for shard in range(shards):
        directory = UF / "pools" / pool_name / ("shard%d" % shard)
        shard_settings = json.loads((directory / "settings.json").read_text())
        comparable = {k: v for k, v in shard_settings.items() if k != "shard"}
        if settings is None:
            settings = comparable
        elif settings != comparable:
            raise ValueError("pool shards were generated under different settings")
        for path in sorted(directory.glob("chunk*.jsonl")):
            manifest = json.loads((directory / (path.stem + ".manifest.json")).read_text())
            if file_hash(path) != manifest["sha256"]:
                raise ValueError("pool chunk hash mismatch: %s" % path)
            with path.open() as stream:
                for line in stream:
                    e = json.loads(line)
                    pool.setdefault(e["prompt_id"], {}).setdefault(
                        e["role"], {})[e["sample_index"]] = e
    return pool, settings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, help="pool name under /work/uf4_20260910/pools")
    ap.add_argument("--pool-shards", type=int, default=4)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--judge", default=str(UF / "assets/Qwen3-14B"))
    ap.add_argument("--judge-revision", default="40c069824f4251a91eefaf281ebe4c544efd3e18")
    ap.add_argument("--rubrics", default=str(SUB / "contract/rubrics_safe.json"))
    ap.add_argument("--panel", default="A")
    ap.add_argument("--draws-per-order", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--chunk-prompts", type=int, default=50,
                    help="prompts per output chunk, so a long run is resumable")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rubric_path = Path(args.rubrics)
    rubrics = json.loads(rubric_path.read_text())["rubrics"]
    pool, pool_settings = load_pool(args.pool, args.pool_shards)
    prompt_ids = sorted(pool)[args.shard::args.shards]
    pool_per_role = int(pool_settings["pool_per_role"])

    out = SUB / "pool_judgments" / args.out_tag / ("shard%d" % args.shard)
    out.mkdir(parents=True, exist_ok=True)
    settings = {
        "pool": args.pool, "pool_shards": args.pool_shards,
        "judge": args.judge, "judge_revision": args.judge_revision,
        "judge_revision_is_a_declaration": ("the local directory carries no upstream "
                                            "snapshot id; this is our record"),
        "feedback_kind": ("direct pairwise LLM judgment on raw text; NOT the UF-4 "
                          "score-induced ModernBERT teacher"),
        "thinking": "disabled via the chat template's enable_thinking=False",
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
        "max_output_tokens": args.max_tokens, "input_budget_tokens": args.max_model_len,
        "panel": args.panel, "draws_per_order": args.draws_per_order,
        "pairs_per_prompt": {"learner_vs_comparator": pool_per_role ** 2,
                             "comparator_vs_comparator":
                                 pool_per_role * (pool_per_role - 1) // 2,
                             "learner_vs_learner": 0},
        "verdicts_per_prompt": ((pool_per_role ** 2
                                 + pool_per_role * (pool_per_role - 1) // 2)
                                * len(rubrics) * 2 * args.draws_per_order),
        "rubrics": sorted(rubrics), "rubrics_sha256": file_hash(rubric_path),
        "template_sha256": digest(TEMPLATE),
        "verdict_grammar": "[[A]] / [[B]] / [[TIE]]; anything else is missing",
        "no_selective_retries": True,
        "seed_rule": ("SHA256('%s' + panel:prompt:role_i:i:role_j:j:rubric:order:draw)"
                      "[:16] mod (2**63-1)" % SEED_NAMESPACE),
        "shard": args.shard, "shards": args.shards, "n_prompts": len(prompt_ids),
        "source_sha256": file_hash(__file__)}
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    tok = AutoTokenizer.from_pretrained(args.judge, local_files_only=True)
    llm = LLM(model=args.judge, tokenizer=args.judge, tensor_parallel_size=1,
              dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=64,
              generation_config="vllm", seed=20260914, trust_remote_code=False)

    def pairs_for(prompt):
        learners, comparators = prompt["learner"], prompt["comparator"]
        for i in sorted(learners):
            for j in sorted(comparators):
                yield ("learner", i, "comparator", j)
        for i, j in itertools.combinations(sorted(comparators), 2):
            yield ("comparator", i, "comparator", j)

    chunks = [prompt_ids[i:i + args.chunk_prompts]
              for i in range(0, len(prompt_ids), args.chunk_prompts)]
    totals = Counter()
    started_all = time.monotonic()
    for chunk_index, chunk in enumerate(chunks):
        dest = out / ("chunk%04d.jsonl" % chunk_index)
        if dest.exists():
            print(json.dumps({"skipped_chunk": chunk_index}), flush=True)
            continue
        requests, params, meta = [], [], []
        truncated = 0
        for pid in chunk:
            prompt = pool[pid]
            instruction = prompt["learner"][0]["prompt"]
            for role_i, i, role_j, j in pairs_for(prompt):
                first_event = prompt[role_i][i]
                second_event = prompt[role_j][j]
                for rubric_name, rubric_text in rubrics.items():
                    for order in (0, 1):
                        a, b = ((first_event, second_event) if order == 0
                                else (second_event, first_event))
                        text = TEMPLATE.format(rubric=rubric_text, instruction=instruction,
                                               response_a=a["response"],
                                               response_b=b["response"])
                        ids = tok.apply_chat_template(
                            [{"role": "user", "content": text}], tokenize=True,
                            add_generation_prompt=True, enable_thinking=False)
                        budget = args.max_model_len - args.max_tokens
                        if len(ids) > budget:
                            ids = ids[:budget]
                            truncated += 1
                        for draw in range(args.draws_per_order):
                            key = "%s:%s:%s:%d:%s:%d:%s:%d:%d" % (
                                args.panel, pid, role_i, i, role_j, j, rubric_name,
                                order, draw)
                            seed = int(digest(SEED_NAMESPACE + key)[:16], 16) % (2**63 - 1)
                            requests.append({"prompt_token_ids": ids})
                            params.append(SamplingParams(
                                n=1, temperature=args.temperature, top_p=args.top_p,
                                top_k=args.top_k, max_tokens=args.max_tokens, seed=seed))
                            meta.append({"prompt_id": pid, "role_i": role_i, "i": i,
                                         "role_j": role_j, "j": j, "rubric": rubric_name,
                                         "order": order, "draw": draw, "panel": args.panel,
                                         "seed": seed})
        started = time.monotonic()
        generated = llm.generate(requests, params, use_tqdm=False)
        elapsed = time.monotonic() - started
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
                                         "status": status,
                                         "output_tokens": len(o.token_ids)},
                                        ensure_ascii=False) + "\n")
        tmp.replace(dest)
        manifest = {"chunk": chunk_index, "prompts": len(chunk),
                    "verdicts": len(requests), "seconds": elapsed,
                    "verdicts_per_second": len(requests) / max(elapsed, 1e-9),
                    "status_counts": dict(counts),
                    "truncated_judge_inputs": truncated,
                    "sha256": file_hash(dest)}
        (out / ("chunk%04d.manifest.json" % chunk_index)).write_text(
            json.dumps(manifest, indent=2) + "\n")
        totals.update(counts)
        totals["verdicts"] += len(requests)
        print(json.dumps({"chunk": chunk_index, **{k: manifest[k] for k in
                                                   ("prompts", "verdicts", "seconds",
                                                    "verdicts_per_second")}}), flush=True)

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
