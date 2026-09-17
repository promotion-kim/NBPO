"""Independent reference wins for one arm's fresh responses on the test panel.

One fresh response per test prompt is compared against the SAME four frozen
base reference responses, under both objectives and both presentation orders:

    1 fresh x 4 references x 2 objectives x 2 orders = 16 verdicts per prompt

W_1 and W_2 in tab:stress_results are the order-averaged win rates of the fresh
response against that reference bank, one per objective, and the row's
seed-wise minimum averages the per-seed minimum over objectives -- which with
one declared seed is that seed's minimum, reported as one seed.

The reference bank is generated once from the base and reused by every arm, so
the arms differ only in the response being judged. The base row uses its own
fresh draw under a different seed tag, so it is a genuine independent draw and
not a copy of a reference.

A whole-prompt paired bootstrap over the common prompts gives the interval, and
the minimum over objectives is recomputed inside every replicate rather than
taken from the point estimates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SUB = Path("/work/sub_20260914")
UF = Path("/work/uf4_20260910")
SEED_NAMESPACE = "20260914-sub-fresh-judge:"
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


def load_responses(tag):
    path = SUB / "responses" / tag / "responses.jsonl"
    by_prompt = defaultdict(dict)
    for line in path.open():
        if not line.strip():
            continue
        e = json.loads(line)
        by_prompt[e["prompt_id"]][e["response_index"]] = e
    return by_prompt, file_hash(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-tag", required=True, help="responses/<tag> of the arm's fresh draw")
    ap.add_argument("--reference-tag", required=True, help="responses/<tag> of the 4 references")
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--judge", default=str(UF / "assets/Qwen3-14B"))
    ap.add_argument("--judge-revision", default="40c069824f4251a91eefaf281ebe4c544efd3e18")
    ap.add_argument("--rubrics", default=str(SUB / "contract/rubrics_safe.json"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rubric_path = Path(args.rubrics)
    rubrics = json.loads(rubric_path.read_text())["rubrics"]
    arm, arm_sha = load_responses(args.arm_tag)
    refs, ref_sha = load_responses(args.reference_tag)
    prompts = sorted(set(arm) & set(refs))
    if not prompts:
        raise SystemExit("no prompts shared between the arm draw and the reference bank")

    out = SUB / "fresh_eval" / args.out_tag
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "verdicts.jsonl"
    if dest.exists():
        print(json.dumps({"skipped": "already judged", "path": str(dest)}), flush=True)
        return 0

    tok = AutoTokenizer.from_pretrained(args.judge, local_files_only=True)
    n_refs = len(next(iter(refs.values())))
    settings = {
        "arm_tag": args.arm_tag, "reference_tag": args.reference_tag,
        "arm_responses_sha256": arm_sha, "reference_responses_sha256": ref_sha,
        "judge": args.judge, "judge_revision": args.judge_revision,
        "judge_revision_is_a_declaration": True,
        "temperature": args.temperature, "max_output_tokens": args.max_tokens,
        "references_per_prompt": n_refs, "objectives": sorted(rubrics),
        "orders": 2, "verdicts_per_prompt": n_refs * len(rubrics) * 2,
        "rubrics_sha256": file_hash(rubric_path), "template_sha256": digest(TEMPLATE),
        "estimand": ("order-averaged win rate of the arm's fresh response against the "
                     "frozen four-response base reference bank, ties 0.5"),
        "not_a_policy_game_value": ("one fresh response per prompt does not identify the "
                                    "stochastic policy's game value"),
        "verdict_grammar": "[[A]] / [[B]] / [[TIE]]; anything else is missing",
        "source_sha256": file_hash(__file__)}
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    # Four concurrent judges on this node run each card at 30-40% utilisation and
    # 22.7 verdicts/s, against 40.2 for a single judge, because the cards starve
    # between batches on CPU-side preparation. These fresh-eval jobs have not
    # started yet, so they get a larger batch from the outset.
    llm = LLM(model=args.judge, tokenizer=args.judge, tensor_parallel_size=1,
              dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_num_seqs=args.max_num_seqs,
              generation_config="vllm", seed=args.seed, trust_remote_code=False)

    requests, params, meta = [], [], []
    truncated = 0
    for pid in prompts:
        fresh = arm[pid][0]
        instruction = fresh["instruction"]
        for r_index in sorted(refs[pid]):
            reference = refs[pid][r_index]
            for rubric_name, rubric_text in rubrics.items():
                for order in (0, 1):
                    a, b = ((fresh, reference) if order == 0 else (reference, fresh))
                    text = TEMPLATE.format(rubric=rubric_text, instruction=instruction,
                                           response_a=a["response"], response_b=b["response"])
                    ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                                  tokenize=True, add_generation_prompt=True,
                                                  enable_thinking=False)
                    budget = args.max_model_len - args.max_tokens
                    if len(ids) > budget:
                        ids = ids[:budget]
                        truncated += 1
                    requests.append({"prompt_token_ids": ids})
                    params.append(SamplingParams(n=1, temperature=args.temperature,
                                                 max_tokens=args.max_tokens))
                    meta.append({"prompt_id": pid, "reference_index": r_index,
                                 "rubric": rubric_name, "order": order})

    print(json.dumps({"phase": "start", "prompts": len(prompts),
                      "verdicts": len(requests)}), flush=True)
    started = time.monotonic()
    generated = llm.generate(requests, params, use_tqdm=False)
    elapsed = time.monotonic() - started

    counts = Counter()
    values = defaultdict(lambda: defaultdict(list))     # prompt -> rubric -> [value]
    with dest.open("x") as stream:
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
                values[m["prompt_id"]][m["rubric"]].append(value)
            counts[status] += 1
            stream.write(json.dumps({**m, "verdict": verdict, "value_for_fresh": value,
                                     "status": status}, ensure_ascii=False) + "\n")

    objectives = sorted(rubrics)
    per_prompt = {}
    for pid, table in values.items():
        if all(len(table.get(o, [])) == n_refs * 2 for o in objectives):
            per_prompt[pid] = [float(np.mean(table[o])) for o in objectives]
    common = sorted(per_prompt)
    mat = np.array([per_prompt[p] for p in common]) if common else np.zeros((0, len(objectives)))

    rng = np.random.default_rng(args.seed)
    point = mat.mean(axis=0) if len(common) else np.zeros(len(objectives))
    boot_w, boot_min = [], []
    for _ in range(args.bootstrap if len(common) else 0):
        idx = rng.integers(0, len(common), len(common))
        means = mat[idx].mean(axis=0)
        boot_w.append(means)
        boot_min.append(means.min())
    boot_w = np.array(boot_w) if boot_w else np.zeros((0, len(objectives)))

    def ci(samples):
        if not len(samples):
            return None
        return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]

    report = {
        "arm_tag": args.arm_tag, "out_tag": args.out_tag,
        "n_prompts_judged": len(prompts), "n_prompts_complete": len(common),
        "verdicts": len(requests), "seconds": elapsed,
        "verdicts_per_second": len(requests) / max(elapsed, 1e-9),
        "status_counts": dict(counts),
        "valid_fraction": counts["ok"] / max(len(requests), 1),
        "truncated_judge_inputs": truncated,
        "objectives": objectives,
        "win_rates": {o: float(point[i]) for i, o in enumerate(objectives)},
        "win_rate_ci95": {o: ci(boot_w[:, i]) for i, o in enumerate(objectives)},
        "min_objective": float(point.min()) if len(common) else None,
        "min_objective_ci95": ci(boot_min),
        "bootstrap": {"replicates": args.bootstrap,
                      "unit": "whole prompt, all objectives and orders resampled together",
                      "minimum_recomputed_per_replicate": True},
        "verdicts_sha256": file_hash(dest)}
    (out / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("arm_tag", "n_prompts_complete", "win_rates", "min_objective",
                       "valid_fraction", "verdicts_per_second")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
