"""MT-Bench single-answer grading with the campaign's local judge and no paid API.

The protocol is MT-Bench's own: the four official single-answer grading prompts
(plain and reference-based, first turn and multi-turn) verbatim, the official
system message, the official "[[rating]]" grammar on a 1-10 scale, and the
official rule that a reference answer is used exactly for the categories
MT-Bench declares need one (math, reasoning, coding). The one substitution is
the grader: GPT-4 is replaced by the frozen local Qwen3-14B already used
everywhere else in this campaign, at the same pinned revision and temperature 0.
That substitution is why the reported number is NOT an MT-Bench score and is
labelled as a local-judge score wherever it is printed.

Two deliberate choices are recorded because they change the numbers:

* Thinking mode is disabled. Qwen3's chat template leaves the assistant turn
  open by default and the model then emits a reasoning block before its answer,
  which consumes the generation budget before any rating appears. Grading asks
  for a short explanation and a rating, so the template is instantiated with
  `enable_thinking=False` and the budget is spent on the graded text.
* An unparsed judgment stays missing. It is never scored 1, never scored 0 and
  never imputed, and the parsed count is reported next to every mean.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/work/uf4_20260910")
NEED_REF_CATS = ("math", "reasoning", "coding")      # MT-Bench's own declaration
SEED_NAMESPACE = "20260912-uf4-mtbench:"
SYSTEM = "You are a helpful assistant."
SCORE = re.compile(r"\[\[(\d+\.?\d*)\]\]")
SCORE_BACKUP = re.compile(r"\[(\d+\.?\d*)\]")
PROTOCOL_FIELDS = ("tokenizer_sha256", "chat_template_sha256", "native_user_only",
                   "temperature", "top_p", "top_k", "repetition_penalty",
                   "max_new_tokens", "seed", "terminal_ids", "max_model_len")

SINGLE_V1 = """[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""

SINGLE_MATH_V1 = """[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[The Start of Reference Answer]
{ref_answer_1}
[The End of Reference Answer]

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""

SINGLE_V1_MULTI = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. You evaluation should focus on the assistant's answer to the second user question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

<|The Start of Assistant A's Conversation with User|>

### User:
{question_1}

### Assistant A:
{answer_1}

### User:
{question_2}

### Assistant A:
{answer_2}

<|The End of Assistant A's Conversation with User|>"""

SINGLE_MATH_V1_MULTI = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user questions. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. You evaluation should focus on the assistant's answer to the second question. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

<|The Start of Reference Answer|>

### User:
{question_1}

### Reference answer:
{ref_answer_1}

### User:
{question_2}

### Reference answer:
{ref_answer_2}

<|The End of Reference Answer|>


<|The Start of Assistant A's Conversation with User|>

### User:
{question_1}

### Assistant A:
{answer_1}

### User:
{question_2}

### Assistant A:
{answer_2}

<|The End of Assistant A's Conversation with User|>"""


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_label(responses_root, label):
    directory = Path(responses_root) / label
    target = directory / "mt_bench.jsonl"
    manifest = json.loads((directory / "mt_bench.manifest.json").read_text())
    settings = json.loads((directory / "mt_bench.settings.json").read_text())
    if manifest["response_file_sha256"] != file_hash(target):
        raise ValueError(f"{label}: response file does not match its manifest")
    rows = [json.loads(line) for line in open(target)]
    if len(rows) != manifest["n_generated"] or len(rows) != 160:
        raise ValueError(f"{label}: expected 160 responses, found {len(rows)}")
    by_key = {(r["question_id"], r["turn"]): r for r in rows}
    if len(by_key) != len(rows):
        raise ValueError(f"{label}: duplicate (question, turn)")
    return by_key, settings, manifest


def build_prompt(question, answers):
    """Return (template_name, prompt_text) for each turn of one question."""
    need_ref = question["category"] in NEED_REF_CATS and "reference" in question
    out = []
    if need_ref:
        out.append(("single-math-v1", SINGLE_MATH_V1.format(
            question=question["turns"][0], ref_answer_1=question["reference"][0],
            answer=answers[1])))
        out.append(("single-math-v1-multi-turn", SINGLE_MATH_V1_MULTI.format(
            question_1=question["turns"][0], question_2=question["turns"][1],
            ref_answer_1=question["reference"][0], ref_answer_2=question["reference"][1],
            answer_1=answers[1], answer_2=answers[2])))
    else:
        out.append(("single-v1", SINGLE_V1.format(
            question=question["turns"][0], answer=answers[1])))
        out.append(("single-v1-multi-turn", SINGLE_V1_MULTI.format(
            question_1=question["turns"][0], question_2=question["turns"][1],
            answer_1=answers[1], answer_2=answers[2])))
    return out


def parse_score(text):
    found = SCORE.findall(text)
    if len(found) == 1:
        return float(found[0]), "ok"
    if len(found) > 1:
        return None, "multiple_ratings"
    backup = SCORE_BACKUP.findall(text)
    if len(backup) == 1:
        return float(backup[0]), "ok_backup_pattern"
    return None, ("no_rating" if not backup else "multiple_ratings_backup")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--base-label", default="uf4mt_base")
    ap.add_argument("--responses-root", default="/work/nbpo_repair_20260909/responses")
    ap.add_argument("--questions", default="/work/nbpo_repair_20260909/data/mt_bench_question.jsonl")
    ap.add_argument("--judge", default=str(ROOT / "assets/Qwen3-14B"))
    ap.add_argument("--judge-revision", default="40c069824f4251a91eefaf281ebe4c544efd3e18")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "analysis/mtbench"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    questions = {q["question_id"]: q for q in
                 (json.loads(line) for line in open(args.questions))}
    labels = list(dict.fromkeys(args.labels))
    if args.base_label not in labels:
        labels.insert(0, args.base_label)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.judge, local_files_only=True)

    loaded, base_settings = {}, None
    for label in labels:
        by_key, settings, manifest = load_label(args.responses_root, label)
        if base_settings is None:
            base_settings = settings
        wrong = [k for k in PROTOCOL_FIELDS if settings.get(k) != base_settings.get(k)]
        if wrong:
            raise ValueError(f"{label}: generation protocol differs from "
                             f"{args.base_label}: {wrong}")
        loaded[label] = (by_key, settings, manifest)

    requests, params, meta = [], [], []
    budget = args.max_model_len - args.max_tokens
    longest = 0
    from vllm import SamplingParams
    for label in labels:
        by_key = loaded[label][0]
        for qid in sorted(questions):
            question = questions[qid]
            answers = {t: by_key[(qid, t)]["output"] for t in (1, 2)}
            for turn, (template_name, text) in enumerate(build_prompt(question, answers), 1):
                ids = tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": text}],
                    add_generation_prompt=True, tokenize=True, enable_thinking=False)
                longest = max(longest, len(ids))
                if len(ids) > budget:
                    raise ValueError(
                        f"judge input for {label} q{qid} t{turn} exceeds the budget "
                        f"({len(ids)} > {budget}); raise the window, never truncate")
                seed = int(digest(f"{SEED_NAMESPACE}{label}:{qid}:{turn}")[:16], 16) % (2 ** 63 - 1)
                requests.append({"prompt_token_ids": ids})
                params.append(SamplingParams(n=1, temperature=args.temperature,
                                             max_tokens=args.max_tokens, seed=seed))
                meta.append({"label": label, "question_id": qid, "turn": turn,
                             "category": question["category"], "template": template_name,
                             "used_reference": template_name.startswith("single-math"),
                             "seed": seed})
    if args.dry_run:
        print(json.dumps({"labels": labels, "judgments": len(requests),
                          "longest_judge_input_tokens": longest, "budget": budget,
                          "thinking_disabled": True,
                          "templates": sorted({m["template"] for m in meta}),
                          "reference_questions": len({m["question_id"] for m in meta
                                                      if m["used_reference"]})}, indent=2))
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    from vllm import LLM
    llm = LLM(model=args.judge, tokenizer=args.judge, tensor_parallel_size=1, dtype="bfloat16",
              max_model_len=args.max_model_len, gpu_memory_utilization=0.90, max_num_seqs=32,
              generation_config="vllm", seed=20260912, trust_remote_code=False)
    started = time.monotonic()
    generated = llm.generate(requests, params, use_tqdm=False)
    elapsed = time.monotonic() - started

    scores, status_counts = defaultdict(dict), defaultdict(int)
    dest = out_dir / "judgments.jsonl"
    with dest.open("w") as stream:
        for m, result in zip(meta, generated):
            text = result.outputs[0].text
            value, status = parse_score(text)
            status_counts[status] += 1
            if value is not None and not (1.0 <= value <= 10.0):
                value, status = None, "out_of_range"
                status_counts[status] += 1
            if value is not None:
                scores[m["label"]][(m["question_id"], m["turn"])] = value
            stream.write(json.dumps({**m, "score": value, "status": status,
                                     "finish_reason": result.outputs[0].finish_reason,
                                     "raw": text[-400:]}, ensure_ascii=False) + "\n")

    rng = np.random.default_rng(20260912)
    qids = sorted(questions)
    summary = {}
    for label in labels:
        got = scores[label]
        turn_means = {}
        for turn in (1, 2):
            vals = [got[(q, turn)] for q in qids if (q, turn) in got]
            turn_means[turn] = float(np.mean(vals)) if vals else None
        flat = [got[(q, t)] for q in qids for t in (1, 2) if (q, t) in got]
        # Resample QUESTIONS, keeping both turns of a question together: the two
        # turns of one question are not independent observations.
        boot = []
        per_q = {q: [got[(q, t)] for t in (1, 2) if (q, t) in got] for q in qids}
        usable = [q for q in qids if per_q[q]]
        for _ in range(args.bootstrap if usable else 0):
            pick = rng.choice(usable, size=len(usable), replace=True)
            boot.append(float(np.mean([v for q in pick for v in per_q[q]])))
        by_cat = defaultdict(list)
        for q in qids:
            by_cat[questions[q]["category"]].extend(per_q[q])
        summary[label] = {
            "score": float(np.mean(flat)) if flat else None,
            "ci95": ([float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
                     if boot else None),
            "turn1": turn_means[1], "turn2": turn_means[2],
            "mean_of_turn_means": (float(np.mean([turn_means[1], turn_means[2]]))
                                   if None not in turn_means.values() else None),
            "n_graded": len(flat), "n_planned": 2 * len(qids),
            "n_questions_with_any": len(usable),
            "by_category": {c: {"score": float(np.mean(v)), "n": len(v)}
                            for c, v in sorted(by_cat.items()) if v},
            "generation_settings_sha256": digest(json.dumps(
                {k: loaded[label][1][k] for k in PROTOCOL_FIELDS}, sort_keys=True))}

    report = {
        "benchmark": "MT-Bench (80 questions, two turns)",
        "grading": "official MT-Bench single-answer grading, 1-10, four official templates",
        "judge": args.judge, "judge_revision": args.judge_revision,
        "judge_thinking_mode": "disabled (enable_thinking=False)",
        "judge_system_message": SYSTEM,
        "decoding": {"temperature": args.temperature, "max_tokens": args.max_tokens,
                     "max_model_len": args.max_model_len},
        "reference_answer_rule": f"used exactly for MT-Bench's declared categories {NEED_REF_CATS}",
        "missing_rule": "an unparsed rating stays missing; it is never imputed",
        "not_official": ("Not an official MT-Bench score: the GPT-4 judge is replaced by the "
                         "local Qwen3-14B grader, so values are comparable across the arms in "
                         "this table and not to published MT-Bench numbers."),
        "n_judgments": len(requests), "seconds": elapsed,
        "longest_judge_input_tokens": longest,
        "status_counts": dict(status_counts),
        "judgments_sha256": file_hash(dest),
        "labels": summary}
    (out_dir / "mtbench_scores.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "labels"}, indent=2), flush=True)
    for label, value in summary.items():
        print(json.dumps({"label": label, "score": value["score"], "ci95": value["ci95"],
                          "turn1": value["turn1"], "turn2": value["turn2"],
                          "n_graded": value["n_graded"]}), flush=True)


if __name__ == "__main__":
    main()
