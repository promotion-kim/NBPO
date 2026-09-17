# Generated from mtbench_judge.py by make_mtbatch.py -- do not edit by hand.
# source sha256 1007bf7e9f74f67431548a78f18e06563af549feb32d9f756967eb54d8033271
# change: --arm becomes --arms and one vLLM engine is shared across them,
#         because five separate 72B loads is twenty-five minutes of loading.
#         Templates, template choice, two-turn conditioning, rating parsing
#         and decoding are untouched.
"""MT-Bench single-answer grading with the native templates and a local judge.

Template choice follows the released judging code: a question that ships a
reference answer is graded with the math template, everything else with the
general one, and the second turn uses the multi-turn variant of whichever
applies. The four templates are the verbatim repository text saved next to this
script, not a paraphrase, and their file hash is recorded in the report.

The judge is the contract's independent evaluator, run locally: greedy, 1024
verdict tokens, 32768 context. Scores are the 1-10 rating parsed from the
required "[[rating]]" form; anything else stays missing and is counted rather
than retried. The reported score is the mean over questions and turns, with the
per-turn and per-category means kept so a single number never hides a category.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MT = Path("/work/sub_20260914/mtbench")
RATING = re.compile(r"\[\[(\d+(?:\.\d+)?)\]\]")


def file_hash(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def judge_one(args, llm, tok):

    from vllm import SamplingParams

    templates = {}
    tpath = MT / "judge_prompts.jsonl"
    for line in tpath.open():
        if line.strip():
            row = json.loads(line)
            templates[row["name"]] = row
    for name in ("single-v1", "single-math-v1", "single-v1-multi-turn",
                 "single-math-v1-multi-turn"):
        if name not in templates:
            raise SystemExit("missing native template %s" % name)

    answers = [json.loads(l) for l in
               (MT / ("responses_%s" % args.arm) / "answers.jsonl").open() if l.strip()]
    if len(answers) != 80:
        raise SystemExit("expected 80 graded questions, found %d" % len(answers))

    sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, top_k=-1,
                        max_tokens=args.max_tokens)

    requests, meta = [], []
    for row in answers:
        has_ref = bool(row.get("reference"))
        for turn in (0, 1):
            if turn == 0:
                name = "single-math-v1" if has_ref else "single-v1"
                fields = {"question": row["turns"][0], "answer": row["answers"][0]}
                if has_ref:
                    fields["ref_answer_1"] = row["reference"][0]
            else:
                name = ("single-math-v1-multi-turn" if has_ref
                        else "single-v1-multi-turn")
                fields = {"question_1": row["turns"][0], "answer_1": row["answers"][0],
                          "question_2": row["turns"][1], "answer_2": row["answers"][1]}
                if has_ref:
                    fields["ref_answer_1"] = row["reference"][0]
                    fields["ref_answer_2"] = row["reference"][1]
            tpl = templates[name]
            user = tpl["prompt_template"].format(**fields)
            chat = ([{"role": "system", "content": tpl["system_prompt"]}]
                    if tpl["system_prompt"].strip() else [])
            chat = chat + [{"role": "user", "content": user}]
            ids = tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
            if len(ids) + args.max_tokens > args.max_model_len:
                meta.append({"question_id": row["question_id"], "turn": turn + 1,
                             "category": row["category"], "template": name,
                             "status": "over_budget"})
                continue
            requests.append({"prompt_token_ids": ids})
            meta.append({"question_id": row["question_id"], "turn": turn + 1,
                         "category": row["category"], "template": name,
                         "status": "queued"})

    queued = [m for m in meta if m["status"] == "queued"]
    started = time.monotonic()
    generated = llm.generate(requests, sp, use_tqdm=False)
    status = Counter()
    records = []
    for m, g in zip(queued, generated):
        text = g.outputs[0].text
        found = RATING.findall(text)
        if len(found) != 1:
            score, state = None, ("no_rating" if not found else "multiple_ratings")
        else:
            value = float(found[0])
            if 1.0 <= value <= 10.0:
                score, state = value, "ok"
            else:
                score, state = None, "out_of_range"
        status[state] += 1
        records.append({**m, "status": state, "score": score})
    for m in meta:
        if m["status"] == "over_budget":
            status["over_budget"] += 1
            records.append({**m, "score": None})

    out = MT / ("judged_%s" % args.arm)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "ratings.jsonl").open("w") as stream:
        for r in records:
            stream.write(json.dumps(r) + "\n")

    ok = [r for r in records if r["status"] == "ok"]
    by_turn = defaultdict(list)
    by_cat = defaultdict(list)
    for r in ok:
        by_turn[r["turn"]].append(r["score"])
        by_cat[r["category"]].append(r["score"])
    report = {
        "arm": args.arm, "judge": args.judge, "judge_revision": args.judge_revision,
        "templates_file": str(tpath), "templates_sha256": file_hash(tpath),
        "templates_are_verbatim_repository_text": True,
        "not_the_official_metric": ("MT-Bench is normally judged by GPT-4; this run uses "
                                    "the contract's local independent evaluator, so the "
                                    "score is comparable across these rows and is not an "
                                    "official MT-Bench number"),
        "questions": len(answers), "judgments_planned": len(meta),
        "status_counts": dict(status),
        "valid_fraction": round(len(ok) / max(len(meta), 1), 6),
        "score_mean": round(float(np.mean([r["score"] for r in ok])), 4) if ok else None,
        "score_by_turn": {str(k): round(float(np.mean(v)), 4) for k, v in sorted(by_turn.items())},
        "score_by_category": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_cat.items())},
        "decoding": {"temperature": 0.0, "max_verdict_tokens": args.max_tokens,
                     "max_model_len": args.max_model_len,
                     "tensor_parallel_size": args.tensor_parallel_size},
        "seconds": round(time.monotonic() - started, 1),
        "source_sha256": file_hash(__file__)}
    (out / "complete.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("arm", "score_mean", "score_by_turn", "valid_fraction",
                       "status_counts")}))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--judge", default="/work/models/bases/Qwen2.5-72B-Instruct")
    ap.add_argument("--judge-revision", default="495f39366efef23836d0cfae4fbe635880d2be31")
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(args.judge, local_files_only=True)
    llm = LLM(model=args.judge, tokenizer=args.judge,
              tensor_parallel_size=args.tensor_parallel_size, dtype="bfloat16",
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_num_seqs=32, generation_config="vllm", seed=20260917,
              trust_remote_code=False)
    done, failed = [], []
    for arm in args.arms:
        one = copy.copy(args)
        one.arm = arm
        target = MT / ("judged_%s" % arm) / "complete.json"
        if target.exists():
            print(json.dumps({"skipped": arm}), flush=True)
            done.append(arm)
            continue
        try:
            judge_one(one, llm, tok)
            done.append(arm)
        except Exception as exc:
            failed.append({"arm": arm, "error": str(exc)[:300]})
            print(json.dumps({"failed": arm, "error": str(exc)[:300]}), flush=True)
    print(json.dumps({"judged": done, "failed": failed}), flush=True)
    if failed:
        raise SystemExit("%d of %d arms failed" % (len(failed), len(args.arms)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
