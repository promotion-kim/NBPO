"""Two-turn MT-Bench generation for one policy.

MT-Bench is not a single-turn benchmark: the second answer is conditioned on the
model's own first answer, so the two turns are generated in sequence with the
first answer appended to the conversation. Decoding is greedy, which is the
declared candidate decoding for this table, and the terminal ids come from the
policy's own tokenizer rather than from a hardcoded family.

Nothing about the questions is altered: all 80 are used, both turns, in the
order the released file lists them, and the reference answers travel with the
record so the judge can pick its template without re-deciding categories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

QUESTIONS = Path("/work/nbpo_repair_20260909/data/mt_bench_question.jsonl")
OUT = Path("/work/sub_20260914/mtbench")


def file_hash(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tokenizer", default=None,
                    help="defaults to the model directory, never a foreign family")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = args.tokenizer or args.model
    tok = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
    questions = [json.loads(l) for l in QUESTIONS.open() if l.strip()]
    if len(questions) != 80:
        raise SystemExit("expected the 80 released questions, found %d" % len(questions))

    out = OUT / ("responses_%s" % args.arm)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "answers.jsonl"
    llm = LLM(model=args.model, tokenizer=tokenizer, dtype="bfloat16",
              max_model_len=args.max_model_len, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_memory_utilization,
              generation_config="vllm", seed=20260916, trust_remote_code=False)
    sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, top_k=-1,
                        repetition_penalty=1.0, max_tokens=args.max_tokens)

    started = time.monotonic()
    conversations = [[{"role": "user", "content": q["turns"][0]}] for q in questions]
    answers = [[] for _ in questions]
    for turn in (0, 1):
        if turn == 1:
            for i, q in enumerate(questions):
                conversations[i].append({"role": "assistant", "content": answers[i][0]})
                conversations[i].append({"role": "user", "content": q["turns"][1]})
        prompts = [{"prompt_token_ids": tok.apply_chat_template(
            c, tokenize=True, add_generation_prompt=True)} for c in conversations]
        over = [i for i, p in enumerate(prompts)
                if len(p["prompt_token_ids"]) + args.max_tokens > args.max_model_len]
        if over:
            raise SystemExit("turn %d exceeds the context for %d questions; raise "
                             "--max-model-len rather than truncating" % (turn + 1, len(over)))
        generated = llm.generate(prompts, sp, use_tqdm=False)
        for i, g in enumerate(generated):
            o = g.outputs[0]
            if o.finish_reason not in ("stop", "length"):
                raise SystemExit("unexpected finish reason %r" % o.finish_reason)
            answers[i].append(o.text)

    rows = []
    for q, ans in zip(questions, answers):
        rows.append({"question_id": q["question_id"], "category": q["category"],
                     "turns": q["turns"], "reference": q.get("reference"),
                     "answers": ans, "arm": args.arm})
    with dest.open("w") as stream:
        for r in rows:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {
        "arm": args.arm, "model": args.model, "tokenizer": tokenizer,
        "questions": len(rows), "turns_per_question": 2,
        "questions_sha256": file_hash(QUESTIONS),
        "decoding": {"temperature": 0.0, "top_p": 1.0, "top_k": -1,
                     "max_new_tokens": args.max_tokens,
                     "max_model_len": args.max_model_len},
        "second_turn": "conditioned on the model's own first answer",
        "with_reference": sum(1 for r in rows if r["reference"]),
        "capped_at_length": sum(1 for r in rows for a in r["answers"] if not a.strip()),
        "answers_sha256": file_hash(dest),
        "seconds": round(time.monotonic() - started, 1),
        "source_sha256": file_hash(__file__)}
    (out / "complete.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("arm", "questions", "with_reference", "seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
