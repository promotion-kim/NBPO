"""Two-turn MT-Bench generation under the campaign's greedy decoding protocol.

MT-Bench cannot be produced by the shared single-message benchmark generator:
its second turn is a follow-up that conditions on the model's *own* first
answer, so the two turns must be generated sequentially and the first answer
must be fed back as an assistant message. Everything else is the protocol the
other benchmarks already use -- same tokenizer and chat template, temperature 0,
top_p 1, 2048 new tokens, fixed seed -- and the eleven decoding/tokenizer fields
are checked against an existing reference run before a single token is emitted,
so a turn-1 answer can never be generated under settings that differ from the
arm it is compared with.

`native_user_only` stays true: no synthetic system prompt is injected. The
second turn is a native (user, assistant, user) conversation, which is what
MT-Bench is, and that is recorded separately as `multi_turn_chat`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

PROTOCOL_FIELDS = ("tokenizer_sha256", "chat_template_sha256", "native_user_only",
                   "temperature", "top_p", "top_k", "repetition_penalty",
                   "max_new_tokens", "seed", "terminal_ids", "max_model_len")


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def object_hash(obj):
    return digest(json.dumps(obj, sort_keys=True, ensure_ascii=False))


def write_json(path, payload):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_jsonl(path, rows):
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_questions(path):
    rows = [json.loads(line) for line in open(path)]
    if len(rows) != 80 or any(len(r["turns"]) != 2 for r in rows):
        raise ValueError("MT-Bench expects 80 questions of exactly two turns")
    if len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate question_id")
    return sorted(rows, key=lambda r: r["question_id"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("/work/nbpo_repair_20260909"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tokenizer", default="/work/models/bases/Llama-3.1-8B-Instruct")
    ap.add_argument("--questions", default=None)
    ap.add_argument("--protocol-reference",
                    default="/work/nbpo_repair_20260909/responses/base/settings.json")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--dry-run", action="store_true",
                    help="build every turn-1 prompt and check the context budget without a GPU")
    args = ap.parse_args()

    questions = load_questions(args.questions or (args.root / "data/mt_bench_question.jsonl"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    end = json.loads((Path(args.tokenizer) / "generation_config.json").read_text())["eos_token_id"]
    end = end if isinstance(end, list) else [end]

    settings = {
        "label": args.label, "model": args.model, "tokenizer": args.tokenizer,
        "tokenizer_sha256": file_hash(Path(args.tokenizer) / "tokenizer.json"),
        "chat_template_sha256": digest(tok.chat_template), "native_user_only": True,
        "temperature": 0.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0,
        "max_new_tokens": args.max_new_tokens, "seed": 20260909, "terminal_ids": end,
        "max_model_len": args.max_model_len,
        "benchmark": "mt_bench", "multi_turn_chat": True, "n_questions": len(questions),
        "questions_sha256": file_hash(args.questions or (args.root / "data/mt_bench_question.jsonl")),
        "source_sha256": file_hash(__file__)}

    reference = json.loads(Path(args.protocol_reference).read_text())
    wrong = [k for k in PROTOCOL_FIELDS if settings.get(k) != reference.get(k)]
    if wrong:
        raise ValueError(f"decoding protocol differs from {args.protocol_reference}: {wrong}")

    first_prompts = [tok.apply_chat_template([{"role": "user", "content": q["turns"][0]}],
                                             tokenize=True, add_generation_prompt=True)
                     for q in questions]
    budget = args.max_model_len - args.max_new_tokens
    over = [q["question_id"] for q, ids in zip(questions, first_prompts) if len(ids) > budget]
    if over:
        raise ValueError(f"turn-1 prompts exceed the shared context budget: {over}")
    # The worst case for turn 2 is a first answer that ran to the cap.
    worst = max(len(ids) for ids in first_prompts) + args.max_new_tokens + \
        max(len(tok(q["turns"][1])["input_ids"]) for q in questions) + 16
    if args.dry_run:
        print(json.dumps({"label": args.label, "questions": len(questions),
                          "protocol_reference_ok": True,
                          "turn1_prompt_tokens_max": max(len(i) for i in first_prompts),
                          "turn2_worst_case_prompt_tokens": worst,
                          "context_budget": budget,
                          "turn2_fits": worst <= budget,
                          "model_exists": Path(args.model).exists()}, indent=2))
        return
    if worst > budget:
        raise ValueError(f"a capped turn-1 answer would overflow turn 2 ({worst} > {budget})")

    model_files = {p.name: file_hash(p) for p in sorted(Path(args.model).glob("*.safetensors"))}
    if not model_files:
        raise ValueError("Evaluation requires a concrete exported full-policy checkpoint")
    settings["model_weights"] = model_files
    out = args.root / "responses" / args.label
    out.mkdir(parents=True, exist_ok=True)
    target, summary_path = out / "mt_bench.jsonl", out / "mt_bench.manifest.json"
    settings_path = out / "mt_bench.settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Existing MT-Bench run under this label has different settings")
    if target.exists():
        if not summary_path.exists() or \
                json.loads(summary_path.read_text())["response_file_sha256"] != file_hash(target):
            raise ValueError("Existing MT-Bench responses incomplete/corrupted")
        print(json.dumps({"skipped": "already generated", "path": str(target)}))
        return
    write_json(settings_path, settings)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tokenizer=args.tokenizer, dtype="bfloat16",
              max_model_len=args.max_model_len, gpu_memory_utilization=0.85,
              generation_config="vllm", seed=20260909, tensor_parallel_size=1)
    sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, repetition_penalty=1.0,
                        max_tokens=args.max_new_tokens, stop_token_ids=end, seed=20260909, n=1)

    started = time.monotonic()
    rows, answers = [], {}
    for turn in (1, 2):
        prompts, prompt_texts = [], []
        for q, first_ids in zip(questions, first_prompts):
            if turn == 1:
                prompts.append(first_ids)
                prompt_texts.append(q["turns"][0])
            else:
                chat = [{"role": "user", "content": q["turns"][0]},
                        {"role": "assistant", "content": answers[q["question_id"]]},
                        {"role": "user", "content": q["turns"][1]}]
                ids = tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
                if len(ids) > budget:
                    raise ValueError(f"turn-2 prompt overflow on q{q['question_id']}")
                prompts.append(ids)
                prompt_texts.append(q["turns"][1])
        outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompts], sp, use_tqdm=False)
        if len(outputs) != len(questions) or any(len(o.outputs) != 1 for o in outputs):
            raise ValueError("Generation coverage mismatch")
        for q, ids, text, output in zip(questions, prompts, prompt_texts, outputs):
            g = output.outputs[0]
            token_ids = list(g.token_ids)
            if not token_ids or g.finish_reason not in ("stop", "length"):
                raise ValueError("Invalid generated event")
            if g.finish_reason == "stop" and token_ids[-1] not in end:
                raise ValueError("Sampled terminal token absent from event")
            if turn == 1:
                answers[q["question_id"]] = g.text
            rows.append({"uid": f"mtbench:{q['question_id']}:t{turn}",
                         "question_id": q["question_id"], "turn": turn,
                         "category": q["category"], "prompt": text, "output": g.text,
                         "prompt_sha256": digest(text), "prompt_token_ids": ids,
                         "output_token_ids": token_ids, "n_output_tokens": len(token_ids),
                         "finish_reason": g.finish_reason, "stop_reason": g.stop_reason,
                         "has_reference": "reference" in q,
                         "model_sha256": object_hash(model_files)})
    write_jsonl(target, rows)
    write_json(summary_path, {
        "n_planned": 2 * len(questions), "n_generated": len(rows), "n_failed": 0,
        "seconds": time.monotonic() - started, "response_file_sha256": file_hash(target),
        "settings_sha256": file_hash(settings_path),
        "question_sha256": object_hash([(q["question_id"], q["turns"]) for q in questions]),
        "tokens": sum(r["n_output_tokens"] for r in rows),
        "capped": sum(r["finish_reason"] == "length" for r in rows),
        "capped_turn1": sum(r["finish_reason"] == "length" and r["turn"] == 1 for r in rows),
        "empty_text": sum(not r["output"].strip() for r in rows)})
    print(f"{args.label}/mt_bench: {len(rows)} responses "
          f"in {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
