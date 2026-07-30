#!/usr/bin/env python3
"""Run one resumable B200 shard of the frozen GPT-4-anchor gpt-oss judge proxy."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from vllm import LLM, SamplingParams


ALPACA_SYSTEM = """You are a highly efficient assistant, who evaluates and selects the best large language model (LLMs) based on the quality of their responses to a given instruction. This process will be used to create a leaderboard reflecting the most accurate and human-preferred answers."""
ALPACA_TEMPLATE = """
I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{{
    "instruction": "{instruction}"
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "m",
        "output": "{output_1}"
    }},
    {{
        "model_identifier": "M",
        "output": "{output_2}"
    }}
}}

## Task

Evaluate the models based on the quality and relevance of their outputs, and select the model that generated the best output. Answer by providing the model identifier of the best model. We will use your output as the name of the best model, so make sure your output only contains one of the following model identifiers and nothing else (no quotes, no spaces, no new lines, ...): m or M.

## Best Model Identifier
""".strip()

ARENA_SYSTEM = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]"."""
ARENA_TEMPLATE = """<|User Prompt|>
{question}

<|The Start of Assistant A's Answer|>
{answer_1}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{answer_2}
<|The End of Assistant B's Answer|>""".strip()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_shard(task_id: str, count: int) -> int:
    return int(hashlib.sha256(task_id.encode()).hexdigest()[:16], 16) % count


def make_tasks(args: argparse.Namespace, protocol: dict) -> list[dict]:
    refs = {
        "alpaca_eval_2": {row["item_id"]: row for row in load_jsonl(args.reference_dir / "alpaca_eval_2_gpt4_reference.jsonl")},
        "arena_hard_v0.1": {row["item_id"]: row for row in load_jsonl(args.reference_dir / "arena_hard_v01_gpt4_reference.jsonl")},
        "mt_bench": {row["item_id"]: row for row in load_jsonl(args.reference_dir / "mt_bench_gpt4_reference.jsonl")},
    }
    prompt_rows = {row["name"]: row for row in load_jsonl(args.mt_judge_prompts)}
    needed = {"single-v1", "single-math-v1", "single-v1-multi-turn", "single-math-v1-multi-turn"}
    if not needed.issubset(prompt_rows):
        raise RuntimeError("missing official FastChat MT-Bench judge prompts")
    tasks: list[dict] = []
    for model in protocol["models"]:
        responses = {row["item_id"]: row for row in load_jsonl(args.responses_root / model / "responses.jsonl")}
        for item_id, reference in refs["alpaca_eval_2"].items():
            answer = responses[item_id]["responses"][0]
            for order, output_1, output_2, candidate_label in (
                ("reference_first", reference["reference_answer"], answer, "M"),
                ("candidate_first", answer, reference["reference_answer"], "m"),
            ):
                task_id = f"alpaca_eval_2|{model}|{item_id}|{order}"
                if stable_shard(task_id, args.num_shards) == args.shard_index:
                    tasks.append({
                        "task_id": task_id, "benchmark": "alpaca_eval_2", "model": model,
                        "item_id": item_id, "order": order, "candidate_label": candidate_label,
                        "system": ALPACA_SYSTEM,
                        "user": ALPACA_TEMPLATE.format(
                            instruction=reference["instruction"], output_1=output_1, output_2=output_2
                        ),
                    })
        for item_id, reference in refs["arena_hard_v0.1"].items():
            answer = responses[item_id]["responses"][0]
            for order, answer_1, answer_2, candidate_label in (
                ("reference_first", reference["reference_answer"], answer, "B"),
                ("candidate_first", answer, reference["reference_answer"], "A"),
            ):
                task_id = f"arena_hard_v0.1|{model}|{item_id}|{order}"
                if stable_shard(task_id, args.num_shards) == args.shard_index:
                    tasks.append({
                        "task_id": task_id, "benchmark": "arena_hard_v0.1", "model": model,
                        "item_id": item_id, "order": order, "candidate_label": candidate_label,
                        "system": ARENA_SYSTEM,
                        "user": ARENA_TEMPLATE.format(
                            question=reference["question"], answer_1=answer_1, answer_2=answer_2
                        ),
                    })
        for item_id, reference in refs["mt_bench"].items():
            answers = responses[item_id]["responses"]
            has_reference = reference["reference_answers"] is not None
            for turn_index in (0, 1):
                if turn_index == 0:
                    key = "single-math-v1" if has_reference else "single-v1"
                    values = {"question": reference["turns"][0], "answer": answers[0]}
                    if has_reference:
                        values["ref_answer_1"] = reference["reference_answers"][0]
                else:
                    key = "single-math-v1-multi-turn" if has_reference else "single-v1-multi-turn"
                    values = {
                        "question_1": reference["turns"][0], "question_2": reference["turns"][1],
                        "answer_1": answers[0], "answer_2": answers[1],
                    }
                    if has_reference:
                        values["ref_answer_1"] = reference["reference_answers"][0]
                        values["ref_answer_2"] = reference["reference_answers"][1]
                prompt = prompt_rows[key]
                task_id = f"mt_bench|{model}|{item_id}|turn{turn_index + 1}"
                if stable_shard(task_id, args.num_shards) == args.shard_index:
                    tasks.append({
                        "task_id": task_id, "benchmark": "mt_bench", "model": model,
                        "item_id": item_id, "turn": turn_index + 1, "category": reference["category"],
                        "has_gpt4_reference": has_reference, "prompt_name": key,
                        "system": prompt["system_prompt"], "user": prompt["prompt_template"].format(**values),
                    })
    return sorted(tasks, key=lambda row: row["task_id"])


def parse_output(task: dict, raw: str) -> tuple[object | None, bool]:
    if task["benchmark"] == "alpaca_eval_2":
        matches = re.findall(r"(?m)^\s*([mM])\s*$", raw)
        harmony = re.findall(r"assistantfinal\s*([mM])\s*$", raw)
        value = harmony[-1] if harmony else (matches[-1] if matches else (raw.strip() if raw.strip() in {"m", "M"} else None))
        return value, value in {"m", "M"}
    if task["benchmark"] == "arena_hard_v0.1":
        matches = re.findall(r"\[\[(A>>B|A>B|A=B|B>A|B>>A)\]\]", raw)
        return (matches[-1], True) if matches else (None, False)
    matches = re.findall(r"\[\[([0-9]+(?:\.[0-9]+)?)\]\]", raw)
    if not matches:
        return None, False
    rating = float(matches[-1])
    return rating, 1.0 <= rating <= 10.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses-root", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--mt-judge-prompts", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--judge-model-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_JUDGING":
        raise RuntimeError("invalid protocol lock")
    for benchmark, filename in (
        ("alpaca_eval_2", "alpaca_eval_2_gpt4_reference.jsonl"),
        ("arena_hard_v0.1", "arena_hard_v01_gpt4_reference.jsonl"),
        ("mt_bench", "mt_bench_gpt4_reference.jsonl"),
    ):
        expected = protocol["references"][benchmark]["normalized_sha256"]
        if file_sha256(args.reference_dir / filename) != expected:
            raise RuntimeError(f"reference hash mismatch: {benchmark}")
    tasks = make_tasks(args, protocol)
    completed: dict[str, dict] = {}
    if args.output.exists():
        completed = {row["task_id"]: row for row in load_jsonl(args.output)}
    pending = [task for task in tasks if task["task_id"] not in completed]
    print(json.dumps({"shard": args.shard_index, "total": len(tasks), "completed": len(completed), "pending": len(pending)}), flush=True)
    if not pending:
        return

    judge = protocol["judge"]
    engine_model = str(args.judge_model_path) if args.judge_model_path else judge["model"]
    engine_revision = None if args.judge_model_path else judge["revision"]
    llm = LLM(
        model=engine_model, revision=engine_revision, dtype=judge["dtype"],
        max_model_len=judge["max_model_len"], tensor_parallel_size=judge["tensor_parallel_size"],
        gpu_memory_utilization=judge["gpu_memory_utilization"], seed=judge["seed"],
        trust_remote_code=judge["trust_remote_code"], enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(
        n=1, temperature=judge["temperature"], top_p=judge["top_p"],
        max_tokens=judge["max_new_tokens"], seed=judge["seed"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            prompts = [tokenizer.apply_chat_template(
                [{"role": "system", "content": task["system"]}, {"role": "user", "content": task["user"]}],
                tokenize=False, add_generation_prompt=True, reasoning_effort=judge["reasoning_effort"],
            ) for task in batch]
            outputs = llm.generate(prompts, params, use_tqdm=True)
            if len(outputs) != len(batch):
                raise RuntimeError("judge output count mismatch")
            for task, output in zip(batch, outputs):
                raw = output.outputs[0].text
                parsed, valid = parse_output(task, raw)
                row = {key: value for key, value in task.items() if key not in {"system", "user"}}
                row.update({
                    "parsed": parsed, "valid": valid, "raw_judge_output": raw,
                    "judge_model": judge["model"], "judge_revision": judge["revision"],
                    "protocol_sha256": protocol["configuration_sha256"],
                })
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(json.dumps({"shard": args.shard_index, "done": min(start + len(batch), len(pending)), "pending_at_start": len(pending)}), flush=True)
    rows = load_jsonl(args.output)
    metadata = {
        "shard_index": args.shard_index, "num_shards": args.num_shards,
        "judgments": len(rows), "valid": sum(bool(row["valid"]) for row in rows),
        "invalid": sum(not bool(row["valid"]) for row in rows),
        "output_sha256": file_sha256(args.output),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        import wandb
        run = wandb.init(
            entity="promotion-kim", project="mnpo",
            id=f"gpt4anchor-gptoss120b-s{args.shard_index}-20260714", resume="allow",
            name=f"gpt4anchor-gptoss120b-s{args.shard_index}",
            config={"protocol": protocol, "shard": args.shard_index, "num_shards": args.num_shards},
        )
        wandb.log(metadata)
        run.summary.update(metadata)
        run.finish()
    except Exception as exc:
        metadata["wandb_error"] = repr(exc)
        args.output.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
