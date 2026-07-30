#!/usr/bin/env python3
"""One-GPU shard of position-swapped Qwen3-32B pairwise judgments."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
from pathlib import Path

from vllm import LLM, SamplingParams


SYSTEM = """You are a careful, impartial evaluator of assistant responses. Compare the two answers only on task fulfillment, factual and logical correctness, relevance, clarity, safety, and (for multi-turn tasks) consistency with the conversation. Do not prefer an answer merely because it is longer, more detailed, or appears first. Return exactly one JSON object with keys winner and reason. winner must be A, B, or tie. Keep reason under 35 words."""


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt_text(source: dict, answer_a: dict, answer_b: dict) -> str:
    turns = []
    for index, question in enumerate(source["turns"]):
        turns.append(f"USER TURN {index + 1}:\n{question}")
        turns.append(f"ANSWER A TURN {index + 1}:\n{answer_a['responses'][index]}")
        turns.append(f"ANSWER B TURN {index + 1}:\n{answer_b['responses'][index]}")
    return "\n\n".join(turns) + "\n\nReturn JSON now."


def chat_prompt(tokenizer, user_text: str) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    signature = inspect.signature(tokenizer.apply_chat_template)
    if "enable_thinking" in signature.parameters or any(
        value.kind == inspect.Parameter.VAR_KEYWORD for value in signature.parameters.values()
    ):
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text}],
        **kwargs,
    )


def parse_judgment(text: str) -> tuple[str | None, str | None]:
    candidates = re.findall(r"\{[^{}]*\}", text, flags=re.S)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        winner = str(value.get("winner", "")).strip().lower()
        if winner in {"a", "b", "tie"}:
            return winner.upper() if winner != "tie" else "tie", str(value.get("reason", "")).strip()
    match = re.search(r'(?i)"?winner"?\s*[:=]\s*"?(A|B|tie)\b', text)
    if match:
        winner = match.group(1)
        return winner.upper() if winner.lower() != "tie" else "tie", None
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses-root", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()
    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    candidate_name = protocol["comparisons"]["candidate"]
    opponents = protocol["comparisons"]["opponents"]
    candidate = {row["item_id"]: row for row in load_jsonl(args.responses_root / candidate_name / "responses.jsonl")}
    opponent_rows = {
        name: {row["item_id"]: row for row in load_jsonl(args.responses_root / name / "responses.jsonl")}
        for name in opponents
    }

    tasks = []
    for opponent in opponents:
        if set(candidate) != set(opponent_rows[opponent]):
            raise RuntimeError(f"item mismatch for {opponent}")
        for item_id in sorted(candidate):
            digest = int(hashlib.sha256(f"{opponent}|{item_id}".encode()).hexdigest()[:16], 16)
            if digest % args.num_shards != args.shard_index:
                continue
            source = candidate[item_id]
            other = opponent_rows[opponent][item_id]
            tasks.append({
                "benchmark": source["benchmark"], "item_id": item_id,
                "opponent": opponent, "order": "ronpo_A",
                "user_prompt": prompt_text(source, source, other),
            })
            tasks.append({
                "benchmark": source["benchmark"], "item_id": item_id,
                "opponent": opponent, "order": "ronpo_B",
                "user_prompt": prompt_text(source, other, source),
            })

    judge = protocol["judge"]
    llm = LLM(
        model=judge["model"], revision=judge["revision"], dtype="bfloat16",
        max_model_len=judge["max_model_len"], gpu_memory_utilization=args.gpu_memory_utilization,
        seed=judge["seed"], trust_remote_code=False, enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    formatted = [chat_prompt(tokenizer, task["user_prompt"]) for task in tasks]
    params = SamplingParams(
        n=1, temperature=judge["temperature"], top_p=judge["top_p"],
        max_tokens=judge["max_new_tokens"], seed=judge["seed"],
        bad_words=["<think>", "</think>"],
    )
    generated = llm.generate(formatted, params, use_tqdm=True)
    if len(generated) != len(tasks):
        raise RuntimeError("judge generation count mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for task, result in zip(tasks, generated):
            raw = result.outputs[0].text
            winner, reason = parse_judgment(raw)
            row = {key: value for key, value in task.items() if key != "user_prompt"}
            row.update({
                "winner": winner, "reason": reason, "raw_judge_output": raw,
                "judge_model": judge["model"], "judge_revision": judge["revision"],
                "valid": winner is not None,
            })
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    valid = sum(1 for row in load_jsonl(args.output) if row["valid"])
    metadata = {
        "shard_index": args.shard_index, "num_shards": args.num_shards,
        "judgments": len(tasks), "valid": valid, "invalid": len(tasks) - valid,
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
