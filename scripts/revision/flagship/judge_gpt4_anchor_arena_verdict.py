#!/usr/bin/env python3
"""Run a resumable full Arena-Hard verdict-only open-weight judge shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from vllm import LLM, SamplingParams

from judge_gpt4_anchor_gptoss import load_jsonl, make_tasks


SYSTEM = """Act as an impartial judge comparing Assistant A and Assistant B on the user prompt. Evaluate correctness, task fulfillment, helpfulness, relevance, clarity, conciseness, and creativity when needed. Do not generate your own alternative answer. Return exactly one final label and nothing else: [[A>>B]], [[A>B]], [[A=B]], [[B>A]], or [[B>>A]]."""


def parse(raw: str) -> tuple[str | None, bool]:
    matches = re.findall(r"\[\[(A>>B|A>B|A=B|B>A|B>>A)\]\]", raw)
    return (matches[-1], True) if matches else (None, False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses-root", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--mt-judge-prompts", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--judge-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    parent = json.loads(args.parent_protocol.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if protocol["parent_protocol_sha256"] != parent["configuration_sha256"]:
        raise RuntimeError("Arena protocol parent hash mismatch")
    tasks = [task for task in make_tasks(args, parent) if task["benchmark"] == "arena_hard_v0.1"]
    for task in tasks:
        task["system"] = SYSTEM
        task["user"] += (
            "\n\nReturn exactly one of these strings now: [[A>>B]], [[A>B]], [[A=B]], [[B>A]], or [[B>>A]]. "
            "Do not return bare A or B."
        )
    completed = {row["task_id"]: row for row in load_jsonl(args.output)} if args.output.exists() else {}
    pending = [task for task in tasks if task["task_id"] not in completed]
    print(json.dumps({"shard": args.shard_index, "total": len(tasks), "completed": len(completed), "pending": len(pending)}), flush=True)
    if not pending:
        return
    judge = protocol["judge"]
    llm = LLM(
        model=str(args.judge_model_path), dtype=judge["dtype"], max_model_len=judge["max_model_len"],
        tensor_parallel_size=1, gpu_memory_utilization=judge["gpu_memory_utilization"],
        seed=judge["seed"], trust_remote_code=False, enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(
        n=1, temperature=judge["temperature"], top_p=judge["top_p"],
        max_tokens=judge["max_new_tokens"], seed=judge["seed"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a" if args.output.exists() else "w", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            prompts = [tokenizer.apply_chat_template(
                [{"role": "system", "content": task["system"]}, {"role": "user", "content": task["user"]}],
                tokenize=False, add_generation_prompt=True, reasoning_effort=judge["reasoning_effort"],
            ) for task in batch]
            outputs = llm.generate(prompts, params, use_tqdm=True)
            for task, output in zip(batch, outputs):
                raw = output.outputs[0].text
                parsed, valid = parse(raw)
                row = {key: value for key, value in task.items() if key not in {"system", "user"}}
                row.update({
                    "parsed": parsed, "valid": valid, "raw_judge_output": raw,
                    "judge_model": judge["model"], "judge_revision": judge["revision"],
                    "protocol_sha256": protocol["configuration_sha256"],
                })
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(json.dumps({"shard": args.shard_index, "done": min(start + len(batch), len(pending))}), flush=True)
    rows = load_jsonl(args.output)
    metadata = {
        "shard_index": args.shard_index, "judgments": len(rows),
        "valid": sum(bool(row["valid"]) for row in rows),
        "invalid": sum(not bool(row["valid"]) for row in rows),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    import wandb
    run = wandb.init(
        entity="promotion-kim", project="mnpo",
        id=f"gpt4anchor-gptoss120b-arena-v3-s{args.shard_index}-20260714", resume="allow",
        name=f"gpt4anchor-gptoss120b-arena-v3-s{args.shard_index}",
        config={"protocol": protocol, "shard": args.shard_index, "num_shards": args.num_shards},
    )
    wandb.log(metadata); run.summary.update(metadata); run.finish()
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
