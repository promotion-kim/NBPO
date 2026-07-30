#!/usr/bin/env python3
"""Run one resumable B200 shard of the frozen sealed all-pairwise judge."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from itertools import combinations
from pathlib import Path

from vllm import LLM, SamplingParams


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_shard(task_id: str, count: int) -> int:
    return int(hashlib.sha256(task_id.encode()).hexdigest()[:16], 16) % count


def parse(raw: str) -> tuple[str | None, bool]:
    matches = re.findall(r"\[\[(A>>B|A>B|A=B|B>A|B>>A)\]\]", raw)
    return (matches[-1], True) if matches else (None, False)


def make_tasks(protocol: dict, work: Path, shard_index: int, num_shards: int) -> list[dict]:
    generations = {
        model: json.loads((work / "generations" / model / "output_42.json").read_text())
        for model in protocol["models"]
    }
    tasks = []
    for prompt_index in range(protocol["sealed_prompt_count"]):
        prompt = str(generations[protocol["models"][0]][prompt_index]["prompt"])
        for left, right in combinations(protocol["models"], 2):
            task_id = f"sealed-pairwise|{prompt_index}|{left}|{right}|seed42"
            if stable_shard(task_id, num_shards) != shard_index:
                continue
            left_first = int(hashlib.sha256((task_id + "|order").encode()).hexdigest()[:16], 16) % 2 == 0
            assistant_a, assistant_b = (left, right) if left_first else (right, left)
            answer_a = str(generations[assistant_a][prompt_index]["generated_text"])
            answer_b = str(generations[assistant_b][prompt_index]["generated_text"])
            tasks.append({
                "task_id": task_id,
                "prompt_index": prompt_index,
                "pair_left": left,
                "pair_right": right,
                "assistant_a_model": assistant_a,
                "assistant_b_model": assistant_b,
                "system": protocol["rubric"]["system"],
                "user": protocol["rubric"]["user_template"].format(
                    prompt=prompt, answer_a=answer_a, answer_b=answer_b
                ),
            })
    return sorted(tasks, key=lambda row: row["task_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--judge-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    protocol = json.loads(args.protocol_lock.read_text())
    if protocol.get("status") != "FROZEN_BEFORE_JUDGING":
        raise RuntimeError("invalid protocol lock")
    if args.num_shards != protocol["randomization"]["num_shards"]:
        raise RuntimeError("shard-count mismatch")
    for model, artifact in protocol["generation_files"].items():
        path = args.work / "generations" / model / "output_42.json"
        if sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"generation hash mismatch: {model}")
    tasks = make_tasks(protocol, args.work, args.shard_index, args.num_shards)
    completed = {row["task_id"]: row for row in load_jsonl(args.output)} if args.output.exists() else {}
    pending = [task for task in tasks if task["task_id"] not in completed]
    print(json.dumps({"shard": args.shard_index, "total": len(tasks), "completed": len(completed), "pending": len(pending)}), flush=True)
    if pending:
        judge = protocol["judge"]
        llm = LLM(
            model=str(args.judge_model_path), dtype=judge["dtype"],
            max_model_len=judge["max_model_len"], tensor_parallel_size=1,
            gpu_memory_utilization=judge["gpu_memory_utilization"], seed=judge["seed"],
            trust_remote_code=judge["trust_remote_code"], enable_prefix_caching=True,
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
                if len(outputs) != len(batch):
                    raise RuntimeError("judge output count mismatch")
                for task, output in zip(batch, outputs):
                    raw = output.outputs[0].text
                    parsed, valid = parse(raw)
                    row = {key: value for key, value in task.items() if key not in {"system", "user"}}
                    row.update({
                        "parsed": parsed,
                        "valid": valid,
                        "raw_judge_output": raw,
                        "judge_model": judge["model"],
                        "judge_revision": judge["revision"],
                        "protocol_sha256": protocol["configuration_sha256"],
                    })
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                print(json.dumps({"shard": args.shard_index, "done": min(start + len(batch), len(pending))}), flush=True)
    rows = load_jsonl(args.output)
    metadata = {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "judgments": len(rows),
        "valid": sum(bool(row["valid"]) for row in rows),
        "invalid": sum(not bool(row["valid"]) for row in rows),
        "output_sha256": sha256(args.output),
        "protocol_sha256": protocol["configuration_sha256"],
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    import wandb
    run_id = f"sealed-pairwise-gptoss120b-s{args.shard_index}-20260714"
    run = wandb.init(
        entity="promotion-kim", project="mnpo", id=run_id, resume="allow",
        name=f"sealed-pairwise-gptoss120b-s{args.shard_index}",
        group="aaai27-qwen3-sealed-pairwise",
        config={"protocol": protocol, "shard": args.shard_index},
    )
    run.log(metadata); run.summary.update(metadata); run.finish()
    metadata["wandb_run_id"] = run_id
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if metadata["invalid"]:
        raise RuntimeError(f"fail closed: {metadata['invalid']} invalid judge outputs")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
