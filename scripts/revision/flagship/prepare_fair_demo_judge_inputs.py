#!/usr/bin/env python3
"""Create and hash position-swapped candidate-vs-base judge tasks without scoring them."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def load_generation(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"invalid generation file: {path}")
    for index, row in enumerate(rows):
        if not str(row.get("prompt", "")).strip() or not str(row.get("generated_text", "")).strip():
            raise RuntimeError(f"empty prompt/response at {path}:{index}")
    return rows


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], help="name=/path/output_42.json")
    parser.add_argument("--evaluator-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock-output", type=Path, required=True)
    args = parser.parse_args()

    evaluator = json.loads(args.evaluator_lock.read_text(encoding="utf-8"))
    if evaluator.get("status") != "LOCKED_BEFORE_ANY_NEW_METHOD_RANKING":
        raise RuntimeError("evaluator is not locked")
    base = load_generation(args.base)
    candidates = {}
    for item in args.candidate:
        name, path = item.split("=", 1)
        if name == "base" or name in candidates:
            raise RuntimeError(f"duplicate or reserved candidate name: {name}")
        candidates[name] = load_generation(Path(path))
    if not candidates:
        raise RuntimeError("no candidates supplied")
    base_prompts = [str(row["prompt"]) for row in base]
    for name, rows in candidates.items():
        if [str(row["prompt"]) for row in rows] != base_prompts:
            raise RuntimeError(f"prompt order differs for {name}")

    tasks = []
    for candidate in sorted(candidates):
        for index, (base_row, candidate_row) in enumerate(zip(base, candidates[candidate])):
            prompt = str(base_row["prompt"])
            base_response = str(base_row["generated_text"])
            candidate_response = str(candidate_row["generated_text"])
            for order in ("candidate_A", "candidate_B"):
                answer_a, answer_b = ((candidate_response, base_response) if order == "candidate_A"
                                      else (base_response, candidate_response))
                tasks.append({
                    "task_id": f"fair-demo-panel-v1|{args.split}|{candidate}|{index}|{order}",
                    "split": args.split, "prompt_index": index, "prompt_sha256": prompt_hash(prompt),
                    "candidate": candidate, "order": order, "prompt": prompt,
                    "answer_a": answer_a, "answer_b": answer_b,
                })
    tasks.sort(key=lambda row: row["task_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in tasks:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(args.output)
    lock = {
        "status": "LOCKED_BEFORE_JUDGING", "locked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "split": args.split, "prompt_count": len(base), "candidates": sorted(candidates),
        "position_swap": True, "tasks_per_candidate": 2 * len(base), "total_tasks_per_judge": len(tasks),
        "input_sha256": sha256(args.output), "evaluator_lock_sha256": sha256(args.evaluator_lock),
        "spent_sealed_split_touched": False,
    }
    atomic_json(args.lock_output, lock)
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
