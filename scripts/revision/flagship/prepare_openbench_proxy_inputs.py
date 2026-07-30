#!/usr/bin/env python3
"""Freeze official AlpacaEval 2, Arena-Hard v0.1, and MT-Bench prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


ALPACA_REVISION = "2edc6fad8be6b14ea7230aabfd08188da6b8b814"
ALPACA_URL = (
    "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/"
    f"{ALPACA_REVISION}/alpaca_eval.json"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arena-repo", type=Path, required=True)
    parser.add_argument("--fastchat-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arena-commit", required=True)
    parser.add_argument("--fastchat-commit", required=True)
    parser.add_argument("--alpaca-repo-commit", required=True)
    args = parser.parse_args()

    arena_path = args.arena_repo / "data/arena-hard-v0.1/question.jsonl"
    mt_path = args.fastchat_repo / "fastchat/llm_judge/data/mt_bench/question.jsonl"
    arena_raw = arena_path.read_bytes()
    mt_raw = mt_path.read_bytes()
    alpaca_raw = urllib.request.urlopen(ALPACA_URL, timeout=120).read()
    alpaca = json.loads(alpaca_raw)
    arena = read_jsonl(arena_path)
    mt_bench = read_jsonl(mt_path)

    if (len(alpaca), len(arena), len(mt_bench)) != (805, 500, 80):
        raise RuntimeError(
            f"unexpected official prompt counts: {(len(alpaca), len(arena), len(mt_bench))}"
        )

    rows: list[dict] = []
    for index, item in enumerate(alpaca):
        rows.append({
            "benchmark": "alpaca_eval_2",
            "item_id": f"alpaca_eval_2:{index:04d}",
            "source_id": index,
            "category": item.get("dataset"),
            "turns": [item["instruction"]],
            "reference_answer": item.get("output"),
            "reference_generator": item.get("generator"),
        })
    for item in arena:
        rows.append({
            "benchmark": "arena_hard_v0.1",
            "item_id": f"arena_hard_v0.1:{item['uid']}",
            "source_id": item["uid"],
            "category": item.get("cluster") or item.get("category"),
            "turns": [item["prompt"]],
        })
    for item in mt_bench:
        rows.append({
            "benchmark": "mt_bench",
            "item_id": f"mt_bench:{item['question_id']}",
            "source_id": item["question_id"],
            "category": item.get("category"),
            "turns": item["turns"],
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = args.output_dir / "official_prompts.jsonl"
    prompts_bytes = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    prompts_path.write_bytes(prompts_bytes)
    (args.output_dir / "alpaca_eval_2_source.json").write_bytes(alpaca_raw)

    manifest = {
        "artifact_type": "zero_cost_open_weight_proxy_prompt_manifest",
        "official_score_reproduction": False,
        "total_items": len(rows),
        "counts": {"alpaca_eval_2": 805, "arena_hard_v0.1": 500, "mt_bench": 80},
        "prompt_sha256": sha256_bytes(prompts_bytes),
        "sources": {
            "alpaca_eval_2": {
                "repo": "tatsu-lab/alpaca_eval",
                "repo_commit": args.alpaca_repo_commit,
                "dataset_repo": "tatsu-lab/alpaca_eval",
                "dataset_revision": ALPACA_REVISION,
                "url": ALPACA_URL,
                "source_sha256": sha256_bytes(alpaca_raw),
            },
            "arena_hard_v0.1": {
                "repo": "lmarena/arena-hard-auto",
                "commit": args.arena_commit,
                "path": "data/arena-hard-v0.1/question.jsonl",
                "source_sha256": sha256_bytes(arena_raw),
            },
            "mt_bench": {
                "repo": "lm-sys/FastChat",
                "commit": args.fastchat_commit,
                "path": "fastchat/llm_judge/data/mt_bench/question.jsonl",
                "source_sha256": sha256_bytes(mt_raw),
            },
        },
    }
    (args.output_dir / "prompt_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
