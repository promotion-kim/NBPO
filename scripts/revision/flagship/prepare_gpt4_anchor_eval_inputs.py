#!/usr/bin/env python3
"""Prepare immutable public GPT-4 anchors for the zero-cost open-weight proxy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


ALPACA_REPO = "tatsu-lab/alpaca_eval"
ALPACA_REVISION = "2edc6fad8be6b14ea7230aabfd08188da6b8b814"
ALPACA_FILE = "alpaca_eval_gpt4_baseline.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses-root", type=Path, required=True)
    parser.add_argument("--models-tsv", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--arena-reference", type=Path, required=True)
    parser.add_argument("--mt-reference", type=Path, required=True)
    parser.add_argument("--mt-questions", type=Path, required=True)
    parser.add_argument("--mt-judge-prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.models_tsv.open(encoding="utf-8") as handle:
        models = list(csv.DictReader(handle, delimiter="\t"))
    model_names = [row["name"] for row in models]
    if len(model_names) != 11 or len(set(model_names)) != 11:
        raise RuntimeError(f"expected 11 unique models, got {model_names}")

    responses: dict[str, dict[str, dict]] = {}
    for model in model_names:
        path = args.responses_root / model / "responses.jsonl"
        rows = load_jsonl(path)
        if len(rows) != 1385:
            raise RuntimeError(f"{model}: expected 1385 responses, got {len(rows)}")
        responses[model] = {row["item_id"]: row for row in rows}
    canonical = responses[model_names[0]]
    for model in model_names[1:]:
        if set(responses[model]) != set(canonical):
            raise RuntimeError(f"response item mismatch: {model}")
        for item_id, row in responses[model].items():
            if row["turns"] != canonical[item_id]["turns"]:
                raise RuntimeError(f"prompt mismatch: {model}/{item_id}")

    alpaca_path = Path(hf_hub_download(
        repo_id=ALPACA_REPO,
        repo_type="dataset",
        revision=ALPACA_REVISION,
        filename=ALPACA_FILE,
        cache_dir=str(args.hf_cache),
    ))
    alpaca_raw = json.loads(alpaca_path.read_text(encoding="utf-8"))
    alpaca_by_instruction = {row["instruction"]: row for row in alpaca_raw}
    alpaca_rows = []
    for item_id, source in sorted(canonical.items()):
        if source["benchmark"] != "alpaca_eval_2":
            continue
        instruction = source["turns"][0]
        reference = alpaca_by_instruction.get(instruction)
        if reference is None:
            raise RuntimeError(f"missing Alpaca GPT-4 reference: {item_id}")
        alpaca_rows.append({
            "benchmark": "alpaca_eval_2", "item_id": item_id,
            "instruction": instruction, "reference_answer": reference["output"],
            "reference_generator": reference.get("generator", "gpt4_1106_preview"),
        })
    if len(alpaca_rows) != 805:
        raise RuntimeError(f"expected 805 Alpaca references, got {len(alpaca_rows)}")

    arena_raw = load_jsonl(args.arena_reference)
    arena_by_id = {str(row.get("question_id", row.get("uid"))): row for row in arena_raw}
    arena_rows = []
    for item_id, source in sorted(canonical.items()):
        if source["benchmark"] != "arena_hard_v0.1":
            continue
        ref = arena_by_id.get(str(source["source_id"]))
        if ref is None:
            raise RuntimeError(f"missing Arena GPT-4 reference: {item_id}")
        try:
            if "messages" in ref:
                content = ref["messages"][1]["content"]
                answer = content["answer"] if isinstance(content, dict) else content
            else:
                answer = ref["choices"][0]["turns"][0]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"bad Arena reference schema: {item_id}") from exc
        arena_rows.append({
            "benchmark": "arena_hard_v0.1", "item_id": item_id,
            "question": source["turns"][0], "reference_answer": answer,
            "reference_generator": "gpt-4-0314", "source_id": str(source["source_id"]),
        })
    if len(arena_rows) != 500:
        raise RuntimeError(f"expected 500 Arena references, got {len(arena_rows)}")

    mt_questions = {str(row["question_id"]): row for row in load_jsonl(args.mt_questions)}
    mt_references = {str(row["question_id"]): row for row in load_jsonl(args.mt_reference)}
    mt_rows = []
    for item_id, source in sorted(canonical.items()):
        if source["benchmark"] != "mt_bench":
            continue
        qid = str(source["source_id"])
        question = mt_questions.get(qid)
        if question is None or question["turns"] != source["turns"]:
            raise RuntimeError(f"missing/mismatched MT-Bench question: {item_id}")
        ref = mt_references.get(qid)
        reference_answers = None
        if ref is not None:
            raw_turns = ref["choices"][0]["turns"]
            reference_answers = [turn["content"] if isinstance(turn, dict) else turn for turn in raw_turns]
        mt_rows.append({
            "benchmark": "mt_bench", "item_id": item_id, "question_id": source["source_id"],
            "category": question["category"], "turns": source["turns"],
            "reference_answers": reference_answers,
            "reference_generator": "gpt-4" if reference_answers else None,
        })
    if len(mt_rows) != 80:
        raise RuntimeError(f"expected 80 MT-Bench questions, got {len(mt_rows)}")
    with_ref = sum(row["reference_answers"] is not None for row in mt_rows)
    if with_ref != 30:
        raise RuntimeError(f"expected 30 MT-Bench GPT-4 references, got {with_ref}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_alpaca = args.output_dir / "alpaca_eval_2_gpt4_reference.jsonl"
    out_arena = args.output_dir / "arena_hard_v01_gpt4_reference.jsonl"
    out_mt = args.output_dir / "mt_bench_gpt4_reference.jsonl"
    write_jsonl(out_alpaca, alpaca_rows)
    write_jsonl(out_arena, arena_rows)
    write_jsonl(out_mt, mt_rows)
    manifest = {
        "models": models,
        "model_names": model_names,
        "response_counts": {name: len(responses[name]) for name in model_names},
        "references": {
            "alpaca_eval_2": {
                "repo": ALPACA_REPO, "revision": ALPACA_REVISION, "file": ALPACA_FILE,
                "source_sha256": sha256(alpaca_path), "normalized_sha256": sha256(out_alpaca), "count": 805,
            },
            "arena_hard_v0.1": {
                "generator": "gpt-4-0314", "source_sha256": sha256(args.arena_reference),
                "normalized_sha256": sha256(out_arena), "count": 500,
            },
            "mt_bench": {
                "generator": "gpt-4", "source_sha256": sha256(args.mt_reference),
                "questions_sha256": sha256(args.mt_questions),
                "judge_prompts_sha256": sha256(args.mt_judge_prompts),
                "normalized_sha256": sha256(out_mt), "count": 80, "reference_count": 30,
            },
        },
    }
    (args.output_dir / "reference_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
