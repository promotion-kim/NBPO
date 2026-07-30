#!/usr/bin/env python3
"""Extract deterministic, auditable position-consistent qualitative examples."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score(row: dict) -> float:
    if row["winner"] == "tie":
        return 0.5
    ronpo = "A" if row["order"] == "ronpo_A" else "B"
    return 1.0 if row["winner"] == ronpo else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses-root", type=Path, required=True)
    parser.add_argument("--judgments-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = {
        row["item_id"]: row
        for row in load_jsonl(args.responses_root / "ronpo_k_only" / "responses.jsonl")
    }
    opponents = ("base", "dpo", "ronpo_full_expect")
    opponent_rows = {
        name: {row["item_id"]: row for row in load_jsonl(args.responses_root / name / "responses.jsonl")}
        for name in opponents
    }
    judgments = []
    for path in sorted(args.judgments_dir.glob("shard_*.jsonl")):
        judgments.extend(row for row in load_jsonl(path) if row["opponent"] in opponents)
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in judgments:
        grouped[(row["benchmark"], row["opponent"], row["item_id"])].append(row)

    examples = []
    for benchmark in ("alpaca_eval_2", "arena_hard_v0.1", "mt_bench"):
        for opponent in opponents:
            candidates = []
            for (row_benchmark, row_opponent, item_id), pair in grouped.items():
                if (row_benchmark, row_opponent) != (benchmark, opponent) or len(pair) != 2:
                    continue
                scores = [score(row) for row in pair]
                if scores == [1.0, 1.0] or scores == [0.0, 0.0]:
                    candidates.append((item_id, scores[0], pair))
            for outcome in (1.0, 0.0):
                selected = next((value for value in sorted(candidates) if value[1] == outcome), None)
                if selected is None:
                    continue
                item_id, _, pair = selected
                source = candidate[item_id]
                examples.append({
                    "selection_rule": "lexicographically first item with identical winner after A/B position swap",
                    "benchmark": benchmark,
                    "opponent": opponent,
                    "outcome": "RONPO win" if outcome == 1.0 else "RONPO loss",
                    "item_id": item_id,
                    "category": source.get("category"),
                    "turns": source["turns"],
                    "ronpo_responses": source["responses"],
                    "opponent_responses": opponent_rows[opponent][item_id]["responses"],
                    "position_swapped_judgments": [
                        {"order": row["order"], "winner": row["winner"], "reason": row["reason"]}
                        for row in sorted(pair, key=lambda value: value["order"])
                    ],
                })
    output = {
        "artifact_type": "deterministically_selected_qualitative_examples",
        "candidate": "ronpo_k_only",
        "opponents": list(opponents),
        "selection_rule": "For each benchmark/opponent/outcome, first lexicographic item where both position-swapped judgments agree.",
        "num_examples": len(examples),
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"num_examples": len(examples), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
