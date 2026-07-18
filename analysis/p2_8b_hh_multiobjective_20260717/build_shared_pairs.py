#!/usr/bin/env python3
"""Build one shared, average-oriented pair set for all locked training arms.

The original P2 invocation fixed a 770-prompt, four-response pool.  Those
values remain the defaults for byte-compatible P2 reconstruction, but are
parameters so later preregistered studies can use a larger shared pool without
changing the pairing rule.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path


OBJECTIVES = ["helpfulness", "harmlessness"]
SCALE = 8.0


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))


def minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--helpfulness", type=Path, required=True)
    parser.add_argument("--harmlessness", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--pairs-per-prompt", type=int, default=3)
    parser.add_argument("--internal-test-prompts", type=int, default=38)
    parser.add_argument(
        "--expected-prompts",
        type=int,
        default=770,
        help="Expected scored prompt count; pass 0 to accept any positive count.",
    )
    parser.add_argument("--expected-responses", type=int, default=4)
    parser.add_argument(
        "--split-salt",
        default="p2",
        help="Deterministic namespace for the internal split and pair ordering.",
    )
    args = parser.parse_args()

    if args.expected_prompts < 0:
        raise ValueError("--expected-prompts must be non-negative")
    if args.expected_responses < 2:
        raise ValueError("--expected-responses must be at least two")

    sources = {}
    for objective, path in [("helpfulness", args.helpfulness), ("harmlessness", args.harmlessness)]:
        rows = read_jsonl(path)
        if args.expected_prompts and len(rows) != args.expected_prompts:
            raise RuntimeError(
                f"{objective}: expected {args.expected_prompts} score rows, got {len(rows)}"
            )
        if not rows:
            raise RuntimeError(f"{objective}: score file is empty")
        sources[objective] = {str(row["prompt_id"]): row for row in rows}
    prompt_ids = sorted(sources["helpfulness"])
    if set(prompt_ids) != set(sources["harmlessness"]):
        raise RuntimeError("objective prompt sets differ")
    internal_test = set(
        sorted(
            prompt_ids,
            key=lambda value: hashlib.sha256(
                f"{args.split_salt}-internal-test|{value}".encode()
            ).hexdigest(),
        )[:args.internal_test_prompts]
    )
    output = {"train": [], "test": []}
    for prompt_id in prompt_ids:
        hrow, srow = sources["helpfulness"][prompt_id], sources["harmlessness"][prompt_id]
        if hrow["all_generated_responses"] != srow["all_generated_responses"]:
            raise RuntimeError(f"response mismatch for {prompt_id}")
        responses = [str(value) for value in hrow["all_generated_responses"]]
        if len(responses) != args.expected_responses:
            raise RuntimeError(
                f"{prompt_id}: expected {args.expected_responses} responses, got {len(responses)}"
            )
        response_count = len(responses)
        raw = {
            "helpfulness": [float(value) for value in hrow["all_rm_scores"]],
            "harmlessness": [float(value) for value in srow["all_rm_scores"]],
        }
        normalized = {name: minmax(values) for name, values in raw.items()}
        avg = [
            sum(normalized[name][index] for name in OBJECTIVES) / len(OBJECTIVES)
            for index in range(response_count)
        ]
        win_probs = [
            sum(sigmoid(SCALE * (avg[i] - avg[j])) for j in range(response_count)) / response_count
            for i in range(response_count)
        ]
        pairs = list(itertools.combinations(range(response_count), 2))
        pairs.sort(
            key=lambda pair: hashlib.sha256(
                f"{args.split_salt}-common-pair|{prompt_id}|{pair[0]}|{pair[1]}".encode()
            ).hexdigest()
        )
        for pair_rank, (left, right) in enumerate(pairs[:args.pairs_per_prompt]):
            if (avg[right], -right) > (avg[left], -left):
                chosen, rejected = right, left
            else:
                chosen, rejected = left, right
            atom = int.from_bytes(
                hashlib.sha256(
                    f"{args.split_salt}-validator-atom|{prompt_id}|{pair_rank}".encode()
                ).digest()[:4],
                "big",
            )
            objective_index = atom % 2
            adversary_index = (atom // 2) % response_count
            name = OBJECTIVES[objective_index]
            objective_gap = (
                sigmoid(SCALE * (normalized[name][chosen] - normalized[name][adversary_index]))
                - sigmoid(SCALE * (normalized[name][rejected] - normalized[name][adversary_index]))
            )
            row = {
                "prompt_id": prompt_id,
                "prompt": hrow["prompt"],
                "source": hrow.get("source"),
                "slice": hrow.get("slice"),
                "behavior_label": hrow.get("behavior_label"),
                "all_generated_responses": responses,
                "objective_names": OBJECTIVES,
                "objective_scores": raw,
                "normalized_objective_scores": normalized,
                "avg_objective_scores": avg,
                "min_objective_scores": [min(normalized[name][i] for name in OBJECTIVES) for i in range(4)],
                "avg_oracle_win_probs": win_probs,
                "homogeneous_oracle": "average_minmax_objectives",
                "homogeneous_oracle_objectives": OBJECTIVES,
                "homogeneous_oracle_preference_scale": SCALE,
                "chosen": responses[chosen],
                "rejected": responses[rejected],
                "chosen_index": chosen,
                "rejected_index": rejected,
                "pair_source": "locked_common_uniform_policy_pair_avg_oriented",
                "ronpo_target": objective_gap,
                "ronpo_objective_name": name,
                "ronpo_objective_index": objective_index,
                "ronpo_objective_gap": objective_gap,
                "ronpo_weight": 1.0,
                "ronpo_adversary_response_index": adversary_index,
                "ronpo_adversary_mass": 1.0,
                "chosen_probs": win_probs[chosen],
                "rejected_probs": win_probs[rejected],
                "avg_oracle_chosen_score": avg[chosen],
                "avg_oracle_rejected_score": avg[rejected],
                "avg_oracle_score_gap": avg[chosen] - avg[rejected],
                "avg_oracle_chosen_win_prob": win_probs[chosen],
                "avg_oracle_rejected_win_prob": win_probs[rejected],
                "ht_target": normalized["harmlessness"][chosen] - normalized["harmlessness"][rejected],
                "ht_target_helpfulness": normalized["helpfulness"][chosen] - normalized["helpfulness"][rejected],
            }
            output["test" if prompt_id in internal_test else "train"].append(row)
    write_jsonl(args.train_output, output["train"])
    write_jsonl(args.test_output, output["test"])
    payload = {
        "status": "complete",
        "prompt_count": len(prompt_ids),
        "pairs_per_prompt": args.pairs_per_prompt,
        "expected_responses": args.expected_responses,
        "split_salt": args.split_salt,
        "train_rows": len(output["train"]),
        "test_rows": len(output["test"]),
        "internal_test_prompts": len(internal_test),
        "train_sha256": sha(args.train_output),
        "test_sha256": sha(args.test_output),
        "orientation": "higher mean of per-prompt minmax-normalized helpfulness and harmlessness",
        "shared_across_all_arms": True,
        "spent_sealed_split_touched": False,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
