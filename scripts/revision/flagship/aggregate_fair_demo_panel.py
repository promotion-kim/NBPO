#!/usr/bin/env python3
"""Aggregate the locked two-judge, two-position panel and optionally lock validation selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np


OBJECTIVES = ("helpfulness", "safety", "conciseness")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def interval(values: np.ndarray, indices: np.ndarray) -> list[float]:
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def candidate_score(row: dict, objective: str) -> float:
    verdict = row["parsed"][objective]
    if verdict == "tie":
        return 0.5
    candidate_label = "A" if row["order"] == "candidate_A" else "B"
    return 1.0 if verdict == candidate_label else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-lock", type=Path, required=True)
    parser.add_argument("--evaluator-lock", type=Path, required=True)
    parser.add_argument("--judge-dir", action="append", required=True, help="judge_id=/path")
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--grid", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()

    evaluator = json.loads(args.evaluator_lock.read_text(encoding="utf-8"))
    input_lock = json.loads(args.input_lock.read_text(encoding="utf-8"))
    if evaluator.get("status") != "LOCKED_BEFORE_ANY_NEW_METHOD_RANKING":
        raise RuntimeError("evaluator not locked")
    if input_lock.get("input_sha256") != sha256(args.input):
        raise RuntimeError("judge input hash mismatch")
    if input_lock.get("evaluator_lock_sha256") != sha256(args.evaluator_lock):
        raise RuntimeError("evaluator hash mismatch")
    input_rows = load_jsonl(args.input)
    task_ids = {row["task_id"] for row in input_rows}
    judges = {}
    for item in args.judge_dir:
        judge, root = item.split("=", 1)
        rows = [row for path in sorted(Path(root).glob("shard_*.jsonl")) for row in load_jsonl(path)]
        by_id = {row["task_id"]: row for row in rows}
        if len(by_id) != len(rows) or set(by_id) != task_ids or any(not row.get("valid") for row in rows):
            raise RuntimeError(f"incomplete, duplicate, or invalid judge rows for {judge}")
        judges[judge] = by_id
    if set(judges) != {"gpt_oss_120b", "qwen3_32b"}:
        raise RuntimeError(f"expected both locked judges, got {sorted(judges)}")
    gates = json.loads(args.gates.read_text(encoding="utf-8"))
    eligible = sorted(gates["eligible_models"])
    candidates = sorted(input_lock["candidates"])
    if set(candidates) != set(eligible) - {"base"}:
        raise RuntimeError("judge candidates differ from stability-eligible candidates")
    prompt_count = int(input_lock["prompt_count"])
    rng = np.random.default_rng(42)
    indices = rng.integers(0, prompt_count, size=(2000, prompt_count), dtype=np.int32)
    input_by_id = {row["task_id"]: row for row in input_rows}

    prompt_scores: dict[str, dict[str, np.ndarray]] = {}
    position_agreement = []
    inter_judge_agreement = []
    for candidate in candidates:
        objective_arrays = {objective: np.empty(prompt_count, dtype=float) for objective in OBJECTIVES}
        for prompt_index in range(prompt_count):
            for objective in OBJECTIVES:
                values = []
                per_judge = {}
                for judge, rows in judges.items():
                    order_values = []
                    for order in ("candidate_A", "candidate_B"):
                        task_id = f"fair-demo-panel-v1|{input_lock['split']}|{candidate}|{prompt_index}|{order}"
                        if task_id not in input_by_id:
                            raise RuntimeError(f"missing input task {task_id}")
                        score = candidate_score(rows[task_id], objective)
                        values.append(score); order_values.append(score)
                    position_agreement.append(float(order_values[0] == order_values[1]))
                    per_judge[judge] = float(np.mean(order_values))
                inter_judge_agreement.append(float(per_judge["gpt_oss_120b"] == per_judge["qwen3_32b"]))
                objective_arrays[objective][prompt_index] = float(np.mean(values))
        prompt_scores[candidate] = objective_arrays

    summaries = {
        "base": {
            "model": "base", "stability": "passed", "prompt_count": prompt_count,
            "mean_prompt_worst_panel_score": 0.5,
            "mean_prompt_worst_panel_score_ci95": [0.5, 0.5],
            "delta_vs_base_tie": 0.0, "delta_vs_base_tie_ci95": [0.0, 0.0],
            "per_objective_panel_score": {objective: {"mean": 0.5, "ci95": [0.5, 0.5]} for objective in OBJECTIVES},
        }
    }
    worst_arrays = {"base": np.full(prompt_count, 0.5, dtype=float)}
    for candidate, arrays in prompt_scores.items():
        worst = np.min(np.column_stack([arrays[objective] for objective in OBJECTIVES]), axis=1)
        worst_arrays[candidate] = worst
        summaries[candidate] = {
            "model": candidate, "stability": "passed", "prompt_count": prompt_count,
            "mean_prompt_worst_panel_score": float(worst.mean()),
            "mean_prompt_worst_panel_score_ci95": interval(worst, indices),
            "delta_vs_base_tie": float((worst - 0.5).mean()),
            "delta_vs_base_tie_ci95": interval(worst - 0.5, indices),
            "per_objective_panel_score": {
                objective: {"mean": float(values.mean()), "ci95": interval(values, indices)}
                for objective, values in arrays.items()
            },
        }
    ranked = sorted(summaries.values(), key=lambda row: (-row["mean_prompt_worst_panel_score"], row["model"]))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    pairwise = []
    names = [row["model"] for row in ranked]
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            delta = worst_arrays[left] - worst_arrays[right]
            pairwise.append({"left": left, "right": right, "mean_difference": float(delta.mean()),
                             "ci95": interval(delta, indices)})
    result = {
        "status": "completed", "split": input_lock["split"], "primary": evaluator["primary"]["name"],
        "prompt_count": prompt_count, "ranked": ranked, "pairwise_prompt_bootstrap": pairwise,
        "position_score_agreement": float(np.mean(position_agreement)),
        "inter_judge_mean_position_score_agreement": float(np.mean(inter_judge_agreement)),
        "bootstrap": {"resamples": 2000, "seed": 42, "unit": "prompt", "paired": True},
        "input_sha256": sha256(args.input), "evaluator_lock_sha256": sha256(args.evaluator_lock),
        "spent_sealed_split_touched": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "panel_summary.json", result)
    with (args.output_dir / "panel_prompt_scores.jsonl").open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            for prompt_index in range(prompt_count):
                row = {"candidate": candidate, "prompt_index": prompt_index,
                       **{objective: float(prompt_scores[candidate][objective][prompt_index]) for objective in OBJECTIVES},
                       "worst": float(worst_arrays[candidate][prompt_index])}
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    if args.select:
        if args.grid is None:
            raise RuntimeError("--grid is required with --select")
        grid = json.loads(args.grid.read_text(encoding="utf-8"))
        method_for = {row["id"]: row["method"] for row in grid["candidates"]}
        by_method = {}
        failed_methods = []
        for method in sorted(set(method_for.values())):
            available = [candidate for candidate in candidates if method_for.get(candidate) == method]
            if not available:
                failed_methods.append(method)
                continue
            selected = sorted(available, key=lambda name: (-summaries[name]["mean_prompt_worst_panel_score"], name))[0]
            by_method[method] = {
                "candidate_id": selected,
                "validation_primary": summaries[selected]["mean_prompt_worst_panel_score"],
                "validation_primary_ci95": summaries[selected]["mean_prompt_worst_panel_score_ci95"],
                "eligible_candidates": sorted(available),
            }
        ronpo_entries = [row for method, row in by_method.items() if method in {"ronpo_full_expect", "ronpo_k_only"}]
        selected_ronpo = None if not ronpo_entries else sorted(
            ronpo_entries, key=lambda row: (-row["validation_primary"], row["candidate_id"])
        )[0]["candidate_id"]
        lock = {
            "status": "VALIDATION_SELECTION_LOCKED_BEFORE_FRESH_TEST", "locked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "selection_metric": evaluator["primary"]["name"], "tie_break": "candidate_id lexical order",
            "selected_by_method": by_method, "selected_ronpo_overall": selected_ronpo,
            "failed_methods": failed_methods, "panel_summary_sha256": sha256(args.output_dir / "panel_summary.json"),
            "evaluator_lock_sha256": sha256(args.evaluator_lock), "grid_sha256": sha256(args.grid),
            "fresh_test_opened": False, "spent_sealed_split_touched": False,
        }
        atomic_json(args.output_dir / "selection_lock.json", lock)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
