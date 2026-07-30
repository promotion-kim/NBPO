#!/usr/bin/env python3
"""Aggregate the pre-registered marginal-then-min local-RM metric."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def interval(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def minmax(values: np.ndarray) -> np.ndarray:
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return np.full_like(values, 0.5, dtype=float)
    return (values - lo) / (hi - lo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--score", action="append", required=True, help="objective=score.jsonl")
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--metric-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--locked-model-set", action="store_true",
                        help="The input models were selected earlier; do not describe this split as selection.")
    args = parser.parse_args()

    lock = json.loads(args.metric_lock.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_BEFORE_ANY_STAGE1_OS_LOCAL_RM_RANKING":
        raise RuntimeError("metric lock is invalid")
    objectives = list(lock["objectives"])
    score_paths = dict(value.split("=", 1) for value in args.score)
    if list(score_paths) != objectives:
        raise RuntimeError(f"objective order/set differs from lock: {list(score_paths)} vs {objectives}")
    merged = json.loads(args.merged.read_text(encoding="utf-8"))
    names = list(merged[0]["response_model_names"])
    n = len(merged)
    if any(row["response_model_names"] != names for row in merged) or "base" not in names:
        raise RuntimeError("merged input alignment failed")
    matrices: dict[str, np.ndarray] = {}
    for objective in objectives:
        rows = load_jsonl(Path(score_paths[objective]))
        if len(rows) != n or any(row["response_model_names"] != names for row in rows):
            raise RuntimeError(f"score alignment failed for {objective}")
        matrix = np.asarray([row["all_rm_scores"] for row in rows], dtype=float)
        if matrix.shape != (n, len(names)) or not np.isfinite(matrix).all():
            raise RuntimeError(f"invalid finite score matrix for {objective}")
        matrices[objective] = matrix
    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    candidates = {row["id"]: row for row in grid["candidates"]}
    gates = json.loads(args.gates.read_text(encoding="utf-8"))
    eligible = set(gates["eligible_candidates"])
    if set(names[1:]) != eligible:
        raise RuntimeError("merged models and gate-eligible models differ")

    base_index = names.index("base")
    rng = np.random.default_rng(lock["bootstrap"]["seed"])
    b = int(lock["bootstrap"]["resamples"])
    indices = rng.integers(0, n, size=(b, n), dtype=np.int32)
    prompt_norm: dict[str, np.ndarray] = {
        objective: np.stack([minmax(row) for row in matrices[objective]]) for objective in objectives
    }
    summary = []
    per_objective = []
    prompt_rows = []
    for model_index, model in enumerate(names):
        marginal_values = []
        marginal_boot = []
        raw_deltas_by_obj = []
        normalized_by_obj = []
        for objective in objectives:
            scores = matrices[objective][:, model_index]
            base = matrices[objective][:, base_index]
            delta = scores - base
            wins = np.where(delta > 0.0, 1.0, np.where(delta < 0.0, 0.0, 0.5))
            if model == "base":
                wins = np.full(n, 0.5, dtype=float)
            boot_delta = delta[indices].mean(axis=1)
            boot_win = wins[indices].mean(axis=1)
            marginal_values.append(float(wins.mean()))
            marginal_boot.append(boot_win)
            raw_deltas_by_obj.append(delta)
            normalized_by_obj.append(prompt_norm[objective][:, model_index])
            per_objective.append({
                "model": model, "objective": objective, "prompt_count": n,
                "mean_raw_score": float(scores.mean()), "mean_base_raw_score": float(base.mean()),
                "mean_paired_raw_delta_vs_base": float(delta.mean()),
                "paired_raw_delta_vs_base_ci95": interval(boot_delta),
                "marginal_win_rate_vs_base": float(wins.mean()),
                "marginal_win_rate_vs_base_ci95": interval(boot_win),
            })
        marginal_array = np.asarray(marginal_values, dtype=float)
        marginal_boot_array = np.stack(marginal_boot, axis=0)
        primary_boot = marginal_boot_array.min(axis=0)
        mean_boot = marginal_boot_array.mean(axis=0)
        spread_boot = marginal_boot_array.max(axis=0) - marginal_boot_array.min(axis=0)
        normalized_matrix = np.stack(normalized_by_obj, axis=1)
        row = {
            "model": model, "method": "base" if model == "base" else candidates[model]["method"],
            "stage": 0 if model == "base" else candidates[model]["stage"], "prompt_count": n,
            "worst_objective_marginal_win_rate": float(marginal_array.min()),
            "worst_objective_marginal_win_rate_ci95": interval(primary_boot),
            "mean_objective_marginal_win_rate": float(marginal_array.mean()),
            "mean_objective_marginal_win_rate_ci95": interval(mean_boot),
            "cross_objective_marginal_spread": float(marginal_array.max() - marginal_array.min()),
            "cross_objective_marginal_spread_ci95": interval(spread_boot),
            "mean_prompt_worst_norm_score_continuity": float(normalized_matrix.min(axis=1).mean()),
            "min_objective_mean_norm_score_continuity": float(normalized_matrix.mean(axis=0).min()),
            "model_path": None if model == "base" else candidates[model]["model_path"],
            "source": "Qwen/Qwen3-8B" if model == "base" else candidates[model]["source"],
        }
        summary.append(row)
        for prompt_index in range(n):
            prompt_rows.append({
                "prompt_index": prompt_index, "model": model,
                **{f"raw_delta_{objective}": float(raw_deltas_by_obj[j][prompt_index])
                   for j, objective in enumerate(objectives)},
                **{f"marginal_win_{objective}": float(
                    0.5 if model == "base" else (1.0 if raw_deltas_by_obj[j][prompt_index] > 0 else
                    0.0 if raw_deltas_by_obj[j][prompt_index] < 0 else 0.5))
                   for j, objective in enumerate(objectives)},
            })
    ranked_candidates = sorted(summary, key=lambda row: (-row["worst_objective_marginal_win_rate"], row["model"]))
    for rank, row in enumerate(ranked_candidates, 1):
        row["global_rank"] = rank
    selected = []
    methods = sorted({row["method"] for row in summary if row["method"] != "base"})
    for method in methods:
        rows = [row for row in summary if row["method"] == method]
        selected.append(sorted(rows, key=lambda row: (-row["worst_objective_marginal_win_rate"], row["model"]))[0])
    selected_with_base = [next(row for row in summary if row["model"] == "base"), *selected]
    selected_ranked = sorted(selected_with_base,
                             key=lambda row: (-row["worst_objective_marginal_win_rate"], row["model"]))
    for rank, row in enumerate(selected_ranked, 1):
        row["selected_set_rank"] = rank
    output = {
        "status": "completed", "split": args.split_name, "prompt_count": n,
        "primary_metric": lock["primary"], "bootstrap": lock["bootstrap"],
        "ranked_all_eligible_candidates": ranked_candidates,
        "selected_candidate_per_method": selected,
        "ranked_selected_method_set": selected_ranked,
        "selection_used_only_this_split": not args.locked_model_set,
        "model_selection_performed_on_this_split": not args.locked_model_set,
        "spent_sealed_split_touched": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "model_summary.json", output)
    for path, rows in [(args.output_dir / "per_objective_scores.csv", per_objective),
                       (args.output_dir / "prompt_level_deltas.csv", prompt_rows)]:
        fields = list(rows[0])
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
