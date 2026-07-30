#!/usr/bin/env python3
"""Aggregate locked panel judgments using marginal-then-min worst-objective score."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np


OBJECTIVES = ("helpfulness", "safety", "conciseness")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def ci(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-scores", type=Path, required=True)
    parser.add_argument("--metric-lock", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()
    lock = json.loads(args.metric_lock.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_BEFORE_VALIDATION_REAGGREGATION_OR_FRESH_MEASUREMENT":
        raise RuntimeError("metric is not locked")
    rows = [json.loads(line) for line in args.prompt_scores.read_text(encoding="utf-8").splitlines() if line.strip()]
    gates = json.loads(args.gates.read_text(encoding="utf-8"))
    eligible = sorted(set(gates["eligible_models"]) - {"base"})
    by_model: dict[str, list[dict]] = {model: [] for model in eligible}
    for row in rows:
        if row["candidate"] in by_model:
            by_model[row["candidate"]].append(row)
    if any(not values for values in by_model.values()):
        raise RuntimeError("eligible model missing prompt scores")
    counts = {len(values) for values in by_model.values()}
    if len(counts) != 1:
        raise RuntimeError(f"unequal prompt counts: {counts}")
    prompt_count = counts.pop()
    if any(sorted(row["prompt_index"] for row in values) != list(range(prompt_count)) for values in by_model.values()):
        raise RuntimeError("prompt indices are incomplete")
    rng = np.random.default_rng(42)
    indices = rng.integers(0, prompt_count, size=(2000, prompt_count), dtype=np.int32)

    summaries = []
    arrays_by_model = {}
    base_objective = np.full((prompt_count, len(OBJECTIVES)), 0.5)
    for model, values in {"base": []}.items():
        bootstrap_objectives = base_objective[indices].mean(axis=1)
        bootstrap_primary = bootstrap_objectives.min(axis=1)
        summaries.append({
            "model": model, "stability": "passed", "prompt_count": prompt_count,
            "worst_objective_marginal": 0.5, "worst_objective_marginal_ci95": [0.5, 0.5],
            "delta_vs_base": 0.0, "delta_vs_base_ci95": [0.0, 0.0],
            "cross_objective_disparity": 0.0, "cross_objective_disparity_ci95": [0.0, 0.0],
            "legacy_mean_prompt_worst": 0.5, "legacy_mean_prompt_worst_ci95": [0.5, 0.5],
            "per_objective_marginal": {objective: {"mean": 0.5, "ci95": [0.5, 0.5], "delta": 0.0,
                                                           "delta_ci95": [0.0, 0.0]}
                                         for objective in OBJECTIVES},
        })
        arrays_by_model[model] = base_objective
    for model, values in by_model.items():
        values = sorted(values, key=lambda row: row["prompt_index"])
        matrix = np.array([[float(row[objective]) for objective in OBJECTIVES] for row in values])
        arrays_by_model[model] = matrix
        marginals = matrix.mean(axis=0)
        boot_obj = matrix[indices].mean(axis=1)
        boot_primary = boot_obj.min(axis=1)
        primary = float(marginals.min())
        disparities = boot_obj.max(axis=1) - boot_obj.min(axis=1)
        legacy = matrix.min(axis=1)
        boot_legacy = legacy[indices].mean(axis=1)
        summaries.append({
            "model": model, "stability": "passed", "prompt_count": prompt_count,
            "worst_objective_marginal": primary, "worst_objective_marginal_ci95": ci(boot_primary),
            "delta_vs_base": primary - 0.5, "delta_vs_base_ci95": ci(boot_primary - 0.5),
            "cross_objective_disparity": float(marginals.max() - marginals.min()),
            "cross_objective_disparity_ci95": ci(disparities),
            "legacy_mean_prompt_worst": float(legacy.mean()), "legacy_mean_prompt_worst_ci95": ci(boot_legacy),
            "per_objective_marginal": {
                objective: {"mean": float(marginals[index]), "ci95": ci(boot_obj[:, index]),
                            "delta": float(marginals[index] - 0.5), "delta_ci95": ci(boot_obj[:, index] - 0.5)}
                for index, objective in enumerate(OBJECTIVES)
            },
        })
    ranked = sorted(summaries, key=lambda row: (-row["worst_objective_marginal"], row["model"]))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    failed = sorted(set(gates.get("failed_models", [])) - {"base"})
    result = {
        "status": "completed", "split": args.split,
        "primary": lock["primary"], "prompt_count": prompt_count,
        "ranked": ranked, "stability_failed_models": failed,
        "bootstrap": lock["bootstrap"],
        "metric_lock_sha256": sha256(args.metric_lock),
        "prompt_scores_sha256": sha256(args.prompt_scores),
        "spent_sealed_split_touched": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "panel_summary.json"
    atomic_json(summary_path, result)
    with (args.output_dir / "per_objective_marginals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["model", "rank", "worst_objective_marginal", "worst_ci_low", "worst_ci_high"]
        for objective in OBJECTIVES:
            header.extend([objective, f"{objective}_ci_low", f"{objective}_ci_high",
                           f"{objective}_delta_vs_base", f"{objective}_delta_ci_low",
                           f"{objective}_delta_ci_high"])
        header.extend(["disparity", "disparity_ci_low", "disparity_ci_high",
                       "legacy_mean_prompt_worst", "legacy_ci_low", "legacy_ci_high"])
        writer.writerow(header)
        for row in ranked:
            values = [row["model"], row["rank"], row["worst_objective_marginal"],
                      *row["worst_objective_marginal_ci95"]]
            for objective in OBJECTIVES:
                objective_row = row["per_objective_marginal"][objective]
                values.extend([objective_row["mean"], *objective_row["ci95"], objective_row["delta"],
                               *objective_row["delta_ci95"]])
            values.extend([row["cross_objective_disparity"], *row["cross_objective_disparity_ci95"],
                           row["legacy_mean_prompt_worst"], *row["legacy_mean_prompt_worst_ci95"]])
            writer.writerow(values)
    if args.select:
        grid = json.loads(args.grid.read_text(encoding="utf-8"))
        method_for = {row["id"]: row["method"] for row in grid["candidates"]}
        summary_for = {row["model"]: row for row in summaries}
        selected_by_method, failed_methods = {}, []
        for method in sorted(set(method_for.values())):
            available = sorted(model for model in eligible if method_for.get(model) == method)
            if not available:
                failed_methods.append(method)
                continue
            selected = sorted(available, key=lambda model: (-summary_for[model]["worst_objective_marginal"], model))[0]
            selected_by_method[method] = {
                "candidate_id": selected,
                "validation_primary": summary_for[selected]["worst_objective_marginal"],
                "validation_primary_ci95": summary_for[selected]["worst_objective_marginal_ci95"],
                "eligible_candidates": available,
            }
        ronpo_rows = [row for method, row in selected_by_method.items()
                      if method in {"ronpo_full_expect", "ronpo_k_only"}]
        selected_ronpo = None if not ronpo_rows else sorted(
            ronpo_rows, key=lambda row: (-row["validation_primary"], row["candidate_id"])
        )[0]["candidate_id"]
        atomic_json(args.output_dir / "selection_lock.json", {
            "status": "VALIDATION_SELECTION_LOCKED_BEFORE_FRESH_TEST",
            "locked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "selection_metric": lock["primary"]["name"],
            "selected_by_method": selected_by_method,
            "selected_ronpo_overall": selected_ronpo,
            "failed_methods": failed_methods,
            "tie_break": "candidate_id lexical order",
            "panel_summary_sha256": sha256(summary_path),
            "metric_lock_sha256": sha256(args.metric_lock),
            "grid_sha256": sha256(args.grid),
            "fresh_test_opened": False,
            "spent_sealed_split_touched": False,
        })
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
