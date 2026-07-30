#!/usr/bin/env python3
"""Seed-average P1 results and paired prompt-bootstrap publication artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mnpo_scripts.evaluate_multi_objective_models import load_scores, parse_named_paths


METHODS = (
    "ronpo_full_expect", "ronpo_k_only", "dpo", "ipo", "simpo", "kto", "sppo_avg", "inpo_avg",
    "ht_mnpo_helpfulness", "ht_mnpo_safety", "ht_mnpo_conciseness",
)
SEEDS = (42, 43, 44)
MODEL_RE = re.compile(r"^(?P<method>.+)__s(?P<seed>\d+)$")


def minmax(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    low, high = float(array.min()), float(array.max())
    if not math.isfinite(low) or not math.isfinite(high) or high - low < 1e-12:
        return np.full_like(array, 0.5)
    return (array - low) / (high - low)


def interval(samples: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-files", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--expected-prompts", type=int, default=604)
    args = parser.parse_args()
    objectives, merged = load_scores(parse_named_paths(args.scored_files))
    records = [row for row in merged.values() if all(name in row["scores"] for name in objectives)]
    if len(records) != args.expected_prompts:
        raise RuntimeError(f"expected {args.expected_prompts} complete prompts, found {len(records)}")
    model_names = records[0]["response_model_names"]
    if "base" not in model_names:
        raise RuntimeError("base model missing")
    for row in records:
        if row["response_model_names"] != model_names:
            raise RuntimeError("model ordering mismatch")

    groups: dict[str, dict[int, int]] = {method: {} for method in METHODS}
    for index, name in enumerate(model_names):
        match = MODEL_RE.match(name)
        if match and match.group("method") in groups:
            groups[match.group("method")][int(match.group("seed"))] = index
    complete_methods = [method for method in METHODS if set(groups[method]) == set(SEEDS)]
    incomplete = {method: sorted(groups[method]) for method in METHODS if method not in complete_methods}
    base_index = model_names.index("base")
    prompt_values = {method: {objective: [] for objective in objectives} for method in complete_methods}
    base_values = {objective: [] for objective in objectives}
    for row in records:
        for objective in objectives:
            normalized = minmax(row["scores"][objective])
            base_values[objective].append(float(normalized[base_index]))
            for method in complete_methods:
                prompt_values[method][objective].append(
                    float(np.mean([normalized[groups[method][seed]] for seed in SEEDS]))
                )

    n = len(records)
    rng = np.random.default_rng(args.bootstrap_seed)
    indices = rng.integers(0, n, size=(args.bootstrap_samples, n), dtype=np.int32)
    rows = []
    per_prompt_worst = {}
    per_prompt_avg = {}
    objective_means = {}
    for method in complete_methods:
        arrays = {objective: np.asarray(prompt_values[method][objective]) for objective in objectives}
        stack = np.stack([arrays[objective] for objective in objectives], axis=0)
        worst, avg = stack.min(axis=0), stack.mean(axis=0)
        per_prompt_worst[method], per_prompt_avg[method] = worst, avg
        objective_means[method] = {objective: float(arrays[objective].mean()) for objective in objectives}
        row: dict[str, Any] = {
            "method": method, "training_seeds": "42,43,44", "num_prompts": n,
            "mean_prompt_worst_norm": float(worst.mean()),
            "mean_prompt_worst_norm_ci95_low": interval(worst[indices].mean(axis=1))[0],
            "mean_prompt_worst_norm_ci95_high": interval(worst[indices].mean(axis=1))[1],
            "mean_prompt_avg_norm": float(avg.mean()),
            "min_objective_mean_norm": min(objective_means[method].values()),
        }
        base_stack = np.stack([np.asarray(base_values[objective]) for objective in objectives], axis=0)
        wins = (stack > base_stack).mean(axis=0) + 0.5 * (stack == base_stack).mean(axis=0)
        row["mean_win_rate_vs_base"] = float(wins.mean())
        for objective in objectives:
            values = arrays[objective]
            row[f"{objective}_mean_norm"] = float(values.mean())
            ci = interval(values[indices].mean(axis=1))
            row[f"{objective}_mean_norm_ci95_low"], row[f"{objective}_mean_norm_ci95_high"] = ci
        rows.append(row)

    for row in rows:
        method = row["method"]
        dominated_by = []
        for other in complete_methods:
            if other == method:
                continue
            left, right = objective_means[other], objective_means[method]
            if all(left[obj] >= right[obj] for obj in objectives) and any(left[obj] > right[obj] for obj in objectives):
                dominated_by.append(other)
        row["pareto_nondominated"] = not dominated_by
        row["dominated_by"] = ",".join(dominated_by)

    ranked = sorted(rows, key=lambda row: (-float(row["mean_prompt_worst_norm"]), row["method"]))
    for rank, row in enumerate(ranked, 1):
        row["worst_objective_rank"] = rank
    paired = []
    for left, right in combinations(complete_methods, 2):
        differences = per_prompt_worst[left] - per_prompt_worst[right]
        ci = interval(differences[indices].mean(axis=1))
        paired.append({
            "left_method": left, "right_method": right,
            "mean_prompt_worst_difference": float(differences.mean()),
            "ci95_low": ci[0], "ci95_high": ci[1], "num_prompts": n,
        })

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "method_summary.csv", ranked)
    write_csv(out / "paired_worst_differences.csv", paired)
    result = {
        "schema_version": 1, "num_prompts": n, "objectives": objectives,
        "normalization": "per-prompt min-max across base and every stability-passing method-seed generation",
        "seed_aggregation": "mean normalized objective per prompt over training seeds 42/43/44",
        "bootstrap": {"unit": "prompt", "paired": True, "samples": args.bootstrap_samples,
                      "seed": args.bootstrap_seed, "interval": "percentile_95"},
        "complete_methods": complete_methods, "incomplete_methods": incomplete,
        "method_summary": ranked, "paper_ranking_valid": len(complete_methods) == len(METHODS),
    }
    (out / "flagship_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    latex = []
    for row in ranked:
        values = [row[f"{objective}_mean_norm"] for objective in objectives]
        latex.append(
            "{} & {} & {:.3f} & {:.3f} & {:.3f} & {:.3f} [{:.3f}, {:.3f}] & {:.3f} \\\\".format(
                row["method"].replace("_", "\\_"), row["training_seeds"], *values,
                row["mean_prompt_worst_norm"], row["mean_prompt_worst_norm_ci95_low"],
                row["mean_prompt_worst_norm_ci95_high"], row["mean_win_rate_vs_base"],
            )
        )
    (out / "latex_rows.tex").write_text("\n".join(latex) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
