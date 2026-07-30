#!/usr/bin/env python3
"""Paired prompt-level Pareto diagnostics for the conflict evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mnpo_scripts.evaluate_multi_objective_models import load_scores, parse_named_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-files", nargs="+", required=True)
    parser.add_argument("--focal-model", default="ronpo")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    objective_names, records = load_scores(parse_named_paths(args.scored_files))
    required = {"helpfulness", "safety", "brevity"}
    if set(objective_names) != required:
        raise ValueError(f"expected {sorted(required)}, got {sorted(objective_names)}")
    complete = [r for r in records.values() if required.issubset(r["scores"])]
    if not complete:
        raise ValueError("no complete prompt records")
    model_names = complete[0]["response_model_names"]
    if args.focal_model not in model_names:
        raise ValueError(f"missing focal model {args.focal_model!r}")
    focal_idx = model_names.index(args.focal_model)
    comparisons = [name for name in model_names if name != args.focal_model]
    rng = np.random.default_rng(args.bootstrap_seed)
    indices = rng.integers(0, len(complete), size=(args.bootstrap_samples, len(complete)))

    def summarize(values: np.ndarray, row: dict[str, object], key: str) -> None:
        values = values.astype(np.float64)
        samples = values[indices].mean(axis=1)
        low, high = np.quantile(samples, [0.025, 0.975])
        row[key] = float(values.mean())
        row[f"{key}_ci95_low"] = float(low)
        row[f"{key}_ci95_high"] = float(high)

    output: list[dict[str, object]] = []
    for comparison in comparisons:
        other_idx = model_names.index(comparison)
        deltas = {
            objective: np.asarray(
                [r["scores"][objective][focal_idx] - r["scores"][objective][other_idx] for r in complete],
                dtype=np.float64,
            )
            for objective in sorted(required)
        }
        help_up = deltas["helpfulness"] > 0
        help_noninferior = deltas["helpfulness"] >= 0
        brief_up = deltas["brevity"] > 0
        brief_noninferior = deltas["brevity"] >= 0
        safety_noninferior = deltas["safety"] >= 0
        row: dict[str, object] = {
            "focal_model": args.focal_model,
            "comparison_model": comparison,
            "num_prompts": len(complete),
        }
        metrics = {
            "help_win_rate": help_up,
            "brevity_win_rate": brief_up,
            "safety_noninferior_rate": safety_noninferior,
            "help_brevity_strict_both_win_rate": help_up & brief_up,
            "help_brevity_joint_noninferior_rate": help_noninferior & brief_noninferior,
            "help_brevity_pareto_win_rate": (help_noninferior & brief_noninferior) & (help_up | brief_up),
            "all_three_joint_noninferior_rate": help_noninferior & brief_noninferior & safety_noninferior,
            "quadrant_help_up_brevity_up_rate": help_up & brief_up,
            "quadrant_help_up_brevity_not_up_rate": help_up & ~brief_up,
            "quadrant_help_not_up_brevity_up_rate": ~help_up & brief_up,
            "quadrant_neither_strictly_up_rate": ~help_up & ~brief_up,
        }
        for key, values in metrics.items():
            summarize(values, row, key)
        output.append(row)

    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in output for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
