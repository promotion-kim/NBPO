#!/usr/bin/env python3
"""Summarize fixed-config seed-42/43 conflict and IFEval replications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = ["base", "dpo_b0p01", "ipo_b0p05", "simpo_b2_g0p6", "sppo_eta0p0075", "inpo_eta0p0075", "ronpo"]
IFEVAL_METRICS = ["mean_prompt_level_strict", "mean_inst_level_strict", "mean_prompt_level_loose", "mean_inst_level_loose"]


def by_model(path: Path) -> dict[str, dict]:
    return {row["model"]: row for row in json.load(path.open())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed42-test-summary", required=True)
    parser.add_argument("--seed43-test-summary", required=True)
    parser.add_argument("--seed42-ifeval", required=True)
    parser.add_argument("--seed43-ifeval", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    tests = [by_model(Path(args.seed42_test_summary)), by_model(Path(args.seed43_test_summary))]
    ifevals = [by_model(Path(args.seed42_ifeval)), by_model(Path(args.seed43_ifeval))]
    reward_key = "mean_primary_prompt_worst_norm_score"

    reward_rows = []
    for method in METHODS:
        vals = [float(rows[method][reward_key]) for rows in tests]
        reward_rows.append({"model": method, "seed42": vals[0], "seed43": vals[1], "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))})
    ifeval_rows = []
    for method in METHODS:
        row: dict[str, object] = {"model": method}
        for metric in IFEVAL_METRICS:
            vals = [float(rows[method][metric]) for rows in ifevals]
            row[metric] = {"seed42": vals[0], "seed43": vals[1], "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))}
        ifeval_rows.append(row)
    ronpo_reward = next(row for row in reward_rows if row["model"] == "ronpo")
    output = {
        "training_seeds": [42, 43],
        "decode_seed": 42,
        "reward_metric": reward_key,
        "reward": reward_rows,
        "ifeval": ifeval_rows,
        "ronpo_reward_win_count_vs_trained_baselines": sum(
            ronpo_reward["mean"] > row["mean"] for row in reward_rows if row["model"] not in {"base", "ronpo"}
        ),
        "notes": ["n=2 training seeds; mean and sample SD are descriptive, not confidence intervals", "base is deterministic and identical across the two training-seed evaluations"],
    }
    Path(args.output_json).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    lines = ["# Two-seed fixed-config summary", "", "## Conflict primary", "", "| Model | Seed 42 | Seed 43 | Mean +/- sample SD |", "|---|---:|---:|---:|"]
    for row in sorted(reward_rows, key=lambda x: x["mean"], reverse=True):
        lines.append(f"| {row['model']} | {row['seed42']:.4f} | {row['seed43']:.4f} | {row['mean']:.4f} +/- {row['sd']:.4f} |")
    for metric in IFEVAL_METRICS:
        lines.extend(["", f"## {metric}", "", "| Model | Seed 42 | Seed 43 | Mean +/- sample SD |", "|---|---:|---:|---:|"])
        ranked = sorted(ifeval_rows, key=lambda x: x[metric]["mean"], reverse=True)
        for row in ranked:
            value = row[metric]
            lines.append(f"| {row['model']} | {value['seed42']:.4f} | {value['seed43']:.4f} | {value['mean']:.4f} +/- {value['sd']:.4f} |")
    lines.extend(["", "n=2 training seeds; values are descriptive means and sample SDs, not seed-level confidence intervals."])
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
