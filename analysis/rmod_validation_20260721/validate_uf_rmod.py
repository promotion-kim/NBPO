#!/usr/bin/env python3
"""Paired UltraFeedback validation for RMOD radar experiments.

Each tag must have one ``<tag>_<objective>.jsonl`` file.  The script aligns
all methods and objectives on the exact prompt intersection before computing
means and paired prompt-bootstrap confidence intervals.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


OBJECTIVES = (
    "instruction_following",
    "truthfulness",
    "honesty",
    "helpfulness",
    "safety",
)


def load_scores(path: Path) -> dict[str, float]:
    scores = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            values = row.get("all_rm_scores", [])
            if values:
                scores[row["prompt"]] = float(np.mean(values))
    return scores


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> list[float]:
    n = len(values)
    draws = rng.integers(0, n, size=(n_boot, n))
    means = values[draws].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True, help="tag=Label")
    parser.add_argument("--reference", required=True, help="label used for paired deltas")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    specs = [item.split("=", 1) for item in args.methods]
    tables = {
        label: {
            objective: load_scores(args.scored_dir / f"{tag}_{objective}.jsonl")
            for objective in OBJECTIVES
        }
        for tag, label in specs
    }
    if args.reference not in tables:
        raise ValueError(f"reference label not found: {args.reference}")

    common = None
    for per_objective in tables.values():
        for scores in per_objective.values():
            prompts = set(scores)
            common = prompts if common is None else common & prompts
    prompt_list = sorted(common or ())
    if not prompt_list:
        raise RuntimeError("no prompts shared by every method and objective")

    rng = np.random.default_rng(args.seed)
    ref = np.array(
        [[tables[args.reference][objective][prompt] for objective in OBJECTIVES]
         for prompt in prompt_list],
        dtype=np.float64,
    )
    summary = {
        "protocol": {
            "objectives": list(OBJECTIVES),
            "prompt_alignment": "exact intersection across every method and objective",
            "n_prompts": len(prompt_list),
            "bootstrap_resamples": args.bootstrap,
            "bootstrap_seed": args.seed,
            "reference": args.reference,
        },
        "methods": {},
    }
    csv_rows = []
    for _, label in specs:
        values = np.array(
            [[tables[label][objective][prompt] for objective in OBJECTIVES]
             for prompt in prompt_list],
            dtype=np.float64,
        )
        means = values.mean(axis=0)
        delta = values - ref
        mean_delta = delta.mean(axis=0)
        delta_ci = [bootstrap_ci(delta[:, j], rng, args.bootstrap) for j in range(len(OBJECTIVES))]
        aggregate = values.mean(axis=1)
        aggregate_delta = aggregate - ref.mean(axis=1)
        record = {
            "n_prompts": len(prompt_list),
            "mean_by_objective": dict(zip(OBJECTIVES, map(float, means))),
            "average_of_objective_means": float(means.mean()),
            "worst_objective_mean": float(means.min()),
            "paired_delta_vs_reference": dict(zip(OBJECTIVES, map(float, mean_delta))),
            "paired_delta_ci95": dict(zip(OBJECTIVES, delta_ci)),
            "paired_average_delta_vs_reference": float(aggregate_delta.mean()),
            "paired_average_delta_ci95": bootstrap_ci(aggregate_delta, rng, args.bootstrap),
        }
        summary["methods"][label] = record
        csv_rows.append({
            "method": label,
            **{objective: means[j] for j, objective in enumerate(OBJECTIVES)},
            "avg": means.mean(),
            "worst": means.min(),
            "avg_delta_vs_reference": aggregate_delta.mean(),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "paired_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.out_dir / "paired_means.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (args.out_dir / "prompt_manifest.json").write_text(
        json.dumps({"count": len(prompt_list), "prompts": prompt_list}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
