#!/usr/bin/env python3
"""Paired prompt bootstrap for EvalScope IFEval review files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = (
    "prompt_level_strict",
    "inst_level_strict",
    "prompt_level_loose",
    "inst_level_loose",
)


def load_reviews(root: Path, model: str) -> dict[int, tuple[dict[str, float], int]]:
    paths = list((root / model).glob("**/reviews/*/ifeval_default.jsonl"))
    if len(paths) != 1:
        raise RuntimeError(f"{model}: expected one IFEval review file, found {paths}")
    rows: dict[int, tuple[dict[str, float], int]] = {}
    with paths[0].open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            score = record["sample_score"]
            values = {k: float(score["score"]["value"][k]) for k in METRICS}
            n_inst = len(score["sample_metadata"]["instruction_id_list"])
            rows[int(record["index"])] = (values, n_inst)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--focal-model", default="ronpo")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    manifest = [line.split("\t", 1)[0] for line in (root / "manifest.tsv").read_text().splitlines()]
    reviews = {model: load_reviews(root, model) for model in manifest}
    indices = sorted(reviews[args.focal_model])
    if len(indices) != 541 or any(sorted(rows) != indices for rows in reviews.values()):
        raise RuntimeError("IFEval prompt IDs are incomplete or not aligned across models")

    n = len(indices)
    rng = np.random.default_rng(args.bootstrap_seed)
    boot = rng.integers(0, n, size=(args.bootstrap_samples, n), dtype=np.int32)

    values: dict[str, dict[str, np.ndarray]] = {}
    n_inst: dict[str, np.ndarray] = {}
    for model, rows in reviews.items():
        values[model] = {
            metric: np.asarray([rows[i][0][metric] for i in indices], dtype=np.float64)
            for metric in METRICS
        }
        n_inst[model] = np.asarray([rows[i][1] for i in indices], dtype=np.float64)

    def point(model: str, metric: str) -> float:
        # EvalScope's per-review instruction-level value is already the mean
        # across that prompt's instructions; the official `mean_inst_level_*`
        # metric then averages those 541 prompt values.
        return float(values[model][metric].mean())

    def bootstrap_metric(model: str, metric: str) -> np.ndarray:
        return values[model][metric][boot].mean(axis=1)

    output: list[dict[str, object]] = []
    focal = args.focal_model
    for comparison in manifest:
        if comparison == focal:
            continue
        for metric in METRICS:
            differences = bootstrap_metric(focal, metric) - bootstrap_metric(comparison, metric)
            low, high = np.quantile(differences, [0.025, 0.975])
            output.append(
                {
                    "focal_model": focal,
                    "comparison_model": comparison,
                    "metric": metric,
                    "focal_score": point(focal, metric),
                    "comparison_score": point(comparison, metric),
                    "difference": point(focal, metric) - point(comparison, metric),
                    "difference_ci95_low": float(low),
                    "difference_ci95_high": float(high),
                    "num_prompts": n,
                    "bootstrap_samples": args.bootstrap_samples,
                    "bootstrap_seed": args.bootstrap_seed,
                }
            )

    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    fields = list(output[0])
    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    lines = [
        "| Comparison | Metric | RONPO | Other | Difference [95% CI] |",
        "|---|---|---:|---:|---:|",
    ]
    for row in output:
        lines.append(
            "| {comparison_model} | {metric} | {focal_score:.4f} | {comparison_score:.4f} | "
            "{difference:+.4f} [{difference_ci95_low:+.4f}, {difference_ci95_high:+.4f}] |".format(**row)
        )
    lines.extend(
        [
            "",
            "Paired percentile intervals resample the 541 prompts with replacement (10,000 samples, seed 42).",
            "EvalScope's instruction-level review value is already averaged within each prompt; all four metrics are therefore resampled and averaged over prompts exactly as in the official report. Intervals are prompt-level, not seed-level, and are not multiplicity-adjusted.",
        ]
    )
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
