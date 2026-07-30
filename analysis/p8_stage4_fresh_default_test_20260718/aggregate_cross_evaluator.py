#!/usr/bin/env python3
"""Aggregate the locked P8 cross-evaluator safety diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--llama", type=Path, nargs="+", required=True)
    parser.add_argument("--qwen", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    models = lock["models"]
    expected = lock["records"]
    evaluator_inputs = {"llama_guard_3": args.llama, "qwen3_guard": args.qwen}
    output = {}
    rng = np.random.default_rng(lock["analysis"]["bootstrap_seed"])
    indices = rng.integers(0, expected, size=(lock["analysis"]["bootstrap_resamples"], expected))
    for name, paths in evaluator_inputs.items():
        rows = []
        for path in paths:
            rows.extend(read_jsonl(path))
        rows.sort(key=lambda row: str(row["prompt_id"]))
        if len(rows) != expected or len({str(row["prompt_id"]) for row in rows}) != expected:
            raise RuntimeError(f"{name}: incomplete or duplicated rows")
        if any(row["response_model_names"] != models or len(row["all_rm_scores"]) != len(models) for row in rows):
            raise RuntimeError(f"{name}: model order or score width mismatch")
        scores = np.asarray([row["all_rm_scores"] for row in rows], dtype=float)
        if scores.shape != (expected, len(models)) or not np.isfinite(scores).all():
            raise RuntimeError(f"{name}: invalid score matrix")
        base = scores[:, models.index("base")]
        summary = []
        for index, model in enumerate(models):
            delta = scores[:, index] - base
            item = {
                "model": model,
                "mean_raw_score": float(scores[:, index].mean()),
                "mean_raw_score_ci95": [float(x) for x in np.quantile(scores[:, index][indices].mean(axis=1), [0.025, 0.975])],
                "paired_difference_vs_base": float(delta.mean()),
                "paired_difference_vs_base_ci95": [float(x) for x in np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])],
            }
            summary.append(item)
        output[name] = summary
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "completed",
        "post_hoc_diagnostic": True,
        "protocol_lock": str(args.lock),
        "protocol_sha256": lock.get("response_file_sha256"),
        "records": expected,
        "models": models,
        "results": output,
        "spent_sealed_split_touched": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["evaluator", "model", "mean_raw_score", "mean_raw_score_ci95", "paired_difference_vs_base", "paired_difference_vs_base_ci95"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for evaluator, rows in output.items():
            for row in rows:
                writer.writerow({"evaluator": evaluator, **row})
    lines = [
        "# P8 Stage-4 cross-evaluator safety diagnostic",
        "",
        "This post-hoc diagnostic rescored fixed P8 responses. It did not select, tune, or retrain any policy.",
        "",
    ]
    for evaluator, rows in output.items():
        lines.extend([f"## {evaluator}", "", "| Model | Raw mean (95% CI) | Paired difference vs Base (95% CI) |", "|---|---:|---:|"])
        for row in rows:
            mean_ci, delta_ci = row["mean_raw_score_ci95"], row["paired_difference_vs_base_ci95"]
            lines.append(f"| {row['model']} | {row['mean_raw_score']:.4f} [{mean_ci[0]:.4f}, {mean_ci[1]:.4f}] | {row['paired_difference_vs_base']:+.4f} [{delta_ci[0]:+.4f}, {delta_ci[1]:+.4f}] |")
        lines.append("")
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
