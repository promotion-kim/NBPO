#!/usr/bin/env python3
"""Fail-closed conflict and score-spread audit for the shared Table-4 pool."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

import numpy as np


def read(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {str(row["prompt_id"]): row for row in rows}


def spearman(x: list[float], y: list[float]) -> float:
    def rank(values: list[float]) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[order[end]] == values[order[start]]:
                end += 1
            result[order[start:end]] = (start + end - 1) / 2.0
            start = end
        return result
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--helpfulness", type=Path, required=True)
    parser.add_argument("--harmlessness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-median-correlation", type=float, default=0.5)
    args = parser.parse_args()
    helpful, harmless = read(args.helpfulness), read(args.harmlessness)
    if set(helpful) != set(harmless) or not helpful:
        raise RuntimeError("objective prompt sets differ or are empty")
    correlations, mismatches, h_ranges, s_ranges = [], 0, [], []
    for prompt_id in sorted(helpful):
        h = [float(value) for value in helpful[prompt_id]["all_rm_scores"]]
        s = [float(value) for value in harmless[prompt_id]["all_rm_scores"]]
        if len(h) != len(s) or len(h) != 4:
            raise RuntimeError(f"{prompt_id}: expected matched four-response scores")
        value = spearman(h, s)
        if math.isfinite(value):
            correlations.append(value)
        mismatches += int(int(np.argmax(h)) != int(np.argmax(s)))
        h_ranges.append(max(h) - min(h))
        s_ranges.append(max(s) - min(s))
    result = {
        "status": "pass" if correlations and median(correlations) <= args.max_median_correlation and mismatches > 0 else "fail",
        "scope": "shared base response pool, before any preference training",
        "prompts": len(helpful),
        "responses_per_prompt": 4,
        "median_prompt_spearman": float(median(correlations)) if correlations else math.nan,
        "mean_prompt_spearman": float(np.mean(correlations)) if correlations else math.nan,
        "nonfinite_prompt_correlations": len(helpful) - len(correlations),
        "reward_argmax_cost_argmax_mismatch_count": mismatches,
        "reward_argmax_cost_argmax_mismatch_rate": mismatches / len(helpful),
        "helpfulness_mean_within_prompt_range": float(np.mean(h_ranges)),
        "harmlessness_mean_within_prompt_range": float(np.mean(s_ranges)),
        "threshold": {"median_spearman_must_be_at_most": args.max_median_correlation, "argmax_mismatch_must_be_positive": True},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
