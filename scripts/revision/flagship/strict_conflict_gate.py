#!/usr/bin/env python3
"""Pre-training conflict gate for three continuous objective score files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key(row: dict[str, Any]) -> str:
    return str(row.get("prompt_id") or row.get("prompt"))


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    x = np.asarray(average_ranks(left), dtype=np.float64)
    y = np.asarray(average_ranks(right), dtype=np.float64)
    if x.std() <= 1e-12 or y.std() <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def parse_named(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected name=path, got {value!r}")
        name, path = value.split("=", 1)
        result.append((name.strip(), Path(path.strip())))
    if len(result) != 3 or len({name for name, _ in result}) != 3:
        raise ValueError("exactly three unique objectives are required")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-files", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--std-floor", type=float, default=0.01)
    parser.add_argument("--max-median-rho", type=float, default=0.0)
    parser.add_argument("--min-top1-mismatch", type=float, default=0.50)
    parser.add_argument("--objective-spec-json", required=True)
    args = parser.parse_args()

    named = parse_named(args.scored_files)
    objective_spec = json.load(open(args.objective_spec_json, encoding="utf-8"))
    by_objective = {name: {key(row): row for row in read_jsonl(path)} for name, path in named}
    names = [name for name, _ in named]
    common = set(by_objective[names[0]])
    for name in names[1:]:
        common &= set(by_objective[name])

    records = []
    for prompt_id in sorted(common):
        first = by_objective[names[0]][prompt_id]
        responses = first.get("all_generated_responses")
        if not isinstance(responses, list) or len(responses) < 2:
            continue
        scores = {}
        valid = True
        for name in names:
            row = by_objective[name][prompt_id]
            values = row.get("all_rm_scores")
            if row.get("prompt") != first.get("prompt") or row.get("all_generated_responses") != responses:
                valid = False
                break
            if not isinstance(values, list) or len(values) != len(responses):
                valid = False
                break
            values = [float(value) for value in values]
            if not all(math.isfinite(value) for value in values):
                valid = False
                break
            scores[name] = values
        if valid:
            records.append((prompt_id, scores))
    if not records:
        raise RuntimeError("no complete prompt records")

    distributions = {
        name: quantiles([value for _, scores in records for value in scores[name]]) for name in names
    }
    pair_rows = []
    all_rhos = []
    all_mismatches = []
    for left, right in combinations(names, 2):
        rhos = []
        mismatches = []
        for _, scores in records:
            rho = spearman(scores[left], scores[right])
            if math.isfinite(rho):
                rhos.append(rho)
            mismatches.append(float(np.argmax(scores[left]) != np.argmax(scores[right])))
        all_rhos.extend(rhos)
        all_mismatches.extend(mismatches)
        pair_rows.append(
            {
                "left": left,
                "right": right,
                "finite_prompt_rhos": len(rhos),
                "spearman_mean": statistics.fmean(rhos) if rhos else float("nan"),
                "spearman_median": statistics.median(rhos) if rhos else float("nan"),
                "top1_mismatch_rate": statistics.fmean(mismatches),
            }
        )

    aggregate_rho = statistics.median(all_rhos) if all_rhos else float("nan")
    aggregate_mismatch = statistics.fmean(all_mismatches)
    variance_pass = all(distributions[name]["std"] > args.std_floor for name in names)
    rho_pass = math.isfinite(aggregate_rho) and aggregate_rho < args.max_median_rho
    mismatch_pass = aggregate_mismatch >= args.min_top1_mismatch
    passed = variance_pass and rho_pass and mismatch_pass
    result = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "num_prompts": len(records),
        "num_responses": sum(len(scores[names[0]]) for _, scores in records),
        "objective_spec": objective_spec,
        "objectives": names,
        "thresholds_preregistered_before_scoring": {
            "raw_score_std_strictly_greater_than": args.std_floor,
            "aggregate_prompt_pair_spearman_median_strictly_less_than": args.max_median_rho,
            "aggregate_pairwise_top1_mismatch_at_least": args.min_top1_mismatch,
        },
        "score_distributions": distributions,
        "pairwise": pair_rows,
        "aggregate_prompt_pair_spearman_median": aggregate_rho,
        "aggregate_pairwise_top1_mismatch_rate": aggregate_mismatch,
        "checks": {
            "nondegenerate_variance": variance_pass,
            "negative_median_cross_objective_spearman": rho_pass,
            "high_top1_mismatch": mismatch_pass,
        },
        "passed": passed,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
