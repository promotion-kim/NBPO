#!/usr/bin/env python3
"""Aggregate locked RM raw deltas and diagnostic-scale standardized worst deltas."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def interval(values: np.ndarray, indices: np.ndarray) -> list[float]:
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--score", action="append", default=[], help="signal=/path/file.jsonl")
    parser.add_argument("--evaluator-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    evaluator = json.loads(args.evaluator_lock.read_text(encoding="utf-8"))
    signals = evaluator["objective_signals"]
    scales = evaluator["secondary"]["diagnostic_control_scale"]
    score_paths = dict(item.split("=", 1) for item in args.score)
    merged = json.loads(args.merged.read_text(encoding="utf-8"))
    if not isinstance(merged, list) or not merged:
        raise RuntimeError("invalid merged generations")
    prompt_count = len(merged)
    names = list(merged[0]["response_model_names"])
    if "base" not in names:
        raise RuntimeError("base response is required")
    if any(row["response_model_names"] != names for row in merged):
        raise RuntimeError("model order differs across merged prompts")
    by_signal: dict[str, np.ndarray] = {}
    for signal in signals:
        if signal == "length_conciseness":
            by_signal[signal] = np.asarray([
                [-math.log1p(len(str(response).split())) for response in row["all_generated_responses"]]
                for row in merged
            ], dtype=float)
            continue
        if signal not in score_paths:
            raise RuntimeError(f"missing score path for locked signal {signal}")
        rows = load_jsonl(Path(score_paths[signal]))
        if len(rows) != prompt_count or any(row["response_model_names"] != names for row in rows):
            raise RuntimeError(f"score alignment failure for {signal}")
        by_signal[signal] = np.asarray([row["all_rm_scores"] for row in rows], dtype=float)
    base_index = names.index("base")
    rng = np.random.default_rng(42)
    indices = rng.integers(0, prompt_count, size=(2000, prompt_count), dtype=np.int32)
    summaries = []
    prompt_rows = []
    for model_index, model in enumerate(names):
        standardized = []
        raw = {}
        for signal in signals:
            values = by_signal[signal][:, model_index]
            base = by_signal[signal][:, base_index]
            delta = values - base
            scale = float(scales[signal])
            if not math.isfinite(scale) or scale <= 0.0:
                raise RuntimeError(f"invalid locked diagnostic scale for {signal}: {scale}")
            standardized.append(delta / scale)
            raw[signal] = {
                "mean_score": float(values.mean()), "mean_base_score": float(base.mean()),
                "mean_paired_delta": float(delta.mean()), "paired_delta_ci95": interval(delta, indices),
            }
        worst = np.min(np.column_stack(standardized), axis=1)
        summaries.append({
            "model": model, "prompt_count": prompt_count, "raw_objectives": raw,
            "mean_prompt_worst_standardized_delta": float(worst.mean()),
            "mean_prompt_worst_standardized_delta_ci95": interval(worst, indices),
        })
        for prompt_index in range(prompt_count):
            prompt_rows.append({
                "model": model, "prompt_index": prompt_index,
                **{f"raw_delta_{signal}": float(by_signal[signal][prompt_index, model_index]
                                                  - by_signal[signal][prompt_index, base_index])
                   for signal in signals},
                "worst_standardized_delta": float(worst[prompt_index]),
            })
    ranked = sorted(summaries, key=lambda row: (-row["mean_prompt_worst_standardized_delta"], row["model"]))
    for rank, row in enumerate(ranked, 1):
        row["secondary_rank"] = rank
    output = {
        "status": "completed", "objective_signals": signals,
        "normalization": evaluator["secondary"]["normalization"],
        "diagnostic_control_scale": scales, "prompt_count": prompt_count,
        "ranked_secondary": ranked,
        "bootstrap": {"resamples": 2000, "seed": 42, "unit": "prompt", "paired": True},
        "spent_sealed_split_touched": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "reward_summary.json", output)
    fields = list(prompt_rows[0])
    with (args.output_dir / "reward_prompt_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(prompt_rows)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
