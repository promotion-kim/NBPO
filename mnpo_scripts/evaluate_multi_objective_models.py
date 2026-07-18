import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def read_records(path: str) -> Iterable[Dict[str, Any]]:
    suffix = Path(path).suffix.lower()
    with open(path, "r", encoding="utf-8") as f:
        if suffix in {".jsonl", ".jsonlines"}:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        elif suffix == ".json":
            data = json.load(f)
            if isinstance(data, list):
                yield from data
            elif isinstance(data, dict):
                for value in data.values():
                    if isinstance(value, list):
                        yield from value
                        return
                yield data
            else:
                raise ValueError(f"Unsupported JSON structure in {path}")
        else:
            raise ValueError(f"Unsupported file suffix: {path}")


def parse_named_paths(values: List[str]) -> List[Tuple[str, str]]:
    out = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected name=path, got {value!r}")
        name, path = value.split("=", 1)
        out.append((name.strip(), path.strip()))
    return out


def prompt_key(record: Dict[str, Any]) -> str:
    return str(record.get("prompt_id") or record["prompt"])


def minmax(values: List[float]) -> List[float]:
    lo = min(values)
    hi = max(values)
    if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) < 1e-12:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def add_ci(row: Dict[str, Any], metric: str, samples: np.ndarray) -> None:
    low, high = np.quantile(samples, [0.025, 0.975])
    row[f"{metric}_ci95_low"] = float(low)
    row[f"{metric}_ci95_high"] = float(high)


def load_scores(named_paths: List[Tuple[str, str]]) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    objective_names = [name for name, _ in named_paths]
    merged: Dict[str, Dict[str, Any]] = {}
    for obj, path in named_paths:
        for record in read_records(path):
            key = prompt_key(record)
            scores = record.get("all_rm_scores")
            model_names = record.get("response_model_names")
            if not isinstance(scores, list) or not isinstance(model_names, list):
                continue
            if key not in merged:
                merged[key] = {
                    "prompt": record["prompt"],
                    "response_model_names": model_names,
                    "scores": {},
                }
            if merged[key]["response_model_names"] != model_names:
                raise ValueError(f"response_model_names mismatch for prompt {key}")
            merged[key]["scores"][obj] = [float(s) for s in scores]
    return objective_names, merged


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model generations under multiple objective scorers.")
    parser.add_argument("--scored_files", nargs="+", required=True, help="Objective scored files, e.g. skywork=...")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_model", default="baseline")
    parser.add_argument(
        "--primary_objectives",
        nargs="+",
        default=None,
        help="Pre-registered objective subset for primary prompt-average/worst metrics; defaults to all objectives.",
    )
    parser.add_argument("--tie_threshold", type=float, default=0.0)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap_samples must be >= 1")

    objective_names, records = load_scores(parse_named_paths(args.scored_files))
    primary_objectives = args.primary_objectives or objective_names
    unknown_primary = sorted(set(primary_objectives) - set(objective_names))
    if unknown_primary:
        raise ValueError(f"primary objectives not present in scored files: {unknown_primary}")
    if len(set(primary_objectives)) != len(primary_objectives):
        raise ValueError("primary objectives must be unique")
    complete = [r for r in records.values() if all(obj in r["scores"] for obj in objective_names)]
    if not complete:
        raise ValueError("No prompts had complete scores for all objectives.")

    model_names = complete[0]["response_model_names"]
    if args.baseline_model not in model_names:
        raise ValueError(f"baseline_model={args.baseline_model!r} not in {model_names}")
    baseline_idx = model_names.index(args.baseline_model)

    per_objective_rows: List[Dict[str, Any]] = []
    model_summary_rows: List[Dict[str, Any]] = []
    pairwise_rows: List[Dict[str, Any]] = []
    paired_difference_rows: List[Dict[str, Any]] = []

    objective_model_norms: Dict[str, Dict[str, List[float]]] = {
        obj: {model: [] for model in model_names} for obj in objective_names
    }
    objective_model_scores: Dict[str, Dict[str, List[float]]] = {
        obj: {model: [] for model in model_names} for obj in objective_names
    }
    prompt_avg_norms: Dict[str, List[float]] = {model: [] for model in model_names}
    prompt_worst_norms: Dict[str, List[float]] = {model: [] for model in model_names}
    primary_prompt_avg_norms: Dict[str, List[float]] = {model: [] for model in model_names}
    primary_prompt_worst_norms: Dict[str, List[float]] = {model: [] for model in model_names}
    objective_vs_baseline: Dict[str, Dict[str, List[float]]] = {
        obj: {model: [] for model in model_names if model != args.baseline_model} for obj in objective_names
    }

    for record in complete:
        per_prompt_norms = {}
        for obj in objective_names:
            scores = record["scores"][obj]
            norm_scores = minmax(scores)
            per_prompt_norms[obj] = norm_scores
            for i, model in enumerate(model_names):
                objective_model_scores[obj][model].append(scores[i])
                objective_model_norms[obj][model].append(norm_scores[i])
                if model != args.baseline_model:
                    delta = scores[i] - scores[baseline_idx]
                    if delta > args.tie_threshold:
                        win = 1.0
                    elif delta < -args.tie_threshold:
                        win = 0.0
                    else:
                        win = 0.5
                    objective_vs_baseline[obj][model].append(win)
        for i, model in enumerate(model_names):
            values = [per_prompt_norms[obj][i] for obj in objective_names]
            primary_values = [per_prompt_norms[obj][i] for obj in primary_objectives]
            prompt_avg_norms[model].append(mean(values))
            prompt_worst_norms[model].append(min(values))
            primary_prompt_avg_norms[model].append(mean(primary_values))
            primary_prompt_worst_norms[model].append(min(primary_values))

    num_prompts = len(complete)
    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap_indices = rng.integers(
        0,
        num_prompts,
        size=(args.bootstrap_samples, num_prompts),
        dtype=np.int32,
    )

    def bootstrap_means(values: List[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        return array[bootstrap_indices].mean(axis=1)

    for obj in objective_names:
        for model in model_names:
            row = {
                "objective": obj,
                "model": model,
                "mean_raw_score": mean(objective_model_scores[obj][model]),
                "mean_prompt_norm_score": mean(objective_model_norms[obj][model]),
            }
            if model != args.baseline_model:
                row["win_rate_vs_baseline"] = mean(objective_vs_baseline[obj][model])
            add_ci(row, "mean_raw_score", bootstrap_means(objective_model_scores[obj][model]))
            add_ci(row, "mean_prompt_norm_score", bootstrap_means(objective_model_norms[obj][model]))
            if model != args.baseline_model:
                add_ci(
                    row,
                    "win_rate_vs_baseline",
                    bootstrap_means(objective_vs_baseline[obj][model]),
                )
            per_objective_rows.append(row)

    for model in model_names:
        norm_by_obj = [mean(objective_model_norms[obj][model]) for obj in objective_names]
        row = {
            "model": model,
            "mean_objective_norm_score": mean(norm_by_obj),
            "min_objective_norm_score": min(norm_by_obj),
            "mean_prompt_avg_norm_score": mean(prompt_avg_norms[model]),
            "mean_prompt_worst_norm_score": mean(prompt_worst_norms[model]),
            "mean_primary_prompt_avg_norm_score": mean(primary_prompt_avg_norms[model]),
            "mean_primary_prompt_worst_norm_score": mean(primary_prompt_worst_norms[model]),
            "std_objective_norm_score": float((sum((x - mean(norm_by_obj)) ** 2 for x in norm_by_obj) / len(norm_by_obj)) ** 0.5),
            "num_prompts": len(complete),
        }
        objective_norm_bootstrap = np.stack(
            [bootstrap_means(objective_model_norms[obj][model]) for obj in objective_names]
        )
        add_ci(row, "mean_objective_norm_score", objective_norm_bootstrap.mean(axis=0))
        add_ci(row, "min_objective_norm_score", objective_norm_bootstrap.min(axis=0))
        add_ci(row, "std_objective_norm_score", objective_norm_bootstrap.std(axis=0))
        add_ci(row, "mean_prompt_avg_norm_score", bootstrap_means(prompt_avg_norms[model]))
        add_ci(row, "mean_prompt_worst_norm_score", bootstrap_means(prompt_worst_norms[model]))
        add_ci(
            row,
            "mean_primary_prompt_avg_norm_score",
            bootstrap_means(primary_prompt_avg_norms[model]),
        )
        add_ci(
            row,
            "mean_primary_prompt_worst_norm_score",
            bootstrap_means(primary_prompt_worst_norms[model]),
        )
        if model != args.baseline_model:
            wins = [mean(objective_vs_baseline[obj][model]) for obj in objective_names]
            row["mean_win_rate_vs_baseline"] = mean(wins)
            row["min_win_rate_vs_baseline"] = min(wins)
            win_bootstrap = np.stack(
                [bootstrap_means(objective_vs_baseline[obj][model]) for obj in objective_names]
            )
            add_ci(row, "mean_win_rate_vs_baseline", win_bootstrap.mean(axis=0))
            add_ci(row, "min_win_rate_vs_baseline", win_bootstrap.min(axis=0))
        model_summary_rows.append(row)

    for left_idx, left in enumerate(model_names):
        for right_idx, right in enumerate(model_names):
            if left_idx >= right_idx:
                continue
            difference_row: Dict[str, Any] = {
                "left_model": left,
                "right_model": right,
            }
            for metric, values_by_model in (
                ("mean_prompt_worst_norm_score", prompt_worst_norms),
                ("mean_prompt_avg_norm_score", prompt_avg_norms),
                ("mean_primary_prompt_worst_norm_score", primary_prompt_worst_norms),
                ("mean_primary_prompt_avg_norm_score", primary_prompt_avg_norms),
            ):
                left_values = np.asarray(values_by_model[left], dtype=np.float64)
                right_values = np.asarray(values_by_model[right], dtype=np.float64)
                differences = left_values - right_values
                difference_row[f"{metric}_difference"] = float(differences.mean())
                add_ci(
                    difference_row,
                    f"{metric}_difference",
                    differences[bootstrap_indices].mean(axis=1),
                )
            paired_difference_rows.append(difference_row)
            for obj in objective_names:
                wins = []
                for record in complete:
                    scores = record["scores"][obj]
                    delta = scores[left_idx] - scores[right_idx]
                    if delta > args.tie_threshold:
                        wins.append(1.0)
                    elif delta < -args.tie_threshold:
                        wins.append(0.0)
                    else:
                        wins.append(0.5)
                pairwise_rows.append(
                    {
                        "objective": obj,
                        "left_model": left,
                        "right_model": right,
                        "left_win_rate": mean(wins),
                    }
                )

    outdir = Path(args.output_dir)
    write_csv(outdir / "per_objective_scores.csv", per_objective_rows)
    write_csv(outdir / "model_summary.csv", model_summary_rows)
    write_csv(outdir / "pairwise_win_rates.csv", pairwise_rows)
    write_csv(outdir / "paired_model_differences.csv", paired_difference_rows)
    with (outdir / "model_summary.json").open("w", encoding="utf-8") as f:
        json.dump(model_summary_rows, f, ensure_ascii=False, indent=2)
    with (outdir / "paired_model_differences.json").open("w", encoding="utf-8") as f:
        json.dump(paired_difference_rows, f, ensure_ascii=False, indent=2)
    with (outdir / "evaluation_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_prompts": num_prompts,
                "objectives": objective_names,
                "primary_objectives": primary_objectives,
                "baseline_model": args.baseline_model,
                "tie_threshold": args.tie_threshold,
                "bootstrap": {
                    "method": "paired_prompt_resampling_with_replacement",
                    "samples": args.bootstrap_samples,
                    "seed": args.bootstrap_seed,
                    "interval": "percentile_95",
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Evaluated {len(complete)} prompts.")
    print(f"Wrote summaries to {outdir}")


if __name__ == "__main__":
    main()
