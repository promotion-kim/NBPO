#!/usr/bin/env python3
"""Aggregate the frozen sealed all-pairwise judge with prompt bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bootstrap(values: np.ndarray, resamples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)), dtype=np.int32)
    samples = values[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def outcome(row: dict) -> dict[str, float]:
    label = str(row["parsed"])
    a, b = row["assistant_a_model"], row["assistant_b_model"]
    if label == "A=B":
        return {a: 0.5, b: 0.5}
    winner = a if label.startswith("A>") else b
    loser = b if winner == a else a
    return {winner: 1.0, loser: 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--judgment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol_lock.read_text())
    rows = []
    shard_metadata = []
    for index in range(protocol["randomization"]["num_shards"]):
        path = args.judgment_dir / f"shard_{index}.jsonl"
        metadata_path = args.judgment_dir / f"shard_{index}.metadata.json"
        rows.extend(load_jsonl(path))
        shard_metadata.append(json.loads(metadata_path.read_text()))
    if len(rows) != protocol["expected_judgments"]:
        raise RuntimeError(f"expected {protocol['expected_judgments']} judgments, got {len(rows)}")
    if len({row["task_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate task IDs")
    if any(not row.get("valid") for row in rows):
        raise RuntimeError("fail closed: invalid judgment present")
    if any(row.get("protocol_sha256") != protocol["configuration_sha256"] for row in rows):
        raise RuntimeError("protocol hash mismatch in judgments")

    per_prompt_model: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        scores = outcome(row)
        prompt_index = int(row["prompt_index"])
        for model, score in scores.items():
            per_prompt_model[prompt_index][model].append(score)
        left, right = row["pair_left"], row["pair_right"]
        per_pair[(left, right)].append(scores[left])
    models = protocol["models"]
    for prompt_index in range(protocol["sealed_prompt_count"]):
        for model in models:
            if len(per_prompt_model[prompt_index][model]) != len(models) - 1:
                raise RuntimeError(f"incomplete opponents for prompt {prompt_index}, model {model}")

    summaries = []
    prompt_vectors = {}
    for model in models:
        values = np.asarray([
            np.mean(per_prompt_model[prompt_index][model])
            for prompt_index in range(protocol["sealed_prompt_count"])
        ], dtype=float)
        prompt_vectors[model] = values
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"non-finite pairwise score for {model}")
        low, high = bootstrap(
            values,
            protocol["aggregation"]["bootstrap_resamples"],
            protocol["aggregation"]["bootstrap_seed"],
        )
        base_values = None
        if model != "base":
            left, right = ("base", model) if models.index("base") < models.index(model) else (model, "base")
            left_scores = np.asarray(per_pair[(left, right)], dtype=float)
            base_values = 1.0 - left_scores if model == right else left_scores
        summaries.append({
            "model": model,
            "mean_pairwise_score": float(values.mean()),
            "ci95_low": low,
            "ci95_high": high,
            "win_rate_vs_base": None if base_values is None else float(base_values.mean()),
            "num_prompts": len(values),
            "opponents_per_prompt": len(models) - 1,
        })
    ranked = sorted(summaries, key=lambda row: (-row["mean_pairwise_score"], row["model"]))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank

    pair_rows = []
    for left, right in combinations(models, 2):
        values = np.asarray(per_pair[(left, right)], dtype=float)
        low, high = bootstrap(
            values,
            protocol["aggregation"]["bootstrap_resamples"],
            protocol["aggregation"]["bootstrap_seed"],
        )
        pair_rows.append({
            "left_model": left,
            "right_model": right,
            "left_score": float(values.mean()),
            "left_ci95_low": low,
            "left_ci95_high": high,
            "num_prompts": len(values),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked[0]))
        writer.writeheader(); writer.writerows(ranked)
    with (args.output_dir / "pairwise_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader(); writer.writerows(pair_rows)

    import wandb
    run_id = "sealed-pairwise-gptoss120b-aggregate-20260714"
    run = wandb.init(
        entity="promotion-kim", project="mnpo", id=run_id, resume="allow",
        name="sealed-pairwise-gptoss120b-aggregate",
        group="aaai27-qwen3-sealed-pairwise",
        config={"protocol": protocol},
    )
    metrics = {}
    for row in ranked:
        prefix = row["model"]
        metrics[f"{prefix}/mean_pairwise_score"] = row["mean_pairwise_score"]
        metrics[f"{prefix}/ci95_low"] = row["ci95_low"]
        metrics[f"{prefix}/ci95_high"] = row["ci95_high"]
        metrics[f"{prefix}/rank"] = row["rank"]
    run.log(metrics); run.summary.update(metrics); wandb_url = run.url; run.finish()
    payload = {
        "status": "completed",
        "official_score": False,
        "description": "Open-weight RM-independent all-pairwise judgment of the preserved sealed generations",
        "protocol_sha256": protocol["configuration_sha256"],
        "judge": protocol["judge"],
        "num_judgments": len(rows),
        "bootstrap": protocol["aggregation"],
        "ranked": ranked,
        "shards": shard_metadata,
        "wandb_run_id": run_id,
        "wandb_url": wandb_url,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Sealed all-pairwise gpt-oss-120b judge",
        "",
        "This is an open-weight, reward-model-independent diagnostic, not an official closed-judge benchmark score.",
        "",
        "| Rank | Model | Pairwise score (95% CI) | Win vs base |",
        "|---:|---|---:|---:|",
    ]
    for row in ranked:
        base_text = "--" if row["win_rate_vs_base"] is None else f"{100 * row['win_rate_vs_base']:.2f}%"
        lines.append(
            f"| {row['rank']} | {row['model']} | {100 * row['mean_pairwise_score']:.2f}% "
            f"[{100 * row['ci95_low']:.2f}, {100 * row['ci95_high']:.2f}] | {base_text} |"
        )
    lines.extend([
        "",
        f"Judgments: {len(rows):,}; 604 prompts; nine opponents per model and prompt; ties count as 0.5.",
        f"W&B: {wandb_url}",
    ])
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    if any(not math.isfinite(value) for row in ranked for value in (row["mean_pairwise_score"], row["ci95_low"], row["ci95_high"])):
        raise RuntimeError("non-finite aggregate")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
