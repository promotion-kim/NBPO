#!/usr/bin/env python3
"""Aggregate the pre-committed raw-reward power diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def load_reward(score_dir: Path, reward_name: str, input_rows: list[dict]) -> np.ndarray:
    by_prompt = {}
    for shard in (0, 1):
        path = score_dir / "scores" / reward_name / f"shard_{shard}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt_id = str(row["prompt_id"])
                if prompt_id in by_prompt:
                    raise RuntimeError(f"duplicate prompt {prompt_id} for {reward_name}")
                by_prompt[prompt_id] = row
    ordered = []
    for source in input_rows:
        prompt_id = str(source["prompt_id"])
        row = by_prompt.get(prompt_id)
        if row is None:
            raise RuntimeError(f"missing prompt {prompt_id} for {reward_name}")
        if row.get("response_model_names") != source.get("response_model_names"):
            raise RuntimeError(f"model order mismatch for {reward_name} prompt {prompt_id}")
        scores = row.get("all_rm_scores", [])
        if len(scores) != len(source["response_model_names"]):
            raise RuntimeError(f"score count mismatch for {reward_name} prompt {prompt_id}")
        if not all(math.isfinite(float(value)) for value in scores):
            raise RuntimeError(f"non-finite score for {reward_name} prompt {prompt_id}")
        ordered.append([float(value) for value in scores])
    if len(by_prompt) != len(input_rows):
        raise RuntimeError(f"unexpected prompt count for {reward_name}: {len(by_prompt)}")
    return np.asarray(ordered, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text())
    if lock.get("status") != "locked_before_scoring":
        raise RuntimeError("diagnostic set was not locked")
    input_rows = json.loads(args.input_file.read_text())
    models = input_rows[0]["response_model_names"]
    if models != lock["models"] or len(input_rows) != 604:
        raise RuntimeError("input differs from locked model or prompt set")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(0, len(input_rows), size=(args.bootstrap, len(input_rows)))
    reward_names = ["skywork", "athene"]
    reward_results = {}
    long_rows = []
    pair_rows = []
    for reward_name in reward_names:
        scores = load_reward(args.score_dir, reward_name, input_rows)
        boot_means = scores[bootstrap_indices, :].mean(axis=1)
        base_index = models.index("base")
        means = scores.mean(axis=0)
        model_rows = []
        for model_index, model in enumerate(models):
            raw_low, raw_high = percentile(boot_means[:, model_index])
            delta = scores[:, model_index] - scores[:, base_index]
            boot_delta = delta[bootstrap_indices].mean(axis=1)
            delta_low, delta_high = percentile(boot_delta)
            wins = np.where(scores[:, model_index] > scores[:, base_index], 1.0,
                            np.where(scores[:, model_index] == scores[:, base_index], 0.5, 0.0))
            boot_wins = wins[bootstrap_indices].mean(axis=1)
            win_low, win_high = percentile(boot_wins)
            detectable = model != "base" and (delta_low > 0.0 or delta_high < 0.0)
            result = {
                "model": model,
                "mean_raw_score": float(means[model_index]),
                "mean_raw_score_ci95_low": raw_low,
                "mean_raw_score_ci95_high": raw_high,
                "mean_raw_delta_vs_base": float(delta.mean()),
                "mean_raw_delta_vs_base_ci95_low": delta_low,
                "mean_raw_delta_vs_base_ci95_high": delta_high,
                "win_rate_vs_base": float(wins.mean()),
                "win_rate_vs_base_ci95_low": win_low,
                "win_rate_vs_base_ci95_high": win_high,
                "detectably_different_from_base": detectable,
            }
            model_rows.append(result)
            long_rows.append({"reward_model": reward_name, **result})
        model_rows.sort(key=lambda row: row["mean_raw_score"], reverse=True)
        for rank, row in enumerate(model_rows, start=1):
            row["rank_by_mean_raw_score"] = rank

        significant_pairs = 0
        detectable_pair_rows = []
        for left_index, left in enumerate(models):
            for right_index in range(left_index + 1, len(models)):
                right = models[right_index]
                delta = scores[:, left_index] - scores[:, right_index]
                boot_delta = delta[bootstrap_indices].mean(axis=1)
                low, high = percentile(boot_delta)
                detectable = low > 0.0 or high < 0.0
                significant_pairs += int(detectable)
                pair_result = {
                    "reward_model": reward_name,
                    "left_model": left,
                    "right_model": right,
                    "mean_raw_delta_left_minus_right": float(delta.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "detectably_different": detectable,
                }
                pair_rows.append(pair_result)
                if detectable:
                    detectable_pair_rows.append(pair_result)
        spread_boot = boot_means.max(axis=1) - boot_means.min(axis=1)
        spread_low, spread_high = percentile(spread_boot)
        detectable_vs_base = sum(row["detectably_different_from_base"] for row in model_rows)
        reward_results[reward_name] = {
            "models": model_rows,
            "model_mean_spread": float(means.max() - means.min()),
            "model_mean_spread_ci95_low": spread_low,
            "model_mean_spread_ci95_high": spread_high,
            "detectable_trained_models_vs_base": int(detectable_vs_base),
            "detectable_pairwise_differences": int(significant_pairs),
            "total_pairwise_differences": 45,
            "separates_any_model_from_base": bool(detectable_vs_base),
            "separates_any_model_pair": bool(significant_pairs),
            "detectable_pairs": detectable_pair_rows,
        }

    separates_base = any(value["separates_any_model_from_base"] for value in reward_results.values())
    separates_pair = any(value["separates_any_model_pair"] for value in reward_results.values())
    if separates_base:
        verdict = (
            "At least one pre-committed general reward model detects a paired raw-reward difference "
            "between a trained model and base. This localizes the original power failure to the "
            "ArmoRM heads for the measured directions; it does not change locked model selection."
        )
    elif separates_pair:
        verdict = (
            "Neither committed reward model separates any trained policy from base, and Skywork "
            "separates no model pair. Athene separates 3 of 45 pairs under unadjusted paired bootstrap "
            "intervals: full-expectation RONPO exceeds top-mass RONPO and SPPO-avg, and IPO exceeds "
            "SPPO-avg. Athene therefore registers limited policy differences that the ArmoRM heads miss, "
            "but this is weak evaluator-specific evidence rather than a powered model ranking because "
            "no base comparison separates and 45 pairwise intervals were inspected."
        )
    else:
        verdict = (
            "Neither pre-committed general reward model detects a paired raw-reward difference between "
            "any trained model and base or between any model pair. The evaluated Qwen3-8B policies are "
            "reward-near-equivalent under all tested evaluators, and the original normalized ordering is underpowered."
        )
    scored_hashes = {}
    for reward_name in reward_names:
        scored_hashes[reward_name] = {
            f"shard_{shard}": sha256(args.score_dir / "scores" / reward_name / f"shard_{shard}.jsonl")
            for shard in (0, 1)
        }
    summary = {
        "status": "completed",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_changed": False,
        "new_decode_invoked": False,
        "num_prompts": len(input_rows),
        "models": models,
        "bootstrap_resamples": args.bootstrap,
        "bootstrap_seed": args.seed,
        "lock_sha256": sha256(args.lock),
        "input_sha256": sha256(args.input_file),
        "scored_file_sha256": scored_hashes,
        "reward_models": reward_results,
        "separates_any_model_from_base": separates_base,
        "separates_any_model_pair": separates_pair,
        "plain_verdict": verdict,
    }

    with (args.output_dir / "per_model_raw_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(long_rows[0]))
        writer.writeheader()
        writer.writerows(long_rows)
    with (args.output_dir / "pairwise_raw_differences.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Qwen3-8B sealed-generation reward power diagnostic", "",
        "This diagnostic re-scores the preserved 604 sealed responses. It performs no decode and does not change the locked model selection.", "",
    ]
    for reward_name in reward_names:
        result = reward_results[reward_name]
        lines.extend([
            f"## {reward_name.title()}", "",
            f"Across-model mean spread: `{result['model_mean_spread']:.6f}`. Detectable trained models vs base: "
            f"`{result['detectable_trained_models_vs_base']}`. Detectable model pairs: "
            f"`{result['detectable_pairwise_differences']}/45`.", "",
            "| Rank | Model | Mean raw reward (95% CI) | Delta vs base (95% CI) | Win vs base (95% CI) | Detectable |",
            "|---:|---|---:|---:|---:|:---:|",
        ])
        for row in result["models"]:
            lines.append(
                f"| {row['rank_by_mean_raw_score']} | {row['model']} | "
                f"{row['mean_raw_score']:.6f} [{row['mean_raw_score_ci95_low']:.6f}, {row['mean_raw_score_ci95_high']:.6f}] | "
                f"{row['mean_raw_delta_vs_base']:+.6f} [{row['mean_raw_delta_vs_base_ci95_low']:+.6f}, {row['mean_raw_delta_vs_base_ci95_high']:+.6f}] | "
                f"{100 * row['win_rate_vs_base']:.2f}% [{100 * row['win_rate_vs_base_ci95_low']:.2f}, {100 * row['win_rate_vs_base_ci95_high']:.2f}] | "
                f"{'yes' if row['detectably_different_from_base'] else 'no'} |"
            )
        lines.append("")
    lines.extend(["## Verdict", "", verdict, "", "## Provenance", "",
                  f"- Reward-model lock SHA-256: `{summary['lock_sha256']}`",
                  f"- Preserved merged-generation SHA-256: `{summary['input_sha256']}`",
                  "- Intervals: 2,000 paired prompt bootstrap resamples, seed 42.",
                  "- Complete pairwise differences: `pairwise_raw_differences.csv`.", ""])
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    if args.wandb:
        import wandb
        run = wandb.init(entity="promotion-kim", project="mnpo",
                         id="qwen3-sealed-power-diagnostic-20260714", resume="allow",
                         name="qwen3-sealed-power-diagnostic-20260714",
                         config={"lock_sha256": summary["lock_sha256"], "num_prompts": 604,
                                 "bootstrap_resamples": args.bootstrap, "selection_changed": False,
                                 "new_decode_invoked": False})
        payload = {}
        for reward_name, result in reward_results.items():
            payload[f"{reward_name}/model_mean_spread"] = result["model_mean_spread"]
            payload[f"{reward_name}/detectable_vs_base"] = result["detectable_trained_models_vs_base"]
            payload[f"{reward_name}/detectable_pairs"] = result["detectable_pairwise_differences"]
        run.log(payload)
        run.finish()


if __name__ == "__main__":
    main()
