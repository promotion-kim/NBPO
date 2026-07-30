#!/usr/bin/env python3
"""Aggregate raw-reward power and normalized validation results for stronger 8B models."""

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


def interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def read_score(path: Path) -> tuple[list[str], np.ndarray]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 128:
        raise RuntimeError(f"expected 128 score rows in {path}, found {len(rows)}")
    models = rows[0].get("response_model_names")
    if not isinstance(models, list) or not models or models[0] != "base":
        raise RuntimeError(f"invalid model order in {path}")
    scores = []
    for row in rows:
        if row.get("response_model_names") != models:
            raise RuntimeError(f"model order changes within {path}")
        values = [float(value) for value in row.get("all_rm_scores", [])]
        if len(values) != len(models) or not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"invalid raw scores in {path}")
        scores.append(values)
    return models, np.asarray(scores, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", nargs="+", required=True,
                        help="evaluator=JSONL raw-score paths")
    parser.add_argument("--normalized-summary", type=Path, required=True)
    parser.add_argument("--gates-summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("status") != "frozen_before_training":
        raise RuntimeError("stronger-training protocol is not frozen")
    scored = {}
    for item in args.scored:
        name, value = item.split("=", 1)
        scored[name] = Path(value)
    expected = {"armo_helpfulness", "armo_safety", "armo_conciseness", "skywork", "athene"}
    if set(scored) != expected:
        raise RuntimeError(f"evaluator set mismatch: {sorted(scored)}")
    models = None
    matrices = {}
    for evaluator, path in scored.items():
        current_models, matrix = read_score(path)
        if models is None:
            models = current_models
        elif current_models != models:
            raise RuntimeError("model sets differ across evaluators")
        matrices[evaluator] = matrix
    assert models is not None
    base = models.index("base")
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, 128, size=(args.bootstrap, 128))
    model_rows = []
    pair_rows = []
    evaluator_summary = {}
    for evaluator, matrix in matrices.items():
        boot_means = matrix[indices, :].mean(axis=1)
        means = matrix.mean(axis=0)
        rows = []
        for model_index, model in enumerate(models):
            mean_low, mean_high = interval(boot_means[:, model_index])
            delta = matrix[:, model_index] - matrix[:, base]
            boot_delta = delta[indices].mean(axis=1)
            delta_low, delta_high = interval(boot_delta)
            wins = np.where(matrix[:, model_index] > matrix[:, base], 1.0,
                            np.where(matrix[:, model_index] == matrix[:, base], 0.5, 0.0))
            boot_wins = wins[indices].mean(axis=1)
            win_low, win_high = interval(boot_wins)
            detectable = model != "base" and (delta_low > 0.0 or delta_high < 0.0)
            row = {
                "evaluator": evaluator, "model": model,
                "mean_raw_score": float(means[model_index]),
                "mean_raw_score_ci95_low": mean_low, "mean_raw_score_ci95_high": mean_high,
                "mean_raw_delta_vs_base": float(delta.mean()),
                "mean_raw_delta_vs_base_ci95_low": delta_low,
                "mean_raw_delta_vs_base_ci95_high": delta_high,
                "win_rate_vs_base": float(wins.mean()),
                "win_rate_vs_base_ci95_low": win_low, "win_rate_vs_base_ci95_high": win_high,
                "detectably_different_from_base": detectable,
            }
            rows.append(row)
            model_rows.append(row)
        pair_detectable = []
        for left_index, left in enumerate(models):
            for right_index in range(left_index + 1, len(models)):
                right = models[right_index]
                delta = matrix[:, left_index] - matrix[:, right_index]
                boot_delta = delta[indices].mean(axis=1)
                low, high = interval(boot_delta)
                result = {
                    "evaluator": evaluator, "left_model": left, "right_model": right,
                    "mean_raw_delta_left_minus_right": float(delta.mean()),
                    "ci95_low": low, "ci95_high": high,
                    "detectably_different": low > 0.0 or high < 0.0,
                }
                pair_rows.append(result)
                if result["detectably_different"]:
                    pair_detectable.append(result)
        spread_boot = boot_means.max(axis=1) - boot_means.min(axis=1)
        spread_low, spread_high = interval(spread_boot)
        evaluator_summary[evaluator] = {
            "model_mean_spread": float(means.max() - means.min()),
            "model_mean_spread_ci95_low": spread_low,
            "model_mean_spread_ci95_high": spread_high,
            "detectable_models_vs_base": [row["model"] for row in rows
                                           if row["detectably_different_from_base"]],
            "detectable_pairs": pair_detectable,
        }

    normalized = json.loads(args.normalized_summary.read_text())
    ranked = normalized.get("ranked", [])
    if {row.get("model") for row in ranked} != set(models):
        raise RuntimeError("normalized summary model set differs from raw scores")
    armo_evaluators = [name for name in matrices if name.startswith("armo_")]
    armo_detectable = any(evaluator_summary[name]["detectable_models_vs_base"]
                          for name in armo_evaluators)
    any_detectable = any(value["detectable_models_vs_base"]
                         for value in evaluator_summary.values())
    detectable_raw_deltas = [
        row for row in model_rows if row["detectably_different_from_base"]
    ]
    leader = ranked[0]["model"] if ranked else None
    ronpo_leads = leader in {"ronpo_full_expect", "ronpo_k_only"}
    normalized_interpretable = bool(armo_detectable)
    if armo_detectable and ronpo_leads:
        verdict = (
            f"Stronger training creates a detectable raw ArmoRM-head shift and {leader} has the "
            "highest validation worst-objective point estimate. This is a genuine validation-split "
            "signal, not a sealed result, and requires confirmation on the pre-registered unopened split."
        )
    elif armo_detectable:
        details = "; ".join(
            f"{row['model']} on {row['evaluator']} "
            f"({row['mean_raw_delta_vs_base']:+.6f}, 95% CI "
            f"[{row['mean_raw_delta_vs_base_ci95_low']:+.6f}, "
            f"{row['mean_raw_delta_vs_base_ci95_high']:+.6f}])"
            for row in detectable_raw_deltas if row["evaluator"].startswith("armo_")
        )
        verdict = (
            "Stronger training creates a detectable raw ArmoRM-head shift, but it does not support "
            f"the robustness claim: {details}. The normalized worst-objective leader is {leader}, "
            "not a RONPO estimator."
        )
    elif any_detectable:
        verdict = (
            "Stronger training moves at least one committed general-reward score relative to base, "
            "but none of the three conflict-objective ArmoRM heads separates a retrained model from base. "
            "Policy movement is measurable, while heterogeneous-objective robustness remains underpowered."
        )
    else:
        verdict = (
            "Stronger training still produces no paired raw-reward difference from base under any of "
            "the five evaluators. The Qwen3-8B heterogeneous-objective comparison remains underpowered."
        )
    gates = json.loads(args.gates_summary.read_text())
    fresh_confirmation_signal = bool(armo_detectable and ronpo_leads)
    summary = {
        "status": "completed", "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "split": "non-sealed 128-prompt validation", "spent_sealed_split_touched": False,
        "models": models, "bootstrap_resamples": args.bootstrap, "bootstrap_seed": args.seed,
        "config_sha256": sha256(args.config),
        "scored_file_sha256": {name: sha256(path) for name, path in scored.items()},
        "gates": gates, "evaluators": evaluator_summary,
        "any_detectable_raw_delta_vs_base": bool(any_detectable),
        "armo_raw_delta_detectable": bool(armo_detectable),
        "detectable_raw_deltas_vs_base": detectable_raw_deltas,
        "normalized_worst_objective_interpretable": normalized_interpretable,
        "normalized_worst_objective_leader": leader,
        "ronpo_leads_normalized_worst_objective": ronpo_leads,
        "requires_fresh_sealed_preregistration": fresh_confirmation_signal,
        "fresh_sealed_preregistration_rule": (
            "Pre-register a fresh unopened confirmation split only when at least one ArmoRM "
            "objective has a paired raw delta versus base whose 95% interval excludes zero "
            "and a RONPO estimator leads the normalized validation worst objective."
        ),
        "normalized_ranking": ranked,
        "plain_verdict": verdict,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "raw_power_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "per_evaluator_model_raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(model_rows[0]))
        writer.writeheader(); writer.writerows(model_rows)
    with (args.output_dir / "pairwise_raw_differences.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader(); writer.writerows(pair_rows)

    lines = ["# Stronger Qwen3-8B validation power check", "",
             "This experiment uses only the existing non-sealed 128-prompt validation split. The spent sealed split was not decoded, rescored, or used for selection.", "",
             "## Stability gates", "",
             "| Model | Result |", "|---|---|"]
    for model in gates["models"]:
        outcome = gates["models"].get(model, {})
        if outcome.get("passed"):
            result = "PASS"
        else:
            gate_path = Path(outcome["artifact"])
            if not gate_path.is_file():
                gate_path = args.gates_summary.parent / f"{model}.json"
            gate_payload = json.loads(gate_path.read_text())
            result = f"FAIL (max repeat run {gate_payload['candidate']['max_repeat_run']})"
        lines.append(f"| {model} | {result} |")
    lines.extend(["", "## Raw-reward power", ""])
    for evaluator in sorted(evaluator_summary):
        result = evaluator_summary[evaluator]
        detectable = ", ".join(result["detectable_models_vs_base"]) or "none"
        lines.append(f"- `{evaluator}`: mean spread `{result['model_mean_spread']:.6f}`; detectable vs base: {detectable}.")
    lines.extend(["", "## Normalized validation ranking", "",
                  f"Interpretability gate: `{'PASS' if normalized_interpretable else 'FAIL'}`. "
                  "The ranking is descriptive only when this gate fails.", "",
                  "| Rank | Model | Worst objective (95% CI) |", "|---:|---|---:|"])
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {row['model']} | {float(row['mean_primary_prompt_worst_norm_score']):.4f} "
            f"[{float(row['mean_primary_prompt_worst_norm_score_ci95_low']):.4f}, "
            f"{float(row['mean_primary_prompt_worst_norm_score_ci95_high']):.4f}] |"
        )
    lines.extend(["", "## Verdict", "", verdict, ""])
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    if args.wandb:
        import wandb
        run = wandb.init(entity="promotion-kim", project="mnpo",
                         id="qwen3-8b-stronger-validation-power-20260715", resume="allow",
                         name="qwen3-8b-stronger-validation-power-20260715",
                         config={"config_sha256": summary["config_sha256"], "split": summary["split"],
                                 "bootstrap_resamples": args.bootstrap, "spent_sealed_split_touched": False})
        payload = {"power/any_raw_delta_vs_base": int(any_detectable),
                   "power/armo_raw_delta_detectable": int(armo_detectable),
                   "power/normalized_interpretable": int(normalized_interpretable)}
        for evaluator, result in evaluator_summary.items():
            payload[f"{evaluator}/mean_spread"] = result["model_mean_spread"]
            payload[f"{evaluator}/detectable_vs_base"] = len(result["detectable_models_vs_base"])
        run.log(payload); run.summary.update(payload); run.finish()


if __name__ == "__main__":
    main()
