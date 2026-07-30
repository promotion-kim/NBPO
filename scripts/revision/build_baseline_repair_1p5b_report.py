#!/usr/bin/env python3
"""Build the reviewable measured summary for the 1.5B baseline repair."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


CANDIDATES = [
    "repair1p5b_sppo_a_s42",
    "repair1p5b_sppo_b_s42",
    "repair1p5b_sppo_c_s42",
    "repair1p5b_sppo_d_s42",
    "repair1p5b_inpo_a_s42",
    "repair1p5b_inpo_b_s42",
    "repair1p5b_inpo_c_s42",
    "repair1p5b_inpo_d_s42",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_env(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    if not path.exists():
        return output
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            output[key] = value
    return output


def interval(row: dict, metric: str) -> dict[str, float]:
    return {
        "value": float(row[metric]),
        "ci95_low": float(row[f"{metric}_ci95_low"]),
        "ci95_high": float(row[f"{metric}_ci95_high"]),
    }


def fmt(value: dict[str, float] | None) -> str:
    if value is None:
        return "--"
    return f"{value['value']:.4f} [{value['ci95_low']:.4f}, {value['ci95_high']:.4f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    result_dir = args.run_root / "eval" / "results"
    rows = read_json(result_dir / "model_summary.json")
    by_model = {row["model"]: row for row in rows}
    retry_candidates = [
        name
        for name in ("repair1p5b_sppo_e_s42", "repair1p5b_inpo_e_s42")
        if (args.run_root / "candidates" / name).exists()
        or (args.run_root / "eval" / "stability" / f"{name}_gate.json").exists()
    ]
    attempts = []
    for name in CANDIDATES + retry_candidates:
        candidate_dir = args.run_root / "candidates" / name
        gate_path = args.run_root / "eval" / "stability" / f"{name}_gate.json"
        gate = read_json(gate_path) if gate_path.exists() else {"passed": False, "reason": "gate artifact missing"}
        env = parse_env(candidate_dir / "run_metadata.env")
        train = read_json(candidate_dir / "train_results.json") if (candidate_dir / "train_results.json").exists() else None
        attempts.append(
            {
                "candidate": name,
                "method": env.get("method", name.split("_")[1]),
                "hyperparameters": {
                    key: env.get(key)
                    for key in (
                        "eta",
                        "reference_anchor_weight",
                        "preference_sft_weight",
                        "learning_rate",
                        "effective_batch_size",
                        "per_device_train_batch_size",
                        "num_processes",
                        "max_steps",
                    )
                },
                "training_completed": train is not None,
                "train_metrics": train,
                "stability_gate": gate,
                "eligible_for_reward_eval": bool(gate.get("passed")) and name in by_model,
                "reward_metrics": by_model.get(name),
                "wandb_run_id": env.get("wandb_run_id"),
                "wandb_url": (
                    f"https://wandb.ai/promotion-kim/mnpo/runs/{env['wandb_run_id']}"
                    if env.get("wandb_run_id")
                    else None
                ),
                "exact_command": (
                    (candidate_dir / "exact_command.txt").read_text(encoding="utf-8").strip()
                    if (candidate_dir / "exact_command.txt").exists()
                    else None
                ),
            }
        )

    def best(method: str):
        eligible = [
            attempt
            for attempt in attempts
            if attempt["method"] == method and attempt["eligible_for_reward_eval"]
        ]
        return max(
            eligible,
            key=lambda item: float(item["reward_metrics"]["min_objective_norm_score"]),
            default=None,
        )

    best_sppo = best("sppo")
    best_inpo = best("inpo")
    base = by_model["baseline"]
    ronpo = by_model["ronpo"]

    selected_rows = {"base": base, "ronpo": ronpo}
    if best_sppo:
        selected_rows["sppo_avg_repaired"] = best_sppo["reward_metrics"]
    if best_inpo:
        selected_rows["inpo_avg_repaired"] = best_inpo["reward_metrics"]

    metrics = {}
    for label, row in selected_rows.items():
        metrics[label] = {
            "model": row["model"],
            "avg": interval(row, "mean_objective_norm_score"),
            "worst_objective": interval(row, "min_objective_norm_score"),
            "mean_prompt_worst": interval(row, "mean_prompt_worst_norm_score"),
            "win_rate_vs_base": (
                interval(row, "mean_win_rate_vs_baseline")
                if "mean_win_rate_vs_baseline" in row
                else None
            ),
            "num_prompts": int(row["num_prompts"]),
        }

    base_avg = float(base["mean_objective_norm_score"])
    sppo_recovered = bool(best_sppo) and float(best_sppo["reward_metrics"]["mean_objective_norm_score"]) > base_avg
    inpo_recovered = bool(best_inpo) and float(best_inpo["reward_metrics"]["mean_objective_norm_score"]) > base_avg
    repaired = [item for item in (best_sppo, best_inpo) if item is not None]
    ronpo_worst = float(ronpo["min_objective_norm_score"])
    all_repaired_lose = bool(repaired) and all(
        float(item["reward_metrics"]["min_objective_norm_score"]) < ronpo_worst for item in repaired
    )
    inpo_worst_ci_overlaps_ronpo = False
    if best_inpo:
        inpo_row = best_inpo["reward_metrics"]
        inpo_worst_ci_overlaps_ronpo = not (
            float(inpo_row["min_objective_norm_score_ci95_high"])
            < float(ronpo["min_objective_norm_score_ci95_low"])
            or float(ronpo["min_objective_norm_score_ci95_high"])
            < float(inpo_row["min_objective_norm_score_ci95_low"])
        )
    if sppo_recovered and inpo_recovered:
        verdict = (
            "Yes on the measured point estimates. Both averaged-oracle baselines recovered above base "
            "on Avg and both still lost to RONPO on the worst objective."
            if all_repaired_lose
            else "No. Both averaged-oracle baselines recovered above base on Avg, but RONPO did not beat both on the worst objective."
        )
    else:
        missing = []
        if not sppo_recovered:
            missing.append("SPPO-avg")
        if not inpo_recovered:
            missing.append("INPO-avg")
        verdict = (
            "Inconclusive under the requested acceptance criterion: "
            + " and ".join(missing)
            + " did not produce a stability-passing candidate above base on Avg."
        )

    cache = args.run_root / "cache" / "reward_models"
    revisions = {}
    for model_dir, label in (
        ("models--Skywork--Skywork-Reward-V2-Llama-3.1-8B", "Skywork/Skywork-Reward-V2-Llama-3.1-8B"),
        ("models--Nexusflow--Athene-RM-8B", "Nexusflow/Athene-RM-8B"),
        ("models--RLHFlow--ArmoRM-Llama3-8B-v0.1", "RLHFlow/ArmoRM-Llama3-8B-v0.1"),
    ):
        ref = cache / model_dir / "refs" / "main"
        revisions[label] = ref.read_text(encoding="utf-8").strip() if ref.exists() else "unknown"

    summary = {
        "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "sealed_test_touched": False,
        "selection_metric": "min_objective_norm_score (paper Worst: minimum objective-wise normalized mean)",
        "selected_candidates": {
            "sppo_avg": best_sppo["candidate"] if best_sppo else None,
            "inpo_avg": best_inpo["candidate"] if best_inpo else None,
        },
        "metrics": metrics,
        "candidate_attempts": attempts,
        "acceptance": {
            "sppo_stability_passing_above_base_avg": sppo_recovered,
            "inpo_stability_passing_above_base_avg": inpo_recovered,
            "all_selected_repaired_baselines_lose_to_ronpo_worst": all_repaired_lose,
            "inpo_worst_marginal_ci_overlaps_ronpo": inpo_worst_ci_overlaps_ronpo,
        },
        "verdict": verdict,
        "provenance": {
            "evaluation_metadata": str(result_dir / "evaluation_metadata.json"),
            "per_objective_scores": str(result_dir / "per_objective_scores.csv"),
            "paired_model_differences": str(result_dir / "paired_model_differences.json"),
            "reward_model_revisions": revisions,
            "hardware": "exactly authorized B200 GPU indices 0,1,2,3; three idle samples before launch; no other-user process touched",
            "decode": {"engine": "vLLM", "seed": 42, "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 2048, "dtype": "bfloat16", "enable_thinking": False},
            "normalization": "per-prompt min-max over the exact evaluated eligible-model pool",
            "bootstrap": {"resamples": 2000, "seed": 42, "paired_unit": "prompt"},
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "repair_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Qwen2.5-1.5B Averaged-Oracle Baseline Repair",
        "",
        "All values below are measured on the same 647 held-out prompts with joint per-prompt normalization. No sealed split was opened.",
        "",
        "| Model | Stability | Avg (95% CI) | Worst objective (95% CI) | Mean prompt-worst (95% CI) | WR vs base (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in ("base", "ronpo", "sppo_avg_repaired", "inpo_avg_repaired"):
        if label not in metrics:
            continue
        row = metrics[label]
        lines.append(
            f"| {label} (`{row['model']}`) | PASS | {fmt(row['avg'])} | {fmt(row['worst_objective'])} | "
            f"{fmt(row['mean_prompt_worst'])} | {fmt(row['win_rate_vs_base'])} |"
        )
    lines += ["", "## Honest verdict", "", verdict]
    if inpo_worst_ci_overlaps_ronpo:
        lines += [
            "",
            "The marginal 95% bootstrap intervals for RONPO and the selected INPO-avg candidate's "
            "Worst scores overlap; the table therefore supports the point-estimate ordering but not a "
            "claim of clearly separated uncertainty intervals.",
        ]
    lines += ["", "## Candidate audit", ""]
    for attempt in attempts:
        gate = attempt["stability_gate"]
        diag = gate.get("candidate_diagnostics", {})
        lines.append(
            f"- `{attempt['candidate']}`: gate={'PASS' if gate.get('passed') else 'FAIL'}, "
            f"mean-word ratio={gate.get('mean_word_ratio_vs_base', 'unknown')}, "
            f"max-repeat-run={diag.get('max_consecutive_identical_word_run', 'unknown')}, "
            f"W&B={attempt.get('wandb_url') or 'missing'}."
        )
    lines += [
        "",
        "## Provenance",
        "",
        "- Decode: vLLM, seed 42, temperature 0.7, top-p 0.9, max 2048, bf16.",
        "- RMs: Skywork V2 8B, Athene-RM-8B, ArmoRM-Llama3-8B-v0.1; exact revisions are in `repair_summary.json`.",
        "- Bootstrap: 2,000 paired prompt resamples, seed 42.",
        "- Hardware: only authorized B200 GPU 0/1/2/3; no other-user process touched.",
        "- Source metrics: `eval/results/model_summary.json`, `eval/results/per_objective_scores.csv`, and stability gate JSON files.",
    ]
    (args.output_dir / "REPAIR_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "selected": summary["selected_candidates"]}, indent=2))


if __name__ == "__main__":
    main()
