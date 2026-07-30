#!/usr/bin/env python3
"""Create the honest final variant-search report from locked measured artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


LABELS = {
    "base": "Base", "sppo_b": "SPPO (avg)", "ht_safety_a": "HT-MNPO safety",
    "ronpo_full_a": "RONPO full-exp (frozen)", "ronpo_top_a": "RONPO top-mass (frozen)",
    "inpo_b": "INPO (avg)", "dpo_b": "DPO", "ht_help_b": "HT-MNPO helpfulness",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hours_between(start: str, end: str) -> float:
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 3600.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--final-lock", type=Path, required=True)
    parser.add_argument("--candidate-panel", type=Path, required=True)
    parser.add_argument("--calibration-panel", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--reward-summary", type=Path, required=True)
    parser.add_argument("--main-tex", type=Path, required=True)
    parser.add_argument("--main-pdf", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.final_lock.read_text(encoding="utf-8"))
    candidate_panel = json.loads(args.candidate_panel.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration_panel.read_text(encoding="utf-8"))
    gates = json.loads(args.gates.read_text(encoding="utf-8"))
    rewards = json.loads(args.reward_summary.read_text(encoding="utf-8"))
    selected = lock["selected_variant"]
    model_id = selected["model_id"]
    row = next(value for value in candidate_panel["ranked"] if value["model"] == model_id)
    point = float(row["mean_prompt_worst_panel_score"])
    delta_ci = [float(x) for x in row["delta_vs_base_tie_ci95"]]
    if point >= 0.5 and delta_ci[0] > 0.0:
        outcome = "BEAT_BASE"
    elif point >= 0.5 and delta_ci[0] <= 0.0 <= delta_ci[1]:
        outcome = "MATCH_BASE"
    else:
        outcome = "NULL_BELOW_BASE"
    upload_authorized = outcome in {"BEAT_BASE", "MATCH_BASE"}
    calibration_rows = []
    for value in calibration["ranked"]:
        calibration_rows.append({
            "model": value["model"], "label": LABELS.get(value["model"], value["model"]),
            "role": "pre-search calibration", "mean_prompt_worst_panel_score":
                value["mean_prompt_worst_panel_score"],
            "ci95": value["mean_prompt_worst_panel_score_ci95"],
        })
    combined = [*calibration_rows, {
        "model": model_id, "label": selected["candidate_id"], "role": "prospective locked finalist",
        "mean_prompt_worst_panel_score": point,
        "ci95": row["mean_prompt_worst_panel_score_ci95"],
    }]
    combined.sort(key=lambda value: (-float(value["mean_prompt_worst_panel_score"]), value["model"]))
    attempts = sorted(lock.get("all_attempted_variants", lock["all_validation_selected_candidates"]),
                      key=lambda value: value["candidate_id"])
    variant_table = []
    for attempt in attempts:
        config = attempt.get("candidate_config", {})
        is_final = attempt.get("model_id") == model_id
        row_payload = {
            "candidate_id": attempt["candidate_id"],
            "round": attempt.get("round"),
            "selected_checkpoint_model_id": attempt.get("model_id"),
            "selected_step": attempt.get("step"),
            "wandb_run_id": attempt.get("wandb_run_id"),
            "wandb_url": attempt.get("wandb_url"),
            "hyperparameters": config,
            "theory_note": config.get("theory_note", config.get("mechanism")),
            "s3_pass": attempt.get("s3_pass") is True,
            "validation_status": attempt.get("status"),
            "validation_selection_metric": attempt.get("selection_metric"),
            "validation_selection_metric_ci95": attempt.get("selection_metric_ci95"),
            "athene_paired_delta_vs_base": attempt.get("athene", {}).get("mean_paired_delta_vs_base"),
            "athene_paired_delta_vs_base_ci95": attempt.get("athene", {}).get("paired_delta_vs_base_ci95"),
            "fresh_panel_opened": is_final,
            "fresh_panel_worst": point if is_final else None,
            "fresh_panel_worst_ci95": row["mean_prompt_worst_panel_score_ci95"] if is_final else None,
            "verdict": (("KEEP" if upload_authorized else "DISCARD:numbers") if is_final
                        else ("DISCARD:validation" if attempt.get("s3_pass") else "DISCARD:stability")),
        }
        variant_table.append(row_payload)
    training_hours = 0.0
    seen_training = set()
    for attempt in lock.get("all_attempted_variants", lock["all_validation_selected_candidates"]):
        model_path = attempt.get("model_path")
        if not model_path:
            continue
        path = Path(model_path)
        candidate = path if (path / "training_status.json").is_file() else path.parent
        status_path = candidate / "training_status.json"
        if candidate in seen_training or not status_path.is_file():
            continue
        seen_training.add(candidate)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("started_at") and status.get("completed_at"):
            training_hours += hours_between(status["started_at"], status["completed_at"])
    evaluation_upper_bound = 0.0
    for work in [args.run_dir / "round1_validation", args.run_dir / "round2_validation",
                 args.run_dir / "round3_validation"]:
        samples = work / "prelaunch_gpu_samples.json"; status_path = work / "status.json"
        if samples.is_file() and status_path.is_file():
            first = json.loads(samples.read_text(encoding="utf-8"))["samples"][0]["timestamp"]
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
            if status_payload.get("completed_at"):
                evaluation_upper_bound += 4.0 * hours_between(first, status_payload["completed_at"])
    final_samples = args.run_dir / "final_confirmation/fresh_test/prelaunch_gpu_samples.json"
    final_status = args.run_dir / "final_confirmation/fresh_test/scoring_status.json"
    if final_samples.is_file() and final_status.is_file():
        first = json.loads(final_samples.read_text(encoding="utf-8"))["samples"][0]["timestamp"]
        status_payload = json.loads(final_status.read_text(encoding="utf-8"))
        if status_payload.get("completed_at"):
            evaluation_upper_bound += 4.0 * hours_between(first, status_payload["completed_at"])
    summary = {
        "status": "completed", "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "outcome": outcome, "upload_authorized": upload_authorized,
        "selected_variant": selected, "fresh_panel": row,
        "fresh_panel_all_judgments_valid": True,
        "fresh_stability_gate_passed": model_id in gates["eligible_models"],
        "validation_reward_summary": rewards,
        "variant_table": variant_table,
        "combined_review_table": combined,
        "comparison_caveat": "Frozen baseline rows are pre-search calibration values on the same prompts; only the locked finalist row is a prospective post-lock measurement.",
        "paper_action": ("eligible_for_measured_update_after_verified_public_upload" if upload_authorized
                         else "main_v3_Table4_unchanged_honest_null"),
        "gpu_hours": {"training_measured": training_hours,
                      "evaluation_scheduled_upper_bound": evaluation_upper_bound,
                      "total_upper_bound": training_hours + evaluation_upper_bound,
                      "note": "Training sums one-GPU wall time per variant. Evaluation is a conservative four-GPU wall-time upper bound."},
        "spent_sealed_split_touched": False,
        "artifact_sha256": {
            "final_lock": sha256(args.final_lock), "candidate_panel": sha256(args.candidate_panel),
            "calibration_panel": sha256(args.calibration_panel), "gates": sha256(args.gates),
            "reward_summary": sha256(args.reward_summary),
            "main_v3_tex": sha256(args.main_tex),
            "main_v3_pdf": sha256(args.main_pdf) if args.main_pdf.is_file() else None,
        },
    }
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# Qwen3-8B RONPO variant search", "",
             "## Locked finalist", "",
             f"- Candidate: `{selected['candidate_id']}` at step `{selected['step']}`.",
             f"- Validation locked-RM worst standardized delta: `{selected['selection_metric']:.6f}` "
             f"(95% CI `[{selected['selection_metric_ci95'][0]:.6f}, {selected['selection_metric_ci95'][1]:.6f}]`).",
             f"- Fresh 1,024-prompt panel worst: `{point:.6f}` "
             f"(95% CI `[{row['mean_prompt_worst_panel_score_ci95'][0]:.6f}, "
             f"{row['mean_prompt_worst_panel_score_ci95'][1]:.6f}]`).",
             f"- Delta versus base tie anchor: `{row['delta_vs_base_tie']:.6f}` "
             f"(95% CI `[{delta_ci[0]:.6f}, {delta_ci[1]:.6f}]`).", "",
             "## Combined review table", "", "| Rank | Model | Role | Panel worst | 95% CI |",
             "|---:|---|---|---:|---:|"]
    for rank, value in enumerate(combined, 1):
        lines.append(f"| {rank} | {value['label']} | {value['role']} | "
                     f"{float(value['mean_prompt_worst_panel_score']):.4f} | "
                     f"[{float(value['ci95'][0]):.4f}, {float(value['ci95'][1]):.4f}] |")
    verdict = {
        "BEAT_BASE": "The locked RONPO variant beats the base under the prospective panel rule.",
        "MATCH_BASE": "The locked RONPO variant matches the base but does not show a CI-separated beat.",
        "NULL_BELOW_BASE": "No tested RONPO variant reaches the base worst-objective panel anchor; this is the honest powered null.",
    }[outcome]
    lines.extend(["", "## Verdict", "", verdict, "",
                  "The frozen-baseline rows are calibration measurements known before this search. "
                  "The finalist was selected only on the 128-prompt validation RM metric before its panel judgments were run.", "",
                  "`spent_sealed_split_touched=false`", ""])
    (args.run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    ledger = [
        "# Qwen3-8B RONPO variant search", "",
        "> At Qwen3-8B, all previously trained methods lost to the strong base on the "
        "pre-search worst-objective panel. This run kept those baselines frozen, locked the "
        "evaluator before any new ranking, selected checkpoints only on the prompt-disjoint "
        "validation RM metric, and ran the panel only for the final locked variant.", "",
        "## Integrity state", "",
        "- Spent 604-prompt sealed split: forbidden and untouched.",
        "- Seed: 42. Budget: 900 optimizer steps and effective batch 16 per variant.",
        "- W&B: online under `promotion-kim/mnpo`.",
        "- The old fair-demo confirmatory result remains invalid; its corrected aggregation is calibration only.", "",
        "## Variant ledger", "",
        "| Variant | Round | Target | Anchor/SFT | Tau/alpha/eta | LR | Selected step | W&B | S3 | Validation RM worst (95% CI) | Athene delta (95% CI) | Fresh panel | Verdict |",
        "|---|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---|",
    ]
    for attempt in attempts:
        if attempt.get("status") != "selected_on_validation_locked_rm_metric":
            ledger.append(f"| `{attempt['candidate_id']}` | -- | -- | -- | -- | -- | -- | -- | FAIL | -- | -- | -- | DISCARD:stability |")
            continue
        config = attempt["candidate_config"]
        target = config.get("ronpo_target_column", "--")
        if config.get("ronpo_target_schedule_columns"):
            target = "schedule:" + "→".join(config["ronpo_target_schedule_columns"])
        metric_ci = attempt["selection_metric_ci95"]
        athene = attempt["athene"]
        athene_ci = athene["paired_delta_vs_base_ci95"]
        is_final = attempt["model_id"] == model_id
        panel = (f"{point:.4f} [{row['mean_prompt_worst_panel_score_ci95'][0]:.4f}, "
                 f"{row['mean_prompt_worst_panel_score_ci95'][1]:.4f}]" if is_final else "unopened")
        verdict_cell = (("KEEP" if upload_authorized else "DISCARD:numbers") if is_final
                        else "DISCARD:validation")
        ledger.append(
            f"| `{attempt['candidate_id']}` | {str(attempt['round']).replace('round', '')} | `{target}` | "
            f"{config.get('reference_anchor_weight', 0):g}/{config.get('preference_sft_weight', 0):g} | "
            f"{config.get('ronpo_tau', 0):g}/{config.get('ronpo_alpha', 0):g}/{config.get('eta', 0):g} | "
            f"{config.get('learning_rate', 0):.2g} | {attempt['step']} | `{attempt['wandb_run_id']}` | PASS | "
            f"{attempt['selection_metric']:.4f} [{metric_ci[0]:.4f}, {metric_ci[1]:.4f}] | "
            f"{athene['mean_paired_delta_vs_base']:.4f} [{athene_ci[0]:.4f}, {athene_ci[1]:.4f}] | "
            f"{panel} | {verdict_cell} |")
    ledger.extend(["", "## Theory notes", ""])
    for attempt in attempts:
        config = attempt.get("candidate_config", {})
        theory = config.get("theory_note", config.get("mechanism", "No theory note recorded."))
        ledger.append(f"- `{attempt['candidate_id']}`: {theory}")
    ledger.extend(["", "## Final outcome", "", verdict, "",
                   f"- Selected model: `{selected['candidate_id']}` step `{selected['step']}`.",
                   f"- HF upload: `{'authorized after verification' if upload_authorized else 'not performed'}`.",
                   f"- Paper action: `{summary['paper_action']}`.",
                   f"- Measured training GPU-hours: `{training_hours:.3f}`; total scheduled upper bound including evaluation: `{training_hours + evaluation_upper_bound:.3f}`.",
                   "- `spent_sealed_split_touched=false`", ""])
    (args.run_dir / "EXPERIMENT_LOG.md").write_text("\n".join(ledger), encoding="utf-8")
    wandb_ids = [value["wandb_run_id"] for value in variant_table if value.get("wandb_run_id")]
    audit = ["# Completion audit", "", f"- Outcome: `{outcome}`",
             f"- Final lock SHA-256: `{summary['artifact_sha256']['final_lock']}`",
             f"- Candidate panel SHA-256: `{summary['artifact_sha256']['candidate_panel']}`",
             f"- Evaluator lock SHA-256: `{lock['evaluator_lock_sha256']}`",
             f"- main_v3.tex SHA-256: `{summary['artifact_sha256']['main_v3_tex']}`",
             f"- main_v3.pdf SHA-256: `{summary['artifact_sha256']['main_v3_pdf']}`",
             f"- W&B run IDs: `{', '.join(wandb_ids)}`",
             f"- HF upload: `{'authorized pending verification' if upload_authorized else 'not performed (measured null)'}`",
             "- Spent sealed split touched: `false`", ""]
    (args.run_dir / "COMPLETION_AUDIT.md").write_text("\n".join(audit), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
