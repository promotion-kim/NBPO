#!/usr/bin/env python3
"""Build the measured OS-only report and audit after fresh local-RM and panel completion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float) -> str:
    return f"{float(value):.4f}"


def ci(value: list[float]) -> str:
    return f"[{fmt(value[0])}, {fmt(value[1])}]"


def local_table(summary: dict) -> list[str]:
    lines = ["| Rank | Model | Worst marginal | 95% CI | Mean marginal | Disparity | Norm. worst |",
             "|---:|---|---:|---|---:|---:|---:|"]
    for row in summary["ranked_all_eligible_candidates"]:
        lines.append(f"| {row['global_rank']} | {row['model']} | {fmt(row['worst_objective_marginal_win_rate'])} | "
                     f"{ci(row['worst_objective_marginal_win_rate_ci95'])} | "
                     f"{fmt(row['mean_objective_marginal_win_rate'])} | "
                     f"{fmt(row['cross_objective_marginal_spread'])} | "
                     f"{fmt(row['mean_prompt_worst_norm_score_continuity'])} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_root
    prereg = load(root / "prereg_lock.json")
    gate = load(root / "full647_gate/gates/summary.json")
    selection = load(root / "fixed647_selection/selection_lock.json")
    fixed = load(root / "fixed647_exact_table4/results/model_summary.json")
    fresh = load(root / "fresh_test/results/localrm/model_summary.json")
    panel = load(root / "fresh/panel_summary.json")
    decision = load(root / "fresh_localrm_decision.json")
    fresh_gates = load(root / "fresh_test/stability_gates/summary.json")
    selected_path = Path(selection["selected"]["model_path"])
    candidates_root = next((path for path in [selected_path, *selected_path.parents]
                            if path.name == "candidates"), None)
    if candidates_root is None:
        raise RuntimeError("selected model path is outside the frozen OS candidate root")
    training_statuses = []
    for path in sorted(candidates_root.glob("*/training_status.json")):
        value = load(path)
        training_statuses.append({"candidate_id": value.get("candidate_id"),
                                  "wandb_run_id": value.get("wandb_run_id"),
                                  "wandb_url": value.get("wandb_url"),
                                  "status": value.get("status"),
                                  "measured_step": value.get("measured_step")})
    fresh_out = root / "fresh"; fresh_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "fresh_test/results/localrm/model_summary.json", fresh_out / "model_summary.json")
    shutil.copy2(root / "fresh_test/results/localrm/per_objective_scores.csv", fresh_out / "per_objective_scores.csv")
    selected_id = selection["selected"]["model_id"]
    selected_profile = next(row for row in gate["rows"] if row["model_id"] == selected_id)
    panel_by_model = {row["model"]: row for row in panel["ranked"]}
    os_panel = panel_by_model.get("ronpo_os")
    report = [
        "# Qwen3-8B RONPO-OS-only stabilization report", "",
        f"Decision: **{decision['decision']}**. {decision['reason']}.", "",
        f"Selected checkpoint: `{selected_id}` (`{selection['selected']['model_path']}`).",
        f"The selection used the fixed 647 prompts only after the unchanged 4096-token gate was frozen. "
        f"The one-shot fresh split contains {fresh['prompt_count']} prompts and was not used for selection.", "",
        "## Full-647 checkpoint stability profile", "",
        "| Recipe | Passing steps | Failing steps | Robust neighbor-pass steps |", "|---|---|---|---|",
    ]
    for profile in gate["profiles"]:
        report.append(f"| {profile['candidate_id']} | {profile['passed_steps']} | {profile['failed_steps']} | {profile['robust_pass_steps']} |")
    report.extend(["", "Selected-step gate evidence:", "",
                   f"- max repeat run: {selected_profile['candidate']['max_repeat_run']}",
                   f"- empty responses: {selected_profile['candidate']['empty_count']}",
                   f"- non-empty paired think spans: {selected_profile['candidate']['think_leak_count']}",
                   f"- mean-word ratio vs base: {fmt(selected_profile['candidate_base_mean_word_ratio'])}",
                   "", "## Fixed-647 exact frozen-baseline comparison", "", *local_table(fixed),
                   "", "## One-shot fresh local-RM comparison", "", *local_table(fresh),
                   "", "## Independent open-weight panel", ""])
    if os_panel:
        report.extend([f"RONPO-OS panel worst-objective marginal: {fmt(os_panel['worst_objective_marginal'])} "
                       f"(95% CI {ci(os_panel['worst_objective_marginal_ci95'])}); "
                       f"delta vs base {fmt(os_panel['delta_vs_base'])} "
                       f"(95% CI {ci(os_panel['delta_vs_base_ci95'])}).", ""])
    report.extend(["| Rank | Model | Panel worst | 95% CI | Disparity | Legacy prompt-worst |",
                   "|---:|---|---:|---|---:|---:|"])
    for row in panel["ranked"]:
        report.append(f"| {row['rank']} | {row['model']} | {fmt(row['worst_objective_marginal'])} | "
                      f"{ci(row['worst_objective_marginal_ci95'])} | {fmt(row['cross_objective_disparity'])} | "
                      f"{fmt(row['legacy_mean_prompt_worst'])} |")
    report.extend(["", "## Provenance", "",
                   f"- metric lock: `{prereg['metric_lock_sha256']}`",
                   f"- stability gate spec: `{prereg['stability_gate_spec_sha256']}`",
                   f"- selected 4096-gate summary: `{sha256(root / 'full647_gate/gates/summary.json')}`",
                   f"- fresh local-RM summary: `{sha256(root / 'fresh/model_summary.json')}`",
                   f"- independent panel summary: `{sha256(root / 'fresh/panel_summary.json')}`",
                   f"- fresh gate failures: `{fresh_gates.get('failed_models', [])}`",
                   f"- W&B runs: `{[(row['candidate_id'], row['wandb_run_id']) for row in training_statuses]}`",
                   "- baseline training launched: `false`", "- spent_sealed_split_touched: `false`", ""])
    (root / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    terminal = decision["decision"] == "FAIL"
    audit = {
        "status": ("COMPLETED_FAIL_NO_UPLOAD_NO_PAPER_EDIT" if terminal
                   else "MEASUREMENT_COMPLETE_PENDING_CONDITIONAL_UPLOAD_AND_PAPER_ACTION"),
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "decision": decision["decision"], "decision_reason": decision["reason"],
        "selected_os": selection["selected"], "wandb_runs": training_statuses,
        "full647_gate_profiles": gate["profiles"],
        "fresh_gate_eligible": fresh_gates["eligible_models"],
        "fresh_gate_failed": fresh_gates.get("failed_models", []),
        "artifacts": {str(path.relative_to(root)): sha256(path) for path in [
            root / "PREREG.md", root / "metric_lock.json", root / "stability_gate_spec.json",
            root / "sweep/baseline_reuse_ledger.json", root / "fixed647_selection/selection_lock.json",
            root / "fixed647_exact_table4/results/model_summary.json", root / "fresh/model_summary.json",
            root / "fresh/panel_summary.json", root / "REPORT.md"]},
        "baseline_training_launched": False,
        "hf_upload_performed": False,
        "paper_edited": False,
        "large_files_deleted": False,
        "spent_sealed_split_touched": False,
    }
    (root / "COMPLETION_AUDIT.md").write_text(
        "# Completion audit\n\n" + json.dumps(audit, indent=2) + "\n\nspent_sealed_split_touched=false\n",
        encoding="utf-8")
    (root / "completion_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (root / "pipeline_status.json").write_text(json.dumps({
        "status": "completed",
        "decision": decision["decision"],
        "reason": decision["reason"],
        "hf_upload_performed": False,
        "paper_edited": False,
        "spent_sealed_split_touched": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision["decision"], "selected": selected_id,
                      "report": str(root / "REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
