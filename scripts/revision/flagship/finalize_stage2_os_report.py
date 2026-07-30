#!/usr/bin/env python3
"""Finalize the honest Stage-1 OS decision and symmetric Stage-2 feasibility audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float) -> str:
    return f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    grid = load(root / "sweep/candidate_grid.json")
    selection = load(root / "sweep/stage1_selection_lock.json")
    val = load(root / "validation/results/model_summary.json")
    val_gates = load(root / "validation/stability_gates/summary.json")
    test = load(root / "fixed647/results/model_summary.json")
    test_gates = load(root / "fixed647/stability_gates/summary.json")
    metric = load(root / "metric_lock.json")
    selected_os = next(row for row in selection["selected"] if row["method"] == "ronpo_os")
    os_test_gate = next(row for row in test_gates["rows"] if row["id"] == selected_os["candidate_id"])
    if os_test_gate["passed"]:
        raise RuntimeError("this finalizer is for the measured fail-closed Phase-A outcome")

    candidate_method = {row["id"]: row["method"] for row in grid["candidates"]}
    eligible_methods = sorted({candidate_method[value] for value in val_gates["eligible_candidates"]})
    required_methods = sorted(grid["candidate_count_by_method"])
    missing_parent = sorted(set(required_methods) - set(eligible_methods))
    feasibility = {
        "status": "BLOCKED_FAIL_CLOSED_NOT_LAUNCHED", "assessed_at": datetime.now().astimezone().isoformat(),
        "phase_a_outcome": "FAIL", "required_stage_matching": True,
        "required_methods": required_methods, "methods_with_eligible_stage1_parent": eligible_methods,
        "methods_without_eligible_stage1_parent": missing_parent,
        "blocking_evidence": {
            "simpo_validation_candidates": [row for row in val_gates["rows"] if row["method"] == "simpo"],
            "selected_os_fixed647_gate": os_test_gate,
        },
        "reason": ("A complete symmetric Stage-2 cannot start because SimPO has no Stage-1 checkpoint "
                   "that passes the unchanged stability gate. Launching only the remaining methods would "
                   "violate the pre-registered stage-matching rule."),
        "stage2_jobs_launched": 0, "spent_sealed_split_touched": False,
    }
    atomic_json(root / "stage2_feasibility.json", feasibility)
    atomic_json(root / "fresh/NOT_OPENED.json", {
        "status": "NOT_DRAWN_NOT_OPENED", "reason": "Selected RONPO-OS failed the fixed-647 stability gate before reward scoring.",
        "independent_judge_invoked": False, "fresh_prompt_manifest_created": False,
        "spent_sealed_split_touched": False})
    decision = {
        "status": "FAIL", "decided_at": datetime.now().astimezone().isoformat(),
        "selected_ronpo_os": selected_os, "fixed647_stability_gate": os_test_gate,
        "reason": "The validation-selected RONPO-OS checkpoint failed the pre-registered fail-closed stability gate on the fixed 647 prompts.",
        "fresh_confirmation_opened": False, "hf_upload_performed": False,
        "table4_edited": False, "stage2_launched": False,
        "spent_sealed_split_touched": False,
    }
    atomic_json(root / "phase_a_decision.json", decision)
    atomic_json(root / "hf_upload_status.json", {
        "status": "NOT_PERFORMED_BY_DECISION_RULE", "reason": "Phase A outcome was FAIL",
        "repos_created": [], "local_large_files_deleted": [], "spent_sealed_split_touched": False})

    wandb_rows = []
    for candidate in grid["candidates"]:
        patterns = []
        if candidate["source"] == "frozen_fair_demo_two_config_grid":
            patterns = [args.experiment_root / "fair_demo_20260715/sweep/wandb" / candidate["id"] / "wandb"]
        elif candidate["source"].startswith("ronpo_variant_search_round"):
            round_name = "round1" if "round1" in candidate["source"] else "round2"
            patterns = [args.experiment_root / "ronpo_variant_search_20260715" / round_name /
                        "wandb" / candidate["config_id"] / "wandb"]
        runs = []
        for directory in patterns:
            runs.extend(sorted(directory.glob("run-*")))
        run_ids = sorted({path.name.split("-", 2)[-1] for path in runs})
        wandb_rows.append({
            "candidate_id": candidate["id"], "method": candidate["method"],
            "wandb_run_ids": run_ids,
            "status": "recovered_from_local_wandb_run_directory" if run_ids else
                      "prior_public_checkpoint_no_current_sweep_run_directory",
            "hf_repo": candidate.get("hf_repo"), "hf_revision": candidate.get("hf_revision"),
        })
    atomic_json(root / "sweep/wandb_run_ids.json", {
        "status": "measured_from_training_artifacts", "rows": wandb_rows,
        "spent_sealed_split_touched": False})

    per_obj = {}
    with (root / "fixed647/results/per_objective_scores.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            per_obj.setdefault(row["model"], {})[row["objective"]] = float(row["marginal_win_rate_vs_base"])
    fixed_rows = test["ranked_selected_method_set"]
    lines = [
        "# Qwen3-8B RONPO-OS Stage-1 check",
        "",
        "## Decision",
        "",
        "**FAIL.** The validation-selected OS checkpoint passed the 128-prompt gate but failed the unchanged "
        "647-prompt gate on one genuine repetition loop. Under the fail-closed rule it was not reward-scored. "
        "The fresh independent-judge split was therefore neither drawn nor opened. No model was uploaded and "
        "Table 4 was not edited.",
        "",
        "## Validation selection",
        "",
        "| Rank | Method | Candidate | Worst marginal | 95% CI |",
        "|---:|---|---|---:|---:|",
    ]
    for row in val["ranked_selected_method_set"]:
        ci = row["worst_objective_marginal_win_rate_ci95"]
        lines.append(f"| {row['selected_set_rank']} | {row['method']} | `{row['model']}` | {fmt(row['worst_objective_marginal_win_rate'])} | [{fmt(ci[0])}, {fmt(ci[1])}] |")
    lines += ["", "SimPO had no eligible row: all three pre-locked Stage-1 candidates failed the repetition gate.",
              "", "## Fixed 647-prompt stability", "",
              "| Method | Candidate | Gate | Word ratio | Max repeat | Failed checks |",
              "|---|---|---|---:|---:|---|"]
    for row in test_gates["rows"]:
        failed = ", ".join(key for key, value in row["checks"].items() if not value) or "none"
        lines.append(f"| {row['method']} | `{row['id']}` | {row['status']} | {row['candidate_base_mean_word_ratio']:.3f} | {row['candidate']['max_repeat_run']} | {failed} |")
    lines += ["", "## Fixed 647-prompt local-RM results (gate-passing models only)", "",
              "| Rank | Method | Worst marginal | 95% CI | Skywork | Athene | ArmoRM |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for row in fixed_rows:
        ci = row["worst_objective_marginal_win_rate_ci95"]
        obj = {key: 0.5 for key in metric["objectives"]} if row["model"] == "base" else per_obj[row["model"]]
        lines.append(f"| {row['selected_set_rank']} | {row['method']} | {fmt(row['worst_objective_marginal_win_rate'])} | [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(obj['skywork'])} | {fmt(obj['athene'])} | {fmt(obj['armo'])} |")
    lines += [
        "", "RONPO-OS, full-expectation RONPO, DPO, HT-MNPO-helpfulness, and HT-MNPO-safety are absent "
        "from this reward table because they failed the fixed-647 stability gate. The top-mass arm passed but "
        "ranked below base and IPO on the pre-registered worst marginal.",
        "", "## Stage-2",
        "", "Stage-2 was not launched. SimPO has no stable Stage-1 parent, so a complete stage-matched comparison "
        "cannot be constructed without violating the symmetric-continuation rule. Partial Stage-2 training was not used.",
        "", "## Provenance", "",
        f"- Metric lock SHA-256: `{sha(root / 'metric_lock.json')}`",
        f"- Candidate grid SHA-256: `{sha(root / 'sweep/candidate_grid.json')}`",
        f"- Fixed test SHA-256: `{selection['fixed_test']['sha256']}` (647 prompts)",
        "- Decode: vLLM, seed 42, temperature 0.7, top-p 0.9, 2048 new tokens, bfloat16, thinking disabled.",
        "- Reward models: exact revisions in `metric_lock.json`; paired 2,000-resample prompt bootstrap, seed 42.",
        "- GPU scope: four authorized B200 GPUs. H200 was not used because the lab allocation was full. Odin2 was not needed.",
        "- No other user's process was stopped, modified, or attached to.",
        "- `spent_sealed_split_touched=false`",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit = [
        "# Completion audit", "", "- Phase A: completed, honest FAIL.",
        "- Validation selection: locked before the fixed 647-prompt decode.",
        "- Fixed 647 decode: completed once for the locked model set.",
        "- Stability: fail-closed; failed candidates preserved and unscored.",
        "- Fresh independent judge: not drawn and not opened.",
        "- Phase B: not launched because a symmetric set has no stable SimPO Stage-1 parent.",
        "- HF upload: not performed under the pre-registered FAIL rule.",
        "- Table 4 / `main_v3.tex`: unchanged under the pre-registered FAIL rule.",
        "- Large artifact deletion: none, because no new upload was authorized by the decision rule.",
        "- Other users' processes: untouched.", "- spent_sealed_split_touched=false", "",
    ]
    (root / "COMPLETION_AUDIT.md").write_text("\n".join(audit), encoding="utf-8")
    print(json.dumps({"decision": "FAIL", "report": str(root / "REPORT.md"),
                      "stage2": feasibility["status"]}, indent=2))


if __name__ == "__main__":
    main()
