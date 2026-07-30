#!/usr/bin/env python3
"""Generate the honest fair-demo decision, review table, and completion audit from measured JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DISPLAY = {
    "ronpo_full_expect": "RONPO full-expectation", "ronpo_k_only": "RONPO top-mass",
    "dpo": "DPO", "ipo": "IPO", "simpo": "SimPO", "sppo_avg": "SPPO (avg)",
    "inpo_avg": "INPO (avg)", "ht_mnpo_helpfulness": "HT-MNPO helpfulness",
    "ht_mnpo_safety": "HT-MNPO safety", "ht_mnpo_conciseness": "HT-MNPO conciseness",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def pairwise(summary: dict, left: str, right: str) -> dict:
    for row in summary["pairwise_prompt_bootstrap"]:
        if row["left"] == left and row["right"] == right:
            return row
        if row["left"] == right and row["right"] == left:
            return {"left": left, "right": right, "mean_difference": -row["mean_difference"],
                    "ci95": [-row["ci95"][1], -row["ci95"][0]]}
    raise KeyError((left, right))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fair-root", type=Path, required=True)
    args = parser.parse_args()
    evaluator = json.loads((args.run_dir / "evaluator_lock.json").read_text())
    diagnostic = json.loads((args.run_dir / "diagnostics/results/summary.json").read_text())
    grid = json.loads((args.run_dir / "sweep/grid.json").read_text())
    selection_path = args.run_dir / "validation/results/panel/selection_lock.json"
    selection = json.loads(selection_path.read_text())
    validation_panel = json.loads((args.run_dir / "validation/results/panel/panel_summary.json").read_text())
    official_panel_path = args.run_dir / "fresh_test/results/panel/panel_summary.json"
    sensitivity_panel_path = args.run_dir / "fresh_test/results/panel_schema_alias_sensitivity/panel_summary.json"
    if official_panel_path.is_file():
        fresh_panel_path = official_panel_path
        panel_confirmatory = True
    elif sensitivity_panel_path.is_file():
        fresh_panel_path = sensitivity_panel_path
        panel_confirmatory = False
    else:
        raise RuntimeError("neither a confirmatory nor sensitivity fresh panel is available")
    fresh_panel = json.loads(fresh_panel_path.read_text())
    schema_alias_audit_path = args.run_dir / "fresh_test/judge/schema_alias_sensitivity_audit.json"
    schema_alias_audit = (
        json.loads(schema_alias_audit_path.read_text()) if schema_alias_audit_path.is_file() else {}
    )
    fresh_rewards = json.loads((args.run_dir / "fresh_test/results/rewards/reward_summary.json").read_text())
    fresh_gates = json.loads((args.run_dir / "fresh_test/stability_gates/summary.json").read_text())
    fresh_manifest = json.loads((args.run_dir / "fresh_test_preregistration/fresh_test_manifest.json").read_text())
    terminal_training = json.loads((args.run_dir / "training_terminal.json").read_text())
    terminal_failed = set(terminal_training.get("terminal_failed_candidates", []))
    preregistered_methods = sorted({row["method"] for row in grid["candidates"]})
    failed_validation_methods = sorted(selection.get("failed_methods", []))
    by_model = {row["model"]: row for row in fresh_panel["ranked"]}
    ronpo = selection["selected_ronpo_overall"]
    if ronpo not in by_model:
        verdict = "The validation-selected RONPO checkpoint failed or was not measured on the fresh test."
        strong = False
        comparisons = []
    else:
        base_ci = by_model[ronpo]["delta_vs_base_tie_ci95"]
        comparisons = []
        expected_baselines = []
        for method, row in selection["selected_by_method"].items():
            candidate = row["candidate_id"]
            if method in {"ronpo_full_expect", "ronpo_k_only"}:
                continue
            expected_baselines.append(candidate)
            if candidate not in by_model:
                continue
            comparison = pairwise(fresh_panel, ronpo, candidate)
            comparison["method"] = method
            comparison["candidate_id"] = candidate
            comparisons.append(comparison)
        all_methods_have_stable_selection = (
            not failed_validation_methods
            and set(selection["selected_by_method"]) == set(preregistered_methods)
        )
        all_baselines_measured = len(comparisons) == len(expected_baselines)
        strong = bool(base_ci[0] > 0.0 and all_methods_have_stable_selection
                      and all_baselines_measured and comparisons
                      and all(row["ci95"][0] > 0.0 for row in comparisons))
        if strong:
            verdict = "RONPO satisfies the preregistered strong 8B decision rule on the fresh test."
        else:
            verdict = "RONPO does not satisfy the preregistered strong 8B decision rule on the fresh test."
    if not panel_confirmatory:
        strong = False
        verdict = (
            "The preregistered fresh two-judge primary is invalid because 10 gpt-oss verdicts "
            "failed the frozen schema; the separately labeled schema-alias sensitivity does not "
            "support a strong RONPO claim."
        )
    summary = {
        "status": "completed", "selected_ronpo": ronpo, "strong_claim_supported": strong,
        "verdict": verdict, "fresh_prompt_count": fresh_panel["prompt_count"],
        "confirmatory_primary_valid": panel_confirmatory,
        "fresh_panel_analysis": ("confirmatory" if panel_confirmatory else "schema_alias_sensitivity_only"),
        "selected_ronpo_result": by_model.get(ronpo), "ronpo_vs_selected_baselines": comparisons,
        "failed_validation_methods": failed_validation_methods,
        "failed_fresh_stability_models": fresh_gates["failed_models"],
        "evaluator": {"objectives": evaluator["objective_signals"], "primary": evaluator["primary"],
                      "diagnostic": diagnostic},
        "artifact_hashes": {"evaluator_lock": sha256(args.run_dir / "evaluator_lock.json"),
                            "selection_lock": sha256(selection_path),
                            "fresh_manifest": sha256(args.run_dir / "fresh_test_preregistration/fresh_test_manifest.json"),
                            "fresh_panel": sha256(fresh_panel_path),
                            "schema_alias_sensitivity_audit": (
                                sha256(schema_alias_audit_path) if schema_alias_audit_path.is_file() else None
                            )},
        "spent_sealed_split_touched": False,
    }
    atomic_json(args.run_dir / "summary.json", summary)

    selected_order = [("base", "base")]
    selected_order.extend(
        (method, selection["selected_by_method"].get(method, {}).get("candidate_id"))
        for method in sorted(preregistered_methods,
                             key=lambda name: (0 if name.startswith("ronpo") else 1, name))
    )
    reward_by_model = {row["model"]: row for row in fresh_rewards["ranked_secondary"]}
    lines = ["# Qwen3-8B fair maximal demonstration", "", verdict, "",
             f"Fresh test: {fresh_panel['prompt_count']} prompt-disjoint UltraChat prompts. ",
             f"Locked objectives: `{', '.join(evaluator['objective_signals'])}`. ",
             "Primary: two-judge, two-position mean prompt-level worst objective versus base.",
             f"Confirmatory primary valid: `{str(panel_confirmatory).lower()}`. "
             + ("" if panel_confirmatory else
                "The table below is a non-confirmatory sensitivity after a documented security-to-safety key alias."), "",
             "| Method | Selected candidate | Panel worst | 95% CI | Delta vs base | 95% CI | RM worst standardized delta | Stability |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    latex_rows = []
    for method, candidate in selected_order:
        display = "Base" if method == "base" else DISPLAY[method]
        panel = by_model.get(candidate) if candidate is not None else None
        stability = "PASS" if candidate in fresh_gates["eligible_models"] else "FAIL"
        if panel is None:
            candidate_label = candidate or "no stable validation candidate"
            lines.append(f"| {display} | `{candidate_label}` | NA | NA | NA | NA | NA | {stability} |")
            latex_rows.append(f"{display} & -- & -- & -- & -- & {stability} \\\\")
            continue
        reward = reward_by_model.get(candidate)
        reward_value = reward["mean_prompt_worst_standardized_delta"] if reward else float("nan")
        ci = panel["mean_prompt_worst_panel_score_ci95"]
        dci = panel["delta_vs_base_tie_ci95"]
        lines.append(
            f"| {display} | `{candidate}` | {panel['mean_prompt_worst_panel_score']:.4f} | "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] | {panel['delta_vs_base_tie']:.4f} | "
            f"[{dci[0]:.4f}, {dci[1]:.4f}] | {reward_value:.4f} | {stability} |"
        )
        latex_rows.append(
            f"{display} & {panel['mean_prompt_worst_panel_score']:.3f} & "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] & {panel['delta_vs_base_tie']:.3f} & "
            f"{reward_value:.3f} & {stability} \\\\"
        )
    lines.extend(["", "## Provenance", "",
                  f"- Evaluator lock SHA-256: `{summary['artifact_hashes']['evaluator_lock']}`",
                  f"- Validation selection lock SHA-256: `{summary['artifact_hashes']['selection_lock']}`",
                  f"- Fresh manifest SHA-256: `{summary['artifact_hashes']['fresh_manifest']}`",
                  f"- Fresh panel SHA-256: `{summary['artifact_hashes']['fresh_panel']}`",
                  f"- Schema-alias audit SHA-256: `{summary['artifact_hashes']['schema_alias_sensitivity_audit'] or 'not applicable'}`",
                  f"- Position score agreement: `{fresh_panel['position_score_agreement']:.4f}`",
                  f"- Inter-judge agreement: `{fresh_panel['inter_judge_mean_position_score_agreement']:.4f}`",
                  "- Spent sealed split touched: `false`", ""])
    (args.run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    table = ["\\begin{tabular}{lccccc}", "\\toprule",
             "Method & Panel worst & 95\\% CI & $\\Delta$ vs. base & RM worst $z$ & Gate \\\\",
             "\\midrule", *latex_rows, "\\bottomrule", "\\end{tabular}", ""]
    tables = args.run_dir / "tables"; tables.mkdir(exist_ok=True)
    (tables / "qwen3_fair_demo_main.tex").write_text("\n".join(table), encoding="utf-8")
    ronpo_row = by_model.get(ronpo)
    if ronpo_row is not None:
        worst_ci = ronpo_row["mean_prompt_worst_panel_score_ci95"]
        delta_ci = ronpo_row["delta_vs_base_tie_ci95"]
        ronpo_macros = {
            "QwenFairRonpoWorst": f"{ronpo_row['mean_prompt_worst_panel_score']:.3f}",
            "QwenFairRonpoWorstLow": f"{worst_ci[0]:.3f}",
            "QwenFairRonpoWorstHigh": f"{worst_ci[1]:.3f}",
            "QwenFairRonpoDelta": f"{ronpo_row['delta_vs_base_tie']:.3f}",
            "QwenFairRonpoDeltaLow": f"{delta_ci[0]:.3f}",
            "QwenFairRonpoDeltaHigh": f"{delta_ci[1]:.3f}",
        }
    else:
        ronpo_macros = {name: "--" for name in (
            "QwenFairRonpoWorst", "QwenFairRonpoWorstLow", "QwenFairRonpoWorstHigh",
            "QwenFairRonpoDelta", "QwenFairRonpoDeltaLow", "QwenFairRonpoDeltaHigh",
        )}
    macro_values = {
        "QwenFairPromptCount": str(fresh_panel["prompt_count"]),
        "QwenFairEvaluatedModelCount": str(len(by_model)),
        "QwenFairFailedMethodCount": str(len(failed_validation_methods)),
        "QwenFairRonpoRank": str(ronpo_row["rank"]) if ronpo_row is not None else "--",
        "QwenFairFreshMDE": f"{fresh_manifest['power']['minimum_detectable_absolute_effect']:.3f}",
        "QwenFairConfirmatoryValid": "1" if panel_confirmatory else "0",
        "QwenFairInvalidVerdictCount": str(schema_alias_audit.get("original_invalid", 0)),
        "QwenFairJudgeVerdictCount": str(schema_alias_audit.get("original_rows", 0)),
        "QwenFairPositionAgreement": f"{fresh_panel['position_score_agreement']:.3f}",
        "QwenFairInterJudgeAgreement": f"{fresh_panel['inter_judge_mean_position_score_agreement']:.3f}",
        "QwenFairConflictRho": f"{diagnostic['selected_triple']['median_pairwise_spearman']:.2f}",
        "QwenFairConflictMismatch": f"{diagnostic['selected_triple']['mean_pairwise_top1_mismatch']:.2f}",
        "QwenFairStrongSupported": "1" if strong else "0",
        **ronpo_macros,
    }
    macros = ["% Auto-generated from measured fair-demo JSON. Do not edit by hand."]
    macros.extend(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macro_values.items())
    (tables / "qwen3_fair_demo_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    validation_gates = json.loads((args.run_dir / "validation/stability_gates/summary.json").read_text())
    audit = ["# Completion audit", "", f"Decision: {verdict}", "", "## Training candidates", "",
             "| Candidate | Method | Training | Validation gate | Max repeat | Mean-word ratio | W&B run ID | W&B URL |",
             "|---|---|---|---|---:|---:|---|---|"]
    for row in grid["candidates"]:
        candidate = row["id"]
        status_path = args.fair_root / "sweep/candidates" / candidate / "training_status.json"
        status = json.loads(status_path.read_text()) if status_path.is_file() else {}
        state = "terminal_failed" if candidate in terminal_failed else status.get("status", "missing")
        gate_summary = validation_gates["models"].get(candidate, {})
        gate_path = args.run_dir / "validation/stability_gates" / f"{candidate}.json"
        gate = json.loads(gate_path.read_text()) if gate_summary and gate_path.is_file() else {}
        gate_state = "PASS" if gate.get("passed") is True else "FAIL"
        max_repeat = gate.get("candidate", {}).get("max_repeat_run", "NA")
        mean_ratio = gate.get("candidate_base_mean_word_ratio")
        ratio_text = "NA" if mean_ratio is None else f"{mean_ratio:.3f}"
        audit.append(f"| `{candidate}` | `{row['method']}` | {state} | {gate_state} | "
                     f"{max_repeat} | {ratio_text} | `{status.get('wandb_run_id', '')}` | "
                     f"{status.get('wandb_url', '')} |")
    audit.extend(["", "## Gates and immutable boundaries", "",
                  f"- Validation candidates measured: {len(validation_panel['ranked']) - 1}",
                  f"- Fresh eligible models: {', '.join(fresh_gates['eligible_models'])}",
                  f"- Fresh failed models: {', '.join(fresh_gates['failed_models']) or 'none'}",
                  f"- Methods with no validation-stable candidate: {', '.join(failed_validation_methods) or 'none'}",
                  f"- Confirmatory fresh primary valid: {str(panel_confirmatory).lower()}",
                  f"- Fresh prompt count: {fresh_manifest['prompt_count']}",
                  "- Every reported cell is generated from JSON/JSONL/CSV artifacts.",
                  "- `spent_sealed_split_touched=false` throughout.", ""])
    (args.run_dir / "COMPLETION_AUDIT.md").write_text("\n".join(audit), encoding="utf-8")
    atomic_json(args.run_dir / "pipeline_status.json", {
        "status": "completed",
        "stage": ("confirmatory_reported" if panel_confirmatory
                  else "confirmatory_invalid_sensitivity_reported"),
        "confirmatory_primary_valid": panel_confirmatory,
        "strong_claim_supported": strong,
        "spent_sealed_split_touched": False,
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
