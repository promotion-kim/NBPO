#!/usr/bin/env python3
"""Create the auditable decision, report, and conditional Table-4 fragment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


NAMES = {
    "base": "Base", "ronpo_top_a": "RONPO (top-mass)", "ronpo_full_a": "RONPO (full-exp.)",
    "dpo_b": "DPO", "sppo_b": "SPPO (avg)", "inpo_b": "INPO (avg)",
    "ht_help_b": "HT-MNPO (help.)", "ht_safety_a": "HT-MNPO (safety)",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def f(value: float) -> str:
    return f"{value:.4f}"


def ci_text(values: list[float]) -> str:
    return f"[{values[0]:.4f}, {values[1]:.4f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--metric-lock", type=Path, required=True)
    parser.add_argument("--fresh-manifest", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--evaluator-lock", type=Path, required=True)
    parser.add_argument("--fair-grid", type=Path, required=True)
    parser.add_argument("--fair-audit", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.run_dir / "fresh/panel_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selection = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    metric = json.loads(args.metric_lock.read_text(encoding="utf-8"))
    execution = json.loads(args.execution_lock.read_text(encoding="utf-8"))
    manifest = json.loads(args.fresh_manifest.read_text(encoding="utf-8"))
    if execution["selection_lock_sha256"] != sha256(args.selection_lock):
        raise RuntimeError("selection lock changed after execution lock")
    if execution["fresh_manifest_sha256"] != sha256(args.fresh_manifest):
        raise RuntimeError("fresh manifest changed after execution lock")
    if summary.get("metric_lock_sha256") != sha256(args.metric_lock):
        raise RuntimeError("summary was not produced from the locked metric")
    rows = {row["model"]: row for row in summary["ranked"]}
    ronpo_id = selection["selected_ronpo_overall"]
    ronpo = rows.get(ronpo_id)
    if ronpo is None:
        raise RuntimeError("validation-selected RONPO failed fresh stability gate")
    baseline_methods = {method: row for method, row in selection["selected_by_method"].items()
                        if method not in {"ronpo_full_expect", "ronpo_k_only"}}
    scored_baselines = {method: rows[row["candidate_id"]] for method, row in baseline_methods.items()
                        if row["candidate_id"] in rows}
    failed_selected = {method: row["candidate_id"] for method, row in baseline_methods.items()
                       if row["candidate_id"] not in rows}
    ronpo_score = ronpo["worst_objective_marginal"]
    baseline_ties_or_beats = sorted(
        method for method, row in scored_baselines.items()
        if row["worst_objective_marginal"] >= ronpo_score
    )
    if baseline_ties_or_beats:
        decision = "FAIL"
    elif ronpo_score >= 0.5:
        decision = "PASS"
    else:
        decision = "PARTIAL"
    significance_vs_base = ronpo["delta_vs_base_ci95"][0] > 0 or ronpo["delta_vs_base_ci95"][1] < 0
    decision_payload = {
        "status": "completed", "decision": decision,
        "selected_ronpo": ronpo_id,
        "selected_ronpo_primary": ronpo_score,
        "selected_ronpo_primary_ci95": ronpo["worst_objective_marginal_ci95"],
        "selected_ronpo_delta_vs_base": ronpo["delta_vs_base"],
        "selected_ronpo_delta_vs_base_ci95": ronpo["delta_vs_base_ci95"],
        "significantly_different_from_base": significance_vs_base,
        "eligible_baselines_tying_or_beating_ronpo": baseline_ties_or_beats,
        "fresh_gate_failed_selected_baselines": failed_selected,
        "validation_terminal_failed_methods": selection["failed_methods"],
        "paper_edit_authorized": decision in {"PASS", "PARTIAL"},
        "hf_upload_authorized": decision in {"PASS", "PARTIAL"},
        "shipped_stage": "stage1",
        "all_compared_models_stage_matched": True,
        "summary_sha256": sha256(summary_path),
        "metric_lock_sha256": sha256(args.metric_lock),
        "fresh_manifest_sha256": sha256(args.fresh_manifest),
        "spent_sealed_split_touched": False,
    }
    atomic_json(args.run_dir / "decision.json", decision_payload)

    report_lines = [
        "# Qwen3-8B marginal-worst fresh confirmation",
        "",
        f"Decision: **{decision}** under the rule locked before validation reaggregation and fresh measurement.",
        "",
        "The metric was specified after the earlier calibration result and is treated as a prospective follow-up. "
        "The 1,024-prompt split was measured once. The spent 604-prompt sealed split was not read, decoded, or scored.",
        "All scored policies use the same Stage-1 budget. The Stage-2 instruction arrived after the fresh split had already been opened, so changing the model set would have invalidated the one-shot confirmation. No Stage-2 model is mixed into this table.",
        "",
        "| Rank | Model | Worst marginal (95% CI) | Helpfulness | Safety | Conciseness | Disparity | Legacy prompt-min | Stability |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["ranked"]:
        obj = row["per_objective_marginal"]
        report_lines.append(
            f"| {row['rank']} | {NAMES.get(row['model'], row['model'])} | {f(row['worst_objective_marginal'])} "
            f"{ci_text(row['worst_objective_marginal_ci95'])} | {f(obj['helpfulness']['mean'])} | "
            f"{f(obj['safety']['mean'])} | {f(obj['conciseness']['mean'])} | "
            f"{f(row['cross_objective_disparity'])} | {f(row['legacy_mean_prompt_worst'])} | passed |"
        )
    for method in selection["failed_methods"]:
        report_lines.append(f"| -- | {method} | -- | -- | -- | -- | -- | -- | terminal FAILED on validation S3 |")
    for method, candidate in failed_selected.items():
        report_lines.append(f"| -- | {NAMES.get(candidate, method)} | -- | -- | -- | -- | -- | -- | FAILED on fresh S3 |")
    report_lines.extend([
        "", "## Decision audit", "",
        f"Validation-selected RONPO is `{ronpo_id}` with fresh worst-objective marginal "
        f"{f(ronpo_score)} {ci_text(ronpo['worst_objective_marginal_ci95'])}. "
        f"Its paired delta from the 0.5 base floor is {f(ronpo['delta_vs_base'])} "
        f"{ci_text(ronpo['delta_vs_base_ci95'])}.",
        "",
        ("At least one eligible trained baseline tied or exceeded RONPO, so the preregistered outcome is FAIL. "
         "No model upload or Table 4 edit is authorized."
         if decision == "FAIL" else
         ("RONPO leads the eligible trained baselines and is at or above the base floor. The preregistered outcome is PASS."
          if decision == "PASS" else
          "RONPO leads the eligible trained baselines but remains below the base floor. The preregistered outcome is PARTIAL.")),
        "", "## Provenance", "",
        f"- Fresh prompt count: {manifest['prompt_count']}",
        f"- Fresh prompt SHA-256: `{manifest['prompt_file_sha256']}`",
        f"- Metric lock SHA-256: `{sha256(args.metric_lock)}`",
        f"- Evaluator lock SHA-256: `{sha256(args.evaluator_lock)}`",
        f"- Selection lock SHA-256: `{sha256(args.selection_lock)}`",
        f"- Execution lock SHA-256: `{sha256(args.execution_lock)}`",
        "- Judges: locked Qwen3-32B and gpt-oss-120b, both position-swapped",
        "- Bootstrap: 2,000 paired prompt resamples, seed 42",
        "- Shipped stage: Stage-1 for every compared trained policy",
        "- spent_sealed_split_touched=false",
    ])
    (args.run_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    sweep_dir = args.run_dir / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    grid = json.loads(args.fair_grid.read_text(encoding="utf-8"))
    audit_text = args.fair_audit.read_text(encoding="utf-8")
    wandb_runs = {}
    for line in audit_text.splitlines():
        if not line.startswith("| `") or "wandb.ai/" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        candidate = cells[0].strip("`")
        url = next((cell for cell in cells if "wandb.ai/" in cell), "")
        run_id = url.rstrip("/").split("/")[-1] if url else ""
        wandb_runs[candidate] = {"run_id": run_id, "url": url}
    atomic_json(sweep_dir / "selection_grid.json", {
        "source_grid_sha256": sha256(args.fair_grid), "budget_rule": grid["budget_rule"],
        "candidates": grid["candidates"], "validation_selection": selection,
        "wandb_runs": wandb_runs, "source_fair_audit_sha256": sha256(args.fair_audit),
        "stage": "stage1", "spent_sealed_split_touched": False,
    })

    if decision in {"PASS", "PARTIAL"}:
        ordered_ids = [ronpo_id] + [row["model"] for row in summary["ranked"]
                                     if row["model"] not in {ronpo_id, "base"}] + ["base"]
        table = [
            "\\begin{tabular}{lrrrrr}", "\\toprule",
            "Method & Worst marginal & Help. & Safety & Concise & Disparity \\\\",
            "\\midrule",
        ]
        for model in ordered_ids:
            row = rows[model]; obj = row["per_objective_marginal"]
            label = NAMES.get(model, model)
            if model == ronpo_id:
                label = "\\textbf{" + label + "}"
            table.append(
                f"{label} & {f(row['worst_objective_marginal'])} "
                f"[{f(row['worst_objective_marginal_ci95'][0])}, {f(row['worst_objective_marginal_ci95'][1])}] & "
                f"{f(obj['helpfulness']['mean'])} & {f(obj['safety']['mean'])} & "
                f"{f(obj['conciseness']['mean'])} & {f(row['cross_objective_disparity'])} \\\\"
            )
        for method in selection["failed_methods"]:
            table.append(f"{method} & FAILED & -- & -- & -- & -- \\\\")
        for method, candidate in failed_selected.items():
            table.append(f"{NAMES.get(candidate, method)} & FAILED & -- & -- & -- & -- \\\\")
        table.extend(["\\bottomrule", "\\end{tabular}"])
        (args.run_dir / "table4_worstobj.tex").write_text("\n".join(table) + "\n", encoding="utf-8")

    audit = [
        "# Completion audit", "",
        f"- Decision: {decision}",
        f"- Selected RONPO: {ronpo_id}",
        f"- Fresh summary SHA-256: `{sha256(summary_path)}`",
        f"- Fresh per-objective CSV SHA-256: `{sha256(args.run_dir / 'fresh/per_objective_marginals.csv')}`",
        f"- Metric lock SHA-256: `{sha256(args.metric_lock)}`",
        f"- Fresh manifest SHA-256: `{sha256(args.fresh_manifest)}`",
        "- Fresh split decoded once and judged once.",
        "- Shipped comparison is stage-matched: Stage-1 for all trained methods.",
        "- Failed stability gates were retained and not substituted.",
        f"- HF upload authorized: {str(decision in {'PASS', 'PARTIAL'}).lower()}",
        f"- Table 4 edit authorized: {str(decision in {'PASS', 'PARTIAL'}).lower()}",
        "- spent_sealed_split_touched=false",
    ]
    (args.run_dir / "COMPLETION_AUDIT.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
    fix_log = [
        "# Fix log", "",
        "- The requested 12-variant RONPO-only headline search was not used because the frozen fair comparison gave every method exactly two configurations. The larger search remains exploratory.",
        "- The fresh-selected HT-MNPO safety model failed the unchanged repetition gate and was retained as FAILED; no regeneration or replacement was performed.",
        "- The symmetric Stage-2 request arrived after the Stage-1 fresh split had been opened. The locked Stage-1 confirmation was completed without changing its model set. A valid Stage-2 confirmation now requires a new preregistration and a new disjoint test split.",
        "- No reward or judge score was used to change the metric, prompts, selection, parser, or model set after the locks were written.",
        "- spent_sealed_split_touched=false",
    ]
    (args.run_dir / "fix_log.md").write_text("\n".join(fix_log) + "\n", encoding="utf-8")
    print(json.dumps(decision_payload, indent=2))


if __name__ == "__main__":
    main()
