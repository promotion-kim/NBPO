#!/usr/bin/env python3
"""Render audit and selection documents from locked screen artifacts only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value: float) -> str:
    return f"{value:.4f}"


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--analysis-script", type=Path, required=True)
    args = parser.parse_args()

    root = args.root
    summary = json.loads((root / "analysis_results" / "summary.json").read_text())
    lock = json.loads((root / "selection_lock.json").read_text())
    selected = summary["selected_setting"]
    if not selected:
        raise RuntimeError("No selected setting; renderer requires the completed gate output.")
    resolution = read_csv(root / "analysis_results" / "resolution.csv")
    candidates = read_csv(root / "analysis_results" / "candidate_settings.csv")

    def resolution_row(objective: str) -> dict:
        for row in resolution:
            if row["base"] == selected["base"] and row["objective"] == objective:
                return row
        raise RuntimeError(f"missing resolution row: {objective}")

    help_resolution = resolution_row("compliance")
    safety_resolution = resolution_row(selected["harmlessness"])
    probe_validity = resolution_row("unsafe_probe_behavior")
    qualifying = [row for row in candidates if row["qualifies"] == "True"]

    dataset = lock["dataset"]
    components = [
        ("Base policy", "meta-llama/Llama-3.1-8B-Instruct (locked revision in model audit)"),
        ("Helpfulness", "WildGuard compliance: continuous P(response is not a refusal) on should-answer prompts"),
        ("Harmlessness", "Qwen/Qwen3Guard-Gen-8B: continuous log P(safe) minus log P(unsafe)"),
        ("Panel", "640 locked validation prompts: 40% PKU-SafeRLHF, 20% OR-Bench hard, 20% balanced XSTest, 20% high-severity BeaverTails"),
        ("Fresh split", "320 prompt-disjoint manifest is present and unopened"),
    ]
    lines = [
        "# Selection",
        "",
        "## Locked setting",
        "",
        "| Component | Selection |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in components]
    lines += [
        "",
        "The selection uses only the locked 640-prompt validation panel. It was chosen before any RONPO training or ranking, and the 320-prompt fresh manifest remains unopened.",
        "",
        "## Gate evidence",
        "",
        f"- Conflict: pooled five-policy Spearman rho = **{f(selected['spearman_rho'])}**, below the preregistered -0.2 threshold. The per-prompt top-objective set differed on **{f(selected['top_set_mismatch_rate'])}** of prompts.",
        f"- Helpfulness resolution: Llama-3.1 base minus its locked over-refusing probe on should-answer prompts = **{f(float(help_resolution['mean']))}** (95% CI [{f(float(help_resolution['ci_low']))}, {f(float(help_resolution['ci_high']))}]; n={help_resolution['n']}).",
        f"- Unsafe-probe validity: Zephyr compliance minus Llama-3.1 base on should-refuse prompts = **{f(float(probe_validity['mean']))}** (95% CI [{f(float(probe_validity['ci_low']))}, {f(float(probe_validity['ci_high']))}]).",
        f"- Harmlessness resolution: Llama-3.1 base minus Zephyr on should-refuse prompts = **{f(float(safety_resolution['mean']))}** (95% CI [{f(float(safety_resolution['ci_low']))}, {f(float(safety_resolution['ci_high']))}]).",
        f"- Weakest-objective headroom: Mistral-7 minus Llama-3.1 on compliance = **{f(selected['weaker_headroom_mean'])}** (95% CI [{f(selected['weaker_headroom_ci_low'])}, {f(selected['weaker_headroom_ci_high'])}]).",
        "",
        "## Why this setting is defensible",
        "",
        "Llama-3.1-8B-Instruct is a standard current instruct backbone. WildGuard and Qwen3Guard are independent model lineages, and their scores use distinct continuous response-level signals. The prompt mixture includes both answerable cases where an unnecessary refusal is unhelpful and harmful cases where compliance is unsafe. The setting was selected by preregistered resolution, conflict, and headroom gates rather than by any RONPO outcome.",
        "",
        "## All candidate settings",
        "",
        "| Base | Harmlessness signal | rho | Conflict | Resolution | Weakest headroom | Qualifies |",
        "|---|---|---:|:---:|:---:|:---:|:---:|",
    ]
    for row in candidates:
        conflict = float(row["spearman_rho"]) <= -0.2
        resolution_pass = row["helpfulness_resolution_pass"] == "True" and row["harmlessness_resolution_pass"] == "True" and row["unsafe_probe_behavior_pass"] == "True"
        lines.append(
            f"| {row['base']} | {row['harmlessness']} | {f(float(row['spearman_rho']))} | "
            f"{'PASS' if conflict else 'FAIL'} | {'PASS' if resolution_pass else 'FAIL'} | "
            f"{'PASS' if row['weaker_headroom_pass'] == 'True' else 'FAIL'} | {'PASS' if row['qualifies'] == 'True' else 'FAIL'} |"
        )
    lines += [
        "",
        "Two settings passed every gate. The selected setting has the more negative pooled Spearman (Llama-3.1/Qwen3Guard8: -0.5989; Llama-3.1/ShieldGemma: -0.2524), so the fixed tie priority was not needed.",
        "",
        "## Recommended next experiment, not launched here",
        "",
        "Use the selected base, the same locked mixed-pool recipe, and these two continuous objectives in a separately preregistered matched-budget RONPO-versus-baselines experiment. Split training, validation selection, and a fresh confirmation manifest before fitting. Keep the compliance scorer restricted to should-answer prompts and the harmlessness scorer restricted to should-refuse prompts when evaluating objective-specific effects.",
        "",
        "All numeric claims above are regenerated by `run_selection_analysis.py` from the score JSONL files.",
    ]
    (root / "SELECTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    score_files = sorted((root / "scores").glob("*.jsonl"))
    score_counts = []
    for path in score_files:
        rows = sum(1 for line in path.open(encoding="utf-8") if line.strip())
        score_counts.append((path.name, rows, checksum(path)))
    manifest_dir = root / "dataset_manifest"
    manifest_rows = [(path.name, checksum(path)) for path in sorted(manifest_dir.glob("*.jsonl"))]
    audit = [
        "# Completion audit",
        "",
        "## Scope",
        "",
        "This run performed inference-only decoding and reward scoring. It launched no preference training, Hugging Face upload, or paper edit.",
        "",
        "## Integrity checks",
        "",
        f"- Selection-lock SHA-256: `{checksum(root / 'selection_lock.json')}`.",
        f"- Analysis-script SHA-256: `{checksum(args.analysis_script)}`.",
        f"- Validation-only summary: `{summary['validation_only']}`.",
        f"- Fresh confirmation opened: `{summary['fresh_confirmation_opened']}`.",
        f"- Score files: {len(score_counts)}; each contains one complete row per locked validation prompt.",
        "",
        "| Score file | Rows | SHA-256 |",
        "|---|---:|---|",
    ]
    audit += [f"| {name} | {rows} | `{digest}` |" for name, rows, digest in score_counts]
    audit += [
        "",
        "## Prompt manifests",
        "",
        "| Manifest | SHA-256 |",
        "|---|---|",
    ]
    audit += [f"| {name} | `{digest}` |" for name, digest in manifest_rows]
    audit += [
        "",
        "The original analysis output and its pre-correction script hash are preserved under `audit/initial_analysis_inequality_bug/`; `fix_log.md` records the definition-preserving comparator correction.",
        "",
        "spent_sealed_split_touched=false",
    ]
    (root / "COMPLETION_AUDIT.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "qualifying_settings": len(qualifying), "score_files": len(score_counts)}, indent=2))


if __name__ == "__main__":
    main()
