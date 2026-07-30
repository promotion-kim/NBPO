#!/usr/bin/env python3
"""Build the deadline report exclusively from measured JSON/CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


METHODS = (
    "base", "ronpo_full_expect", "ronpo_k_only", "dpo", "ipo", "simpo",
    "sppo_avg", "inpo_avg", "ht_mnpo_helpfulness", "ht_mnpo_safety",
    "ht_mnpo_conciseness",
)
ACADEMIC = (
    "MMLU", "MMLU-Pro", "GPQA", "ARC-Challenge", "HellaSwag", "TruthfulQA",
    "Winogrande", "GSM8K", "Minerva-Math", "AIME-24", "HumanEval",
)


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value, digits=3) -> str:
    if value is None or value == "":
        return "UNKNOWN"
    if value == "BLOCKED":
        return "BLOCKED"
    return f"{float(value):.{digits}f}"


def rank_map(values: dict[str, float]) -> dict[str, int]:
    return {
        method: 1 + sum(other > value + 1e-12 for other in values.values())
        for method, value in values.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed-dir", type=Path, required=True)
    parser.add_argument("--ifeval-json", type=Path, required=True)
    parser.add_argument("--academic-dir", type=Path, required=True)
    parser.add_argument("--blockers", type=Path, required=True)
    parser.add_argument("--validation-summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latex-output", type=Path, default=None)
    args = parser.parse_args()

    sealed_summary_path = args.sealed_dir / "ranked_sealed_summary.json"
    sealed_rows = {}
    if sealed_summary_path.is_file():
        payload = json.loads(sealed_summary_path.read_text())
        sealed_rows = {row["model"]: row for row in payload.get("ranked", [])}
    per_head_rows = read_csv(args.sealed_dir / "per_objective_scores.csv")
    per_head = {(row["model"], row["objective"]): row for row in per_head_rows}

    ifeval_rows = {}
    if args.ifeval_json.is_file():
        payload = json.loads(args.ifeval_json.read_text())
        ifeval_rows = {row["method"]: row for row in payload.get("rows", [])}

    academic_rows = {row["method"]: row for row in read_csv(args.academic_dir / "benchmark_table.csv")}
    long_rows = read_csv(args.academic_dir / "metrics_long.csv")
    blocker = json.loads(args.blockers.read_text()) if args.blockers.is_file() else {}
    validation_ronpo = None
    if args.validation_summary and args.validation_summary.is_file():
        validation_payload = json.loads(args.validation_summary.read_text())
        validation_ronpo = next(
            (row for row in validation_payload.get("ranked", [])
             if row.get("model") == "ronpo_full_expect"),
            None,
        )

    worst_values = {
        method: float(row["mean_primary_prompt_worst_norm_score"])
        for method, row in sealed_rows.items()
    }
    ifeval_values = {
        method: float(row["ifeval_prompt_strict_percent"])
        for method, row in ifeval_rows.items()
    }
    worst_ranks = rank_map(worst_values)
    ifeval_ranks = rank_map(ifeval_values)

    capability_macro = {}
    for method in METHODS:
        values = []
        if method in ifeval_values:
            values.append(ifeval_values[method])
        academic = academic_rows.get(method, {})
        for benchmark in ACADEMIC:
            value = academic.get(benchmark)
            if value not in (None, "", "BLOCKED"):
                values.append(float(value))
        if len(values) == 11:  # IFEval + ten accessible academic tasks; GPQA is blocked.
            capability_macro[method] = sum(values) / len(values)
    capability_ranks = rank_map(capability_macro)

    columns = [
        "Method", "Worst norm", "Worst 95% CI", "Worst rank", "Avg norm", "Help raw", "Safety raw",
        "Concise raw", "IFEval", "IFEval rank", *ACADEMIC, "Capability macro", "Macro rank",
    ]
    lines = [
        "# Qwen3-8B AAAI-27 revision results — 2026-07-14",
        "",
        f"Generated at {datetime.now().astimezone().isoformat(timespec='seconds')} exclusively from the JSON/CSV artifacts listed below.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |",
    ]
    for method in METHODS:
        sealed = sealed_rows.get(method, {})
        academic = academic_rows.get(method, {})
        row = [
            method,
            fmt(sealed.get("mean_primary_prompt_worst_norm_score")),
            (f"[{fmt(sealed.get('mean_primary_prompt_worst_norm_score_ci95_low'))}, "
             f"{fmt(sealed.get('mean_primary_prompt_worst_norm_score_ci95_high'))}]"
             if sealed else "UNKNOWN"),
            str(worst_ranks.get(method, "UNKNOWN")),
            fmt(sealed.get("mean_primary_prompt_avg_norm_score")),
            fmt(per_head.get((method, "helpfulness"), {}).get("mean_raw_score")),
            fmt(per_head.get((method, "safety"), {}).get("mean_raw_score")),
            fmt(per_head.get((method, "conciseness"), {}).get("mean_raw_score")),
            fmt(ifeval_rows.get(method, {}).get("ifeval_prompt_strict_percent"), 2),
            str(ifeval_ranks.get(method, "UNKNOWN")),
        ]
        row.extend(fmt(academic.get(benchmark), 2) for benchmark in ACADEMIC)
        row.extend([fmt(capability_macro.get(method), 2), str(capability_ranks.get(method, "UNKNOWN"))])
        lines.append("| " + " | ".join(row) + " |")

    ronpo_worst_rank = worst_ranks.get("ronpo_full_expect")
    ronpo_ifeval_rank = ifeval_ranks.get("ronpo_full_expect")
    ronpo_macro_rank = capability_ranks.get("ronpo_full_expect")
    base_macro = capability_macro.get("base")
    ronpo_macro = capability_macro.get("ronpo_full_expect")
    no_regression = None if base_macro is None or ronpo_macro is None else ronpo_macro >= base_macro
    strict_regressions = []
    if "base" in ifeval_values and "ronpo_full_expect" in ifeval_values:
        if ifeval_values["ronpo_full_expect"] < ifeval_values["base"]:
            strict_regressions.append("IFEval")
    base_academic = academic_rows.get("base", {})
    ronpo_academic = academic_rows.get("ronpo_full_expect", {})
    for benchmark in ACADEMIC:
        base_value, ronpo_value = base_academic.get(benchmark), ronpo_academic.get(benchmark)
        if base_value not in (None, "", "BLOCKED") and ronpo_value not in (None, "", "BLOCKED"):
            if float(ronpo_value) < float(base_value):
                strict_regressions.append(benchmark)
    success = (
        ronpo_worst_rank == 1 and ronpo_ifeval_rank is not None and ronpo_ifeval_rank <= 2
        and ronpo_macro_rank is not None and ronpo_macro_rank <= 2 and no_regression is True
    )
    lines.extend([
        "", "## Honest verdict", "",
        f"- RONPO worst-objective rank: `{ronpo_worst_rank if ronpo_worst_rank is not None else 'unknown'}`.",
        (f"- Non-sealed selection diagnostic (not the headline test): RONPO full-expect validation "
         f"worst-objective rank `{validation_ronpo['validation_worst_objective_rank']}`, score "
         f"`{float(validation_ronpo['mean_primary_prompt_worst_norm_score']):.6f}`."
         if validation_ronpo else
         "- Non-sealed selection diagnostic: `unknown`."),
        f"- RONPO IFEval rank: `{ronpo_ifeval_rank if ronpo_ifeval_rank is not None else 'unknown'}`.",
        f"- RONPO accessible-suite macro rank: `{ronpo_macro_rank if ronpo_macro_rank is not None else 'unknown'}`.",
        f"- Accessible-suite macro no-regression vs base: `{str(no_regression).lower() if no_regression is not None else 'unknown'}`.",
        f"- Strict per-task regressions vs base: `{', '.join(strict_regressions) if strict_regressions else 'none measured'}`.",
        f"- Requested combined success criterion: `{str(success).lower()}` (requires measured worst rank 1, IFEval and capability macro rank <=2, and macro no-regression).",
        "- GPQA: `BLOCKED` because the official dataset is gated; no mirror or imputed value is used.",
        "", "## Machine-readable sources", "",
        f"- Sealed ranks and CIs: `{sealed_summary_path}`",
        f"- Sealed per-head scores: `{args.sealed_dir / 'per_objective_scores.csv'}`",
        f"- IFEval: `{args.ifeval_json}`",
        f"- Academic table: `{args.academic_dir / 'benchmark_table.csv'}`",
        f"- Per-cell academic source JSON paths: `{args.academic_dir / 'metrics_long.csv'}`",
        f"- GPQA blocker: `{args.blockers}`; evidence `{blocker.get('evidence_log', 'unknown')}`",
        f"- Non-sealed model-selection reward summary: `{args.validation_summary or 'unknown'}`",
        "", "## Per-cell academic provenance", "",
        "| Method | Benchmark | Status | Value (%) | Source JSON / blocker evidence |",
        "| --- | --- | --- | ---: | --- |",
    ])
    for row in long_rows:
        value = row.get("score_percent") or "BLOCKED"
        source = row.get("source_json") or row.get("blocker_evidence") or "UNKNOWN"
        lines.append(f"| {row['method']} | {row['benchmark']} | {row.get('status', 'MEASURED')} | {value} | `{source}` |")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    if args.latex_output:
        latex_rows = []
        for method in METHODS:
            sealed = sealed_rows.get(method, {})
            fields = [
                method.replace("_", r"\_"),
                fmt(sealed.get("mean_primary_prompt_worst_norm_score")),
                fmt(sealed.get("mean_primary_prompt_avg_norm_score")),
                fmt(per_head.get((method, "helpfulness"), {}).get("mean_raw_score")),
                fmt(per_head.get((method, "safety"), {}).get("mean_raw_score")),
                fmt(per_head.get((method, "conciseness"), {}).get("mean_raw_score")),
                fmt(ifeval_rows.get(method, {}).get("ifeval_prompt_strict_percent"), 2),
                fmt(capability_macro.get(method), 2),
            ]
            fields = ["--" if value == "UNKNOWN" else value for value in fields]
            latex_rows.append(" & ".join(fields) + r" \\")
        args.latex_output.parent.mkdir(parents=True, exist_ok=True)
        args.latex_output.write_text("\n".join(latex_rows) + "\n")


if __name__ == "__main__":
    main()
