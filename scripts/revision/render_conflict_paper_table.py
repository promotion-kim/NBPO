#!/usr/bin/env python3
"""Render complete conflict-evaluation results into Markdown and LaTeX rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


LABELS = {
    "base": "Base",
    "dpo_b0p01": "DPO",
    "ipo_b0p05": "IPO",
    "simpo_b2_g0p6": "SimPO",
    "sppo_eta0p0075": "SPPO",
    "inpo_eta0p0075": "INPO",
    "ronpo": "RONPO",
}
ORDER = list(LABELS)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def value(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--collapse-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-tex", required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    summaries = {row["model"]: row for row in rows(result_dir / "model_summary.csv")}
    per_objective = {
        (row["model"], row["objective"]): row
        for row in rows(result_dir / "per_objective_scores.csv")
    }
    collapse = {row["model"]: row for row in rows(Path(args.collapse_csv))}
    missing = [name for name in ORDER if name not in summaries]
    if missing:
        raise RuntimeError(f"missing expected models: {missing}")
    objective_names = {objective for _, objective in per_objective}
    expected = {"helpfulness", "safety", "brevity"}
    if objective_names != expected:
        raise RuntimeError(f"expected objectives {sorted(expected)}, got {sorted(objective_names)}")

    metric_keys = {
        "help": lambda name: value(per_objective[(name, "helpfulness")], "mean_prompt_norm_score"),
        "safe": lambda name: value(per_objective[(name, "safety")], "mean_prompt_norm_score"),
        "brief": lambda name: value(per_objective[(name, "brevity")], "mean_prompt_norm_score"),
        "avg": lambda name: value(summaries[name], "mean_primary_prompt_avg_norm_score"),
        "worst": lambda name: value(summaries[name], "mean_primary_prompt_worst_norm_score"),
        "all_worst": lambda name: value(summaries[name], "mean_prompt_worst_norm_score"),
    }
    best = {key: max(fn(name) for name in ORDER) for key, fn in metric_keys.items()}

    def formatted(name: str, key: str, latex: bool = False) -> str:
        number = metric_keys[key](name)
        text = f"{number:.3f}"
        if abs(number - best[key]) < 5e-7:
            return f"\\textbf{{{text}}}" if latex else f"**{text}**"
        return text

    markdown = [
        "| Method | Help | Safety | Brevity | H--B Avg | H--B Worst [95% CI] | All-3 Worst | Mean words |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    latex: list[str] = []
    for name in ORDER:
        summary = summaries[name]
        low = value(summary, "mean_primary_prompt_worst_norm_score_ci95_low")
        high = value(summary, "mean_primary_prompt_worst_norm_score_ci95_high")
        words = value(collapse[name], "mean_words")
        worst_md = f"{formatted(name, 'worst')} [{low:.3f}, {high:.3f}]"
        worst_tex = f"{formatted(name, 'worst', True)} [{low:.3f}, {high:.3f}]"
        markdown.append(
            "| {label} | {help} | {safe} | {brief} | {avg} | {worst} | {all_worst} | {words:.1f} |".format(
                label=LABELS[name],
                help=formatted(name, "help"),
                safe=formatted(name, "safe"),
                brief=formatted(name, "brief"),
                avg=formatted(name, "avg"),
                worst=worst_md,
                all_worst=formatted(name, "all_worst"),
                words=words,
            )
        )
        latex.append(
            "{label} & {help} & {safe} & {brief} & {avg} & {worst} & {all_worst} & {words:.1f} \\\\".format(
                label=LABELS[name],
                help=formatted(name, "help", True),
                safe=formatted(name, "safe", True),
                brief=formatted(name, "brief", True),
                avg=formatted(name, "avg", True),
                worst=worst_tex,
                all_worst=formatted(name, "all_worst", True),
                words=words,
            )
        )

    Path(args.output_md).write_text("\n".join(markdown) + "\n", encoding="utf-8")
    Path(args.output_tex).write_text("\n".join(latex) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
