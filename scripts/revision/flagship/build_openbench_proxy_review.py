#!/usr/bin/env python3
"""Build a compact review report only from measured proxy artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def pct(value: float) -> str:
    return f"{100 * value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.root / "results/summary.json").read_text(encoding="utf-8"))
    protocol = json.loads((args.root / "protocol_lock.json").read_text(encoding="utf-8"))
    pipeline = json.loads((args.root / "pipeline_status.json").read_text(encoding="utf-8"))
    with (args.root / "results/paired_item_scores.csv").open(encoding="utf-8") as handle:
        paired = list(csv.DictReader(handle))
    comparisons = summary["comparisons"]

    def rows(benchmark: str) -> list[dict]:
        return sorted(
            (row for row in comparisons if row["benchmark"] == benchmark),
            key=lambda row: row["proxy_win_rate"], reverse=True,
        )

    lines = [
        "# Qwen3-8B open-benchmark local-judge proxy review",
        "",
        "**Verdict: RONPO top-mass is not the best model under this measured proxy.** "
        "It does not beat every baseline by the equal-benchmark macro point estimate. This is not an official AlpacaEval 2, Arena-Hard, or MT-Bench score.",
        "",
        "## Macro head-to-head result",
        "",
        "| Candidate | Opponent | RONPO proxy WR | 95% item-bootstrap CI | Position agreement |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows("macro_average"):
        lines.append(
            f"| ronpo_k_only | {row['opponent']} | {pct(row['proxy_win_rate'])} | "
            f"[{pct(row['ci95_low'])}, {pct(row['ci95_high'])}] | {pct(row['position_agreement'])} |"
        )

    lines.extend(["", "## Benchmark-specific head-to-head proxy WR", ""])
    opponents = protocol["comparisons"]["opponents"]
    lookup = {(row["benchmark"], row["opponent"]): row for row in comparisons}
    lines.extend([
        "| Opponent | AlpacaEval 2 | Arena-Hard v0.1 | MT-Bench | Equal-benchmark macro |",
        "|---|---:|---:|---:|---:|",
    ])
    for opponent in opponents:
        lines.append(
            f"| {opponent} | {pct(lookup[('alpaca_eval_2', opponent)]['proxy_win_rate'])} | "
            f"{pct(lookup[('arena_hard_v0.1', opponent)]['proxy_win_rate'])} | "
            f"{pct(lookup[('mt_bench', opponent)]['proxy_win_rate'])} | "
            f"{pct(lookup[('macro_average', opponent)]['proxy_win_rate'])} |"
        )

    lines.extend(["", "## Position-bias audit", ""])
    for benchmark in ("all", "alpaca_eval_2", "arena_hard_v0.1", "mt_bench"):
        selected = paired if benchmark == "all" else [row for row in paired if row["benchmark"] == benchmark]
        a = sum(float(row["ronpo_A_score"]) for row in selected) / len(selected)
        b = sum(float(row["ronpo_B_score"]) for row in selected) / len(selected)
        pair = sum(float(row["ronpo_pair_score"]) for row in selected) / len(selected)
        agreement = sum(row["position_agreement"] == "True" for row in selected) / len(selected)
        lines.append(
            f"- `{benchmark}`: RONPO-as-A {pct(a)}, RONPO-as-B {pct(b)}, symmetric pair mean {pct(pair)}, position agreement {pct(agreement)} (n={len(selected)})."
        )

    lines.extend([
        "",
        "The large first-position bias is why no one-sided judgment is reported. Every item score is the mean of its two A/B-swapped judgments. The low agreement also limits how strongly this single-judge proxy can be generalized.",
        "",
        "## Provenance",
        "",
        f"- Prompts: {protocol['prompt_manifest']['counts']} (SHA-256 `{protocol['official_prompts_sha256']}`).",
        f"- Judge: `{protocol['judge']['model']}` at revision `{protocol['judge']['revision']}`.",
        f"- Raw judgments: {summary['num_raw_judgments']}; paired items: {summary['num_paired_items']}; bootstrap: {summary['bootstrap_resamples']} resamples, seed {summary['bootstrap_seed']}.",
        f"- W&B: [{pipeline['wandb_run_id']}]({pipeline['wandb_url']}).",
        "- Exact generation, judge, source-revision, and model-revision settings are in `protocol_lock.json`.",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
