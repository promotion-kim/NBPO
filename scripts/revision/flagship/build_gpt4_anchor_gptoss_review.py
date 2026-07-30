#!/usr/bin/env python3
"""Build an honest review from measured GPT-4-anchor proxy artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(value: float) -> str:
    return f"{100 * value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arena-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.summary.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    arena_protocol = json.loads(args.arena_protocol.read_text(encoding="utf-8"))
    rows = result["summaries"]
    lookup = {(row["model"], row["benchmark"]): row for row in rows}

    def leader(benchmark: str, field: str = "score") -> dict:
        return max((row for row in rows if row["benchmark"] == benchmark), key=lambda row: row[field])

    top = lookup[("ronpo_k_only", "alpaca_eval_2")]
    full = lookup[("ronpo_full_expect", "alpaca_eval_2")]
    arena_top = lookup[("ronpo_k_only", "arena_hard_v0.1")]
    arena_full = lookup[("ronpo_full_expect", "arena_hard_v0.1")]
    mt_top = lookup[("ronpo_k_only", "mt_bench")]
    mt_full = lookup[("ronpo_full_expect", "mt_bench")]
    alpaca_leader = leader("alpaca_eval_2")
    lc_leader = leader("alpaca_eval_2", "length_controlled_win_rate")
    arena_leader = leader("arena_hard_v0.1")
    mt_leader = leader("mt_bench")

    lines = [
        "# Review: GPT-4-anchor / gpt-oss-120b zero-cost proxy", "",
        "## Outcome", "",
        "The desired outcome was **not measured**: neither RONPO estimator ranks first on AlpacaEval 2, "
        "Arena-Hard v0.1, or MT-Bench under this frozen open-weight proxy.", "",
        "| RONPO estimator | Alpaca symmetric WR | Alpaca LC proxy | Arena anchor WR | MT-Bench /10 |",
        "|---|---:|---:|---:|---:|",
        f"| top-mass (`ronpo_k_only`) | {pct(top['score'])} [{pct(top['ci95_low'])}, {pct(top['ci95_high'])}] (#{top['rank']}) | "
        f"{pct(top['length_controlled_win_rate'])} (#{top['lc_rank']}) | "
        f"{pct(arena_top['score'])} [{pct(arena_top['ci95_low'])}, {pct(arena_top['ci95_high'])}] (#{arena_top['rank']}) | "
        f"{mt_top['score']:.3f} [{mt_top['ci95_low']:.3f}, {mt_top['ci95_high']:.3f}] (#{mt_top['rank']}) |",
        f"| full-expectation | {pct(full['score'])} [{pct(full['ci95_low'])}, {pct(full['ci95_high'])}] (#{full['rank']}) | "
        f"{pct(full['length_controlled_win_rate'])} (#{full['lc_rank']}) | "
        f"{pct(arena_full['score'])} [{pct(arena_full['ci95_low'])}, {pct(arena_full['ci95_high'])}] (#{arena_full['rank']}) | "
        f"{mt_full['score']:.3f} [{mt_full['ci95_low']:.3f}, {mt_full['ci95_high']:.3f}] (#{mt_full['rank']}) |",
        "", "Measured leaders:", "",
        f"- Alpaca symmetric raw: `{alpaca_leader['model']}` at {pct(alpaca_leader['score'])}.",
        f"- Alpaca length-controlled proxy: `{lc_leader['model']}` at {pct(lc_leader['length_controlled_win_rate'])}.",
        f"- Arena GPT-4 anchor proxy: `{arena_leader['model']}` at {pct(arena_leader['score'])}.",
        f"- MT-Bench absolute score: `{mt_leader['model']}` at {mt_leader['score']:.3f}/10.",
        "", "The prompt-bootstrap intervals overlap broadly, so this run does not support a statistical superiority claim.",
        "", "## Protocol and provenance", "",
        f"- Judge: `{protocol['judge']['model']}` revision `{protocol['judge']['revision']}`, reasoning effort "
        f"`{protocol['judge']['reasoning_effort']}`, deterministic temperature {protocol['judge']['temperature']}, seed {protocol['judge']['seed']}.",
        f"- Alpaca: {protocol['references']['alpaca_eval_2']['count']} public GPT-4-turbo references; source SHA-256 "
        f"`{protocol['references']['alpaca_eval_2']['source_sha256']}`.",
        f"- Arena: {protocol['references']['arena_hard_v0.1']['count']} public GPT-4-0314 references; source SHA-256 "
        f"`{protocol['references']['arena_hard_v0.1']['source_sha256']}`.",
        f"- MT-Bench: {protocol['references']['mt_bench']['count']} questions and "
        f"{protocol['references']['mt_bench']['reference_count']} GPT-4 reference-answer questions; reference SHA-256 "
        f"`{protocol['references']['mt_bench']['source_sha256']}`.",
        f"- Final inputs: {result['num_judgments']:,} effective judgments; 2,000 prompt bootstrap resamples, seed 42.",
        f"- RONPO top-mass position agreement: Alpaca {pct(top['position_agreement'])}%, Arena {pct(arena_top['position_agreement'])}%.",
        f"- Alpaca LC `df_gamed.csv`: revision `{result['alpaca_length_control']['df_gamed_revision']}`, SHA-256 "
        f"`{result['alpaca_length_control']['df_gamed_sha256']}`.",
        f"- Final Arena adaptation lock: `{arena_protocol['configuration_sha256']}`. "
        "It was frozen before aggregation and reran all models/positions; no invalid output was assigned a score.",
        f"- W&B aggregate: [{result['wandb_run_id']}]({result['wandb_url']}).", "",
        "## Interpretation", "",
        "These are **not official AlpacaEval 2, Arena-Hard, or MT-Bench scores**. The public GPT-4 anchors are official/public "
        "artifacts, but the closed judge was replaced by `gpt-oss-120b`; no validated equivalence to GPT-5.4-mini exists. "
        "Arena additionally required a fully symmetric verdict-only adaptation to obtain fail-closed parse completeness. "
        "Use this as a zero-cost diagnostic or appendix result, not as a drop-in replacement for MNPO Table 2.", "",
        "Because RONPO is not first here, selecting only favorable qualitative examples would not repair the quantitative result "
        "and would be misleading. The stronger paper evidence remains the separately measured conflict-objective robustness/IFEval study.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
