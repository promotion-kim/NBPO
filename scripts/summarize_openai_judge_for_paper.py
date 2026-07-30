#!/usr/bin/env python3
"""Create paper-ready summaries from OpenAI pairwise judge artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


DISPLAY = {
    "baseline": "Base",
    "htmnpo_skywork_s2": "HT-MNPO Skywork S2",
    "htmnpo_athene_s2": "HT-MNPO Athene S2",
    "htmnpo_armo_s2": "HT-MNPO ArmoRM S2",
    "ronpo_s2_ckpt1400": "RONPO S2 checkpoint-1400",
    "ronpo_s2_ckpt2457": "RONPO S2 checkpoint-2457",
}
ORDER = [
    "baseline",
    "htmnpo_skywork_s2",
    "htmnpo_athene_s2",
    "htmnpo_armo_s2",
    "ronpo_s2_ckpt1400",
    "ronpo_s2_ckpt2457",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(xs: list[float], q: float) -> float:
    vals = sorted(xs)
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def fmt(x: float) -> str:
    if x != x:
        return "-"
    return f"{x:.4f}"


def ci_text(est: float, lo: float, hi: float) -> str:
    return f"{fmt(est)} [{fmt(lo)}, {fmt(hi)}]"


def load_prompt_subset(path: Path | None, limit: int | None = None) -> set[str] | None:
    if path is None:
        return None
    ids: list[str] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ids.append(row["prompt_id"])
            if limit is not None and len(ids) >= limit:
                break
    return set(ids)


def load_judgments(path: Path, subset_ids: set[str] | None = None) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(path):
        if subset_ids is not None and row["prompt_id"] not in subset_ids:
            continue
        if row.get("winner_model") == "parse_error":
            continue
        try:
            left_score = float(row["left_score"])
        except (TypeError, ValueError):
            continue
        if left_score != left_score:
            continue
        rows.append(
            {
                "prompt_id": row["prompt_id"],
                "left_model": row["left_model"],
                "right_model": row["right_model"],
                "winner_model": row["winner_model"],
                "left_score": left_score,
                "confidence": float(row["confidence"]),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, float]]]:
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[(row["left_model"], row["right_model"])].append(row)

    pair_rows: list[dict[str, Any]] = []
    model_scores: dict[str, list[float]] = defaultdict(list)
    pair_wr: dict[str, dict[str, float]] = defaultdict(dict)
    for left, right in sorted(by_pair):
        vals = by_pair[(left, right)]
        scores = [row["left_score"] for row in vals]
        left_wr = sum(scores) / len(scores)
        right_wr = 1.0 - left_wr
        tie_rate = sum(1 for row in vals if row["winner_model"] == "tie") / len(vals)
        mean_conf = statistics.mean(row["confidence"] for row in vals)
        pair_rows.append(
            {
                "left_model": left,
                "right_model": right,
                "n": len(vals),
                "left_win_rate": left_wr,
                "right_win_rate": right_wr,
                "tie_rate": tie_rate,
                "mean_confidence": mean_conf,
            }
        )
        model_scores[left].append(left_wr)
        model_scores[right].append(right_wr)
        pair_wr[left][right] = left_wr
        pair_wr[right][left] = right_wr

    scoreboard = []
    for model in ORDER:
        scores = model_scores.get(model, [])
        scoreboard.append(
            {
                "model": model,
                "display": DISPLAY.get(model, model),
                "mean_pairwise_win_rate": sum(scores) / len(scores) if scores else float("nan"),
                "num_matchups": len(scores),
            }
        )
    scoreboard.sort(key=lambda row: row["mean_pairwise_win_rate"], reverse=True)
    return scoreboard, pair_rows, pair_wr


def bootstrap(rows: list[dict[str, Any]], iterations: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[row["prompt_id"]].append(row)
    prompt_ids = sorted(by_prompt)
    rng = random.Random(seed)

    observed_scoreboard, observed_pairs, _ = summarize(rows)
    model_samples: dict[str, list[float]] = defaultdict(list)
    pair_samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    for _ in range(iterations):
        sample_rows: list[dict[str, Any]] = []
        for pid in rng.choices(prompt_ids, k=len(prompt_ids)):
            sample_rows.extend(by_prompt[pid])
        sample_scoreboard, sample_pairs, _ = summarize(sample_rows)
        for row in sample_scoreboard:
            model_samples[row["model"]].append(row["mean_pairwise_win_rate"])
        for row in sample_pairs:
            pair_samples[(row["left_model"], row["right_model"])].append(row["left_win_rate"])

    model_ci = []
    for row in observed_scoreboard:
        samples = model_samples[row["model"]]
        model_ci.append(
            {
                **row,
                "ci_low": percentile(samples, 0.025),
                "ci_high": percentile(samples, 0.975),
                "bootstrap_iterations": iterations,
            }
        )
    pair_ci = []
    for row in observed_pairs:
        samples = pair_samples[(row["left_model"], row["right_model"])]
        pair_ci.append(
            {
                **row,
                "ci_low": percentile(samples, 0.025),
                "ci_high": percentile(samples, 0.975),
                "bootstrap_iterations": iterations,
            }
        )
    return model_ci, pair_ci


def make_report(
    out_dir: Path,
    full_models: list[dict[str, Any]],
    full_pairs: list[dict[str, Any]],
    stress_models: list[dict[str, Any]],
    stress_pairs: list[dict[str, Any]],
    coverage: dict[str, Any],
) -> str:
    lines = [
        "# GPT-5.5 Judge Evaluation for RONPO Stage 2",
        "",
        f"Artifact directory: `{out_dir}`",
        "",
        "## Protocol",
        "",
        "- Judge: `gpt-5.5-2026-04-23` via OpenAI Batch API.",
        "- Compared models: Base, HT-MNPO Skywork/Athene/ArmoRM S2, RONPO S2 checkpoint-1400, RONPO S2 checkpoint-2457.",
        "- Prompt set: 647 held-out UltraFeedback prompts from the existing stage-2 generation artifact.",
        "- Comparison: all pairwise model pairs per prompt; response order deterministically randomized per prompt-pair.",
        "- Win rate: ties count as 0.5; confidence intervals are prompt-level paired bootstrap intervals.",
        "",
        "## Coverage",
        "",
        f"- Expected full judgments: `{coverage['expected_full']}`.",
        f"- Parsed full judgments: `{coverage['parsed_full']}`.",
        f"- Missing/failed full judgments: `{coverage['missing_full']}`.",
        f"- Parsed high-disagreement judgments: `{coverage['parsed_stress']}`.",
        "",
        "## Full Held-Out Set",
        "",
        "| Rank | Model | Mean pairwise win rate | 95% CI | Matchups |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(full_models, 1):
        lines.append(
            f"| {rank} | {row['display']} | {fmt(row['mean_pairwise_win_rate'])} | "
            f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | {row['num_matchups']} |"
        )
    lines.extend(
        [
            "",
            "### Key Pairwise Results",
            "",
            "| Pair | n | Left WR | 95% CI | Tie |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    wanted = {
        ("baseline", "ronpo_s2_ckpt2457"),
        ("baseline", "ronpo_s2_ckpt1400"),
        ("htmnpo_skywork_s2", "ronpo_s2_ckpt2457"),
        ("htmnpo_athene_s2", "ronpo_s2_ckpt2457"),
        ("htmnpo_armo_s2", "ronpo_s2_ckpt2457"),
        ("ronpo_s2_ckpt1400", "ronpo_s2_ckpt2457"),
    }
    for row in full_pairs:
        key = (row["left_model"], row["right_model"])
        if key not in wanted:
            continue
        pair = f"{DISPLAY.get(row['left_model'], row['left_model'])} vs {DISPLAY.get(row['right_model'], row['right_model'])}"
        lines.append(
            f"| {pair} | {row['n']} | {fmt(row['left_win_rate'])} | "
            f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | {fmt(row['tie_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## High-Disagreement Stress Subset",
            "",
            "| Rank | Model | Mean pairwise win rate | 95% CI | Matchups |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for rank, row in enumerate(stress_models, 1):
        lines.append(
            f"| {rank} | {row['display']} | {fmt(row['mean_pairwise_win_rate'])} | "
            f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | {row['num_matchups']} |"
        )
    lines.extend(
        [
            "",
            "## Conservative Interpretation",
            "",
            "The full held-out GPT-5.5 judge evaluation is reward-model-independent evidence for stage-2 preference quality. "
            "It should be reported separately from local reward-model tables because the evaluator and metric differ. "
            "If coverage is complete or near-complete after retries, the result is appropriate for a main or appendix paper table.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stress-csv", default="analysis/stage2_robustness_20260626/disagreement_prompts.csv")
    parser.add_argument("--stress-limit", type=int, default=162)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--expected-full", type=int, default=9705)
    parser.add_argument("--seed", type=int, default=20260627)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    full_rows = load_judgments(Path(args.judgments))
    stress_rows = load_judgments(Path(args.judgments), load_prompt_subset(Path(args.stress_csv), args.stress_limit))

    full_models, full_pairs = bootstrap(full_rows, args.bootstrap_iterations, args.seed)
    stress_models, stress_pairs = bootstrap(stress_rows, args.bootstrap_iterations, args.seed + 1)
    coverage = {
        "expected_full": args.expected_full,
        "parsed_full": len(full_rows),
        "missing_full": args.expected_full - len(full_rows),
        "parsed_stress": len(stress_rows),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "full_model_scoreboard_ci.csv", full_models)
    write_csv(out_dir / "full_pairwise_ci.csv", full_pairs)
    write_csv(out_dir / "stress_model_scoreboard_ci.csv", stress_models)
    write_csv(out_dir / "stress_pairwise_ci.csv", stress_pairs)
    (out_dir / "report.md").write_text(
        make_report(out_dir, full_models, full_pairs, stress_models, stress_pairs, coverage),
        encoding="utf-8",
    )
    print((out_dir / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
