#!/usr/bin/env python3
"""Aggregate star-shaped RONPO pairwise judgments with item bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPECTED = {"alpaca_eval_2": 805, "arena_hard_v0.1": 500, "mt_bench": 80}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ronpo_score(row: dict) -> float:
    if row["winner"] == "tie":
        return 0.5
    ronpo_winner = "A" if row["order"] == "ronpo_A" else "B"
    return 1.0 if row["winner"] == ronpo_winner else 0.0


def interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(2000, len(values)))
    samples = values[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-dir", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    rows = []
    for path in sorted(args.judge_dir.glob("shard_*.jsonl")):
        rows.extend(load_jsonl(path))
    expected_judgments = protocol["comparisons"]["expected_judgments"]
    if len(rows) != expected_judgments:
        raise RuntimeError(f"expected {expected_judgments} judgments, got {len(rows)}")
    invalid = [row for row in rows if not row.get("valid")]
    if invalid:
        raise RuntimeError(f"fail closed: {len(invalid)} invalid judge outputs")

    by_pair: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_pair[(row["opponent"], row["benchmark"], row["item_id"])].append(row)
    pair_rows = []
    for (opponent, benchmark, item_id), judgments in sorted(by_pair.items()):
        if len(judgments) != 2 or {x["order"] for x in judgments} != {"ronpo_A", "ronpo_B"}:
            raise RuntimeError(f"missing position swap: {(opponent, benchmark, item_id)}")
        scores = [ronpo_score(row) for row in judgments]
        pair_rows.append({
            "opponent": opponent, "benchmark": benchmark, "item_id": item_id,
            "ronpo_pair_score": sum(scores) / 2,
            "position_agreement": scores[0] == scores[1],
            "ronpo_A_score": next(ronpo_score(x) for x in judgments if x["order"] == "ronpo_A"),
            "ronpo_B_score": next(ronpo_score(x) for x in judgments if x["order"] == "ronpo_B"),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "paired_item_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    summaries = []
    rng = np.random.default_rng(42)
    opponents = protocol["comparisons"]["opponents"]
    for opponent in opponents:
        for benchmark in list(EXPECTED) + ["macro_average", "pooled_items"]:
            selected = [row for row in pair_rows if row["opponent"] == opponent]
            if benchmark in EXPECTED:
                selected = [row for row in selected if row["benchmark"] == benchmark]
                if len(selected) != EXPECTED[benchmark]:
                    raise RuntimeError(f"count mismatch for {opponent}/{benchmark}: {len(selected)}")
                values = np.asarray([row["ronpo_pair_score"] for row in selected], dtype=float)
                low, high = interval(values, rng)
                mean = float(values.mean())
                agreement = float(np.mean([row["position_agreement"] for row in selected]))
                n = len(values)
            elif benchmark == "pooled_items":
                values = np.asarray([row["ronpo_pair_score"] for row in selected], dtype=float)
                low, high = interval(values, rng)
                mean = float(values.mean())
                agreement = float(np.mean([row["position_agreement"] for row in selected]))
                n = len(values)
            else:
                grouped = {
                    name: np.asarray(
                        [row["ronpo_pair_score"] for row in selected if row["benchmark"] == name],
                        dtype=float,
                    )
                    for name in EXPECTED
                }
                means = np.asarray([grouped[name].mean() for name in EXPECTED])
                mean = float(means.mean())
                boot = []
                for _ in range(2000):
                    boot.append(np.mean([
                        values[rng.integers(0, len(values), size=len(values))].mean()
                        for values in grouped.values()
                    ]))
                low, high = [float(x) for x in np.quantile(boot, [0.025, 0.975])]
                agreement = float(np.mean([row["position_agreement"] for row in selected]))
                n = len(selected)
            summaries.append({
                "candidate": "ronpo_k_only", "opponent": opponent, "benchmark": benchmark,
                "n_items": n, "proxy_win_rate": mean,
                "ci95_low": low, "ci95_high": high,
                "position_agreement": agreement,
            })

    strict_point_win = all(
        row["proxy_win_rate"] > 0.5 for row in summaries if row["benchmark"] == "macro_average"
    )
    confident_win = all(
        row["ci95_low"] > 0.5 for row in summaries if row["benchmark"] == "macro_average"
    )
    result = {
        "score_label": protocol["score_label"],
        "candidate": "ronpo_k_only",
        "strictly_beats_every_opponent_by_macro_point_estimate": strict_point_win,
        "ci95_lower_bound_above_0.5_for_every_opponent": confident_win,
        "num_raw_judgments": len(rows),
        "num_paired_items": len(pair_rows),
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 42,
        "comparisons": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    macro = sorted(
        (row for row in summaries if row["benchmark"] == "macro_average"),
        key=lambda row: row["proxy_win_rate"], reverse=True,
    )
    lines = [
        "# Zero-cost open-weight benchmark proxy",
        "",
        f"This is **not an official AlpacaEval 2, Arena-Hard, or MT-Bench score**. {protocol['score_label']}.",
        "",
        "| RONPO candidate | Opponent | Macro proxy WR | 95% CI | Position agreement |",
        "|---|---|---:|---:|---:|",
    ]
    for row in macro:
        lines.append(
            f"| ronpo_k_only | {row['opponent']} | {100*row['proxy_win_rate']:.2f} | "
            f"[{100*row['ci95_low']:.2f}, {100*row['ci95_high']:.2f}] | {100*row['position_agreement']:.2f} |"
        )
    lines.extend([
        "",
        f"Strict point-estimate win against every opponent: **{strict_point_win}**.",
        f"All 95% CI lower bounds above 50%: **{confident_win}**.",
    ])
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
