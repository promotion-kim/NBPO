#!/usr/bin/env python3
"""Summarize raw P10 Stage-1 Beaver scores on the fixed P8 panel.

The diagnostic deliberately uses raw scores, not per-prompt normalized ranks.
It describes policy movement and is not a method-selection result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ARMS = ["ronpo_os", "ronpo_topmass", "inpo_avg", "sppo_avg", "simpo", "ipo", "dpo", "ht_mnpo_harmless", "ht_mnpo_helpfulness"]


def load_single(path: Path) -> dict[str, float]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        names, scores = row["response_model_names"], row["all_rm_scores"]
        if len(names) != 1 or len(scores) != 1:
            raise RuntimeError(f"not a single-policy score file: {path}")
        values[str(row["prompt_id"])] = float(scores[0])
    if len(values) != 1000:
        raise RuntimeError(f"{path}: expected 1000 unique prompts, found {len(values)}")
    return values


def base_from_merged(path: Path) -> dict[str, float]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        index = row["response_model_names"].index("base")
        values[str(row["prompt_id"])] = float(row["all_rm_scores"][index])
    if len(values) != 1000:
        raise RuntimeError(f"{path}: expected 1000 base prompt scores, found {len(values)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    out = args.experiment / "stage1_eval_p8_locked_panel" / "raw_beaver_diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    base_scores = {
        objective: base_from_merged(args.experiment / "stage2_eval_p8_locked_panel" / "scores" / f"{objective}.jsonl")
        for objective in ("helpfulness", "harmlessness")
    }
    prompt_ids = sorted(base_scores["helpfulness"])
    if prompt_ids != sorted(base_scores["harmlessness"]):
        raise RuntimeError("base scorer prompt IDs do not align")
    rows = []
    for arm in ARMS:
        row = {"model": arm}
        for objective in ("helpfulness", "harmlessness"):
            values = load_single(args.experiment / "stage1_eval_p8_locked_panel" / "scores_individual" / arm / f"{objective}.jsonl")
            if sorted(values) != prompt_ids:
                raise RuntimeError(f"{arm}/{objective}: prompt IDs do not align with base")
            deltas = [values[key] - base_scores[objective][key] for key in prompt_ids]
            row[f"{objective}_raw"] = sum(values.values()) / len(values)
            row[f"{objective}_delta_vs_base"] = sum(deltas) / len(deltas)
        rows.append(row)
    payload = {
        "status": "completed",
        "scope": "Raw Beaver Stage-1 diagnostic on the already-open P8 1,000-prompt panel; no normalization, selection, or fresh evaluation.",
        "base_raw_means": {key: sum(value.values()) / len(value) for key, value in base_scores.items()},
        "rows": rows,
        "spent_sealed_split_touched": False,
    }
    out.joinpath("summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["# P10 Stage-1 raw Beaver diagnostic", "", payload["scope"], "", "| Method | Help. Δ vs Base | Harmless Δ vs Base |", "|---|---:|---:|"]
    lines.extend(f"| {row['model']} | {row['helpfulness_delta_vs_base']:+.4f} | {row['harmlessness_delta_vs_base']:+.4f} |" for row in rows)
    out.joinpath("REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
