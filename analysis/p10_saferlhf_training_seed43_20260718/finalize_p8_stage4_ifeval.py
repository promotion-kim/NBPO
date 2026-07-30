#!/usr/bin/env python3
"""Aggregate locked Stage-4 IFEval artifacts without selecting a model."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_p8_stage4_ifeval import MODELS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.output.joinpath("ifeval_lock.json").read_text(encoding="utf-8"))
    locked_models = lock.get("requested_models", lock.get("all_stage4_appendix_models"))
    if not isinstance(locked_models, list) or not all(isinstance(name, str) for name in locked_models):
        raise RuntimeError("IFEval lock has no valid immutable model list")
    rows, missing = [], []
    for name in locked_models:
        path = args.output / name / "results.json"
        if not path.exists():
            missing.append(name)
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "completed" or not isinstance(value.get("score"), (float, int)):
            raise RuntimeError(f"invalid IFEval result: {path}")
        rows.append({"method": name, "ifeval_prompt_strict": float(value["score"]), "source": value["source"]})
    if missing:
        raise RuntimeError(f"incomplete locked IFEval cohort: {missing}")
    base = next(row["ifeval_prompt_strict"] for row in rows if row["method"] == "base")
    for row in rows:
        row["delta_vs_base_pp"] = 100.0 * (row["ifeval_prompt_strict"] - base)
        row["ifeval_prompt_strict_percent"] = 100.0 * row["ifeval_prompt_strict"]
    args.output.joinpath("results.json").write_text(json.dumps({
        "status": "completed", "metric": "IFEval prompt_level_strict_acc", "num_fewshot": 0,
        "rows": rows, "spent_sealed_split_touched": False,
    }, indent=2) + "\n", encoding="utf-8")
    with args.output.joinpath("results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ["# P8 Stage-4 IFEval", "", "| Method | Prompt-strict | Δ vs Base (pp) |", "|---|---:|---:|"]
    lines.extend(f"| {row['method']} | {row['ifeval_prompt_strict_percent']:.2f} | {row['delta_vs_base_pp']:+.2f} |" for row in rows)
    lines.append("")
    lines.append("All rows were fixed in `ifeval_lock.json` before evaluation. This is a capability measurement, not a selection criterion.")
    args.output.joinpath("REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
