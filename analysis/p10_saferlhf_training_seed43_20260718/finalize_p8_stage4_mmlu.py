#!/usr/bin/env python3
"""Aggregate the immutable Stage-4 MMLU cohort."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    lock = json.loads((args.output / "capability_lock.json").read_text(encoding="utf-8"))
    rows, missing = [], []
    for name in lock["requested_models"]:
        path = args.output / name / "result.json"
        if not path.exists(): missing.append(name); continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "completed" or not isinstance(result.get("score"), (float, int)):
            raise RuntimeError(f"invalid result: {path}")
        rows.append({"method": name, "mmlu_acc": float(result["score"]), "source": result["source"]})
    if missing: raise RuntimeError(f"incomplete locked cohort: {missing}")
    base = next(r["mmlu_acc"] for r in rows if r["method"] == "base")
    for row in rows:
        row["mmlu_percent"] = 100 * row["mmlu_acc"]; row["delta_vs_base_pp"] = 100 * (row["mmlu_acc"] - base)
    (args.output / "results.json").write_text(json.dumps({"status": "completed", "task": "mmlu", "num_fewshot": 5,
        "rows": rows, "spent_sealed_split_touched": False}, indent=2) + "\n", encoding="utf-8")
    with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = ["# P8 Stage-4 MMLU", "", "| Method | Accuracy | Δ vs Base (pp) |", "|---|---:|---:|"]
    lines.extend(f"| {r['method']} | {r['mmlu_percent']:.2f} | {r['delta_vs_base_pp']:+.2f} |" for r in rows)
    lines.extend(["", "All rows were fixed in `capability_lock.json` before evaluation. This is a capability measurement, not a selection criterion."])
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
