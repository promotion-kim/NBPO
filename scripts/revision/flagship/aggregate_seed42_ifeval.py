#!/usr/bin/env python3
"""Create measured IFEval CSV/JSON/LaTeX artifacts from lm-eval outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def score(model_root: Path) -> tuple[float, str] | None:
    found = []
    for path in sorted(model_root.rglob("*.json")) if model_root.exists() else ():
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        entry = payload.get("results", {}).get("ifeval") if isinstance(payload, dict) else None
        if not isinstance(entry, dict):
            continue
        values = [
            float(value) for key, value in entry.items()
            if key.split(",", 1)[0] == "prompt_level_strict_acc"
            and isinstance(value, (int, float)) and math.isfinite(value)
        ]
        if len(values) == 1:
            found.append((values[0], str(path)))
    return found[0] if len(found) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    provenance = json.loads((args.work / "provenance.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []
    for model in provenance["models"]:
        method = model["method"]
        measured = score(args.work / "raw" / method)
        if measured is None:
            missing.append(method)
            continue
        value, source = measured
        rows.append({
            "method": method, "seed": model.get("seed"),
            "ifeval_prompt_strict_raw": f"{value:.12g}",
            "ifeval_prompt_strict_percent": f"{100.0 * value:.8f}",
            "source_json": source,
        })
    base = next((float(row["ifeval_prompt_strict_percent"]) for row in rows if row["method"] == "base"), None)
    for row in rows:
        value = float(row["ifeval_prompt_strict_percent"])
        row["delta_vs_base_pp"] = "" if base is None else f"{value - base:.8f}"
    fields = ["method", "seed", "ifeval_prompt_strict_raw", "ifeval_prompt_strict_percent", "delta_vs_base_pp", "source_json"]
    with (args.output_dir / "ifeval_seed42.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "ifeval_seed42.json").write_text(json.dumps({
        "metric": "prompt_level_strict_acc", "num_fewshot": 0,
        "rows": rows, "missing": missing,
        "complete": not missing and not provenance.get("missing"),
        "p1_sealed_test_opened": False,
    }, indent=2) + "\n")
    latex = [
        row["method"].replace("_", "\\_") + " & " + f"{float(row['ifeval_prompt_strict_percent']):.2f}" + r" \\"
        for row in rows
    ]
    (args.output_dir / "ifeval_seed42_latex_rows.tex").write_text("\n".join(latex) + "\n")
    print(json.dumps({"measured": len(rows), "missing": missing}, indent=2))


if __name__ == "__main__":
    main()
