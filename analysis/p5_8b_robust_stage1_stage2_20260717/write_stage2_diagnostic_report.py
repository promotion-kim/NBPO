#!/usr/bin/env python3
"""Render a diagnostic-only Stage-2 reward table from scored JSON outputs.

This script never reclassifies a stability result.  It simply places the
measured gate status beside raw per-head rewards and the protocol's normalized
average/worst aggregates, making length-drift effects inspectable without
turning failed models into paper-eligible results.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--pool-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, default=None)
    args = parser.parse_args()
    result = json.loads(args.result_json.read_text(encoding="utf-8"))
    audit = json.loads(args.pool_audit.read_text(encoding="utf-8"))
    status = audit["gate_status"]
    rows = []
    for row in result["ranking"]:
        gate = status[row["model"]]
        rows.append({
            **row,
            "stability_status": "passed" if gate["passed"] else "failed_diagnostic_only",
            "length_ratio": gate["length_ratio"],
        })
    payload = {
        "status": "complete",
        "scope": "diagnostic_only_includes_stability_failed_models; do not use for paper ranking, model selection, or a stability-gated claim",
        "primary_metric": result["primary"],
        "normalization": result["normalization"],
        "records": result["records"],
        "models": rows,
        "bootstrap": result["bootstrap"],
        "input_sha256": {
            "result_json": hashlib.sha256(args.result_json.read_bytes()).hexdigest(),
            "pool_audit": hashlib.sha256(args.pool_audit.read_bytes()).hexdigest(),
        },
        "spent_sealed_split_touched": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "diagnostic_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    columns = [
        "model", "display_name", "stability_status", "length_ratio", "helpfulness_raw", "harmlessness_raw",
        "mean_objective_norm_score", "mean_prompt_worst_norm_score", "mean_prompt_worst_norm_score_ci95",
        "mean_win_rate_vs_base", "min_win_rate_vs_base",
    ]
    with (args.output_dir / "diagnostic_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in columns} for row in rows])
    lines = [
        "# P5 Stage-2 diagnostic reward table including stability failures",
        "",
        "This diagnostic reuses existing 49-prompt generations. It includes RONPO rows that failed the frozen length-ratio gate only to inspect their scores. These rows are not eligible for paper ranking, selection, or any stability-gated claim.",
        "",
        "Raw columns are Beaver reward helpfulness and negative Beaver cost harmlessness. Avg and Worst are the protocol's per-prompt min-max normalized aggregates over this complete diagnostic pool.",
        "",
        "| Method | Stability | Length ratio | Help. raw | Harmless raw | Avg norm. | Worst norm. (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ci = row["mean_prompt_worst_norm_score_ci95"]
        ratio = "n/a" if row["length_ratio"] is None else f"{row['length_ratio']:.4f}"
        lines.append(
            f"| {row['display_name']} | {row['stability_status']} | {ratio} | "
            f"{row['helpfulness_raw']:.6f} | {row['harmlessness_raw']:.6f} | "
            f"{row['mean_objective_norm_score']:.4f} | {row['mean_prompt_worst_norm_score']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] |"
        )
    (args.output_dir / "DIAGNOSTIC_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.status_file is not None:
        with args.status_file.open("a", encoding="utf-8") as handle:
            failed = [row["model"] for row in rows if row["stability_status"] != "passed"]
            handle.write(
                f"{dt.datetime.now().astimezone().isoformat(timespec='seconds')} "
                "P5 diagnostic-only Stage-2 reward evaluation completed on existing 49-prompt generations; "
                f"stability-failed rows retained only for diagnosis: {','.join(failed)}. "
                "Not eligible for paper ranking or selection; no sealed split touched.\n"
            )
    print(json.dumps({"summary": str(args.output_dir / "diagnostic_summary.json"), "models": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
