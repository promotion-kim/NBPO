#!/usr/bin/env python3
"""Generate measured P2 CSV/JSON/LaTeX tables from lm-eval result JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def load_result_files(model_root: Path) -> list[dict]:
    payloads = []
    for path in sorted(model_root.rglob("*.json")):
        if path.name.endswith(".completed.json"):
            continue
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("results"), dict):
            payload["_source_json"] = str(path)
            payloads.append(payload)
    return payloads


def select_entry(payloads: list[dict], task: str) -> tuple[dict, str] | None:
    found = []
    for payload in payloads:
        # lm-eval 0.4.12 emits aggregate tasks such as MMLU, MMLU-Pro and
        # Minerva-Math identically in both `results` and `groups`.  Count the
        # physical result JSON once, preferring `results`; fall back to
        # `groups` only when the former has no aggregate entry.
        entry = payload.get("results", {}).get(task)
        if not isinstance(entry, dict):
            entry = payload.get("groups", {}).get(task)
        if isinstance(entry, dict):
            found.append((entry, payload["_source_json"]))
    if len(found) != 1:
        return None
    return found[0]


def select_metric(entry: dict, metric: str, filter_name: str | None) -> tuple[str, float] | None:
    candidates = []
    for key, value in entry.items():
        base, _, suffix = key.partition(",")
        if base != metric or not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        if filter_name is not None and suffix != filter_name:
            continue
        if key.endswith("_stderr") or "stderr" in key:
            continue
        candidates.append((key, float(value)))
    if len(candidates) != 1:
        return None
    return candidates[0]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def tex(value: float | None) -> str:
    return "--" if value is None else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--blockers", default=None)
    args = parser.parse_args()
    root = Path(args.root)
    protocol = json.loads(Path(args.protocol).read_text())
    blocked_benchmarks = {}
    if args.blockers:
        blocker_payload = json.loads(Path(args.blockers).read_text())
        blocker_rows = blocker_payload if isinstance(blocker_payload, list) else [blocker_payload]
        blocked_benchmarks = {
            str(row["benchmark"]): row for row in blocker_rows
            if isinstance(row, dict) and row.get("status") == "BLOCKED" and row.get("benchmark")
        }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (root / "models.tsv").open(encoding="utf-8") as handle:
        models = list(csv.DictReader(handle, delimiter="\t"))
    long_rows = []
    values: dict[str, dict[str, float]] = {}
    missing = []
    for model in models:
        name = model["name"]
        payloads = load_result_files(root / "raw" / name)
        values[name] = {}
        for bench in protocol["benchmarks"]:
            selected = select_entry(payloads, bench["task"])
            if selected is None:
                if bench["paper_name"] in blocked_benchmarks:
                    blocker = blocked_benchmarks[bench["paper_name"]]
                    long_rows.append({
                        "model": name, "method": model["method"], "seed": model["seed"],
                        "benchmark": bench["paper_name"], "task": bench["task"],
                        "num_fewshot": bench["num_fewshot"], "metric_key": "",
                        "score_raw": "", "score_percent": "", "status": "BLOCKED",
                        "source_json": "", "blocker_evidence": blocker.get("evidence_log", ""),
                    })
                    continue
                missing.append({"model": name, "benchmark": bench["paper_name"], "reason": "task_entry_count_not_one"})
                continue
            entry, source = selected
            metric = select_metric(entry, bench["primary_metric"], bench.get("filter"))
            if metric is None:
                missing.append({"model": name, "benchmark": bench["paper_name"], "reason": "primary_metric_count_not_one"})
                continue
            metric_key, score = metric
            score_pct = 100.0 * score
            values[name][bench["paper_name"]] = score_pct
            long_rows.append({
                "model": name, "method": model["method"], "seed": model["seed"],
                "benchmark": bench["paper_name"], "task": bench["task"],
                "num_fewshot": bench["num_fewshot"], "metric_key": metric_key,
                "score_raw": f"{score:.12g}", "score_percent": f"{score_pct:.8f}",
                "source_json": source,
                "status": "MEASURED", "blocker_evidence": "",
            })

    bench_names = [item["paper_name"] for item in protocol["benchmarks"]]
    base = values.get("base", {})
    table_rows = []
    group_rows = []
    for model in models:
        name = model["name"]
        row = {"model": name, "method": model["method"], "seed": model["seed"]}
        for bench in bench_names:
            row[bench] = (
                "BLOCKED" if bench in blocked_benchmarks
                else "" if bench not in values[name] else f"{values[name][bench]:.8f}"
            )
            row[f"delta_vs_base_{bench}"] = (
                "" if bench not in values[name] or bench not in base
                else f"{values[name][bench] - base[bench]:.8f}"
            )
        table_rows.append(row)
        grow = {"model": name, "method": model["method"], "seed": model["seed"]}
        for group, members in protocol["group_averages"].items():
            available_members = [member for member in members if member not in blocked_benchmarks]
            group_values = [values[name].get(member) for member in available_members]
            grow[group] = "" if any(value is None for value in group_values) else f"{sum(group_values) / len(group_values):.8f}"
            grow[f"{group}_n"] = len(available_members)
            grow[f"{group}_blocked"] = ",".join(member for member in members if member in blocked_benchmarks)
        group_rows.append(grow)

    long_fields = ["model", "method", "seed", "benchmark", "task", "num_fewshot", "metric_key", "score_raw", "score_percent", "status", "source_json", "blocker_evidence"]
    write_csv(out / "metrics_long.csv", long_rows, long_fields)
    table_fields = ["model", "method", "seed"] + [field for bench in bench_names for field in (bench, f"delta_vs_base_{bench}")]
    write_csv(out / "benchmark_table.csv", table_rows, table_fields)
    group_names = list(protocol["group_averages"])
    group_fields = ["model", "method", "seed"] + [field for group in group_names for field in (group, f"{group}_n", f"{group}_blocked")]
    write_csv(out / "group_table.csv", group_rows, group_fields)

    latex = []
    for row in table_rows:
        label = row["method"].replace("_", "\\_")
        latex.append(label + " & " + " & ".join("BLOCKED" if row[b] == "BLOCKED" else tex(float(row[b])) if row[b] else "--" for b in bench_names) + r" \\")
    (out / "benchmark_latex_rows.tex").write_text("\n".join(latex) + "\n")
    group_latex = []
    for row in group_rows:
        label = row["method"].replace("_", "\\_")
        group_latex.append(label + " & " + " & ".join(tex(float(row[g])) if row[g] else "--" for g in group_names) + r" \\")
    (out / "group_latex_rows.tex").write_text("\n".join(group_latex) + "\n")

    summary = {
        "models": [row["name"] for row in models],
        "benchmarks": bench_names,
        "measured_cells": sum(row["status"] == "MEASURED" for row in long_rows),
        "blocked_cells": sum(row["status"] == "BLOCKED" for row in long_rows),
        "expected_cells": len(models) * len(bench_names),
        "complete_except_declared_blockers": not missing and len(long_rows) == len(models) * len(bench_names),
        "blocked_benchmarks": blocked_benchmarks,
        "missing": missing,
        "no_regression_definition": "Reported as signed percentage-point difference from base; no post-hoc equivalence margin is imposed.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if missing:
        raise SystemExit(f"P2 aggregation incomplete: {len(missing)} missing measured cells")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
