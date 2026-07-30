#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


METRICS = (
    "mean_prompt_level_strict",
    "mean_inst_level_strict",
    "mean_prompt_level_loose",
    "mean_inst_level_loose",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    seen = set()
    for root in args.roots:
        for report_path in sorted(root.glob("**/reports/*/ifeval.json")):
            report = json.loads(report_path.read_text(encoding="utf-8"))
            model = report["model_name"]
            key = (str(root.resolve()), model)
            if key in seen:
                continue
            seen.add(key)
            metric_map = {item["name"]: item["score"] for item in report["metrics"]}
            row = {
                "suite": root.name,
                "model": model,
                "num": report["num"],
                **{metric: metric_map[metric] for metric in METRICS},
                "mean_output_tokens": report.get("perf_metrics", {}).get("summary", {}).get("usage", {}).get(
                    "output_tokens", {}).get("mean"),
                "report": str(report_path),
            }
            rows.append(row)

    rows.sort(key=lambda row: (-row[METRICS[0]], row["suite"], row["model"]))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["suite", "model"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
