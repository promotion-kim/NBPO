#!/usr/bin/env python3
"""Build a Table-4-style seed-42/43 mean and sample-SD report.

Each training seed is first evaluated and min-max normalized independently on
the same 1,000 prompts and the same canonical method pool.  This script then
computes the arithmetic mean and sample standard deviation across the two
seed-level aggregates.  It never pools prompts across seeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


DISPLAY = {
    "ronpo_os_stage4": "RONPO (OS)",
    "ronpo_topmass_stage4": "RONPO (top-mass; ablation)",
    "ht_mnpo_helpfulness_stage4": "HT-MNPO (help.)",
    "ht_mnpo_harmless_stage4": "HT-MNPO (harml.)",
    "inpo_avg_stage4": "INPO-avg",
    "sppo_avg_stage4": "SPPO-avg",
    "simpo_stage4": "SimPO",
    "ipo_stage4": "IPO",
    "dpo_stage4": "DPO",
    "base": "Base",
}
MAIN_MODELS = [
    "ronpo_os_stage4",
    "ht_mnpo_helpfulness_stage4",
    "ht_mnpo_harmless_stage4",
    "inpo_avg_stage4",
    "sppo_avg_stage4",
    "simpo_stage4",
    "ipo_stage4",
    "dpo_stage4",
    "base",
]
ALL_MODELS = ["ronpo_os_stage4", "ronpo_topmass_stage4", *MAIN_MODELS[1:]]
METRICS = {
    "helpfulness_norm": "Helpful.",
    "harmlessness_norm": "Harmless.",
    "mean_objective_norm_score": "Avg",
    "mean_prompt_worst_norm_score": "Worst",
    "mean_win_rate_vs_base": "WR_B",
    "min_win_rate_vs_base": "wWR_B",
}
TABLE_METRICS = list(METRICS)[:4]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(f"incomplete input: {path}")
    return payload


def finite_number(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite {label}")
    return result


def decorate(model: str, metric: str, text: str, stats: dict[str, dict[str, dict[str, float]]], models: list[str], latex: bool) -> str:
    trained = [candidate for candidate in models if candidate != "base"]
    ordered = sorted(trained, key=lambda candidate: (-stats[candidate][metric]["mean"], candidate))
    if model == ordered[0]:
        return f"\\textbf{{{text}}}" if latex else f"**{text}**"
    if len(ordered) > 1 and model == ordered[1]:
        return f"\\underline{{{text}}}" if latex else f"<u>{text}</u>"
    return text


def fmt(value: float, sd: float, latex: bool = False) -> str:
    # Four decimals keep the small but nonzero across-seed variation visible;
    # three-decimal Table-4 rounding would print SimPO's SD as 0.000.
    return f"{value:.4f} $\\pm$ {sd:.4f}" if latex else f"{value:.4f} ± {sd:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed42", type=Path, required=True)
    parser.add_argument("--seed43", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = {42: load(args.seed42), 43: load(args.seed43)}
    reference = inputs[42]
    for seed, payload in inputs.items():
        for field in ("primary", "normalization", "objectives", "records", "eligible_models"):
            if payload.get(field) != reference.get(field):
                raise RuntimeError(f"protocol incompatibility in {field}: seed42={reference.get(field)!r}, seed{seed}={payload.get(field)!r}")
        if payload.get("records") != 1000 or payload.get("bootstrap") != reference.get("bootstrap"):
            raise RuntimeError(f"records/bootstrap mismatch for seed {seed}")
        if payload.get("spent_sealed_split_touched") is not False:
            raise RuntimeError(f"sealed-split audit failed for seed {seed}")

    by_seed = {
        seed: {row["model"]: row for row in payload["ranking"]}
        for seed, payload in inputs.items()
    }
    if any(set(rows) != set(ALL_MODELS) for rows in by_seed.values()):
        raise RuntimeError(f"canonical model set mismatch: {[sorted(rows) for rows in by_seed.values()]}")

    stats: dict[str, dict[str, dict[str, float]]] = {}
    long_rows = []
    for model in ALL_MODELS:
        stats[model] = {}
        for metric in METRICS:
            values = [finite_number(by_seed[seed][model][metric], f"{seed}/{model}/{metric}") for seed in (42, 43)]
            stats[model][metric] = {
                "seed42": values[0], "seed43": values[1],
                "mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
            }
            long_rows.append({
                "model": model, "display_name": DISPLAY[model], "metric": metric,
                **stats[model][metric], "n_seeds": 2,
            })

    with (args.output_dir / "seed42_seed43_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(long_rows[0])); writer.writeheader(); writer.writerows(long_rows)

    ranked_main = sorted(MAIN_MODELS[:-1], key=lambda model: (-stats[model]["mean_prompt_worst_norm_score"]["mean"], model)) + ["base"]
    ranked_all = sorted(ALL_MODELS[:-1], key=lambda model: (-stats[model]["mean_prompt_worst_norm_score"]["mean"], model)) + ["base"]

    lines = [
        "# Stage-4 SafeRLHF, training seeds 42 and 43",
        "",
        "Values are the mean ± sample standard deviation across the two training seeds (n=2). "
        "Each seed was evaluated on the same 1,000 prompts with decode seed 42 and normalized independently over the same method pool.",
        "",
        "| Method | Helpful. | Harmless. | Avg | Worst |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in ranked_main:
        cells = []
        for metric in TABLE_METRICS:
            item = stats[model][metric]
            cells.append(decorate(model, metric, fmt(item["mean"], item["sample_sd"]), stats, ranked_main, False))
        lines.append(f"| {DISPLAY[model]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Sample SD with two seeds is descriptive only; it is not a confidence interval or a stable variance estimate.",
        "RONPO top-mass is retained in the full audit table as an estimator ablation and omitted from the main Table-4-style method table.",
        "",
    ]
    (args.output_dir / "TABLE4_SEED42_43.md").write_text("\n".join(lines), encoding="utf-8")

    full = [
        "# Full two-seed metrics, including the top-mass estimator ablation",
        "",
        "| Method | Helpful. | Harmless. | Avg | Worst | WR_B | wWR_B |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ranked_all:
        cells = [fmt(stats[model][metric]["mean"], stats[model][metric]["sample_sd"]) for metric in METRICS]
        full.append(f"| {DISPLAY[model]} | " + " | ".join(cells) + " |")
    full.append("")
    (args.output_dir / "FULL_TWO_SEED_METRICS.md").write_text("\n".join(full), encoding="utf-8")

    tex = [
        "% Generated from seed-level JSON; values are mean and sample SD over training seeds 42 and 43.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4.5pt}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method & Helpful. & Harmless. & Avg & Worst \\\\",
        "\\midrule",
    ]
    for model in ranked_main:
        cells = []
        for metric in TABLE_METRICS:
            item = stats[model][metric]
            cells.append(decorate(model, metric, fmt(item["mean"], item["sample_sd"], True), stats, ranked_main, True))
        tex.append(f"{DISPLAY[model]} & " + " & ".join(cells) + " \\\\")
        if model == ranked_main[-2]:
            tex.append("\\midrule")
    tex += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{\\textbf{Model-scale robustness on PKU-SafeRLHF across training seeds 42 and 43.} Values are mean $\\pm$ sample standard deviation over two training seeds. Each seed uses the same held-out 1{,}000-prompt panel, Beaver reward and negated Beaver cost evaluators, fixed decode seed 42, and per-prompt min--max normalization over the same method pool. Best in bold, second best underlined.}",
        "\\label{tab:saferlhf-robust-two-seed}",
        "\\end{table}",
        "",
    ]
    (args.output_dir / "table4_seed42_seed43.tex").write_text("\n".join(tex), encoding="utf-8")

    provenance = {
        "status": "complete",
        "summary": "mean and sample standard deviation over training seeds 42 and 43",
        "n_seeds": 2,
        "training_seeds": [42, 43],
        "decode_seed": 42,
        "records_per_seed": 1000,
        "objectives": reference["objectives"],
        "normalization": reference["normalization"],
        "bootstrap_in_seed_files": reference["bootstrap"],
        "seed_input_sha256": {"42": sha256(args.seed42), "43": sha256(args.seed43)},
        "prompt_manifest": str(args.prompt_manifest),
        "prompt_manifest_sha256": sha256(args.prompt_manifest),
        "main_table_models": ranked_main,
        "full_audit_models": ranked_all,
        "topmass_handling": "reported in full audit as estimator ablation; omitted from Table-4-style main table",
        "caution": "sample SD at n=2 is descriptive and is not a confidence interval",
        "spent_sealed_split_touched": False,
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir), "ranking": ranked_main}, indent=2))


if __name__ == "__main__":
    main()
