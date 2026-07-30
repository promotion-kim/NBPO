#!/usr/bin/env python3
"""Build the Qwen Table-3 block as mean and sample SD over three seeds."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


ORDER = ["ronpo_os", "ht_mnpo_helpfulness", "ht_mnpo_harmless", "inpo_avg", "sppo_avg", "simpo", "ipo", "dpo", "base"]
DISPLAY = {"ronpo_os": "RONPO (OS)", "ht_mnpo_helpfulness": "HT-MNPO (help.)", "ht_mnpo_harmless": "HT-MNPO (harml.)", "inpo_avg": "INPO-avg", "sppo_avg": "SPPO-avg", "simpo": "SimPO", "ipo": "IPO", "dpo": "DPO", "base": "Base"}
METRICS = ["helpfulness_norm", "harmlessness_norm", "mean_objective_norm_score", "mean_prompt_worst_norm_score", "mean_win_rate_vs_base", "min_win_rate_vs_base"]
HEADERS = ["Help.", "Harmless", "Avg", "Worst", "WR_B", "wWR_B"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(f"incomplete evaluation: {path}")
    return payload


def fmt(mean: float, sd: float, latex: bool = False) -> str:
    return f"{mean:.4f}$\\pm${sd:.4f}" if latex else f"{mean:.4f} ± {sd:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    seeds = (42, 43, 44)
    paths = {seed: args.root / f"evaluation/s{seed}/results/ranked_validation_summary.json" for seed in seeds}
    data = {seed: load(path) for seed, path in paths.items()}
    reference = data[42]
    for seed, payload in data.items():
        for field in ("records", "primary", "normalization", "objectives", "bootstrap", "eligible_models"):
            if payload.get(field) != reference.get(field):
                raise RuntimeError(f"incompatible {field} for seed {seed}")
        if payload.get("records") != 1000 or payload.get("spent_sealed_split_touched") is not False:
            raise RuntimeError(f"protocol audit failed for seed {seed}")
    common_lock = json.loads((args.root / "evaluation/common_eligible_arms.json").read_text(encoding="utf-8"))
    common = common_lock["eligible_models"]
    if reference["eligible_models"] != common:
        raise RuntimeError("evaluation pool differs from the locked three-seed intersection")
    rows = {seed: {row["model"].removesuffix("_stage4"): row for row in payload["ranking"]} for seed, payload in data.items()}
    if any(set(seed_rows) != set(common) for seed_rows in rows.values()):
        raise RuntimeError("ranked model set is not identical across seeds")
    stats, csv_rows = {}, []
    for model in common:
        stats[model] = {}
        for metric in METRICS:
            values = [float(rows[seed][model][metric]) for seed in seeds]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"non-finite metric: {model}/{metric}")
            item = {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), **{f"seed{seed}": value for seed, value in zip(seeds, values)}}
            stats[model][metric] = item
            csv_rows.append({"model": model, "metric": metric, **item})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "three_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    trained = [model for model in ORDER if model in common and model != "base"]
    ranked = sorted(trained, key=lambda model: (-stats[model]["mean_prompt_worst_norm_score"]["mean"], model))
    rank = {model: index for index, model in enumerate(ranked)}
    def decorate(model: str, metric: str, value: str, latex: bool = False) -> str:
        ordered = sorted(trained, key=lambda item: (-stats[item][metric]["mean"], item))
        if model == ordered[0]: return f"\\textbf{{{value}}}" if latex else f"**{value}**"
        if len(ordered) > 1 and model == ordered[1]: return f"\\underline{{{value}}}" if latex else f"<u>{value}</u>"
        return value
    md = ["# Qwen2.5-7B SafeRLHF Stage-4, seeds 42/43/44", "", "Values are mean ± sample standard deviation over training seeds. Every seed uses the same 1,000 prompts and the same common eligible model pool.", "", "| Method | " + " | ".join(HEADERS) + " |", "|---|" + "---:|" * len(HEADERS)]
    tex_rows = []
    for model in ORDER:
        if model not in common:
            md.append(f"| {DISPLAY[model]} | " + " | ".join(["--"] * len(METRICS)) + " |")
            tex_rows.append(f"{DISPLAY[model]} & " + " & ".join(["--"] * len(METRICS)) + r" \\")
            continue
        values_md, values_tex = [], []
        for metric in METRICS:
            item = stats[model][metric]
            values_md.append(decorate(model, metric, fmt(item["mean"], item["sample_sd"]), False))
            values_tex.append(decorate(model, metric, fmt(item["mean"], item["sample_sd"], True), True))
        md.append(f"| {DISPLAY[model]} | " + " | ".join(values_md) + " |")
        tex_rows.append(f"{DISPLAY[model]} & " + " & ".join(values_tex) + r" \\")
    md += ["", f"Common eligible pool: {', '.join(common)}.", f"Excluded fail-closed: {', '.join(common_lock['excluded_arms']) or 'none'}.", "Sample SD is descriptive, not a confidence interval.", ""]
    (args.output_dir / "TABLE3_QWEN_SEED42_43_44.md").write_text("\n".join(md), encoding="utf-8")
    tex = ["% Auto-generated from three compatible seed-level JSON summaries.", r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lrrrrrr}", r"\toprule", "Method & Help. & Harmless & Avg & Worst & WR$_{\\mathrm{B}}$ & wWR$_{\\mathrm{B}}$ \\\\", r"\midrule", *tex_rows, r"\bottomrule", r"\end{tabular}%", "}", r"\caption{Qwen2.5-7B-Instruct SafeRLHF Stage-4 results. Values are mean $\pm$ sample standard deviation across training seeds 42, 43, and 44 on one common 1,000-prompt evaluation and normalization pool. Dashes denote a method excluded fail-closed because it lacked a valid Stage-4 stability-gate pass in at least one seed.}", r"\label{tab:qwen25-saferlhf-stage4-three-seed}", r"\end{table}", ""]
    (args.output_dir / "table3_qwen25_seed42_43_44.tex").write_text("\n".join(tex), encoding="utf-8")
    provenance = {"status": "complete", "training_seeds": list(seeds), "n_seeds": 3, "aggregation": "arithmetic mean and sample SD of seed-level aggregates; prompts are not pooled", "common_eligible_models": common, "excluded_arms": common_lock["excluded_arms"], "seed_summary_sha256": {str(seed): sha256(path) for seed, path in paths.items()}, "common_pool_lock_sha256": sha256(args.root / "evaluation/common_eligible_arms.json"), "spent_sealed_split_touched": False}
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "ranking": ranked, "excluded": common_lock["excluded_arms"]}, indent=2))


if __name__ == "__main__":
    main()
