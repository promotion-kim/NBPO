#!/usr/bin/env python3
"""Aggregate the preregistered Beaver two-objective Table-4 evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


DISPLAY = {
    "base": "Base",
    "ronpo_os_confirmatory": "RONPO (OS, confirmatory)",
    "inpo_avg": "INPO (avg)",
    "sppo_avg": "SPPO (avg)",
    "simpo": "SimPO",
    "ipo": "IPO",
    "dpo": "DPO",
    "ht_mnpo_harmless": "HT-MNPO (harmless)",
    "ht_mnpo_helpfulness": "HT-MNPO (help.)",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ci(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def macro_name(model: str, field: str) -> str:
    return "SL" + "".join(part.title() for part in model.replace("-", "_").split("_")) + "".join(part.title() for part in field.split("_"))


def latex_cell(macro: str, model: str, ranked: list[str]) -> str:
    if ranked and model == ranked[0]:
        return f"\\textbf{{\\{macro}}}"
    if len(ranked) > 1 and model == ranked[1]:
        return f"\\underline{{\\{macro}}}"
    return f"\\{macro}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--helpfulness", type=Path, required=True)
    parser.add_argument("--harmlessness", type=Path, required=True)
    parser.add_argument("--pool-audit", type=Path, required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--train-pair-count", type=int, required=True)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--effective-batch", type=int, default=16)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit, calibration, manifest = map(load, [args.pool_audit, args.calibration, args.data_manifest])
    models = list(audit["eligible_models"])
    if not models or models[0] != "base":
        raise RuntimeError("base must be the first eligible model")
    h_rows, s_rows = read_jsonl(args.helpfulness), read_jsonl(args.harmlessness)
    if len(h_rows) != len(s_rows) or not h_rows:
        raise RuntimeError("objective score row count mismatch")
    h_map = {str(row["prompt_id"]): row for row in h_rows}
    s_map = {str(row["prompt_id"]): row for row in s_rows}
    prompt_ids = sorted(h_map)
    if set(prompt_ids) != set(s_map) or len(prompt_ids) != len(h_rows):
        raise RuntimeError("objective prompt ids mismatch or duplicate")
    raw = np.empty((len(prompt_ids), 2, len(models)), dtype=np.float64)
    metadata: dict[str, dict] = {}
    for pidx, prompt_id in enumerate(prompt_ids):
        for oidx, row in enumerate((h_map[prompt_id], s_map[prompt_id])):
            if list(row["response_model_names"]) != models:
                raise RuntimeError(f"model order mismatch at {prompt_id}")
            values = np.asarray(row["all_rm_scores"], dtype=np.float64)
            if values.shape != (len(models),) or not np.isfinite(values).all():
                raise RuntimeError(f"invalid scores at {prompt_id}")
            raw[pidx, oidx] = values
            metadata[prompt_id] = row
    lo, hi = raw.min(axis=2, keepdims=True), raw.max(axis=2, keepdims=True)
    norm = np.where(hi == lo, 0.5, (raw - lo) / (hi - lo))
    prompt_avg, prompt_worst = norm.mean(axis=1), norm.min(axis=1)
    base_idx = models.index("base")
    wins = (raw > raw[:, :, base_idx, None]).astype(float)
    wins += 0.5 * (raw == raw[:, :, base_idx, None])
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(prompt_ids), size=(args.bootstrap_resamples, len(prompt_ids)))
    rows, per_prompt = [], []
    for midx, model in enumerate(models):
        primary_values = prompt_worst[:, midx]
        objective_win = wins[:, :, midx].mean(axis=0)
        row = {
            "model": model,
            "display_name": DISPLAY.get(model, model),
            "status": "eligible",
            "records": len(prompt_ids),
            "helpfulness_norm": float(norm[:, 0, midx].mean()),
            "harmlessness_norm": float(norm[:, 1, midx].mean()),
            "helpfulness_raw": float(raw[:, 0, midx].mean()),
            "harmlessness_raw": float(raw[:, 1, midx].mean()),
            "mean_objective_norm_score": float(prompt_avg[:, midx].mean()),
            "mean_prompt_worst_norm_score": float(primary_values.mean()),
            "mean_prompt_worst_norm_score_ci95": ci(primary_values[indices].mean(axis=1)),
            "mean_win_rate_vs_baseline": float(objective_win.mean()),
            "min_win_rate_vs_baseline": float(objective_win.min()),
            "helpfulness_win_vs_base": float(objective_win[0]),
            "harmlessness_win_vs_base": float(objective_win[1]),
        }
        rows.append(row)
        for pidx, prompt_id in enumerate(prompt_ids):
            per_prompt.append({"prompt_id": prompt_id, "model": model,
                               "helpfulness_raw": float(raw[pidx, 0, midx]), "harmlessness_raw": float(raw[pidx, 1, midx]),
                               "helpfulness_norm": float(norm[pidx, 0, midx]), "harmlessness_norm": float(norm[pidx, 1, midx]),
                               "prompt_avg_norm": float(prompt_avg[pidx, midx]), "prompt_worst_norm": float(prompt_worst[pidx, midx])})
    nonbase = [row for row in rows if row["model"] != "base"]
    ranked = sorted(nonbase, key=lambda row: (-row["mean_prompt_worst_norm_score"], row["model"]))
    presentation = ranked + [next(row for row in rows if row["model"] == "base")]
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    presentation[-1]["rank"] = len(presentation)
    trained_nonronpo = [row for row in rows if row["model"] != "base" and not row["model"].startswith("ronpo_")]
    best_worst_baseline = max(trained_nonronpo, key=lambda row: (row["mean_prompt_worst_norm_score"], row["model"])) if trained_nonronpo else None
    best_avg_baseline = max(trained_nonronpo, key=lambda row: (row["mean_objective_norm_score"], row["model"])) if trained_nonronpo else None
    os_row = next((row for row in rows if row["model"] == "ronpo_os_confirmatory"), None)
    gates = {"status": "failed_missing"}
    if os_row and best_worst_baseline and best_avg_baseline:
        oi, wi, ai = models.index(os_row["model"]), models.index(best_worst_baseline["model"]), models.index(best_avg_baseline["model"])
        worst_dist = (prompt_worst[:, oi] - prompt_worst[:, wi])[indices].mean(axis=1)
        avg_dist = (prompt_avg[:, oi] - prompt_avg[:, ai])[indices].mean(axis=1)
        worst_ci, avg_ci = ci(worst_dist), ci(avg_dist)
        gates = {
            "status": "pass" if worst_ci[0] > 0 and avg_ci[0] > -0.02 else "fail",
            "worst_gate_pass": bool(worst_ci[0] > 0),
            "average_floor_pass": bool(avg_ci[0] > -0.02),
            "worst_comparator": best_worst_baseline["model"],
            "average_comparator": best_avg_baseline["model"],
            "worst_paired_difference": float((prompt_worst[:, oi] - prompt_worst[:, wi]).mean()),
            "worst_paired_difference_ci95": worst_ci,
            "avg_paired_difference": float((prompt_avg[:, oi] - prompt_avg[:, ai]).mean()),
            "avg_paired_difference_ci95": avg_ci,
            "bootstrap_resamples": args.bootstrap_resamples, "bootstrap_seed": args.seed,
        }
    failures = []
    for model in audit.get("failed_models", []):
        path = args.gate_root / f"{model}.json"
        failures.append({"model": model, "status": "stability_failed", "gate": str(path)})
    for model in audit.get("training_failed_models", []):
        failures.append({"model": model, "status": "training_failed"})
    macros = []
    numeric_fields = ["helpfulness_norm", "harmlessness_norm", "mean_objective_norm_score", "mean_prompt_worst_norm_score", "mean_win_rate_vs_baseline", "min_win_rate_vs_baseline"]
    for row in presentation:
        for field in numeric_fields:
            macros.append(f"\\newcommand{{\\{macro_name(row['model'], field)}}}{{{row[field]:.3f}}}")
    macro_path = args.output_dir / "saferlhf_table4_macros.tex"
    macro_path.write_text("\n".join(macros) + "\n", encoding="utf-8")
    ranks = {field: [row["model"] for row in sorted(rows, key=lambda item: (-item[field], item["model"]))[:2]] for field in numeric_fields[:4]}
    wr_ranks = [row["model"] for row in sorted(nonbase, key=lambda item: (-item["mean_win_rate_vs_baseline"], item["model"]))[:2]]
    wwr_ranks = [row["model"] for row in sorted(nonbase, key=lambda item: (-item["min_win_rate_vs_baseline"], item["model"]))[:2]]
    tex_rows = []
    for row in presentation:
        if row["model"] == "base":
            wr, wwr = "--", "--"
        else:
            wr = latex_cell(macro_name(row["model"], "mean_win_rate_vs_baseline"), row["model"], wr_ranks)
            wwr = latex_cell(macro_name(row["model"], "min_win_rate_vs_baseline"), row["model"], wwr_ranks)
        body = (
            f"{row['display_name']} & {latex_cell(macro_name(row['model'], 'helpfulness_norm'), row['model'], ranks['helpfulness_norm'])} & "
            f"{latex_cell(macro_name(row['model'], 'harmlessness_norm'), row['model'], ranks['harmlessness_norm'])} & "
            f"{latex_cell(macro_name(row['model'], 'mean_objective_norm_score'), row['model'], ranks['mean_objective_norm_score'])} & "
            f"{latex_cell(macro_name(row['model'], 'mean_prompt_worst_norm_score'), row['model'], ranks['mean_prompt_worst_norm_score'])} & {wr} & {wwr} "
        )
        tex_rows.append(body + "\\\\")
    head = calibration["reward_helpfulness_vs_human_better"]
    cost = calibration["cost_harmlessness_vs_human_safer"]
    epochs = args.steps * args.effective_batch / args.train_pair_count
    caption = (
        "Llama-3.1-8B-Instruct on PKU-SafeRLHF dual-preference conflict prompts. "
        f"Helpfulness is Beaver reward ({head['accuracy']:.2%} human-better agreement on {head['rows']} conflict rows); "
        f"harmlessness is negative Beaver cost ({cost['accuracy']:.2%} human-safer agreement). "
        f"The shared training pool has {args.train_pair_count} pairs ({epochs:.2f} effective epochs at {args.steps} steps and batch {args.effective_batch}); "
        f"validation has {len(prompt_ids)} prompt-disjoint conflict prompts. Values are per-prompt min--max normalized over eligible models. "
        f"RONPO OS worst-objective CI is [{os_row['mean_prompt_worst_norm_score_ci95'][0]:.3f}, {os_row['mean_prompt_worst_norm_score_ci95'][1]:.3f}]. "
        "The small validation panel is an explicit power limitation."
    ) if os_row else "No eligible confirmatory RONPO OS arm."
    table = [
        "% Generated only by build_table4_saferlhf.py from JSON/CSV artifacts.",
        "\\begin{table}[t]", "\\centering", "\\scriptsize", "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrrrr}", "\\toprule",
        " & \\multicolumn{2}{c}{Per-objective norm.} & \\multicolumn{4}{c}{Aggregate} " + "\\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-7}",
        "Method & Help. & Harmless & Avg & Worst & WR$_{\\mathrm{B}}$ & wWR$_{\\mathrm{B}}$ " + "\\\\",
        "\\midrule", *tex_rows, "\\bottomrule", "\\end{tabular}%", "}",
        f"\\caption{{{caption}}}", "\\label{tab:qwen3-robust-validation}", "\\end{table}", "",
    ]
    (args.output_dir / "table4_saferlhf.tex").write_text("\n".join(table), encoding="utf-8")
    with (args.output_dir / "per_objective_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(presentation[0]))
        writer.writeheader(); writer.writerows(presentation)
    with (args.output_dir / "per_prompt_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_prompt[0]))
        writer.writeheader(); writer.writerows(per_prompt)
    summary = {
        "status": "complete", "primary": "mean_prompt_worst_norm_score", "normalization": "per-prompt minmax across eligible model pool; constant objective prompt=0.5",
        "records": len(prompt_ids), "eligible_models": models, "ranking": presentation, "gate": gates,
        "failures": failures, "bootstrap": {"resamples": args.bootstrap_resamples, "seed": args.seed, "unit": "prompt", "paired": True},
        "input_sha256": {"helpfulness": sha(args.helpfulness), "harmlessness": sha(args.harmlessness), "pool_audit": sha(args.pool_audit)},
        "spent_sealed_split_touched": False,
    }
    (args.output_dir / "model_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md = ["# SafeRLHF Table 4", "", "| Method | Help. | Harmless | Avg | Worst (95% CI) | WR_B | wWR_B |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in presentation:
        wr = "--" if row["model"] == "base" else f"{row['mean_win_rate_vs_baseline']:.3f}"
        wwr = "--" if row["model"] == "base" else f"{row['min_win_rate_vs_baseline']:.3f}"
        md.append(f"| {row['display_name']} | {row['helpfulness_norm']:.3f} | {row['harmlessness_norm']:.3f} | {row['mean_objective_norm_score']:.3f} | {row['mean_prompt_worst_norm_score']:.3f} [{row['mean_prompt_worst_norm_score_ci95'][0]:.3f}, {row['mean_prompt_worst_norm_score_ci95'][1]:.3f}] | {wr} | {wwr} |")
    for fail in failures:
        md.append(f"| {DISPLAY.get(fail['model'], fail['model'])} | FAILED | FAILED | FAILED | FAILED | FAILED | FAILED |")
    md += ["", "## Preregistered gate", "", "```json", json.dumps(gates, indent=2), "```", ""]
    (args.output_dir / "TABLE4.md").write_text("\n".join(md), encoding="utf-8")
    (args.output_dir / "GATE.md").write_text("# Preregistered gate\n\n```json\n" + json.dumps(gates, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps({"summary": str(args.output_dir / "model_summary.json"), "gate": gates, "ranking": [(row["model"], row["mean_prompt_worst_norm_score"]) for row in presentation]}, indent=2))


if __name__ == "__main__":
    main()
