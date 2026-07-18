#!/usr/bin/env python3
"""Regenerate all two-objective model metrics and the preregistered gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


OBJECTIVES = ("helpfulness", "harmlessness")
LOCKED_STRETCH = ("ht_mnpo_helpful", "sppo_avg", "simpo", "ipo", "ronpo_full_expect")
DISPLAY = {
    "base": "Base", "ronpo_os": "RONPO (OS)", "ronpo_topmass": "RONPO (top-mass)",
    "inpo_avg": "INPO (avg)", "ht_mnpo_harmless": "HT-MNPO (harmless)",
    "ht_mnpo_helpful": "HT-MNPO (help.)", "sppo_avg": "SPPO (avg)",
    "simpo": "SimPO", "ipo": "IPO", "ronpo_full_expect": "RONPO (full-exp)",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def bootstrap_mean(values: np.ndarray, indices: np.ndarray) -> list[float]:
    return percentile(values[indices].mean(axis=1))


def format_ci(value: float, ci: list[float]) -> str:
    return f"{value:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"


def latex_ranked(value: float, model: str, best: str, second: str) -> str:
    rendered = f"{value:.3f}"
    if model == best:
        return f"\\textbf{{{rendered}}}"
    if model == second:
        return f"\\underline{{{rendered}}}"
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--helpfulness", type=Path, required=True)
    parser.add_argument("--harmlessness", type=Path, required=True)
    parser.add_argument("--pool-audit", type=Path, required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "fresh"], required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--primary-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = json.loads(args.pool_audit.read_text(encoding="utf-8"))
    models = list(audit["eligible_models"])
    if models[0] != "base":
        raise RuntimeError("base must be first")
    objective_rows = {name: read_jsonl(path) for name, path in [
        ("helpfulness", args.helpfulness), ("harmlessness", args.harmlessness),
    ]}
    if any(len(rows) != args.expected_records for rows in objective_rows.values()):
        raise RuntimeError("score row count mismatch")
    by_objective = {}
    metadata = {}
    for objective, rows in objective_rows.items():
        mapping = {str(row["prompt_id"]): row for row in rows}
        if len(mapping) != len(rows):
            raise RuntimeError(f"duplicate prompt ids: {objective}")
        by_objective[objective] = mapping
        for prompt_id, row in mapping.items():
            if list(row["response_model_names"]) != models or len(row["all_rm_scores"]) != len(models):
                raise RuntimeError(f"model order or score length mismatch: {objective}/{prompt_id}")
            metadata[prompt_id] = row
    prompt_ids = sorted(by_objective["helpfulness"])
    if set(prompt_ids) != set(by_objective["harmlessness"]):
        raise RuntimeError("objective prompt sets differ")
    raw = np.empty((len(prompt_ids), len(OBJECTIVES), len(models)), dtype=np.float64)
    norm = np.empty_like(raw)
    for pidx, prompt_id in enumerate(prompt_ids):
        for oidx, objective in enumerate(OBJECTIVES):
            values = np.asarray(by_objective[objective][prompt_id]["all_rm_scores"], dtype=np.float64)
            if not np.isfinite(values).all():
                raise RuntimeError(f"non-finite scores: {objective}/{prompt_id}")
            raw[pidx, oidx] = values
            lo, hi = float(values.min()), float(values.max())
            norm[pidx, oidx] = 0.5 if hi == lo else (values - lo) / (hi - lo)
    prompt_avg = norm.mean(axis=1)
    prompt_worst = norm.min(axis=1)
    base_index = models.index("base")
    wins = (raw > raw[:, :, base_index, None]).astype(np.float64)
    wins += 0.5 * (raw == raw[:, :, base_index, None])
    bootstrap_indices = np.random.default_rng(args.seed).integers(
        0, len(prompt_ids), size=(args.bootstrap_resamples, len(prompt_ids))
    )
    summaries = []
    per_prompt_rows = []
    for midx, model in enumerate(models):
        objective_norm = norm[:, :, midx].mean(axis=0)
        objective_raw = raw[:, :, midx].mean(axis=0)
        objective_win = wins[:, :, midx].mean(axis=0)
        primary_values = prompt_worst[:, midx]
        primary = float(primary_values.mean())
        primary_ci = bootstrap_mean(primary_values, bootstrap_indices)
        should_answer = np.asarray([
            str(metadata[prompt_id].get("behavior_label", "")) == "should_answer" for prompt_id in prompt_ids
        ])
        should_refuse = np.asarray([
            str(metadata[prompt_id].get("behavior_label", "")) == "should_refuse" for prompt_id in prompt_ids
        ])
        record = {
            "model": model, "display_name": DISPLAY.get(model, model), "status": "eligible",
            "records": len(prompt_ids),
            "mean_prompt_worst_norm_score": primary,
            "mean_prompt_worst_norm_score_ci95": primary_ci,
            "mean_objective_norm_score": float(prompt_avg[:, midx].mean()),
            "mean_win_rate_vs_baseline": float(objective_win.mean()),
            "min_win_rate_vs_baseline": float(objective_win.min()),
            "helpfulness_norm": float(objective_norm[0]),
            "harmlessness_norm": float(objective_norm[1]),
            "helpfulness_raw": float(objective_raw[0]),
            "harmlessness_raw": float(objective_raw[1]),
            "helpfulness_win_vs_base": float(objective_win[0]),
            "harmlessness_win_vs_base": float(objective_win[1]),
            "should_answer_helpfulness_norm": float(norm[should_answer, 0, midx].mean()) if should_answer.any() else None,
            "should_refuse_harmlessness_norm": float(norm[should_refuse, 1, midx].mean()) if should_refuse.any() else None,
        }
        summaries.append(record)
        for pidx, prompt_id in enumerate(prompt_ids):
            per_prompt_rows.append({
                "prompt_id": prompt_id, "model": model,
                "source": metadata[prompt_id].get("source"), "slice": metadata[prompt_id].get("slice"),
                "behavior_label": metadata[prompt_id].get("behavior_label"),
                "helpfulness_raw": raw[pidx, 0, midx], "harmlessness_raw": raw[pidx, 1, midx],
                "helpfulness_norm": norm[pidx, 0, midx], "harmlessness_norm": norm[pidx, 1, midx],
                "prompt_avg_norm": prompt_avg[pidx, midx], "prompt_worst_norm": prompt_worst[pidx, midx],
            })
    summaries.sort(key=lambda row: (-row["mean_prompt_worst_norm_score"], row["model"]))
    for rank, record in enumerate(summaries, start=1):
        record["rank"] = rank
    trained_non_ronpo = [row for row in summaries if row["model"] != "base" and not row["model"].startswith("ronpo_")]
    best_baseline = trained_non_ronpo[0] if trained_non_ronpo else None
    gates = {}
    if best_baseline:
        bidx = models.index(best_baseline["model"])
        for model in ("ronpo_os", "ronpo_topmass"):
            if model not in models:
                gates[model] = {"status": "failed_missing_or_ineligible", "pass": False}
                continue
            midx = models.index(model)
            differences = prompt_worst[:, midx] - prompt_worst[:, bidx]
            distribution = differences[bootstrap_indices].mean(axis=1)
            ci = percentile(distribution)
            gates[model] = {
                "status": "pass" if ci[0] > 0 else "fail", "pass": bool(ci[0] > 0),
                "best_non_ronpo_trained_arm": best_baseline["model"],
                "mean_paired_difference": float(differences.mean()), "ci95": ci,
                "resamples": args.bootstrap_resamples, "seed": args.seed,
            }
    failures = []
    for model in audit.get("failed_models", []):
        gate_path = args.gate_root / f"{model}.json"
        failures.append({"model": model, "status": "stability_failed", "gate": str(gate_path)})
    not_run = [model for model in LOCKED_STRETCH if model not in models and model not in audit.get("failed_models", [])]
    reported_summaries = summaries
    if args.primary_only:
        keep = {
            "rank", "model", "display_name", "status", "records",
            "mean_prompt_worst_norm_score", "mean_prompt_worst_norm_score_ci95",
        }
        reported_summaries = [{key: value for key, value in row.items() if key in keep} for row in summaries]
    summary = {
        "status": "complete", "split": args.split, "records": len(prompt_ids),
        "normalization": "per-prompt minmax across eligible model pool; constant objective prompt=0.5",
        "primary": "mean_prompt_worst_norm_score", "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed, "eligible_models": models, "failures": failures,
        "ranking": reported_summaries, "preregistered_gates": gates, "not_run_arms": not_run,
        "reporting_scope": "primary_only" if args.primary_only else "primary_and_preregistered_secondaries",
        "input_sha256": {
            "helpfulness": sha(args.helpfulness), "harmlessness": sha(args.harmlessness),
            "pool_audit": sha(args.pool_audit),
        },
        "spent_sealed_split_touched": False,
    }
    summary_path = args.output_dir / "model_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.primary_only:
        lines = [
            f"# {args.split.title()} primary-only confirmation", "",
            "| Rank | Method | Mean prompt worst normalized score (95% CI) |",
            "|---:|---|---:|",
        ]
        for row in summaries:
            lines.append(
                f"| {row['rank']} | {row['display_name']} | "
                f"{format_ci(row['mean_prompt_worst_norm_score'], row['mean_prompt_worst_norm_score_ci95'])} |"
            )
        for failure in failures:
            lines.append(f"| -- | {DISPLAY.get(failure['model'], failure['model'])} | FAILED stability gate |")
        for model in not_run:
            lines.append(f"| -- | {DISPLAY.get(model, model)} | NOT RUN (stretch scope) |")
        lines += ["", "## Preregistered paired gates", ""]
        for model, gate in gates.items():
            lines.append(f"- `{model}`: **{gate['status'].upper()}**; {json.dumps(gate, sort_keys=True)}")
        (args.output_dir / "FRESH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps({"summary": str(summary_path), "primary_only": True, "ranking": [(r['model'], r['mean_prompt_worst_norm_score']) for r in summaries], "gates": gates}, indent=2))
        return
    with (args.output_dir / "per_prompt_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_prompt_rows[0]))
        writer.writeheader(); writer.writerows(per_prompt_rows)
    with (args.output_dir / "per_objective_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "rank", "model", "status", "records", "mean_prompt_worst_norm_score",
            "mean_objective_norm_score", "mean_win_rate_vs_baseline", "min_win_rate_vs_baseline",
            "helpfulness_norm", "harmlessness_norm", "helpfulness_raw", "harmlessness_raw",
            "helpfulness_win_vs_base", "harmlessness_win_vs_base",
            "should_answer_helpfulness_norm", "should_refuse_harmlessness_norm",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(summaries)
    lines = [
        f"# {args.split.title()} two-objective table", "",
        "| Rank | Method | Help. | Harmless | Avg | Worst (95% CI) | WR_B | wWR_B |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        wr_markdown = "--" if row["model"] == "base" else f"{row['mean_win_rate_vs_baseline']:.4f}"
        wwr_markdown = "--" if row["model"] == "base" else f"{row['min_win_rate_vs_baseline']:.4f}"
        lines.append(
            f"| {row['rank']} | {row['display_name']} | {row['helpfulness_norm']:.4f} | "
            f"{row['harmlessness_norm']:.4f} | {row['mean_objective_norm_score']:.4f} | "
            f"{format_ci(row['mean_prompt_worst_norm_score'], row['mean_prompt_worst_norm_score_ci95'])} | "
            f"{wr_markdown} | {wwr_markdown} |"
        )
    for failure in failures:
        lines.append(f"| -- | {DISPLAY.get(failure['model'], failure['model'])} | FAILED | FAILED | FAILED | FAILED | FAILED | FAILED |")
    for model in not_run:
        lines.append(f"| -- | {DISPLAY.get(model, model)} | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |")
    (args.output_dir / "TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    def top_two(field: str, include_base: bool = True) -> tuple[str, str]:
        candidates = [row for row in summaries if include_base or row["model"] != "base"]
        ordered = sorted(candidates, key=lambda row: (-row[field], row["model"]))
        return ordered[0]["model"], ordered[1]["model"] if len(ordered) > 1 else ""
    best_avg, second_avg = top_two("mean_objective_norm_score")
    best_worst, second_worst = top_two("mean_prompt_worst_norm_score")
    best_wr, second_wr = top_two("mean_win_rate_vs_baseline", include_base=False)
    best_wwr, second_wwr = top_two("min_win_rate_vs_baseline", include_base=False)
    tex = []
    for row in summaries:
        name = row["display_name"].replace("_", "\\_")
        avg_cell = latex_ranked(row["mean_objective_norm_score"], row["model"], best_avg, second_avg)
        worst_cell = latex_ranked(row["mean_prompt_worst_norm_score"], row["model"], best_worst, second_worst)
        if row["model"] == "base":
            wr_cell = wwr_cell = "--"
        else:
            wr_cell = latex_ranked(row["mean_win_rate_vs_baseline"], row["model"], best_wr, second_wr)
            wwr_cell = latex_ranked(row["min_win_rate_vs_baseline"], row["model"], best_wwr, second_wwr)
        tex.append(
            f"{name} & {row['helpfulness_norm']:.3f} & {row['harmlessness_norm']:.3f} & "
            f"{avg_cell} & {worst_cell} & {wr_cell} & {wwr_cell} \\\\"
        )
    for failure in failures:
        tex.append(f"{DISPLAY.get(failure['model'], failure['model'])} & \\multicolumn{{6}}{{c}}{{FAILED stability gate}} \\\\")
    for model in not_run:
        tex.append(f"{DISPLAY.get(model, model)} & \\multicolumn{{6}}{{c}}{{NOT RUN (stretch scope)}} \\\\")
    (args.output_dir / "table_two_objective.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    gate_lines = [f"# {args.split.title()} preregistered gate", ""]
    for model, gate in gates.items():
        gate_lines.append(f"- `{model}`: **{gate['status'].upper()}**; {json.dumps(gate, sort_keys=True)}")
    (args.output_dir / "GATE.md").write_text("\n".join(gate_lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "ranking": [(r['model'], r['mean_prompt_worst_norm_score']) for r in summaries], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
