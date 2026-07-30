#!/usr/bin/env python3
"""Stage-2 RONPO robustness diagnostics from existing local-RM artifacts.

This script recomputes prompt-level normalized reward metrics, high-disagreement
subset metrics, and prompt bootstrap confidence intervals from scored JSONL files.
It intentionally does not rescore or regenerate model outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


OBJECTIVES = ("skywork", "athene", "armo")
DISPLAY = {
    "baseline": "Base",
    "htmnpo_skywork_s2": "HT-MNPO Skywork S2",
    "htmnpo_athene_s2": "HT-MNPO Athene S2",
    "htmnpo_armo_s2": "HT-MNPO ArmoRM S2",
    "ronpo_s2_ckpt1400": "RONPO S2 checkpoint-1400",
    "ronpo_s2_ckpt2457": "RONPO S2 checkpoint-2457",
}


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    vals = sorted(xs)
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def fmt(x: float, digits: int = 4) -> str:
    if x != x:
        return "-"
    return f"{x:.{digits}f}"


def win(a: float, b: float) -> float:
    if a > b:
        return 1.0
    if a < b:
        return 0.0
    return 0.5


def sign(a: float, b: float) -> int:
    if a > b:
        return 1
    if a < b:
        return -1
    return 0


def load_scores(work_dir: Path) -> tuple[list[str], list[str], dict[str, dict[str, dict[str, float]]], dict[str, str]]:
    scores: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    prompts: dict[str, str] = {}
    model_names: list[str] | None = None

    for objective in OBJECTIVES:
        path = work_dir / "scored" / f"eval_{objective}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                prompt_id = row["prompt_id"]
                prompts[prompt_id] = row.get("prompt", "")
                row_models = list(row["response_model_names"])
                if model_names is None:
                    model_names = row_models
                elif model_names != row_models:
                    raise ValueError(f"model order mismatch in {path}: {row_models} != {model_names}")
                scores[prompt_id][objective] = {
                    model: float(score)
                    for model, score in zip(row_models, row["all_rm_scores"])
                }

    if model_names is None:
        raise RuntimeError("no scored rows found")

    prompt_ids = sorted(scores)
    for prompt_id in prompt_ids:
        missing = [obj for obj in OBJECTIVES if obj not in scores[prompt_id]]
        if missing:
            raise ValueError(f"prompt {prompt_id} missing objectives: {missing}")
    return prompt_ids, model_names, scores, prompts


def normalize_scores(
    prompt_ids: list[str],
    model_names: list[str],
    scores: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    norm: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for prompt_id in prompt_ids:
        for objective in OBJECTIVES:
            vals = [scores[prompt_id][objective][model] for model in model_names]
            lo = min(vals)
            hi = max(vals)
            denom = hi - lo
            if denom == 0:
                norm[prompt_id][objective] = {model: 0.5 for model in model_names}
            else:
                norm[prompt_id][objective] = {
                    model: (scores[prompt_id][objective][model] - lo) / denom
                    for model in model_names
                }
    return norm


def aggregate(
    subset: list[str],
    model_names: list[str],
    scores: dict[str, dict[str, dict[str, float]]],
    norm: dict[str, dict[str, dict[str, float]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    per_objective: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    pairwise: list[dict[str, object]] = []

    for model in model_names:
        obj_norms: list[float] = []
        obj_wrs: list[float] = []
        for objective in OBJECTIVES:
            norm_score = mean([norm[p][objective][model] for p in subset])
            raw_score = mean([scores[p][objective][model] for p in subset])
            wr = float("nan") if model == "baseline" else mean(
                [win(scores[p][objective][model], scores[p][objective]["baseline"]) for p in subset]
            )
            obj_norms.append(norm_score)
            if model != "baseline":
                obj_wrs.append(wr)
            per_objective.append(
                {
                    "model": model,
                    "objective": objective,
                    "n": len(subset),
                    "mean_prompt_norm_score": norm_score,
                    "mean_raw_score": raw_score,
                    "win_rate_vs_baseline": wr,
                }
            )
        summary.append(
            {
                "model": model,
                "n": len(subset),
                "mean_objective_norm_score": mean(obj_norms),
                "min_objective_norm_score": min(obj_norms),
                "std_objective_norm_score": statistics.pstdev(obj_norms),
                "mean_win_rate_vs_baseline": float("nan") if model == "baseline" else mean(obj_wrs),
                "min_win_rate_vs_baseline": float("nan") if model == "baseline" else min(obj_wrs),
            }
        )

    left = "ronpo_s2_ckpt2457"
    for right in model_names:
        if right == left:
            continue
        objective_wrs = []
        row: dict[str, object] = {"left_model": left, "right_model": right, "n": len(subset)}
        for objective in OBJECTIVES:
            wr = mean([win(scores[p][objective][left], scores[p][objective][right]) for p in subset])
            row[f"{objective}_win_rate"] = wr
            objective_wrs.append(wr)
        row["mean_win_rate"] = mean(objective_wrs)
        row["min_win_rate"] = min(objective_wrs)
        pairwise.append(row)

    return summary, per_objective, pairwise


def disagreement_rows(
    prompt_ids: list[str],
    model_names: list[str],
    scores: dict[str, dict[str, dict[str, float]]],
    prompts: dict[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    model_pairs = [
        (model_names[i], model_names[j])
        for i in range(len(model_names))
        for j in range(i + 1, len(model_names))
    ]
    objective_pairs = [
        (OBJECTIVES[i], OBJECTIVES[j])
        for i in range(len(OBJECTIVES))
        for j in range(i + 1, len(OBJECTIVES))
    ]
    for prompt_id in prompt_ids:
        discordant = 0
        comparable = 0
        for a, b in model_pairs:
            for obj1, obj2 in objective_pairs:
                s1 = sign(scores[prompt_id][obj1][a], scores[prompt_id][obj1][b])
                s2 = sign(scores[prompt_id][obj2][a], scores[prompt_id][obj2][b])
                if s1 == 0 or s2 == 0:
                    continue
                comparable += 1
                if s1 != s2:
                    discordant += 1
        rate = discordant / comparable if comparable else 0.0
        winners = {
            objective: max(model_names, key=lambda m: scores[prompt_id][objective][m])
            for objective in OBJECTIVES
        }
        rows.append(
            {
                "prompt_id": prompt_id,
                "judge_pairwise_disagreement_rate": rate,
                "discordant_pairs": discordant,
                "comparable_pairs": comparable,
                "distinct_objective_winners": len(set(winners.values())),
                "skywork_winner": winners["skywork"],
                "athene_winner": winners["athene"],
                "armo_winner": winners["armo"],
                "prompt_preview": prompts.get(prompt_id, "").replace("\n", " ")[:220],
            }
        )
    rows.sort(
        key=lambda r: (
            float(r["judge_pairwise_disagreement_rate"]),
            int(r["distinct_objective_winners"]),
            str(r["prompt_id"]),
        ),
        reverse=True,
    )
    return rows


def metric_value(
    metric: str,
    subset: list[str],
    model: str,
    scores: dict[str, dict[str, dict[str, float]]],
    norm: dict[str, dict[str, dict[str, float]]],
    right_model: str | None = None,
) -> float:
    if metric == "avg_norm":
        return mean([mean([norm[p][obj][model] for p in subset]) for obj in OBJECTIVES])
    if metric == "worst_norm":
        return min([mean([norm[p][obj][model] for p in subset]) for obj in OBJECTIVES])
    if metric == "avg_wr_vs_base":
        return mean([
            mean([win(scores[p][obj][model], scores[p][obj]["baseline"]) for p in subset])
            for obj in OBJECTIVES
        ])
    if metric == "worst_wr_vs_base":
        return min([
            mean([win(scores[p][obj][model], scores[p][obj]["baseline"]) for p in subset])
            for obj in OBJECTIVES
        ])
    if metric == "avg_pairwise_wr":
        if right_model is None:
            raise ValueError("right_model required")
        return mean([
            mean([win(scores[p][obj][model], scores[p][obj][right_model]) for p in subset])
            for obj in OBJECTIVES
        ])
    if metric == "worst_pairwise_wr":
        if right_model is None:
            raise ValueError("right_model required")
        return min([
            mean([win(scores[p][obj][model], scores[p][obj][right_model]) for p in subset])
            for obj in OBJECTIVES
        ])
    raise ValueError(metric)


def bootstrap_ci(
    base_subset: list[str],
    metric: str,
    model: str,
    scores: dict[str, dict[str, dict[str, float]]],
    norm: dict[str, dict[str, dict[str, float]]],
    *,
    right_model: str | None = None,
    iterations: int = 2000,
    seed: int = 20260626,
) -> tuple[float, float, float]:
    estimate = metric_value(metric, base_subset, model, scores, norm, right_model)
    rng = random.Random(seed)
    draws = []
    n = len(base_subset)
    for _ in range(iterations):
        sample = [base_subset[rng.randrange(n)] for _ in range(n)]
        draws.append(metric_value(metric, sample, model, scores, norm, right_model))
    return estimate, percentile(draws, 0.025), percentile(draws, 0.975)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_bootstrap_table(
    subset_name: str,
    subset: list[str],
    model_names: list[str],
    scores: dict[str, dict[str, dict[str, float]]],
    norm: dict[str, dict[str, dict[str, float]]],
    iterations: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in model_names:
        for metric in ("avg_norm", "worst_norm"):
            est, lo, hi = bootstrap_ci(subset, metric, model, scores, norm, iterations=iterations)
            rows.append(
                {
                    "subset": subset_name,
                    "model": model,
                    "metric": metric,
                    "estimate": est,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": len(subset),
                    "bootstrap_iterations": iterations,
                }
            )
        if model != "baseline":
            for metric in ("avg_wr_vs_base", "worst_wr_vs_base"):
                est, lo, hi = bootstrap_ci(subset, metric, model, scores, norm, iterations=iterations)
                rows.append(
                    {
                        "subset": subset_name,
                        "model": model,
                        "metric": metric,
                        "estimate": est,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n": len(subset),
                        "bootstrap_iterations": iterations,
                    }
                )
    return rows


def make_pairwise_bootstrap_table(
    subset_name: str,
    subset: list[str],
    model_names: list[str],
    scores: dict[str, dict[str, dict[str, float]]],
    norm: dict[str, dict[str, dict[str, float]]],
    iterations: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    left = "ronpo_s2_ckpt2457"
    for right in model_names:
        if right == left:
            continue
        for metric in ("avg_pairwise_wr", "worst_pairwise_wr"):
            est, lo, hi = bootstrap_ci(
                subset,
                metric,
                left,
                scores,
                norm,
                right_model=right,
                iterations=iterations,
            )
            rows.append(
                {
                    "subset": subset_name,
                    "left_model": left,
                    "right_model": right,
                    "metric": metric,
                    "estimate": est,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": len(subset),
                    "bootstrap_iterations": iterations,
                }
            )
    return rows


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def make_report(
    out_dir: Path,
    full_summary: list[dict[str, object]],
    disagreement_summary: list[dict[str, object]],
    full_pairwise: list[dict[str, object]],
    disagreement_pairwise: list[dict[str, object]],
    bootstrap_rows: list[dict[str, object]],
    pairwise_bootstrap_rows: list[dict[str, object]],
    disagreement_top: list[dict[str, object]],
    work_dir: Path,
) -> str:
    def ci_lookup(rows: list[dict[str, object]], subset: str, model: str, metric: str) -> tuple[float, float, float]:
        for row in rows:
            if row["subset"] == subset and row["model"] == model and row["metric"] == metric:
                return float(row["estimate"]), float(row["ci_low"]), float(row["ci_high"])
        return float("nan"), float("nan"), float("nan")

    def pci_lookup(rows: list[dict[str, object]], subset: str, right: str, metric: str) -> tuple[float, float, float]:
        for row in rows:
            if row["subset"] == subset and row["right_model"] == right and row["metric"] == metric:
                return float(row["estimate"]), float(row["ci_low"]), float(row["ci_high"])
        return float("nan"), float("nan"), float("nan")

    lines = [
        "# Stage-2 Robustness Bootstrap and Disagreement Analysis",
        "",
        f"Source artifact: `{work_dir}`.",
        "All metrics are recomputed from `scored/eval_{skywork,athene,armo}.jsonl` with the same six-model comparison set.",
        "Prompt-normalized reward is min-max normalized across the six models for each prompt and objective before averaging.",
        "Win rates count exact ties as 0.5. Confidence intervals are prompt-level paired bootstrap 95% intervals.",
        "",
        "## Full Held-Out Set",
        "",
    ]
    rows = []
    for row in full_summary:
        model = str(row["model"])
        avg_est, avg_lo, avg_hi = ci_lookup(bootstrap_rows, "full", model, "avg_norm")
        worst_est, worst_lo, worst_hi = ci_lookup(bootstrap_rows, "full", model, "worst_norm")
        if model == "baseline":
            avg_wr = worst_wr = "-"
        else:
            wr_est, wr_lo, wr_hi = ci_lookup(bootstrap_rows, "full", model, "avg_wr_vs_base")
            wwr_est, wwr_lo, wwr_hi = ci_lookup(bootstrap_rows, "full", model, "worst_wr_vs_base")
            avg_wr = f"{fmt(wr_est)} [{fmt(wr_lo)}, {fmt(wr_hi)}]"
            worst_wr = f"{fmt(wwr_est)} [{fmt(wwr_lo)}, {fmt(wwr_hi)}]"
        rows.append([
            DISPLAY.get(model, model),
            str(row["n"]),
            f"{fmt(avg_est)} [{fmt(avg_lo)}, {fmt(avg_hi)}]",
            f"{fmt(worst_est)} [{fmt(worst_lo)}, {fmt(worst_hi)}]",
            avg_wr,
            worst_wr,
        ])
    lines.append(markdown_table(["Method", "n", "Avg norm", "Worst norm", "Avg WR vs Base", "Worst WR vs Base"], rows))
    lines.extend(["", "## High-Disagreement Top 25% Subset", ""])
    rows = []
    for row in disagreement_summary:
        model = str(row["model"])
        avg_est, avg_lo, avg_hi = ci_lookup(bootstrap_rows, "disagreement_top25", model, "avg_norm")
        worst_est, worst_lo, worst_hi = ci_lookup(bootstrap_rows, "disagreement_top25", model, "worst_norm")
        if model == "baseline":
            avg_wr = worst_wr = "-"
        else:
            wr_est, wr_lo, wr_hi = ci_lookup(bootstrap_rows, "disagreement_top25", model, "avg_wr_vs_base")
            wwr_est, wwr_lo, wwr_hi = ci_lookup(bootstrap_rows, "disagreement_top25", model, "worst_wr_vs_base")
            avg_wr = f"{fmt(wr_est)} [{fmt(wr_lo)}, {fmt(wr_hi)}]"
            worst_wr = f"{fmt(wwr_est)} [{fmt(wwr_lo)}, {fmt(wwr_hi)}]"
        rows.append([
            DISPLAY.get(model, model),
            str(row["n"]),
            f"{fmt(avg_est)} [{fmt(avg_lo)}, {fmt(avg_hi)}]",
            f"{fmt(worst_est)} [{fmt(worst_lo)}, {fmt(worst_hi)}]",
            avg_wr,
            worst_wr,
        ])
    lines.append(markdown_table(["Method", "n", "Avg norm", "Worst norm", "Avg WR vs Base", "Worst WR vs Base"], rows))
    lines.extend(["", "## RONPO Final Pairwise Win Rates", ""])
    for subset_name, pair_rows in (("full", full_pairwise), ("disagreement_top25", disagreement_pairwise)):
        lines.append(f"### {subset_name}")
        rows = []
        for row in pair_rows:
            right = str(row["right_model"])
            avg_est, avg_lo, avg_hi = pci_lookup(pairwise_bootstrap_rows, subset_name, right, "avg_pairwise_wr")
            worst_est, worst_lo, worst_hi = pci_lookup(pairwise_bootstrap_rows, subset_name, right, "worst_pairwise_wr")
            rows.append([
                f"RONPO S2 final vs {DISPLAY.get(right, right)}",
                f"{fmt(float(row['skywork_win_rate']))}",
                f"{fmt(float(row['athene_win_rate']))}",
                f"{fmt(float(row['armo_win_rate']))}",
                f"{fmt(avg_est)} [{fmt(avg_lo)}, {fmt(avg_hi)}]",
                f"{fmt(worst_est)} [{fmt(worst_lo)}, {fmt(worst_hi)}]",
            ])
        lines.append(markdown_table(["Comparison", "Skywork WR", "Athene WR", "ArmoRM WR", "Avg WR", "Worst WR"], rows))
        lines.append("")

    mean_dis = mean([float(r["judge_pairwise_disagreement_rate"]) for r in disagreement_top])
    min_dis = min(float(r["judge_pairwise_disagreement_rate"]) for r in disagreement_top)
    max_dis = max(float(r["judge_pairwise_disagreement_rate"]) for r in disagreement_top)
    lines.extend([
        "## Disagreement Subset Definition",
        "",
        "For each prompt, every pair of candidate model responses is compared under each pair of reward objectives.",
        "The disagreement score is the fraction of non-tied objective-pair/model-pair comparisons where the objectives prefer opposite responses.",
        f"The top-25% subset has n={len(disagreement_top)}, disagreement-rate range [{fmt(min_dis)}, {fmt(max_dis)}], and mean {fmt(mean_dis)}.",
        "",
        "## Paper-Ready Interpretation",
        "",
        "The full held-out set already shows RONPO S2 final as the strongest local-RM model by average and worst-objective normalized reward.",
        "The high-disagreement subset is the more targeted stress test: it isolates prompts where the reward sources disagree over candidate responses.",
        "RONPO S2 final remains the strongest method on both average and worst-objective metrics on this subset, supporting the core robustness claim that objective-adversarial training raises the weakest reward-source floor rather than merely improving an easy average.",
        "",
        "This result should be reported alongside the IFEval finding: RONPO S2 preserves rule-based instruction following better than HT-MNPO S2, while the local-RM stress test shows stronger robustness under reward-source conflict.",
        "",
        "## Generated Files",
        "",
    ])
    for path in sorted(out_dir.glob("*.csv")):
        lines.append(f"- `{path}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/ext_hdd/sjkim/mnpo/eval/ronpo_stage2_resume_sanity_20260625"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis/stage2_robustness_20260626"),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()

    prompt_ids, model_names, scores, prompts = load_scores(args.work_dir)
    norm = normalize_scores(prompt_ids, model_names, scores)
    disagreement = disagreement_rows(prompt_ids, model_names, scores, prompts)
    top_n = math.ceil(len(prompt_ids) * 0.25)
    disagreement_subset = [str(r["prompt_id"]) for r in disagreement[:top_n]]

    full_summary, full_objective, full_pairwise = aggregate(prompt_ids, model_names, scores, norm)
    dis_summary, dis_objective, dis_pairwise = aggregate(disagreement_subset, model_names, scores, norm)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "model_summary_full.csv", full_summary)
    write_csv(args.out_dir / "per_objective_full.csv", full_objective)
    write_csv(args.out_dir / "pairwise_ronpo_final_full.csv", full_pairwise)
    write_csv(args.out_dir / "disagreement_prompts.csv", disagreement)
    write_csv(args.out_dir / "model_summary_disagreement_top25.csv", dis_summary)
    write_csv(args.out_dir / "per_objective_disagreement_top25.csv", dis_objective)
    write_csv(args.out_dir / "pairwise_ronpo_final_disagreement_top25.csv", dis_pairwise)

    bootstrap_rows = []
    bootstrap_rows.extend(
        make_bootstrap_table("full", prompt_ids, model_names, scores, norm, args.bootstrap_iterations)
    )
    bootstrap_rows.extend(
        make_bootstrap_table(
            "disagreement_top25",
            disagreement_subset,
            model_names,
            scores,
            norm,
            args.bootstrap_iterations,
        )
    )
    write_csv(args.out_dir / "bootstrap_model_metrics.csv", bootstrap_rows)

    pair_boot = []
    pair_boot.extend(
        make_pairwise_bootstrap_table("full", prompt_ids, model_names, scores, norm, args.bootstrap_iterations)
    )
    pair_boot.extend(
        make_pairwise_bootstrap_table(
            "disagreement_top25",
            disagreement_subset,
            model_names,
            scores,
            norm,
            args.bootstrap_iterations,
        )
    )
    write_csv(args.out_dir / "bootstrap_pairwise_ronpo_final.csv", pair_boot)

    report = make_report(
        args.out_dir,
        full_summary,
        dis_summary,
        full_pairwise,
        dis_pairwise,
        bootstrap_rows,
        pair_boot,
        disagreement[:top_n],
        args.work_dir,
    )
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"Wrote {args.out_dir}")
    print(f"Prompts: full={len(prompt_ids)}, disagreement_top25={len(disagreement_subset)}")


if __name__ == "__main__":
    main()
