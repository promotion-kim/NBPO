#!/usr/bin/env python3
"""Aggregate the fixed P4 validation panel for the two new Stage-1 arms.

This is a descriptive, same-split comparison only.  It never reads the spent
604-prompt split and writes every reported number from the two scorer JSONL
files.  Normalization is recomputed over the complete eligible pool.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


DISPLAY = {
    "base": "Base",
    "ronpo_os_confirmatory": "RONPO (OS, prior Stage-1)",
    "inpo_avg": "INPO (avg, prior Stage-1)",
    "sppo_avg": "SPPO (avg, prior Stage-1)",
    "simpo": "SimPO (prior Stage-1)",
    "ipo": "IPO (prior Stage-1)",
    "dpo": "DPO (prior Stage-1)",
    "ht_mnpo_harmless": "HT-MNPO (harmless, prior Stage-1)",
    "ht_mnpo_helpfulness": "HT-MNPO (help., prior Stage-1)",
    "ronpo_topmass_stage1_replicate": "RONPO (top-mass, Stage-1 replicate)",
    "ronpo_softmin_lb_stage1": "RONPO (soft-min lower-bound target, Stage-1)",
    "ronpo_os_stage2": "RONPO (OS, Stage-2)",
    "ronpo_topmass_stage2": "RONPO (top-mass, Stage-2)",
    "inpo_avg_stage2": "INPO (avg, Stage-2)",
    "sppo_avg_stage2": "SPPO (avg, Stage-2)",
    "simpo_stage2": "SimPO (Stage-2)",
    "ipo_stage2": "IPO (Stage-2)",
    "dpo_stage2": "DPO (Stage-2)",
    "ht_mnpo_harmless_stage2": "HT-MNPO (harmless, Stage-2)",
    "ht_mnpo_helpfulness_stage2": "HT-MNPO (help., Stage-2)",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ci(samples: np.ndarray) -> list[float]:
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--helpfulness", type=Path, required=True)
    parser.add_argument("--harmlessness", type=Path, required=True)
    parser.add_argument("--pool-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scope",
        default="descriptive fixed P4 49-prompt validation panel; not a fresh confirmation",
        help="Verbatim scope statement preserved in the machine-readable summary and report.",
    )
    parser.add_argument(
        "--report-title",
        default="P5 Stage-1 fixed-panel comparison",
        help="Human-readable report title. Does not affect any metric.",
    )
    parser.add_argument(
        "--ronpo-arm",
        action="append",
        default=[],
        help="Eligible RONPO arm to compare against the best non-RONPO trained arm. May be repeated.",
    )
    parser.add_argument(
        "--comparison-label",
        default="best eligible non-RONPO baseline",
        help="Report-only label for the fixed comparison set; does not affect metrics.",
    )
    parser.add_argument(
        "--display-name-map",
        type=Path,
        default=None,
        help="Optional JSON object mapping model IDs to report-only display names.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    display = dict(DISPLAY)
    if args.display_name_map is not None:
        overrides = json.loads(args.display_name_map.read_text(encoding="utf-8"))
        if not isinstance(overrides, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in overrides.items()):
            raise RuntimeError("display-name map must be a JSON object of string model IDs to string labels")
        display.update(overrides)
    audit = json.loads(args.pool_audit.read_text(encoding="utf-8"))
    # Standard paper pools name gate-passing entries ``eligible_models``.  A
    # separately labelled diagnostic pool intentionally has no eligible set
    # because it preserves failed rows, so it supplies ``models`` instead.
    models = list(audit.get("eligible_models", audit.get("models", [])))
    if not models or models[0] != "base":
        raise RuntimeError("base must be the first eligible model")
    hrows, srows = read_jsonl(args.helpfulness), read_jsonl(args.harmlessness)
    hmap = {str(row["prompt_id"]): row for row in hrows}
    smap = {str(row["prompt_id"]): row for row in srows}
    prompt_ids = sorted(hmap)
    if not prompt_ids or set(prompt_ids) != set(smap) or len(prompt_ids) != len(hrows):
        raise RuntimeError("scorer prompt IDs are missing, duplicate, or mismatched")
    raw = np.empty((len(prompt_ids), 2, len(models)), dtype=np.float64)
    for pidx, prompt_id in enumerate(prompt_ids):
        for oidx, row in enumerate((hmap[prompt_id], smap[prompt_id])):
            if list(row["response_model_names"]) != models:
                raise RuntimeError(f"model ordering mismatch on prompt {prompt_id}")
            values = np.asarray(row["all_rm_scores"], dtype=np.float64)
            if values.shape != (len(models),) or not np.isfinite(values).all():
                raise RuntimeError(f"non-finite or malformed score at {prompt_id}")
            raw[pidx, oidx] = values
    lo, hi = raw.min(axis=2, keepdims=True), raw.max(axis=2, keepdims=True)
    norm = np.where(hi == lo, 0.5, (raw - lo) / (hi - lo))
    prompt_avg, prompt_worst = norm.mean(axis=1), norm.min(axis=1)
    base_idx = models.index("base")
    wins = (raw > raw[:, :, base_idx, None]).astype(float)
    wins += 0.5 * (raw == raw[:, :, base_idx, None])
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(prompt_ids), size=(args.bootstrap, len(prompt_ids)))
    rows, per_prompt = [], []
    for midx, model in enumerate(models):
        objective_wins = wins[:, :, midx].mean(axis=0)
        row = {
            "model": model,
            "display_name": display.get(model, model),
            "records": len(prompt_ids),
            "helpfulness_raw": float(raw[:, 0, midx].mean()),
            "harmlessness_raw": float(raw[:, 1, midx].mean()),
            "helpfulness_norm": float(norm[:, 0, midx].mean()),
            "harmlessness_norm": float(norm[:, 1, midx].mean()),
            "mean_objective_norm_score": float(prompt_avg[:, midx].mean()),
            "mean_prompt_worst_norm_score": float(prompt_worst[:, midx].mean()),
            "mean_prompt_worst_norm_score_ci95": ci(prompt_worst[:, midx][indices].mean(axis=1)),
            "mean_win_rate_vs_base": float(objective_wins.mean()),
            "min_win_rate_vs_base": float(objective_wins.min()),
        }
        rows.append(row)
        for pidx, prompt_id in enumerate(prompt_ids):
            per_prompt.append({
                "prompt_id": prompt_id, "model": model,
                "helpfulness_raw": float(raw[pidx, 0, midx]),
                "harmlessness_raw": float(raw[pidx, 1, midx]),
                "helpfulness_norm": float(norm[pidx, 0, midx]),
                "harmlessness_norm": float(norm[pidx, 1, midx]),
                "prompt_avg_norm": float(prompt_avg[pidx, midx]),
                "prompt_worst_norm": float(prompt_worst[pidx, midx]),
            })
    ranked = sorted((row for row in rows if row["model"] != "base"), key=lambda row: (-row["mean_prompt_worst_norm_score"], row["model"]))
    for index, row in enumerate(ranked, 1):
        row["rank"] = index
    base = next(row for row in rows if row["model"] == "base")
    base["rank"] = len(rows)
    presentation = ranked + [base]
    comparisons = {}
    comparison_arms = args.ronpo_arm or ["ronpo_topmass_stage1_replicate", "ronpo_softmin_lb_stage1"]
    for arm in comparison_arms:
        if arm not in models:
            continue
        nonronpo = [row for row in rows if row["model"] != "base" and not row["model"].startswith("ronpo_")]
        comparator = max(nonronpo, key=lambda row: (row["mean_prompt_worst_norm_score"], row["model"]))
        delta = prompt_worst[:, models.index(arm)] - prompt_worst[:, models.index(comparator["model"])]
        comparisons[arm] = {
            "worst_comparator": comparator["model"],
            "paired_prompt_worst_difference": float(delta.mean()),
            "paired_prompt_worst_difference_ci95": ci(delta[indices].mean(axis=1)),
        }
    fieldnames = list(presentation[0])
    with (args.output_dir / "per_objective_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(presentation)
    with (args.output_dir / "per_prompt_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_prompt[0])); writer.writeheader(); writer.writerows(per_prompt)
    result = {
        "status": "complete", "scope": args.scope,
        "primary": "mean_prompt_worst_norm_score",
        "normalization": "per-prompt minmax over the full eligible model pool; constant objective prompt=0.5",
        "objectives": ["Beaver reward helpfulness", "negative Beaver cost harmlessness"],
        "records": len(prompt_ids), "eligible_models": models, "ranking": presentation,
        "new_arm_paired_comparisons": comparisons,
        "bootstrap": {"resamples": args.bootstrap, "seed": args.seed, "unit": "prompt", "paired": True},
        "input_sha256": {"helpfulness": sha(args.helpfulness), "harmlessness": sha(args.harmlessness), "pool_audit": sha(args.pool_audit)},
        "spent_sealed_split_touched": False,
    }
    (args.output_dir / "ranked_validation_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = [f"# {args.report_title}", "", result["scope"], "", "| Method | Help. | Harmless | Avg | Worst (95% CI) |", "|---|---:|---:|---:|---:|"]
    for row in presentation:
        interval = row["mean_prompt_worst_norm_score_ci95"]
        report.append(f"| {row['display_name']} | {row['helpfulness_norm']:.4f} | {row['harmlessness_norm']:.4f} | {row['mean_objective_norm_score']:.4f} | {row['mean_prompt_worst_norm_score']:.4f} [{interval[0]:.4f}, {interval[1]:.4f}] |")
    report += ["", f"## Paired comparisons against the {args.comparison_label}", "", "```json", json.dumps(comparisons, indent=2), "```", ""]
    (args.output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"summary": str(args.output_dir / "ranked_validation_summary.json"), "models": models, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
