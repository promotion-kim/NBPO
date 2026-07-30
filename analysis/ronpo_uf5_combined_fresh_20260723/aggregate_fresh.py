#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np

OBJECTIVES = ("instruction_following", "truthfulness", "honesty", "helpfulness", "safety")


def quantile_boot(values, rng, n_boot):
    n = len(values)
    means = values[rng.integers(0, n, (n_boot, n))].mean(1)
    return [float(x) for x in np.quantile(means, (.025, .975))]


def robust_diff_boot(a, b, rng, n_boot):
    n = a.shape[0]
    idx = rng.integers(0, n, (n_boot, n))
    diffs = a[idx].mean(1).min(1) - b[idx].mean(1).min(1)
    return float(a.mean(0).min() - b.mean(0).min()), [float(x) for x in np.quantile(diffs, (.025, .975))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--pool-audit", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    audit = json.loads(a.pool_audit.read_text()); models = audit["models"]
    mats, prompts = [], None
    for objective in OBJECTIVES:
        rows = [json.loads(x) for x in (a.scores / f"joint_{objective}.jsonl").read_text().splitlines() if x]
        this = [x["prompt"] for x in rows]
        if prompts is None: prompts = this
        elif prompts != this: raise RuntimeError("prompt order mismatch")
        mats.append(np.asarray([x["all_rm_scores"] for x in rows], dtype=np.float64))
    values = np.stack(mats, axis=2)
    if values.shape != (647, len(models), 5): raise RuntimeError(values.shape)
    rng = np.random.default_rng(a.seed); base = values[:, 0, :]
    summary = {"protocol": {"objectives": list(OBJECTIVES), "n_prompts": 647, "single_batch_context": True, "bootstrap_resamples": a.bootstrap, "bootstrap_seed": a.seed, "fresh_panel": True}, "methods": {}, "comparisons": {}}
    table = []
    for j, model in enumerate(models):
        x = values[:, j, :]; means = x.mean(0); delta = x - base; k = int(np.argmin(means))
        record = {
            "mean_by_objective": dict(zip(OBJECTIVES, map(float, means))),
            "average_of_objective_means": float(means.mean()),
            "worst_objective": OBJECTIVES[k], "worst_objective_mean": float(means[k]),
            "worst_head_paired_delta_vs_base": float(delta[:, k].mean()),
            "worst_head_paired_delta_ci95": quantile_boot(delta[:, k], rng, a.bootstrap),
            "paired_delta_vs_base": dict(zip(OBJECTIVES, map(float, delta.mean(0)))),
            "paired_delta_ci95": {o: quantile_boot(delta[:, q], rng, a.bootstrap) for q, o in enumerate(OBJECTIVES)},
            "paired_average_delta_vs_base": float(delta.mean(1).mean()),
            "paired_average_delta_ci95": quantile_boot(delta.mean(1), rng, a.bootstrap),
        }
        summary["methods"][model["label"]] = record
        table.append({"method": model["label"], **record["mean_by_objective"], "avg": record["average_of_objective_means"], "worst": record["worst_objective_mean"]})
    labels = [x["label"] for x in models]
    if "RONPO-COMB-S6" not in labels or "RMOD K=16" not in labels: raise RuntimeError("primary models missing")
    c, r = labels.index("RONPO-COMB-S6"), labels.index("RMOD K=16")
    estimate, ci = robust_diff_boot(values[:, c, :], values[:, r, :], rng, a.bootstrap)
    summary["comparisons"]["primary_combined_s6_minus_rmod_k16"] = {"estimate": estimate, "ci95": ci, "pass": bool(ci[0] > 0), "statistic": "min_head_marginal_mean_difference"}
    for label in ("Base", "RONPO-SS-S4", "RONPO-FB-S5", "RONPO-COMB-S5"):
        if label in labels:
            j = labels.index(label); est, comp_ci = robust_diff_boot(values[:, c, :], values[:, j, :], rng, a.bootstrap)
            summary["comparisons"][f"combined_s6_minus_{label.lower().replace(' ','_')}"] = {"estimate": est, "ci95": comp_ci}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "paired_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (a.out / "paired_means.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)


if __name__ == "__main__":
    main()
