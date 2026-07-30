#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np

HEADS = ("instruction_following", "truthfulness", "honesty", "helpfulness", "safety")
BASELINES = ("Base", "RMOD K=16", "INPO", "SPPO", "DPO", "IPO", "SimPO")
RONPO = "RONPO-COMB-S6"


def ci(x, rng, n_boot):
    n = len(x)
    draw = x[rng.integers(0, n, (n_boot, n))].mean(1)
    return [float(v) for v in np.quantile(draw, (0.025, 0.975))]


def robust_delta(x, y, rng, n_boot):
    n = len(x)
    idx = rng.integers(0, n, (n_boot, n))
    draw = x[idx].mean(1).min(1) - y[idx].mean(1).min(1)
    return {
        "estimate": float(x.mean(0).min() - y.mean(0).min()),
        "ci95": [float(v) for v in np.quantile(draw, (0.025, 0.975))],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--pool-audit", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    audit = json.loads(a.pool_audit.read_text())
    models = audit["models"]
    labels = [m["label"] for m in models]
    expected = [*BASELINES[:2], "RONPO-S2", RONPO, *BASELINES[2:]]
    if labels != expected:
        raise RuntimeError(f"policy order changed: {labels}")

    arrays, prompts = [], None
    for head in HEADS:
        with (a.scores / f"joint_{head}.jsonl").open() as f:
            rows = [json.loads(x) for x in f if x.strip()]
        current = [r["prompt"] for r in rows]
        if prompts is None:
            prompts = current
        elif prompts != current:
            raise RuntimeError("prompt order differs across reward heads")
        arrays.append(np.asarray([r["all_rm_scores"] for r in rows], dtype=np.float64))
    values = np.stack(arrays, axis=2)
    if values.shape != (586, 9, 5):
        raise RuntimeError(f"unexpected score tensor {values.shape}")

    rng = np.random.default_rng(a.seed)
    result = {
        "protocol": {
            "n_prompts": 586,
            "policy_count": 9,
            "heads": list(HEADS),
            "single_bf16_batch_context": True,
            "bootstrap_resamples": a.bootstrap,
            "bootstrap_seed": a.seed,
        },
        "methods": {},
        "comparisons": {},
    }
    table = []
    base_i = labels.index("Base")
    for j, label in enumerate(labels):
        x = values[:, j]
        means = x.mean(0)
        worst_i = int(means.argmin())
        delta = x - values[:, base_i]
        record = {
            "mean_by_objective": dict(zip(HEADS, map(float, means))),
            "average_of_objective_means": float(means.mean()),
            "worst_objective": HEADS[worst_i],
            "worst_objective_mean": float(means[worst_i]),
            "paired_delta_vs_base": dict(zip(HEADS, map(float, delta.mean(0)))),
            "paired_delta_vs_base_ci95": {h: ci(delta[:, k], rng, a.bootstrap) for k, h in enumerate(HEADS)},
            "paired_average_delta_vs_base": float(delta.mean(1).mean()),
            "paired_average_delta_vs_base_ci95": ci(delta.mean(1), rng, a.bootstrap),
        }
        result["methods"][label] = record
        table.append({"method": label, **record["mean_by_objective"],
                      "avg": record["average_of_objective_means"],
                      "worst": record["worst_objective_mean"]})

    r = labels.index(RONPO)
    rx = values[:, r]
    baseline_idx = [labels.index(x) for x in BASELINES]
    baseline_means = values[:, baseline_idx].mean(0)
    envelope = baseline_means.max(0)
    head_margin = rx.mean(0) - envelope
    n = len(rx)
    idx = rng.integers(0, n, (a.bootstrap, n))
    r_draw = rx[idx].mean(1)
    b_draw = np.stack([values[:, j][idx].mean(1) for j in baseline_idx], axis=2)
    envelope_draw = b_draw.max(2)
    min_margin_draw = (r_draw - envelope_draw).min(1)
    primary = {
        "statistic": "min_head(ronpo_s6_mean - max_non_ronpo_mean)",
        "estimate": float(head_margin.min()),
        "ci95": [float(v) for v in np.quantile(min_margin_draw, (0.025, 0.975))],
        "margin_by_objective": dict(zip(HEADS, map(float, head_margin))),
        "baseline_envelope_by_objective": dict(zip(HEADS, map(float, envelope))),
    }
    primary["pass"] = bool(primary["ci95"][0] > 0)
    result["comparisons"]["primary_all_objective_envelope"] = primary

    for label in BASELINES:
        j = labels.index(label)
        delta = rx - values[:, j]
        result["comparisons"][f"{RONPO}_minus_{label}"] = {
            "paired_delta_by_objective": dict(zip(HEADS, map(float, delta.mean(0)))),
            "paired_delta_ci95": {h: ci(delta[:, k], rng, a.bootstrap) for k, h in enumerate(HEADS)},
            "average_delta": float(delta.mean(1).mean()),
            "average_delta_ci95": ci(delta.mean(1), rng, a.bootstrap),
            "robust_floor_delta": robust_delta(rx, values[:, j], rng, a.bootstrap),
            "gain_range": float(np.ptp(delta.mean(0))),
            "gain_std": float(np.std(delta.mean(0))),
        }

    means = values.mean(0)
    lo, hi = means.min(0), means.max(0)
    normalized = (means - lo) / np.where(hi > lo, hi - lo, 1.0)
    result["figure5_normalized"] = {
        label: dict(zip(HEADS, map(float, normalized[j]))) for j, label in enumerate(labels)
    }

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "paired_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (a.out / "per_policy_means.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=table[0].keys())
        w.writeheader()
        w.writerows(table)


if __name__ == "__main__":
    main()
