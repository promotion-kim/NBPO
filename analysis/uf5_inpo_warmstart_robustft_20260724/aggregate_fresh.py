#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np

HEADS = ("instruction_following", "truthfulness", "honesty", "helpfulness", "safety")
RONPO = "RONPO-FT-S2"
CONTROL = "INPO-CONTROL-S2"


def ci(x, rng, n_boot):
    n = len(x)
    draw = x[rng.integers(0, n, (n_boot, n))].mean(1)
    return [float(v) for v in np.quantile(draw, (0.025, 0.975))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--pool-audit", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    audit = json.loads(a.pool_audit.read_text())
    labels = [x["label"] for x in audit["models"]]
    expected = ["Base", "INPO-S1-PARENT", "DPO", "IPO", "SPPO", "SimPO", CONTROL, RONPO]
    if labels != expected:
        raise RuntimeError(f"policy order changed: {labels}")

    mats, prompts = [], None
    for head in HEADS:
        with (a.scores / f"joint_{head}.jsonl").open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        current = [x["prompt"] for x in rows]
        if prompts is None:
            prompts = current
        elif prompts != current:
            raise RuntimeError("prompt order differs across heads")
        mats.append(np.asarray([x["all_rm_scores"] for x in rows], dtype=np.float64))
    values = np.stack(mats, axis=2)
    if values.shape != (646, 8, 5):
        raise RuntimeError(values.shape)
    rng = np.random.default_rng(a.seed)
    result = {
        "protocol": {
            "n_prompts": 646,
            "heads": list(HEADS),
            "single_bf16_batch_context": True,
            "bootstrap_resamples": a.bootstrap,
            "bootstrap_seed": a.seed,
        },
        "methods": {},
        "comparisons": {},
    }
    rows_out = []
    for j, label in enumerate(labels):
        means = values[:, j].mean(0)
        result["methods"][label] = {
            "mean_by_objective": dict(zip(HEADS, map(float, means))),
            "average_of_objective_means": float(means.mean()),
            "worst_objective": HEADS[int(means.argmin())],
            "worst_objective_mean": float(means.min()),
        }
        rows_out.append({"method": label, **dict(zip(HEADS, map(float, means))),
                         "avg": float(means.mean()), "worst": float(means.min())})

    r = labels.index(RONPO)
    for comparator in ("Base", "INPO-S1-PARENT", "DPO", "IPO", "SPPO", "SimPO", CONTROL):
        c = labels.index(comparator)
        delta = values[:, r] - values[:, c]
        n = len(delta)
        idx = rng.integers(0, n, (a.bootstrap, n))
        head_draw = delta[idx].mean(1)
        min_draw = head_draw.min(1)
        means = delta.mean(0)
        record = {
            "paired_delta_by_objective": dict(zip(HEADS, map(float, means))),
            "paired_delta_ci95": {h: ci(delta[:, k], rng, a.bootstrap) for k, h in enumerate(HEADS)},
            "minimum_head_delta": float(means.min()),
            "minimum_head_delta_ci95": [float(v) for v in np.quantile(min_draw, (0.025, 0.975))],
            "average_delta": float(delta.mean(1).mean()),
            "average_delta_ci95": ci(delta.mean(1), rng, a.bootstrap),
            "gain_range": float(np.ptp(means)),
            "gain_std": float(np.std(means)),
        }
        result["comparisons"][f"{RONPO}_minus_{comparator}"] = record
    primary = result["comparisons"][f"{RONPO}_minus_{CONTROL}"]
    primary["pass"] = bool(primary["minimum_head_delta_ci95"][0] > 0)
    result["primary"] = {
        "name": "RONPO robust fine-tuning versus matched INPO continuation",
        **primary,
    }

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "paired_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (a.out / "per_policy_means.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        w.writeheader()
        w.writerows(rows_out)


if __name__ == "__main__":
    main()
