#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

OBJECTIVES = ("instruction_following", "truthfulness", "honesty", "helpfulness", "safety")


def read_stable_jsonl(path, attempts=8):
    previous = None
    error = None
    for _ in range(attempts):
        try:
            data = path.read_bytes()
            current = (len(data), hashlib.sha256(data).hexdigest())
            rows = [json.loads(line) for line in data.splitlines() if line]
            if current == previous:
                return rows
            previous = current
        except Exception as exc:
            error = exc
            previous = None
        time.sleep(1)
    raise RuntimeError(f"unstable or invalid merged JSONL {path}: {error}")


def ci(values, rng, n_boot):
    n = len(values)
    means = values[rng.integers(0, n, size=(n_boot, n))].mean(1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


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
    mats = []
    prompts = None
    for objective in OBJECTIVES:
        rows = read_stable_jsonl(a.scores / f"joint_{objective}.jsonl")
        if prompts is None:
            prompts = [r["prompt"] for r in rows]
        elif prompts != [r["prompt"] for r in rows]:
            raise RuntimeError("objective prompt order mismatch")
        mat = np.asarray([r["all_rm_scores"] for r in rows], dtype=np.float64)
        if mat.shape != (audit["prompt_count"], audit["model_count"]):
            raise RuntimeError(f"bad {objective} shape {mat.shape}")
        mats.append(mat)
    values = np.stack(mats, axis=2)  # prompt, model, objective
    base = values[:, 0, :]
    rng = np.random.default_rng(a.seed)
    summary = {
        "protocol": {
            "objectives": list(OBJECTIVES), "n_prompts": len(prompts),
            "single_batch_context": True, "bootstrap_resamples": a.bootstrap,
            "bootstrap_seed": a.seed, "reference": models[0]["label"],
        },
        "methods": {},
    }
    table = []
    for j, model in enumerate(models):
        x = values[:, j, :]
        means = x.mean(0)
        delta = x - base
        worst_idx = int(np.argmin(means))
        worst_delta = delta[:, worst_idx]
        avg_delta = delta.mean(1)
        wci = ci(worst_delta, rng, a.bootstrap)
        record = {
            "n_prompts": len(prompts),
            "mean_by_objective": dict(zip(OBJECTIVES, map(float, means))),
            "average_of_objective_means": float(means.mean()),
            "worst_objective": OBJECTIVES[worst_idx],
            "worst_objective_mean": float(means[worst_idx]),
            "worst_head_paired_delta_vs_base": float(worst_delta.mean()),
            "worst_head_paired_delta_ci95": wci,
            "separates": bool(wci[0] > 0 or wci[1] < 0),
            "paired_delta_vs_base": dict(zip(OBJECTIVES, map(float, delta.mean(0)))),
            "paired_delta_ci95": {o: ci(delta[:, k], rng, a.bootstrap) for k, o in enumerate(OBJECTIVES)},
            "paired_average_delta_vs_base": float(avg_delta.mean()),
            "paired_average_delta_ci95": ci(avg_delta, rng, a.bootstrap),
        }
        summary["methods"][model["label"]] = record
        table.append({"method": model["label"], **record["mean_by_objective"], "avg": record["average_of_objective_means"], "worst": record["worst_objective_mean"], "worst_delta": record["worst_head_paired_delta_vs_base"], "worst_ci_low": wci[0], "worst_ci_high": wci[1]})
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "paired_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (a.out / "paired_means.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    (a.out / "prompt_manifest.json").write_text(json.dumps({"count": len(prompts), "prompts": prompts}, indent=2) + "\n")


if __name__ == "__main__":
    main()
