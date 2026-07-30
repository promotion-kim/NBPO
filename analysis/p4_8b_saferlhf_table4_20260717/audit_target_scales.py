#!/usr/bin/env python3
"""Measure, rather than conceal, the matched-budget arms' target scales."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk


def stats(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "mean_abs": float(np.abs(values).mean()),
        "p95_abs": float(np.quantile(np.abs(values), 0.95)),
        "count": int(values.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--kappa-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.kappa_lock.read_text())
    train = load_from_disk(str(args.input_dir))["train"]
    cols = set(train.column_names)
    choices = {float(row["entropy_target"]): float(row["selected_kappa"]) for row in lock["selected"]}
    tag = lambda value: f"{value:g}".replace(".", "p")
    out = {"scope": "pre-training target-scale audit; no outcome metric used", "rows": len(train), "arms": {}}
    for objective, kappa in sorted(choices.items()):
        column = f"target_os_k{tag(kappa)}"
        out["arms"][f"ronpo_os_entropy_{objective:g}"] = {"column": column, **stats(np.asarray(train[column], dtype=np.float64))}
    for column in [f"target_topmass_k{tag(choices[0.55])}", "target_uniform"]:
        out["arms"][column] = {"column": column, **stats(np.asarray(train[column], dtype=np.float64))}
    for column in ["ht_target", "ht_target_helpfulness"]:
        if column in cols:
            values = 0.0075 * np.asarray(train[column], dtype=np.float64)
            out["arms"][f"ht_mnpo_{column}"] = {"column": column, "trainer_scale": 0.0075, **stats(values)}
    out["arms"]["inpo_avg"] = {"analytic_target": "1/(2*eta)", "eta": 0.0075, **stats(np.full(len(train), 1.0 / (2 * 0.0075)))}
    out["arms"]["mnpo"] = {"analytic_target": "1/(2*eta)", "eta": 0.0075, **stats(np.full(len(train), 1.0 / (2 * 0.0075)))}
    out["arms"]["ipo"] = {"analytic_target": "1/(2*dpo_beta)", "dpo_beta": 0.05, **stats(np.full(len(train), 1.0 / (2 * 0.05)))}
    out["arms"]["dpo"] = {"analytic_target": "logistic pairwise objective; no fixed regression target"}
    out["arms"]["sppo_avg"] = {"analytic_target": "history probability objective; no fixed scalar target"}
    out["arms"]["simpo"] = {"analytic_target": "logistic pairwise objective; no fixed regression target"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
