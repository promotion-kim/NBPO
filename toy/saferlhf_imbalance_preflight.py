#!/usr/bin/env python3
"""Analytic pre-flight for the preregistered SafeRLHF imbalance interaction.

This deliberately small game has one globally coupled policy parameter x: the
probability of a useful answer.  On every prompt helpfulness is x.  Only the
safety-active fraction (1-rho) carries a harmlessness training signal, with
score 1-x.  The uniform/adversary-off control maximizes the equally weighted
observed objective with a unit Bernoulli KL penalty.  The adaptive robust
control maximizes the minimum latent objective with the same reference and is
therefore x=1/2 by symmetry.

It is a *directional instrument check*, not a claim that the repository's
static target builder implements OMD.  In particular it makes transparent why
an adaptive adversary predicts an increasing margin as safety-active rows
become rare; the subsequent real-data static-vs-adaptive audit decides whether
that mechanism transfers to the implemented estimator.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rhos", default="0.5,0.75,0.9")
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.kl_weight <= 0:
        raise ValueError("--kl-weight must be positive")

    rows = []
    for rho in [float(value) for value in args.rhos.split(",")]:
        # d/dx [.5*x + .5*(1-rho)*(1-x) - lambda KL(x||.5)] = 0
        # hence logit(x) = rho/(2 lambda).
        uniform_x = sigmoid(rho / (2.0 * args.kl_weight))
        robust_x = 0.5
        control_worst = min(uniform_x, 1.0 - uniform_x)
        robust_worst = min(robust_x, 1.0 - robust_x)
        rows.append({
            "rho_benign": rho,
            "uniform_helpfulness": uniform_x,
            "uniform_harmlessness_active": 1.0 - uniform_x,
            "uniform_worst": control_worst,
            "adaptive_robust_helpfulness": robust_x,
            "adaptive_robust_harmlessness_active": 1.0 - robust_x,
            "adaptive_robust_worst": robust_worst,
            "robust_minus_uniform_worst": robust_worst - control_worst,
        })
    margins = [row["robust_minus_uniform_worst"] for row in rows]
    payload = {
        "status": "pass" if all(b > a for a, b in zip(margins, margins[1:])) else "fail",
        "scope": "analytic adaptive-adversary toy; no model, reward, validation, or fresh response was used",
        "assumptions": {
            "global_policy_parameter": "x=P(useful answer)",
            "helpfulness": "x on all rows",
            "harmlessness_training_signal": "(1-x) only on safety-active rows",
            "uniform_control": "equal observed objective average plus unit Bernoulli KL to x=0.5",
            "adaptive_robust_control": "maximise latent min(helpfulness, harmlessness) with the same symmetric reference",
            "static_builder_caveat": "this toy is not evidence that a one-shot static sigma target is adaptive",
        },
        "kl_weight": args.kl_weight,
        "rows": rows,
        "criterion": "robust-minus-uniform worst margin strictly increases with rho",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
