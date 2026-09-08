#!/usr/bin/env python3
"""The attainable floor on the Section-7 regression gate, computed before training.

``h_t`` is a function of ``(prompt, learner pair)`` alone. The sampled Eq. (24)
target additionally depends on the drawn opponent ``j ~ nu*_k`` and on two
Bernoulli flips, and no policy can see either. So the best predictor available
to ANY policy is the conditional mean

    m(x, i1, i2) = sum_k lambda_k * ( A[k,x,i1,:] - A[k,x,i2,:] ) . nu[k,x]

and the smallest attainable ``MSE / Var(target)`` is ``1 - Var(m)/Var(target)``.

This is a property of the target construction, not of the fit. Computing it
first is the difference between "the policy failed the gate" and "the gate was
unreachable", which are different papers.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tensor-dir", type=Path, required=True)
    ap.add_argument("--solver-dir", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, default=None,
                    help="optional realized pairs file; its empirical variance is "
                         "compared against the analytic one as a consistency check")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sol = json.loads((args.solver_dir / "solution.json").read_text())
    lam = np.asarray(sol["lambda_raw"], float)                 # (K,)
    A = np.load(args.tensor_dir / "tensor_policy.npz")["A"]    # (K, X, I, J) centered
    nu = np.load(args.solver_dir / "nu_update.npz")["nu"]      # (K, X, J)
    K, X, I, J = A.shape

    P = A + 0.5                                                # P_k(y_i > z_j)
    # conditional mean and conditional variance of ONE objective's z_k
    Ej = np.einsum("kxij,kxj->kxi", P, nu)                     # E_j P(i beats z)
    Ej2 = np.einsum("kxij,kxj->kxi", P * P, nu)                # E_j P^2

    pairs = list(itertools.combinations(range(I), 2))
    m = np.zeros((X, len(pairs)))
    v = np.zeros((X, len(pairs)))
    for c, (i1, i2) in enumerate(pairs):
        # z_k = b1 - b2, with b1,b2 | j independent Bernoulli(P[..,i1,j]),(P[..,i2,j])
        # E[z_k] = E_j[p1 - p2]
        mk = Ej[:, :, i1] - Ej[:, :, i2]                       # (K, X)
        # E[z_k^2] = E_j[ p1(1-p1) + p2(1-p2) + (p1-p2)^2 ]
        ez2 = (Ej[:, :, i1] - Ej2[:, :, i1]) + (Ej[:, :, i2] - Ej2[:, :, i2]) \
            + np.einsum("kxj,kxj->kx", (P[:, :, i1, :] - P[:, :, i2, :]) ** 2, nu)
        vk = ez2 - mk ** 2
        m[:, c] = lam @ mk                                     # objectives are independent draws
        v[:, c] = (lam ** 2) @ vk

    target_var = float(m.var() + v.mean())                     # law of total variance
    predictable = float(m.var())
    floor = 1.0 - predictable / target_var

    out = {
        "n_prompts": int(X), "n_learner_responses": int(I),
        "n_comparators": int(J), "n_pairs_per_prompt": len(pairs),
        "lambda_raw": lam.tolist(),
        "conditional_mean_rms": float(np.sqrt((m ** 2).mean())),
        "conditional_mean_variance": predictable,
        "mean_conditional_variance": float(v.mean()),
        "total_target_variance_analytic": target_var,
        "target_rms_analytic": float(np.sqrt((m ** 2).mean() + v.mean())),
        "max_attainable_r2": predictable / target_var,
        "normalized_mse_var_floor": floor,
        "gate_threshold": 0.90,
        "gate_reachable": bool(floor < 0.90),
        "interpretation": (
            "normalized_mse_var cannot go below this floor for ANY policy, because "
            "the sampled target carries opponent-draw and Bernoulli noise that h_t "
            "is not a function of. A floor above the threshold means the gate is "
            "unreachable by construction, not that the policy failed."),
    }
    if args.pairs:
        z = np.asarray([json.loads(l)["nbpo_weighted_z"] for l in args.pairs.open()], float)
        out["empirical"] = {"n_rows": int(z.size), "variance": float(z.var()),
                            "rms": float(np.sqrt((z ** 2).mean())),
                            "fraction_zero": float((z == 0).mean()),
                            "analytic_vs_empirical_variance_ratio":
                                float(target_var / z.var()) if z.var() else None}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
