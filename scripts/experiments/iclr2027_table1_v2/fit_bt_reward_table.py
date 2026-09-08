#!/usr/bin/env python3
"""Fit the scalar Bradley-Terry reward table the BT-RM control needs.

The BT-RM row is a **scalar reward model on the same judged pairs**, so it needs
``r[k, x, i]``, one number per response, not a pairwise matrix. This fits that
from the frozen BT ensemble's own pairwise probabilities.

The fit is over **all 16 responses jointly** -- eight learner and eight
comparator -- because the observed comparison graph is the bipartite
learner-by-comparator block plus the comparator-by-comparator tournament. That
graph is connected, so the scores are identified up to the usual additive
constant, which is fixed by centring on the comparator pool. Fitting the learner
pool alone would not be identified: no learner is ever compared to another
learner.

Minimizing the Bradley-Terry cross-entropy against the ensemble's probabilities
is exactly "the best scalar summary of what this preference model says", which is
what the matched control is supposed to be.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def fit_prompt(P_lc, P_cc, iters=400, lr=0.5, tol=1e-12):
    """Scores for [learners, comparators] from the two observed blocks.

    Gradient descent on the BT cross-entropy. Convex in the scores, so a plain
    descent with a fixed step is adequate and deterministic.
    """
    I, J = P_lc.shape
    n = I + J
    s = np.zeros(n)
    li, ci = np.arange(I), np.arange(I, n)
    prev = None
    for _ in range(iters):
        d_lc = s[li][:, None] - s[ci][None, :]
        q_lc = 1.0 / (1.0 + np.exp(-d_lc))
        d_cc = s[ci][:, None] - s[ci][None, :]
        q_cc = 1.0 / (1.0 + np.exp(-d_cc))
        g = np.zeros(n)
        e_lc = q_lc - P_lc
        g[li] += e_lc.sum(axis=1)
        g[ci] -= e_lc.sum(axis=0)
        e_cc = q_cc - P_cc
        np.fill_diagonal(e_cc, 0.0)
        g[ci] += e_cc.sum(axis=1) - e_cc.sum(axis=0)
        s -= lr * g / max(1, n)
        s -= s[ci].mean()                     # centre on the comparator pool
        loss = float(np.abs(e_lc).mean() + np.abs(e_cc).mean())
        if prev is not None and abs(prev - loss) < tol:
            break
        prev = loss
    return s[li], s[ci], prev


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probs-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pool-size", type=int, default=8)
    args = ap.parse_args()

    n = args.pool_size
    pol = np.load(args.probs_dir / "probs_bt_policy.npz")["p"].astype(np.float64)
    ref = np.load(args.probs_dir / "probs_bt_ref.npz")["p"].astype(np.float64)
    pol = pol[:, :, :, :n, :n].mean(axis=0)
    ref = ref[:, :, :, :n, :n].mean(axis=0)
    K, X = pol.shape[0], pol.shape[1]

    r_l = np.zeros((K, X, n))
    r_c = np.zeros((K, X, n))
    resid = []
    for k in range(K):
        for x in range(X):
            a, b, loss = fit_prompt(pol[k, x], ref[k, x])
            r_l[k, x], r_c[k, x] = a, b
            resid.append(loss)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "bt_reward_table.npz",
                        r_learner=r_l, r_reference=r_c)
    (args.out_dir / "bt_reward_meta.json").write_text(json.dumps({
        "source": str(args.probs_dir),
        "pool_size": n, "shape": list(r_l.shape),
        "fit": ("Bradley-Terry cross-entropy against the frozen BT ensemble's own "
                "pairwise probabilities, over all 2n responses jointly"),
        "identifiability": ("the observed graph is the bipartite learner-by-"
                            "comparator block plus the comparator tournament, which "
                            "is connected; scores are centred on the comparator pool"),
        "mean_abs_probability_residual": float(np.mean(resid)),
        "max_abs_probability_residual": float(np.max(resid)),
        "reward_scale": {"learner_sd": float(r_l.std()),
                         "reference_sd": float(r_c.std())},
    }, indent=2))
    print(f"r_learner {r_l.shape}, r_reference {r_c.shape} -> {args.out_dir}")
    print(f"  mean |P - sigma(dr)| residual {np.mean(resid):.4f} "
          f"(max {np.max(resid):.4f})")
    print(f"  reward sd: learner {r_l.std():.4f}, reference {r_c.std():.4f}")


if __name__ == "__main__":
    main()
