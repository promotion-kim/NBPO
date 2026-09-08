#!/usr/bin/env python3
"""Algorithm 1's acceptance test: is every objective's surplus positive out of sample?

The regression gate asks whether the policy reproduces the Eq. (26) target. That
is a means, not the end. Algorithm 1 accepts a candidate on a different
condition -- that held-out empirical surpluses are positive for every objective --
and a policy could in principle move the game values in the right direction while
scoring poorly as a regression. The two are measured separately here so neither
stands in for the other.

No new generation or judging is needed. The candidate's log-probabilities over
the frozen response pool are already in the scored precompute, so its induced
distribution on that pool is a softmax over them, and the frozen payoff tensor
turns that distribution into game values directly.

The pool distribution is the candidate renormalized ON THE POOL, which is what
the finite-pool problem optimizes over -- not the candidate's distribution over
all strings. That restriction is the same one the solver made.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def game_value(pi_x, A_kx, mu, beta):
    """Soft-min game value: -beta * log sum_j mu_j exp(-(pi . A[:, j]) / beta)."""
    m = pi_x @ A_kx                      # (J,) expected centered payoff per comparator
    z = -m / beta
    zmax = z.max()
    return -beta * (zmax + np.log(np.sum(mu * np.exp(z - zmax))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", nargs="+", required=True, help="label=<scored dir>")
    ap.add_argument("--tensor-dir", type=Path, required=True)
    ap.add_argument("--solver-dir", type=Path, required=True)
    ap.add_argument("--heldout-split", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from datasets import load_from_disk

    meta = json.loads((args.tensor_dir / "meta.json").read_text())
    A = np.load(args.tensor_dir / "tensor_policy.npz")["A"]        # (K, X, I, J)
    A_ref = np.load(args.tensor_dir / "tensor_ref.npz")["A"]       # (K, X, J', J)
    sol = json.loads((args.solver_dir / "solution.json").read_text())
    beta = np.asarray(sol["config"]["beta"], float)
    K, X, I, J = A.shape
    mu = np.full(J, 1.0 / J)
    x_of = {p: i for i, p in enumerate(meta["prompt_ids"])}
    i_of = {r: i for i, r in enumerate(meta["policy_learner_ids"])}
    assign = json.loads(args.heldout_split.read_text())["assignment"]
    objectives = meta["objectives"]

    # the disagreement point: the reference pool played against itself
    mu_ref = np.full(A_ref.shape[2], 1.0 / A_ref.shape[2])
    d = np.array([[game_value(mu_ref, A_ref[k, x], mu, beta[k]) for x in range(X)]
                  for k in range(K)])                              # (K, X)

    report = {"objectives": objectives, "beta": beta.tolist(),
              "acceptance_rule": ("Algorithm 1 promotes a candidate only if the held-out "
                                  "empirical surplus is positive for EVERY objective"),
              "pool_note": ("the candidate is renormalized over the frozen response pool, "
                            "which is the set the finite-pool problem optimizes over"),
              "arms": {}}

    # pi_t itself, as the baseline every arm has to be read against: "negative
    # surplus" means nothing until you know what the starting policy scores.
    pi_t_logp = np.zeros((X, I))          # uniform on the pool, which is what pi_t was
    specs = [("pi_t_uniform_BASELINE", None)] + [tuple(sp.split("=", 1)) for sp in args.scored]
    for label, path in specs:
        if path is None:
            logp = pi_t_logp
        else:
            dd = load_from_disk(str(Path(path) / "precomputed"))["train"].to_dict()
            # recover per-(prompt, response) log-probabilities from the pair rows
            logp = np.full((X, I), np.nan)
            for pid, cid, rid, lc, lr in zip(
                    dd["prompt_id"], dd["chosen_response_id"], dd["rejected_response_id"],
                    dd["reference_chosen_logps"], dd["reference_rejected_logps"]):
                x = x_of[pid]
                logp[x, i_of[cid]] = lc
                logp[x, i_of[rid]] = lr
        halves = np.array([assign.get(p, "?") for p in meta["prompt_ids"]])

        arm = {}
        for name in ("validation", "test"):
            xs = [x for x in range(X) if halves[x] == name and np.isfinite(logp[x]).all()]
            if path is None:
                xs = [x for x in range(X) if halves[x] == name]
            if not xs:
                continue
            s = np.zeros((K, len(xs)))
            for c, x in enumerate(xs):
                z = logp[x] - logp[x].max()
                pi_x = np.exp(z); pi_x /= pi_x.sum()
                for k in range(K):
                    s[k, c] = game_value(pi_x, A[k, x], mu, beta[k]) - d[k, x]
            per_obj = {objectives[k]: {
                "mean_surplus": float(s[k].mean()),
                "fraction_prompts_positive": float((s[k] > 0).mean()),
            } for k in range(K)}
            arm[name] = {
                "n_prompts": len(xs),
                "per_objective": per_obj,
                "worst_objective_mean_surplus": float(min(
                    v["mean_surplus"] for v in per_obj.values())),
                "accepts": bool(all(v["mean_surplus"] > 0 for v in per_obj.values())),
            }
        report["arms"][label] = arm

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'arm':>18} {'half':>11} " +
          " ".join(f"{o[:12]:>13}" for o in objectives) + f" {'worst':>9} {'accepts':>8}")
    for label, halves in report["arms"].items():
        for name, a in halves.items():
            vals = " ".join(f"{a['per_objective'][o]['mean_surplus']:+13.5f}"
                            for o in objectives)
            print(f"{label:>18} {name:>11} {vals} "
                  f"{a['worst_objective_mean_surplus']:+9.5f} {str(a['accepts']):>8}")


if __name__ == "__main__":
    main()
