#!/usr/bin/env python3
"""Controlled nontransitivity stress test: does the adaptive game buy anything?

The whole NBPO argument is that a *game-valued* objective can represent pair
interactions a scalar reward cannot. On real judged pools that claim is hard to
test, because the amount of genuine intransitivity is unknown and small. Here it
is dialled directly.

For each objective and prompt, the centered payoff is

    Delta_k^(alpha) = clip( (1-alpha) * Delta_k^base + alpha * C_k, -1/2, 1/2 )

* ``Delta_k^base`` is **scalar-compatible by construction**: built from a latent
  utility ``u_k`` as ``g(u_k(i) - u_k(j))``, so a Bradley-Terry model can fit it
  exactly and every bargaining rule sees a transitive world at ``alpha = 0``;
* ``C_k`` is an independently seeded **cyclic circulation**: skew-symmetric with
  zero diagonal and, crucially, *no* consistent utility ordering, so it is pure
  pair interaction.

At ``alpha = 0`` a scalar reward model loses nothing. As ``alpha`` grows the
scalar representation is provably unable to fit the payoff, while the adaptive
game is. If the adaptive-minus-BT gap does not widen with ``alpha``, the
paper's central mechanism does not do what it claims -- so this is a test the
method can fail.

Nothing here is method-specific: one payoff tensor per (alpha, seed) is handed
unchanged to every method.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

from mnpo_scripts.nbpo_core import kl_divergence, uniform_policy
from mnpo_scripts.nbpo_generic import (
    KSUndefinedError, matched_weight_l1, solve_finite_pool,
)
from mnpo_scripts.nbpo_representations import (
    AdaptiveGameRepresentation, BTRewardRepresentation, FixedReferenceRepresentation,
)

ALPHAS = (0.00, 0.25, 0.50, 0.75, 1.00)


def skew(M):
    """Project onto the skew-symmetric subspace and zero the diagonal."""
    M = 0.5 * (M - np.swapaxes(M, -1, -2))
    idx = np.arange(M.shape[-1])
    M[..., idx, idx] = 0.0
    return M


def base_tensor(rng, K, X, I, spread=1.0):
    """Scalar-compatible payoff: a latent utility squashed through tanh.

    Exactly representable by a Bradley-Terry model, so alpha=0 is the case where
    a scalar reward gives up nothing.
    """
    u = rng.normal(0, spread, size=(K, X, I))
    d = u[..., :, None] - u[..., None, :]
    return skew(0.45 * np.tanh(d)), u


def circulation(rng, K, X, I):
    """Cyclic circulation: skew-symmetric, zero diagonal, no consistent ordering.

    Built from a random cyclic permutation so that i beats i+1 beats ... beats i,
    then randomly re-signed per prompt. Its Borda scores are ~0, so it carries no
    utility signal at all -- only pair interaction.
    """
    C = np.zeros((K, X, I, I))
    for k in range(K):
        for x in range(X):
            perm = rng.permutation(I)
            M = np.zeros((I, I))
            for a in range(I):
                for b in range(1, I // 2 + 1):
                    if I % 2 == 0 and b == I // 2 and a >= I // 2:
                        continue
                    M[perm[a], perm[(a + b) % I]] = 0.45
                    M[perm[(a + b) % I], perm[a]] = -0.45
            C[k, x] = M
    return skew(C)


def mix(base, C, alpha):
    return np.clip((1 - alpha) * base + alpha * C, -0.5, 0.5)


def bt_fit(P, iters=500):
    """Fit Bradley-Terry to P = 0.5 + Delta; return (deviance per edge, scores).

    The deviance is the paper-relevant number: it is how much of the payoff a
    SCALAR representation provably cannot explain.
    """
    I = P.shape[-1]
    w = np.ones(I)
    for _ in range(iters):
        new = np.zeros(I)
        for i in range(I):
            num = sum(P[i, j] for j in range(I) if j != i)
            den = sum(1.0 / (w[i] + w[j]) for j in range(I) if j != i)
            new[i] = num / den if den > 0 else w[i]
        new = np.clip(new, 1e-9, None)
        w = new / new.mean()
    dev, n = 0.0, 0
    for i, j in itertools.combinations(range(I), 2):
        q = w[i] / (w[i] + w[j])
        for o, p in ((P[i, j], q), (1 - P[i, j], 1 - q)):
            if o > 0 and p > 0:
                dev += 2 * o * math.log(o / p)
        n += 1
    return dev / max(1, n), np.log(w)


def exploitability(A, pi):
    """Mean over objectives of how much a best-responding opponent can gain.

    ``max_z E_{y~pi} A_k(y,z)`` minus what the policy itself achieves. Zero at a
    Nash equilibrium of the symmetric game, and it does not depend on any
    method's own representation -- which is what makes it usable as a common
    yardstick.
    """
    vals = []
    for k in range(A.shape[0]):
        r = np.einsum("xi,xij->xj", pi, A[k])
        vals.append(float(np.mean(r.max(axis=-1) - (r * pi).sum(axis=-1))))
    return float(np.mean(vals))


def evaluate_common(A_np, pi, beta):
    """Score ANY method's policy on the one true payoff, under one evaluator.

    Comparing each method's own reported surplus is meaningless across
    representations: a game surplus lives in centered-preference units
    (|A| <= 1/2) while a BT-reward surplus lives in log-odds units, so the two
    are not on the same scale and the difference between them is an artifact of
    units. Every policy is therefore re-scored here against the SAME
    adaptive-game payoff, which is the ground truth the tensor was built from.
    """
    K, X, I, _ = A_np.shape
    A = torch.from_numpy(A_np)
    mu = uniform_policy(X, I)
    rep = AdaptiveGameRepresentation(A, A, mu,
                                     torch.full((K,), beta, dtype=torch.float64))
    s = rep.surplus(pi if torch.is_tensor(pi) else torch.from_numpy(pi))
    sn = s.numpy()
    return {
        "true_min_surplus": float(sn.min()),
        "true_avg_surplus": float(sn.mean()),
        "true_nash_welfare": (float(np.sum(np.log(sn))) if (sn > 0).all() else None),
        "true_nash_welfare_defined": bool((sn > 0).all()),
        "true_exploitability": exploitability(A_np, pi.numpy() if torch.is_tensor(pi) else pi),
    }


def solve_all(A_np, beta, eta, M, R, common_l1=None):
    """Every matched method on one payoff tensor.

    Two conditions are produced, because they answer different questions and
    conflating them is misleading:

    ``native``       each Nash variant keeps its own RAW lambda, which is the
                     deployed configuration. But raw lambda is 1/s, so as
                     surpluses approach zero NBPO's weight norm explodes -- on a
                     pure-cycle payoff it reaches 268 against fixed-reference's
                     57 -- and the comparison silently becomes one of step size.
    ``matched_step`` every method is solved at one common ||w||_1, so what
                     differs is only the DIRECTION of the weight vector. This is
                     the condition that isolates the representation.
    """
    K, X, I, _ = A_np.shape
    A = torch.from_numpy(A_np)
    mu = uniform_policy(X, I)
    reps = {
        "adaptive_game": AdaptiveGameRepresentation(
            A, A, mu, torch.full((K,), beta, dtype=torch.float64)),
        "fixed_reference": FixedReferenceRepresentation(A, A, mu),
    }
    # BT reward: the best scalar summary of the SAME payoff, fitted per prompt.
    r = np.zeros((K, X, I))
    for k in range(K):
        for x in range(X):
            _, s = bt_fit(0.5 + A_np[k, x])
            r[k, x] = s
    reps["bt_reward"] = BTRewardRepresentation(
        torch.from_numpy(r), torch.from_numpy(r), mu)

    out = {}
    nash = {}
    for name, rep in reps.items():
        nash[name] = solve_finite_pool(rep, "nash", eta=eta, M=M, R=R, gamma=0.5)
    l1_game = matched_weight_l1(nash["adaptive_game"])
    l1_bt = matched_weight_l1(nash["bt_reward"])

    out["NBPO"] = nash["adaptive_game"]
    out["Fixed-reference Nash"] = nash["fixed_reference"]
    out["BT-RM-Nash"] = nash["bt_reward"]
    out["Game-utilitarian"] = solve_finite_pool(
        reps["adaptive_game"], "utilitarian", eta=eta, R=R, weight_l1=l1_game)
    out["BT-RM-utilitarian"] = solve_finite_pool(
        reps["bt_reward"], "utilitarian", eta=eta, R=R, weight_l1=l1_bt)
    try:
        out["Game-KS"] = solve_finite_pool(
            reps["adaptive_game"], "kalai_smorodinsky", eta=eta, R=R,
            weight_l1=l1_game, ks_kwargs=dict(stage1_iters=300, ideal_iters=40))
    except KSUndefinedError as exc:
        out["Game-KS"] = exc

    # --- matched-step condition -------------------------------------------
    # A common weight norm for every representation, so only the direction of
    # the weight vector differs. Fixed-reference's norm is used as the anchor
    # because it is the one that does not blow up as surpluses approach zero.
    L = common_l1 if common_l1 is not None else matched_weight_l1(nash["fixed_reference"])
    matched = {}
    for name, rep_key, agg in (("NBPO", "adaptive_game", "nash"),
                               ("Fixed-reference Nash", "fixed_reference", "nash"),
                               ("BT-RM-Nash", "bt_reward", "nash"),
                               ("Game-utilitarian", "adaptive_game", "utilitarian"),
                               ("BT-RM-utilitarian", "bt_reward", "utilitarian")):
        rep = reps[rep_key]
        if agg == "nash":
            # keep the Nash DIRECTION, rescale to the common norm
            w = nash[rep_key].weights
            w = w * (L / float(w.sum()))
            from mnpo_scripts.nbpo_generic import solve_proximal
            sol = solve_proximal(rep, uniform_policy(X, I), w, eta, R)
            matched[name] = (sol.pi, w)
        else:
            sol = solve_finite_pool(rep, "utilitarian", eta=eta, R=R, weight_l1=L)
            matched[name] = (sol.pi, sol.weights)
    return out, matched, L


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--objectives", type=int, default=4)
    ap.add_argument("--prompts", type=int, default=40)
    ap.add_argument("--responses", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--dual-iterations", type=int, default=1500)
    ap.add_argument("--fixed-point-iterations", type=int, default=3)
    args = ap.parse_args()

    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(20260908 + seed)
        base, _ = base_tensor(rng, args.objectives, args.prompts, args.responses)
        C = circulation(rng, args.objectives, args.prompts, args.responses)
        for alpha in ALPHAS:
            A = mix(base, C, alpha)
            # structural guarantees, asserted rather than assumed
            assert np.abs(A + np.swapaxes(A, -1, -2)).max() < 1e-12, "not skew-symmetric"
            idx = np.arange(args.responses)
            assert np.abs(A[..., idx, idx]).max() == 0.0, "nonzero diagonal"
            P = 0.5 + A
            assert P.min() >= 0.0 and P.max() <= 1.0, "invalid probabilities"

            devs = [bt_fit(0.5 + A[k, x])[0]
                    for k in range(args.objectives) for x in range(args.prompts)]
            sols, matched, common_l1 = solve_all(
                A, args.beta, args.eta, args.dual_iterations,
                args.fixed_point_iterations)
            target = sols["NBPO"]
            t_ref = (torch.log(target.pi) - torch.log(target.pi_t)).numpy()

            # the uniform reference, scored on the same yardstick, as a floor
            base_eval = evaluate_common(A, uniform_policy(args.prompts,
                                                          args.responses), args.beta)
            for name, s in sols.items():
                if isinstance(s, Exception):
                    rows.append({"seed": seed, "alpha": alpha, "method": name,
                                 "status": "undefined", "reason": str(s)[:200]})
                    continue
                own = s.surplus.numpy()
                t = (torch.log(s.pi) - torch.log(s.pi_t)).numpy()
                rows.append({
                    "seed": seed, "alpha": alpha, "method": name, "status": "ok",
                    # common yardstick: every policy scored on the true payoff
                    **evaluate_common(A, s.pi, args.beta),
                    "reference_min_surplus": base_eval["true_min_surplus"],
                    # the method's OWN reported surplus, kept but not comparable
                    # across representations (different units)
                    "own_min_surplus": float(own.min()),
                    "own_avg_surplus": float(own.mean()),
                    "bt_deviance_per_edge": float(np.mean(devs)),
                    "distance_to_nbpo_target": float(np.abs(t - t_ref).mean()),
                    "proximal_kl": s.proximal_kl,
                    "weight_l1": float(s.weights.sum()),
                    "fixed_point_residual": s.fixed_point_residual,
                    "extra_map_residual": s.extra_map_residual,
                    "target_identity_residual": s.target_log_ratio_check(),
                    "condition": "native",
                })
            for name, (pi_m, w_m) in matched.items():
                rows.append({
                    "seed": seed, "alpha": alpha, "method": name, "status": "ok",
                    "condition": "matched_step",
                    **evaluate_common(A, pi_m, args.beta),
                    "reference_min_surplus": base_eval["true_min_surplus"],
                    "weight_l1": float(w_m.sum()),
                    "common_weight_l1": common_l1,
                    "bt_deviance_per_edge": float(np.mean(devs)),
                })
        print(f"seed {seed} done", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "controlled_nontransitivity_raw.json").write_text(
        json.dumps({"config": vars(args) | {"out_dir": str(args.out_dir)},
                    "alphas": list(ALPHAS), "rows": rows}, indent=2, default=str))

    # aggregate: the gaps that carry the paper's claim
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        if r.get("status") == "ok":
            agg[(r["condition"], r["alpha"], r["method"])].append(r)
    lines = ["condition,alpha,method,n_seeds,true_min_surplus_mean,true_min_surplus_std,"
             "true_avg_surplus_mean,true_nash_welfare_mean,true_exploitability_mean,"
             "bt_deviance_mean,distance_to_nbpo_target_mean,reference_min_surplus_mean"]
    for (cond, alpha, method), rs in sorted(agg.items()):
        f = lambda k: [x[k] for x in rs if x.get(k) is not None]
        m = lambda v: (sum(v) / len(v)) if v else ""
        sd = lambda v: (statistics_stdev(v) if len(v) > 1 else 0.0) if v else ""
        lines.append(",".join(map(str, [
            cond, alpha, method, len(rs), m(f("true_min_surplus")), sd(f("true_min_surplus")),
            m(f("true_avg_surplus")), m(f("true_nash_welfare")),
            m(f("true_exploitability")), m(f("bt_deviance_per_edge")),
            m(f("distance_to_nbpo_target")), m(f("reference_min_surplus"))])))
    (args.out_dir / "controlled_nontransitivity.csv").write_text("\n".join(lines) + "\n")

    mean = lambda rs, k: sum(x[k] for x in rs) / len(rs)
    for cond in ("native", "matched_step"):
        print(f"\n=== {cond} === (all policies scored on the SAME true payoff)")
        print("alpha  BTdev   reference     NBPO   FixedRef  BT-RM-Nash  "
              "adaptive-fixed  adaptive-BT   NBPO ||w||")
        for alpha in ALPHAS:
            g = agg.get((cond, alpha, "NBPO"), [])
            if not g:
                continue
            gm = mean(g, "true_min_surplus")
            cells = {}
            for nm in ("Fixed-reference Nash", "BT-RM-Nash"):
                rs = agg.get((cond, alpha, nm), [])
                cells[nm] = mean(rs, "true_min_surplus") if rs else float("nan")
            print(f"{alpha:5.2f}  {mean(g,'bt_deviance_per_edge'):6.4f}  "
                  f"{mean(g,'reference_min_surplus'):9.5f}  {gm:8.5f}  "
                  f"{cells['Fixed-reference Nash']:9.5f}  {cells['BT-RM-Nash']:10.5f}  "
                  f"{gm - cells['Fixed-reference Nash']:+13.5f}  "
                  f"{gm - cells['BT-RM-Nash']:+11.5f}  {mean(g,'weight_l1'):10.2f}")


def statistics_stdev(v):
    import statistics
    return statistics.stdev(v)


if __name__ == "__main__":
    main()
