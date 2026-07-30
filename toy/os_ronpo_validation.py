"""Step 2a - decoy-game validation of OS-RONPO against the three RONPO
policy-reward estimators, on the exact matrix game used by
`toy/validate_theory.py` (robust LP value V* = 0.3289 at seed 0).

We isolate the POLICY-side estimator: the adversary side uses the exact
cost c_t = A^T pi_t for every arm, so any difference is due solely to how
r_t(y) = (A s_t)(y) is estimated. Arms:

  full-exp   : r = A @ s_t                       (exact expectation; reference)
  single-pair: (k,a) ~ s_t;  r = Ber(A[:,ka])    (existing stochastic, Cor. 1)
  OS-RONPO   : a_k ~ q_t(.|k); r = sum_k w_t(k) Ber(A[:,(k,a_k)])   (new)
  top-mass   : ka* = argmax_ka s_t;  r = A[:,ka*] (deterministic top-1 truncation)

Claims tested:
  (T1) unbiased arms (full-exp, single-pair, OS) drive Theta_t -> 0 at O(1/T);
       OS matches full-exp and reaches the robust Nash.
  (P3) OS has far smaller run-to-run variance than single-pair.
  (bias) top-mass optimizes a truncated reward: does its worst-objective floor
       reach V*, or does it stall / chase one objective? Reported honestly.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_theory import (
    build_decoy_game, payoff_matrix, mirror_prox_saddle, lse, kl_from_logs,
    robust_lp_value,
)


def omd_step(log_pi, log_s, r, c, log_mu, log_s0, eta, tau, kappa):
    nlp = eta * r + eta * tau * log_mu + (1.0 - eta * tau) * log_pi
    nlp -= lse(nlp)
    nls = -eta * c + eta * kappa * log_s0 + (1.0 - eta * kappa) * log_s
    nls -= lse(nls)
    return nlp, nls


def policy_reward_estimate(arm, A, log_s, n_obj, n_act, rng):
    """Return an estimate of r_t(y) = (A s_t)(y) under the named arm."""
    s = np.exp(log_s)
    if arm == "full-exp":
        return A @ s
    if arm == "single-pair":
        ka = rng.choice(A.shape[1], p=s)
        return (rng.random(A.shape[0]) < A[:, ka]).astype(float)
    if arm == "os-ronpo":
        S = s.reshape(n_obj, n_act)
        w = S.sum(axis=1)                       # omega_t(k)
        q = S / w[:, None]                      # q_t(a|k)
        r = np.zeros(A.shape[0])
        for k in range(n_obj):
            a_k = rng.choice(n_act, p=q[k])
            r += w[k] * (rng.random(A.shape[0]) < A[:, k * n_act + a_k]).astype(float)
        return r
    if arm == "top-mass":
        ka = int(np.argmax(s))                  # top-1 mass atom (deterministic)
        return A[:, ka]                          # truncated point-mass reward
    raise ValueError(arm)


def run_arm(arm, A, log_mu, log_s0, tau, kappa, T, lp_star, ls_star,
            n_obj, n_act, seed):
    rng = np.random.default_rng(seed)
    m = min(tau, kappa)
    t0 = int(np.ceil(2.0 * max(tau, kappa) / m)) + 1
    log_pi, log_s = log_mu.copy(), log_s0.copy()
    rec = sorted(set(np.unique(np.round(np.logspace(0, np.log10(T), 120)).astype(int))))
    ts, theta, floor = [], [], []
    ri = 0
    for t in range(T):
        eta = 2.0 / (m * (t + t0))
        r = policy_reward_estimate(arm, A, log_s, n_obj, n_act, rng)
        c = A.T @ np.exp(log_pi)                 # exact adversary cost (isolate policy est.)
        log_pi, log_s = omd_step(log_pi, log_s, r, c, log_mu, log_s0, eta, tau, kappa)
        if ri < len(rec) and (t + 1) == rec[ri]:
            ts.append(t + 1)
            theta.append(kl_from_logs(lp_star, log_pi) + kl_from_logs(ls_star, log_s))
            floor.append(float((A.T @ np.exp(log_pi)).min()))  # min_{k,a} c_pi(k,a)
            ri += 1
    return np.array(ts), np.array(theta), np.array(floor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=60000)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--out-dir", default="results/os_ronpo_toy_20260715")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    P = build_decoy_game(seed=0)
    A = payoff_matrix(P)
    n_obj, n_act = P.shape[0], P.shape[1]
    n, ka = A.shape
    log_mu = np.full(n, -np.log(n))
    log_s0 = np.full(ka, -np.log(ka))
    tau = kappa = 0.05
    vstar = robust_lp_value(A)
    lp_star, ls_star, its, res = mirror_prox_saddle(A, log_mu, log_s0, tau, kappa)
    print(f"Game: n_obj={n_obj} n_act={n_act} | robust LP value V*={vstar:.4f} "
          f"| saddle residual {res:.1e}")

    arms = ["full-exp", "single-pair", "os-ronpo", "top-mass"]
    det = {"full-exp", "top-mass"}                # deterministic arms: 1 seed suffices
    results = {}
    for arm in arms:
        ns = 1 if arm in det else args.seeds
        thetas, floors = [], []
        for sd in range(ns):
            ts, th, fl = run_arm(arm, A, log_mu, log_s0, tau, kappa, args.T,
                                 lp_star, ls_star, n_obj, n_act, seed=100 + sd)
            thetas.append(th); floors.append(fl)
        results[arm] = dict(ts=ts, theta=np.array(thetas), floor=np.array(floors))

    # ---- numeric summary ----
    print(f"\n{'arm':12s} {'Theta_final(mean)':>18s} {'Theta_final(std)':>17s} "
          f"{'floor_final':>12s}  (V*={vstar:.4f})")
    for arm in arms:
        R = results[arm]
        thf = R["theta"][:, -1]
        flf = R["floor"][:, -1]
        print(f"{arm:12s} {thf.mean():18.3e} {thf.std():17.3e} {flf.mean():12.4f}")

    # variance-reduction ratio OS vs single-pair (Prop 3, at matched t)
    sp = results["single-pair"]["theta"]
    osr = results["os-ronpo"]["theta"]
    # compare std across seeds at each recorded t (use median over t of std ratio)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.nanmedian(sp.std(0) / np.where(osr.std(0) > 0, osr.std(0), np.nan))
    print(f"\nProp 3 check: median std(single-pair)/std(OS) over t = {ratio:.2f}x "
          f"(>1 means OS lower-variance)")

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    col = {"full-exp": "#1f77b4", "single-pair": "#d62728",
           "os-ronpo": "#2ca02c", "top-mass": "#9467bd"}
    for arm in arms:
        R = results[arm]; ts = R["ts"]
        th = R["theta"]; mean = th.mean(0)
        ax1.loglog(ts, np.maximum(mean, 1e-16), color=col[arm], lw=1.7, label=arm)
        if th.shape[0] > 1:
            ax1.fill_between(ts, np.maximum(th.min(0), 1e-16), np.maximum(th.max(0), 1e-16),
                             color=col[arm], alpha=0.18, lw=0)
    t0 = int(np.ceil(2.0 * max(tau, kappa) / min(tau, kappa))) + 1
    G = 2.0 + max(tau, kappa); m = min(tau, kappa)
    theta0 = kl_from_logs(lp_star, log_mu) + kl_from_logs(ls_star, log_s0)
    M = max(t0 * theta0, 4.0 * G * G / (m * m))
    tref = results["os-ronpo"]["ts"].astype(float)
    ax1.loglog(tref, M / (tref + t0), "k--", lw=1.0, label=r"bound $M/(t{+}t_0)$")
    ax1.set_xlabel("iteration t"); ax1.set_ylabel(r"$\Theta_t=D(z^\star\Vert z_t)$")
    ax1.set_title("(a) Last-iterate convergence to robust Nash")
    ax1.legend(fontsize=7.5, frameon=False); ax1.grid(alpha=0.25, which="both")

    for arm in arms:
        R = results[arm]; ts = R["ts"]; fl = R["floor"].mean(0)
        ax2.semilogx(ts, fl, color=col[arm], lw=1.7, label=arm)
    ax2.axhline(vstar, color="gray", ls=":", lw=1.1, label=rf"$V^\star={vstar:.4f}$")
    ax2.set_xlabel("iteration t"); ax2.set_ylabel(r"worst-objective floor $\min_{k,a}c_{\pi_t}(k,a)$")
    ax2.set_title("(b) Worst-objective floor")
    ax2.legend(fontsize=7.5, frameon=False); ax2.grid(alpha=0.25, which="both")
    fig.tight_layout()
    out = os.path.join(args.out_dir, "os_ronpo_toy_validation.png")
    fig.savefig(out, dpi=150); fig.savefig(out.replace(".png", ".pdf"))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
