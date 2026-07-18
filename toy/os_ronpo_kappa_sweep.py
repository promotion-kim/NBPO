"""Step 2a follow-up - does annealing the adversary temperature kappa let the
OS-RONPO / full-expectation target (which both converge to the regularized
robust Nash) exceed top-mass on the HARD worst-objective floor
min_{k,a} c_{pi}(k,a)?

For each kappa (policy temperature tau fixed = 0.05, matching the paper's
soft-min panel):
  * floor_nash  = min_{k,a} c_{pi*_kappa}   -- exact regularized Nash via mirror-prox.
                  This is exactly what full-exp and OS-RONPO converge to.
  * floor_os    = OS-RONPO OMD run to (near) convergence  -- confirms it tracks floor_nash.
  * floor_top   = top-mass OMD, tail-averaged               -- the biased incumbent.
Decision: pick the smallest kappa where floor_nash clearly > floor_top AND OS
still converges (Theta small) within a practical iteration budget.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_theory import (build_decoy_game, payoff_matrix, mirror_prox_saddle,
                             kl_from_logs, robust_lp_value)
from os_ronpo_validation import omd_step, policy_reward_estimate


def run_arm_tail(arm, A, log_mu, log_s0, tau, kappa, T, lp_star, ls_star,
                 n_obj, n_act, seed, tail=0.2):
    """Run OMD; return (Theta_final, tail-averaged worst-floor)."""
    rng = np.random.default_rng(seed)
    m = min(tau, kappa)
    t0 = int(np.ceil(2.0 * max(tau, kappa) / m)) + 1
    log_pi, log_s = log_mu.copy(), log_s0.copy()
    tail_start = int((1.0 - tail) * T)
    floor_acc, floor_n = 0.0, 0
    for t in range(T):
        eta = 2.0 / (m * (t + t0))
        r = policy_reward_estimate(arm, A, log_s, n_obj, n_act, rng)
        c = A.T @ np.exp(log_pi)
        log_pi, log_s = omd_step(log_pi, log_s, r, c, log_mu, log_s0, eta, tau, kappa)
        if t >= tail_start:
            floor_acc += float((A.T @ np.exp(log_pi)).min()); floor_n += 1
    theta = kl_from_logs(lp_star, log_pi) + kl_from_logs(ls_star, log_s)
    return theta, floor_acc / max(floor_n, 1)


def main():
    P = build_decoy_game(seed=0)
    A = payoff_matrix(P)
    n_obj, n_act = P.shape[0], P.shape[1]
    n = A.shape[0]
    log_mu = np.full(n, -np.log(n))
    log_s0 = np.full(A.shape[1], -np.log(A.shape[1]))
    vstar = robust_lp_value(A)
    tau = 0.05
    kappas = [0.05, 0.03, 0.02, 0.01, 0.007, 0.005, 0.003, 0.002]

    print(f"V* (unregularized robust LP) = {vstar:.4f} | tau fixed = {tau}\n")
    print(f"{'kappa':>7} {'floor_nash(OS/full)':>19} {'floor_top':>11} "
          f"{'nash-top':>9} {'floor_os(OMD)':>13} {'Theta_os':>10} {'Theta_top':>10}")
    rows = []
    for kappa in kappas:
        lp_star, ls_star, _, res = mirror_prox_saddle(A, log_mu, log_s0, tau, kappa)
        floor_nash = float((A.T @ np.exp(lp_star)).min())
        T = int(min(300_000, 40_000 * (0.05 / kappa)))
        th_top, fl_top = run_arm_tail("top-mass", A, log_mu, log_s0, tau, kappa, T,
                                      lp_star, ls_star, n_obj, n_act, seed=0)
        # OS: average a few seeds
        os_th, os_fl = [], []
        for sd in range(5):
            t_, f_ = run_arm_tail("os-ronpo", A, log_mu, log_s0, tau, kappa, T,
                                  lp_star, ls_star, n_obj, n_act, seed=10 + sd)
            os_th.append(t_); os_fl.append(f_)
        th_os, fl_os = np.mean(os_th), np.mean(os_fl)
        rows.append((kappa, floor_nash, fl_top, fl_os, th_os, th_top, T))
        print(f"{kappa:>7.3f} {floor_nash:>19.4f} {fl_top:>11.4f} "
              f"{floor_nash-fl_top:>+9.4f} {fl_os:>13.4f} {th_os:>10.2e} {th_top:>10.2e}")

    # decision
    print("\nDecision scan (floor_nash - floor_top > 0 means OS/full-exp wins the hard floor):")
    winners = [r for r in rows if r[1] - r[2] > 0.002]
    if winners:
        best = max(winners, key=lambda r: r[1] - r[2])
        print(f"  -> OS/full-exp EXCEEDS top-mass for kappa in "
              f"{[f'{r[0]:.3f}' for r in winners]}; "
              f"largest margin at kappa={best[0]:.3f} "
              f"(nash {best[1]:.4f} vs top {best[2]:.4f}, +{best[1]-best[2]:.4f}).")
    else:
        print("  -> NO kappa gives OS/full-exp a clear hard-floor advantage over top-mass.")

    # figure
    ks = np.array([r[0] for r in rows])
    fn = np.array([r[1] for r in rows]); ft = np.array([r[2] for r in rows])
    fo = np.array([r[3] for r in rows])
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.semilogx(ks, fn, "o-", color="#1f77b4", lw=1.8, label="reg. Nash floor (OS / full-exp)")
    ax.semilogx(ks, fo, "^--", color="#2ca02c", lw=1.4, label="OS-RONPO OMD (tail avg)")
    ax.semilogx(ks, ft, "s-", color="#9467bd", lw=1.8, label="top-mass (tail avg)")
    ax.axhline(vstar, color="gray", ls=":", lw=1.1, label=rf"$V^\star={vstar:.4f}$")
    ax.set_xlabel(r"adversary temperature $\kappa$ (annealed; $\tau=0.05$ fixed)")
    ax.set_ylabel(r"hard worst-objective floor $\min_{k,a}c_{\pi}(k,a)$")
    ax.set_title("Does annealing $\\kappa$ let OS/full-exp beat top-mass?")
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25, which="both")
    ax.invert_xaxis()
    fig.tight_layout()
    out = "results/os_ronpo_toy_20260715/os_ronpo_kappa_sweep.png"
    fig.savefig(out, dpi=150); fig.savefig(out.replace(".png", ".pdf"))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
