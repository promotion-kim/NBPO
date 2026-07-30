"""Numerical validation of RONPO theory on the decoy matrix game.

Produces `figures/ronpo_lastiter_validation.png` with two panels:

  (a) Last-iterate convergence Theta_t = KL(pi*||pi_t) + KL(s*||s_t) on
      log-log axes for tau = kappa = 0.05:
        - exact OMD dynamics (deterministic r_t, c_t): converges below the
          Theorem-3 bound (faster, since z* is a fixed point of the exact map);
        - stochastic OMD with single-pair Bernoulli oracle feedback
          (Corollary 1 setting): matches the Theta(1/t) rate;
        - the explicit bound M/(t+t0) from Theorem 3 / Corollary 1.
  (b) Soft-min limit (Prop. 1): -kappa log E_{s0} exp(-c_{pi*}/kappa) and
      min_{k,a} c_{pi*}(k,a) both converge to the robust LP value V* as
      kappa -> 0.

Usage:
    python scripts/validate_theory.py --seed 0 --T 100000

Seed 0 reproduces the game instance with robust LP value V* = 0.3289
(the instance of Table `tab:toy`). The rate is theorem-guaranteed for
every instance, so the qualitative conclusion is seed-independent.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- game
def build_decoy_game(n_actions=8, n_obj=3, seed=0, bg_hi=0.56,
                     specialist=0.82, decoy_hi=0.78, decoy_lo=0.06,
                     decoy_idx=3):
    rng = np.random.default_rng(seed)
    P = np.empty((n_obj, n_actions, n_actions))
    for k in range(n_obj):
        M = np.full((n_actions, n_actions), 0.5)
        for i in range(n_actions):
            for j in range(i + 1, n_actions):
                p = bg_hi if rng.random() < 0.5 else 1.0 - bg_hi
                M[i, j], M[j, i] = p, 1.0 - p
        a_k = k
        for j in range(n_actions):
            if j != a_k:
                M[a_k, j], M[j, a_k] = specialist, 1.0 - specialist
        hi = decoy_hi if k < 2 else decoy_lo
        for j in range(n_actions):
            if j != decoy_idx:
                M[decoy_idx, j], M[j, decoy_idx] = hi, 1.0 - hi
        P[k] = M
    return P


def payoff_matrix(P):
    n_obj, n, _ = P.shape
    return np.transpose(P, (1, 0, 2)).reshape(n, n_obj * n)


def lse(v):
    m = v.max()
    return m + np.log(np.exp(v - m).sum())


# --------------------------------------------------- exact / stochastic OMD
def omd_step(log_pi, log_s, r, c, log_mu, log_s0, eta, tau, kappa):
    new_log_pi = eta * r + eta * tau * log_mu + (1.0 - eta * tau) * log_pi
    new_log_pi -= lse(new_log_pi)
    new_log_s = -eta * c + eta * kappa * log_s0 + (1.0 - eta * kappa) * log_s
    new_log_s -= lse(new_log_s)
    return new_log_pi, new_log_s


def mirror_prox_saddle(A, log_mu, log_s0, tau, kappa, gamma=0.05,
                       iters=400000, tol=1e-15):
    log_pi, log_s = log_mu.copy(), log_s0.copy()

    def F(lp, ls):
        pi, s = np.exp(lp), np.exp(ls)
        return (-(A @ s) + tau * (lp - log_mu + 1.0),
                (A.T @ pi) + kappa * (ls - log_s0 + 1.0))

    for it in range(iters):
        Fp, Fs = F(log_pi, log_s)
        lpw = log_pi - gamma * Fp; lpw -= lse(lpw)
        lsw = log_s - gamma * Fs;  lsw -= lse(lsw)
        Fpw, Fsw = F(lpw, lsw)
        nlp = log_pi - gamma * Fpw; nlp -= lse(nlp)
        nls = log_s - gamma * Fsw;  nls -= lse(nls)
        res = np.abs(nlp - log_pi).max() + np.abs(nls - log_s).max()
        log_pi, log_s = nlp, nls
        if res < tol:
            break
    return log_pi, log_s, it, res


def kl_from_logs(log_a, log_b):
    return max(float(np.sum(np.exp(log_a) * (log_a - log_b))), 1e-300)


def theta_trace(A, log_mu, log_s0, tau, kappa, T, lp_star, ls_star,
                stochastic=False, rng=None):
    m = min(tau, kappa)
    t0 = int(np.ceil(2.0 * max(tau, kappa) / m)) + 1
    log_pi, log_s = log_mu.copy(), log_s0.copy()
    n, ka = A.shape
    record = sorted(set(
        np.unique(np.round(np.logspace(0, np.log10(T), 160)).astype(int))))
    ts, thetas = [], []
    ridx = 0
    for t in range(T):
        eta = 2.0 / (m * (t + t0))
        if stochastic:
            pair = rng.choice(ka, p=np.exp(log_s))          # (k,a) ~ s_t
            r = (rng.random(n) < A[:, pair]).astype(float)  # z_y ~ Ber
            y = rng.choice(n, p=np.exp(log_pi))             # y ~ pi_t
            c = (rng.random(ka) < A[y, :]).astype(float)    # z_{k,a} ~ Ber
        else:
            r = A @ np.exp(log_s)
            c = A.T @ np.exp(log_pi)
        log_pi, log_s = omd_step(log_pi, log_s, r, c, log_mu, log_s0,
                                 eta, tau, kappa)
        if ridx < len(record) and (t + 1) == record[ridx]:
            ts.append(t + 1)
            thetas.append(kl_from_logs(lp_star, log_pi)
                          + kl_from_logs(ls_star, log_s))
            ridx += 1
    return np.array(ts), np.array(thetas), t0


def robust_lp_value(A):
    from scipy.optimize import linprog
    n, ka = A.shape
    cvec = np.zeros(n + 1); cvec[-1] = -1.0
    A_ub = np.hstack([-A.T, np.ones((ka, 1))])
    A_eq = np.zeros((1, n + 1)); A_eq[0, :n] = 1.0
    res = linprog(cvec, A_ub=A_ub, b_ub=np.zeros(ka), A_eq=A_eq,
                  b_eq=np.array([1.0]),
                  bounds=[(0, None)] * n + [(None, None)], method="highs")
    return float(res.x[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=100000)
    ap.add_argument("--n-stoch", type=int, default=8)
    ap.add_argument("--out", type=str,
                    default="figures/ronpo_lastiter_validation.png")
    args = ap.parse_args()

    P = build_decoy_game(seed=args.seed)
    A = payoff_matrix(P)
    n, ka = A.shape
    log_mu = np.full(n, -np.log(n))
    log_s0 = np.full(ka, -np.log(ka))
    vstar = robust_lp_value(A)
    print(f"Robust LP value V* = {vstar:.4f}")

    tau = kappa = 0.05
    m, G = min(tau, kappa), 2.0 + max(tau, kappa)
    lp_star, ls_star, its, res = mirror_prox_saddle(A, log_mu, log_s0,
                                                    tau, kappa)
    print(f"saddle solved in {its} MP iters (residual {res:.2e})")
    theta0 = (kl_from_logs(lp_star, log_mu) + kl_from_logs(ls_star, log_s0))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]

    # exact dynamics
    ts_e, th_e, t0 = theta_trace(A, log_mu, log_s0, tau, kappa, args.T,
                                 lp_star, ls_star, stochastic=False)
    ax.loglog(ts_e, np.maximum(th_e, 1e-16), color="#1f77b4", lw=1.6,
              label="exact OMD (Thm. 3)")

    # stochastic dynamics, averaged
    rng = np.random.default_rng(123)
    acc = None
    for i in range(args.n_stoch):
        ts_s, th_s, _ = theta_trace(A, log_mu, log_s0, tau, kappa, args.T,
                                    lp_star, ls_star, stochastic=True,
                                    rng=rng)
        acc = th_s if acc is None else acc + th_s
    th_s_mean = acc / args.n_stoch
    ax.loglog(ts_s, th_s_mean, color="#d62728", lw=1.6,
              label=f"stochastic OMD (Cor. 1), mean of {args.n_stoch} runs")

    # explicit bound M/(t+t0)
    M = max(t0 * theta0, 4.0 * G * G / (m * m))
    tref = np.array(ts_s, dtype=float)
    ax.loglog(tref, M / (tref + t0), "k--", lw=1.2,
              label=r"bound $M/(t+t_0)$")
    mask = ts_s >= args.T // 10
    slope = np.polyfit(np.log(ts_s[mask]), np.log(th_s_mean[mask]), 1)[0]
    print(f"Stochastic empirical slope (last decade): {slope:.3f} "
          "(theory: -1)")
    ax.set_xlabel(r"iteration $t$")
    ax.set_ylabel(r"$\Theta_t=D(z^\star\Vert z_t)$")
    ax.set_title(r"(a) Last-iterate rate, $\tau=\kappa=0.05$")
    ax.set_ylim(1e-13, 1e4)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax.grid(alpha=0.25, which="both")

    # panel (b): soft-min limit
    kappas = np.logspace(-3, 0.3, 22)
    soft_vals, hard_vals = [], []
    for kap in kappas:
        lps, lss, _, _ = mirror_prox_saddle(A, log_mu, log_s0, tau, kap)
        c = A.T @ np.exp(lps)
        soft_vals.append(-kap * lse(log_s0 - c / kap))
        hard_vals.append(c.min())
    ax = axes[1]
    ax.semilogx(kappas, soft_vals, "o-", ms=3, lw=1.4, color="#1f77b4",
                label=r"soft-min value (Eq. soft-min)")
    ax.semilogx(kappas, hard_vals, "s--", ms=3, lw=1.4, color="#d62728",
                label=r"$\min_{k,a}c_{\pi^\star_\kappa}(k,a)$")
    ax.axhline(vstar, color="gray", lw=1.0, ls=":",
               label=rf"robust LP value $V^\star={vstar:.4f}$")
    ax.set_xlabel(r"adversary temperature $\kappa$")
    ax.set_ylabel("inner value at saddle policy")
    ax.set_title("(b) Soft-min limit (Prop. 1)")
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()