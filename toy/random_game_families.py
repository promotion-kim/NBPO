"""Random-game-family validation of RONPO's convergence theory (CPU only).

Addresses the "one hand-designed matrix / eight runs" critique:
  (a) exact-OMD bound ratio Theta_t / (M/(t+t0)) across N random games;
  (b) exact OMD vs mirror-prox (EGPO-style extragradient) on the lifted game,
      x-axis = exact-gradient evaluations (MP costs 2 per iteration);
  (c) stochastic Bernoulli OMD, mean over seeds, bands across games;
  (d) robust-value gaps V*_LP - V_min(pi_saddle) for the lifted (RONPO) vs
      averaged-oracle saddle, on uniform tournaments and randomized decoy games;
  plus a one-at-a-time sensitivity table over |Y|, K, pool coverage,
  tau=kappa, and payoff noise.

Deterministic (master seed 0). Outputs JSON, a 2x2 PDF figure for the paper,
and a LaTeX sensitivity-table fragment.

Usage: python toy/random_game_families.py [--games 20 --T 5000]
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_theory import (build_decoy_game, payoff_matrix, lse, omd_step,
                             kl_from_logs, robust_lp_value)

BLUE, RED, GRAY = "#1f77b4", "#d62728", "#7f7f7f"


# ------------------------------------------------------------ game families
def build_uniform_game(n, K, rng, noise=0.0):
    """Random tournaments: P_k(i,j) ~ U(0.05, 0.95), skew-symmetric."""
    P = np.full((K, n, n), 0.5)
    iu = np.triu_indices(n, 1)
    for k in range(K):
        p = rng.uniform(0.05, 0.95, size=len(iu[0]))
        if noise > 0:
            p = np.clip(p + rng.normal(0, noise, size=p.shape), 0.01, 0.99)
        P[k][iu] = p
        P[k].T[iu] = 1.0 - p
    return P


def build_random_decoy_game(n, K, rng):
    """The decoy construction of Appendix tab:toy with randomized parameters."""
    return build_decoy_game(
        n_actions=n, n_obj=K, seed=int(rng.integers(2**31)),
        bg_hi=rng.uniform(0.52, 0.62), specialist=rng.uniform(0.75, 0.90),
        decoy_hi=rng.uniform(0.70, 0.85), decoy_lo=rng.uniform(0.02, 0.15),
        decoy_idx=int(rng.integers(K, n)))


def lifted_payoff(P, cov, rng):
    """A_{y,(k,a)} for a random opponent pool of size ceil(cov*n)."""
    K, n, _ = P.shape
    pool = np.sort(rng.choice(n, size=max(2, round(cov * n)), replace=False))
    sub = P[:, :, pool]                       # (K, n, |pool|)
    return np.transpose(sub, (1, 0, 2)).reshape(n, K * len(pool))


# ------------------------------------------------------------ solvers
def mirror_prox(A, log_mu, log_s0, tau, kappa, gamma, iters, tol=0.0,
                lp_star=None, ls_star=None, record=None):
    """Extragradient (mirror-prox) on the lifted game; optionally trace Theta."""
    log_pi, log_s = log_mu.copy(), log_s0.copy()

    def F(lp, ls):
        return (-(A @ np.exp(ls)) + tau * (lp - log_mu + 1.0),
                (A.T @ np.exp(lp)) + kappa * (ls - log_s0 + 1.0))

    ts, th = [], []
    for it in range(iters):
        Fp, Fs = F(log_pi, log_s)
        lpw = log_pi - gamma * Fp; lpw -= lse(lpw)
        lsw = log_s - gamma * Fs;  lsw -= lse(lsw)
        Fpw, Fsw = F(lpw, lsw)
        nlp = log_pi - gamma * Fpw; nlp -= lse(nlp)
        nls = log_s - gamma * Fsw;  nls -= lse(nls)
        res = np.abs(nlp - log_pi).max() + np.abs(nls - log_s).max()
        log_pi, log_s = nlp, nls
        if record is not None and (it + 1) in record:
            ts.append(2 * (it + 1))  # gradient evaluations
            th.append(kl_from_logs(lp_star, log_pi)
                      + kl_from_logs(ls_star, log_s))
        if tol and res < tol:
            break
    return log_pi, log_s, np.array(ts), np.array(th)


def omd_trace(A, log_mu, log_s0, tau, kappa, T, lp_star, ls_star, record,
              rng=None, seeds=0):
    """Exact (seeds=0) or Bernoulli-stochastic (mean over seeds) OMD trace."""
    m = min(tau, kappa)
    t0 = int(np.ceil(2.0 * max(tau, kappa) / m)) + 1
    n, ka = A.shape
    runs = max(1, seeds)
    acc = np.zeros(len(record))
    for _ in range(runs):
        log_pi, log_s = log_mu.copy(), log_s0.copy()
        th, ridx = [], 0
        for t in range(T):
            eta = 2.0 / (m * (t + t0))
            if seeds:
                pair = rng.choice(ka, p=np.exp(log_s))
                r = (rng.random(n) < A[:, pair]).astype(float)
                y = rng.choice(n, p=np.exp(log_pi))
                c = (rng.random(ka) < A[y, :]).astype(float)
            else:
                r, c = A @ np.exp(log_s), A.T @ np.exp(log_pi)
            log_pi, log_s = omd_step(log_pi, log_s, r, c, log_mu, log_s0,
                                     eta, tau, kappa)
            if ridx < len(record) and (t + 1) == record[ridx]:
                th.append(kl_from_logs(lp_star, log_pi)
                          + kl_from_logs(ls_star, log_s))
                ridx += 1
        acc += np.array(th)
    return acc / runs, t0


def run_config(n, K, cov, tk, noise, games, T, master, stoch_seeds=3):
    """One sensitivity cell: per-game bound ratios, MP compare, stoch slope."""
    tau = kappa = tk
    m, G = tk, 2.0 + tk
    record = sorted(set(
        np.unique(np.round(np.logspace(0, np.log10(T), 60)).astype(int))))
    rec_mp = set(np.unique(np.round(
        np.logspace(0, np.log10(T // 2), 60)).astype(int)))
    out = dict(ratio_curves=[], exact_curves=[], mp_curves=[], stoch_curves=[],
               final_ratio=[], bound_ok=[], slope=[], record=record)
    for g in range(games):
        rng = np.random.default_rng([master, g])
        A = lifted_payoff(build_uniform_game(n, K, rng, noise), cov, rng)
        na, ka = A.shape
        log_mu = np.full(na, -np.log(na))
        log_s0 = np.full(ka, -np.log(ka))
        lp_st, ls_st, _, _ = mirror_prox(A, log_mu, log_s0, tau, kappa,
                                         gamma=0.1, iters=100000, tol=1e-13)
        theta0 = kl_from_logs(lp_st, log_mu) + kl_from_logs(ls_st, log_s0)
        th_e, t0 = omd_trace(A, log_mu, log_s0, tau, kappa, T, lp_st, ls_st,
                             record)
        M = max(t0 * theta0, 4.0 * G * G / (m * m))
        bound = M / (np.array(record, float) + t0)
        ratio = np.maximum(th_e, 1e-300) / bound
        out["ratio_curves"].append(ratio.tolist())
        out["exact_curves"].append(np.maximum(th_e, 1e-300).tolist())
        out["final_ratio"].append(float(ratio[-1]))
        out["bound_ok"].append(bool((th_e <= bound).all()))
        _, _, ts_mp, th_mp = mirror_prox(A, log_mu, log_s0, tau, kappa,
                                         gamma=0.25, iters=T // 2,
                                         lp_star=lp_st, ls_star=ls_st,
                                         record=rec_mp)
        out["mp_evals"], out["mp_curves"] = ts_mp.tolist(), \
            out["mp_curves"] + [np.maximum(th_mp, 1e-300).tolist()]
        th_s, _ = omd_trace(A, log_mu, log_s0, tau, kappa, T, lp_st, ls_st,
                            record, rng=rng, seeds=stoch_seeds)
        out["stoch_curves"].append(th_s.tolist())
        rec = np.array(record)
        mask = rec >= T // 10
        out["slope"].append(float(np.polyfit(
            np.log(rec[mask]), np.log(th_s[mask]), 1)[0]))
    return out


def value_gaps(family, games, master, n=8, K=3, tau=0.05, kappa=0.05):
    """V*_LP - V_min for the lifted (RONPO) vs averaged-oracle saddle policy."""
    gaps_r, gaps_a = [], []
    for g in range(games):
        rng = np.random.default_rng([master, 7, g])
        P = (build_uniform_game(n, K, rng) if family == "uniform"
             else build_random_decoy_game(n, K, rng))
        A = payoff_matrix(P)
        log_mu = np.full(n, -np.log(n))
        log_s0 = np.full(A.shape[1], -np.log(A.shape[1]))
        vstar = robust_lp_value(A)
        lp_r, _, _, _ = mirror_prox(A, log_mu, log_s0, tau, kappa,
                                    gamma=0.1, iters=100000, tol=1e-12)
        gaps_r.append(vstar - float((A.T @ np.exp(lp_r)).min()))
        A_avg = P.mean(axis=0)
        lp_a, _, _, _ = mirror_prox(A_avg, log_mu, log_mu, tau, tau,
                                    gamma=0.1, iters=100000, tol=1e-12)
        gaps_a.append(vstar - float((A.T @ np.exp(lp_a)).min()))
    return np.array(gaps_r), np.array(gaps_a)


def band(ax, x, curves, color, label):
    q = np.quantile(np.array(curves), [0.0, 0.5, 1.0], axis=0)
    ax.loglog(x, q[1], color=color, lw=1.6, label=label)
    ax.fill_between(x, np.maximum(q[0], 1e-16), q[2], color=color, alpha=0.18,
                    lw=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--T", type=int, default=5000)
    ap.add_argument("--outdir", type=str, default="ronpo_aaai")
    args = ap.parse_args()
    base = dict(n=8, K=3, cov=1.0, tk=0.05, noise=0.0)
    axes_grid = [("$|\\mathcal{Y}|$", "n", [4, 16, 32]),
                 ("$K$", "K", [2, 5, 8]),
                 ("coverage", "cov", [0.25, 0.5, 0.75]),
                 ("$\\tau=\\kappa$", "tk", [0.01, 0.2, 0.5]),
                 ("noise $\\sigma$", "noise", [0.05, 0.1, 0.2])]

    print("base config ...")
    res = {"base": run_config(games=args.games, T=args.T, master=0,
                              stoch_seeds=8, **base)}
    for _, key, vals in axes_grid:
        for v in vals:
            cfg = dict(base); cfg[key] = v
            tag = f"{key}={v}"
            print(tag, "...")
            res[tag] = run_config(games=args.games, T=args.T, master=0, **cfg)

    print("value gaps ...")
    gaps = {fam: value_gaps(fam, args.games, master=0)
            for fam in ("uniform", "decoy")}

    # ---------------------------------------------------------------- figure
    b = res["base"]
    rec = np.array(b["record"], float)
    fig, ax = plt.subplots(2, 2, figsize=(9.4, 6.6))

    a = ax[0, 0]
    band(a, rec, b["ratio_curves"], BLUE,
         f"median / range over {args.games} games")
    a.axhline(1.0, color="k", ls="--", lw=1.2, label="Theorem 2 bound")
    a.set_xlabel("iteration $t$")
    a.set_ylabel(r"$\Theta_t \,/\, (M/(t+t_0))$")
    a.set_title("(a) Exact OMD vs. Theorem bound, random games")
    a.legend(fontsize=7.5, frameon=False, loc="lower left")

    a = ax[0, 1]
    band(a, rec, b["exact_curves"], BLUE, "OMD (Thm. 2 schedule)")
    mp = np.maximum(np.array(b["mp_curves"]), 1e-16)
    qmp = np.quantile(mp, [0.0, 0.5, 1.0], axis=0)
    evals = np.array(b["mp_evals"], float)
    a.loglog(evals, qmp[1], color=RED, lw=1.6,
             label="mirror-prox (extragradient)")
    a.fill_between(evals, np.maximum(qmp[0], 1e-16), qmp[2], color=RED,
                   alpha=0.18, lw=0)
    a.set_xlabel("exact-gradient evaluations")
    a.set_ylabel(r"$\Theta$")
    a.set_title("(b) OMD vs. extragradient on the lifted game")
    a.legend(fontsize=7.5, frameon=False, loc="lower left")

    a = ax[1, 0]
    band(a, rec, b["stoch_curves"], RED,
         f"stochastic OMD, mean of 8 seeds")
    a.loglog(rec, b["stoch_curves"][0][0] * rec[0] / rec, color=GRAY, ls=":",
             lw=1.2, label=r"$\Theta(1/t)$ guide")
    med_slope = float(np.median(b["slope"]))
    a.set_xlabel("iteration $t$")
    a.set_ylabel(r"$\Theta_t$")
    a.set_title(f"(c) Bernoulli feedback (median slope {med_slope:.2f})")
    a.legend(fontsize=7.5, frameon=False, loc="lower left")

    a = ax[1, 1]
    data = [gaps["uniform"][0], gaps["uniform"][1],
            gaps["decoy"][0], gaps["decoy"][1]]
    bp = a.boxplot(data, positions=[1, 2, 3.5, 4.5], widths=0.7,
                   patch_artist=True, medianprops=dict(color="k"))
    for patch, col in zip(bp["boxes"], [BLUE, RED, BLUE, RED]):
        patch.set_facecolor(col); patch.set_alpha(0.45)
    a.set_xticks([1.5, 4], ["uniform tournaments", "randomized decoy"])
    a.set_ylabel(r"$V^\star_{\mathrm{LP}} - V_{\min}(\pi)$")
    a.set_title("(d) Robust-value gap of the saddle policy")
    a.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=BLUE, alpha=0.45),
                      plt.Rectangle((0, 0), 1, 1, fc=RED, alpha=0.45)],
             labels=["lifted (RONPO)", "averaged oracle"],
             fontsize=7.5, frameon=False, loc="upper left")
    for s in ax.ravel():
        s.grid(alpha=0.25, which="both")
    fig.tight_layout()
    figpath = os.path.join(args.outdir, "figures", "ronpo_random_families.pdf")
    fig.savefig(figpath)
    print("saved", figpath)

    # ------------------------------------------------------- sensitivity table
    def row(tag, label, val):
        r = res[tag]
        fr = np.array(r["final_ratio"]); sl = np.array(r["slope"])
        ok = sum(r["bound_ok"])
        m, e = f"{np.median(fr):.1e}".split("e")
        return (f"{label} & {val} & ${m}\\times10^{{{int(e)}}}$ & "
                f"{ok}/{args.games} & ${np.median(sl):.2f}$ \\\\")

    lines = [row("base", "base", "--")]
    for label, key, vals in axes_grid:
        for v in vals:
            lines.append(row(f"{key}={v}", label, v))
    frag = os.path.join(args.outdir, "random_family_sensitivity.tex")
    header = ("% AUTO-GENERATED by toy/random_game_families.py; do not "
              "hand-edit\n\\begin{tabular}{llccc}\n\\toprule\nAxis & Value & "
              "Ratio$_{T}$ & Bound & Slope \\\\\n\\midrule\n")
    with open(frag, "w") as f:
        f.write(header + "\n".join(lines)
                + "\n\\bottomrule\n\\end{tabular}\n")
    print("saved", frag)

    # ----------------------------------------------------------------- JSON
    summary = {
        "games_per_config": args.games, "T": args.T,
        "bound_ok_total": int(sum(sum(r["bound_ok"]) for r in res.values())),
        "configs": len(res),
        "base_median_final_ratio": float(np.median(res["base"]["final_ratio"])),
        "base_median_slope": float(np.median(res["base"]["slope"])),
        "gap_uniform_ronpo_median": float(np.median(gaps["uniform"][0])),
        "gap_uniform_avg_median": float(np.median(gaps["uniform"][1])),
        "gap_decoy_ronpo_median": float(np.median(gaps["decoy"][0])),
        "gap_decoy_avg_median": float(np.median(gaps["decoy"][1])),
    }
    rngb = np.random.default_rng(0)
    for fam in ("uniform", "decoy"):
        d = gaps[fam][1] - gaps[fam][0]
        boots = np.median(rngb.choice(d, (10000, len(d))), axis=1)
        summary[f"gap_diff_{fam}_median"] = float(np.median(d))
        summary[f"gap_diff_{fam}_ci95"] = [float(np.quantile(boots, .025)),
                                           float(np.quantile(boots, .975))]
    os.makedirs("toy/random_family_outputs", exist_ok=True)
    with open("toy/random_family_outputs/results.json", "w") as f:
        json.dump({"summary": summary,
                   "sensitivity": {k: {kk: vv for kk, vv in v.items()
                                       if kk in ("final_ratio", "bound_ok",
                                                 "slope")}
                                   for k, v in res.items()},
                   "gaps": {k: [v[0].tolist(), v[1].tolist()]
                            for k, v in gaps.items()}}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
