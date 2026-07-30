#!/usr/bin/env python3
"""Controlled preference-game demonstration of Anchored BPO (no GPU, no LLM).

A policy is a distribution pi over N candidate responses; response i has a known
anchored-surplus vector s_i in R^K (surplus vs the reference mu, one per judge).
The feasible surplus set is the convex hull {sum_i pi_i s_i : pi in simplex}, the
exact analogue of "mix the policy's own samples". On this set we evaluate the four
selection rules and the two claims that separate bargaining from its corners:

  IR  : the utilitarian rule (averaging) can push a judge's surplus below 0;
        NBS and KS keep every judge strictly positive.
  COV : degrade judge k by label noise (s_k -> lambda_k s_k). NBS and KS select
        the SAME policy (both are scale-invariant per coordinate), so the realized
        clean surplus is unchanged; utilitarian and egalitarian (maxmin) drift.

Outputs the two result tables (with LaTeX rows) and a K=2 feasible-set figure.
Pure numpy for the solver; matplotlib only for the figure.
"""
from __future__ import annotations
import numpy as np


def simplex_grid(n: int, res: int = 200):
    """All lattice points of the (n-1)-simplex with denominator res (rows sum to 1)."""
    def rec(k, rem):
        if k == 1:
            yield (rem,)
            return
        for i in range(rem + 1):
            for tail in rec(k - 1, rem - i):
                yield (i,) + tail
    return np.array(list(rec(n, res)), dtype=float) / res


def solve(S: np.ndarray, rule: str, eps=1e-9):
    """S: (N,K) surplus vectors. Grid-search the simplex (exact for small N).
    Returns the argmax mixture pi and its realized surplus vector s = pi @ S."""
    res = {3: 300, 4: 90}.get(S.shape[0], 45)   # keep the lattice ~<200k points
    P = simplex_grid(S.shape[0], res)
    Sm = P @ S
    if rule == "uniform":                       # utilitarian: max sum_k s_k
        obj = Sm.sum(1)
    elif rule == "maxmin":                      # egalitarian: max min_k s_k
        obj = Sm.min(1)
    elif rule == "nbs":                         # Nash: max sum_k log s_k, s>0
        pos = np.all(Sm > 0, axis=1)
        obj = np.where(pos, np.sum(np.log(np.where(pos[:, None], Sm, 1.0)), axis=1), -np.inf)
    elif rule == "ks":                          # Kalai-Smorodinsky: max min_k s_k/u*_k
        obj = (Sm / np.maximum(S.max(0), eps)).min(1)
    else:
        raise ValueError(rule)
    j = int(np.argmax(obj))
    return P[j], Sm[j]


RULES = ["uniform", "maxmin", "nbs", "ks"]
PRETTY = {"uniform": "Utilitarian (avg)", "maxmin": "Egalitarian (maxmin)",
          "nbs": "Nash (NBS)", "ks": "Kalai--Smorodinsky"}


def table(S, title):
    K = S.shape[1]
    print(f"\n=== {title}  (K={K}, N={S.shape[0]}) ===")
    print(f"{'rule':22} " + " ".join(f"s{k+1:>7}" for k in range(K)) +
          f" {'worst':>8} {'IR':>6}")
    res = {}
    for r in RULES:
        pi, s = solve(S, r)
        res[r] = (pi, s)
        print(f"{PRETTY[r]:22} " + " ".join(f"{v:>8.3f}" for v in s) +
              f" {s.min():>8.3f} {('yes' if np.all(s>0) else 'NO'):>6}")
    return res


def latex_rows(S, res):
    for r in RULES:
        s = res[r][1]
        cells = " & ".join(f"{v:.3f}" for v in s)
        ir = r"\checkmark" if np.all(s > 0) else r"\times"
        print(f"  {PRETTY[r]} & {cells} & {s.min():.3f} & ${ir}$ \\\\")


def figure_k2(path):
    """K=2 bargaining diagram: feasible surplus set, ideal point $u^\\ast$, the
    segment from $d$ to $u^\\ast$ on which KS lies, and the four solutions. The
    utilitarian vertex leaves the positive orthant (judge 2 below the reference)
    while the three bargaining solutions stay strictly inside and apart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # A game whose max-sum vertex sacrifices judge 2 below the reference; the
    # bargaining solutions stay interior and separate from one another.
    S = np.array([[0.80, -0.10], [0.55, 0.12], [0.38, 0.30], [0.12, 0.38], [-0.08, 0.40]])
    F = simplex_grid(S.shape[0], 40) @ S
    ustar = S.max(0)                                    # ideal (utopia) point
    fig, ax = plt.subplots(figsize=(4.7, 4.0))
    ax.scatter(F[:, 0], F[:, 1], s=3, color="0.87", zorder=1, rasterized=True,
               label="feasible set")
    # positive-orthant axes; the region below the dotted line is $s_2<0$
    ax.axhline(0, color="0.55", lw=0.9, ls=":")
    ax.axvline(0, color="0.55", lw=0.9, ls=":")
    # segment d -> u*, the locus of the Kalai--Smorodinsky solution
    ax.plot([0, ustar[0]], [0, ustar[1]], color="0.4", lw=1.1, ls="--", zorder=3)
    ax.plot(ustar[0], ustar[1], marker="*", color="0.15", ms=13, zorder=5)
    ax.annotate(r"ideal $u^\ast$", (ustar[0], ustar[1]), textcoords="offset points",
                xytext=(-4, 7), fontsize=8, ha="right")
    # disagreement point (the reference), kept clear of any label
    ax.plot(0, 0, "ks", ms=7, zorder=6)
    ax.annotate(r"disagreement $d$", (0, 0), textcoords="offset points",
                xytext=(9, 4), fontsize=8)
    sols = {
        "uniform": ("Utilitarian (avg)", "D", "#d62728", (-8, 2), "right"),
        "maxmin":  ("Egalitarian", "v", "#2ca02c", (-10, -12), "right"),
        "nbs":     ("Nash (NBPO)", "o", "#1f77b4", (5, 9), "left"),
        "ks":      ("Kalai--Smorodinsky", "^", "#9467bd", (9, -5), "left"),
    }
    for r, (lab, m, c, off, ha) in sols.items():
        _, s = solve(S, r)
        ax.plot(s[0], s[1], m, color=c, ms=10, mec="k", mew=0.6, zorder=7)
        ax.annotate(lab, (s[0], s[1]), textcoords="offset points", xytext=off,
                    fontsize=8, color=c, ha=ha, fontweight="bold")
        print(f"  fig {r:8} -> s=({s[0]:+.3f}, {s[1]:+.3f})")
    ax.set_xlim(-0.19, 0.96); ax.set_ylim(-0.21, 0.49)
    ax.set_xlabel("surplus to judge 1, $s_1$")
    ax.set_ylabel("surplus to judge 2, $s_2$")
    ax.set_title("Feasible surplus set and the four solutions", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"[figure] wrote {path}")


def main():
    # ---- Claim 1: individual rationality. Judge 3 is "expensive": only response C
    # serves it, and C is costly for judges 1,2. Averaging abandons judge 3. ----
    S_ir = np.array([
        [0.35, 0.30, -0.25],   # A: great for 1,2; sacrifices 3
        [0.30, 0.35, -0.25],   # B: great for 1,2; sacrifices 3
        [-0.12, -0.12, 0.45],  # C: the only response that serves judge 3
    ])
    res_ir = table(S_ir, "Claim 1 (IR): averaging drops the expensive judge")
    print("LaTeX:")
    latex_rows(S_ir, res_ir)

    # ---- Claim 2: covariance. Clean game where every rule is IR-valid; then
    # degrade judge 3 (lambda_3=0.3). Report each rule's realized CLEAN s_3. ----
    S_cov = np.array([
        [0.20, 0.20, 0.40],    # A: clean-best sum, driven by judge 3
        [0.33, 0.33, 0.10],    # B: runner-up, light on judge 3
        [0.45, 0.08, 0.14],    # C: strong judge 1
        [0.08, 0.45, 0.14],    # D: strong judge 2
    ])
    lam = np.array([1.0, 1.0, 0.3])
    base = {r: solve(S_cov, r) for r in RULES}
    noisy = {r: solve(S_cov * lam, r) for r in RULES}
    print("\n=== Claim 2 (covariance): degrade judge 3, lambda_3=0.3 ===")
    print(f"{'rule':22} {'s3 (clean policy)':>18} {'s3 (noisy policy)':>18} {'drift':>8}")
    for r in RULES:
        s3c = float(base[r][0] @ S_cov[:, 2])
        s3n = float(noisy[r][0] @ S_cov[:, 2])   # noise-selected policy, judged clean
        print(f"{PRETTY[r]:22} {s3c:>18.3f} {s3n:>18.3f} {s3n - s3c:>+8.3f}")
    print("LaTeX:")
    for r in RULES:
        s3c = float(base[r][0] @ S_cov[:, 2]); s3n = float(noisy[r][0] @ S_cov[:, 2])
        print(f"  {PRETTY[r]} & {s3c:.3f} & {s3n:.3f} & ${s3n - s3c:+.3f}$ \\\\")

    figure_k2("ronpo_aaai/figures/synthetic_bargain.pdf")


if __name__ == "__main__":
    main()
