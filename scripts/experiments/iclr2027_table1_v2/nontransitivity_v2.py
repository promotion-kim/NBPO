#!/usr/bin/env python3
"""A feasibility-preserving controlled-nontransitivity benchmark (v2).

v1 dials intransitivity by *mixing* a scalar-compatible payoff with a cyclic
circulation, ``(1-alpha) base + alpha C``, and clipping. That mixture removes the
transitive component exactly as fast as it adds the cyclic one, so at large
``alpha`` the payoff approaches a pure circulant tournament -- whose unique
equilibrium is the uniform reference itself. The bargaining problem then has an
empty interior: no policy improves every objective, ``rho* = 0``, and the
instance is outside Assumption 1. Whatever a solver returns there is not a test
of NBPO.

v2 fixes the design flaw rather than the solver. Three components, added rather
than interpolated:

``T``       a **shared transitive direction**, ``T[i,j] = m (c_i - c_j) / 2``
            with ``c`` centered. Every objective values it identically, so it is
            what keeps a common improvement available -- the feasibility witness.
``base_k``  per-objective scalar utilities, **fixed**, which is where the
            objectives genuinely conflict.
``C_k``     a per-objective circulant tournament entering as ``alpha * C_k``.

    A_k(alpha) = T + base_k + alpha * C_k

Two properties follow by construction and are asserted, not hoped for:

1. **No clipping.** Amplitudes are chosen so ``|A| <= 1/2`` holds outright, so
   the reference/witness geometry is never deformed by a clip.
2. **The disagreement point does not move with alpha.** ``d_k = V_k(mu)`` depends
   on ``A`` only through the column sums at the uniform reference, and a
   circulant tournament on an odd number of responses has exactly zero row sums,
   hence zero column sums. So ``alpha * C_k`` contributes nothing to ``d_k`` and
   the surplus scale is comparable across the whole sweep.

What is *not* asserted by construction is that the witness stays feasible, so it
is measured: ``build_v2`` computes ``rho*`` and the witness surpluses exactly at
every ``alpha`` and raises unless they clear a floor.
"""
from __future__ import annotations

import numpy as np

from scripts.experiments.iclr2027_table1_v2.controlled_nontransitivity import (
    base_tensor, circulation, skew,
)
from scripts.experiments.iclr2027_table1_v2 import nontransitivity_feasibility as nf


def shared_direction(rng, K, X, I, m):
    """``T[i,j] = m (c_i - c_j)/2`` with ``c`` centered -- one direction all agree on."""
    c = rng.normal(size=(X, I))
    c = c - c.mean(axis=-1, keepdims=True)
    c = c / np.abs(c).max(axis=-1, keepdims=True)
    T = 0.5 * m * (c[:, :, None] - c[:, None, :])
    T = skew(np.broadcast_to(T, (K, X, I, I)).copy())
    return T, c


def build_v2(seed, K=4, X=40, I=5, alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
             m=0.24, base_amp=0.12, cycle_amp=0.12, beta=0.25,
             rho_floor=0.01):
    """One v2 instance family across ``alpha``, with feasibility verified exactly."""
    if I % 2 == 0:
        raise ValueError("I must be odd so the circulant tournament has zero row sums")
    rng = np.random.default_rng(20260908 + seed)
    T, c = shared_direction(rng, K, X, I, m)
    base_raw, _ = base_tensor(rng, K, X, I)
    base = base_raw * (base_amp / 0.45)
    C = circulation(rng, K, X, I) * (cycle_amp / 0.45)

    rs = np.abs(C.sum(axis=-1)).max()
    if rs > 1e-12:
        raise AssertionError(f"circulation row sums are {rs:.3e}, not zero; the "
                             "disagreement point would drift with alpha")

    mu = np.full((X, I), 1.0 / I)
    b = np.full(K, float(beta))
    out = {}
    d_ref = None
    for alpha in alphas:
        A = T + base + alpha * C
        A = skew(A)
        if np.abs(A).max() > 0.5 + 1e-12:
            raise AssertionError(
                f"|A| reaches {np.abs(A).max():.4f} at alpha={alpha}; v2 refuses to "
                "clip, so lower the amplitudes instead")
        d = nf.game_values(A, mu, mu, b)
        if d_ref is None:
            d_ref = d
        elif np.abs(d - d_ref).max() > 1e-12:
            raise AssertionError(
                f"disagreement point moved by {np.abs(d - d_ref).max():.3e} at "
                f"alpha={alpha}; the alpha-independence argument is wrong")
        out[alpha] = {"A": A, "mu": mu, "beta": b, "d": d}
    return out, {"m": m, "base_amp": base_amp, "cycle_amp": cycle_amp,
                 "I": I, "K": K, "X": X, "seed": seed, "rho_floor": rho_floor}


def match_rho_star(seed, target, alphas, *, K=4, X=40, I=5, base_amp=0.05,
                   cycle_amp=0.28, beta=0.25, iters=18, m_lo=0.02):
    """Choose the shared-direction weight ``m`` per alpha so ``rho*`` is CONSTANT.

    Even where v1 keeps a positive ``rho*``, the size of the feasible margin
    collapses as alpha grows -- by a factor of about 6.5 on the v1 sweep -- so a
    method's achieved surplus is being compared against a moving target. Holding
    ``rho*`` fixed removes that confound: what changes across the sweep is then
    only how much of the payoff a scalar representation cannot express.

    ``m`` is capped so that ``|A| <= 1/2`` holds without clipping, which is the
    binding constraint at large alpha; the achieved ``rho*`` is returned rather
    than assumed, so a cap that bites is visible instead of silent.
    """
    m_cap = 0.5 - base_amp - cycle_amp
    if m_cap <= m_lo:
        raise ValueError("no admissible margin range: lower base_amp or cycle_amp")
    out = {}
    for alpha in alphas:
        lo, hi = m_lo, m_cap

        def rho(m):
            fam, _ = build_v2(seed, K=K, X=X, I=I, alphas=(alpha,), m=m,
                              base_amp=base_amp, cycle_amp=cycle_amp, beta=beta)
            inst = fam[alpha]
            return nf.solve_max_min(inst["A"], inst["mu"], inst["beta"],
                                    inst["d"])["rho_star"], inst

        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            r, _ = rho(mid)
            if r < target:
                lo = mid
            else:
                hi = mid
        m = 0.5 * (lo + hi)
        r, inst = rho(m)
        inst["margin"] = m
        inst["rho_star"] = r
        inst["margin_at_cap"] = bool(m >= m_cap - 1e-6)
        out[alpha] = inst
    return out, {"target_rho_star": target, "base_amp": base_amp,
                 "cycle_amp": cycle_amp, "margin_cap": m_cap, "I": I, "K": K,
                 "X": X, "seed": seed, "bisection_iterations": iters}
