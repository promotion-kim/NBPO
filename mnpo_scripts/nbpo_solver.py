"""Finite-pool fixed-point and projected dual solver for NBPO (Algorithm 1).

Implements, on a frozen finite response pool:

- the fixed-point approximation of the weighted proximal solve
  ``pi_t^*(lambda)`` (Eq. (18) ``eq:weighted-policy``) by alternating
  regularized-opponent reweighting (Eq. (7)) and the exponential update
  (Eq. (21) ``eq:fixed-opponent-update``), always centered at the proximal
  center ``pi_t``;
- projected dual gradient descent on the raw multipliers,
  ``lambda <- clamp(lambda - gamma_m (s_hat - 1/lambda), lambda_min, lambda_max)``
  (Eq. (27) ``eq:dual-update``), whose gradient is Eq. (19)
  ``eq:dual-gradient`` and whose KKT point satisfies ``lambda_k = 1/s_k``
  (Eq. (20) ``eq:kkt-weights``);
- matched finite-game controls (utilitarian, absolute max-min, surplus
  max-min) sharing the same payoff tensors, ``mu``, ``beta``, ``d``, response
  pool, and optimizer budget.

Scope note: the ``M`` = 4e3--3e5 dual iterations of the manuscript run on this
frozen finite pool -- each iteration is a cheap tensor computation. Fitting a
neural policy to the resulting targets is the *subsequent realization step*
(``scripts/nbpo/build_nbpo_pairs.py`` + ``loss_type: nbpo``); no 8B model is
retrained inside the dual loop.

Non-negotiables honored here: lambda is never normalized to sum to one in the
NBPO path; nonpositive surpluses are never replaced by ``max(s, eps)``; the
logged lambda is raw.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import torch

from mnpo_scripts.nbpo_core import (
    as_float64,
    compute_disagreement_point,
    compute_margins,
    compute_objective_gradient,
    compute_regularized_game_value,
    compute_regularized_opponent,
    kl_divergence,
    opponent_entropy,
    opponent_ess,
    uniform_policy,
    validate_distribution,
    validate_payoff_tensor,
    weighted_policy_update,
)

AGGREGATIONS = ("nash", "utilitarian", "absolute_maxmin", "surplus_maxmin")


def resolve_gamma_schedule(gamma: Union[float, Sequence[float]], M: int) -> torch.Tensor:
    """Dual step sizes ``gamma_{0:M-1}``: a scalar is broadcast, a sequence must have length M."""
    g = as_float64(gamma)
    if g.dim() == 0:
        g = g.expand(M).clone()
    if g.shape != (M,):
        raise ValueError(f"gamma must be a scalar or length-{M} schedule, got {tuple(g.shape)}")
    if not torch.isfinite(g).all() or (g <= 0).any():
        raise ValueError("gamma must be finite and strictly positive")
    return g


@dataclass
class WeightedPolicySolution:
    """Result of the fixed-point weighted proximal solve at fixed lambda."""

    pi: torch.Tensor  # (X, I)
    nu: torch.Tensor  # (K, X, J) -- the opponent that produced the final pi
    q: torch.Tensor  # (K, X, I)
    fixed_point_residual: float  # max |pi_next - pi_r| at the last iteration
    iterations: int


def solve_weighted_policy(
    A: torch.Tensor,
    mu: torch.Tensor,
    pi_t: torch.Tensor,
    lam: torch.Tensor,
    beta: torch.Tensor,
    eta: float,
    R: int,
    *,
    pi_init: Optional[torch.Tensor] = None,
    damping: float = 0.0,
) -> WeightedPolicySolution:
    """Fixed-point solve of Eq. (18) at fixed ``lam`` (Section 5.2 of the manuscript).

    Each round builds the adaptive opponent from the current iterate ``pi_r``
    (Eq. (7)), forms ``q`` (Eq. (9)), and applies the exponential update
    Eq. (21) -- which is ALWAYS centered at the proximal center ``pi_t``, never
    at ``pi_r``; the iterate enters only through the opponent.

    ``R`` is the fixed-point budget. ``R = 1`` is allowed and is the
    manuscript's disclosed practical approximation (one opponent reweighting
    per dual update); it must be reported as an approximation, which the
    returned ``fixed_point_residual`` quantifies. ``damping`` in [0, 1) mixes
    the previous iterate back in for stability at large ``eta * lam``.
    """
    if R < 1:
        raise ValueError("R must be at least 1")
    if not (0.0 <= damping < 1.0):
        raise ValueError("damping must lie in [0, 1)")
    pi_t = validate_distribution(pi_t, "pi_t", require_full_support=True)
    pi_r = pi_t if pi_init is None else validate_distribution(pi_init, "pi_init")
    residual = float("inf")
    nu = q = None
    for _ in range(R):
        margins = compute_margins(A, pi_r)
        nu = compute_regularized_opponent(margins, mu, beta)
        q = compute_objective_gradient(A, nu)
        pi_next = weighted_policy_update(pi_t, q, lam, eta)
        if damping > 0.0:
            pi_next = (1.0 - damping) * pi_next + damping * pi_r
        residual = (pi_next - pi_r).abs().max().item()
        pi_r = pi_next
    return WeightedPolicySolution(
        pi=pi_r, nu=nu, q=q, fixed_point_residual=residual, iterations=R
    )


def policy_game_values(
    A: torch.Tensor, mu: torch.Tensor, pi: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """``V_{k,beta_k}(pi)`` (Eq. (8)) for a policy on the finite pool: shape (K,)."""
    return compute_regularized_game_value(compute_margins(A, pi), mu, beta, form="softmin")


@dataclass
class DualSolveResult:
    """Full output of a finite-pool solve (Nash dual or a matched control)."""

    aggregation: str
    lam: torch.Tensor  # (K,) RAW dual weights (Nash) or fixed/adversarial weights
    V: torch.Tensor  # (K,) game values of the final policy
    d: torch.Tensor  # (K,) disagreement point V_{k,beta}(mu)
    surplus: torch.Tensor  # (K,) s = V - d (raw; may be nonpositive)
    pi: torch.Tensor  # (X, I) final finite-pool policy
    nu: torch.Tensor  # (K, X, J) final opponents
    q: torch.Tensor  # (K, X, I)
    kkt_residual: Optional[float]  # ||s - 1/lam||_inf (Nash only)
    control_residual: Optional[float]  # duality gap of the max-min controls
    fixed_point_residual: float
    opponent_entropy: torch.Tensor  # (K,)
    opponent_ess: torch.Tensor  # (K,)
    history: List[dict] = field(default_factory=list)
    config: dict = field(default_factory=dict)


def _log_row(m, lam, V, d, s, nu, fp_residual, extra=None):
    row = {
        "iteration": int(m),
        "lambda_raw": [float(v) for v in lam],
        "V": [float(v) for v in V],
        "d": [float(v) for v in d],
        "surplus": [float(v) for v in s],
        "min_surplus": float(s.min()),
        "kkt_residual": float((s - 1.0 / lam).abs().max()),
        "opponent_entropy": [float(v) for v in opponent_entropy(nu)],
        "opponent_ess": [float(v) for v in opponent_ess(nu)],
        "fixed_point_residual": float(fp_residual),
    }
    if extra:
        row.update(extra)
    return row


def solve_nbpo_dual(
    A_policy: torch.Tensor,
    A_ref: torch.Tensor,
    mu: torch.Tensor,
    beta: torch.Tensor,
    eta: float,
    gamma: Union[float, Sequence[float]],
    M: int,
    R: int,
    lambda_box: Sequence[float] = (1e-3, 1e3),
    *,
    lambda_init: Optional[torch.Tensor] = None,
    pi_t: Optional[torch.Tensor] = None,
    aggregation: str = "nash",
    warm_start_policy: bool = True,
    damping: float = 0.0,
    adversary_step: float = 1.0,
    log_every: int = 0,
) -> DualSolveResult:
    """Projected dual descent (Eq. (27)) or a matched control, on the frozen pool.

    Nash path (``aggregation="nash"``): ``M`` iterations of
    ``lambda <- Pi_Lambda[lambda - gamma_m (s_hat(pi_t^*(lambda)) - 1/lambda)]``
    with the box ``Lambda = [lambda_min, lambda_max]^K``. ``lambda`` stays RAW
    throughout -- no normalization -- and nonpositive surpluses are used as-is.
    ``lambda_init`` supports warm-starting from the previous outer stage; it
    defaults to ones (t = 0, per Algorithm 1).

    Matched controls share (A, mu, beta, d, pool, eta, M, R):
    - ``utilitarian``: fixed uniform weights (lambda = 1), no dual variable;
    - ``absolute_maxmin`` / ``surplus_maxmin``: adversarial two-player solve --
      the policy takes the same exponential step against the adversary's
      weights, the adversary runs multiplicative weights toward the worst
      objective (on V or on s respectively); ``control_residual`` reports the
      final duality gap ``<w, v> - min_k v_k``. This is NOT a static one-hot on
      the pre-training worst objective.
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
    A_policy = validate_payoff_tensor(A_policy, "A_policy")
    A_ref = validate_payoff_tensor(A_ref, "A_ref")
    mu = validate_distribution(mu, "mu", require_full_support=True)
    K, X, I, _ = A_policy.shape
    beta = as_float64(beta)
    if beta.dim() == 0:
        beta = beta.expand(K).clone()
    gamma_sched = resolve_gamma_schedule(gamma, M)
    lo, hi = float(lambda_box[0]), float(lambda_box[1])
    if not (0.0 < lo < hi):
        raise ValueError(f"lambda box must satisfy 0 < lambda_min < lambda_max, got {lambda_box}")
    if pi_t is None:
        pi_t = uniform_policy(X, I)
    pi_t = validate_distribution(pi_t, "pi_t", require_full_support=True)

    # Disagreement point from the SEPARATE reference-as-learner tensor (Eq. (10)).
    d = compute_disagreement_point(A_ref, mu, beta)

    if lambda_init is None:
        lam = torch.ones(K, dtype=torch.float64)
    else:
        lam = as_float64(lambda_init).clone()
        if lam.shape != (K,):
            raise ValueError(f"lambda_init must have shape ({K},), got {tuple(lam.shape)}")
    if (lam <= 0).any():
        raise ValueError("lambda_init must be strictly positive")

    history: List[dict] = []
    pi_warm: Optional[torch.Tensor] = None
    control_residual: Optional[float] = None

    if aggregation in ("nash", "utilitarian"):
        if aggregation == "utilitarian":
            lam = torch.ones(K, dtype=torch.float64)
        for m in range(M):
            sol = solve_weighted_policy(
                A_policy, mu, pi_t, lam, beta, eta, R, pi_init=pi_warm, damping=damping
            )
            if warm_start_policy:
                pi_warm = sol.pi
            V = policy_game_values(A_policy, mu, sol.pi, beta)
            s = V - d
            if log_every and (m % log_every == 0):
                history.append(_log_row(m, lam, V, d, s, sol.nu, sol.fixed_point_residual))
            if aggregation == "nash":
                grad = s - 1.0 / lam  # Eq. (19)
                lam = torch.clamp(lam - gamma_sched[m] * grad, min=lo, max=hi)  # Eq. (27)
    else:
        # Adversarial max-min controls: w on the simplex, multiplicative weights.
        w = torch.full((K,), 1.0 / K, dtype=torch.float64)
        for m in range(M):
            sol = solve_weighted_policy(
                A_policy, mu, pi_t, w, beta, eta, R, pi_init=pi_warm, damping=damping
            )
            if warm_start_policy:
                pi_warm = sol.pi
            V = policy_game_values(A_policy, mu, sol.pi, beta)
            s = V - d
            v = V if aggregation == "absolute_maxmin" else s
            if log_every and (m % log_every == 0):
                history.append(
                    _log_row(
                        m, w, V, d, s, sol.nu, sol.fixed_point_residual,
                        extra={"duality_gap": float((w * v).sum() - v.min())},
                    )
                )
            log_w = torch.log(w) - adversary_step * v
            w = torch.exp(log_w - torch.logsumexp(log_w, dim=0))
        lam = w

    # Final weighted-policy solution at the final multipliers (Algorithm 1, line 10).
    final = solve_weighted_policy(
        A_policy, mu, pi_t, lam, beta, eta, R, pi_init=pi_warm, damping=damping
    )
    V = policy_game_values(A_policy, mu, final.pi, beta)
    s = V - d
    if aggregation in ("absolute_maxmin", "surplus_maxmin"):
        v = V if aggregation == "absolute_maxmin" else s
        control_residual = float((lam * v).sum() - v.min())
    kkt = float((s - 1.0 / lam).abs().max()) if aggregation == "nash" else None
    history.append(
        _log_row(
            M, lam, V, d, s, final.nu, final.fixed_point_residual,
            extra=({"duality_gap": control_residual} if control_residual is not None else None),
        )
    )
    return DualSolveResult(
        aggregation=aggregation,
        lam=lam,
        V=V,
        d=d,
        surplus=s,
        pi=final.pi,
        nu=final.nu,
        q=final.q,
        kkt_residual=kkt,
        control_residual=control_residual,
        fixed_point_residual=final.fixed_point_residual,
        opponent_entropy=opponent_entropy(final.nu),
        opponent_ess=opponent_ess(final.nu),
        history=history,
        config={
            "aggregation": aggregation,
            "beta": [float(b) for b in beta],
            "eta": float(eta),
            "gamma": [float(g) for g in gamma_sched],
            "M": int(M),
            "R": int(R),
            "R_is_approximation": R == 1,
            "lambda_box": [lo, hi],
            "warm_start_policy": bool(warm_start_policy),
            "damping": float(damping),
            "adversary_step": float(adversary_step),
        },
    )


def proximal_divergence(pi: torch.Tensor, pi_t: torch.Tensor) -> float:
    """``D(pi || pi_t) = mean_x KL(pi(.|x) || pi_t(.|x))`` (below Eq. (15))."""
    pi = validate_distribution(pi, "pi")
    pi_t = validate_distribution(pi_t, "pi_t", require_full_support=True)
    return float(kl_divergence(pi, pi_t).mean())


def dual_objective_phi(
    A_policy: torch.Tensor,
    A_ref: torch.Tensor,
    mu: torch.Tensor,
    beta: torch.Tensor,
    eta: float,
    lam: torch.Tensor,
    *,
    pi_t: Optional[torch.Tensor] = None,
    R: int = 400,
    damping: float = 0.5,
    residual_tol: float = 1e-10,
) -> float:
    """Evaluate the dual function ``phi_t(lambda)`` of Eq. (17) ``eq:prox-dual``.

    ``phi_t(lambda) = -sum_k log lambda_k - K
    + max_pi { sum_k lambda_k s_k(pi) - (1/eta) D(pi || pi_t) }``.

    The inner max is solved to fixed-point convergence (verified against
    ``residual_tol``), so the envelope theorem gives
    ``d phi / d lambda_k = s_k(pi^*(lambda)) - 1/lambda_k`` (Eq. (19)) -- the
    identity the finite-difference test checks.
    """
    A_policy = validate_payoff_tensor(A_policy, "A_policy")
    K, X, I, _ = A_policy.shape
    if pi_t is None:
        pi_t = uniform_policy(X, I)
    lam = as_float64(lam)
    beta = as_float64(beta)
    if beta.dim() == 0:
        beta = beta.expand(K).clone()
    sol = solve_weighted_policy(A_policy, mu, pi_t, lam, beta, eta, R, damping=damping)
    if sol.fixed_point_residual > residual_tol:
        raise RuntimeError(
            f"inner solve did not converge: residual {sol.fixed_point_residual:.3e} "
            f"> tol {residual_tol:.3e}; increase R or damping"
        )
    d = compute_disagreement_point(A_ref, mu, beta)
    s = policy_game_values(A_policy, mu, sol.pi, beta) - d
    inner = float((lam * s).sum()) - proximal_divergence(sol.pi, pi_t) / eta
    return float(-torch.log(lam).sum()) - K + inner
