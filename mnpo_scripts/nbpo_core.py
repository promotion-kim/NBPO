"""Finite-pool core math for finite-temperature NBPO (Algorithm 1 of the ICLR manuscript).

This module implements the population quantities of the manuscript
(``game_nbpo_iclr/main_v2.tex``) restricted to a frozen finite response pool, in
float64 torch. It is the paper-exact path: nothing here uses scalar reward
models, Bradley--Terry reconstruction, or fixed-anchor surpluses. The legacy
fixed-reference (beta = infinity) Anchored-BPO pipeline lives in ``scripts/bpo``
and is intentionally untouched.

Conventions
-----------
- ``A[k, x, i, j] = P_k(y_i > z_j | x) - 1/2`` -- the centered pairwise payoff,
  Eq. (2) ``eq:centered``. ``k`` indexes objectives, ``x`` prompts, ``i``
  learner responses, ``j`` reference-comparator responses.
- ``pi[x, i]`` is the learner policy over its response pool for prompt ``x``;
  ``mu[x, j]`` is the reference comparator distribution (uniform over sampled
  comparators unless stated otherwise).
- ``beta[k] > 0`` is the opponent temperature of objective ``k``.
- The disagreement point ``d_k = V_{k,beta_k}(mu)`` (Eq. (10)
  ``eq:game-surplus``) must be computed from a *separate* reference-as-learner
  tensor ``A_ref[k, x, i, j]`` via the same game value -- it is generally
  nonzero and is never replaced by ``g_k(mu, mu) = 0``.

All distribution arguments are validated (finite, nonnegative, rows summing to
one, and full support where the math requires it).
"""
from __future__ import annotations

from typing import Union

import torch

TensorLike = Union[torch.Tensor, list, tuple, float]

_SUM_ATOL = 1e-8


def as_float64(value: TensorLike) -> torch.Tensor:
    """Coerce to a float64 tensor (the finite-pool solver precision)."""
    return torch.as_tensor(value, dtype=torch.float64)


def validate_distribution(
    p: torch.Tensor,
    name: str,
    *,
    dim: int = -1,
    require_full_support: bool = False,
    atol: float = _SUM_ATOL,
) -> torch.Tensor:
    """Check that ``p`` is a proper probability distribution along ``dim``.

    Raises ``ValueError`` on non-finite values, negative entries, rows that do
    not sum to one within ``atol``, or (when ``require_full_support``) zero
    entries.
    """
    p = as_float64(p)
    if not torch.isfinite(p).all():
        raise ValueError(f"{name} contains non-finite values")
    if (p < 0).any():
        raise ValueError(f"{name} contains negative entries")
    sums = p.sum(dim=dim)
    if not torch.allclose(sums, torch.ones_like(sums), atol=atol, rtol=0.0):
        worst = (sums - 1.0).abs().max().item()
        raise ValueError(f"{name} rows must sum to one (worst deviation {worst:.3e})")
    if require_full_support and (p <= 0).any():
        raise ValueError(f"{name} must have full support (found zero entries)")
    return p


def validate_game_utility_tensor(A: torch.Tensor, name: str = "A") -> torch.Tensor:
    """Check a game-utility tensor ``A[k, x, i, j]``: 4-D and finite, any scale.

    The game algebra (Eqs. (6)-(9), (21)) is covariant under the positive
    rescaling ``A_k -> c_k A_k, beta_k -> c_k beta_k`` (Theorem `thm:scale`),
    so the solver must accept utilities outside the probability range. Range
    checks belong to the RAW artifact, see ``validate_centered_preference_tensor``.
    """
    A = as_float64(A)
    if A.dim() != 4:
        raise ValueError(f"{name} must have shape (K, X, I, J), got {tuple(A.shape)}")
    if not torch.isfinite(A).all():
        raise ValueError(f"{name} contains non-finite values")
    return A


def validate_centered_preference_tensor(A: torch.Tensor, name: str = "A") -> torch.Tensor:
    """Check a RAW centered-preference tensor ``P - 1/2`` (Eq. (2)): 4-D, finite, in [-1/2, 1/2].

    Applies to artifacts written from judge verdicts. Rescaled utilities
    (``c_k A_k``) are NOT centered preferences and must go through
    ``validate_game_utility_tensor`` instead.
    """
    A = validate_game_utility_tensor(A, name)
    if (A.abs() > 0.5 + _SUM_ATOL).any():
        raise ValueError(f"{name} entries must lie in [-1/2, 1/2] (centered preferences)")
    return A


# Backward-compatible alias: the math functions validate utilities, not ranges.
validate_payoff_tensor = validate_game_utility_tensor

REFERENCE_SKEW_TOL = 1e-12


REFERENCE_CONSTRUCTIONS = ("shared_pool", "independent_samples")


def reference_skew_residual(A_ref: torch.Tensor) -> float:
    """``max |A + A^T|``; zero iff the tensor is exactly skew-symmetric."""
    A_ref = as_float64(A_ref)
    if A_ref.shape[-1] != A_ref.shape[-2]:
        return float("nan")
    return float((A_ref + A_ref.transpose(-1, -2)).abs().max())


def validate_reference_tensor(A_ref: torch.Tensor, name: str = "A_ref",
                              construction: str = "shared_pool") -> torch.Tensor:
    """Check the reference-as-learner tensor against its DECLARED construction.

    ``shared_pool`` -- one response set sits on both sides of the game, which is
    the paper's assumption: ``A_k(i, j) = -A_k(j, i)`` and ``A_k(i, i) = 0``
    (Eqs. (1)-(2)). The builder enforces this exactly (one judged unordered pair
    -> ``a`` and ``-a``, diagonal zero); legacy artifacts are projected and the
    residual recorded. Square shape, exact skew symmetry and a zero diagonal are
    hard requirements here.

    ``independent_samples`` -- learner and comparator are two INDEPENDENT draws
    from ``mu`` (e.g. four sampled responses judged against four other sampled
    responses). This also estimates ``d_k = V_{k,beta_k}(mu)`` -- without the
    self-comparison ties the shared pool forces -- but the two supports are
    different response sets, so skew symmetry does not hold and must NOT be
    imposed. The residual is returned for reporting, never used to reject.

    The construction is never inferred: callers read it from the artifact's
    metadata, so an asymmetric tensor can never be accepted silently.
    """
    if construction not in REFERENCE_CONSTRUCTIONS:
        raise ValueError(
            f"reference construction must be one of {REFERENCE_CONSTRUCTIONS}, got {construction!r}"
        )
    A_ref = validate_game_utility_tensor(A_ref, name)
    if construction == "independent_samples":
        return A_ref
    if A_ref.shape[2] != A_ref.shape[3]:
        raise ValueError(
            f"{name} declared construction='shared_pool' must be square over one response "
            f"set (I == J), got {tuple(A_ref.shape)}"
        )
    diag = torch.diagonal(A_ref, dim1=-2, dim2=-1)
    if (diag != 0).any():
        raise ValueError(f"{name} diagonal must be exactly zero (Eq. (1): P(y > y) = 1/2)")
    skew = reference_skew_residual(A_ref)
    if skew >= REFERENCE_SKEW_TOL:
        raise ValueError(
            f"{name} is not skew-symmetric: max |A + A^T| = {skew:.3e} >= {REFERENCE_SKEW_TOL:g}"
        )
    return A_ref


def _validate_positive_vector(v: TensorLike, k: int, name: str) -> torch.Tensor:
    v = as_float64(v)
    if v.dim() == 0:
        v = v.expand(k).clone()
    if v.shape != (k,):
        raise ValueError(f"{name} must be a scalar or shape ({k},), got {tuple(v.shape)}")
    if not torch.isfinite(v).all() or (v <= 0).any():
        raise ValueError(f"{name} must be finite and strictly positive")
    return v


def compute_margins(A: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """Per-comparator margins ``r[k, x, j] = sum_i pi[x, i] A[k, x, i, j]``.

    This is ``r_{k,pi}(z | x) = E_{y ~ pi}[A_k(y, z | x)]``, Eq. (6) ``eq:r-def``.
    """
    A = validate_payoff_tensor(A)
    pi = validate_distribution(pi, "pi")
    if pi.shape != (A.shape[1], A.shape[2]):
        raise ValueError(
            f"pi must have shape (X={A.shape[1]}, I={A.shape[2]}), got {tuple(pi.shape)}"
        )
    return torch.einsum("xi,kxij->kxj", pi, A)


def compute_regularized_opponent(
    r: torch.Tensor, mu: torch.Tensor, beta: TensorLike
) -> torch.Tensor:
    """KL-regularized optimal opponent, Eq. (7) ``eq:regularized-opponent``.

    ``nu[k, x, j]`` proportional to ``mu[x, j] * exp(-r[k, x, j] / beta[k])``,
    normalized over ``j``. Computed in log space for stability.
    """
    r = as_float64(r)
    if r.dim() != 3:
        raise ValueError(f"r must have shape (K, X, J), got {tuple(r.shape)}")
    if not torch.isfinite(r).all():
        raise ValueError("r contains non-finite values")
    mu = validate_distribution(mu, "mu", require_full_support=True)
    if mu.shape != r.shape[1:]:
        raise ValueError(f"mu must have shape (X, J) = {tuple(r.shape[1:])}, got {tuple(mu.shape)}")
    beta = _validate_positive_vector(beta, r.shape[0], "beta")
    log_w = torch.log(mu).unsqueeze(0) - r / beta.view(-1, 1, 1)
    log_nu = log_w - torch.logsumexp(log_w, dim=-1, keepdim=True)
    return validate_distribution(torch.exp(log_nu), "nu")


def compute_regularized_game_value(
    r: torch.Tensor,
    mu: torch.Tensor,
    beta: TensorLike,
    *,
    form: str = "softmin",
    per_prompt: bool = False,
) -> torch.Tensor:
    """Entropy-regularized game value ``V_{k,beta_k}``, Eq. (8) ``eq:soft-value``.

    ``form="softmin"``: the closed form
    ``V[k] = mean_x -beta[k] * logsumexp_j(log mu[x, j] - r[k, x, j] / beta[k])``
    (the log of a mu-weighted expectation; the mean over prompts is OUTSIDE the log).

    ``form="primal"``: the equivalent primal expression at the optimal opponent,
    ``V[k] = mean_x [ sum_j nu[k,x,j] r[k,x,j] + beta[k] * KL(nu[k,x] || mu[x]) ]``
    (Definition 1, Eq. (5) ``eq:game-value`` evaluated at Eq. (7)).

    The two forms agree to numerical precision; both are exposed so tests can
    verify the identity. With ``per_prompt=True`` returns the (K, X) values
    before averaging over prompts.
    """
    r = as_float64(r)
    mu = validate_distribution(mu, "mu", require_full_support=True)
    beta = _validate_positive_vector(beta, r.shape[0], "beta")
    if form == "softmin":
        log_w = torch.log(mu).unsqueeze(0) - r / beta.view(-1, 1, 1)
        v_kx = -beta.view(-1, 1) * torch.logsumexp(log_w, dim=-1)
    elif form == "primal":
        nu = compute_regularized_opponent(r, mu, beta)
        kl = kl_divergence(nu, mu.unsqueeze(0).expand_as(nu))
        v_kx = (nu * r).sum(dim=-1) + beta.view(-1, 1) * kl
    else:
        raise ValueError(f"unknown form {form!r}; expected 'softmin' or 'primal'")
    return v_kx if per_prompt else v_kx.mean(dim=-1)


def kl_divergence(p: torch.Tensor, q: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """``KL(p || q)`` along ``dim``; ``0 log 0 = 0``; requires ``q > 0`` where ``p > 0``."""
    p = as_float64(p)
    q = as_float64(q)
    if ((p > 0) & (q <= 0)).any():
        raise ValueError("KL(p||q) undefined: q has zero mass where p is positive")
    ratio = torch.where(p > 0, p / q.clamp_min(1e-300), torch.ones_like(p))
    return (p * torch.log(ratio)).sum(dim=dim)


def compute_objective_gradient(A: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """Gradient representer ``q[k, x, i] = sum_j nu[k, x, j] A[k, x, i, j]``.

    This is ``q_{k,pi}(y | x) = E_{z ~ nu*}[A_k(y, z | x)]``, Eq. (9)
    ``eq:game-gradient`` -- the centered version (uses ``A_k``, not ``P_k``).
    """
    A = validate_payoff_tensor(A)
    nu = validate_distribution(nu, "nu")
    if nu.shape != (A.shape[0], A.shape[1], A.shape[3]):
        raise ValueError(
            f"nu must have shape (K, X, J) = {(A.shape[0], A.shape[1], A.shape[3])}, "
            f"got {tuple(nu.shape)}"
        )
    return torch.einsum("kxj,kxij->kxi", nu, A)


def weighted_policy_update(
    pi_center: torch.Tensor,
    q: torch.Tensor,
    lam: TensorLike,
    eta: float,
) -> torch.Tensor:
    """Exponential (mirror-descent) update, Eq. (21) ``eq:fixed-opponent-update``.

    ``pi_new[x, i]`` proportional to
    ``pi_center[x, i] * exp(eta * sum_k lam[k] * q[k, x, i])``.

    ``pi_center`` is the *proximal center* ``pi_t`` of the stage subproblem
    (Eq. (15)); it is NOT the fixed-point iterate ``pi^(r)``, which enters only
    through ``q``. ``lam`` is the RAW dual vector -- it is never normalized here
    (normalizing it rescales the effective step; see tests).
    """
    pi_center = validate_distribution(pi_center, "pi_center", require_full_support=True)
    q = as_float64(q)
    if not torch.isfinite(q).all():
        raise ValueError("q contains non-finite values")
    lam = _validate_positive_vector(lam, q.shape[0], "lam")
    if not (eta > 0):
        raise ValueError("eta must be strictly positive")
    score = torch.einsum("k,kxi->xi", lam, q)
    log_new = torch.log(pi_center) + eta * score
    log_new = log_new - torch.logsumexp(log_new, dim=-1, keepdim=True)
    return validate_distribution(torch.exp(log_new), "pi_new")


def compute_surplus(V_pi: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Surplus ``s_k = V_{k,beta_k}(pi) - d_k``, Eq. (10) ``eq:game-surplus``.

    ``d`` must be the game value of the reference (``d_k = V_{k,beta_k}(mu)``)
    computed from the reference-as-learner tensor; it is not assumed zero.
    """
    V_pi = as_float64(V_pi)
    d = as_float64(d)
    if V_pi.shape != d.shape:
        raise ValueError(f"V_pi {tuple(V_pi.shape)} and d {tuple(d.shape)} must match")
    if not torch.isfinite(V_pi).all() or not torch.isfinite(d).all():
        raise ValueError("game values must be finite")
    return V_pi - d


def opponent_entropy(nu: torch.Tensor) -> torch.Tensor:
    """Mean (over prompts) Shannon entropy of the opponent, per objective: (K,)."""
    nu = validate_distribution(nu, "nu")
    plogp = torch.where(nu > 0, nu * torch.log(nu), torch.zeros_like(nu))
    return (-plogp.sum(dim=-1)).mean(dim=-1)


def opponent_ess(nu: torch.Tensor) -> torch.Tensor:
    """Mean (over prompts) effective sample size ``1 / sum_j nu_j^2``, per objective: (K,)."""
    nu = validate_distribution(nu, "nu")
    return (1.0 / (nu**2).sum(dim=-1)).mean(dim=-1)


def uniform_policy(n_prompts: int, n_responses: int) -> torch.Tensor:
    """Uniform distribution over a finite response pool: shape (X, I).

    On the frozen pool, the empirical distribution of i.i.d. samples from a
    policy is uniform over its own sampled responses; this is the finite-pool
    representation of both ``pi_0 = mu`` and the reference-as-learner in
    ``d_k = V_{k,beta_k}(mu)``.
    """
    if n_prompts <= 0 or n_responses <= 0:
        raise ValueError("pool dimensions must be positive")
    return torch.full((n_prompts, n_responses), 1.0 / n_responses, dtype=torch.float64)


def compute_disagreement_point(
    A_ref: torch.Tensor, mu: torch.Tensor, beta: TensorLike,
    construction: str = "shared_pool",
) -> torch.Tensor:
    """``d_k = V_{k,beta_k}(mu)`` from the reference-as-learner tensor (Eq. (10)).

    ``A_ref[k, x, i, j]`` holds centered payoffs of reference responses (as
    learner, index ``i``) against reference comparators (index ``j``). The
    reference learner is uniform over its own pool. ``d`` is generally
    negative for skew-symmetric payoffs at beta < infinity and is never
    replaced by ``g_k(mu, mu) = 0``.
    """
    A_ref = validate_reference_tensor(A_ref, "A_ref", construction)
    pi_ref = uniform_policy(A_ref.shape[1], A_ref.shape[2])
    r_ref = compute_margins(A_ref, pi_ref)
    return compute_regularized_game_value(r_ref, mu, beta, form="softmin")
