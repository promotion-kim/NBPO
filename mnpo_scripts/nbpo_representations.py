"""Objective representations for the finite-pool NBPO family.

A *representation* answers one question about a policy on the frozen response
pool: what is objective ``k`` worth, and which direction improves it?  It
exposes

* ``opponent_and_gradient(pi) -> (nu, q)`` -- the comparator distribution the
  objective is measured against and the gradient representer
  ``q[k, x, i]``;
* ``game_values(pi) -> V[k]`` -- the value of the objective at ``pi``;
* ``disagreement -> d[k]`` -- the value of the reference, *measured* from the
  reference-as-learner construction (never hard-coded to zero);

so that ``s_k(pi) = V_k(pi) - d_k``.

Three representations share one aggregation/solver/realization stack, and they
are the only thing that differs between the matched Table-1 rows:

``adaptive_game``
    The manuscript's finite-temperature game (Eqs. (6)-(10)).  The comparator
    is the KL-regularized best-responding opponent ``nu*_k`` (Eq. (7)), so it
    *moves with the learner*; ``V`` is the soft-min game value (Eq. (8)).

``fixed_reference``
    The comparator is frozen at the empirical reference ``mu``:

        q^FR_k(y|x) = E_{z ~ mu(.|x)} Delta_k(y, z|x)
        V^FR_k(pi)  = E_{y ~ pi(.|x)} q^FR_k(y|x)
        d^FR_k      = V^FR_k(mu)

    This is *exactly* the ``beta -> infinity`` limit of ``adaptive_game``
    (Eq. (7) sends ``nu* -> mu``), which is what
    ``tests/test_nbpo_representations.py`` checks numerically.  It is the
    matched control that isolates the adaptive opponent, and it is implemented
    here -- in the modern finite-pool path -- not by calling the legacy
    ``scripts/bpo`` pipeline.

``bt_reward``
    The objective is a *scalar* Bradley-Terry reward model, one per objective,
    read off a per-response score table ``r_tilde[k, x, i]`` that was
    normalized on training reference responses only.  There is no opponent;
    ``q`` is the reward itself and ``V`` is its expectation, so

        s^RM_k(pi) = E_pi[r_tilde_k] - E_mu[r_tilde_k].

    ``nu`` is reported as ``mu`` purely so the artifact has one uniform shape;
    ``has_opponent`` is False and the Eq. (26) opponent-sampling path is not
    used for this representation.

Everything is float64 (the finite-pool solver precision).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from mnpo_scripts.nbpo_core import (
    as_float64,
    compute_disagreement_point,
    compute_margins,
    compute_objective_gradient,
    compute_regularized_game_value,
    compute_regularized_opponent,
    uniform_policy,
    validate_distribution,
    validate_game_utility_tensor,
    validate_reference_tensor,
)

REPRESENTATIONS = ("adaptive_game", "fixed_reference", "bt_reward")


def _check_learner_pi(pi: torch.Tensor, X: int, I: int, name: str = "pi") -> torch.Tensor:
    pi = validate_distribution(pi, name)
    if pi.shape != (X, I):
        raise ValueError(f"{name} must have shape (X={X}, I={I}), got {tuple(pi.shape)}")
    return pi


@dataclass
class RepresentationInfo:
    """Everything an artifact needs to say which representation produced it."""

    type: str
    policy_adaptive: bool
    has_opponent: bool
    beta: Optional[list]
    reference_construction: Optional[str]
    detail: dict


class ObjectiveRepresentation:
    """Interface shared by every matched Table-1 row."""

    type: str = "abstract"
    policy_adaptive: bool = False
    has_opponent: bool = False

    K: int
    X: int
    I: int
    J: int

    def opponent_and_gradient(self, pi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def game_values(self, pi: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def disagreement(self) -> torch.Tensor:
        raise NotImplementedError

    def surplus(self, pi: torch.Tensor) -> torch.Tensor:
        """``s_k(pi) = V_k(pi) - d_k`` (Eq. (10)); raw, never clamped."""
        return self.game_values(pi) - self.disagreement

    def info(self) -> RepresentationInfo:
        raise NotImplementedError


class AdaptiveGameRepresentation(ObjectiveRepresentation):
    """The manuscript's finite-temperature game: the comparator best-responds.

    ``nu*_k`` is Eq. (7) at the current policy, ``q_k`` is Eq. (9) against it,
    and ``V_{k,beta_k}`` is the soft-min value of Eq. (8).  ``d_k`` comes from
    the separate reference-as-learner tensor through the identical value map.
    """

    type = "adaptive_game"
    policy_adaptive = True
    has_opponent = True

    def __init__(self, A_policy, A_ref, mu, beta, reference_construction: str = "shared_pool"):
        self.A = validate_game_utility_tensor(A_policy, "A_policy")
        self.A_ref = validate_reference_tensor(A_ref, "A_ref", reference_construction)
        self.mu = validate_distribution(mu, "mu", require_full_support=True)
        self.K, self.X, self.I, self.J = self.A.shape
        beta = as_float64(beta)
        if beta.dim() == 0:
            beta = beta.expand(self.K).clone()
        if beta.shape != (self.K,) or not bool((beta > 0).all()):
            raise ValueError("beta must be a positive vector of length K")
        self.beta = beta
        self.reference_construction = reference_construction
        self._d = compute_disagreement_point(self.A_ref, self.mu, self.beta, reference_construction)

    def opponent_and_gradient(self, pi):
        pi = _check_learner_pi(pi, self.X, self.I)
        nu = compute_regularized_opponent(compute_margins(self.A, pi), self.mu, self.beta)
        return nu, compute_objective_gradient(self.A, nu)

    def game_values(self, pi):
        pi = _check_learner_pi(pi, self.X, self.I)
        return compute_regularized_game_value(
            compute_margins(self.A, pi), self.mu, self.beta, form="softmin")

    @property
    def disagreement(self):
        return self._d

    def info(self):
        return RepresentationInfo(
            type=self.type, policy_adaptive=True, has_opponent=True,
            beta=[float(b) for b in self.beta],
            reference_construction=self.reference_construction,
            detail={"opponent": "kl_regularized_best_response_eq7",
                    "value": "soft_min_game_value_eq8"})


class FixedReferenceRepresentation(ObjectiveRepresentation):
    """Comparator frozen at the empirical reference ``mu`` (the beta -> infinity limit).

    ``q^FR`` does not depend on the learner, so the weighted proximal solve of
    Eq. (21) is exact in a single map: the fixed-point residual is zero by
    construction, and ``R`` has no effect.  That is a property of the
    representation, and the solver reports it rather than pretending an
    approximation was made.

    ``d^FR_k = V^FR_k(mu)`` is *measured* from the reference-as-learner tensor
    with ``mu`` on both sides -- the same construction ``adaptive_game`` uses.
    For an exactly skew-symmetric shared pool with uniform ``mu`` this evaluates
    to zero; the code computes it rather than assuming it, and the value is
    written into the artifact.
    """

    type = "fixed_reference"
    policy_adaptive = False
    has_opponent = True  # the opponent exists and is exactly mu

    def __init__(self, A_policy, A_ref, mu, reference_construction: str = "shared_pool",
                 mu_learner=None):
        self.A = validate_game_utility_tensor(A_policy, "A_policy")
        self.A_ref = validate_reference_tensor(A_ref, "A_ref", reference_construction)
        self.mu = validate_distribution(mu, "mu", require_full_support=True)
        self.K, self.X, self.I, self.J = self.A.shape
        self.reference_construction = reference_construction
        # q^FR[k, x, i] = sum_j mu[x, j] A[k, x, i, j]   (constant in pi)
        self._q = torch.einsum("xj,kxij->kxi", self.mu, self.A)
        # The formal opponent is mu itself, broadcast to the (K, X, J) shape the
        # rest of the stack expects.
        self._nu = self.mu.unsqueeze(0).expand(self.K, self.X, self.J).contiguous()
        # d^FR = V^FR(mu): the SAME reference distribution on both sides of the
        # reference game, exactly as compute_disagreement_point requires.
        if mu_learner is None:
            expected = 1.0 / self.A_ref.shape[2]
            if not bool((self.mu - expected).abs().max() < 1e-9):
                raise ValueError(
                    "fixed_reference d_k = V^FR(mu) needs the same mu on both sides of the "
                    "reference game; pass mu_learner explicitly for a nonuniform mu")
            pi_ref = uniform_policy(self.A_ref.shape[1], self.A_ref.shape[2])
        else:
            pi_ref = _check_learner_pi(mu_learner, self.A_ref.shape[1], self.A_ref.shape[2],
                                       "mu_learner")
        q_ref = torch.einsum("xj,kxij->kxi", self.mu, self.A_ref)
        self._d = torch.einsum("xi,kxi->k", pi_ref, q_ref) / float(self.A_ref.shape[1])

    def opponent_and_gradient(self, pi):
        _check_learner_pi(pi, self.X, self.I)
        return self._nu, self._q

    def game_values(self, pi):
        pi = _check_learner_pi(pi, self.X, self.I)
        return torch.einsum("xi,kxi->k", pi, self._q) / float(self.X)

    @property
    def disagreement(self):
        return self._d

    def info(self):
        return RepresentationInfo(
            type=self.type, policy_adaptive=False, has_opponent=True, beta=None,
            reference_construction=self.reference_construction,
            detail={"opponent": "empirical_reference_mu",
                    "value": "linear_expected_centered_preference",
                    "equals_adaptive_game_limit": "beta -> infinity"})


class BTRewardRepresentation(ObjectiveRepresentation):
    """Scalar Bradley-Terry reward models, one per objective.

    ``r_tilde[k, x, i]`` are the *already normalized* learner-pool rewards and
    ``r_tilde_ref[k, x, j]`` the reference-pool rewards under the same frozen
    ``(mu_k_ref, sigma_k_ref)``.  Then

        V^RM_k(pi) = E_pi[r_tilde_k],   d^RM_k = E_mu[r_tilde_k],

    so ``s^RM_k(pi) = E_pi[r_tilde_k] - E_mu[r_tilde_k]``.  There is no
    opponent: ``nu`` is returned as ``mu`` only to keep one artifact shape, and
    ``has_opponent`` is False so the Eq. (26) opponent-sampling path is skipped.
    """

    type = "bt_reward"
    policy_adaptive = False
    has_opponent = False

    def __init__(self, r_learner, r_reference, mu, normalization: Optional[dict] = None):
        self._r = as_float64(r_learner)
        r_ref = as_float64(r_reference)
        if self._r.dim() != 3 or r_ref.dim() != 3:
            raise ValueError("reward tables must have shape (K, X, I) / (K, X, J)")
        if self._r.shape[:2] != r_ref.shape[:2]:
            raise ValueError("learner and reference reward tables disagree on (K, X)")
        if not torch.isfinite(self._r).all() or not torch.isfinite(r_ref).all():
            raise ValueError("reward tables contain non-finite values")
        self.K, self.X, self.I = self._r.shape
        self.J = r_ref.shape[2]
        self.mu = validate_distribution(mu, "mu", require_full_support=True)
        if self.mu.shape != (self.X, self.J):
            raise ValueError(f"mu must have shape ({self.X}, {self.J}), got {tuple(self.mu.shape)}")
        self._nu = self.mu.unsqueeze(0).expand(self.K, self.X, self.J).contiguous()
        self._d = torch.einsum("xj,kxj->k", self.mu, r_ref) / float(self.X)
        self.normalization = dict(normalization or {})

    def opponent_and_gradient(self, pi):
        _check_learner_pi(pi, self.X, self.I)
        return self._nu, self._r

    def game_values(self, pi):
        pi = _check_learner_pi(pi, self.X, self.I)
        return torch.einsum("xi,kxi->k", pi, self._r) / float(self.X)

    @property
    def disagreement(self):
        return self._d

    def info(self):
        return RepresentationInfo(
            type=self.type, policy_adaptive=False, has_opponent=False, beta=None,
            reference_construction=None,
            detail={"opponent": "none_scalar_reward",
                    "value": "expected_normalized_reward",
                    "normalization": self.normalization})


def build_representation(kind: str, **kwargs) -> ObjectiveRepresentation:
    """Construct a representation by its config name."""
    if kind == "adaptive_game":
        return AdaptiveGameRepresentation(**kwargs)
    if kind == "fixed_reference":
        return FixedReferenceRepresentation(**kwargs)
    if kind == "bt_reward":
        return BTRewardRepresentation(**kwargs)
    raise ValueError(f"representation must be one of {REPRESENTATIONS}, got {kind!r}")
