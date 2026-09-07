"""Soft Bradley--Terry reward models: the scalar-reward half of the matched table.

``BT-RM-Nash`` and ``BT-RM-utilitarian`` exist to answer one question -- does the
*game-valued* objective buy anything over a scalar reward model, holding the
aggregation fixed?  For that to be a fair question the reward models must be fit
to **exactly the same judged comparisons** the game tensors are built from, with
the same ties, the same swap averaging, and no extra supervision.

Two things in this module are load-bearing:

**Soft labels.**  The judge emits ``[[A]] / [[B]] / [[TIE]]`` in both
presentation orders.  Scoring ``q = 1 / 0.5 / 0`` and swap-averaging

    p_hat(A > B) = 0.5 * (q_AB + 1 - q_BA)

gives a probability in ``{0, 0.25, 0.5, 0.75, 1}`` that keeps ties and keeps
order disagreements.  Dropping ties would throw away the comparisons the
objectives disagree on most -- exactly the ones the bargaining story is about --
and would make the reward model fit a different labelling than the tensor.  So
the loss is the soft cross-entropy

    L = -p_hat log sigmoid(r_A - r_B) - (1 - p_hat) log sigmoid(r_B - r_A),

whose minimum over ``delta = r_A - r_B`` is at ``sigmoid(delta) = p_hat``.  Its
value at the optimum is the binary entropy of ``p_hat``, which is the floor the
reported validation NLL has to be read against -- a soft-label NLL cannot reach
zero, and comparing it to zero would make a well-calibrated model look broken.

**Reference-only normalization.**  The learned scale of a BT reward is
arbitrary, so surpluses from two independently trained heads are not comparable
until they share a scale.  ``r_tilde = (r - mu_ref) / max(sigma_ref, 1e-6)``
uses the *training reference responses only*: using the learner pool would leak
the thing being measured into the normalizer, and using the evaluation pool
would leak the test set.  ``mu_ref`` and ``sigma_ref`` are frozen and hashed,
and ``s^RM_k(pi) = E_pi[r_tilde_k] - E_mu[r_tilde_k]`` is invariant to that
choice of offset while being covariant in scale.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional

import torch

VERDICT_SCORES = {"A": 1.0, "TIE": 0.5, "B": 0.0}


def verdict_to_q(verdict: str) -> float:
    """``[[A]] -> 1.0``, ``[[TIE]] -> 0.5``, ``[[B]] -> 0.0`` for the shown order."""
    v = str(verdict).strip().upper().strip("[]")
    if v not in VERDICT_SCORES:
        raise ValueError(f"unparseable verdict {verdict!r}; expected A, B or TIE")
    return VERDICT_SCORES[v]


def swap_averaged_probability(q_ab: float, q_ba: float) -> float:
    """``p_hat(A > B) = 0.5 (q_AB + 1 - q_BA)``.

    ``q_ab`` scores the presentation with the semantic A first; ``q_ba`` scores
    the presentation with the semantic B first, so ``1 - q_ba`` is that order's
    vote for A.  A judge that always picks the first slot gives ``q_ab = 1``,
    ``q_ba = 1`` and therefore ``p_hat = 0.5``: position bias cancels instead of
    becoming a preference.
    """
    for name, q in (("q_ab", q_ab), ("q_ba", q_ba)):
        if not (0.0 <= float(q) <= 1.0):
            raise ValueError(f"{name} must lie in [0, 1], got {q}")
    return 0.5 * (float(q_ab) + 1.0 - float(q_ba))


def centered_margin(p_hat: float) -> float:
    """``Delta_hat = p_hat - 1/2``, the entry of the centered tensor (Eq. (2))."""
    return float(p_hat) - 0.5


def soft_bt_loss(reward_a: torch.Tensor, reward_b: torch.Tensor,
                 p_hat: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Soft Bradley--Terry cross-entropy; ties contribute at ``p_hat = 0.5``."""
    delta = reward_a - reward_b
    p = p_hat.to(delta.dtype)
    if bool(((p < 0) | (p > 1)).any()):
        raise ValueError("p_hat must lie in [0, 1]")
    loss = -(p * torch.nn.functional.logsigmoid(delta)
             + (1.0 - p) * torch.nn.functional.logsigmoid(-delta))
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"unknown reduction {reduction!r}")


def soft_label_entropy(p_hat: torch.Tensor) -> torch.Tensor:
    """Binary entropy of the soft label: the irreducible floor of ``soft_bt_loss``.

    Report validation NLL against this, never against zero.
    """
    p = p_hat.double().clamp(1e-12, 1 - 1e-12)
    return -(p * p.log() + (1 - p) * (1 - p).log())


# --- diagnostics ------------------------------------------------------------

def brier_score(p_pred: torch.Tensor, p_hat: torch.Tensor) -> float:
    return float(((p_pred.double() - p_hat.double()) ** 2).mean())


def expected_calibration_error(p_pred: torch.Tensor, p_hat: torch.Tensor,
                               n_bins: int = 10) -> float:
    """Binned |mean predicted - mean observed|, weighted by bin occupancy."""
    p_pred = p_pred.double()
    p_hat = p_hat.double()
    edges = torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float64)
    total, n = 0.0, p_pred.numel()
    if n == 0:
        return float("nan")
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p_pred > lo) & (p_pred <= hi) if i else (p_pred >= lo) & (p_pred <= hi)
        if not bool(m.any()):
            continue
        total += float(m.sum()) / n * float((p_pred[m].mean() - p_hat[m].mean()).abs())
    return total


def hard_accuracy_excluding_ties(delta: torch.Tensor, p_hat: torch.Tensor,
                                 tie_tol: float = 1e-9) -> Optional[float]:
    """Accuracy on comparisons whose *label* is not an exact tie.

    Ties are excluded from the accuracy only -- they stay in the loss.  A
    hard accuracy computed over ties would score a coin flip on an
    unanswerable comparison.
    """
    keep = (p_hat.double() - 0.5).abs() > tie_tol
    if not bool(keep.any()):
        return None
    pred = (delta[keep] > 0).double()
    truth = (p_hat.double()[keep] > 0.5).double()
    return float((pred == truth).double().mean())


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    den = float(a.norm() * b.norm())
    return float(a.dot(b)) / den if den > 0 else float("nan")


# --- reference-only normalization ------------------------------------------

@dataclass
class RewardNormalization:
    """Frozen ``(mu_ref, sigma_ref)`` per objective, plus the hash that pins it."""

    mu_ref: torch.Tensor      # (K,)
    sigma_ref: torch.Tensor   # (K,)
    n_reference_responses: int
    source: str
    sha256: str

    def apply(self, r: torch.Tensor) -> torch.Tensor:
        """``r_tilde = (r - mu_ref) / max(sigma_ref, 1e-6)`` broadcast over (K, X, *)."""
        r = r.double()
        if r.shape[0] != self.mu_ref.numel():
            raise ValueError(f"reward table has K={r.shape[0]}, normalization has "
                             f"K={self.mu_ref.numel()}")
        shape = (-1,) + (1,) * (r.dim() - 1)
        return (r - self.mu_ref.view(shape)) / self.sigma_ref.clamp_min(1e-6).view(shape)

    def to_dict(self) -> dict:
        return {"mu_ref": [float(v) for v in self.mu_ref],
                "sigma_ref": [float(v) for v in self.sigma_ref],
                "n_reference_responses": self.n_reference_responses,
                "source": self.source, "sha256": self.sha256,
                "formula": "(r - mu_ref) / max(sigma_ref, 1e-6)",
                "fitted_on": "training reference responses only"}


def fit_reward_normalization(r_reference: torch.Tensor, source: str) -> RewardNormalization:
    """Fit ``(mu_ref, sigma_ref)`` on the training REFERENCE responses only.

    ``r_reference`` has shape ``(K, X, J)``.  The learner pool is deliberately
    not used: it is the thing the surplus measures, and normalizing by it would
    fold the answer into the scale.
    """
    r = r_reference.double()
    if r.dim() != 3:
        raise ValueError(f"r_reference must be (K, X, J), got {tuple(r.shape)}")
    flat = r.reshape(r.shape[0], -1)
    mu = flat.mean(dim=1)
    sigma = flat.std(dim=1, unbiased=True)
    payload = json.dumps({"mu": [float(v) for v in mu],
                          "sigma": [float(v) for v in sigma],
                          "n": int(flat.shape[1]), "source": source},
                         sort_keys=True).encode()
    return RewardNormalization(mu_ref=mu, sigma_ref=sigma,
                               n_reference_responses=int(flat.shape[1]),
                               source=source,
                               sha256=hashlib.sha256(payload).hexdigest())
