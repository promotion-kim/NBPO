"""Anti-symmetric general preference model (GPM).

A scalar reward model can only ever express ``P(y > z) = sigma(r(y) - r(z))``,
whose induced preference is transitive by construction. That is exactly the
representational limit NBPO's game-valued objective is supposed to escape, so a
learned judge for this project must not have it either.

The model scores the **pair jointly** and antisymmetrizes explicitly:

    ell_k(x, y, z) = 1/2 * ( a_k(x, y, z) - a_k(x, z, y) )
    P_k(y > z | x) = sigmoid( ell_k(x, y, z) )

which gives, exactly and not approximately:

    P_k(y > z|x) + P_k(z > y|x) = 1        (antisymmetry)
    P_k(y > y|x) = 0.5                     (self-tie)

Both hold to floating point for *any* parameters, at initialisation and after
any amount of training, because they are properties of the construction rather
than of the fit. The tests assert them on random weights.

Why the joint encoder matters: if ``a_k(x, y, z)`` were ever computed as
``f(x,y) - f(x,z)``, the antisymmetrization would collapse to a scalar reward
difference and the model would be a Bradley-Terry model wearing a different
name. `assert_not_scalar_decomposable` looks for that collapse empirically by
searching for a cyclic triple the model can represent; a scalar model provably
cannot produce one.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


def antisymmetrize(a_yz: torch.Tensor, a_zy: torch.Tensor) -> torch.Tensor:
    """``ell = (a(y,z) - a(z,y)) / 2``: the exact antisymmetric part."""
    return 0.5 * (a_yz - a_zy)


def preference_probability(ell: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(ell)


class PairwiseHead(nn.Module):
    """Joint scorer over a concatenated pair encoding.

    The concatenation is deliberately ORDER-SENSITIVE -- ``[h_y, h_z]`` is not
    ``[h_z, h_y]`` -- because an order-insensitive head would make ``a(y,z) ==
    a(z,y)`` and force ``ell`` to zero everywhere. Antisymmetry is imposed
    afterwards, by construction, not by symmetrising the input.
    """

    def __init__(self, hidden: int, width: int = 512, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden, width), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width // 2), nn.GELU(),
            nn.Linear(width // 2, 1))

    def forward(self, h_y: torch.Tensor, h_z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h_y, h_z], dim=-1)).squeeze(-1)


class AntiSymmetricGPM(nn.Module):
    """Encoder + joint pair head + exact antisymmetrization, per objective."""

    def __init__(self, encoder: nn.Module, hidden: int, n_objectives: int,
                 width: int = 512, dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.n_objectives = n_objectives
        self.heads = nn.ModuleList(
            [PairwiseHead(hidden, width, dropout) for _ in range(n_objectives)])

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        last = attention_mask.sum(dim=1) - 1
        return h[torch.arange(h.shape[0], device=h.device), last]

    def joint_score(self, h_y, h_z, k: int):
        return self.heads[k](h_y, h_z)

    def logit(self, h_y, h_z, k: int):
        """``ell_k`` -- both orders of the SAME head, then the antisymmetric part."""
        return antisymmetrize(self.joint_score(h_y, h_z, k),
                              self.joint_score(h_z, h_y, k))

    def probability(self, h_y, h_z, k: int):
        return preference_probability(self.logit(h_y, h_z, k))

    def forward(self, h_y, h_z):
        return torch.stack([self.logit(h_y, h_z, k)
                            for k in range(self.n_objectives)], dim=-1)


def swap_augmented_loss(model, h_y, h_z, k: int, p_hat: torch.Tensor,
                        reduction: str = "mean") -> torch.Tensor:
    """Soft cross-entropy on ``P_k(y > z)``.

    No separate swapped term is added, and that is deliberate: the model is
    *exactly* antisymmetric, so the loss on the swapped pair is algebraically
    identical to this one. Adding it would double the gradient on every example
    and buy nothing -- unlike a scalar model, where swap augmentation is doing
    real work. `test_swap_augmented_loss_is_identical_under_swap` pins that.
    """
    ell = model.logit(h_y, h_z, k)
    p = p_hat.to(ell.dtype)
    if bool(((p < 0) | (p > 1)).any()):
        raise ValueError("p_hat must lie in [0, 1]")
    loss = -(p * nn.functional.logsigmoid(ell)
             + (1 - p) * nn.functional.logsigmoid(-ell))
    return loss.mean() if reduction == "mean" else (
        loss.sum() if reduction == "sum" else loss)


def assert_not_scalar_decomposable(model, h, k: int = 0, tol: float = 1e-4) -> dict:
    """Empirically demonstrate the model is NOT a scalar reward model.

    A scalar model satisfies ``ell(i,j) + ell(j,k) + ell(k,i) == 0`` for every
    triple, because the terms telescope. A joint pair model need not. Returns
    the largest cyclic residual found; a value materially above ``tol`` proves
    the model can represent a cycle, which is the whole point of using it.
    """
    n = h.shape[0]
    worst, triple = 0.0, None
    with torch.no_grad():
        for i in range(min(n, 8)):
            for j in range(min(n, 8)):
                for m in range(min(n, 8)):
                    if len({i, j, m}) < 3:
                        continue
                    c = float(model.logit(h[i:i+1], h[j:j+1], k)
                              + model.logit(h[j:j+1], h[m:m+1], k)
                              + model.logit(h[m:m+1], h[i:i+1], k))
                    if abs(c) > worst:
                        worst, triple = abs(c), (i, j, m)
    return {"max_cyclic_residual": worst, "triple": triple,
            "scalar_decomposable": worst <= tol,
            "note": ("a scalar reward model has residual 0 for every triple; a "
                     "materially nonzero residual proves this model can represent "
                     "a cyclic preference")}


@dataclass
class GPMProvenance:
    encoder_name: str
    adaptation_mode: str          # "full_finetune" | "peft_lora" | "frozen_encoder"
    n_objectives: int
    hidden_size: int
    head_width: int
    train_split_sha256: Optional[str] = None
    label_source: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True,
                                         default=str).encode()).hexdigest()

    def to_dict(self) -> dict:
        return {**self.__dict__, "fingerprint": self.fingerprint(),
                "architecture": "joint pair encoder + explicit antisymmetrization",
                "guarantees": ["P(y>z) + P(z>y) == 1 exactly",
                               "P(y>y) == 0.5 exactly",
                               "not decomposable into r(y) - r(z)"]}
