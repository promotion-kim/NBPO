"""Reference loss formulas used by the AAAI revision smoke tests.

These helpers are intentionally small and tensor-only.  They let us verify the
loss values used by our launch configs against the official trainer formulas on
a fixed toy batch before running expensive model-scale jobs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_sigmoid_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """TRL DPOTrainer loss_type='sigmoid' with reverse-KL default."""

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios
    return (
        -F.logsigmoid(beta * logits) * (1.0 - label_smoothing)
        - F.logsigmoid(-beta * logits) * label_smoothing
    )


def ipo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """TRL DPOTrainer loss_type='ipo'.

    TRL computes IPO on completion-length-normalized log probabilities.
    The caller is responsible for supplying the normalized logps.
    """

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios
    return (logits - 1.0 / (2.0 * beta)) ** 2


def simpo_sigmoid_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    beta: float,
    gamma_over_beta: float,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Official SimPO sigmoid loss.

    The official implementation subtracts gamma/beta before multiplying by beta.
    """

    logits = (policy_chosen_logps - policy_rejected_logps) - gamma_over_beta
    return (
        -F.logsigmoid(beta * logits) * (1.0 - label_smoothing)
        - F.logsigmoid(-beta * logits) * label_smoothing
    )


def kto_pair_losses(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    policy_kl_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    reference_kl_logps: torch.Tensor,
    beta: float,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TRL KTOTrainer loss pieces for desirable and undesirable samples.

    This is only a formula check.  Full KTO training must still use a pointwise
    dataset with KL samples, matching TRL's KTOTrainer protocol.
    """

    kl = (policy_kl_logps - reference_kl_logps).mean().detach().clamp(min=0.0)
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps
    chosen_losses = desirable_weight * (1.0 - F.sigmoid(beta * (chosen_logratios - kl)))
    rejected_losses = undesirable_weight * (1.0 - F.sigmoid(beta * (kl - rejected_logratios)))
    return chosen_losses, rejected_losses
