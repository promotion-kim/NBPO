"""Soft Bradley--Terry reward models: the scalar-reward arm of the matched table.

The BT rows only isolate "game value vs scalar reward" if the reward models see
the same comparisons, the same ties and the same swap averaging as the game
tensors. These tests pin the label algebra, the loss, and the frozen
reference-only normalization that makes two independently trained heads
comparable.
"""
import math

import pytest
import torch

from mnpo_scripts.bt_reward import (
    brier_score,
    centered_margin,
    expected_calibration_error,
    fit_reward_normalization,
    hard_accuracy_excluding_ties,
    soft_bt_loss,
    soft_label_entropy,
    swap_averaged_probability,
    verdict_to_q,
)
from mnpo_scripts.nbpo_core import uniform_policy
from mnpo_scripts.nbpo_representations import BTRewardRepresentation


# --- label algebra ----------------------------------------------------------

@pytest.mark.parametrize("verdict,q", [("[[A]]", 1.0), ("[[TIE]]", 0.5), ("[[B]]", 0.0),
                                       ("A", 1.0), ("tie", 0.5)])
def test_verdict_scoring(verdict, q):
    assert verdict_to_q(verdict) == q


def test_unparseable_verdict_is_rejected_not_defaulted():
    with pytest.raises(ValueError, match="unparseable verdict"):
        verdict_to_q("[[MAYBE]]")


def test_swap_averaging_cancels_pure_position_bias():
    """A judge that always picks whichever response is shown first carries no
    preference: q_AB = 1 and q_BA = 1 must give exactly 0.5."""
    assert swap_averaged_probability(1.0, 1.0) == 0.5
    assert centered_margin(swap_averaged_probability(1.0, 1.0)) == 0.0


def test_swap_averaging_is_exactly_skew_symmetric():
    """p_hat(B > A) = 1 - p_hat(A > B) for every pair of verdicts."""
    for q_ab in (0.0, 0.5, 1.0):
        for q_ba in (0.0, 0.5, 1.0):
            fwd = swap_averaged_probability(q_ab, q_ba)
            rev = swap_averaged_probability(q_ba, q_ab)
            assert fwd + rev == pytest.approx(1.0, abs=1e-15)
            assert centered_margin(fwd) == pytest.approx(-centered_margin(rev), abs=1e-15)


def test_consistent_agreement_gives_a_decisive_label():
    assert swap_averaged_probability(1.0, 0.0) == 1.0     # both orders say A
    assert swap_averaged_probability(0.0, 1.0) == 0.0     # both orders say B


def test_tie_and_split_labels_land_on_the_declared_grid():
    seen = {swap_averaged_probability(a, b)
            for a in (0.0, 0.5, 1.0) for b in (0.0, 0.5, 1.0)}
    assert seen == {0.0, 0.25, 0.5, 0.75, 1.0}


# --- the loss ---------------------------------------------------------------

@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_soft_bt_loss_is_minimized_where_sigmoid_matches_the_label(p):
    p_hat = torch.tensor([p], dtype=torch.float64)
    if p in (0.0, 1.0):
        pytest.skip("hard label: the optimum is at infinity")
    delta = torch.tensor([math.log(p / (1 - p))], dtype=torch.float64, requires_grad=True)
    loss = soft_bt_loss(delta, torch.zeros_like(delta), p_hat)
    loss.backward()
    assert abs(float(delta.grad)) < 1e-12
    # and the optimal value is the binary entropy, not zero
    assert float(loss) == pytest.approx(float(soft_label_entropy(p_hat)), abs=1e-12)


def test_soft_bt_loss_reduces_to_standard_bt_on_hard_labels():
    ra = torch.tensor([1.2, -0.3], dtype=torch.float64)
    rb = torch.tensor([0.1, 0.8], dtype=torch.float64)
    p = torch.tensor([1.0, 1.0], dtype=torch.float64)
    expected = -torch.nn.functional.logsigmoid(ra - rb)
    assert torch.allclose(soft_bt_loss(ra, rb, p, reduction="none"), expected)


def test_soft_bt_loss_is_symmetric_under_swapping_the_pair():
    ra = torch.tensor([0.7], dtype=torch.float64)
    rb = torch.tensor([-0.4], dtype=torch.float64)
    p = torch.tensor([0.75], dtype=torch.float64)
    assert soft_bt_loss(ra, rb, p) == pytest.approx(
        float(soft_bt_loss(rb, ra, 1.0 - p)), abs=1e-14)


def test_ties_are_kept_and_pull_the_margin_to_zero():
    """A tie is a real observation at p = 0.5, not a row to delete."""
    p = torch.tensor([0.5], dtype=torch.float64)
    delta = torch.tensor([0.9], dtype=torch.float64, requires_grad=True)
    soft_bt_loss(delta, torch.zeros_like(delta), p).backward()
    assert float(delta.grad) > 0        # pushes delta down toward 0


def test_loss_rejects_out_of_range_labels():
    with pytest.raises(ValueError, match=r"p_hat must lie"):
        soft_bt_loss(torch.zeros(2), torch.zeros(2), torch.tensor([1.4, 0.2]))


# --- diagnostics ------------------------------------------------------------

def test_hard_accuracy_excludes_exact_ties_only():
    delta = torch.tensor([1.0, -1.0, 5.0], dtype=torch.float64)
    p_hat = torch.tensor([0.75, 0.75, 0.5], dtype=torch.float64)
    assert hard_accuracy_excluding_ties(delta, p_hat) == pytest.approx(0.5)
    assert hard_accuracy_excluding_ties(delta, torch.full((3,), 0.5, dtype=torch.float64)) is None


def test_perfect_calibration_scores_zero_ece_and_matching_brier():
    p = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float64)
    assert expected_calibration_error(p, p) == pytest.approx(0.0, abs=1e-12)
    assert brier_score(p, p) == pytest.approx(0.0, abs=1e-12)
    assert brier_score(p, 1.0 - p) > 0.0


# --- reference-only normalization ------------------------------------------

def test_normalization_is_fitted_on_reference_responses_only():
    gen = torch.Generator().manual_seed(1)
    K, X, J = 3, 20, 4
    r_ref = torch.randn(K, X, J, dtype=torch.float64, generator=gen) * 3.0 + 5.0
    norm = fit_reward_normalization(r_ref, source="unit-test")
    z = norm.apply(r_ref)
    assert torch.allclose(z.reshape(K, -1).mean(1), torch.zeros(K, dtype=torch.float64),
                          atol=1e-12)
    assert torch.allclose(z.reshape(K, -1).std(1, unbiased=True),
                          torch.ones(K, dtype=torch.float64), atol=1e-12)
    assert norm.n_reference_responses == X * J
    assert len(norm.sha256) == 64
    assert norm.to_dict()["fitted_on"] == "training reference responses only"
    # a different fit must produce a different hash
    other = fit_reward_normalization(r_ref * 2, source="unit-test")
    assert other.sha256 != norm.sha256


def test_a_learner_pool_cannot_change_the_frozen_normalizer():
    """The normalizer is a property of the reference pool; scoring more learner
    responses must not move it."""
    gen = torch.Generator().manual_seed(2)
    r_ref = torch.randn(2, 10, 4, dtype=torch.float64, generator=gen)
    a = fit_reward_normalization(r_ref, source="s")
    b = fit_reward_normalization(r_ref, source="s")
    assert a.sha256 == b.sha256
    assert torch.equal(a.mu_ref, b.mu_ref) and torch.equal(a.sigma_ref, b.sigma_ref)


def test_degenerate_sigma_is_floored_not_divided_by_zero():
    r_ref = torch.full((2, 5, 4), 3.0, dtype=torch.float64)
    norm = fit_reward_normalization(r_ref, source="s")
    out = norm.apply(r_ref)
    assert torch.isfinite(out).all()
    assert float(out.abs().max()) == 0.0


def test_rm_surplus_is_affine_invariant_under_the_normalization():
    """s^RM = E_pi[r_tilde] - E_mu[r_tilde] must not depend on an affine
    re-parameterization of the raw reward head, which is what licenses
    comparing two independently trained objectives."""
    gen = torch.Generator().manual_seed(3)
    K, X, I, J = 3, 25, 4, 4
    r = torch.randn(K, X, I, dtype=torch.float64, generator=gen)
    r_ref = torch.randn(K, X, J, dtype=torch.float64, generator=gen)
    mu = uniform_policy(X, J)
    pi = torch.rand(X, I, dtype=torch.float64, generator=gen)
    pi = pi / pi.sum(-1, keepdim=True)

    def surplus(raw, raw_ref):
        norm = fit_reward_normalization(raw_ref, source="s")
        return BTRewardRepresentation(norm.apply(raw), norm.apply(raw_ref), mu).surplus(pi)

    base = surplus(r, r_ref)
    a = torch.tensor([2.0, 0.5, 3.0], dtype=torch.float64).view(K, 1, 1)
    b = torch.tensor([-1.0, 4.0, 0.0], dtype=torch.float64).view(K, 1, 1)
    moved = surplus(a * r + b, a * r_ref + b)
    assert torch.allclose(base, moved, atol=1e-12)
