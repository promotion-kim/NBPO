"""Objective representations: the three matched Table-1 rows differ only here.

Covers the invariants the matched comparison rests on:

* ``fixed_reference`` really is the ``beta -> infinity`` limit of
  ``adaptive_game`` (that is what makes "NBPO vs Fixed-reference Nash" isolate
  the adaptive opponent and nothing else);
* the disagreement point is *measured* from the reference-as-learner
  construction in every representation, never hard-coded to zero;
* ``bt_reward`` surpluses are invariant to a common offset and covariant under
  positive scaling, which is what licenses the frozen ``(mu_ref, sigma_ref)``
  normalization.
"""
import pytest
import torch

from mnpo_scripts.nbpo_core import uniform_policy
from mnpo_scripts.nbpo_representations import (
    REPRESENTATIONS,
    AdaptiveGameRepresentation,
    BTRewardRepresentation,
    FixedReferenceRepresentation,
    build_representation,
)


def skew_reference(K, X, I, gen):
    B = (torch.rand(K, X, I, I, dtype=torch.float64, generator=gen) - 0.5) * 0.8
    A_ref = 0.5 * (B - B.transpose(-1, -2))
    return A_ref - torch.diag_embed(torch.diagonal(A_ref, dim1=-2, dim2=-1))


@pytest.fixture
def toy():
    gen = torch.Generator().manual_seed(11)
    K, X, I, J = 3, 7, 4, 4
    A = (torch.rand(K, X, I, J, dtype=torch.float64, generator=gen) - 0.5) * 0.8
    A_ref = skew_reference(K, X, I, gen)
    mu = uniform_policy(X, J)
    pi = torch.rand(X, I, dtype=torch.float64, generator=gen)
    pi = pi / pi.sum(-1, keepdim=True)
    return A, A_ref, mu, pi


def test_registry_names():
    assert REPRESENTATIONS == ("adaptive_game", "fixed_reference", "bt_reward")


@pytest.mark.parametrize("beta_big,tol", [(1e4, 1e-5), (1e6, 1e-7)])
def test_fixed_reference_is_the_beta_to_infinity_limit(toy, beta_big, tol):
    """Eq. (7) sends nu* -> mu as beta -> infinity, so q, V and d all converge."""
    A, A_ref, mu, pi = toy
    K = A.shape[0]
    fr = FixedReferenceRepresentation(A, A_ref, mu)
    big = AdaptiveGameRepresentation(A, A_ref, mu,
                                     torch.full((K,), beta_big, dtype=torch.float64))
    nu_fr, q_fr = fr.opponent_and_gradient(pi)
    nu_big, q_big = big.opponent_and_gradient(pi)
    assert (nu_fr - nu_big).abs().max() < tol
    assert (q_fr - q_big).abs().max() < tol
    assert (fr.game_values(pi) - big.game_values(pi)).abs().max() < tol
    assert (fr.disagreement - big.disagreement).abs().max() < tol
    assert (fr.surplus(pi) - big.surplus(pi)).abs().max() < tol


def test_fixed_reference_opponent_is_exactly_mu(toy):
    A, A_ref, mu, pi = toy
    fr = FixedReferenceRepresentation(A, A_ref, mu)
    nu, _ = fr.opponent_and_gradient(pi)
    for k in range(A.shape[0]):
        assert torch.equal(nu[k], mu)


def test_fixed_reference_q_does_not_move_with_the_policy(toy):
    """The whole point of the control: the comparator is frozen."""
    A, A_ref, mu, pi = toy
    fr = FixedReferenceRepresentation(A, A_ref, mu)
    other = uniform_policy(pi.shape[0], pi.shape[1])
    assert torch.equal(fr.opponent_and_gradient(pi)[1], fr.opponent_and_gradient(other)[1])
    assert not fr.policy_adaptive


def test_adaptive_q_does_move_with_the_policy(toy):
    A, A_ref, mu, pi = toy
    ad = AdaptiveGameRepresentation(A, A_ref, mu, torch.full((3,), 0.25, dtype=torch.float64))
    other = uniform_policy(pi.shape[0], pi.shape[1])
    assert (ad.opponent_and_gradient(pi)[1] - ad.opponent_and_gradient(other)[1]).abs().max() > 1e-6
    assert ad.policy_adaptive


def test_disagreement_is_measured_not_zero(toy):
    """d is computed from A_ref. On an exactly skew-symmetric shared pool with
    uniform mu the *measured* value is 0 -- the code must arrive there, not
    assume it -- while at finite beta the adaptive game gives d < 0."""
    A, A_ref, mu, pi = toy
    fr = FixedReferenceRepresentation(A, A_ref, mu)
    assert fr.disagreement.abs().max() < 1e-15          # measured, and exactly zero here
    ad = AdaptiveGameRepresentation(A, A_ref, mu, torch.full((3,), 0.25, dtype=torch.float64))
    assert (ad.disagreement < 0).all()                   # soft-min of a skew game is negative
    # ... and a nonzero reference pool must move it.
    A_ref2 = A_ref * 2.0
    ad2 = AdaptiveGameRepresentation(A, A_ref2, mu, torch.full((3,), 0.25, dtype=torch.float64))
    assert (ad2.disagreement - ad.disagreement).abs().max() > 1e-6


def test_bt_reward_surplus_offset_invariant_and_scale_covariant():
    gen = torch.Generator().manual_seed(3)
    K, X, I, J = 4, 6, 4, 4
    r = torch.randn(K, X, I, dtype=torch.float64, generator=gen)
    r_ref = torch.randn(K, X, J, dtype=torch.float64, generator=gen)
    mu = uniform_policy(X, J)
    pi = torch.rand(X, I, dtype=torch.float64, generator=gen)
    pi = pi / pi.sum(-1, keepdim=True)
    base = BTRewardRepresentation(r, r_ref, mu)
    off = torch.tensor([1.0, -2.0, 0.5, 7.0], dtype=torch.float64).view(K, 1, 1)
    shifted = BTRewardRepresentation(r + off, r_ref + off, mu)
    assert (base.surplus(pi) - shifted.surplus(pi)).abs().max() < 1e-12
    c = torch.tensor([2.0, 0.5, 3.0, 1.5], dtype=torch.float64).view(K, 1, 1)
    scaled = BTRewardRepresentation(r * c, r_ref * c, mu)
    assert (scaled.surplus(pi) - c.view(K) * base.surplus(pi)).abs().max() < 1e-12
    assert not base.has_opponent


def test_build_representation_dispatch_and_rejection(toy):
    A, A_ref, mu, _ = toy
    assert build_representation("fixed_reference", A_policy=A, A_ref=A_ref, mu=mu).type == \
        "fixed_reference"
    with pytest.raises(ValueError, match="representation must be one of"):
        build_representation("nope")


def test_nonuniform_mu_without_mu_learner_is_rejected(toy):
    """d_k = V(mu) needs the SAME mu on both sides; guessing is refused."""
    A, A_ref, mu, _ = toy
    bad = mu.clone()
    bad[0, 0] += 0.1
    bad = bad / bad.sum(-1, keepdim=True)
    with pytest.raises(ValueError, match="same mu on both sides"):
        FixedReferenceRepresentation(A, A_ref, bad)
