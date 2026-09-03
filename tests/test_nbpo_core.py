"""Core-math tests for the paper-exact NBPO module (mnpo_scripts/nbpo_core.py).

Covers spec tests 1 (opponent normalization), 2 (sign), 3 (game-value
identity), 4 (nonzero disagreement), 5 (fixed-reference limit), the core half
of 9 (positive scaling covariance at matched opponents), and 18 (cycle toy
game distinguishing policies with equal fixed-reference margins).
"""
import torch

from mnpo_scripts.nbpo_core import (
    compute_disagreement_point,
    compute_margins,
    compute_objective_gradient,
    compute_regularized_game_value,
    compute_regularized_opponent,
    uniform_policy,
    weighted_policy_update,
)


def make_game(K=2, X=3, I=4, J=4, seed=0, scale=0.4):
    g = torch.Generator().manual_seed(seed)
    A = (torch.rand(K, X, I, J, generator=g, dtype=torch.float64) - 0.5) * 2 * scale
    A_ref = (torch.rand(K, X, J, J, generator=g, dtype=torch.float64) - 0.5) * 2 * scale
    A_ref = (A_ref - A_ref.transpose(-1, -2)) / 2  # skew-symmetric, zero diagonal
    mu = uniform_policy(X, J)
    pi = uniform_policy(X, I)
    beta = torch.full((K,), 0.25, dtype=torch.float64)
    return A, A_ref, mu, pi, beta


def test_opponent_normalization_matches_definition():
    A, _, mu, pi, beta = make_game()
    r = compute_margins(A, pi)
    nu = compute_regularized_opponent(r, mu, beta)
    assert (nu > 0).all()
    assert torch.allclose(nu.sum(dim=-1), torch.ones_like(nu.sum(dim=-1)), atol=1e-12)
    # direct (non-log-space) computation of mu * exp(-r/beta), normalized
    w = mu.unsqueeze(0) * torch.exp(-r / beta.view(-1, 1, 1))
    expected = w / w.sum(dim=-1, keepdim=True)
    assert torch.allclose(nu, expected, atol=1e-12)


def test_opponent_sign_lower_margin_gets_more_mass():
    A, _, mu, pi, beta = make_game()
    r = compute_margins(A, pi)
    nu = compute_regularized_opponent(r, mu, beta)
    for k in range(r.shape[0]):
        for x in range(r.shape[1]):
            order_r = torch.argsort(r[k, x])          # ascending margin
            order_nu = torch.argsort(nu[k, x], descending=True)  # descending mass
            assert torch.equal(order_r, order_nu)


def test_game_value_identity_softmin_equals_primal():
    A, _, mu, pi, beta = make_game(K=3, X=5, seed=7)
    r = compute_margins(A, pi)
    v_soft = compute_regularized_game_value(r, mu, beta, form="softmin")
    v_primal = compute_regularized_game_value(r, mu, beta, form="primal")
    assert torch.allclose(v_soft, v_primal, atol=1e-12)


def test_nonzero_disagreement_two_action_skew_game():
    # A_ref = [[0, a], [-a, 0]]: g(mu, mu) = 0 but V_beta(mu) < 0 for finite beta.
    a = 0.3
    A_ref = torch.tensor([[[[0.0, a], [-a, 0.0]]]], dtype=torch.float64)
    mu = uniform_policy(1, 2)
    beta = torch.tensor([0.25], dtype=torch.float64)
    d = compute_disagreement_point(A_ref, mu, beta)
    assert d.item() < 0.0, "implementation must not replace d by g(mu,mu)=0"
    # analytic: -beta * log(0.5 e^{a/(2 beta)} + 0.5 e^{-a/(2 beta)})
    import math

    expected = -0.25 * math.log(0.5 * math.exp(a / 0.5) + 0.5 * math.exp(-a / 0.5))
    assert abs(d.item() - expected) < 1e-12


def test_fixed_reference_limit_large_beta():
    # For beta -> infinity, V_beta(pi) - V_beta(mu) -> g(pi, mu) (error O(1/beta)).
    A, A_ref, mu, _, _ = make_game(K=2, X=4, seed=3)
    g = torch.Generator().manual_seed(11)
    pi_logits = torch.rand(A.shape[1], A.shape[2], generator=g, dtype=torch.float64)
    pi = torch.softmax(pi_logits, dim=-1)
    beta_large = torch.full((2,), 1e6, dtype=torch.float64)
    v_pi = compute_regularized_game_value(compute_margins(A, pi), mu, beta_large)
    d = compute_disagreement_point(A_ref, mu, beta_large)
    fixed_margin = torch.einsum("xi,kxij,xj->k", pi, A, mu) / A.shape[1]
    assert torch.allclose(v_pi - d, fixed_margin, atol=1e-5)


def test_scaling_covariance_at_matched_opponents():
    # A_k -> c_k A_k with beta_k -> c_k beta_k: nu invariant, V and q scale by c_k,
    # and with lambda_k -> lambda_k / c_k the weighted score sum_k lambda_k q_k
    # is unchanged, so the exponential policy update is identical.
    A, _, mu, pi, beta = make_game(K=3, X=4, seed=5, scale=0.25)
    c = torch.tensor([0.2, 1.0, 1.7], dtype=torch.float64)
    A_s = A * c.view(-1, 1, 1, 1)
    beta_s = beta * c
    r, r_s = compute_margins(A, pi), compute_margins(A_s, pi)
    nu, nu_s = (compute_regularized_opponent(r, mu, beta),
                compute_regularized_opponent(r_s, mu, beta_s))
    assert torch.allclose(nu, nu_s, atol=1e-12)
    V, V_s = (compute_regularized_game_value(r, mu, beta),
              compute_regularized_game_value(r_s, mu, beta_s))
    assert torch.allclose(V * c, V_s, atol=1e-12)
    q, q_s = compute_objective_gradient(A, nu), compute_objective_gradient(A_s, nu_s)
    assert torch.allclose(q * c.view(-1, 1, 1), q_s, atol=1e-12)
    lam = torch.tensor([2.0, 0.7, 1.3], dtype=torch.float64)
    pi_new = weighted_policy_update(pi, q, lam, eta=1.0)
    pi_new_s = weighted_policy_update(pi, q_s, lam / c, eta=1.0)
    assert torch.allclose(pi_new, pi_new_s, atol=1e-12)


def test_cycle_toy_game_adaptive_value_separates_equal_fixed_margins():
    # Rock-paper-scissors payoff with a uniform reference: EVERY policy has
    # fixed-reference margin g(pi, mu) = 0 (each row of A sums to 0), yet the
    # adaptive game value distinguishes the uniform policy (V = 0) from a
    # concentrated one (V < 0: the adaptive opponent exploits it). A scalar
    # fixed-reference margin cannot represent this at all.
    a = 0.4
    rps = torch.tensor([[0.0, a, -a], [-a, 0.0, a], [a, -a, 0.0]], dtype=torch.float64)
    A = rps.view(1, 1, 3, 3)
    mu = uniform_policy(1, 3)
    beta = torch.tensor([0.25], dtype=torch.float64)
    pi_uniform = uniform_policy(1, 3)
    pi_rock = torch.tensor([[0.98, 0.01, 0.01]], dtype=torch.float64)
    margin_u = torch.einsum("xi,kxij,xj->k", pi_uniform, A, mu)
    margin_r = torch.einsum("xi,kxij,xj->k", pi_rock, A, mu)
    assert torch.allclose(margin_u, margin_r, atol=1e-12)  # both exactly zero
    v_u = compute_regularized_game_value(compute_margins(A, pi_uniform), mu, beta)
    v_r = compute_regularized_game_value(compute_margins(A, pi_rock), mu, beta)
    assert abs(v_u.item()) < 1e-12
    assert v_r.item() < v_u.item() - 1e-3
