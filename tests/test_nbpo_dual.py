"""Dual-solver tests (mnpo_scripts/nbpo_solver.py).

Covers spec tests 6 (dual finite-difference), 7 (KKT convergence on a jointly
feasible game), 8 (raw vs normalized lambda), the KKT half of 9 (scaling
covariance at the converged point), and the max-min control residual.
"""
import pytest
import torch

from mnpo_scripts.nbpo_core import (
    compute_margins,
    compute_objective_gradient,
    compute_regularized_opponent,
    uniform_policy,
    weighted_policy_update,
)
from mnpo_scripts.nbpo_solver import (
    dual_objective_phi,
    policy_game_values,
    solve_nbpo_dual,
    solve_weighted_policy,
)


def feasible_game(K=2, X=4, I=4, J=4, seed=0, amplitude=1.0):
    """Jointly improvable game: both objectives prefer low-index responses.

    ``amplitude`` shrinks all payoffs so that a scale test's c_k * A_k stays a
    valid centered payoff in [-1/2, 1/2].
    """
    g = torch.Generator().manual_seed(seed)
    strength = torch.tensor([0.35, 0.15, -0.15, -0.35], dtype=torch.float64) * amplitude
    noise = (torch.rand(K, X, I, J, generator=g, dtype=torch.float64) - 0.5) * 0.1 * amplitude
    A = (strength.view(1, 1, -1, 1) + noise).clamp(-0.45, 0.45)
    A_ref = (torch.rand(K, X, J, J, generator=g, dtype=torch.float64) - 0.5) * 0.2 * amplitude
    A_ref = (A_ref - A_ref.transpose(-1, -2)) / 2
    mu = uniform_policy(X, J)
    beta = torch.full((K,), 0.25, dtype=torch.float64)
    return A, A_ref, mu, beta


def test_dual_finite_difference_matches_surplus_minus_inverse_lambda():
    A, A_ref, mu, beta = feasible_game()
    lam0 = torch.tensor([1.5, 3.0], dtype=torch.float64)  # O(1)-O(10): contraction regime
    eps = 1e-6
    # the inner solve is verified converged (residual tol) inside dual_objective_phi
    from mnpo_scripts.nbpo_core import compute_disagreement_point

    d = compute_disagreement_point(A_ref, mu, beta)
    sol = solve_weighted_policy(A, mu, uniform_policy(4, 4), lam0, beta, 1.0, R=400, damping=0.5)
    assert sol.fixed_point_residual < 1e-10
    s = policy_game_values(A, mu, sol.pi, beta) - d
    for k in range(2):
        lp, lm = lam0.clone(), lam0.clone()
        lp[k] += eps
        lm[k] -= eps
        fd = (dual_objective_phi(A, A_ref, mu, beta, 1.0, lp)
              - dual_objective_phi(A, A_ref, mu, beta, 1.0, lm)) / (2 * eps)
        analytic = float(s[k] - 1.0 / lam0[k])  # Eq. (19)
        assert abs(fd - analytic) < 1e-5, (k, fd, analytic)


def test_kkt_convergence_on_jointly_feasible_game():
    A, A_ref, mu, beta = feasible_game()
    res = solve_nbpo_dual(A, A_ref, mu, beta, eta=1.0, gamma=0.5, M=6000, R=1,
                          lambda_box=(1e-3, 1e3), damping=0.5)
    assert (res.surplus > 0).all(), "constructed game should be jointly improvable"
    assert res.kkt_residual < 1e-4, res.kkt_residual
    # KKT identity lambda_k = 1 / s_k (Eq. (20)) holds at the converged point
    assert torch.allclose(res.lam, 1.0 / res.surplus, rtol=1e-2)


def test_raw_vs_normalized_lambda_changes_the_update():
    A, _, mu, beta = feasible_game()
    pi_t = uniform_policy(4, 4)
    lam = torch.tensor([4.0, 1.0], dtype=torch.float64)
    r = compute_margins(A, pi_t)
    nu = compute_regularized_opponent(r, mu, beta)
    q = compute_objective_gradient(A, nu)
    pi_raw = weighted_policy_update(pi_t, q, lam, eta=1.0)
    lam_norm = lam / lam.sum()
    pi_norm = weighted_policy_update(pi_t, q, lam_norm, eta=1.0)
    assert not torch.allclose(pi_raw, pi_norm, atol=1e-6), \
        "normalizing lambda must change the exponential update"
    pi_rescaled = weighted_policy_update(pi_t, q, lam_norm, eta=float(lam.sum()))
    assert torch.allclose(pi_raw, pi_rescaled, atol=1e-12), \
        "...unless eta is rescaled by the normalizer"
    # the Nash path itself keeps lambda raw: the solver result is not simplex-normalized
    _, A_ref, _, _ = feasible_game()
    res = solve_nbpo_dual(A, A_ref, mu, beta, eta=1.0, gamma=0.5, M=2000, R=1, damping=0.5)
    assert abs(float(res.lam.sum()) - 1.0) > 1e-3, "reproduction mode must keep raw lambda"


def test_scaling_covariance_at_the_kkt_point():
    # Under A_k -> c_k A_k, beta_k -> c_k beta_k: V, d, s scale by c_k and the
    # converged lambda scales by 1/c_k (KKT: lambda = 1/s). Asserted at the
    # converged point, not along the (non-covariant) dual trajectory.
    A, A_ref, mu, beta = feasible_game(seed=2, amplitude=0.3)  # c*A stays in [-1/2, 1/2]
    c = torch.tensor([3.0, 0.5], dtype=torch.float64)
    kwargs = dict(eta=1.0, gamma=0.3, M=8000, R=1, damping=0.5)
    res = solve_nbpo_dual(A, A_ref, mu, beta, **kwargs)
    res_s = solve_nbpo_dual(A * c.view(-1, 1, 1, 1), A_ref * c.view(-1, 1, 1, 1),
                            mu, beta * c, **kwargs)
    assert res.kkt_residual < 1e-3 and res_s.kkt_residual < 1e-3
    assert torch.allclose(res_s.V, res.V * c, rtol=1e-3, atol=1e-6)
    assert torch.allclose(res_s.d, res.d * c, rtol=1e-3, atol=1e-6)
    assert torch.allclose(res_s.surplus, res.surplus * c, rtol=1e-3, atol=1e-6)
    assert torch.allclose(res_s.lam, res.lam / c, rtol=2e-2)
    # opponents invariant up to the two runs' dual-convergence residuals
    # (exact invariance at matched policies is asserted in test_nbpo_core)
    assert torch.allclose(res_s.nu, res.nu, atol=1e-4)


def test_maxmin_controls_report_small_duality_gap_and_are_not_static_onehot():
    A, A_ref, mu, beta = feasible_game(seed=4)
    res = solve_nbpo_dual(A, A_ref, mu, beta, eta=1.0, gamma=0.5, M=1500, R=1,
                          aggregation="surplus_maxmin", damping=0.5, adversary_step=0.5)
    assert res.control_residual is not None and res.control_residual < 5e-3
    assert res.kkt_residual is None  # KKT residual is a Nash-path diagnostic
    # the adversarial weights live on the simplex but were not fixed a priori
    assert abs(float(res.lam.sum()) - 1.0) < 1e-9


def test_scale_covariance_with_payoffs_outside_probability_range():
    # A_k -> c A_k with c = 5 leaves the probability range; the solver must accept
    # the rescaled UTILITIES (only raw artifacts are range-checked) and be exactly
    # covariant once beta_k -> c beta_k: nu invariant, V/d scale by c, lambda by
    # 1/c, sum_k lambda_k q_k invariant, selected policy and pair orientation invariant.
    from mnpo_scripts.nbpo_core import validate_centered_preference_tensor
    A, A_ref, mu, beta = feasible_game(seed=2, amplitude=1.0)
    c = 5.0
    assert float((c * A).abs().max()) > 0.5
    with pytest.raises(ValueError, match="centered"):
        validate_centered_preference_tensor(c * A)          # raw check rejects it ...
    kwargs = dict(eta=1.0, gamma=0.3, M=6000, R=2, damping=0.5)
    res = solve_nbpo_dual(A, A_ref, mu, beta, **kwargs)
    res_s = solve_nbpo_dual(c * A, c * A_ref, mu, beta * c, **kwargs)   # ... the solver accepts it
    assert res.kkt_residual < 1e-4 and res_s.kkt_residual < 1e-4
    assert torch.allclose(res_s.V, res.V * c, rtol=1e-4, atol=1e-8)
    assert torch.allclose(res_s.d, res.d * c, rtol=1e-4, atol=1e-8)
    assert torch.allclose(res_s.surplus, res.surplus * c, rtol=1e-4, atol=1e-8)
    assert torch.allclose(res_s.lam, res.lam / c, rtol=5e-3)
    assert torch.allclose(res_s.nu_final_policy, res.nu_final_policy, atol=1e-5)
    score = torch.einsum("k,kxi->xi", res.lam, res.q_update)
    score_s = torch.einsum("k,kxi->xi", res_s.lam, res_s.q_update)
    assert torch.allclose(score, score_s, atol=1e-4)                      # sum_k lambda_k q_k
    assert torch.equal(res.pi.argmax(-1), res_s.pi.argmax(-1))            # selected policy
    # pair orientation: sign of the weighted-score difference for every unordered pair
    d1 = score[:, :, None] - score[:, None, :]
    d2 = score_s[:, :, None] - score_s[:, None, :]
    mask = d1.abs() > 1e-6
    assert torch.equal(torch.sign(d1[mask]), torch.sign(d2[mask]))


def test_nu_update_and_nu_final_are_labeled_separately():
    A, A_ref, mu, beta = feasible_game(seed=3)
    g = torch.Generator().manual_seed(9)
    pi_init = torch.softmax(torch.rand(4, 4, generator=g, dtype=torch.float64) * 3, dim=-1)
    lam = torch.tensor([3.0, 4.0], dtype=torch.float64)
    sol = solve_weighted_policy(A, mu, uniform_policy(4, 4), lam, beta, 1.0, R=1, pi_init=pi_init)
    # R=1: the opponent that generated pi (from pi_init) is not the opponent at pi
    assert not torch.allclose(sol.nu_update, sol.nu_final_policy, atol=1e-6)
    assert torch.allclose(sol.nu_update, compute_regularized_opponent(
        compute_margins(A, pi_init), mu, beta), atol=1e-12)
    assert torch.allclose(sol.nu_final_policy, compute_regularized_opponent(
        compute_margins(A, sol.pi), mu, beta), atol=1e-12)
    assert sol.extra_map_residual > 1e-6             # not a fixed point after one map
    assert sol.nu is sol.nu_update                   # alias points at the UPDATE opponent
    res = solve_nbpo_dual(A, A_ref, mu, beta, eta=1.0, gamma=0.3, M=50, R=1)
    assert hasattr(res, "nu_update") and hasattr(res, "nu_final_policy")
    assert res.opponent_entropy.shape == (2,)        # computed from nu_final_policy


def test_projected_kkt_residual_at_lambda_box_boundary():
    # Objective 1 is infeasible (every payoff -0.3, so s_1 < 0 for every policy):
    # the dual pushes lambda_1 to the upper bound, where lambda = 1/s cannot hold.
    A, A_ref, mu, beta = feasible_game(seed=5)
    A = A.clone(); A[1] = -0.3
    hi = 10.0
    res = solve_nbpo_dual(A, A_ref, mu, beta, eta=1.0, gamma=0.5, M=3000, R=1,
                          lambda_box=(1e-3, hi), damping=0.3)
    assert res.lambda_at_upper_bound == [1]
    assert abs(float(res.lam[1]) - hi) < 1e-9
    assert res.surplus[1] < 0
    assert res.inverse_surplus_residual > 0.1          # ||s - 1/lambda|| is NOT small here ...
    assert res.projected_kkt_residual < 1e-6           # ... but the projected residual is
    assert res.gamma_ref == 0.5
