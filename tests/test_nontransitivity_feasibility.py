"""The exact primal geometry solver: gradients, certificates, and the identity
that ties it to the alternating fixed point it is meant to audit."""
import numpy as np
import pytest
import torch

from mnpo_scripts.nbpo_core import uniform_policy
from mnpo_scripts.nbpo_generic import solve_proximal
from mnpo_scripts.nbpo_representations import AdaptiveGameRepresentation
from scripts.experiments.iclr2027_table1_v2 import nontransitivity_feasibility as nf
from scripts.experiments.iclr2027_table1_v2.controlled_nontransitivity import (
    base_tensor, circulation, mix,
)


def small_instance(seed=0, alpha=0.3, K=3, X=6, I=4, beta=0.25):
    rng = np.random.default_rng(seed)
    base, _ = base_tensor(rng, K, X, I)
    A = mix(base, circulation(rng, K, X, I), alpha)
    mu = np.full((X, I), 1.0 / I)
    b = np.full(K, beta)
    d = nf.game_values(A, mu, mu, b)      # reference-as-learner on the same pool
    return A, mu, b, d


def test_game_value_matches_the_torch_representation():
    """The numpy path must reproduce the audited torch value map exactly."""
    A, mu, beta, _ = small_instance()
    rng = np.random.default_rng(7)
    pi = rng.dirichlet(np.ones(A.shape[2]), size=A.shape[1])
    At = torch.from_numpy(A)
    rep = AdaptiveGameRepresentation(At, At, torch.from_numpy(mu),
                                     torch.from_numpy(beta))
    got = nf.game_values(A, pi, mu, beta)
    want = rep.game_values(torch.from_numpy(pi)).numpy()
    assert np.allclose(got, want, atol=1e-13, rtol=0)


def test_value_gradient_matches_finite_differences():
    A, mu, beta, _ = small_instance()
    rng = np.random.default_rng(11)
    pi = rng.dirichlet(np.ones(A.shape[2]), size=A.shape[1])
    g = nf.value_gradient(A, pi, mu, beta)
    eps = 1e-7
    for k in (0, 2):
        for x in (0, 3):
            for i in (1, 2):
                p = pi.copy()
                p[x, i] += eps
                hi = nf.game_values(A, p, mu, beta)[k]
                p[x, i] -= 2 * eps
                lo = nf.game_values(A, p, mu, beta)[k]
                assert abs((hi - lo) / (2 * eps) - g[k, x, i]) < 1e-6


def test_game_value_is_concave_on_random_segments():
    """Concavity is the reason an exact solver exists at all, so it is checked."""
    A, mu, beta, _ = small_instance()
    rng = np.random.default_rng(3)
    for _ in range(20):
        p = rng.dirichlet(np.ones(A.shape[2]), size=A.shape[1])
        q = rng.dirichlet(np.ones(A.shape[2]), size=A.shape[1])
        t = rng.uniform(0.1, 0.9)
        mid = nf.game_values(A, t * p + (1 - t) * q, mu, beta)
        ends = t * nf.game_values(A, p, mu, beta) + (1 - t) * nf.game_values(A, q, mu, beta)
        assert (mid >= ends - 1e-12).all()


def test_certificate_is_an_upper_bound_at_arbitrary_points():
    A, mu, beta, d = small_instance()
    rng = np.random.default_rng(5)
    best = -np.inf
    for _ in range(200):
        pi = rng.dirichlet(np.ones(A.shape[2]) * rng.uniform(0.2, 3), size=A.shape[1])
        best = max(best, float((nf.game_values(A, pi, mu, beta) - d).min()))
    pi0 = np.full((A.shape[1], A.shape[2]), 1.0 / A.shape[2])
    ub, _ = nf.certify_max_min(A, pi0, mu, beta, d)
    assert ub >= best - 1e-9


def test_max_min_solution_is_certified_tight():
    A, mu, beta, d = small_instance(alpha=0.2)
    r = nf.solve_max_min(A, mu, beta, d)
    assert r["rho_star_gap"] < 1e-4
    assert r["primal_feasibility_residual"] < 1e-8


def test_exact_proximal_equals_the_converged_fixed_point():
    """The identity that makes this an audit rather than a second opinion.

    pi ~ pi_t exp(eta sum_k w_k q_k(pi)) is exactly the stationarity condition of
    the concave program  max_pi sum_k w_k s_k(pi) - D(pi||pi_t)/eta,  so a
    converged alternating fixed point and the direct primal maximizer must be the
    same policy. Any disagreement means one of them is wrong.
    """
    A, mu, beta, d = small_instance(alpha=0.4, X=5, I=4, K=3)
    At = torch.from_numpy(A)
    rep = AdaptiveGameRepresentation(At, At, torch.from_numpy(mu),
                                     torch.from_numpy(beta))
    pi_ref = np.full((A.shape[1], A.shape[2]), 1.0 / A.shape[2])
    w = np.array([0.7, 1.3, 1.0])
    eta = 1.0
    converged = solve_proximal(rep, uniform_policy(A.shape[1], A.shape[2]),
                               torch.from_numpy(w), eta, R=400, damping=0.5)
    assert converged.extra_map_residual < 1e-10
    exact, _ = nf.solve_weighted(A, mu, beta, d, w, pi0=converged.pi.numpy(),
                                 eta=eta, pi_ref=pi_ref)
    assert nf.total_variation(exact, converged.pi.numpy()) < 1e-6


def test_nash_refuses_an_infeasible_start_instead_of_clamping():
    A, mu, beta, d = small_instance(alpha=1.0)
    pi_ref = np.full((A.shape[1], A.shape[2]), 1.0 / A.shape[2])
    with pytest.raises(nf.NashInfeasible):
        nf.solve_nash(A, mu, beta, d, pi0=pi_ref)


def test_uniform_reference_has_exactly_zero_surplus():
    """s_k(mu) = V_k(mu) - d_k = 0 by construction of the disagreement point."""
    A, mu, beta, d = small_instance()
    s = nf.game_values(A, mu, mu, beta) - d
    assert np.abs(s).max() < 1e-14
