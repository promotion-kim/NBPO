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


# --------------------------------------------------------------------------
# The exact inner solver, ported into the production stack.
# --------------------------------------------------------------------------

def _rep(A, mu, beta):
    At = torch.from_numpy(A)
    return AdaptiveGameRepresentation(At, At, torch.from_numpy(mu),
                                      torch.from_numpy(beta))


def test_exact_inner_solve_matches_the_converged_fixed_point_where_one_exists():
    """At weights small enough for the map to contract, both routes must agree."""
    from mnpo_scripts.nbpo_generic import solve_proximal_exact
    A, mu, beta, _ = small_instance(alpha=0.3, K=3, X=5, I=4)
    rep = _rep(A, mu, beta)
    pi_t = uniform_policy(A.shape[1], A.shape[2])
    w = torch.tensor([0.8, 1.1, 0.6], dtype=torch.float64)
    it = solve_proximal(rep, pi_t, w, 1.0, R=400, damping=0.5)
    ex = solve_proximal_exact(rep, pi_t, w, 1.0)
    assert it.extra_map_residual < 1e-10
    assert nf.total_variation(ex.pi.numpy(), it.pi.numpy()) < 1e-6


def test_exact_inner_solve_preserves_the_target_log_ratio_identity():
    """The Eq. (26) identity must hold to float64, not merely to optimizer tolerance.

    `write_generic_solution_artifact` refuses above 1e-9, and a raw optimizer
    iterate only satisfies the identity to its own stationarity tolerance. The
    solver therefore returns the Eq. (21) map APPLIED AT the maximizer, which
    makes the identity exact by construction.
    """
    from mnpo_scripts.nbpo_generic import solve_proximal_exact
    A, mu, beta, _ = small_instance(alpha=1.0, K=4, X=6, I=4)
    rep = _rep(A, mu, beta)
    pi_t = uniform_policy(A.shape[1], A.shape[2])
    w = torch.tensor([120.0, 80.0, 200.0, 60.0], dtype=torch.float64)   # raw Nash scale
    ex = solve_proximal_exact(rep, pi_t, w, 1.0)
    lr = torch.log(ex.pi) - torch.log(pi_t)
    score = 1.0 * torch.einsum("k,kxi->xi", w, ex.q_update)
    diff = lr - score
    identity = float((diff - diff.mean(dim=-1, keepdim=True)).abs().max())
    assert identity < 1e-9, identity


def test_exact_inner_solve_beats_the_r_step_map_where_the_map_diverges():
    """The audit's finding, pinned as a test at production scale-of-weights."""
    from mnpo_scripts.nbpo_generic import solve_finite_pool
    A, mu, beta, d = small_instance(alpha=1.0, K=4, X=8, I=4)
    rep = _rep(A, mu, beta)
    fp = solve_finite_pool(rep, "nash", eta=1.0, M=300, R=3,
                           inner_solver="fixed_point")
    ex = solve_finite_pool(rep, "nash", eta=1.0, M=300, R=3, inner_solver="exact")
    assert fp.fixed_point_residual > 0.5          # the map is not converging
    assert ex.fixed_point_residual < 1e-4         # the exact solve is
    assert float(ex.surplus.min()) > float(fp.surplus.min())
    assert ex.target_log_ratio_check() < 1e-9
    assert ex.config["aggregation"] == "nash"
    assert ex.config["inner_solver"] == "exact"


def test_exact_inner_solver_is_a_no_op_for_a_constant_map_representation():
    """fixed_reference and bt_reward already land on their fixed point in one step."""
    from mnpo_scripts.nbpo_generic import solve_proximal_exact
    from mnpo_scripts.nbpo_representations import FixedReferenceRepresentation
    A, mu, beta, _ = small_instance(alpha=0.6, K=3, X=5, I=4)
    At = torch.from_numpy(A)
    rep = FixedReferenceRepresentation(At, At, torch.from_numpy(mu))
    pi_t = uniform_policy(A.shape[1], A.shape[2])
    w = torch.tensor([2.0, 0.5, 1.0], dtype=torch.float64)
    a = solve_proximal(rep, pi_t, w, 1.0, R=1)
    b = solve_proximal_exact(rep, pi_t, w, 1.0)
    assert torch.equal(a.pi, b.pi)


def test_default_inner_solver_leaves_every_existing_result_untouched():
    from mnpo_scripts.nbpo_generic import solve_finite_pool
    A, mu, beta, _ = small_instance(alpha=0.4, K=3, X=5, I=4)
    rep = _rep(A, mu, beta)
    a = solve_finite_pool(rep, "nash", eta=1.0, M=50, R=3)
    b = solve_finite_pool(rep, "nash", eta=1.0, M=50, R=3, inner_solver="fixed_point")
    assert torch.equal(a.pi, b.pi)
    assert torch.equal(a.weights, b.weights)


def test_unknown_inner_solver_is_refused():
    from mnpo_scripts.nbpo_generic import solve_finite_pool
    A, mu, beta, _ = small_instance()
    with pytest.raises(ValueError, match="inner_solver"):
        solve_finite_pool(_rep(A, mu, beta), "nash", eta=1.0, M=2, R=1,
                          inner_solver="magic")


def test_separable_and_joint_exact_solvers_agree():
    """Two independent implementations of the same concave program.

    The audit module solves it jointly over all X*I variables; the production
    solver exploits the fact that it separates across prompts and solves X small
    problems. Agreement is the cross-check that neither is subtly wrong -- and
    the separable form is the one that can run at 7000 prompts.
    """
    from mnpo_scripts.nbpo_generic import solve_proximal_exact
    A, mu, beta, d = small_instance(alpha=0.8, K=4, X=7, I=4)
    rep = _rep(A, mu, beta)
    pi_t = uniform_policy(A.shape[1], A.shape[2])
    w = np.array([40.0, 12.0, 25.0, 60.0])
    joint, _ = nf.solve_weighted(A, mu, beta, d, w, eta=1.0, pi_ref=mu)
    sep = solve_proximal_exact(rep, pi_t, torch.from_numpy(w), 1.0)
    assert nf.total_variation(sep.update_source_pi.numpy(), joint) < 1e-5


def test_exact_inner_solve_cost_is_linear_in_the_number_of_prompts():
    """The property that makes it usable at scale, asserted rather than claimed."""
    import time
    from mnpo_scripts.nbpo_generic import solve_proximal_exact
    timings = []
    for X in (10, 40):
        A, mu, beta, _ = small_instance(alpha=0.7, K=4, X=X, I=4)
        rep = _rep(A, mu, beta)
        pi_t = uniform_policy(X, 4)
        w = torch.tensor([30.0, 30.0, 30.0, 30.0], dtype=torch.float64)
        solve_proximal_exact(rep, pi_t, w, 1.0)          # warm the imports
        t = time.time()
        solve_proximal_exact(rep, pi_t, w, 1.0)
        timings.append(time.time() - t)
    # 4x the prompts must not cost more than ~8x: linear, with slack for noise
    assert timings[1] < 8 * timings[0] + 0.5
