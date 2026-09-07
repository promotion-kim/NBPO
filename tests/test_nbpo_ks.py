"""Kalai--Smorodinsky: the properties that make it KS and not a renamed max-min.

The manuscript's weak baseline was raw surplus max-min.  KS differs from it by
normalizing each surplus by that objective's *ideal* -- the best it could do on
its own while everyone stays individually rational -- and by refining the
resulting flat face for Pareto efficiency.  These tests pin exactly those two
things, plus the invariances the rule is supposed to have, and check the solved
``rho*`` against an independent brute-force search over the same attainable set.
"""
import pytest
import torch

from mnpo_scripts.nbpo_core import uniform_policy
from mnpo_scripts.nbpo_generic import (
    KSUndefinedError,
    matched_weight_l1,
    solve_finite_pool,
    solve_kalai_smorodinsky,
    solve_proximal,
)
from mnpo_scripts.nbpo_representations import (
    AdaptiveGameRepresentation,
    BTRewardRepresentation,
)
from tests.test_nbpo_generic_solver import make_pool

KS_KW = dict(stage1_iters=400, ideal_iters=50)


@pytest.fixture(scope="module")
def adaptive():
    A, A_ref, mu = make_pool(seed=7, K=3, X=64)
    return AdaptiveGameRepresentation(A, A_ref, mu,
                                      torch.full((3,), 0.25, dtype=torch.float64))


@pytest.fixture(scope="module")
def matched_l1(adaptive):
    return matched_weight_l1(solve_finite_pool(adaptive, "nash", eta=1.0, R=3,
                                               M=400, gamma=0.5))


@pytest.fixture(scope="module")
def ks(adaptive, matched_l1):
    return solve_kalai_smorodinsky(adaptive, uniform_policy(adaptive.X, adaptive.I),
                                   eta=1.0, R=3, weight_l1=matched_l1, **KS_KW)


# --- the defining property --------------------------------------------------

def test_ks_equalizes_the_ideal_normalized_surpluses(ks):
    """The KS point sits on the ray from d to the ideal: all s_k/u_k are equal."""
    v = ks.normalized_surplus
    assert float(v.max() - v.min()) < 5e-3, f"normalized surpluses not equalized: {v}"
    assert ks.rho_star == pytest.approx(float(v.min()), abs=1e-3)


def test_ideal_point_dominates_the_achieved_surplus(ks):
    """u_k is an ideal: no objective may exceed its own ideal at the KS point."""
    assert (ks.surplus <= ks.u_ideal + 1e-9).all()
    assert (ks.u_ideal > 0).all()
    assert 0.0 < ks.rho_star < 1.0


def test_individual_rationality_is_checked_and_reported(ks):
    """s_k >= 0 for every objective, and the violation is reported either way."""
    assert ks.ir_violation == pytest.approx(0.0, abs=1e-12)
    assert (ks.surplus >= -1e-12).all()
    assert ks.ideal_feasible is True


def test_stage1_and_stage2_residuals_are_within_the_declared_tolerance(ks):
    assert ks.stage1_residual <= 1e-4, ks.stage1_residual
    assert ks.stage2_gain >= 0.0
    assert ks.diagnostics["tolerance"] == 1e-4


def test_stage3_reports_the_achieved_proximal_divergence(ks):
    assert ks.stage3_kl > 0.0 and torch.isfinite(torch.tensor(ks.stage3_kl))


def test_equalization_spread_is_reported_separately_from_the_stage1_residual(ks):
    """Two different things that both look like "how well did KS do".

    The Stage-1 residual can be zero -- or negative, when the Pareto tilt beats
    the saddle-point average -- while the normalized surpluses are still spread
    out from an under-converged solve. The artifact reports both so neither can
    stand in for the other.
    """
    spread = ks.diagnostics["normalized_surplus_spread"]
    v = ks.normalized_surplus
    assert spread == pytest.approx(float(v.max() - v.min()), abs=1e-12)
    assert spread >= 0.0
    assert "NEGATIVE means" in ks.diagnostics["stage1_residual_sign_note"]


# --- independent reference implementation -----------------------------------

def _brute_force_rho(rep, pi_t, eta, R, u, l1, steps=24):
    """Grid search over the weight simplex: an implementation that shares no
    code path with the Hedge saddle point the solver uses."""
    K = rep.K
    assert K == 3, "grid written for K = 3"
    best = -float("inf")
    best_w = None
    for a in range(steps + 1):
        for b in range(steps + 1 - a):
            c = steps - a - b
            w = torch.tensor([a, b, c], dtype=torch.float64) / steps
            if float(w.sum()) == 0:
                continue
            s = rep.surplus(solve_proximal(rep, pi_t, w * l1, eta, R).pi)
            val = float((s / u).min())
            if val > best:
                best, best_w = val, w
    return best, best_w


def test_rho_star_matches_an_independent_grid_search(adaptive, ks, matched_l1):
    pi_t = uniform_policy(adaptive.X, adaptive.I)
    grid_rho, grid_w = _brute_force_rho(adaptive, pi_t, 1.0, 3, ks.u_ideal, matched_l1)
    # The solver must not be beaten by the grid (up to the grid's own coarseness).
    assert ks.rho_star >= grid_rho - 2e-3, (ks.rho_star, grid_rho, grid_w)
    # ... and it must actually attain what it claims.
    assert float((ks.surplus / ks.u_ideal).min()) >= grid_rho - 2e-3


# --- KS vs raw max-min ------------------------------------------------------

def _scale_mismatched_rep(c_small=0.02):
    """Two objectives with wildly different attainable ranges.

    Raw surplus max-min equalizes the *raw* surpluses, which drags the
    wide-range objective down to the narrow one's scale; KS equalizes the
    normalized ones.  This is the case where calling raw max-min "KS" would be
    wrong, so the two must not coincide.
    """
    gen = torch.Generator().manual_seed(21)
    K, X, I, J = 2, 40, 4, 4
    r = torch.randn(K, X, I, dtype=torch.float64, generator=gen)
    r_ref = torch.randn(K, X, J, dtype=torch.float64, generator=gen)
    r[0] *= c_small
    r_ref[0] *= c_small
    return BTRewardRepresentation(r, r_ref, uniform_policy(X, J))


def test_ks_is_not_the_raw_surplus_maxmin_solution():
    rep = _scale_mismatched_rep()
    pi_t = uniform_policy(rep.X, rep.I)
    l1 = matched_weight_l1(solve_finite_pool(rep, "nash", eta=1.0, R=1, M=400, gamma=0.5))
    ks = solve_kalai_smorodinsky(rep, pi_t, eta=1.0, R=1, weight_l1=l1, **KS_KW)
    mm = solve_finite_pool(rep, "surplus_maxmin", eta=1.0, R=1, M=400, weight_l1=l1)
    # different policies ...
    assert float((ks.pi - mm.pi).abs().max()) > 1e-3
    # ... and KS is strictly better on its own (normalized) criterion.
    ks_norm = float((ks.surplus / ks.u_ideal).min())
    mm_norm = float((mm.surplus / ks.u_ideal).min())
    assert ks_norm > mm_norm + 1e-3, (ks_norm, mm_norm)


def test_ks_is_not_pareto_dominated_by_the_maxmin_point():
    """A Pareto-dominated answer would be strictly worse on every objective."""
    rep = _scale_mismatched_rep()
    pi_t = uniform_policy(rep.X, rep.I)
    l1 = matched_weight_l1(solve_finite_pool(rep, "nash", eta=1.0, R=1, M=400, gamma=0.5))
    ks = solve_kalai_smorodinsky(rep, pi_t, eta=1.0, R=1, weight_l1=l1, **KS_KW)
    mm = solve_finite_pool(rep, "surplus_maxmin", eta=1.0, R=1, M=400, weight_l1=l1)
    assert not bool((mm.surplus > ks.surplus + 1e-12).all())


# --- invariances ------------------------------------------------------------

def test_offsetting_every_objective_leaves_the_solution_unchanged():
    """s = V - d is offset-free, so a constant added to the objective scale
    cannot move the bargaining solution."""
    gen = torch.Generator().manual_seed(5)
    K, X, I, J = 3, 40, 4, 4
    r = torch.randn(K, X, I, dtype=torch.float64, generator=gen)
    r_ref = torch.randn(K, X, J, dtype=torch.float64, generator=gen)
    mu = uniform_policy(X, J)
    pi_t = uniform_policy(X, I)
    base = BTRewardRepresentation(r, r_ref, mu)
    off = torch.tensor([3.0, -1.0, 10.0], dtype=torch.float64).view(K, 1, 1)
    shifted = BTRewardRepresentation(r + off, r_ref + off, mu)
    l1 = matched_weight_l1(solve_finite_pool(base, "nash", eta=1.0, R=1, M=400, gamma=0.5))
    a = solve_kalai_smorodinsky(base, pi_t, 1.0, 1, weight_l1=l1, **KS_KW)
    b = solve_kalai_smorodinsky(shifted, pi_t, 1.0, 1, weight_l1=l1, **KS_KW)
    assert float((a.pi - b.pi).abs().max()) < 1e-9
    assert float((a.u_ideal - b.u_ideal).abs().max()) < 1e-9
    assert a.rho_star == pytest.approx(b.rho_star, abs=1e-9)


def test_uniform_rescaling_with_a_compensating_weight_budget_is_exact():
    """Scaling every objective by c and the weight budget by 1/c reproduces the
    same policy exactly, and rho* (a ratio) is untouched."""
    gen = torch.Generator().manual_seed(6)
    K, X, I, J = 3, 40, 4, 4
    r = torch.randn(K, X, I, dtype=torch.float64, generator=gen)
    r_ref = torch.randn(K, X, J, dtype=torch.float64, generator=gen)
    mu, pi_t = uniform_policy(X, J), uniform_policy(X, I)
    c = 4.0
    base = BTRewardRepresentation(r, r_ref, mu)
    scaled = BTRewardRepresentation(r * c, r_ref * c, mu)
    a = solve_kalai_smorodinsky(base, pi_t, 1.0, 1, weight_l1=2.0, **KS_KW)
    b = solve_kalai_smorodinsky(scaled, pi_t, 1.0, 1, weight_l1=2.0 / c, **KS_KW)
    assert float((a.pi - b.pi).abs().max()) < 1e-9
    assert a.rho_star == pytest.approx(b.rho_star, abs=1e-9)
    assert float((b.u_ideal - c * a.u_ideal).abs().max()) < 1e-9


def test_the_ks_criterion_itself_is_independent_scale_invariant():
    """``s_k / u_k`` is invariant under *independent* positive rescaling, which
    is the substance of KS's scale invariance: the policy that equalizes the
    normalized surpluses before the rescaling still equalizes them after.

    Note what is NOT claimed: inside a *fixed-norm* proximal family, rescaling
    objective k by c_k also reshapes which policies are reachable at a given
    weight budget, so the solved policy is not invariant. The criterion is; the
    attainable set is not. The test states the true one.
    """
    gen = torch.Generator().manual_seed(8)
    K, X, I, J = 3, 40, 4, 4
    r = torch.randn(K, X, I, dtype=torch.float64, generator=gen)
    r_ref = torch.randn(K, X, J, dtype=torch.float64, generator=gen)
    mu, pi_t = uniform_policy(X, J), uniform_policy(X, I)
    base = BTRewardRepresentation(r, r_ref, mu)
    ks = solve_kalai_smorodinsky(base, pi_t, 1.0, 1, weight_l1=2.0, **KS_KW)
    c = torch.tensor([0.25, 3.0, 1.0], dtype=torch.float64)
    scaled = BTRewardRepresentation(r * c.view(K, 1, 1), r_ref * c.view(K, 1, 1), mu)
    s_scaled = scaled.surplus(ks.pi)
    u_scaled = c * ks.u_ideal
    assert float((s_scaled / u_scaled - ks.normalized_surplus).abs().max()) < 1e-9


def test_symmetric_game_gives_symmetric_weights():
    """Two objectives that are exchangeable under a permutation of the response
    pool must receive equal weight and equal normalized surplus.

    ``r_0 = (a, b, c, c)``, ``r_1 = (b, a, c, c)``: swapping responses 0 and 1
    maps objective 0 onto objective 1 and leaves the uniform proximal centre
    fixed, so the whole bargaining problem is invariant under that swap.  The
    conflict is real (raising objective 0 means moving mass onto response 0,
    which lowers objective 1), and the solution is not the proximal centre.
    """
    X, I, J = 30, 4, 4
    gen = torch.Generator().manual_seed(4)
    a = torch.randn(X, dtype=torch.float64, generator=gen)
    b = torch.randn(X, dtype=torch.float64, generator=gen)
    c = torch.randn(X, dtype=torch.float64, generator=gen)
    r = torch.stack([torch.stack([a, b, c, c], dim=1),
                     torch.stack([b, a, c, c], dim=1)])
    pa = torch.randn(X, dtype=torch.float64, generator=gen)
    pb = torch.randn(X, dtype=torch.float64, generator=gen)
    pc = torch.randn(X, dtype=torch.float64, generator=gen)
    r_ref = torch.stack([torch.stack([pa, pb, pc, pc], dim=1),
                         torch.stack([pb, pa, pc, pc], dim=1)])
    rep = BTRewardRepresentation(r, r_ref, uniform_policy(X, J))
    pi_t = uniform_policy(X, I)
    ks = solve_kalai_smorodinsky(rep, pi_t, 1.0, 1, weight_l1=2.0, **KS_KW)
    assert ks.ideal_feasible, ks.diagnostics["ideal_point_definitions"]
    assert abs(float(ks.u_ideal[0] - ks.u_ideal[1])) < 1e-6      # symmetric ideal point
    assert abs(float(ks.weights[0] - ks.weights[1])) < 2e-2
    assert abs(float(ks.normalized_surplus[0] - ks.normalized_surplus[1])) < 5e-3
    assert float((ks.pi - pi_t).abs().max()) > 1e-3               # not the trivial answer


def test_zero_sum_pool_reports_that_the_ideal_point_is_not_individually_rational():
    """``s_1 = -s_0`` exactly: the only individually rational point is s = 0, so
    the bargaining set above the disagreement point is a single point.

    The solver must refuse rather than normalize by a float-noise ideal.
    """
    X, I, J = 30, 4, 4
    gen = torch.Generator().manual_seed(4)
    base = torch.randn(X, I, dtype=torch.float64, generator=gen)
    base_ref = torch.randn(X, J, dtype=torch.float64, generator=gen)
    rep = BTRewardRepresentation(torch.stack([base, -base]),
                                 torch.stack([base_ref, -base_ref]),
                                 uniform_policy(X, J))
    with pytest.raises(KSUndefinedError) as exc:
        solve_kalai_smorodinsky(rep, uniform_policy(X, I), 1.0, 1, weight_l1=2.0, **KS_KW)
    diag = exc.value.diagnostics
    # The IR set is the single point s = 0, so u_k is positive only by float
    # noise -- orders of magnitude below the attainable range. Dividing by it
    # would have produced a rho* that looks like a solved problem.
    assert diag["nonpositive_objectives"] == [0, 1]
    assert max(diag["u_ideal"]) < 1e-9
    assert diag["attainable_ideal_scale"] > 1e-3


# --- degeneracy is reported, not hidden -------------------------------------

def test_undefined_ideal_point_raises_with_diagnostics():
    """When no policy lifts an objective above its disagreement point there is
    no bargaining set; the solver must say so rather than clamp the surplus."""
    gen = torch.Generator().manual_seed(2)
    K, X, I, J = 2, 20, 4, 4
    r = torch.randn(K, X, I, dtype=torch.float64, generator=gen)
    r_ref = torch.randn(K, X, J, dtype=torch.float64, generator=gen)
    r[0] = r[0] - 50.0                       # objective 0 can never reach its reference
    rep = BTRewardRepresentation(r, r_ref, uniform_policy(X, J))
    with pytest.raises(KSUndefinedError) as exc:
        solve_kalai_smorodinsky(rep, uniform_policy(X, I), 1.0, 1, weight_l1=1.0, **KS_KW)
    diag = exc.value.diagnostics
    assert 0 in diag["nonpositive_objectives"]
    assert diag["u_ideal"][0] < 0
    assert "u_vertex" in diag and "ideal_point_definitions" in diag
