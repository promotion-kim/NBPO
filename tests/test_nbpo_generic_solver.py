"""The unified (representation x aggregation) finite-pool solver.

Two things must hold for the matched Table-1 comparison to mean anything:

1. the new generic path reproduces the audited legacy solver *exactly* on the
   adaptive-game Nash configuration, so no reported NBPO number silently
   changes when the baselines are added;
2. the pairwise realization target is the same functional form for every
   aggregation, with ``eta`` applied exactly once, so the rows differ in the
   direction of the weight vector and not in the step size.
"""
import itertools

import pytest
import torch

from mnpo_scripts.nbpo_core import uniform_policy, weighted_policy_update
from mnpo_scripts.nbpo_generic import (
    AGGREGATIONS,
    exp_update,
    matched_weight_l1,
    solve_finite_pool,
    solve_proximal,
)
from mnpo_scripts.nbpo_representations import (
    AdaptiveGameRepresentation,
    BTRewardRepresentation,
    FixedReferenceRepresentation,
)
from mnpo_scripts.nbpo_solver import solve_nbpo_dual


def make_pool(seed=5, K=3, X=48, I=4, n_pool=8, spread=0.5, gain=0.4):
    """A pool whose learner and comparator responses are exchangeable draws.

    That is the real geometry at outer stage 0 (both groups sampled from the
    same frozen reference), so ``s(pi_t) ~ 0`` and the bargaining set is
    nondegenerate -- unlike an unstructured random tensor, where the ideal point
    can legitimately fail to dominate the disagreement point.
    """
    gen = torch.Generator().manual_seed(seed)
    lat = torch.randn(K, X, n_pool, dtype=torch.float64, generator=gen) * spread
    def payoff(a, b):
        return gain * torch.tanh(lat[:, :, a].unsqueeze(-1) - lat[:, :, b].unsqueeze(-2))
    learner, comp = list(range(I)), list(range(I, n_pool))
    A = payoff(learner, comp)
    raw = payoff(comp, comp)
    A_ref = 0.5 * (raw - raw.transpose(-1, -2))
    A_ref = A_ref - torch.diag_embed(torch.diagonal(A_ref, dim1=-2, dim2=-1))
    return A, A_ref, uniform_policy(X, len(comp))


@pytest.fixture(scope="module")
def pool():
    return make_pool()


@pytest.fixture(scope="module")
def adaptive(pool):
    A, A_ref, mu = pool
    return AdaptiveGameRepresentation(A, A_ref, mu,
                                      torch.full((A.shape[0],), 0.25, dtype=torch.float64))


def test_aggregation_registry():
    assert AGGREGATIONS == ("nash", "utilitarian", "kalai_smorodinsky",
                            "absolute_maxmin", "surplus_maxmin")


@pytest.mark.parametrize("R", [1, 3, 5])
def test_generic_nash_reproduces_the_legacy_solver_bitwise(pool, R):
    """No reported NBPO number may move because the baselines were added."""
    A, A_ref, mu = pool
    beta = torch.full((A.shape[0],), 0.25, dtype=torch.float64)
    legacy = solve_nbpo_dual(A, A_ref, mu, beta, eta=0.5, gamma=0.5, M=120, R=R)
    rep = AdaptiveGameRepresentation(A, A_ref, mu, beta)
    new = solve_finite_pool(rep, "nash", eta=0.5, M=120, R=R, gamma=0.5)
    assert torch.equal(legacy.lam, new.weights)
    assert torch.equal(legacy.pi, new.pi)
    assert torch.equal(legacy.surplus, new.surplus)
    assert torch.equal(legacy.d, new.d)
    assert legacy.kkt_residual == new.kkt_residual
    assert legacy.projected_kkt_residual == new.projected_kkt_residual
    assert legacy.fixed_point_residual == new.fixed_point_residual


def test_exp_update_matches_the_audited_positive_weight_implementation(pool):
    A, A_ref, mu = pool
    rep = AdaptiveGameRepresentation(A, A_ref, mu,
                                     torch.full((3,), 0.25, dtype=torch.float64))
    pi_t = uniform_policy(rep.X, rep.I)
    _, q = rep.opponent_and_gradient(pi_t)
    w = torch.tensor([1.3, 0.7, 2.1], dtype=torch.float64)
    assert torch.equal(exp_update(pi_t, q, w, 0.7), weighted_policy_update(pi_t, q, w, 0.7))


def test_exp_update_allows_zero_weights_but_not_negative(pool):
    A, A_ref, mu = pool
    rep = AdaptiveGameRepresentation(A, A_ref, mu,
                                     torch.full((3,), 0.25, dtype=torch.float64))
    pi_t = uniform_policy(rep.X, rep.I)
    _, q = rep.opponent_and_gradient(pi_t)
    exp_update(pi_t, q, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), 0.5)
    with pytest.raises(ValueError, match="nonnegative"):
        exp_update(pi_t, q, torch.tensor([1.0, -0.1, 0.0], dtype=torch.float64), 0.5)


# --- the realization target -------------------------------------------------

REPRESENTATION_IDS = ["adaptive_game", "fixed_reference", "bt_reward"]


def build_rep(kind, pool):
    A, A_ref, mu = pool
    if kind == "adaptive_game":
        return AdaptiveGameRepresentation(A, A_ref, mu,
                                          torch.full((A.shape[0],), 0.25, dtype=torch.float64))
    if kind == "fixed_reference":
        return FixedReferenceRepresentation(A, A_ref, mu)
    gen = torch.Generator().manual_seed(99)
    K, X, J = A.shape[0], A.shape[1], A.shape[3]
    r = torch.randn(K, X, A.shape[2], dtype=torch.float64, generator=gen)
    r_ref = torch.randn(K, X, J, dtype=torch.float64, generator=gen)
    return BTRewardRepresentation(r, r_ref, mu)


def solve_matched(rep, aggregation, eta=0.5, R=3):
    """Solve one row at the matched weight norm.

    NBPO keeps its raw lambda; every control is solved at ``||lambda||_1`` so
    the rows differ only in the direction of the weight vector.  This is the
    protocol the real runs use, so the tests exercise it too.
    """
    nash = solve_finite_pool(rep, "nash", eta=eta, R=R, M=200, gamma=0.5)
    if aggregation == "nash":
        return nash
    l1 = matched_weight_l1(nash)
    kw = dict(eta=eta, R=R, M=200, weight_l1=l1)
    if aggregation == "kalai_smorodinsky":
        kw["ks_kwargs"] = dict(stage1_iters=150, ideal_iters=30)
    return solve_finite_pool(rep, aggregation, **kw)


CPU_TOY_MATRIX = [
    ("adaptive_game", "nash"),
    ("fixed_reference", "nash"),
    ("bt_reward", "nash"),
    ("adaptive_game", "utilitarian"),
    ("adaptive_game", "kalai_smorodinsky"),
    ("bt_reward", "utilitarian"),
]


@pytest.mark.parametrize("rep_kind,aggregation", CPU_TOY_MATRIX)
def test_cpu_toy_end_to_end_every_required_combination(pool, rep_kind, aggregation):
    """Section 15's required generic CPU toy run for all six combinations."""
    rep = build_rep(rep_kind, pool)
    sol = solve_matched(rep, aggregation)
    assert sol.representation == rep_kind and sol.aggregation == aggregation
    assert torch.isfinite(sol.surplus).all() and torch.isfinite(sol.weights).all()
    assert (sol.weights >= 0).all()
    assert torch.allclose(sol.pi.sum(-1), torch.ones(sol.pi.shape[0], dtype=torch.float64))
    # The artifact carries everything Section 3 requires.
    for field in ("pi_t", "pi", "q_update", "nu_update", "target_log_ratio", "V", "d",
                  "surplus", "weights"):
        assert getattr(sol, field) is not None
    assert sol.config["eta_applications"] == 1
    assert sol.config["weights_are_raw"] is True


@pytest.mark.parametrize("rep_kind,aggregation", CPU_TOY_MATRIX)
def test_pairwise_target_equals_eta_times_weighted_q_difference(pool, rep_kind, aggregation):
    """Section 3's numeric requirement, for every representation/aggregation.

        [log pi*(y) - log pi_t(y)] - [log pi*(y') - log pi_t(y')]
            == eta * sum_k w_k (q_k(y) - q_k(y'))
    """
    rep = build_rep(rep_kind, pool)
    sol = solve_matched(rep, aggregation)
    score = sol.eta * torch.einsum("k,kxi->xi", sol.weights, sol.q_update)
    worst = 0.0
    for i, j in itertools.combinations(range(rep.I), 2):
        lhs = sol.target_log_ratio[:, i] - sol.target_log_ratio[:, j]
        rhs = score[:, i] - score[:, j]
        worst = max(worst, float((lhs - rhs).abs().max()))
    assert worst < 1e-12, f"pairwise target mismatch {worst:.3e}"
    assert sol.target_log_ratio_check() < 1e-12


def test_eta_is_applied_exactly_once(pool):
    """Doubling eta must double the pairwise target, not quadruple it.

    Applying eta twice anywhere in the chain would make the target scale like
    eta^2; this pins the exponent to a single application.
    """
    rep = FixedReferenceRepresentation(*pool)   # constant q -> exact closed form
    w = torch.tensor([0.4, 0.3, 0.3], dtype=torch.float64)
    pi_t = uniform_policy(rep.X, rep.I)
    _, q = rep.opponent_and_gradient(pi_t)
    diffs = []
    for eta in (0.25, 0.5, 1.0):
        pi = exp_update(pi_t, q, w, eta)
        lr = torch.log(pi) - torch.log(pi_t)
        diffs.append(float((lr[:, 0] - lr[:, 1]).abs().mean()))
    assert abs(diffs[1] / diffs[0] - 2.0) < 1e-9
    assert abs(diffs[2] / diffs[1] - 2.0) < 1e-9


def test_fixed_representations_have_an_exact_one_step_solve(pool):
    """When q does not depend on pi the Eq. (21) map is constant, so one
    application lands on its fixed point at every R.

    ``extra_map_residual`` -- one further application of the map to the returned
    policy -- is the invariant that says so, and it is exactly zero.
    ``fixed_point_residual`` is the last-iteration *change*, so at R = 1 it is
    the (nonzero) move away from the proximal centre and only vanishes from
    R >= 2. Both are reported as-is; neither is special-cased.
    """
    for kind in ("fixed_reference", "bt_reward"):
        rep = build_rep(kind, pool)
        first = None
        for R in (1, 2, 3, 5):
            sol = solve_finite_pool(rep, "utilitarian", eta=0.5, R=R, weight_l1=1.0)
            assert sol.extra_map_residual == 0.0
            assert sol.config["R_is_approximation"] is False
            if R >= 2:
                assert sol.fixed_point_residual == 0.0
            if first is None:
                first = sol.pi
            else:
                assert torch.equal(sol.pi, first)   # R genuinely has no effect


def test_adaptive_representation_reports_R_as_an_approximation(adaptive):
    sol = solve_finite_pool(adaptive, "utilitarian", eta=0.5, R=1, weight_l1=1.0)
    assert sol.config["R_is_approximation"] is True
    assert sol.fixed_point_residual > 0.0
    deeper = solve_finite_pool(adaptive, "utilitarian", eta=0.5, R=8, weight_l1=1.0)
    assert deeper.extra_map_residual < sol.extra_map_residual


def test_weights_are_never_normalized_in_the_nash_path(pool):
    """Non-negotiable 6: raw lambda reaches the target; normalizing it would
    silently rescale the step."""
    A, A_ref, mu = pool
    rep = AdaptiveGameRepresentation(A, A_ref, mu,
                                     torch.full((3,), 0.25, dtype=torch.float64))
    sol = solve_finite_pool(rep, "nash", eta=1.0, M=300, R=3, gamma=0.5)
    assert float(sol.weights.sum()) > 1.5           # raw lambda ~ 1/s, far off the simplex
    assert sol.config["weights_normalized_for_training"] is False
    # and lambda_k = 1/s_k at the KKT point
    assert sol.kkt_residual == pytest.approx(
        float((sol.surplus - 1.0 / sol.weights).abs().max()))


def test_surpluses_are_never_clamped(pool):
    """A nonpositive surplus must survive into the artifact, not become eps."""
    A, A_ref, mu = pool
    # Make objective 0 unreachable by flipping its payoff against the learner.
    A2 = A.clone()
    A2[0] = -0.5
    rep = FixedReferenceRepresentation(A2, A_ref, mu)
    sol = solve_finite_pool(rep, "utilitarian", eta=0.5, R=1, weight_l1=1.0)
    assert float(sol.surplus[0]) < 0.0
