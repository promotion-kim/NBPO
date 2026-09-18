"""The canonical target mode must be the exact finite-pool expectation.

`sampled` and `rao_blackwell` are estimators of a quantity; `canonical` is that
quantity. The tests below pin the difference, because the three are easy to
confuse by name and only one of them has zero variance.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from scripts.nbpo.build_nbpo_pairs import TARGET_MODES, build_rows


def _fixture(K=2, X=3, I=4, J=5, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-0.4, 0.4, size=(K, X, I, J))
    nu = rng.dirichlet(np.ones(J), size=(K, X))
    lam = np.array([1.7, 2.3])[:K]
    objectives = [f"obj{k}" for k in range(K)]
    meta = {
        "prompt_ids": [f"p{x}" for x in range(X)],
        "policy_learner_ids": [f"policy:s{i}" for i in range(I)],
        "comparator_ids": [f"ref:r{j}" for j in range(J)],
    }
    policy = {f"s{i}": {f"p{x}": {"prompt": f"prompt {x}",
                                  "generated_text": f"resp {i} for {x}"}
                        for x in range(X)} for i in range(I)}
    return A, nu, lam, objectives, meta, policy


def _rows(mode, seed=0):
    A, nu, lam, objectives, meta, policy = _fixture()
    return build_rows(meta["prompt_ids"], objectives, A, nu, lam,
                      np.array([0.25] * len(objectives)), policy, None,
                      np.random.default_rng(seed), mode, meta, {}), (A, nu, lam, meta)


def test_canonical_is_a_registered_mode():
    assert "canonical" in TARGET_MODES


def test_canonical_matches_the_closed_form_expectation():
    rows, (A, nu, lam, meta) = _rows("canonical")
    x_of = {p: i for i, p in enumerate(meta["prompt_ids"])}
    i_of = {r: i for i, r in enumerate(meta["policy_learner_ids"])}
    for r in rows:
        x = x_of[r["prompt_id"]]
        a, b = i_of[r["chosen_response_id"]], i_of[r["rejected_response_id"]]
        want = sum(lam[k] * float((nu[k, x] * (A[k, x, a, :] - A[k, x, b, :])).sum())
                   for k in range(len(lam)))
        assert r["nbpo_weighted_z"] == pytest.approx(want, abs=1e-12)


def test_canonical_is_deterministic_across_seeds():
    a, _ = _rows("canonical", seed=0)
    b, _ = _rows("canonical", seed=999)
    assert [r["nbpo_weighted_z"] for r in a] == [r["nbpo_weighted_z"] for r in b]


def test_sampled_is_not_deterministic_across_seeds():
    a, _ = _rows("sampled", seed=0)
    b, _ = _rows("sampled", seed=999)
    assert [r["nbpo_weighted_z"] for r in a] != [r["nbpo_weighted_z"] for r in b]


def test_canonical_records_that_no_opponent_was_drawn():
    rows, _ = _rows("canonical")
    for r in rows:
        assert set(r["opponent_response_id"].values()) == {"expectation:nu_star"}


def test_sampled_and_rao_blackwell_are_unbiased_for_the_canonical_value():
    """Averaging many draws converges to the canonical target, not merely
    correlates with it -- that is what makes them estimators of the same thing.

    The tolerance is the estimator's OWN standard error, computed from the draws
    rather than guessed: a fixed epsilon would either be so loose it tests
    nothing or so tight it fails on sampling noise, which is exactly what a
    first attempt at this test did.
    """
    canon, _ = _rows("canonical")
    key = lambda r: (r["prompt_id"], r["chosen_response_id"], r["rejected_response_id"])
    want = {key(r): r["nbpo_weighted_z"] for r in canon}
    n = 400
    for mode in ("rao_blackwell", "sampled"):
        draws = {k: [] for k in want}
        for s in range(n):
            for r in _rows(mode, seed=s)[0]:
                draws[key(r)].append(r["nbpo_weighted_z"])
        worst_z = 0.0
        for k, vals in draws.items():
            v = np.asarray(vals)
            se = v.std(ddof=1) / np.sqrt(len(v))
            if se == 0:
                assert v.mean() == pytest.approx(want[k], abs=1e-12)
                continue
            worst_z = max(worst_z, abs(v.mean() - want[k]) / se)
        # ~18 rows checked at once, so allow a Bonferroni-ish 4 standard errors
        assert worst_z < 4.0, f"{mode} deviates by {worst_z:.2f} standard errors"


def test_canonical_has_strictly_smaller_spread_than_the_estimators():
    canon, _ = _rows("canonical")
    spread = lambda rows: float(np.std([r["nbpo_weighted_z"] for r in rows]))
    # the canonical value has no draw noise at all, so pooling many sampled
    # draws must show a wider spread of realized values
    sampled = [r for s in range(20) for r in _rows("sampled", seed=s)[0]]
    rb = [r for s in range(20) for r in _rows("rao_blackwell", seed=s)[0]]
    assert spread(sampled) > spread(rb) > spread(canon) * 0.99
