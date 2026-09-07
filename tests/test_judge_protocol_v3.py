"""v3 judge semantics: the algebra that makes uncertainty legible.

The v2 protocol's defect was not arithmetic but representation -- a confident
tie and a position flip both collapsed to `Delta = 0` and became
indistinguishable. v3 keeps the choice distribution, so the two remain
separable. These tests pin that separation, and pin that v3 reduces exactly to
v2 when the distribution is one-hot (otherwise the two protocols would not be
comparable on the same metrics).
"""
import math

import pytest

from scripts.experiments.iclr2027_table1_v2.judge_protocol_v3 import (
    CHOICES,
    FORWARD,
    REVERSE,
    build_observation,
    combine,
    hard_probs_from_verdict,
    needs_adjudication,
    normalized_entropy,
    semantic_score,
)


def P(a, b, t):
    return {"A": a, "B": b, "TIE": t}


def obs(order, a, b, t, template="t0"):
    return build_observation(P(a, b, t), order, template)


# --- the semantic mapping ---------------------------------------------------

def test_forward_and_reverse_credit_the_learner_from_opposite_slots():
    """Forward shows the learner as A; reverse shows it as B."""
    assert semantic_score(P(1, 0, 0), FORWARD) == 1.0
    assert semantic_score(P(1, 0, 0), REVERSE) == 0.0
    assert semantic_score(P(0, 1, 0), FORWARD) == 0.0
    assert semantic_score(P(0, 1, 0), REVERSE) == 1.0


def test_a_tie_splits_evenly_in_both_orders():
    assert semantic_score(P(0, 0, 1), FORWARD) == 0.5
    assert semantic_score(P(0, 0, 1), REVERSE) == 0.5


def test_an_unknown_order_is_refused():
    with pytest.raises(ValueError, match="unknown presentation order"):
        semantic_score(P(1, 0, 0), "whichever")


# --- v3 reduces to v2 on one-hot distributions -----------------------------

V2_CASES = [
    ("A", "B", 1.0),    # forward A, reverse B  -> learner wins both orders
    ("B", "A", 0.0),    # forward B, reverse A  -> comparator wins both orders
    ("A", "A", 0.5),    # first-position contradiction
    ("B", "B", 0.5),    # second-position contradiction
    ("TIE", "TIE", 0.5),
]


@pytest.mark.parametrize("fwd,rev,expected", V2_CASES)
def test_hard_verdicts_reproduce_the_v2_swap_mapping_exactly(fwd, rev, expected):
    got = combine([build_observation(hard_probs_from_verdict(fwd), FORWARD, "t0"),
                   build_observation(hard_probs_from_verdict(rev), REVERSE, "t0")])
    assert got["p_hat"] == pytest.approx(expected, abs=1e-12)


def test_an_unparseable_hard_verdict_is_refused():
    with pytest.raises(ValueError, match="unparseable verdict"):
        hard_probs_from_verdict("[[MAYBE]]")


# --- the point of v3 --------------------------------------------------------

def test_a_confident_tie_and_a_position_flip_have_the_same_delta():
    """Both are Delta = 0. This is the v2 collapse, and it is unavoidable --
    Delta is a scalar."""
    tie = combine([obs(FORWARD, 0.01, 0.01, 0.98), obs(REVERSE, 0.01, 0.01, 0.98)])
    flip = combine([obs(FORWARD, 0.98, 0.01, 0.01), obs(REVERSE, 0.98, 0.01, 0.01)])
    assert tie["delta"] == pytest.approx(0.0, abs=1e-9)
    assert flip["delta"] == pytest.approx(0.0, abs=1e-9)


def test_but_v3_separates_them_on_the_diagnostics_that_matter():
    """...and THIS is what v2 could not do. The confident tie has a large tie
    probability, low entropy and no order gap; the position flip has a near-1
    order gap and negligible tie mass."""
    tie = combine([obs(FORWARD, 0.01, 0.01, 0.98), obs(REVERSE, 0.01, 0.01, 0.98)])
    flip = combine([obs(FORWARD, 0.98, 0.01, 0.01), obs(REVERSE, 0.98, 0.01, 0.01)])
    assert tie["mean_tie_probability"] > 0.9 and flip["mean_tie_probability"] < 0.1
    assert tie["order_gap"] < 0.05 and flip["order_gap"] > 0.9
    assert tie["mean_normalized_entropy"] < 0.15


def test_genuine_uncertainty_is_a_third_distinguishable_state():
    """Flat over A/B/TIE: no order gap, low tie mass, but maximal entropy."""
    flat = combine([obs(FORWARD, 1 / 3, 1 / 3, 1 / 3), obs(REVERSE, 1 / 3, 1 / 3, 1 / 3)])
    assert flat["delta"] == pytest.approx(0.0, abs=1e-9)
    assert flat["order_gap"] < 1e-9
    assert flat["mean_normalized_entropy"] == pytest.approx(1.0, abs=1e-9)
    assert flat["mean_tie_probability"] == pytest.approx(1 / 3, abs=1e-9)


def test_a_stable_semantic_preference_survives_the_swap():
    """Forward prefers A and reverse prefers B: the learner really is better."""
    stable = combine([obs(FORWARD, 0.9, 0.05, 0.05), obs(REVERSE, 0.05, 0.9, 0.05)])
    assert stable["delta"] > 0.35
    assert stable["order_gap"] < 0.05
    assert stable["hard_argmax_verdict"] in ("A", "B")


# --- entropy ----------------------------------------------------------------

def test_normalized_entropy_spans_zero_to_one():
    assert normalized_entropy(P(1, 0, 0)) == pytest.approx(0.0, abs=1e-12)
    assert normalized_entropy(P(1 / 3, 1 / 3, 1 / 3)) == pytest.approx(1.0, abs=1e-12)
    assert 0.0 < normalized_entropy(P(0.5, 0.3, 0.2)) < 1.0


# --- aggregation guards -----------------------------------------------------

def test_probabilities_must_be_a_restricted_softmax():
    with pytest.raises(ValueError, match="restricted softmax"):
        build_observation(P(0.5, 0.2, 0.1), FORWARD, "t0")


def test_a_single_order_cannot_produce_a_semantic_score():
    """Averaging one order carries exactly the position bias swap averaging
    exists to cancel."""
    with pytest.raises(ValueError, match="both presentation orders"):
        combine([obs(FORWARD, 0.9, 0.05, 0.05)])


def test_multiple_templates_average_within_each_order():
    a = combine([obs(FORWARD, 1.0, 0.0, 0.0, "t0"), obs(FORWARD, 0.0, 1.0, 0.0, "t1"),
                 obs(REVERSE, 0.0, 1.0, 0.0, "t0"), obs(REVERSE, 1.0, 0.0, 0.0, "t1")])
    assert a["semantic_score_forward"] == pytest.approx(0.5)
    assert a["semantic_score_reverse"] == pytest.approx(0.5)
    assert a["p_hat"] == pytest.approx(0.5)
    assert a["n_observations"] == 4


def test_delta_is_antisymmetric_under_swapping_the_roles():
    """Relabelling which response is the learner must negate Delta."""
    fwd, rev = P(0.8, 0.1, 0.1), P(0.15, 0.75, 0.10)
    y = combine([build_observation(fwd, FORWARD, "t0"),
                 build_observation(rev, REVERSE, "t0")])
    # swapping roles turns the forward rendering into the reverse one and vice versa
    z = combine([build_observation(rev, FORWARD, "t0"),
                 build_observation(fwd, REVERSE, "t0")])
    assert y["delta"] == pytest.approx(-z["delta"], abs=1e-12)


# --- adaptive adjudication --------------------------------------------------

def test_adjudication_triggers_on_a_large_order_gap():
    flip = combine([obs(FORWARD, 0.98, 0.01, 0.01), obs(REVERSE, 0.98, 0.01, 0.01)])
    assert needs_adjudication(flip, order_gap_threshold=0.25, entropy_threshold=0.9)


def test_adjudication_triggers_on_high_entropy_even_without_an_order_gap():
    flat = combine([obs(FORWARD, 1 / 3, 1 / 3, 1 / 3), obs(REVERSE, 1 / 3, 1 / 3, 1 / 3)])
    assert flat["order_gap"] < 1e-9
    assert needs_adjudication(flat, order_gap_threshold=0.25, entropy_threshold=0.9)


def test_a_confident_stable_pair_is_not_adjudicated():
    stable = combine([obs(FORWARD, 0.95, 0.03, 0.02), obs(REVERSE, 0.03, 0.95, 0.02)])
    assert not needs_adjudication(stable, order_gap_threshold=0.25, entropy_threshold=0.9)


def test_a_confident_tie_is_not_adjudicated_either():
    """A confident tie is an answer, not an abstention -- re-querying it would
    burn budget on the pairs the judge is surest about."""
    tie = combine([obs(FORWARD, 0.02, 0.02, 0.96), obs(REVERSE, 0.02, 0.02, 0.96)])
    assert not needs_adjudication(tie, order_gap_threshold=0.25, entropy_threshold=0.9)
