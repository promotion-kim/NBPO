"""The semantic swap mapping, pinned against hand-constructed verdict pairs.

The v2 audit failed its swap-consistency gate. Before concluding anything about
the judge from that, the arithmetic that turns two ordered verdicts into one
semantic preference has to be shown correct -- because every explanation of the
failure changes if this mapping is wrong.

Conventions, stated once so the cases below are readable:

* ``learner_first``  ("forward") shows the learner ``y`` in the Response A slot
  and the comparator ``z`` in the Response B slot;
* ``comparator_first`` ("reverse") shows ``z`` as Response A and ``y`` as
  Response B;
* ``parse_verdict`` returns the win probability **of whatever is in the A
  slot**;
* ``to_policy_win`` converts that to the win probability **of the learner**,
  which is the quantity the tensor is built from.

The six cases are the ones that distinguish a correct mapping from the two
plausible wrong ones (forgetting to flip on reverse, or flipping twice).
"""
import numpy as np
import pytest

from scripts.nbpo.build_preference_tensor import (
    aggregate_cells,
    fill_policy_tensor,
    fill_reference_tensor,
)
from scripts.nbpo.judge_pairwise_matrix import parse_verdict, to_policy_win

FORWARD, REVERSE = "learner_first", "comparator_first"


def semantic_p_hat(forward_verdict: str, reverse_verdict: str) -> float:
    """p_hat(y > z) from the two raw judge completions, via the production path."""
    f = to_policy_win(parse_verdict(forward_verdict), FORWARD)
    r = to_policy_win(parse_verdict(reverse_verdict), REVERSE)
    return 0.5 * (f + r)


# --- Section A, cases 1-5 ---------------------------------------------------

CASES = [
    # (name, forward completion, reverse completion, expected p_hat(y > z))
    ("1. forward A wins, reverse B wins -> y wins in both orders",
     "[[A]]", "[[B]]", 1.0),
    ("2. forward B wins, reverse A wins -> z wins in both orders",
     "[[B]]", "[[A]]", 0.0),
    ("3. forward A wins, reverse A wins -> first-position contradiction",
     "[[A]]", "[[A]]", 0.5),
    ("4. forward B wins, reverse B wins -> second-position contradiction",
     "[[B]]", "[[B]]", 0.5),
    ("5. tie / tie",
     "[[TIE]]", "[[TIE]]", 0.5),
]


@pytest.mark.parametrize("name,fwd,rev,expected", CASES,
                         ids=[c[0].split(".")[0] for c in CASES])
def test_semantic_swap_mapping(name, fwd, rev, expected):
    assert semantic_p_hat(fwd, rev) == pytest.approx(expected, abs=1e-12), name


def test_the_two_position_contradictions_are_indistinguishable_from_a_tie():
    """The mechanically important consequence: a judge that always picks the
    first slot, and a judge that says the two responses are equal, produce the
    SAME tensor entry. The tensor cannot tell noise from indifference."""
    assert semantic_p_hat("[[A]]", "[[A]]") == semantic_p_hat("[[TIE]]", "[[TIE]]")
    assert semantic_p_hat("[[B]]", "[[B]]") == semantic_p_hat("[[TIE]]", "[[TIE]]")


def test_half_verdicts_land_on_the_declared_grid():
    seen = {semantic_p_hat(a, b)
            for a in ("[[A]]", "[[B]]", "[[TIE]]")
            for b in ("[[A]]", "[[B]]", "[[TIE]]")}
    assert seen == {0.0, 0.25, 0.5, 0.75, 1.0}


def test_mapping_is_exactly_antisymmetric_under_relabelling():
    """Swapping which response we call the learner must negate the margin."""
    for fwd, rev in [("[[A]]", "[[B]]"), ("[[B]]", "[[A]]"), ("[[A]]", "[[TIE]]"),
                     ("[[TIE]]", "[[B]]"), ("[[A]]", "[[A]]")]:
        forward_margin = semantic_p_hat(fwd, rev) - 0.5
        # relabelling swaps the roles of the two orders AND the two slots
        swapped = semantic_p_hat(rev, fwd) - 0.5
        assert forward_margin == pytest.approx(-swapped, abs=1e-12), (fwd, rev)


def test_a_reverse_verdict_that_is_not_flipped_would_fail_these_cases():
    """Guards against the obvious wrong implementation: if `to_policy_win` did
    not flip on the reverse order, cases 1 and 3 would swap answers."""
    def broken(fwd, rev):
        return 0.5 * (parse_verdict(fwd) + parse_verdict(rev))
    assert broken("[[A]]", "[[B]]") == 0.5            # would erase a real preference
    assert broken("[[A]]", "[[A]]") == 1.0            # would invent one from position bias
    assert semantic_p_hat("[[A]]", "[[B]]") == 1.0
    assert semantic_p_hat("[[A]]", "[[A]]") == 0.5


# --- Section A, case 6: skew symmetry --------------------------------------

def _rows(pairs, pool, objective="o1", prompt="p1"):
    out = []
    for (lid, cid), (fwd, rev) in pairs.items():
        for order, completion in ((FORWARD, fwd), (REVERSE, rev)):
            out.append({
                "prompt_id": prompt, "objective": objective, "learner_pool": pool,
                "learner_response_id": lid, "comparator_response_id": cid,
                "presentation_order": order, "valid": True,
                "policy_win": to_policy_win(parse_verdict(completion), order),
            })
    return out


def test_reference_tensor_is_exactly_skew_symmetric_with_a_zero_diagonal():
    """Case 6. One response set on both sides, so A(i,j) = -A(j,i) and A(i,i)=0
    must hold to the bit, not to a tolerance."""
    ids = ["r1", "r2", "r3"]
    verdicts = {("r1", "r2"): ("[[A]]", "[[B]]"),    # r1 > r2 decisively
                ("r1", "r3"): ("[[A]]", "[[TIE]]"),  # r1 > r3 weakly
                ("r2", "r3"): ("[[B]]", "[[B]]")}    # position contradiction -> 0
    payoff = aggregate_cells(_rows(verdicts, "reference"))
    A, stats = fill_reference_tensor(payoff, ["o1"], ["p1"], ids)
    assert np.abs(A + np.swapaxes(A, -1, -2)).max() == 0.0
    assert np.abs(np.diagonal(A, axis1=-2, axis2=-1)).max() == 0.0
    # the builder must also SAY it is skew-symmetric, and must not have had to
    # project to get there: each unordered pair was judged once, in both orders
    assert stats["skew_residual_post"] == 0.0
    assert stats["diagonal_zero"] is True
    assert stats["skew_projection_applied"] is False
    assert A[0, 0, 0, 1] == pytest.approx(0.5)    # r1 vs r2
    assert A[0, 0, 1, 0] == pytest.approx(-0.5)
    assert A[0, 0, 0, 2] == pytest.approx(0.25)   # r1 vs r3
    assert A[0, 0, 1, 2] == pytest.approx(0.0)    # contradiction erased


def test_policy_tensor_carries_the_semantic_margin_unsymmetrized():
    """Learner and comparator are different response sets, so no symmetry is
    imposed -- the entry is just the swap-averaged semantic margin."""
    verdicts = {("y1", "z1"): ("[[A]]", "[[B]]"),
                ("y1", "z2"): ("[[B]]", "[[B]]"),
                ("y2", "z1"): ("[[TIE]]", "[[TIE]]"),
                ("y2", "z2"): ("[[B]]", "[[A]]")}
    payoff = aggregate_cells(_rows(verdicts, "policy"))
    A = fill_policy_tensor(payoff, ["o1"], ["p1"], ["y1", "y2"], ["z1", "z2"])
    assert A[0, 0, 0, 0] == pytest.approx(+0.5)
    assert A[0, 0, 0, 1] == pytest.approx(0.0)
    assert A[0, 0, 1, 0] == pytest.approx(0.0)
    assert A[0, 0, 1, 1] == pytest.approx(-0.5)


def test_a_cell_judged_in_only_one_order_is_refused_not_halved():
    rows = [r for r in _rows({("y1", "z1"): ("[[A]]", "[[B]]")}, "policy")
            if r["presentation_order"] == FORWARD]
    with pytest.raises(RuntimeError, match="only one presentation order"):
        aggregate_cells(rows)
