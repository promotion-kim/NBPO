"""v4 decoding robustness: the marker must survive, and absence must stay absent.

v3's protocol lost 4-21% of pairs because a deliberative rationale ran past its
96-token budget and the verdict marker was never emitted. The repair is a bigger
budget, a stop that fires on the marker, and one deterministic retry. Each of
those can fail quietly, so each is pinned here:

* a stop that trims the stop string would delete the very thing being parsed;
* a parser that takes the FIRST marker would read a marker quoted inside the
  rationale rather than the verdict;
* an "unparseable -> TIE" fallback would convert a decoding failure into a
  confident statement of indifference, which is the worst possible repair.
"""
import pytest

from scripts.experiments.iclr2027_table1_v2.judge_protocol_v3 import (
    hard_probs_from_verdict, load_protocol,
)
from scripts.experiments.iclr2027_table1_v2.run_judge_protocol import parse_marker
from pathlib import Path

V4 = Path("experiments/iclr2027_table1_v2/judge_protocol_v4/candidates")


class FakeOut:
    def __init__(self, text, n_tokens, finish_reason="stop"):
        self.text = text
        self.token_ids = list(range(n_tokens))
        self.finish_reason = finish_reason


# --- marker parsing ---------------------------------------------------------

def test_marker_is_found_when_the_stop_string_is_retained():
    assert parse_marker("A is clearer and better sourced.\n[[A]]") == "A"
    assert parse_marker("Neither is materially better.\n[[TIE]]") == "TIE"


def test_a_trimmed_stop_string_leaves_nothing_to_parse():
    """Documents why include_stop_str_in_output matters: without the marker in
    the text there is no verdict, and the row must go invalid rather than be
    guessed from the rationale."""
    assert parse_marker("A is clearer and better sourced.\n") is None


@pytest.mark.parametrize("n", [96, 128, 256, 511])
def test_marker_at_various_positions_including_the_old_boundary(n):
    """96 is exactly where v3 was truncating."""
    text = ("word " * n) + "\n[[B]]"
    assert parse_marker(text) == "B"


def test_no_marker_is_invalid_and_never_becomes_a_tie():
    truncated = "Response A covers the required steps whereas Response B omits the"
    assert parse_marker(truncated) is None
    with pytest.raises(ValueError):
        hard_probs_from_verdict(parse_marker(truncated))


def test_multiple_markers_take_the_LAST_one():
    """A rationale may quote the options before committing; the verdict is the
    final marker, not the first one mentioned."""
    text = "The choice is between [[A]] and [[B]]. B is materially better.\n[[B]]"
    assert parse_marker(text) == "B"


def test_a_truncated_rationale_yields_no_verdict():
    assert parse_marker("Response A begins by restating the question, which") is None


# --- protocol wiring --------------------------------------------------------

@pytest.mark.parametrize("name", ["P2_long", "P2_balanced"])
def test_v4_candidates_declare_the_repaired_decoding(name):
    p = load_protocol(V4 / f"{name}.yaml")
    assert p.max_tokens == 512
    assert p.retry_max_tokens == 1024
    assert p.stop_after_marker is True
    assert p.always_all_templates is True     # no adaptive omission during development
    assert p.decoding["temperature"] == 0.0   # retry must be deterministic


def test_p2_long_changes_only_decoding_from_the_frozen_p2():
    """P2-long must be attributable to budget alone, so every semantic field is
    byte-identical to the frozen v3 protocol.

    Including the template list. An earlier version cut it to two for cost parity
    with P2-balanced, and an ablation showed that cost the deterministic gate on
    truthfulness; cost is the last selection criterion, so accuracy is not traded
    for it.
    """
    import yaml
    frozen = yaml.safe_load(Path("experiments/iclr2027_table1_v2/judge_protocol_v3/protocol.yaml").read_text())
    lng = yaml.safe_load((V4 / "P2_long.yaml").read_text())
    for k in ("common_system", "verdict_instruction", "labels", "rubric_file",
              "rubric_sha256", "objective_procedures", "kind"):
        assert lng[k] == frozen[k], k
    assert lng["templates"] == frozen["templates"]
    assert lng["decoding"] == frozen["decoding"]


def test_both_v4_candidates_share_the_decoding_repair():
    """They may differ in template count -- cost is the last criterion -- but the
    decoding repair under test must be identical, or the comparison is about
    something else."""
    import yaml
    a = yaml.safe_load((V4 / "P2_long.yaml").read_text())
    b = yaml.safe_load((V4 / "P2_balanced.yaml").read_text())
    assert a["max_tokens"] == b["max_tokens"] == 512
    assert a["retry_max_tokens"] == b["retry_max_tokens"] == 1024
    assert a["stop_after_marker"] == b["stop_after_marker"] is True


def test_p2_balanced_pairs_each_wording_with_both_analysis_orders():
    import yaml
    bal = yaml.safe_load((V4 / "P2_balanced.yaml").read_text())
    by_wording = {}
    for t in bal["templates"]:
        by_wording.setdefault(t["user_template"], []).append(t["extra_system"])
    assert len(by_wording) == 2                      # two distinct wordings
    for variants in by_wording.values():
        assert len(variants) == 2                    # each seen in both orders
        assert any("first assess Response A" in v for v in variants)
        assert any("first assess Response B" in v for v in variants)
        assert all(v.startswith("__PROCEDURE__") for v in variants)


def test_p2_balanced_is_balanced_across_wording_and_analysis_order():
    """Two wordings x two analysis orders, none adaptively skipped, so analysis
    order is balanced by construction and wording diversity is not sacrificed."""
    import yaml
    b = yaml.safe_load((V4 / "P2_balanced.yaml").read_text())
    ids = [t["id"] for t in b["templates"]]
    assert ids == ["w0o0", "w0o1", "w1o0", "w1o1"]
    a_first = [t for t in b["templates"] if "first assess Response A" in t["extra_system"]]
    b_first = [t for t in b["templates"] if "first assess Response B" in t["extra_system"]]
    assert len(a_first) == len(b_first) == 2
    assert load_protocol(V4 / "P2_balanced.yaml").always_all_templates is True


# --- semantic mapping survives the retry ------------------------------------

def test_semantic_swap_mapping_is_unchanged_by_a_retry():
    """A retried rendering re-enters the identical algebra; the retry changes the
    token budget, never the orientation."""
    from scripts.experiments.iclr2027_table1_v2.judge_protocol_v3 import (
        FORWARD, REVERSE, build_observation, combine,
    )
    first_try = build_observation(hard_probs_from_verdict("A"), FORWARD, "t0")
    retried = build_observation(hard_probs_from_verdict("B"), REVERSE, "t0")
    retried["first_pass_invalid"] = True
    retried["retried_at_max_tokens"] = 1024
    agg = combine([first_try, retried])
    assert agg["p_hat"] == pytest.approx(1.0)     # learner wins in both orders
