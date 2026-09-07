"""The v2 rubric schema: one shared evaluator instruction, four criteria.

v1 repeated the whole prompt per objective, so "how to judge" drifted between
the four texts and every change had to be made four times. v2 states the
instruction once and gives each objective only its criterion. These tests pin
the composition, keep v1 loading unchanged (published numbers keep the rubric
that produced them), and make sure an ambiguous file is refused rather than
silently resolved.
"""
from pathlib import Path

import pytest
import yaml

from scripts.nbpo.nbpo_common import RubricUnavailableError, load_objective_rubrics

V1 = Path("training_configs/nbpo/objectives/ultrafeedback.yaml")
V2 = Path("training_configs/nbpo/objectives/ultrafeedback_v2.yaml")
OBJECTIVES = ["instruction_following", "truthfulness", "honesty", "helpfulness"]


def test_v1_still_loads_under_the_per_objective_schema():
    r = load_objective_rubrics(V1, OBJECTIVES)
    assert r["version"] == 1
    for o in OBJECTIVES:
        assert r["rubrics"][o]["schema"] == "per_objective"
        assert "{prompt}" in r["rubrics"][o]["user_template"]


def test_v2_composes_the_shared_instruction_with_each_criterion():
    cfg = yaml.safe_load(V2.read_text())
    r = load_objective_rubrics(V2, OBJECTIVES)
    assert r["version"] == 2
    for o in OBJECTIVES:
        got = r["rubrics"][o]
        assert got["schema"] == "shared_instruction"
        assert got["system"].startswith(cfg["common_system"].rstrip())
        assert cfg["objectives"][o]["criterion"].strip() in got["system"]
        # ... and no OTHER objective's criterion leaked in
        for other in OBJECTIVES:
            if other != o:
                assert cfg["objectives"][other]["criterion"].strip() not in got["system"]


def test_v2_declares_the_three_verdict_tokens_and_forbids_other_criteria():
    r = load_objective_rubrics(V2, OBJECTIVES)
    for o in OBJECTIVES:
        sysmsg = r["rubrics"][o]["system"]
        for token in ("[[A]]", "[[B]]", "[[TIE]]"):
            assert token in sysmsg
        assert "Evaluate ONLY the objective stated below" in sysmsg
        assert "rather than its position" in sysmsg     # the position-bias clause


def test_v2_template_is_formattable_by_the_judge():
    """The judge substitutes {prompt}, {a}, {b}; the file is written with the
    self-describing {response_a}/{response_b}, rewritten once at load."""
    r = load_objective_rubrics(V2, OBJECTIVES)
    t = r["rubrics"]["helpfulness"]["user_template"]
    assert "{response_a}" not in t and "{a}" in t
    out = t.format(prompt="P", a="AAA", b="BBB")
    assert "[USER REQUEST]\nP" in out and "AAA" in out and "BBB" in out
    assert out.rstrip().endswith("[VERDICT]")


def test_every_objective_gets_a_distinct_rubric_hash():
    r = load_objective_rubrics(V2, OBJECTIVES)
    hashes = {o: r["rubrics"][o]["rubric_sha256"] for o in OBJECTIVES}
    assert len(set(hashes.values())) == len(OBJECTIVES)
    assert all(len(h) == 64 for h in hashes.values())


def test_the_rubric_hash_covers_the_text_not_just_the_file(tmp_path):
    """Two files can carry the same content hash and still differ in what one
    objective saw; the recorded hash must move when the TEXT moves."""
    cfg = yaml.safe_load(V2.read_text())
    before = load_objective_rubrics(V2, ["honesty"])["rubrics"]["honesty"]["rubric_sha256"]
    cfg["objectives"]["honesty"]["criterion"] += "\nAnd also prefer shorter answers."
    p = tmp_path / "edited.yaml"
    p.write_text(yaml.safe_dump(cfg))
    after = load_objective_rubrics(p, ["honesty"])["rubrics"]["honesty"]["rubric_sha256"]
    assert before != after


def test_v1_and_v2_hashes_differ_for_the_same_objective():
    a = load_objective_rubrics(V1, ["truthfulness"])["rubrics"]["truthfulness"]
    b = load_objective_rubrics(V2, ["truthfulness"])["rubrics"]["truthfulness"]
    assert a["rubric_sha256"] != b["rubric_sha256"]
    assert a["system"] != b["system"]


def test_mixing_both_schemas_for_one_objective_is_refused(tmp_path):
    cfg = yaml.safe_load(V2.read_text())
    cfg["objectives"]["honesty"]["system"] = "an override"
    cfg["objectives"]["honesty"]["user_template"] = "{prompt}{a}{b}"
    p = tmp_path / "ambiguous.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(RubricUnavailableError, match="BOTH"):
        load_objective_rubrics(p, ["honesty"])


def test_a_criterion_without_a_shared_instruction_is_refused(tmp_path):
    cfg = yaml.safe_load(V2.read_text())
    del cfg["common_system"]
    p = tmp_path / "no_common.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(RubricUnavailableError, match="no common_system"):
        load_objective_rubrics(p, ["honesty"])


def test_an_unformattable_template_is_refused_at_load_not_mid_run(tmp_path):
    cfg = yaml.safe_load(V2.read_text())
    cfg["common_user_template"] = "{prompt} {response_a} {response_b} {mystery}"
    p = tmp_path / "bad_template.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(RubricUnavailableError, match="cannot\n?\\s*format|unknown placeholder"):
        load_objective_rubrics(p, ["honesty"])


def test_a_missing_objective_is_named_rather_than_invented():
    with pytest.raises(RubricUnavailableError, match="not defined"):
        load_objective_rubrics(V2, ["conciseness"])
