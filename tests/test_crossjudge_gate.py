import importlib.util
from pathlib import Path


PATH = (Path(__file__).parents[1] / "analysis" /
        "game_nbpo_neural_bridge_20260825" / "validate_crossjudge_gate.py")
SPEC = importlib.util.spec_from_file_location("crossjudge_gate", PATH)
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


def gate(passed=True, invalid=0):
    return {"judgment_rows_seen": 38400, "invalid_judgments": invalid,
            "complete_prompt_intersection": 200, "paper_gate_pass": passed,
            "summary": {name: {} for name in MODULE.ARMS},
            "primary_contrast": {"lower": .01}}


def test_complete_selector_crossjudge_passes():
    assert MODULE.validate(gate())["paper_crossjudge_gate_pass"]


def test_selector_or_integrity_failure_fails_closed():
    assert not MODULE.validate(gate(passed=False))["paper_crossjudge_gate_pass"]
    assert not MODULE.validate(gate(invalid=385))["paper_crossjudge_gate_pass"]
