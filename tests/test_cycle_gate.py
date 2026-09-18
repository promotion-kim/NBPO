import importlib.util
from pathlib import Path


PATH = (Path(__file__).parents[1] / "analysis" /
        "game_nbpo_neural_bridge_20260825" / "evaluate_cycle_gate.py")
SPEC = importlib.util.spec_from_file_location("cycle_gate", PATH)
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


def block(rows, prompts, lower):
    return {"judgment_rows": rows, "invalid_row_rate": 0.0,
            "objectives": {x: {"prompts": prompts, "strict_triples": 1,
                                "cyclic_triple_rate_ci": [lower, .2]}
                           for x in MODULE.OBJECTIVES}}


def test_resolved_complete_cycle_gate_passes():
    summary = {"slices": {"combined": block(4 * 28 * 2, 1, .01),
                           "Base_only": block(4 * 6 * 2, 1, 0),
                           "phase0_only": block(4 * 6 * 2, 1, 0)}}
    assert MODULE.evaluate(summary, prompts=1)["paper_cycle_claim_pass"]


def test_zero_lower_bound_fails_claim_gate():
    summary = {"slices": {"combined": block(4 * 28 * 2, 1, 0),
                           "Base_only": block(4 * 6 * 2, 1, 0),
                           "phase0_only": block(4 * 6 * 2, 1, 0)}}
    result = MODULE.evaluate(summary, prompts=1)
    assert result["integrity_pass"]
    assert not result["paper_cycle_claim_pass"]
