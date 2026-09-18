import importlib.util
from pathlib import Path


PATH = (Path(__file__).parents[1] / "analysis" /
        "game_nbpo_neural_bridge_20260825" / "analyze_cycle_matrix.py")
SPEC = importlib.util.spec_from_file_location("cycle_matrix", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def edge(i, j, win, left="Base", right="Base", order="ij"):
    return {"prompt_index": 0, "objective": "helpfulness",
            "response_i": i, "response_j": j, "oriented_i_win": win,
            "label_i": left, "label_j": right, "order": order}


def test_cycle_and_label_slices():
    # 0 > 1, 1 > 2, 2 > 0; duplicate rows represent the two swapped orders.
    rows = []
    for item in (edge(0, 1, 1), edge(1, 2, 1), edge(0, 2, 0)):
        rows.extend((item, dict(item)))
    result = MODULE.summarize(rows, bootstrap=10, seed=42)
    assert result["cyclic_triples"] == 1
    assert result["strict_triples"] == 1
    assert result["tied_edges"] == 0
    assert result["invalid_edges"] == 0
    assert result["slices"]["Base_only"]["cyclic_triple_rate"] == 1.0


def test_swap_disagreement_becomes_tied_edge():
    rows = [edge(0, 1, 1), edge(0, 1, 0)]
    result = MODULE.summarize(rows, bootstrap=10, seed=42)
    assert result["tied_edges"] == 1
    assert result["tied_edge_rate"] == 1.0
