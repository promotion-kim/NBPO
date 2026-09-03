"""End-to-end pipeline tests (scripts/nbpo).

Covers spec tests 15 (stage gate rejects nonpositive-surplus candidates and
retains the parent), 16 (Nash welfare reported as null when any surplus is
nonpositive), 17 (legacy isolation: the paper-exact path imports no scalar
RM / Bradley-Terry builders and never touches all_rm_scores), plus the
CPU-only dry run and CLI executability.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from mnpo_scripts.nbpo_core import uniform_policy
from scripts.nbpo.eval_game_value import evaluate_game_value
from scripts.nbpo.run_nbpo_stage import apply_gate, check_disjoint, hash_checkpoint_dir

REPO = Path(__file__).parents[1]
ENV = {**os.environ, "PYTHONPATH": str(REPO)}


def _tensors(surplus_sign=+1.0):
    """(A_policy, A_ref, beta) whose min surplus has the requested sign."""
    K, X, I, J = 2, 3, 4, 4
    A_policy = torch.full((K, X, I, J), 0.2 * surplus_sign, dtype=torch.float64)
    A_policy[1] *= 0.5
    A_ref = torch.zeros(K, X, J, J, dtype=torch.float64)  # d = 0 exactly here
    beta = torch.full((K,), 0.25, dtype=torch.float64)
    return A_policy, A_ref, beta


# ---------------------------------------------------------------- test 16 --


def test_nash_welfare_null_when_any_surplus_nonpositive():
    A_policy, A_ref, beta = _tensors(surplus_sign=-1.0)
    out = evaluate_game_value(A_policy, A_ref, beta)
    assert out["min_surplus"] <= 0
    assert out["nash_welfare_defined"] is False
    assert out["nash_welfare"] is None  # never clamped before the log


def test_nash_welfare_defined_when_all_surpluses_positive():
    A_policy, A_ref, beta = _tensors(surplus_sign=+1.0)
    out = evaluate_game_value(A_policy, A_ref, beta)
    assert out["min_surplus"] > 0
    assert out["nash_welfare_defined"] is True
    import math

    expected = sum(math.log(s) for s in out["surplus"])
    assert abs(out["nash_welfare"] - expected) < 1e-9


# ---------------------------------------------------------------- test 15 --


def _checkpoint(path: Path, marker: str):
    path.mkdir(parents=True, exist_ok=True)
    (path / "MARKER.json").write_text(json.dumps({"id": marker}))
    return path


def test_gate_rejects_nonpositive_surplus_and_retains_parent(tmp_path):
    parent = _checkpoint(tmp_path / "pi_t", "parent")
    candidate = _checkpoint(tmp_path / "pi_next.candidate", "candidate")
    parent_hash = hash_checkpoint_dir(parent)
    gate = apply_gate(-0.01, parent, candidate, tmp_path / "pi_next")
    assert gate["accepted"] is False
    assert gate["promoted_path"] == str(parent)
    assert not (tmp_path / "pi_next").exists()      # nothing promoted
    assert candidate.exists()                        # candidate kept for inspection
    assert hash_checkpoint_dir(parent) == parent_hash  # parent untouched
    # boundary: min surplus exactly 0 is also a rejection (Algorithm 1: <= 0)
    gate0 = apply_gate(0.0, parent, candidate, tmp_path / "pi_next")
    assert gate0["accepted"] is False


def test_gate_promotes_atomically_on_positive_surplus(tmp_path):
    parent = _checkpoint(tmp_path / "pi_t", "parent")
    candidate = _checkpoint(tmp_path / "pi_next.candidate", "candidate")
    gate = apply_gate(0.05, parent, candidate, tmp_path / "pi_next")
    assert gate["accepted"] is True
    assert (tmp_path / "pi_next" / "MARKER.json").exists()
    assert not candidate.exists()  # os.replace moved, not copied


def test_disjointness_is_enforced():
    check_disjoint({"a", "b"}, {"c"}, {"d"})
    import pytest

    with pytest.raises(ValueError, match="disjoint"):
        check_disjoint({"a", "b"}, {"b"}, {"d"})
    with pytest.raises(ValueError, match="disjoint"):
        check_disjoint({"a"}, {"c"}, {"c"})


# ---------------------------------------------------------------- test 17 --

FORBIDDEN_SUBSTRINGS = ["all_rm_scores"]
FORBIDDEN_IMPORTS = [
    "on_policy_data_gen",            # scalar RM annotation stack
    "reward_model_annotate",
    "rm_armo", "rm_beaver", "rm_hh",
    "build_os_ronpo_targets",        # scalar-score target builder
    "build_bpo_pairs",               # fixed-reference legacy builder
    "bradley",
]


def test_legacy_isolation_no_scalar_rm_or_bt_imports():
    paths = sorted((REPO / "scripts" / "nbpo").glob("*.py")) + \
        sorted((REPO / "mnpo_scripts").glob("nbpo_*.py"))
    assert len(paths) >= 8, "expected the full paper-exact module set"
    for path in paths:
        text = path.read_text()
        for bad in FORBIDDEN_SUBSTRINGS:
            assert bad not in text, f"{path.name} mentions {bad!r}"
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                for bad in FORBIDDEN_IMPORTS:
                    assert bad.lower() not in stripped.lower(), \
                        f"{path.name} imports the legacy scalar path: {stripped!r}"


# ----------------------------------------------------------- CPU dry run --


def test_cpu_dry_run_completes_without_an_llm(tmp_path):
    workdir = tmp_path / "work"
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.nbpo.run_nbpo_stage",
         "--config", "tests/fixtures/nbpo_toy/config.yaml",
         "--workdir", str(workdir), "--dry-run"],
        cwd=REPO, env=ENV, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    for step in ("rubrics", "disjointness", "judge_train", "tensor_train", "solve_dual",
                 "build_pairs", "train", "eval_monitoring", "gate", "complete"):
        assert f"step={step} status=ok" in proc.stdout, (step, proc.stdout)
    assert "vllm" not in proc.stdout.lower()
    record_files = list(workdir.glob("*_record.json"))
    assert record_files, "gate record must be written"
    record = json.loads(record_files[0].read_text())
    assert record["stage"] == 0 and "monitoring" in record
    assert record["monitoring"]["nash_welfare_defined"] in (True, False)
    solution = json.loads((workdir / "solver" / "solution.json").read_text())
    assert solution["config"]["R"] == 1 and solution["config"]["R_is_approximation"]
    assert solution["kkt_residual"] is not None


def test_every_cli_is_executable():
    for module in ("scripts.nbpo.judge_pairwise_matrix",
                   "scripts.nbpo.build_preference_tensor",
                   "scripts.nbpo.build_nbpo_pairs",
                   "scripts.nbpo.solve_nbpo_dual",
                   "scripts.nbpo.eval_game_value",
                   "scripts.nbpo.run_nbpo_stage"):
        proc = subprocess.run([sys.executable, "-m", module, "--help"],
                              cwd=REPO, env=ENV, capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, (module, proc.stderr[-500:])
