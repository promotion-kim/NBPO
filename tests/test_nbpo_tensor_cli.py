"""The preference-tensor CLI, run end to end.

`build_preference_tensor` had a `NameError` on the line that writes its
provenance block: it called `implementation_contract()` without importing it, so
the CLI aborted the moment it tried to write `meta.json`. Every existing test
either checked `--help` or called the pure helpers directly, so nothing executed
`main()` and the bug survived a commit.

This runs the real CLI over a small hand-built pool and asserts on the artifact
it produces -- which is the only thing that would have caught it.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).parents[1]
ENV = {**os.environ, "PYTHONPATH": str(REPO)}
OBJECTIVES = ["instruction_following", "truthfulness"]
RUBRIC = REPO / "training_configs/nbpo/objectives/ultrafeedback_v2.yaml"

PROMPTS = [f"p{i}" for i in range(3)]
LEARNER_SEEDS = ["s101", "s102", "s103", "s104"]
REF_SEEDS = ["r201", "r202", "r203", "r204"]


def _responses(tmp, seed, tag):
    path = tmp / f"{tag}_{seed}.json"
    path.write_text(json.dumps([
        {"prompt_id": p, "prompt": f"prompt text {p}",
         "generated_text": f"response from {seed} for {p}"} for p in PROMPTS]))
    return f"{seed}={path}"


def _verdicts(tmp):
    """A complete bank: both presentation orders for every cross pair and every
    unordered reference pair, which is what the builder requires."""
    rows = []
    rng = np.random.default_rng(0)

    def add(pool, lid, cid, lseed, cseed, pid, obj):
        for order in ("learner_first", "comparator_first"):
            win = float(rng.choice([0.0, 0.5, 1.0]))
            rows.append({
                "prompt_id": pid, "objective": obj, "learner_pool": pool,
                "learner_response_id": lid, "comparator_response_id": cid,
                "learner_seed": lseed, "comparator_seed": cseed,
                "presentation_order": order, "policy_win": win, "valid": True,
                "judge_model": "unit-test", "rubric_version": 2, "attempt": 0,
                "raw_judge_output": "[[TIE]]", "parsed_verdict": "TIE",
                "comparison_content_hash": f"{pool}{lid}{cid}{pid}{obj}{order}",
            })

    for pid in PROMPTS:
        for obj in OBJECTIVES:
            for ls in LEARNER_SEEDS:
                for rs in REF_SEEDS:
                    add("policy", f"policy:{ls}", f"ref:{rs}", ls, rs, pid, obj)
            for i, a in enumerate(REF_SEEDS):
                for b in REF_SEEDS[i + 1:]:
                    add("reference", f"ref:{a}", f"ref:{b}", a, b, pid, obj)
    path = tmp / "verdicts.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tensor_cli")
    out = tmp / "tensor"
    cmd = [sys.executable, "-m", "scripts.nbpo.build_preference_tensor",
           "--verdicts", str(_verdicts(tmp)),
           "--policy-files", *[_responses(tmp, s, "policy") for s in LEARNER_SEEDS],
           "--reference-files", *[_responses(tmp, s, "ref") for s in REF_SEEDS],
           "--objectives", ",".join(OBJECTIVES),
           "--objectives-config", str(RUBRIC),
           "--out-dir", str(out)]
    proc = subprocess.run(cmd, cwd=REPO, env=ENV, capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return out


def test_cli_writes_all_three_artifacts(built):
    for name in ("tensor_policy.npz", "tensor_ref.npz", "meta.json"):
        assert (built / name).exists(), name


def test_meta_carries_the_implementation_contract(built):
    """The regression: this block is what the missing import blew up on."""
    meta = json.loads((built / "meta.json").read_text())
    assert meta["implementation_type"] == "finite_pool_one_shot_neural_realization"
    assert meta["dual_policy_representation"] == "finite_response_distribution"
    assert meta["neural_fits_per_outer_stage"] == 1


def test_meta_declares_the_reference_construction(built):
    """The solver refuses a tensor that does not say how its reference was built."""
    meta = json.loads((built / "meta.json").read_text())
    assert meta["reference_construction"] in ("shared_pool", "independent_samples")
    assert meta["objectives"] == OBJECTIVES


def test_tensor_shapes_and_range(built):
    A = np.load(built / "tensor_policy.npz")["A"]
    R = np.load(built / "tensor_ref.npz")["A"]
    assert A.shape == (len(OBJECTIVES), len(PROMPTS), 4, 4)
    assert R.shape == (len(OBJECTIVES), len(PROMPTS), 4, 4)
    assert np.isfinite(A).all() and np.abs(A).max() <= 0.5 + 1e-9


def test_reference_tensor_is_exactly_skew_symmetric_with_a_zero_diagonal(built):
    R = np.load(built / "tensor_ref.npz")["A"]
    assert np.abs(R + np.swapaxes(R, -1, -2)).max() == 0.0
    assert np.abs(np.diagonal(R, axis1=-2, axis2=-1)).max() == 0.0


def test_the_solver_accepts_what_the_builder_wrote(built, tmp_path):
    """The two CLIs have to agree about the artifact, not just each be runnable."""
    out = tmp_path / "solution"
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.nbpo.solve_nbpo_dual",
         "--tensor-dir", str(built), "--out-dir", str(out),
         "--beta", "0.25", "--eta", "1.0", "--gamma", "0.5",
         "-M", "50", "-R", "2"],
        cwd=REPO, env=ENV, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    sol = json.loads((out / "solution.json").read_text())
    assert sol["representation"] == "adaptive_game" and sol["aggregation"] == "nash"
    assert sol["target_log_ratio_identity_holds"] is True


# --- the solve -> pair-target handoff --------------------------------------

def test_opponent_temperature_travels_only_with_a_representation_that_has_one():
    """`build_nbpo_pairs` reads `config.beta` to stamp `opponent_beta` on every
    row. Only the adaptive game has an opponent temperature; a fixed-reference or
    scalar-reward solve must record None rather than a number it never used, and
    an adaptive-game artifact that lost its beta must be refused rather than
    stamped with a guess.
    """
    import torch
    from mnpo_scripts.nbpo_generic import solve_finite_pool
    from mnpo_scripts.nbpo_representations import (AdaptiveGameRepresentation,
                                                   FixedReferenceRepresentation)
    from tests.test_nbpo_generic_solver import make_pool

    A, A_ref, mu = make_pool(seed=3, K=3, X=24)
    beta = torch.full((3,), 0.25, dtype=torch.float64)
    adaptive = solve_finite_pool(AdaptiveGameRepresentation(A, A_ref, mu, beta),
                                 "nash", eta=1.0, M=40, R=2, gamma=0.5)
    fixed = solve_finite_pool(FixedReferenceRepresentation(A, A_ref, mu),
                              "nash", eta=1.0, M=40, R=2, gamma=0.5)
    assert adaptive.config["beta"] == [0.25, 0.25, 0.25]
    assert fixed.config["beta"] is None


@pytest.mark.parametrize("representation,beta,expected", [
    ("adaptive_game", [0.25, 0.25], [0.25, 0.25]),
    ("fixed_reference", None, None),
    ("bt_reward", None, None),
])
def test_pair_builder_resolves_the_opponent_temperature_per_representation(
        representation, beta, expected):
    from scripts.nbpo.build_nbpo_pairs import resolve_opponent_betas
    got = resolve_opponent_betas({"representation": representation,
                                  "config": {"beta": beta}})
    if expected is None:
        assert got is None
    else:
        assert list(got) == expected


def test_pair_builder_refuses_an_adaptive_artifact_that_lost_its_beta():
    from scripts.nbpo.build_nbpo_pairs import resolve_opponent_betas
    with pytest.raises(KeyError, match="cannot"):
        resolve_opponent_betas({"representation": "adaptive_game", "config": {}})
    # a legacy artifact with no representation field is treated as adaptive_game,
    # which is what it was -- not waved through
    with pytest.raises(KeyError):
        resolve_opponent_betas({"config": {}})
