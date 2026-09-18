"""Portable regressions for the prompt-wise acceptance gate.

Synthetic throughout: a two-objective panel, three prompts, eight occurrences,
built in a temporary directory. No pod paths, no model weights, no forward pass
-- the candidate's log probabilities are supplied, which is the mode the gate
exposes for exactly this reason. The cases are the audit's acceptance tests for
D01 (the manuscript's predicate is universal, not a coverage fraction), D02 (a
requested fit that cannot be evaluated is a rejection, and a non-finite nMSE
never promotes) and D03 (a beta that disagrees with the solved target, and a
candidate with no weights, both fail closed).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SNAP = Path(__file__).resolve().parents[1] / "analysis/sub_20260914/code_snapshot_20260917_union"
GATE = SNAP / "nbpo_local_gate.py"
POOL, NPROMPT = 8, 3
TOL = 1e-8
SOLVER = SNAP / "solve_pros4_targets_uw1.py"


def _objectives() -> int:
    """The panel's objective count, read from the module the gate will import.

    Hardcoding it made the fixture disagree with the module and the shape check
    -- correctly -- refused the shard. The count belongs to the solver.
    """
    spec = importlib.util.spec_from_file_location("gate_solver_probe", SOLVER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return len(mod.OBJECTIVES)


K = _objectives()


def _solver(tmp: Path):
    spec = importlib.util.spec_from_file_location("gate_solver_src", SOLVER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _panel(tmp: Path, beta: float = 0.25, margin: float = 0.30):
    """A panel whose reference triangle is zero, so d = 0 and the surplus is V(p)."""
    solver = _solver(tmp)
    pids = ["p%d" % i for i in range(NPROMPT)]

    scores = tmp / "scores"
    d = scores / "shard0"
    d.mkdir(parents=True)
    # occurrence 0 LOSES to the reference bank and the rest win, so shifting a
    # prompt's mass onto occurrence 0 drives that prompt's surplus negative. A
    # constant payoff would make every mixture score the same and the fixture
    # could not express a mixed array at all.
    LR = np.full((K, NPROMPT, POOL, POOL), margin)
    LR[:, :, 0, :] = -margin
    RR = np.zeros((K, NPROMPT, POOL, POOL))
    LL = np.zeros((K, NPROMPT, POOL, POOL))
    npz = d / "chunk0000.npz"
    np.savez(npz, prompt_ids=np.array(pids), A_policy=LR, A_ref=RR,
             A_LL=LL, A_LR=LR, A_RR=RR)
    (d / "chunk0000.manifest.json").write_text(json.dumps({
        "prompts": NPROMPT, "sha256": solver.file_hash(npz),
        "bank_ids": {"A_policy": ["learner", "comparator"],
                     "A_ref": ["comparator", "comparator"],
                     "A_LL": ["learner", "learner"],
                     "A_LR": ["learner", "comparator"],
                     "A_RR": ["comparator", "comparator"]}}))
    (d / "complete_shard0.json").write_text(json.dumps({
        "tensor_role_schema": "lr_rr_v2", "gpm_teacher": {}, "bt_teacher": "absent",
        "reference_construction": "independent"}))

    targets = tmp / "targets" / "t1"
    (targets / "dev").mkdir(parents=True)
    np.savez(targets / "dev_per_prompt.npz", prompt_ids=np.array(pids),
             pi=np.full((NPROMPT, POOL), 1.0 / POOL),
             weights=np.ones((K, NPROMPT)), g=np.zeros((NPROMPT, POOL)),
             min_surplus=np.full(NPROMPT, margin),
             identity_residual=np.zeros(NPROMPT))
    (targets / "complete.json").write_text(json.dumps({"beta": beta, "eta": 1.0,
                                                       "aggregation": "prompt_wise_nash"}))
    (targets / "dev" / "complete.json").write_text(json.dumps({
        "n_prompts": NPROMPT, "solver_solution_sha256": "0" * 64, "pairs_sha256": "0" * 64}))

    ckpt = tmp / "cand"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"weights")
    (ckpt / "config.json").write_text("{}")
    parent = tmp / "parent"
    parent.mkdir()
    (parent / "model.safetensors").write_bytes(b"parent")
    return scores, targets, ckpt, parent, pids


def _logprobs(tmp: Path, pids, per_prompt_shift):
    """per_prompt_shift[i] tilts prompt i's mass toward occurrence 0."""
    parent, cand = {}, {}
    for i, pid in enumerate(pids):
        for j in range(POOL):
            key = "%s:learner:%d" % (pid, j)
            parent[key] = 0.0
            cand[key] = float(per_prompt_shift[i] if j == 0 else 0.0)
    path = tmp / "lp.json"
    path.write_text(json.dumps({"parent": parent, "candidate": cand}))
    return path


def _state(tmp: Path, nmse):
    path = tmp / "trainer_state.json"
    hist = [] if nmse is None else [{"eval_nbpo/nmse": nmse}]
    path.write_text(json.dumps({"log_history": hist}))
    return path


def _run(tmp, scores, targets, ckpt, parent, lp, *extra, profile="paper-nbpo"):
    out = tmp / "gate.json"
    cmd = [sys.executable, str(GATE), "--candidate", str(ckpt), "--parent", str(parent),
           "--targets", str(targets), "--pool", "unused", "--scores", str(scores),
           "--solver-module", str(SOLVER),
           "--split", "dev", "--profile", profile, "--logprobs", str(lp),
           "--out", str(out), *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          env={"PYTHONPATH": "%s:%s" % (SNAP.parents[2], SNAP), "PATH": "/usr/bin:/bin"})
    rec = json.loads(out.read_text()) if out.exists() else None
    return proc.returncode, rec, proc.stderr


@pytest.fixture
def panel(tmp_path):
    return (tmp_path,) + _panel(tmp_path)


def test_paper_predicate_rejects_a_mixed_surplus_array(panel):
    """D01: two positive prompts and one negative must NOT promote."""
    tmp, scores, targets, ckpt, parent, pids = panel
    # prompt 2 is pushed onto the losing occurrence; the other two are left alone
    lp = _logprobs(tmp, pids, [0.0, 0.0, 12.0])
    rc, rec, _ = _run(tmp, scores, targets, ckpt, parent, lp)
    assert rec is not None
    assert rec["profile"] == "paper-nbpo"
    assert rec["accepted"] is False
    assert rc != 0
    assert rec["promotion"] is None
    assert rec["measured"]["prompts_failing_the_local_predicate"] >= 1


def test_paper_predicate_accepts_when_every_prompt_clears(panel):
    tmp, scores, targets, ckpt, parent, pids = panel
    lp = _logprobs(tmp, pids, [0.0, 0.0, 0.0])      # uniform: surplus is the full margin
    rc, rec, err = _run(tmp, scores, targets, ckpt, parent, lp)
    assert rec is not None, err
    assert rec["accepted"] is True
    assert rc == 0
    assert min(r["min_surplus"] for r in rec["per_prompt"]) > TOL


def test_coverage_profile_is_labelled_as_not_the_contract(panel):
    tmp, scores, targets, ckpt, parent, pids = panel
    lp = _logprobs(tmp, pids, [0.0, 0.0, 12.0])
    rc, rec, _ = _run(tmp, scores, targets, ckpt, parent, lp, profile="coverage")
    assert rec["is_the_manuscript_contract"] is False
    assert "RELAXED" in rec["predicate"]


@pytest.mark.parametrize("nmse", [None, float("nan"), float("inf"), float("-inf"), -1.0, 1.2])
def test_unusable_or_out_of_range_nmse_never_promotes(panel, nmse):
    """D02: a requested fit that is missing or non-finite is a rejection."""
    tmp, scores, targets, ckpt, parent, pids = panel
    lp = _logprobs(tmp, pids, [0.0, 0.0, 0.0])
    state = _state(tmp, nmse)
    rc, rec, _ = _run(tmp, scores, targets, ckpt, parent, lp,
                      "--nmse-from", str(state), profile="paper-global")
    assert rec["accepted"] is False
    assert rec["promotion"] is None
    assert rc != 0


def test_global_profile_accepts_a_finite_in_range_nmse(panel):
    tmp, scores, targets, ckpt, parent, pids = panel
    lp = _logprobs(tmp, pids, [0.0, 0.0, 0.0])
    state = _state(tmp, 0.8)
    rc, rec, err = _run(tmp, scores, targets, ckpt, parent, lp,
                        "--nmse-from", str(state), profile="paper-global")
    assert rec is not None, err
    assert rec["checks"]["fit"] is True
    assert rec["accepted"] is True


def test_beta_disagreeing_with_the_solved_target_fails_closed(panel):
    """D03: the game is the one that was solved, not the one a flag names."""
    tmp, scores, targets, ckpt, parent, pids = panel
    lp = _logprobs(tmp, pids, [0.0, 0.0, 0.0])
    rc, rec, err = _run(tmp, scores, targets, ckpt, parent, lp, "--beta", "0.05")
    assert rec is None
    assert rc != 0
    assert "disagrees" in err


def test_candidate_without_weights_is_not_fingerprinted(panel):
    """D03: a path with no weights must fail rather than digest an empty set."""
    tmp, scores, targets, ckpt, parent, pids = panel
    lp = _logprobs(tmp, pids, [0.0, 0.0, 0.0])
    empty = tmp / "nonexistent"
    rc, rec, err = _run(tmp, scores, targets, empty, parent, lp,
                        "--promote-to", str(tmp / "policy"))
    assert rc != 0
    assert not (tmp / "policy").exists()


def test_shard_count_is_inferred_not_assumed_four(panel):
    tmp, scores, targets, ckpt, parent, pids = panel
    lp = _logprobs(tmp, pids, [0.0, 0.0, 0.0])
    rc, rec, err = _run(tmp, scores, targets, ckpt, parent, lp)
    assert rec is not None, err
    assert rec["score_shards"] == 1
