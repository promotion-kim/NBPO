"""A04 acceptance: a tiny synthetic stage, with the trainer stubbed.

The audit asks for a stage that invokes the gate and then hands the returned
policy to whatever comes next: candidate passes -> the next stage uses the
candidate; candidate fails, or no decision exists -> the next stage uses the
parent, and nothing evaluates the candidate. The trainer is replaced by a stub
module that writes a checkpoint, so no GPU, no weights download and no forward
pass are involved; the gate runs for real on cached log probabilities.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from test_nbpo_local_gate import (GATE, SNAP, SOLVER, _logprobs, _panel, _state)

CONTROLLER = SNAP / "nbpo_stage_controller.py"


def _stub_trainer(tmp: Path, ckpt: Path, fail: bool = False):
    """A module invocable as -m that stands in for the real trainer."""
    pkg = tmp / "stub_trainer"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "__main__.py").write_text(textwrap.dedent("""
        import sys
        from pathlib import Path
        if %r:
            sys.exit(7)
        d = Path(%r)
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.safetensors").write_bytes(b"stub")
        (d / "config.json").write_text("{}")
        print("stub trainer wrote", d)
        """ % (fail, str(ckpt))))
    return pkg


def _stage(tmp, scores, targets, ckpt, parent, lp, *extra, profile="paper-nbpo",
           trainer="stub_trainer", skip_train=False):
    out = tmp / "stage.json"
    cmd = [sys.executable, str(CONTROLLER), "--config", str(tmp / "cfg.yaml"),
           "--candidate", str(ckpt), "--parent", str(parent),
           "--targets", str(targets), "--scores", str(scores),
           "--solver-module", str(SOLVER), "--pool", "unused",
           "--split", "dev", "--profile", profile, "--nproc", "1",
           "--trainer", trainer, "--gate", str(GATE), "--out", str(out), *extra]
    if skip_train:
        cmd.append("--skip-train")
    (tmp / "cfg.yaml").write_text("stub: true\n")
    env = {"PYTHONPATH": "%s:%s:%s" % (tmp, SNAP.parents[2], SNAP), "PATH": "/usr/bin:/bin"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    rec = json.loads(out.read_text()) if out.exists() else None
    return proc.returncode, rec, proc.stdout + proc.stderr


@pytest.fixture
def panel(tmp_path):
    return (tmp_path,) + _panel(tmp_path)


def _patch_gate_logprobs(tmp, lp):
    """The controller calls the gate without --logprobs, so give it a wrapper."""
    wrapper = tmp / "gate_wrapper.py"
    wrapper.write_text(textwrap.dedent("""
        import runpy, sys
        sys.argv = sys.argv + ["--logprobs", %r]
        sys.argv[0] = %r
        runpy.run_path(%r, run_name="__main__")
        """ % (str(lp), str(GATE), str(GATE))))
    return wrapper


def test_acceptance_hands_the_candidate_to_the_next_stage(panel):
    tmp, scores, targets, ckpt, parent, pids = panel
    lp = _logprobs(tmp, pids, [0.0, 0.0, 0.0])          # every prompt clears
    _stub_trainer(tmp, ckpt)
    rc, rec, log = _stage(tmp, scores, targets, ckpt, parent, lp,
                          "--gate", str(_patch_gate_logprobs(tmp, lp)))
    assert rec is not None, log
    assert rec["accepted"] is True
    assert str(ckpt) in rec["policy_for_the_next_stage"] or \
        "accepted" in rec["policy_for_the_next_stage"]
    assert rc == 0


def test_rejection_hands_the_parent_to_the_next_stage(panel):
    tmp, scores, targets, ckpt, parent, pids = panel
    lp = _logprobs(tmp, pids, [0.0, 0.0, 12.0])          # one prompt fails
    _stub_trainer(tmp, ckpt)
    rc, rec, log = _stage(tmp, scores, targets, ckpt, parent, lp,
                          "--gate", str(_patch_gate_logprobs(tmp, lp)))
    assert rec is not None, log
    assert rec["accepted"] is False
    assert rec["policy_for_the_next_stage"] == str(parent)
    assert rc != 0, "a rejection must stop a shell pipeline"


def test_no_decision_is_a_rejection_not_a_pass(panel):
    tmp, scores, targets, ckpt, parent, pids = panel
    lp = _logprobs(tmp, pids, [0.0, 0.0, 0.0])
    _stub_trainer(tmp, ckpt)
    missing = tmp / "no_such_gate.py"
    rc, rec, log = _stage(tmp, scores, targets, ckpt, parent, lp, "--gate", str(missing))
    assert rec is not None, log
    assert rec["accepted"] is False
    assert rec["policy_for_the_next_stage"] == str(parent)
    assert "missing decision" in rec["reason"] or "no decision" in rec["reason"]
    assert rc != 0


def test_a_failed_trainer_leaves_the_parent(panel):
    tmp, scores, targets, ckpt, parent, pids = panel
    lp = _logprobs(tmp, pids, [0.0, 0.0, 0.0])
    _stub_trainer(tmp, ckpt, fail=True)
    rc, rec, log = _stage(tmp, scores, targets, tmp / "never_written", parent, lp)
    assert rec is not None, log
    assert rec["accepted"] is False
    assert rec["policy_for_the_next_stage"] == str(parent)
    assert rc != 0


def test_a_candidate_without_weights_is_not_judged(panel):
    tmp, scores, targets, ckpt, parent, pids = panel
    lp = _logprobs(tmp, pids, [0.0, 0.0, 0.0])
    empty = tmp / "empty_ckpt"
    empty.mkdir()
    rc, rec, log = _stage(tmp, scores, targets, empty, parent, lp, skip_train=True)
    assert rec is not None, log
    assert rec["accepted"] is False
    assert "no weights" in rec["reason"]
    assert rec["policy_for_the_next_stage"] == str(parent)
