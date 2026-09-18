"""Pod-data smoke check for the gate; the portable regressions live in tests/.

Superseded as the acceptance evidence by tests/test_nbpo_local_gate.py and
tests/test_nbpo_stage_controller.py, which are synthetic, need no /work data and
run in a clean checkout. This file is kept because it exercises the gate against
the real uw1c solve, which those cannot. It exits non-zero when any recorded
assertion fails.

A04 acceptance: the gate accepts, rejects and never promotes on a bad value.

Runs the real nbpo_local_gate.py against the real uw1c dev solve and the real
score tensors, with the forward pass replaced by supplied log-probabilities, so
the decision logic and the promotion filesystem behaviour are exercised without
competing for a card. Three cases:

  accept   a candidate whose mass moves toward each prompt's solved p*
  reject   a candidate whose mass moves away from it
  refuse   a candidate with a non-finite log-probability

and in every rejecting case the parent must be what the stage returns and
nothing may be promoted.
"""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

CODE = "/work/sub_20260914/code"
T = "/work/uf4_20260910/targets/uw1c_pw_nbpo"
POOL = 8
z = np.load(T + "/dev_per_prompt.npz", allow_pickle=True)
pids = [str(p) for p in z["prompt_ids"]]
pi = np.asarray(z["pi"], dtype=np.float64)

def write_logprobs(path, mode):
    """parent flat; candidate tilted toward (or away from) the solved p*."""
    parent, cand = {}, {}
    for x, pid in enumerate(pids):
        ps = np.clip(pi[x], 1e-12, None)
        for i in range(POOL):
            cid = "%s:learner:%d" % (pid, i)
            parent[cid] = 0.0
            if mode == "toward":
                cand[cid] = float(np.log(ps[i]) - np.log(1.0 / POOL))
            elif mode == "away":
                cand[cid] = float(-(np.log(ps[i]) - np.log(1.0 / POOL)))
            elif mode == "nonfinite":
                cand[cid] = float("nan") if i == 0 else 0.0
    Path(path).write_text(json.dumps({"parent": parent, "candidate": cand}))

def run(mode, promote_dir, coverage_min="0.5"):
    with tempfile.TemporaryDirectory() as d:
        lp = str(Path(d) / "lp.json"); write_logprobs(lp, mode)
        out = str(Path(d) / "gate.json")
        link = str(Path(promote_dir) / ("policy_%s" % mode))
        r = subprocess.run(
            [sys.executable, CODE + "/nbpo_local_gate.py",
             "--candidate", "/work/uf4_20260910/arms/uw1c_pw_nbpo/checkpoint-70",
             "--parent", "/work/models/bases/Qwen2.5-7B-Instruct",
             "--targets", T, "--pool", "uw1c",
             "--scores", "/work/uf4_20260910/scores/uw1c",
             "--solver-module", CODE + "/solve_pros4_targets_uw1c.py",
             "--split", "dev", "--beta", "0.25",
             "--coverage-min", coverage_min, "--nmse-max", "1.0",
             "--logprobs", lp, "--promote-to", link, "--out", out],
            capture_output=True, text=True)
        rec = json.loads(Path(out).read_text()) if Path(out).exists() else None
        return r.returncode, rec, Path(link)

fail = []
with tempfile.TemporaryDirectory() as promote:
    for mode, want_accept in (("toward", True), ("away", False), ("nonfinite", False)):
        rc, rec, link = run(mode, promote)
        if rec is None:
            fail.append("%s: no record written (rc=%d)" % (mode, rc)); continue
        got = rec["accepted"]
        print("  %-10s accepted=%-5s coverage=%.3f nonfinite=%d rc=%d promoted=%s"
              % (mode, got, rec["measured"]["coverage"],
                 rec["measured"]["nonfinite_prompts"], rc,
                 bool(rec.get("promotion"))))
        if got != want_accept:
            fail.append("%s: accepted=%s, wanted %s" % (mode, got, want_accept))
        if not got:
            if rec["policy_after_this_stage"] != rec["parent"]:
                fail.append("%s: a rejection did not return the parent" % mode)
            if rec.get("promotion") is not None or link.exists():
                fail.append("%s: a rejection promoted something" % mode)
            if rc == 0:
                fail.append("%s: a rejection exited 0" % mode)
        else:
            if rec["policy_after_this_stage"] != rec["candidate"]:
                fail.append("%s: an acceptance did not return the candidate" % mode)
            if not link.is_symlink():
                fail.append("%s: an acceptance wrote no symlink" % mode)
            else:
                acc = link.resolve() / "ACCEPTED.json"
                if not acc.exists():
                    fail.append("%s: the accepted dir has no record" % mode)
                else:
                    a = json.loads(acc.read_text())
                    if not a.get("fingerprint"):
                        fail.append("%s: the accepted record has no fingerprint" % mode)
    # a threshold above what the good candidate achieves must reject it
    rc, rec, link = run("toward", promote, coverage_min="1.01")
    print("  %-10s accepted=%-5s (coverage-min 1.01)" % ("threshold", rec["accepted"]))
    if rec["accepted"]:
        fail.append("an unreachable coverage threshold still accepted")

print()
print("A04 VERIFIED" if not fail else "STILL BROKEN: " + "; ".join(fail))
# a recorded assertion that failed has to change the process status, or a caller
# that only checks the exit code reads a broken run as a clean one
raise SystemExit(0 if not fail else 1)
