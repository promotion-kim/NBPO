"""One stage of the prompt-wise pipeline: fit a candidate, gate it, hand on a policy.

The audit's standing finding was that nothing in the campaign called the gate.
The traced route was

    job generator -> torch.distributed.run -> run_mnpo -> train() -> save_model()

and whatever came after read the saved checkpoint directly, so a candidate that
Algorithm 1 would have rejected was evaluated as though it had been accepted.
Writing a gate did not fix that; something has to invoke it and then use what it
returns. This is that something.

The stage is:

  1 train      run the trainer on the declared config, unless --skip-train and a
                checkpoint is already there
  2 monitor    locate the candidate and the fixed certified development
                monitoring subset the gate will read, and verify they exist
                before spending anything on the decision
  3 gate       call nbpo_local_gate.py under the declared profile
  4 promote    on acceptance the gate has written a versioned accepted directory;
                this records which policy the NEXT stage must use
  5 hand on    write stage.json naming the accepted-or-parent policy, and exit
                non-zero when the candidate was rejected so a shell pipeline
                stops rather than continuing on a checkpoint that failed

The policy this returns is the ONLY thing a downstream evaluator may load. That
is the whole point: after a rejection the parent is what gets evaluated, and the
stage reports no improvement rather than a smaller one.

The Global Nash control keeps its own aggregate evaluator. Passing
--profile paper-global here runs the control's contract; the default is the
NBPO-family predicate. The two are not interchangeable and the record says which
one produced a decision.

This controller exists from 2026-09-18. It does not make earlier runs gated.
Every number reported before it came from an ungated projection, and a stage
record is the only evidence that a checkpoint passed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd, label, log, env=None):
    """Run one step, streaming into the stage log, and return its exit code."""
    with log.open("a") as stream:
        stream.write("\n===== %s =====\n%s\n" % (label, " ".join(map(str, cmd))))
        stream.flush()
        proc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT,
                              env=env or os.environ.copy())
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the trainer config for this stage")
    ap.add_argument("--candidate", required=True,
                    help="where the trainer writes, and what the gate judges")
    ap.add_argument("--parent", required=True,
                    help="the policy retained if the candidate is rejected")
    ap.add_argument("--targets", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--solver-module", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--profile", default="paper-nbpo",
                    choices=("paper-nbpo", "paper-global", "coverage"))
    ap.add_argument("--surplus-tol", type=float, default=1e-8)
    ap.add_argument("--nmse-max", type=float, default=1.0)
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--skip-train", action="store_true",
                    help="use the checkpoint already at --candidate; the decision is "
                         "still made and still recorded")
    ap.add_argument("--trainer", default="mnpo_scripts.run_mnpo")
    ap.add_argument("--gate", default=str(HERE / "nbpo_local_gate.py"))
    ap.add_argument("--out", required=True, help="stage.json")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".log")
    record = {
        "stage": "prompt_wise",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": args.config, "candidate": args.candidate, "parent": args.parent,
        "profile": args.profile,
        "not_retroactive": ("this controller exists from 2026-09-18; a checkpoint is gated "
                            "only if a stage record names it"),
        "steps": [],
    }

    def finish(policy, accepted, why, code):
        record.update({"accepted": accepted, "policy_for_the_next_stage": policy,
                       "reason": why,
                       "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        out.write_text(json.dumps(record, indent=1) + "\n")
        print(json.dumps({k: record[k] for k in
                          ("accepted", "policy_for_the_next_stage", "reason")}, indent=1))
        return code

    # ---- 1 train ----
    if args.skip_train:
        record["steps"].append({"step": "train", "skipped": True})
    else:
        rc = run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                  "--nnodes=1", "--nproc_per_node=%d" % args.nproc,
                  "-m", args.trainer, args.config], "1_train", log)
        record["steps"].append({"step": "train", "returncode": rc})
        if rc != 0:
            return finish(args.parent, False,
                          "the trainer failed; there is no candidate to judge and the "
                          "parent stands", 2)

    # ---- 2 monitor ----
    cand = Path(args.candidate)
    weights = [f for f in cand.rglob("*") if f.suffix in (".safetensors", ".bin")] \
        if cand.exists() else []
    monitor = Path(args.targets) / ("%s_per_prompt.npz" % args.split)
    record["steps"].append({"step": "monitor", "candidate_exists": cand.exists(),
                            "weight_files": len(weights),
                            "monitoring_subset": str(monitor),
                            "monitoring_subset_exists": monitor.exists()})
    if not weights:
        return finish(args.parent, False,
                      "the candidate holds no weights, so there is nothing to judge", 2)
    if not monitor.exists():
        return finish(args.parent, False,
                      "the certified development monitoring subset is absent; the "
                      "decision cannot be made and the parent stands", 2)

    # ---- 3 gate ----
    gate_out = out.with_name(out.stem + ".gate.json")
    cmd = [sys.executable, args.gate,
           "--candidate", str(cand), "--parent", args.parent,
           "--targets", args.targets, "--scores", args.scores,
           "--solver-module", args.solver_module, "--pool", args.pool,
           "--split", args.split, "--profile", args.profile,
           "--surplus-tol", str(args.surplus_tol), "--nmse-max", str(args.nmse_max),
           "--promote-to", str(Path(args.candidate).parent /
                               ("%s_accepted" % Path(args.candidate).name)),
           "--out", str(gate_out)]
    state = cand / "trainer_state.json"
    if args.profile == "paper-global" and state.exists():
        cmd += ["--nmse-from", str(state)]
    rc = run(cmd, "3_gate", log)
    record["steps"].append({"step": "gate", "returncode": rc, "record": str(gate_out)})

    if not gate_out.exists():
        # No decision is not a pass. A gate that could not run leaves the stage
        # with nothing to stand on, and the parent is what the next stage sees.
        return finish(args.parent, False,
                      "the gate produced no decision record (exit %d); a missing decision "
                      "is treated as a rejection" % rc, 2)
    decision = json.loads(gate_out.read_text())
    record["gate"] = {k: decision.get(k) for k in
                      ("profile", "predicate", "accepted", "checks", "measured",
                       "promotion", "is_the_manuscript_contract")}

    # ---- 4/5 promote and hand on ----
    if decision.get("accepted"):
        promoted = (decision.get("promotion") or {}).get("symlink") or args.candidate
        return finish(promoted, True,
                      "the candidate passed %s and is the policy for the next stage"
                      % decision.get("predicate"), 0)
    return finish(args.parent, False,
                  "the candidate failed %s; the parent is retained and is what the next "
                  "stage and the evaluator must use" % decision.get("predicate"), 3)


if __name__ == "__main__":
    raise SystemExit(main())
