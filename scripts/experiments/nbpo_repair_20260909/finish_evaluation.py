"""Finish the two planned jobs the aborted lane never reached, then close the chain.

The evaluation controller runs its scoring lanes in parallel and raises when any
lane fails, so lane 2's HarmBench failure also stopped teacher_rm shard 2 and the
teacher aggregate that follows all four shards. Those two jobs are run here from
the controller's own frozen plan.json, with the arguments it recorded and no
changes.

The chain's failure.json is preserved under a new name rather than deleted, and
the complete.json written in its place names every substitution, so a reader can
see that HarmBench was scored by harmbench_primary_v3 after two recorded
failures rather than by the job the plan named.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/work/nbpo_repair_20260909")
PREFIX = "scripts.experiments.nbpo_repair_20260909."
CHAIN = ROOT / "controllers/evaluation_chain_v1"
SUBSTITUTIONS = {"harmbench_primary_v1": "harmbench_primary_v3"}


def exit_code(job):
    path = ROOT / "jobs" / job / "exit.json"
    return json.loads(path.read_text())["exit_code"] if path.exists() else None


def run_planned(spec):
    name = spec["name"]
    if exit_code(name) == 0:
        return {"job": name, "skipped": "already succeeded"}
    if (ROOT / "jobs" / name).exists():
        raise ValueError(f"{name} already has a job directory; inspect it before re-running")
    command = ["python3", "-m", PREFIX + "artifact_job", "--root", str(ROOT), "--job", name,
               "--gpu", spec["gpu"], "--stack", spec["stack"], "--", *spec["command"]]
    subprocess.run(command, cwd=ROOT / "code", check=True,
                   env=dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD="spawn"))
    if exit_code(name) != 0:
        raise ValueError(f"{name} failed; stop and diagnose")
    return {"job": name, "seconds": json.loads((ROOT / "jobs" / name / "exit.json").read_text())["seconds"]}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    plan = json.loads((CHAIN / "plan.json").read_text())

    while exit_code("harmbench_primary_v3") is None:
        if not (ROOT / "jobs/harmbench_primary_v3").exists():
            raise ValueError("The HarmBench recovery has not been launched")
        time.sleep(20)
    if exit_code("harmbench_primary_v3") != 0:
        raise ValueError("HarmBench still failing; diagnose before closing the chain")

    done = []
    for lane in plan["scoring_lanes"].values():
        for spec in lane:
            if spec["name"] in SUBSTITUTIONS:
                continue
            done.append(run_planned(spec))
    done.append(run_planned(plan["teacher_aggregate"]))

    planned = ([job["name"] for job in plan["validation"] + plan["generation"]]
               + [job["name"] for lane in plan["scoring_lanes"].values() for job in lane]
               + [plan["teacher_aggregate"]["name"]])
    outstanding = [name for name in planned
                   if exit_code(SUBSTITUTIONS.get(name, name)) != 0]
    if outstanding:
        raise ValueError(f"Not every planned job has a successful run: {outstanding}")

    failure = CHAIN / "failure.json"
    if failure.exists():
        failure.rename(CHAIN / "failure.harmbench_lane_v1.json")
    (CHAIN / "complete.json").write_text(json.dumps({
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "labels": plan["labels"], "paid_judge_api_calls": 0,
        "completed_by": "finish_evaluation.py after the lane-2 HarmBench failure",
        "substitutions": SUBSTITUTIONS,
        "substitution_reason": ("harmbench_primary_v1 and _v2 both died at vLLM engine start: "
                                "HarmBench's eval_utils.py initializes CUDA in the parent before "
                                "vLLM forks its engine core. harmbench_primary_v3 is the same "
                                "command with VLLM_WORKER_MULTIPROC_METHOD=spawn. Both failed job "
                                "directories and the chain's original failure.json are retained."),
        "recovered_jobs": done,
        "remaining": "Follow-up queue, result audit and manuscript integration; not campaign acceptance",
    }, indent=2) + "\n")
    print(json.dumps({"closed": True, "recovered": done}, indent=2), flush=True)


if __name__ == "__main__":
    main()
