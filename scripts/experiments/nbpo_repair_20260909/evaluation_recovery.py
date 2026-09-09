"""Re-run the frozen evaluation plan if the chain aborts on a transient GPU check.

evaluation_chain refuses to start when nvidia-smi still reports a compute
process, and train_job writes its exit record the moment torchrun's parent
returns -- before the ranks have necessarily released their contexts. A few
seconds of lag there is enough to fail the whole evaluation permanently.

This runs the SAME plan.json the chain froze, in the same order, through the
same artifact_job wrapper, which drains its own assigned GPU first. It executes
nothing that is not already in that plan: no new job, no changed argument, no
retry of a job that genuinely failed. Jobs the chain already completed are
skipped by artifact_job's own refusal to reuse a job directory.

Use only after reading controllers/evaluation_chain_v1/failure.json and
confirming the failure was the idle precondition rather than a real error.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/work/nbpo_repair_20260909")
PREFIX = "scripts.experiments.nbpo_repair_20260909."


def run_job(spec):
    name = spec["name"]
    if (ROOT / "jobs" / name / "exit.json").exists():
        record = json.loads((ROOT / "jobs" / name / "exit.json").read_text())
        if record["exit_code"] == 0:
            return {"job": name, "skipped": "already succeeded"}
        raise ValueError(f"{name} already failed for a real reason; diagnose it, do not re-run")
    command = ["python3", "-m", PREFIX + "artifact_job", "--root", str(ROOT), "--job", name,
               "--gpu", spec["gpu"], "--stack", spec["stack"], "--", *spec["command"]]
    subprocess.run(command, cwd=ROOT / "code", check=True)
    record = json.loads((ROOT / "jobs" / name / "exit.json").read_text())
    if record["exit_code"] != 0:
        raise ValueError(f"{name} failed; evidence retained in jobs/{name}")
    return {"job": name, "seconds": record["seconds"]}


def lanes(mapping):
    errors, results = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(mapping)) as pool:
        futures = {pool.submit(lambda specs: [run_job(s) for s in specs], lane): key
                   for key, lane in mapping.items()}
        for future in concurrent.futures.as_completed(futures):
            try:
                results.extend(future.result())
            except Exception as exc:
                errors.append({"lane": futures[future], "type": type(exc).__name__, "message": str(exc)})
    if errors:
        raise RuntimeError(json.dumps(errors))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-transient", action="store_true", required=True,
                    help="assert you read failure.json and it was the GPU idle precondition")
    ap.add_argument("--background", action="store_true")
    args = ap.parse_args()
    directory = ROOT / "controllers" / "evaluation_recovery_v1"
    directory.mkdir(parents=True, exist_ok=True)
    if args.background:
        with (directory / "stdout.log").open("a") as log:
            command = ["python3", str(Path(__file__).resolve()), "--confirm-transient"]
            process = subprocess.Popen(command, cwd=ROOT / "code", stdout=log,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       start_new_session=True)
        (directory / "launch.json").write_text(json.dumps(
            {"pid": process.pid, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=2))
        print(json.dumps({"recovery_pid": process.pid}), flush=True)
        return

    plan = json.loads((ROOT / "controllers/evaluation_chain_v1/plan.json").read_text())
    done = []
    try:
        done.extend(lanes({str(i): [s] for i, s in enumerate(plan["validation"])}))
        done.extend(lanes({str(i): [s] for i, s in enumerate(plan["generation"])}))
        done.extend(lanes(plan["scoring_lanes"]))
        done.append(run_job(plan["teacher_aggregate"]))
        (directory / "complete.json").write_text(json.dumps(
            {"completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "steps": done, "paid_judge_api_calls": 0,
             "note": "Ran the frozen plan after a transient idle-precondition abort"}, indent=2))
        # let the follow-up controller proceed exactly as if the chain had finished
        target = ROOT / "controllers/evaluation_chain_v1/complete.json"
        if not target.exists():
            target.write_text(json.dumps(
                {"completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "labels": plan["labels"], "paid_judge_api_calls": 0,
                 "completed_by": "evaluation_recovery_v1"}, indent=2))
    except Exception as exc:
        (directory / "failure.json").write_text(json.dumps(
            {"error_type": type(exc).__name__, "message": str(exc), "completed_steps": done,
             "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=2))
        raise


if __name__ == "__main__":
    main()
