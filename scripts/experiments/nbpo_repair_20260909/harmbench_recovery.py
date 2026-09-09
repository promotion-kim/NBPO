"""Re-run only the HarmBench lane, which died before it computed anything.

jobs/harmbench_primary_v1 failed at vLLM engine start with "CUDA driver
initialization failed", while the two generation engines on GPUs 0 and 1 were
still tearing down. The same dependency stack initializes CUDA on GPU 2 without
complaint now, and the job produced no scores at all, so this is re-running an
infrastructure failure rather than retrying a result.

The failed job directory is left exactly as it is, as the evidence. The re-run
uses new job and output names, identical arguments otherwise, and this script
refuses to start unless the recorded failure really is that CUDA-init error.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
from pathlib import Path

ROOT = Path("/work/nbpo_repair_20260909")
PREFIX = "scripts.experiments.nbpo_repair_20260909."
HARMBENCH = "/work/hf_cache/hub/models--cais--HarmBench-Llama-2-13b-cls/snapshots/bda705349d1144fa618770bea64d99ce54e3835b"
SIGNATURE = "CUDA driver initialization failed"


def confirm_transient(job):
    exit_record = json.loads((ROOT / "jobs" / job / "exit.json").read_text())
    if exit_record["exit_code"] == 0:
        raise ValueError(f"{job} succeeded; nothing to recover")
    log = (ROOT / "jobs" / job / "stdout.log").read_text(errors="ignore")
    if SIGNATURE not in log:
        raise ValueError(f"{job} did not fail with {SIGNATURE!r}; diagnose it instead of re-running")
    if re.search(r"harmful", log) and "summary" in log:
        raise ValueError(f"{job} produced scores; do not re-run a completed evaluation")
    return {"original_job": job, "exit_code": exit_record["exit_code"],
            "failure_signature": SIGNATURE, "seconds": exit_record["seconds"]}


def run(job, gpu, labels, out):
    if (ROOT / "jobs" / job).exists():
        raise ValueError(f"{job} already exists; choose a fresh name")
    command = ["python3", "-m", PREFIX + "artifact_job", "--root", str(ROOT), "--job", job,
               "--gpu", str(gpu), "--stack", "harmbench", "--",
               "python3", "-m", PREFIX + "evaluate_responses", "--root", str(ROOT),
               "--mode", "harmbench", "--labels", *labels,
               "--harmbench-model", HARMBENCH,
               "--harmbench-repo", str(ROOT / "external/HarmBench"),
               "--out", str(out)]
    subprocess.run(command, cwd=ROOT / "code", check=True)
    record = json.loads((ROOT / "jobs" / job / "exit.json").read_text())
    if record["exit_code"] != 0:
        raise ValueError(f"{job} failed again; stop and diagnose")
    return {"job": job, "seconds": record["seconds"], "labels": list(labels), "out": str(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--background", action="store_true")
    args = ap.parse_args()
    directory = ROOT / "controllers" / "harmbench_recovery_v1"
    directory.mkdir(parents=True, exist_ok=True)
    if args.background:
        with (directory / "stdout.log").open("a") as log:
            process = subprocess.Popen(["python3", str(Path(__file__).resolve())],
                                       cwd=ROOT / "code", stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        (directory / "launch.json").write_text(json.dumps({"pid": process.pid}, indent=2))
        print(json.dumps({"recovery_pid": process.pid}), flush=True)
        return
    evidence = confirm_transient("harmbench_primary_v1")
    done = [run("harmbench_primary_v2", 2, ("wbc_primary_v1", "mse_primary_v1"),
                ROOT / "evaluations/harmbench_primary_v2")]
    done.append(run("harmbench_base_v1", 2, ("base",), ROOT / "evaluations/harmbench_base_v1"))
    (directory / "complete.json").write_text(json.dumps(
        {"completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "recovered_from": evidence, "steps": done,
         "note": "The failed job directory is retained unmodified as the evidence."}, indent=2))
    print(json.dumps({"recovered": done}, indent=2), flush=True)


if __name__ == "__main__":
    main()
