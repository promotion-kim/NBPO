"""Re-run the HarmBench lane under a start method its dependencies allow.

Both jobs/harmbench_primary_v1 and _v2 died at vLLM engine start with "CUDA
driver initialization failed". That is not transient and not the GPU: the cause
is ordering. evaluate_responses.harmbench() calls load_harmbench_helpers()
before constructing the LLM, and HarmBench's own eval_utils.py imports torch and
touches CUDA while doing so. vLLM v1 then forks its engine core, and a forked
child cannot re-initialize CUDA in a process where the parent already has.

Reproduced directly: a forked child raises "Cannot re-initialize CUDA in forked
subprocess" after loading those helpers and succeeds without loading them, and
the same child spawned instead of forked succeeds either way. So the fix is
VLLM_WORKER_MULTIPROC_METHOD=spawn, which changes how the engine process starts
and nothing about what is scored.

Both failed job directories are left exactly as they are, as the evidence. The
re-run uses new job and output names with otherwise identical arguments, and
this script refuses to start unless the recorded failure really is that error
and produced no scores.
"""
from __future__ import annotations

import argparse
import datetime
import os
import json
import re
import subprocess
from pathlib import Path

ROOT = Path("/work/nbpo_repair_20260909")
PREFIX = "scripts.experiments.nbpo_repair_20260909."
HARMBENCH = "/work/hf_cache/hub/models--cais--HarmBench-Llama-2-13b-cls/snapshots/bda705349d1144fa618770bea64d99ce54e3835b"
SIGNATURE = "CUDA driver initialization failed"
START_METHOD = "spawn"  # HarmBench's eval_utils initializes CUDA before vLLM forks


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
    environment = dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD="spawn")
    if (ROOT / "jobs" / job).exists():
        raise ValueError(f"{job} already exists; choose a fresh name")
    command = ["python3", "-m", PREFIX + "artifact_job", "--root", str(ROOT), "--job", job,
               "--gpu", str(gpu), "--stack", "harmbench", "--",
               "python3", "-m", PREFIX + "evaluate_responses", "--root", str(ROOT),
               "--mode", "harmbench", "--labels", *labels,
               "--harmbench-model", HARMBENCH,
               "--harmbench-repo", str(ROOT / "external/HarmBench"),
               "--out", str(out)]
    subprocess.run(command, cwd=ROOT / "code", check=True, env=environment)
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
    evidence = [confirm_transient(job) for job in ("harmbench_primary_v1", "harmbench_primary_v2")]
    done = [run("harmbench_primary_v3", 2, ("wbc_primary_v1", "mse_primary_v1"),
                ROOT / "evaluations/harmbench_primary_v3")]
    done.append(run("harmbench_base_v1", 2, ("base",), ROOT / "evaluations/harmbench_base_v1"))
    (directory / "complete.json").write_text(json.dumps(
        {"completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "recovered_from": evidence, "steps": done,
         "note": "The failed job directory is retained unmodified as the evidence."}, indent=2))
    print(json.dumps({"recovered": done}, indent=2), flush=True)


if __name__ == "__main__":
    main()
