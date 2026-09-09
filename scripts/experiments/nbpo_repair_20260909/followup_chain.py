"""Everything that still has to happen after the primary evaluation, unattended.

The previous session was cut off mid-run, so this controller is written to
survive losing its client: it waits on the primary evaluation controller's own
completion file, then executes a fixed queue. Nothing here selects a checkpoint,
changes a horizon, retries a failed job, or kills another process. If a step
fails the queue stops and the evidence is kept.

Queue, in order:
  1. base HarmBench under tonight's protocol -- the primary chain scores only the
     two arms, so without this there is no matched base number for that column.
  2. both short-horizon arms of the 2x2 declared in protocols/ before they were run.
  3. generation + scoring for each on the same frozen panel.
  4. pool-drift diagnostics for every checkpoint, base included as the control.
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
SKYWORK = "/work/hf_cache/hub/models--Skywork--Skywork-Reward-V2-Qwen3-8B/snapshots/6f19fdefb933293d4898bdb59a96f7223d998659"
HARMBENCH = "/work/hf_cache/hub/models--cais--HarmBench-Llama-2-13b-cls/snapshots/bda705349d1144fa618770bea64d99ce54e3835b"
SHORT_ARMS = ("wbc_short_primary_v1", "mse_short_primary_v1")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def artifact(name, gpu, stack, module, *arguments):
    """Run one step through the campaign's own job wrapper, for provenance."""
    existing = ROOT / "jobs" / name / "exit.json"
    if existing.exists():
        record = json.loads(existing.read_text())
        if record["exit_code"] == 0:
            return {"job": name, "skipped": "already succeeded"}
        raise ValueError(f"{name} already failed for a real reason; diagnose it, do not re-run")
    if (ROOT / "jobs" / name).exists():
        raise ValueError(f"{name} left a job directory with no exit record; inspect it")
    command = ["python3", "-m", PREFIX + "artifact_job", "--root", str(ROOT), "--job", name,
               "--gpu", str(gpu), "--stack", stack, "--",
               "python3", "-m", PREFIX + module, *map(str, arguments)]
    # HarmBench's official eval_utils.py initializes CUDA when it is imported, before the
    # classifier engine is built, so vLLM's forked engine core cannot re-initialize CUDA and
    # reports it as a missing GPU. Spawning changes how the engine process starts and nothing
    # about what is scored.
    subprocess.run(command, cwd=ROOT / "code", check=True,
                   env=dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD="spawn"))
    record = json.loads((ROOT / "jobs" / name / "exit.json").read_text())
    if record["exit_code"] != 0:
        raise ValueError(f"{name} failed; evidence retained in jobs/{name}")
    return {"job": name, "seconds": record["seconds"]}


def gpus_idle():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    return not out.strip()


def wait_for_idle_gpus(limit_seconds=3600):
    start = time.monotonic()
    while not gpus_idle():
        if time.monotonic() - start > limit_seconds:
            raise ValueError("GPUs still busy; inspect what is running, never kill it")
        time.sleep(20)


def pool_drift(label, model, gpu=1):
    out = ROOT / "analysis_claude" / f"pool_drift_{label}_dev.json"
    if out.exists():
        return {"pool_drift": label, "skipped": "already computed"}
    env = dict(os.environ)
    env.update({"PYTHONPATH": f"{ROOT}/deps_train:{ROOT}/code", "CUDA_VISIBLE_DEVICES": str(gpu),
                "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    log = ROOT / "analysis_claude" / f"pool_drift_{label}.log"
    with log.open("w") as handle:
        subprocess.run(["python3", str(ROOT / "analysis_claude/pool_drift.py"),
                        "--policy", str(model), "--label", label, "--split", "dev"],
                       cwd=ROOT / "code", env=env, check=True, stdout=handle, stderr=subprocess.STDOUT)
    return {"pool_drift": label, "output": str(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--background", action="store_true")
    args = ap.parse_args()
    directory = ROOT / "controllers" / "followup_chain_v1"
    directory.mkdir(parents=True, exist_ok=True)
    if args.background:
        with (directory / "stdout.log").open("a") as log:
            command = ["python3", str(Path(__file__).resolve())]
            process = subprocess.Popen(command, cwd=ROOT / "code", stdout=log,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       start_new_session=True)
        write_json(directory / "launch.json", {"pid": process.pid, "started_utc": now()})
        print(json.dumps({"controller_pid": process.pid, "wait_for": "evaluation_chain_v1"}), flush=True)
        return

    done = []
    try:
        evaluation = ROOT / "controllers/evaluation_chain_v1"
        while not (evaluation / "complete.json").exists():
            if (evaluation / "failure.json").exists():
                raise ValueError("Primary evaluation failed; diagnose it before running follow-ups")
            time.sleep(30)
        write_json(directory / "started.json", {"time_utc": now()})
        wait_for_idle_gpus()

        done.append(artifact("harmbench_base_v1", 2, "harmbench", "evaluate_responses",
                             "--root", ROOT, "--mode", "harmbench", "--labels", "base",
                             "--harmbench-model", HARMBENCH,
                             "--harmbench-repo", ROOT / "external/HarmBench",
                             "--out", ROOT / "evaluations/harmbench_base_v1"))

        declared = json.loads((ROOT / "protocols/short_horizon_prospective_v2.json").read_text())
        if tuple(a["arm"] for a in declared["arms"]) != SHORT_ARMS or any(
                a["horizon_updates"] != 250 for a in declared["arms"]):
            raise ValueError("The declaration does not match the arms about to run")

        for arm in SHORT_ARMS:
            wait_for_idle_gpus()
            if not (ROOT / "jobs" / arm).exists():
                subprocess.run(["python3", "-m", PREFIX + "train_job", "--root", str(ROOT),
                                "--config", str(ROOT / "configs" / f"{arm}.yaml"),
                                "--job", arm], cwd=ROOT / "code", check=True)
            record = json.loads((ROOT / "jobs" / arm / "exit.json").read_text())
            if record["exit_code"] != 0:
                raise ValueError(f"{arm} failed; evidence retained, no restart")
            state = json.loads((ROOT / "arms" / arm / "trainer_state.json").read_text())
            if state["global_step"] != 250:
                raise ValueError(f"{arm} did not reach its declared horizon")
            done.append({"job": arm, "seconds": record["seconds"], "global_step": state["global_step"]})

        for arm in SHORT_ARMS:
            wait_for_idle_gpus()
            done.append(artifact(f"{arm}_generation_v1", 0, "eval", "generate_eval",
                                 "--root", ROOT, "--model", ROOT / "arms" / arm, "--label", arm))
            done.append(artifact(f"skywork_{arm}_v1", 0, "eval", "evaluate_responses",
                                 "--root", ROOT, "--mode", "skywork", "--labels", arm,
                                 "--rm", SKYWORK, "--out", ROOT / f"evaluations/skywork_{arm}_v1"))
            done.append(artifact(f"harmbench_{arm}_v2", 2, "harmbench", "evaluate_responses",
                                 "--root", ROOT, "--mode", "harmbench", "--labels", arm,
                                 "--harmbench-model", HARMBENCH,
                                 "--harmbench-repo", ROOT / "external/HarmBench",
                                 "--out", ROOT / f"evaluations/harmbench_{arm}_v2"))
            done.append(artifact(f"deterministic_{arm}_v1", "cpu", "eval", "evaluate_responses",
                                 "--root", ROOT, "--mode", "deterministic", "--labels", arm,
                                 "--out", ROOT / f"evaluations/deterministic_{arm}_v1"))
            done.append(artifact(f"saferlhf_{arm}_v1", 3, "eval", "evaluate_saferlhf",
                                 "--root", ROOT, "--labels", arm,
                                 "--base-score-dir", ROOT / "evaluations/saferlhf_base_v1/base",
                                 "--encoder", ROOT / "assets/roberta-base",
                                 "--out", ROOT / f"evaluations/saferlhf_{arm}_v1"))
            done.append(artifact(f"xstest_{arm}_unadjudicated_v1", "cpu", "eval", "score_xstest_refusal",
                                 "--root", ROOT, "--labels", arm,
                                 "--output-dir", ROOT / f"evaluations/xstest_{arm}_unadjudicated_v1",
                                 "--unadjudicated-reason",
                                 "Pinned official WildGuard unavailable: existing authorized identity returns HTTP403; no terms/authentication changes made"))

        wait_for_idle_gpus()
        for label, model in (("base", "/work/models/bases/Llama-3.1-8B-Instruct"),
                             ("wbc_primary_v1", ROOT / "arms/wbc_primary_v1"),
                             ("mse_primary_v1", ROOT / "arms/mse_primary_v1"),
                             ("wbc_short_primary_v1", ROOT / "arms/wbc_short_primary_v1"),
                             ("mse_short_primary_v1", ROOT / "arms/mse_short_primary_v1")):
            done.append(pool_drift(label, model))

        write_json(directory / "complete.json", {"completed_utc": now(), "steps": done,
            "paid_judge_api_calls": 0,
            "remaining": "Report and manuscript integration; not campaign acceptance"})
    except Exception as exc:
        write_json(directory / "failure.json", {"error_type": type(exc).__name__,
            "message": str(exc), "time_utc": now(), "completed_steps": done})
        raise


if __name__ == "__main__":
    main()
