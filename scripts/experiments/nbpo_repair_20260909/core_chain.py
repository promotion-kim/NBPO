"""Train each core arm and evaluate it the moment it exits, not after the batch.

Reads the manifest frozen before any of these arms ran. Every artifact job gets
VLLM_WORKER_MULTIPROC_METHOD=spawn, because HarmBench's eval_utils initializes
CUDA before vLLM forks its engine core. Nothing here selects a checkpoint,
changes a horizon, retries a genuine failure, or touches another process.
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
SPAWN = {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def exit_code(job):
    p = ROOT / "jobs" / job / "exit.json"
    return json.loads(p.read_text())["exit_code"] if p.exists() else None


def artifact(name, gpu, stack, module, *arguments):
    code = exit_code(name)
    if code == 0:
        return {"job": name, "skipped": "already succeeded"}
    if code is not None:
        raise ValueError(f"{name} already failed; diagnose it rather than re-running")
    if (ROOT / "jobs" / name).exists():
        raise ValueError(f"{name} has a job directory with no exit record; inspect it")
    command = ["python3", "-m", PREFIX + "artifact_job", "--root", str(ROOT), "--job", name,
               "--gpu", str(gpu), "--stack", stack, "--",
               "python3", "-m", PREFIX + module, *map(str, arguments)]
    subprocess.run(command, cwd=ROOT / "code", check=True, env=dict(os.environ, **SPAWN))
    if exit_code(name) != 0:
        raise ValueError(f"{name} failed; evidence retained")
    return {"job": name, "seconds": json.loads((ROOT / "jobs" / name / "exit.json").read_text())["seconds"]}


def gpus_idle():
    out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    return not out.strip()


def wait_idle(limit=3600):
    start = time.monotonic()
    while not gpus_idle():
        if time.monotonic() - start > limit:
            raise ValueError("GPUs still busy; inspect what is running, never kill it")
        time.sleep(20)


def train(arm, horizon=250):
    if exit_code(arm) == 0:
        return {"job": arm, "skipped": "already trained"}
    wait_idle()
    subprocess.run(["python3", "-m", PREFIX + "train_job", "--root", str(ROOT),
                    "--config", str(ROOT / "configs" / f"{arm}.yaml"), "--job", arm],
                   cwd=ROOT / "code", check=True)
    if exit_code(arm) != 0:
        raise ValueError(f"{arm} training failed; evidence retained, no restart")
    state = json.loads((ROOT / "arms" / arm / "trainer_state.json").read_text())
    if state["global_step"] != horizon:
        raise ValueError(f"{arm} stopped at {state['global_step']}, not the declared {horizon}")
    record = json.loads((ROOT / "jobs" / arm / "exit.json").read_text())
    return {"job": arm, "seconds": record["seconds"], "global_step": state["global_step"],
            "gpu_hours": 4 * record["seconds"] / 3600}


def evaluate(arm):
    done = [artifact(f"{arm}_generation_v1", 0, "eval", "generate_eval",
                     "--root", ROOT, "--model", ROOT / "arms" / arm, "--label", arm)]
    done.append(artifact(f"skywork_{arm}_v1", 0, "eval", "evaluate_responses",
                         "--root", ROOT, "--mode", "skywork", "--labels", arm,
                         "--rm", SKYWORK, "--out", ROOT / f"evaluations/skywork_{arm}_v1"))
    done.append(artifact(f"harmbench_{arm}_v1", 2, "harmbench", "evaluate_responses",
                         "--root", ROOT, "--mode", "harmbench", "--labels", arm,
                         "--harmbench-model", HARMBENCH,
                         "--harmbench-repo", ROOT / "external/HarmBench",
                         "--out", ROOT / f"evaluations/harmbench_{arm}_v1"))
    done.append(artifact(f"deterministic_{arm}_v1", "cpu", "eval", "evaluate_responses",
                         "--root", ROOT, "--mode", "deterministic", "--labels", arm,
                         "--out", ROOT / f"evaluations/deterministic_{arm}_v1"))
    done.append(artifact(f"saferlhf_{arm}_v1", 3, "eval", "evaluate_saferlhf",
                         "--root", ROOT, "--labels", arm,
                         "--base-score-dir", ROOT / "evaluations/saferlhf_base_v1/base",
                         "--encoder", ROOT / "assets/roberta-base",
                         "--out", ROOT / f"evaluations/saferlhf_{arm}_v1"))
    done.append(artifact(f"xstest_{arm}_v1", "cpu", "eval", "score_xstest_refusal",
                         "--root", ROOT, "--labels", arm,
                         "--output-dir", ROOT / f"evaluations/xstest_{arm}_v1",
                         "--unadjudicated-reason",
                         "Pinned official WildGuard unavailable: existing authorized identity returns HTTP403; no terms/authentication changes made"))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--background", action="store_true")
    args = ap.parse_args()
    directory = ROOT / "controllers" / "core_chain_v1"
    directory.mkdir(parents=True, exist_ok=True)
    if args.background:
        with (directory / "stdout.log").open("a") as log:
            p = subprocess.Popen(["python3", str(Path(__file__).resolve())], cwd=ROOT / "code",
                                 stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 start_new_session=True)
        (directory / "launch.json").write_text(json.dumps({"pid": p.pid, "started_utc": now()}, indent=2))
        print(json.dumps({"controller_pid": p.pid}), flush=True)
        return

    manifest = json.loads((ROOT / "protocols/submission_core_manifest_v1.json").read_text())
    arms = [a["name"] for a in manifest["arms"]]
    done = []
    try:
        for arm in arms:
            done.append(train(arm))
            (directory / "progress.json").write_text(json.dumps({"time_utc": now(), "steps": done}, indent=2))
            wait_idle()
            done.extend(evaluate(arm))
            (directory / "progress.json").write_text(json.dumps({"time_utc": now(), "steps": done}, indent=2))
            print(json.dumps({"arm": arm, "state": "trained and evaluated"}), flush=True)
        (directory / "complete.json").write_text(json.dumps(
            {"completed_utc": now(), "arms": arms, "steps": done, "paid_judge_api_calls": 0}, indent=2))
    except Exception as exc:
        (directory / "failure.json").write_text(json.dumps(
            {"error_type": type(exc).__name__, "message": str(exc), "time_utc": now(),
             "completed_steps": done}, indent=2))
        raise


if __name__ == "__main__":
    main()
