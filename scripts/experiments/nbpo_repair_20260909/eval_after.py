"""Evaluate one already-training arm the moment its training exits."""
from __future__ import annotations
import argparse, datetime, json, os, subprocess, time
from pathlib import Path

ROOT = Path("/work/nbpo_repair_20260909")
PREFIX = "scripts.experiments.nbpo_repair_20260909."
SKYWORK = "/work/hf_cache/hub/models--Skywork--Skywork-Reward-V2-Qwen3-8B/snapshots/6f19fdefb933293d4898bdb59a96f7223d998659"
HARMBENCH = "/work/hf_cache/hub/models--cais--HarmBench-Llama-2-13b-cls/snapshots/bda705349d1144fa618770bea64d99ce54e3835b"


def exit_code(job):
    p = ROOT / "jobs" / job / "exit.json"
    return json.loads(p.read_text())["exit_code"] if p.exists() else None


def artifact(name, gpu, stack, module, *arguments):
    code = exit_code(name)
    if code == 0:
        return {"job": name, "skipped": "already succeeded"}
    if code is not None:
        raise ValueError(f"{name} already failed; diagnose rather than re-run")
    subprocess.run(["python3", "-m", PREFIX + "artifact_job", "--root", str(ROOT), "--job", name,
                    "--gpu", str(gpu), "--stack", stack, "--",
                    "python3", "-m", PREFIX + module, *map(str, arguments)],
                   cwd=ROOT / "code", check=True,
                   env=dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD="spawn"))
    if exit_code(name) != 0:
        raise ValueError(f"{name} failed; evidence retained")
    return {"job": name, "seconds": json.loads((ROOT / "jobs" / name / "exit.json").read_text())["seconds"]}


def wait_idle(limit=5400):
    start = time.monotonic()
    while subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        if time.monotonic() - start > limit:
            raise ValueError("GPUs still busy; inspect, never kill")
        time.sleep(20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--horizon", type=int, default=250)
    ap.add_argument("--background", action="store_true")
    args = ap.parse_args()
    directory = ROOT / "controllers" / f"eval_after_{args.arm}"
    directory.mkdir(parents=True, exist_ok=True)
    if args.background:
        with (directory / "stdout.log").open("a") as log:
            p = subprocess.Popen(["python3", str(Path(__file__).resolve()), "--arm", args.arm,
                                  "--horizon", str(args.horizon)],
                                 cwd=ROOT / "code", stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        (directory / "launch.json").write_text(json.dumps({"pid": p.pid}, indent=2))
        print(json.dumps({"controller_pid": p.pid, "arm": args.arm}), flush=True)
        return
    arm = args.arm
    done = []
    try:
        while exit_code(arm) is None:
            if not (ROOT / "jobs" / arm).exists():
                raise ValueError(f"{arm} is not training")
            time.sleep(30)
        if exit_code(arm) != 0:
            raise ValueError(f"{arm} training failed; evidence retained")
        state = json.loads((ROOT / "arms" / arm / "trainer_state.json").read_text())
        if state["global_step"] != args.horizon:
            raise ValueError(f"{arm} stopped at {state['global_step']}, not {args.horizon}")
        wait_idle()
        done.append(artifact(f"{arm}_generation_v1", 0, "eval", "generate_eval",
                             "--root", ROOT, "--model", ROOT / "arms" / arm, "--label", arm))
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
        (directory / "complete.json").write_text(json.dumps(
            {"completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "arm": arm, "steps": done, "paid_judge_api_calls": 0}, indent=2))
        print(json.dumps({"arm": arm, "state": "evaluated"}), flush=True)
    except Exception as exc:
        (directory / "failure.json").write_text(json.dumps(
            {"error_type": type(exc).__name__, "message": str(exc), "completed_steps": done,
             "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=2))
        raise


if __name__ == "__main__":
    main()
