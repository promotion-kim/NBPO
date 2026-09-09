"""Finish the utilitarian control at three seeds, then evaluate both new arms.

Trains back to back so the GPUs are never idle between arms, then runs the same
frozen evaluation panel each of the other arms received.
"""
from __future__ import annotations
import datetime, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path("/work/nbpo_repair_20260909")
PREFIX = "scripts.experiments.nbpo_repair_20260909."
SKYWORK = "/work/hf_cache/hub/models--Skywork--Skywork-Reward-V2-Qwen3-8B/snapshots/6f19fdefb933293d4898bdb59a96f7223d998659"
HARMBENCH = "/work/hf_cache/hub/models--cais--HarmBench-Llama-2-13b-cls/snapshots/bda705349d1144fa618770bea64d99ce54e3835b"
ARMS = ["utilitarian_mse_s43", "utilitarian_mse_s44"]
D = ROOT / "controllers" / "util_chain_v1"


def code(job):
    p = ROOT / "jobs" / job / "exit.json"
    return json.loads(p.read_text())["exit_code"] if p.exists() else None


def idle():
    return not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()


def wait_idle(limit=5400):
    t = time.monotonic()
    while not idle():
        if time.monotonic() - t > limit:
            raise ValueError("GPUs still busy; inspect, never kill")
        time.sleep(20)


def artifact(name, gpu, stack, module, *a):
    if code(name) == 0:
        return {"job": name, "skipped": "already succeeded"}
    if code(name) is not None:
        raise ValueError(f"{name} already failed")
    subprocess.run(["python3", "-m", PREFIX + "artifact_job", "--root", str(ROOT), "--job", name,
                    "--gpu", str(gpu), "--stack", stack, "--",
                    "python3", "-m", PREFIX + module, *map(str, a)],
                   cwd=ROOT / "code", check=True,
                   env=dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD="spawn"))
    if code(name) != 0:
        raise ValueError(f"{name} failed")
    return {"job": name, "seconds": json.loads((ROOT / "jobs" / name / "exit.json").read_text())["seconds"]}


def evaluate(arm):
    out = [artifact(f"{arm}_generation_v1", 0, "eval", "generate_eval",
                    "--root", ROOT, "--model", ROOT / "arms" / arm, "--label", arm)]
    out.append(artifact(f"skywork_{arm}_v1", 0, "eval", "evaluate_responses", "--root", ROOT,
                        "--mode", "skywork", "--labels", arm, "--rm", SKYWORK,
                        "--out", ROOT / f"evaluations/skywork_{arm}_v1"))
    out.append(artifact(f"harmbench_{arm}_v1", 2, "harmbench", "evaluate_responses", "--root", ROOT,
                        "--mode", "harmbench", "--labels", arm, "--harmbench-model", HARMBENCH,
                        "--harmbench-repo", ROOT / "external/HarmBench",
                        "--out", ROOT / f"evaluations/harmbench_{arm}_v1"))
    out.append(artifact(f"deterministic_{arm}_v1", "cpu", "eval", "evaluate_responses", "--root", ROOT,
                        "--mode", "deterministic", "--labels", arm,
                        "--out", ROOT / f"evaluations/deterministic_{arm}_v1"))
    out.append(artifact(f"saferlhf_{arm}_v1", 3, "eval", "evaluate_saferlhf", "--root", ROOT,
                        "--labels", arm, "--base-score-dir", ROOT / "evaluations/saferlhf_base_v1/base",
                        "--encoder", ROOT / "assets/roberta-base",
                        "--out", ROOT / f"evaluations/saferlhf_{arm}_v1"))
    return out


def main():
    D.mkdir(parents=True, exist_ok=True)
    if "--background" in sys.argv:
        with (D / "stdout.log").open("a") as log:
            p = subprocess.Popen(["python3", str(Path(__file__).resolve())], cwd=ROOT / "code",
                                 stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 start_new_session=True)
        (D / "launch.json").write_text(json.dumps({"pid": p.pid}, indent=2))
        print(json.dumps({"controller_pid": p.pid, "arms": ARMS}), flush=True)
        return
    done = []
    try:
        for arm in ARMS:
            while code(arm) is None and (ROOT / "jobs" / arm).exists():
                time.sleep(30)                      # s43 was launched by hand; wait it out
            if code(arm) is None:
                wait_idle()
                subprocess.run(["python3", "-m", PREFIX + "train_job", "--root", str(ROOT),
                                "--config", str(ROOT / "configs" / f"{arm}.yaml"), "--job", arm],
                               cwd=ROOT / "code", check=True)
            if code(arm) != 0:
                raise ValueError(f"{arm} training failed")
            state = json.loads((ROOT / "arms" / arm / "trainer_state.json").read_text())
            if state["global_step"] != 250:
                raise ValueError(f"{arm} stopped at {state['global_step']}")
            done.append({"job": arm, "global_step": 250,
                         "seconds": json.loads((ROOT / "jobs" / arm / "exit.json").read_text())["seconds"]})
            (D / "progress.json").write_text(json.dumps({"steps": done}, indent=2))
        for arm in ARMS:
            wait_idle()
            done.extend(evaluate(arm))
            (D / "progress.json").write_text(json.dumps({"steps": done}, indent=2))
        (D / "complete.json").write_text(json.dumps(
            {"completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "arms": ARMS, "steps": done, "paid_judge_api_calls": 0}, indent=2))
    except Exception as exc:
        (D / "failure.json").write_text(json.dumps(
            {"error_type": type(exc).__name__, "message": str(exc), "completed_steps": done,
             "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=2))
        raise


if __name__ == "__main__":
    main()
