"""Fixed primary-checkpoint validation, generation and API-free evaluation.

Waits for BOTH matched training arms to finish, then owns four single-GPU
lanes. Each lane runs sequentially; the longest lanes remain independent.
Teacher-RM diagnostics follow primary scores on that lane. No evaluation
outcome selects a checkpoint, changes a horizon, or launches another arm.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import subprocess
import time
from pathlib import Path

from scripts.experiments.nbpo_repair_20260909.common import file_hash, write_json

PREFIX = "scripts.experiments.nbpo_repair_20260909."
SKYWORK = "/work/hf_cache/hub/models--Skywork--Skywork-Reward-V2-Qwen3-8B/snapshots/6f19fdefb933293d4898bdb59a96f7223d998659"
HARMBENCH = "/work/hf_cache/hub/models--cais--HarmBench-Llama-2-13b-cls/snapshots/bda705349d1144fa618770bea64d99ce54e3835b"


def source_contract(root):
    folder = "scripts/experiments/nbpo_repair_20260909/"
    names = [folder + name + ".py" for name in (
        "artifact_job", "evaluation_chain", "validate_export", "generate_eval", "evaluate_responses",
        "evaluate_saferlhf", "score_teacher_rm", "score_xstest_refusal", "common", "score_pool")]
    names += ["mnpo_scripts/gpm.py", "mnpo_scripts/nbpo_core.py", "mnpo_scripts/precompute_provenance.py",
        "scripts/experiments/nbpo_downstream_local_v1/generate_panel.py",
        "scripts/experiments/nbpo_downstream_local_v1/score_gsm8k.py"]
    return {name: file_hash(root / "code" / name) for name in names}


def verify_sources(root):
    expected = json.loads((root / "controllers/evaluation_chain_v1/source_contract.json").read_text())
    if source_contract(root) != expected:
        raise ValueError("Frozen evaluation source contract changed; inspect before launching")


def job(name, gpu, stack, module, *arguments):
    return {"name": name, "gpu": str(gpu), "stack": stack,
        "command": ["python3", "-m", PREFIX + module, *map(str, arguments)]}


def plan(root):
    labels = ["wbc_primary_v1", "mse_primary_v1"]
    controller = root / "controllers/primary_chain_v1/complete.json"
    validations = [job(f"{arm}_export_validation_v1", gpu, "train", "validate_export",
        "--root", root, "--arm", arm, "--controller-manifest", controller,
        "--inventory", root / "provenance/inventory.json", "--output", root / f"probes/export_validation_{arm}_v1",
        "--stage", "full") for gpu, arm in enumerate(("wbc", "mse"))]
    generation = [job(f"{label}_generation_v1", gpu, "eval", "generate_eval", "--root", root,
        "--model", root / "arms" / label, "--label", label) for gpu, label in enumerate(labels)]
    lanes = {}
    for gpu, label in enumerate(labels):
        lanes[str(gpu)] = [job(f"skywork_{label}_v1", gpu, "eval", "evaluate_responses",
            "--root", root, "--mode", "skywork", "--labels", label, "--rm", SKYWORK,
            "--out", root / f"evaluations/skywork_{label}_v1")]
    lanes["2"] = [job("harmbench_primary_v1", 2, "harmbench", "evaluate_responses",
        "--root", root, "--mode", "harmbench", "--labels", *labels,
        "--harmbench-model", HARMBENCH, "--harmbench-repo", root / "external/HarmBench",
        "--out", root / "evaluations/harmbench_primary_v1")]
    lanes["3"] = [job("saferlhf_primary_v1", 3, "eval", "evaluate_saferlhf",
        "--root", root, "--labels", *labels,
        "--base-score-dir", root / "evaluations/saferlhf_base_v1/base",
        "--encoder", root / "assets/roberta-base",
        "--out", root / "evaluations/saferlhf_primary_v1")]
    for gpu in range(4):
        lanes[str(gpu)].append(job(f"teacher_rm_nash_shard{gpu}_v1", gpu, "eval", "score_teacher_rm",
            "--mode", "score", "--root", root, "--teacher-name", "nash_repair_v2",
            "--rm", SKYWORK, "--out", root / "teacher_rm/nash_repair_v2_v1", "--shard", gpu,
            "--device", "cuda:0"))
    lanes["cpu"] = [job("deterministic_primary_v1", "cpu", "eval", "evaluate_responses",
        "--root", root, "--mode", "deterministic", "--labels", *labels,
        "--out", root / "evaluations/deterministic_primary_v1"),
        job("xstest_primary_unadjudicated_v1", "cpu", "eval", "score_xstest_refusal",
        "--root", root, "--labels", "base", *labels,
        "--output-dir", root / "evaluations/xstest_primary_unadjudicated_v1",
        "--unadjudicated-reason", "Pinned official WildGuard unavailable: existing authorized identity returns HTTP403; no terms/authentication changes made")]
    aggregate = job("teacher_rm_nash_aggregate_v1", "cpu", "eval", "score_teacher_rm",
        "--mode", "aggregate", "--root", root, "--teacher-name", "nash_repair_v2",
        "--rm", SKYWORK, "--out", root / "teacher_rm/nash_repair_v2_v1")
    return {"validation": validations, "generation": generation, "scoring_lanes": lanes,
        "teacher_aggregate": aggregate, "labels": labels,
        "checkpoint_selection": "fixed_final1750_both_arms", "paid_judge_api_calls": 0}


def run_job(root, spec):
    verify_sources(root)
    command = ["python3", "-m", PREFIX + "artifact_job", "--root", str(root), "--job", spec["name"],
        "--gpu", spec["gpu"], "--stack", spec["stack"], "--", *spec["command"]]
    subprocess.run(command, cwd=root / "code", check=True)
    exit_record = json.loads((root / "jobs" / spec["name"] / "exit.json").read_text())
    if exit_record["exit_code"] != 0:
        raise ValueError(f"Job {spec['name']} did not succeed")


def run_lane(root, specs):
    for spec in specs:
        run_job(root, spec)


def parallel_lanes(root, lanes):
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(lanes)) as executor:
        futures = {executor.submit(run_lane, root, lane): key for key, lane in lanes.items()}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                errors.append({"lane": futures[future], "type": type(exc).__name__, "message": str(exc)})
    if errors:
        raise RuntimeError(json.dumps(errors))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--background", action="store_true")
    ap.add_argument("--print-plan", action="store_true")
    args = ap.parse_args()
    if args.root != Path("/work/nbpo_repair_20260909"):
        raise ValueError("Unexpected campaign root")
    resolved = plan(args.root)
    if args.print_plan:
        print(json.dumps(resolved, indent=2))
        return
    directory = args.root / "controllers/evaluation_chain_v1"
    directory.mkdir(parents=True, exist_ok=True)
    if args.background:
        write_json(directory / "plan.json", resolved)
        write_json(directory / "source_contract.json", source_contract(args.root))
        with (directory / "stdout.log").open("x") as log:
            command = ["python3", "-m", PREFIX + "evaluation_chain", "--root", str(args.root)]
            process = subprocess.Popen(command, cwd=args.root / "code", stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        write_json(directory / "launch.json", {"pid": process.pid, "command": command,
            "source_sha256": file_hash(__file__), "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        print(json.dumps({"controller_pid": process.pid, "wait_for": "both_primary_training_arms"}), flush=True)
        return
    if json.loads((directory / "plan.json").read_text()) != resolved:
        raise ValueError("Fixed evaluation plan changed")
    try:
        verify_sources(args.root)
        primary = args.root / "controllers/primary_chain_v1"
        while not (primary / "complete.json").exists():
            if (primary / "failure.json").exists():
                raise ValueError("Primary training controller failed; no premature GPU evaluation")
            time.sleep(15)
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if processes.strip():
            raise ValueError("GPU compute processes remain after primary completion; inspect, never kill")
        parallel_lanes(args.root, {str(i): [s] for i, s in enumerate(resolved["validation"])})
        write_json(directory / "exports_validated.json", {"labels": resolved["labels"]})
        parallel_lanes(args.root, {str(i): [s] for i, s in enumerate(resolved["generation"])})
        write_json(directory / "generation_complete.json", {"labels": resolved["labels"]})
        parallel_lanes(args.root, resolved["scoring_lanes"])
        run_job(args.root, resolved["teacher_aggregate"])
        write_json(directory / "complete.json", {"completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "labels": resolved["labels"], "paid_judge_api_calls": 0,
            "remaining": "Independent result audit, manuscript/report integration, optional compute-budgeted control; not automatic campaign acceptance"})
    except Exception as exc:
        write_json(directory / "failure.json", {"error_type": type(exc).__name__, "message": str(exc),
            "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    main()
