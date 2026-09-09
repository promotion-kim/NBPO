"""Budget-only conditional launch of one frozen optional L1-matched WBC arm.

Prepare/print-plan are CPU-only and do not launch a background process. The
separately approved background mode waits for BOTH primary controllers, then
uses observed WBC wall time, a fixed 1.25 multiplier, two hours optional
evaluation reserve, and two hours reporting reserve. No quality metric enters
the decision. There is no automatic optional evaluation, retry, or other arm.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import time
from pathlib import Path

import yaml

from scripts.experiments.nbpo_repair_20260909.common import file_hash, write_json
from scripts.experiments.nbpo_repair_20260909.primary_chain import completed_arm

PREFIX = "scripts.experiments.nbpo_repair_20260909."
ARM = "utilitarian_l1matched"
LABEL = ARM + "_primary_v1"
TEACHER = "utilitarian_l1matched_v1"
DIRECTORY = "controllers/optional_control_chain_v1"
BUDGET_RULE = {"wbc_wall_multiplier": 1.25, "optional_validation_generation_scoring_reserve_seconds": 7200.,
               "final_reporting_reserve_seconds": 7200., "decision_inputs": "wall-clock time and upstream completion only; no fitting/benchmark outcomes"}
CONFIG_DIFFERENCES = {"dataset_mixer", "output_dir", "run_name", "nbpo_expected_dataset_manifest_sha256",
                      "nbpo_expected_solver_artifact_sha256"}


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


def check_config_match(primary, optional, root, dataset_hash, solver_hash):
    """Exact primary WBC settings, except the declared data/teacher/output fields."""
    changed = {key for key in primary.keys() | optional.keys() if primary.get(key) != optional.get(key)}
    if changed != CONFIG_DIFFERENCES:
        raise ValueError(f"Optional/primary WBC config mismatch beyond exact allowed fields: {sorted(changed)}")
    expected = {"dataset_mixer": {str(root / "datasets" / TEACHER): 1.0},
        "output_dir": str(root / "arms" / LABEL), "run_name": LABEL,
        "nbpo_expected_dataset_manifest_sha256": dataset_hash, "nbpo_expected_solver_artifact_sha256": solver_hash,
        "loss_type": "nbpo_wbc", "max_steps": 1750, "seed": 42, "data_seed": 42,
        "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
        "train_split": "train", "eval_split": "dev", "load_best_model_at_end": False,
        "nbpo_online_reference": False, "nbpo_verify_reference_initialization": False,
        "nbpo_num_candidates": 8, "eta": 1., "logp_reduction": "sum"}
    if any(optional.get(key) != value for key, value in expected.items()):
        raise ValueError("Optional config violates fixed1750 WBC/data/output contract")
    if optional.get("nbpo_profile_updates", 0) or optional.get("resume_from_checkpoint"):
        raise ValueError("Optional primary arm cannot be a profile or a resumed run")
    return sorted(changed)


def budget_decision(wbc_seconds, now, deadline):
    if isinstance(wbc_seconds, bool) or not isinstance(wbc_seconds, (int, float)) or not math.isfinite(wbc_seconds) or wbc_seconds <= 0:
        raise ValueError("Budget requires actual positive finite completed WBC job wall time")
    if now.tzinfo is None or deadline.tzinfo is None:
        raise ValueError("Budget times must be timezone-aware")
    remaining = (deadline-now).total_seconds()
    forecast = float(wbc_seconds)*BUDGET_RULE["wbc_wall_multiplier"]
    required = forecast + BUDGET_RULE["optional_validation_generation_scoring_reserve_seconds"] + BUDGET_RULE["final_reporting_reserve_seconds"]
    return {"status": "budget_fits" if remaining >= required else "skipped_budget_only",
        "launch_permitted_by_budget": remaining >= required, "observed_wbc_job_wall_seconds": float(wbc_seconds),
        "optional_training_forecast_seconds": forecast, "remaining_seconds": remaining,
        "required_seconds_including_evaluation_and_reporting": required, "budget_rule": BUDGET_RULE,
        "decision_utc": now.astimezone(dt.timezone.utc).isoformat(), "deadline": deadline.isoformat(),
        "quality_metrics_read_for_decision": False}


def dependency_state(root):
    """Failure takes precedence, including contradictory complete+failure files."""
    folders = {name: root / "controllers" / name for name in ("primary_chain_v1", "evaluation_chain_v1")}
    failures = {name: {"path": str(folder / "failure.json"), "sha256": file_hash(folder / "failure.json")}
                for name, folder in folders.items() if (folder / "failure.json").exists()}
    if failures:
        return {"state": "upstream_failed", "failures": failures}
    missing = [name for name, folder in folders.items() if not (folder / "complete.json").exists()]
    if missing:
        return {"state": "waiting", "missing": missing}
    primary_path, evaluation_path = (folders[name] / "complete.json" for name in folders)
    primary, evaluation = (json.loads(path.read_text()) for path in (primary_path, evaluation_path))
    arms = primary.get("arms", [])
    if (len(arms) != 2 or {row.get("arm") for row in arms} != {"wbc", "mse"}
            or any(row.get("global_step") != 1750 for row in arms)
            or set(evaluation.get("labels", [])) != {"wbc_primary_v1", "mse_primary_v1"}):
        raise ValueError("Upstream completion does not attest BOTH fixed1750 primary arms and their evaluation")
    wbc = next(row for row in arms if row["arm"] == "wbc")
    for record in arms:
        path = root / "jobs" / f"{record['arm']}_primary_v1" / "exit.json"
        if file_hash(path) != record["job_exit_sha256"] or json.loads(path.read_text()).get("exit_code") != 0:
            raise ValueError("Primary completed-arm job exit attestation mismatch")
    wbc_path = root / "jobs/wbc_primary_v1/exit.json"
    seconds = json.loads(wbc_path.read_text())["seconds"]
    if seconds != wbc["runtime_seconds"]:
        raise ValueError("Actual WBC job wall time differs from controller attestation")
    return {"state": "ready", "actual_wbc_seconds": seconds, "wbc_exit_sha256": file_hash(wbc_path),
        "primary_complete_sha256": file_hash(primary_path), "evaluation_complete_sha256": file_hash(evaluation_path)}


def check_hashes(root, hashes):
    for raw, expected in hashes.items():
        path = Path(raw)
        if not path.resolve().is_relative_to(root.resolve()) or file_hash(path) != expected:
            raise ValueError(f"Frozen optional source/input hash changed: {raw}")


def build_plan(root):
    """Read-only construction; all pins are recorded before optional training."""
    from mnpo_scripts.precompute_provenance import verify_precompute_manifest
    horizon_path = root / "protocols/primary_horizon_frozen_v1.json"
    horizon = json.loads(horizon_path.read_text())
    primary_config = Path(horizon["configs"]["wbc"]["path"])
    if file_hash(primary_config) != horizon["configs"]["wbc"]["sha256"]:
        raise ValueError("Primary WBC config no longer matches its frozen horizon")
    deadline = dt.datetime.fromisoformat(horizon["deadline_kst"])
    if deadline.tzinfo is None or horizon["optimizer_updates_each"] != 1750:
        raise ValueError("Primary horizon/deadline contract is invalid")
    config = root / "configs" / f"{LABEL}.yaml"
    dataset = root / "datasets" / TEACHER
    teacher = root / "teachers" / TEACHER
    dataset_hash = file_hash(dataset / "precompute_manifest.json")
    verify_precompute_manifest(str(dataset), dataset_hash)
    control = json.loads((teacher / "complete.json").read_text())["control"]
    nash_source = root / "teachers/nash_repair_v2/train/solver/solution.json"
    original = json.loads(nash_source.read_text())
    expected_l1 = sum(original["lambda_raw"])
    expected_weights = [expected_l1/2., expected_l1/2.]
    if (control["source_train_nash_solution_sha256"] != file_hash(nash_source)
            or control["control_weights_raw"] != expected_weights or control["train_nash_lambda_l1"] != expected_l1
            or control["simplex_normalized"] is not False or control["weights_refit_on_dev_or_test"] is not False
            or control["nash_dual_certificate_status"] != "not_applicable"):
        raise ValueError("Optional teacher is not the frozen train-Nash-only L1-matched control")
    artifacts = [horizon_path, primary_config, config, nash_source,
        dataset / "precompute_manifest.json", dataset / "nbpo_dataset_provenance.json",
        teacher / "complete.json", teacher / "inputs.json", teacher / "dataset_provenance.json",
        root / "controllers/evaluation_chain_v1/source_contract.json"]
    for split in ("train", "dev", "test"):
        solution_path = teacher / split / "solver/solution.json"
        solution = json.loads(solution_path.read_text())
        if (solution["aggregation"] != "utilitarian" or solution["lambda_raw"] != expected_weights
                or solution["certificate"]["certified"] is not True
                or solution["certificate"]["nash_dual_scope"] != "not_applicable"):
            raise ValueError("Optional split teacher changed raw weights/aggregation/certificate scope")
        artifacts.extend([solution_path, teacher / split / "independent_numpy_crosscheck.json"])
        for name, sha in solution["artifact_hashes"].items():
            path = solution_path.parent / name
            if file_hash(path) != sha:
                raise ValueError("Optional canonical solver artifact hash mismatch")
            artifacts.append(path)
    train_hash = file_hash(teacher / "train/solver/solution.json")
    differences = check_config_match(yaml.safe_load(primary_config.read_text()), yaml.safe_load(config.read_text()),
                                     root, dataset_hash, train_hash)
    launch_path = root / "jobs/wbc_primary_v1/launch.json"
    launch = json.loads(launch_path.read_text())
    source_names = [name for name in launch["source_hashes"]
        if name.startswith(("mnpo_scripts/", "alignment/")) or name == "scripts/simpo_trainer.py"]
    source_names += ["scripts/experiments/nbpo_repair_20260909/train_job.py",
        "scripts/experiments/nbpo_repair_20260909/common.py",
        "training_configs/nbpo/repair_20260909/train_common.yaml",
        "training_configs/nbpo/repair_20260909/deepspeed_zero2.json"]
    for name in source_names:
        if name not in launch["source_hashes"] or file_hash(root / "code" / name) != launch["source_hashes"][name]:
            raise ValueError("Optional training source differs from actual primary WBC source")
    source_names += ["scripts/experiments/nbpo_repair_20260909/"+name+".py"
                     for name in ("optional_control_chain", "train_job", "primary_chain", "common")]
    source_names += ["training_configs/nbpo/repair_20260909/"+name for name in
                     ("train_common.yaml", "deepspeed_zero2.json")]
    artifacts.append(launch_path)
    sources = {str(root / "code" / name): file_hash(root / "code" / name) for name in sorted(set(source_names))}
    return {"arm": ARM, "label": LABEL, "root": str(root), "config_path": str(config),
        "config_sha256": file_hash(config), "dataset_path": str(dataset), "dataset_manifest_sha256": dataset_hash,
        "source_hashes": sources, "artifact_hashes": {str(path): file_hash(path) for path in artifacts},
        "allowed_config_differences_from_primary_wbc": differences, "control": control,
        "deadline": deadline.isoformat(), "budget_rule": BUDGET_RULE,
        "authorization": "Conditional: BOTH primary training and evaluation controllers succeed, budget fits, and all four GPUs idle; no outcome-dependent selection",
        "authorization_basis": "2026-09-09 15:47 KST throughput/time-budget review, not fitting or benchmark results",
        "gpu_count": 4, "gpu_ids": [0, 1, 2, 3], "optimizer_updates": 1750, "policy_seed": 42,
        "automatic_optional_evaluation": False, "automatic_retry": False, "other_methods_or_seeds": False,
        "nash_dual_certificate_status": "not_applicable"}


def verify_plan(root, plan):
    if (plan["root"] != str(root) or plan["arm"] != ARM or plan["label"] != LABEL
            or plan["budget_rule"] != BUDGET_RULE or plan["gpu_count"] != 4
            or plan["optimizer_updates"] != 1750 or plan["automatic_optional_evaluation"] is not False):
        raise ValueError("Optional frozen plan changed its authorized scope")
    check_hashes(root, plan["source_hashes"])
    check_hashes(root, plan["artifact_hashes"])
    from mnpo_scripts.precompute_provenance import verify_precompute_manifest
    verify_precompute_manifest(plan["dataset_path"], plan["dataset_manifest_sha256"])


def gpu_processes():
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
    if output and any(not line.strip().isdigit() for line in output.splitlines()):
        raise ValueError("Cannot prove all four GPUs idle from unrecognized query output")
    return output


def launch_training(root, plan):
    command = ["python3", "-m", PREFIX+"train_job", "--root", str(root),
               "--config", plan["config_path"], "--job", LABEL]
    subprocess.run(command, check=True, cwd=root / "code")
    return completed_arm(root, ARM)


def decide_and_launch(root, plan, dependency, *, now_fn=utcnow, query=gpu_processes, launch=launch_training):
    if dependency.get("state") != "ready":
        raise ValueError("Both upstream controllers must succeed before any optional GPU work")
    directory = root / DIRECTORY
    deadline = dt.datetime.fromisoformat(plan["deadline"])
    decision = budget_decision(dependency["actual_wbc_seconds"], now_fn(), deadline)
    if decision["launch_permitted_by_budget"]:
        if query():
            raise ValueError("GPU compute PIDs remain after upstream completion; inspect, never kill or overlap")
        # Recheck after the idle query so even the tiny pre-launch delay is charged.
        decision = budget_decision(dependency["actual_wbc_seconds"], now_fn(), deadline)
    record = {**decision, "upstream": dependency, "plan_sha256": file_hash(directory / "plan.json")}
    write_json(directory / "decision.json", record)
    if not decision["launch_permitted_by_budget"]:
        write_json(directory / "skipped.json", {**record, "gpu_training_launched": False,
            "reason": "Insufficient remaining wall-clock budget under the frozen rule; no quality outcomes consulted"})
        return None
    result = launch(root, plan)
    write_json(directory / "complete.json", {"arms": [result], "completed_utc": now_fn().isoformat(),
        "plan_sha256": file_hash(directory / "plan.json"), "decision_sha256": file_hash(directory / "decision.json"),
        "nash_dual_certificate_status": "not_applicable", "automatic_optional_evaluation": False,
        "remaining": "Separate root-authorized optional export validation/evaluation and reporting; no automatic evaluation or campaign acceptance"})
    return result


def run(root, expected_plan_hash):
    directory, path = root / DIRECTORY, root / DIRECTORY / "plan.json"
    if not expected_plan_hash or file_hash(path) != expected_plan_hash:
        raise ValueError("Run requires the reviewed immutable plan SHA256")
    plan = json.loads(path.read_text())
    if any((directory / name).exists() for name in ("complete.json", "skipped.json", "failure.json", "run_started.json")):
        raise ValueError("Optional controller is single-run; no automatic resume/retry")
    write_json(directory / "run_started.json", {"pid": os.getpid(), "plan_sha256": expected_plan_hash, "started_utc": utcnow().isoformat()})
    try:
        verify_plan(root, plan)
        while True:
            dependency = dependency_state(root)
            if dependency["state"] == "upstream_failed":
                raise ValueError("Upstream controller failed; optional training not started: "+json.dumps(dependency))
            if dependency["state"] == "ready":
                break
            time.sleep(15)
        verify_plan(root, plan)
        # A newly written failure must win even after a long hash verification.
        dependency = dependency_state(root)
        decide_and_launch(root, plan, dependency)
    except Exception as error:
        write_json(directory / "failure.json", {"error_type": type(error).__name__, "message": str(error),
            "time_utc": utcnow().isoformat(), "plan_sha256": expected_plan_hash,
            "gpu_training_job_created": (root / "jobs" / LABEL).exists(), "automatic_retry": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("print-plan", "prepare", "background", "run"), required=True)
    parser.add_argument("--expected-plan-sha256")
    args = parser.parse_args()
    if args.root != Path("/work/nbpo_repair_20260909"):
        raise ValueError("Unexpected optional-control campaign root")
    directory = args.root / DIRECTORY
    if args.mode in ("print-plan", "prepare"):
        plan = build_plan(args.root)
        if args.mode == "prepare":
            directory.mkdir(parents=True, exist_ok=False)
            write_json(directory / "plan.json", plan)
            print(json.dumps({"plan": str(directory / "plan.json"), "sha256": file_hash(directory / "plan.json"),
                              "status": "prepared_only_no_background_or_GPU_launch"}))
        else:
            print(json.dumps(plan, indent=2))
        return
    if args.mode == "run":
        run(args.root, args.expected_plan_sha256)
        return
    path = directory / "plan.json"
    if not args.expected_plan_sha256 or file_hash(path) != args.expected_plan_sha256:
        raise ValueError("Background launch requires reviewed plan SHA256")
    verify_plan(args.root, json.loads(path.read_text()))
    if any((directory / name).exists() for name in ("launch.json", "run_started.json", "failure.json", "skipped.json", "complete.json")):
        raise ValueError("Optional controller already launched or terminal")
    with (directory / "stdout.log").open("x") as log:
        command = ["python3", "-m", PREFIX+"optional_control_chain", "--root", str(args.root), "--mode", "run",
                   "--expected-plan-sha256", args.expected_plan_sha256]
        process = subprocess.Popen(command, cwd=args.root / "code", stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
    write_json(directory / "launch.json", {"pid": process.pid, "command": command, "plan_sha256": args.expected_plan_sha256,
        "source_sha256": file_hash(__file__), "started_utc": utcnow().isoformat(),
        "wait_for": ["primary_chain_v1", "evaluation_chain_v1"], "conditional_budget_rule": BUDGET_RULE})
    print(json.dumps({"controller_pid": process.pid, "status": "waiting_for_both_upstreams_and_budget", "plan_sha256": args.expected_plan_sha256}))


if __name__ == "__main__":
    main()
