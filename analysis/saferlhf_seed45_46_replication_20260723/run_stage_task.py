#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ARMS = {
    "ronpo_os": ("ronpo", "target_os_k0p1"),
    "inpo_avg": ("inpo", None),
    "sppo_avg": ("sppo", None),
    "simpo": ("simpo", None),
    "ipo": ("ipo", None),
    "dpo": ("dpo", None),
    "ht_mnpo_harmless": ("ht_mnpo", "ht_target"),
    "ht_mnpo_helpfulness": ("ht_mnpo", "ht_target_helpfulness"),
}


def passed(path):
    try:
        data = json.loads(path.read_text())
        return data.get("status") == "passed" and data.get("passed") is True
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def completed(path):
    try:
        data = json.loads(path.read_text())
        return data.get("status") == "completed" and data.get("finite_metrics") is True
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def run(command, env, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write("command=" + json.dumps(command) + "\n")
        handle.flush()
        subprocess.run(
            command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True
        )


def train(runner, common, output, env, log):
    for phase in ("smoke", "full"):
        status = output / phase / "job_status.json"
        if completed(status):
            continue
        if status.exists():
            raise RuntimeError(f"terminal failed training is preserved: {status}")
        flag = "--run-stage" if runner.name == "train_continuation_arm.py" else "--stage"
        run([env["VENV_PYTHON"], str(runner), *common, flag, phase], env, log)
        if not completed(status):
            raise RuntimeError(f"training did not complete: {status}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--seed", type=int, choices=[45, 46], required=True)
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required")
    seed_root = args.root / f"seed{args.seed}"
    stage12 = seed_root / "stage12"
    stage3 = seed_root / "stage3"
    stage4 = seed_root / "stage4"
    loss, target = ARMS[args.arm]
    target_args = ["--target-column", target] if target else []
    env = os.environ.copy()
    env.update(
        {
            "TRAIN_SEED": str(args.seed),
            "RUN_PREFIX": f"p18s{args.seed}",
            "VENV_PYTHON": str(args.venv / "bin/python"),
            "PYTHONPATH": str(args.project),
        }
    )
    legacy = args.project / "analysis/p10_saferlhf_training_seed43_20260718"
    if args.stage == 1:
        experiment, label, runner = stage12, "stage1", legacy / "train_stage1_arm.py"
        common = [
            "--project", str(args.project), "--venv", str(args.venv),
            "--experiment", str(experiment), "--base", args.base,
            "--arm", args.arm, "--loss-type", loss, "--gpu", str(args.gpu),
            "--seed", str(args.seed), "--run-prefix", f"p18s{args.seed}",
            *target_args,
        ]
    elif args.stage == 2:
        experiment, label, runner = stage12, "stage2", legacy / "train_stage2_arm.py"
        parent = stage12 / f"stage1/{args.arm}/train/full"
        parent_gate = stage12 / f"stage1_stability_p8_locked_panel/gates/{args.arm}.json"
        if not passed(parent_gate):
            raise RuntimeError(f"Stage-1 parent gate not passed: {parent_gate}")
        pool = experiment / f"stage2/{args.arm}/pool"
        if not (pool / "PREPARED").is_file():
            prep_env = env | {
                "PROJECT": str(args.project),
                "P4": str(args.project / "results/p4_8b_saferlhf_table4_20260717"),
                "VLLM_GPU_MEMORY_UTILIZATION": "0.55",
            }
            run(
                [
                    "bash",
                    str(args.project / "analysis/p5_8b_robust_stage1_stage2_20260717/prepare_stage2_pool.sh"),
                    str(experiment), args.arm, str(parent), str(args.gpu), str(args.gpu),
                ],
                prep_env, args.log,
            )
        run(
            [
                str(args.venv / "bin/python"), str(legacy / "audit_stage2_pool.py"),
                "--pool", str(pool), "--expected-prompts", "2500",
            ],
            env, args.log,
        )
        common = [
            "--project", str(args.project), "--venv", str(args.venv),
            "--experiment", str(experiment), "--arm", args.arm,
            "--parent-model", str(parent), "--loss-type", loss,
            "--gpu", str(args.gpu), "--seed", str(args.seed),
            "--run-prefix", f"p18s{args.seed}", *target_args,
        ]
    else:
        experiment = stage3 if args.stage == 3 else stage4
        label = f"stage{args.stage}"
        prior = stage12 if args.stage == 3 else stage3
        parent = prior / f"stage{args.stage - 1}/{args.arm}/train/full"
        parent_gate = prior / f"stage{args.stage - 1}_stability_p8_locked_panel/gates/{args.arm}.json"
        if not passed(parent_gate):
            raise RuntimeError(f"parent gate not passed: {parent_gate}")
        pool = experiment / f"{label}/{args.arm}/pool"
        if not (pool / "PREPARED").is_file():
            run(
                [
                    "bash", str(legacy / "prepare_continuation_pool.sh"),
                    str(args.project), str(experiment), label, args.arm,
                    str(parent), str(args.gpu), str(args.gpu),
                ],
                env, args.log,
            )
        run(
            [
                str(args.venv / "bin/python"), str(legacy / "audit_stage2_pool.py"),
                "--pool", str(pool), "--expected-prompts", "2500",
            ],
            env, args.log,
        )
        runner = legacy / "train_continuation_arm.py"
        common = [
            "--project", str(args.project), "--venv", str(args.venv),
            "--experiment", str(experiment), "--continuation-stage", label,
            "--arm", args.arm, "--parent-model", str(parent),
            "--loss-type", loss, "--gpu", str(args.gpu), "--seed", str(args.seed),
            "--run-prefix", f"p18s{args.seed}", *target_args,
        ]
    output = experiment / f"{label}/{args.arm}/train"
    train(runner, common, output, env, args.log)
    gate = experiment / f"{label}_stability_p8_locked_panel/gates/{args.arm}.json"
    if not passed(gate):
        run(
            [
                "bash", str(legacy / "decode_and_gate_continuation.sh"),
                str(args.project), str(experiment), label, args.arm, str(args.gpu),
            ],
            env, args.log,
        )
    if not passed(gate):
        raise RuntimeError(f"fail-closed stability gate failed: {gate}")
    print(json.dumps({"status": "completed", "seed": args.seed, "stage": args.stage, "arm": args.arm}))


if __name__ == "__main__":
    main()
