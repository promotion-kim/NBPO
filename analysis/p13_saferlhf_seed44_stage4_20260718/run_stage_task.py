#!/usr/bin/env python3
"""Run one fail-closed seed-44 arm/stage task on one authorized GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ARMS = {
    "ronpo_os": ["ronpo", "target_os_k0p1"],
    "ronpo_topmass": ["ronpo", "target_topmass_k0p1"],
    "inpo_avg": ["inpo", None], "sppo_avg": ["sppo", None],
    "simpo": ["simpo", None], "ipo": ["ipo", None], "dpo": ["dpo", None],
    "ht_mnpo_harmless": ["ht_mnpo", "ht_target"],
    "ht_mnpo_helpfulness": ["ht_mnpo", "ht_target_helpfulness"],
}


def passed(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("passed") is True and data.get("status") == "passed"
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def completed(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("status") == "completed" and data.get("finite_metrics") is True
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def run(command: list[str], env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("command=" + json.dumps(command) + "\n")
        handle.flush()
        subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def train(runner: Path, common: list[str], output: Path, env: dict[str, str], log: Path) -> None:
    for phase in ("smoke", "full"):
        status = output / phase / "job_status.json"
        if completed(status):
            continue
        if status.exists():
            raise RuntimeError(f"terminal failed training cannot be retried: {status}")
        flag = "--run-stage" if runner.name == "train_continuation_arm.py" else "--stage"
        run([env["VENV_PYTHON"], str(runner), *common, flag, phase], env, log)
        if not completed(status):
            raise RuntimeError(f"training did not complete cleanly: {status}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True); p.add_argument("--venv", type=Path, required=True)
    p.add_argument("--stage12", type=Path, required=True); p.add_argument("--stage3", type=Path, required=True)
    p.add_argument("--stage4", type=Path, required=True); p.add_argument("--base", required=True)
    p.add_argument("--stage", type=int, choices=[1, 2, 3, 4], required=True); p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--gpu", type=int, required=True); p.add_argument("--log", type=Path, required=True)
    a = p.parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required ephemerally")
    loss, target = ARMS[a.arm]
    target_args = ["--target-column", target] if target else []
    env = os.environ.copy()
    env.update({"TRAIN_SEED": "44", "RUN_PREFIX": "p13", "VENV_PYTHON": str(a.venv / "bin/python"), "PYTHONPATH": str(a.project)})
    legacy = a.project / "analysis/p10_saferlhf_training_seed43_20260718"
    if a.stage == 1:
        exp, label, parent = a.stage12, "stage1", None
        runner = legacy / "train_stage1_arm.py"
        common = ["--project", str(a.project), "--venv", str(a.venv), "--experiment", str(exp), "--base", a.base,
                  "--arm", a.arm, "--loss-type", loss, "--gpu", str(a.gpu), "--seed", "44", "--run-prefix", "p13", *target_args]
    elif a.stage == 2:
        exp, label, parent = a.stage12, "stage2", a.stage12 / f"stage1/{a.arm}/train/full"
        parent_gate = a.stage12 / f"stage1_stability_p8_locked_panel/gates/{a.arm}.json"
        if not passed(parent_gate): raise RuntimeError(f"Stage-1 parent gate not passed: {parent_gate}")
        pool = exp / f"stage2/{a.arm}/pool"
        if not (pool / "PREPARED").is_file():
            prep_env = env | {"PROJECT": str(a.project), "P4": str(a.project / "results/p4_8b_saferlhf_table4_20260717"), "VLLM_GPU_MEMORY_UTILIZATION": "0.55"}
            run(["bash", str(a.project / "analysis/p5_8b_robust_stage1_stage2_20260717/prepare_stage2_pool.sh"), str(exp), a.arm, str(parent), str(a.gpu), str(a.gpu)], prep_env, a.log)
        run([str(a.venv / "bin/python"), str(legacy / "audit_stage2_pool.py"), "--pool", str(pool), "--expected-prompts", "2500"], env, a.log)
        runner = legacy / "train_stage2_arm.py"
        common = ["--project", str(a.project), "--venv", str(a.venv), "--experiment", str(exp), "--arm", a.arm,
                  "--parent-model", str(parent), "--loss-type", loss, "--gpu", str(a.gpu), "--seed", "44", "--run-prefix", "p13", *target_args]
    else:
        exp = a.stage3 if a.stage == 3 else a.stage4
        label = f"stage{a.stage}"
        prior = a.stage12 if a.stage == 3 else a.stage3
        parent_label = f"stage{a.stage - 1}"
        parent = prior / f"{parent_label}/{a.arm}/train/full"
        parent_gate = prior / f"{parent_label}_stability_p8_locked_panel/gates/{a.arm}.json"
        if not passed(parent_gate): raise RuntimeError(f"parent gate not passed: {parent_gate}")
        pool = exp / f"{label}/{a.arm}/pool"
        if not (pool / "PREPARED").is_file():
            run(["bash", str(legacy / "prepare_continuation_pool.sh"), str(a.project), str(exp), label, a.arm, str(parent), str(a.gpu), str(a.gpu)], env, a.log)
        run([str(a.venv / "bin/python"), str(legacy / "audit_stage2_pool.py"), "--pool", str(pool), "--expected-prompts", "2500"], env, a.log)
        runner = legacy / "train_continuation_arm.py"
        common = ["--project", str(a.project), "--venv", str(a.venv), "--experiment", str(exp), "--continuation-stage", label,
                  "--arm", a.arm, "--parent-model", str(parent), "--loss-type", loss, "--gpu", str(a.gpu), "--seed", "44", "--run-prefix", "p13", *target_args]
    output = exp / f"{label}/{a.arm}/train"
    train(runner, common, output, env, a.log)
    gate = exp / f"{label}_stability_p8_locked_panel/gates/{a.arm}.json"
    if not passed(gate):
        run(["bash", str(legacy / "decode_and_gate_continuation.sh"), str(a.project), str(exp), label, a.arm, str(a.gpu)], env, a.log)
    if not passed(gate):
        raise RuntimeError(f"fail-closed stability gate failed: {gate}")
    print(json.dumps({"status": "completed", "stage": a.stage, "arm": a.arm, "gate": str(gate)}))


if __name__ == "__main__":
    main()
