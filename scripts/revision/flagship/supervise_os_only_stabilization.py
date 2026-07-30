#!/usr/bin/env python3
"""Resume-safe handoff from OS-only training to 647 gate and locked selection."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


def run(command: list[str], cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=True)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--fixed647", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--general-rm-cache", type=Path, required=True)
    parser.add_argument("--armo-cache", type=Path, required=True)
    parser.add_argument("--prior-metric-lock", type=Path, required=True)
    parser.add_argument("--deadline", required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    logs = args.result_root / "logs"
    status_path = args.result_root / "pipeline_status.json"
    while True:
        if datetime.now().astimezone() >= deadline:
            raise RuntimeError("deadline reached before OS training completed")
        try:
            training = json.loads((args.train_root / "training_manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            time.sleep(20); continue
        if training.get("status") == "completed":
            break
        if training.get("status") == "failed":
            atomic_json(status_path, {"status": "failed", "stage": "training",
                        "failures": training.get("failures"), "spent_sealed_split_touched": False})
            raise RuntimeError("OS-only training failed")
        atomic_json(status_path, {"status": "waiting", "stage": "training",
                    "training": training, "spent_sealed_split_touched": False})
        time.sleep(20)
    manifest = args.result_root / "sweep/checkpoint_manifest.json"
    if not manifest.is_file():
        run([args.python, str(args.project / "scripts/revision/flagship/build_ronpo_variant_checkpoint_manifest.py"),
             "--round", f"os_only={args.train_root}", "--grid", f"os_only={args.grid}",
             "--checkpoint-selection-split", "fixed 647-prompt held-out validation set",
             "--checkpoint-selection-metric", "min_local_rm_marginal_win_rate_vs_base",
             "--output", str(manifest)], args.project, logs / "build_checkpoint_manifest.log")
    gate_work = args.result_root / "full647_gate"
    gate_summary = gate_work / "gates/summary.json"
    if not gate_summary.is_file():
        atomic_json(status_path, {"status": "running", "stage": "full647_4096_stability_gate",
                    "spent_sealed_split_touched": False})
        run([args.python, str(args.project / "scripts/revision/flagship/run_os_only_full647_stability_gate.py"),
             "--project", str(args.project), "--python", args.python, "--manifest", str(manifest),
             "--fixed647", str(args.fixed647), "--base-model", args.base_model,
             "--work", str(gate_work), "--gpus", "0,1,2,3", "--hf-cache", str(args.hf_cache)],
            args.project, logs / "full647_gate_driver.log")
    gate = json.loads(gate_summary.read_text(encoding="utf-8"))
    if not gate.get("eligible_model_ids"):
        atomic_json(status_path, {"status": "terminal_failed", "stage": "full647_4096_stability_gate",
                    "reason": "no OS checkpoint passed", "spent_sealed_split_touched": False})
        return
    fresh = args.result_root / "fresh_preregistration/fresh_test_manifest.json"
    if not fresh.is_file():
        atomic_json(status_path, {"status": "blocked_before_reward_ranking",
                    "stage": "fresh_manifest_lock", "reason": "fresh manifest absent; no reward score computed",
                    "gate_summary": str(gate_summary), "spent_sealed_split_touched": False})
        return
    fresh_value = json.loads(fresh.read_text(encoding="utf-8"))
    if fresh_value.get("status") != "FRESH_TEST_PROMPTS_LOCKED_UNOPENED_BEFORE_OS_RANKING":
        raise RuntimeError("fresh manifest is not locked and unopened")
    selection_work = args.result_root / "fixed647_selection"
    atomic_json(status_path, {"status": "running", "stage": "fixed647_os_reward_selection",
                "spent_sealed_split_touched": False})
    run([args.python, str(args.project / "scripts/revision/flagship/run_os_only_fixed647_selection.py"),
         "--project", str(args.project), "--python", args.python, "--manifest", str(manifest),
         "--gate-summary", str(gate_summary), "--grid", str(args.grid),
         "--metric-lock", str(args.prior_metric_lock),
         "--new-metric-lock", str(args.result_root / "metric_lock.json"),
         "--fixed647", str(args.fixed647),
         "--base-generation", str(args.project / "results/p1_8b_stage2_20260716/fixed647/generations/base/output_42.json"),
         "--work", str(selection_work), "--general-rm-cache", str(args.general_rm_cache),
         "--armo-cache", str(args.armo_cache), "--hf-cache", str(args.hf_cache)],
        args.project, logs / "fixed647_selection_driver.log")
    selection = json.loads((selection_work / "selection_lock.json").read_text(encoding="utf-8"))
    atomic_json(status_path, {"status": "completed", "stage": "fixed647_os_selection_locked",
                "selection_status": selection.get("status"),
                "selected_model_id": (selection.get("selected") or {}).get("model_id"),
                "spent_sealed_split_touched": False})


if __name__ == "__main__":
    main()
