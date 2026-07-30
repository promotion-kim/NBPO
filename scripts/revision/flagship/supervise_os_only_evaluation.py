#!/usr/bin/env python3
"""Resume-safe OS-only fixed-647, one-shot fresh, and independent-panel evaluation."""

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
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--fixed647", type=Path, required=True)
    parser.add_argument("--prior-fixed647", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--general-rm-cache", type=Path, required=True)
    parser.add_argument("--armo-cache", type=Path, required=True)
    parser.add_argument("--evaluator-lock", type=Path, required=True)
    parser.add_argument("--local-metric-lock", type=Path, required=True)
    parser.add_argument("--panel-metric-lock", type=Path, required=True)
    parser.add_argument("--qwen-judge", type=Path, required=True)
    parser.add_argument("--gpt-judge", type=Path, required=True)
    parser.add_argument("--deadline", required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    selection = args.result_root / "fixed647_selection/selection_lock.json"
    status = args.result_root / "evaluation_status.json"
    while not selection.is_file():
        if datetime.now().astimezone() >= deadline:
            raise RuntimeError("deadline reached before OS selection lock")
        pipeline = {}
        try:
            pipeline = json.loads((args.result_root / "pipeline_status.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        if pipeline.get("status") in {"failed", "terminal_failed"}:
            atomic_json(status, {"status": "terminal_failed_before_evaluation", "pipeline": pipeline,
                        "spent_sealed_split_touched": False})
            return
        atomic_json(status, {"status": "waiting_for_selection_lock", "pipeline": pipeline,
                    "spent_sealed_split_touched": False})
        time.sleep(20)
    selected = json.loads(selection.read_text(encoding="utf-8"))
    if selected.get("status") != "OS_SELECTION_LOCKED_BEFORE_FRESH_TEST_EXECUTION":
        atomic_json(status, {"status": "terminal_failed_before_evaluation",
                    "selection_status": selected.get("status"), "spent_sealed_split_touched": False})
        return
    ledger = args.result_root / "sweep/baseline_reuse_ledger.json"
    fixed_work = args.result_root / "fixed647_exact_table4"
    if not (fixed_work / "results/model_summary.json").is_file():
        atomic_json(status, {"status": "running", "stage": "fixed647_exact_table4",
                    "spent_sealed_split_touched": False})
        run([args.python, str(args.project / "scripts/revision/flagship/run_os_only_exact_fixed647_panel.py"),
             "--project", str(args.project), "--python", args.python, "--ledger", str(ledger),
             "--selection-lock", str(selection),
             "--selection-work", str(args.result_root / "fixed647_selection"),
             "--fixed647", str(args.fixed647), "--prior-fixed647", str(args.prior_fixed647),
             "--metric-lock", str(args.local_metric_lock), "--work", str(fixed_work),
             "--hf-cache", str(args.hf_cache), "--general-rm-cache", str(args.general_rm_cache),
             "--armo-cache", str(args.armo_cache)],
            args.project, args.result_root / "logs/exact_fixed647_driver.log")
    fresh_manifest = args.result_root / "fresh_preregistration/fresh_test_manifest.json"
    fresh_results = args.result_root / "fresh_test/results/localrm/model_summary.json"
    if not fresh_results.is_file():
        atomic_json(status, {"status": "running", "stage": "fresh_one_shot_localrm",
                    "spent_sealed_split_touched": False})
        run([args.python, str(args.project / "scripts/revision/flagship/run_os_only_fresh_localrm.py"),
             "--project", str(args.project), "--python", args.python, "--ledger", str(ledger),
             "--selection-lock", str(selection), "--fresh-manifest", str(fresh_manifest),
             "--metric-lock", str(args.local_metric_lock), "--run-dir", str(args.result_root),
             "--hf-cache", str(args.hf_cache), "--general-rm-cache", str(args.general_rm_cache),
             "--armo-cache", str(args.armo_cache)],
            args.project, args.result_root / "logs/fresh_localrm_driver.log")
    panel = args.result_root / "fresh/panel_summary.json"
    if not panel.is_file():
        atomic_json(status, {"status": "running", "stage": "fresh_independent_panel",
                    "spent_sealed_split_touched": False})
        run([args.python, str(args.project / "scripts/revision/flagship/run_table4_fresh_panel.py"),
             "--project", str(args.project), "--python", args.python,
             "--flagship-root", str(args.hf_cache.parents[2]), "--run-dir", str(args.result_root),
             "--grid", str(args.result_root / "fresh_test/grid.json"),
             "--evaluator-lock", str(args.evaluator_lock),
             "--metric-lock", str(args.panel_metric_lock),
             "--qwen-judge", str(args.qwen_judge), "--gpt-judge", str(args.gpt_judge)],
            args.project, args.result_root / "logs/fresh_panel_driver.log")
    atomic_json(status, {"status": "completed", "stage": "fresh_localrm_and_independent_panel",
                "fixed647_summary": str(fixed_work / "results/model_summary.json"),
                "fresh_localrm_summary": str(fresh_results), "fresh_panel_summary": str(panel),
                "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "spent_sealed_split_touched": False})


if __name__ == "__main__":
    main()
