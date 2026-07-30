#!/usr/bin/env python3
"""Resume the fair demo after an immutable evaluator lock without re-running diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(command: list[str], *, cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=True)


def completed(path: Path, stage: str) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("status") == "completed" and payload.get("stage") == stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--flagship-root", type=Path, required=True)
    parser.add_argument("--fair-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--general-rm-cache", type=Path, required=True)
    parser.add_argument("--qwen-judge", type=Path, required=True)
    parser.add_argument("--gpt-judge", type=Path, required=True)
    args = parser.parse_args()

    logs = args.run_dir / "pipeline_logs"
    status_path = args.run_dir / "pipeline_status.json"
    evaluator_lock = args.run_dir / "evaluator_lock.json"
    evaluator_sha = args.run_dir / "evaluator_lock.sha256"
    evaluator = json.loads(evaluator_lock.read_text(encoding="utf-8"))
    expected_hash = evaluator_sha.read_text(encoding="utf-8").split()[0]
    if evaluator.get("status") != "LOCKED_BEFORE_ANY_NEW_METHOD_RANKING":
        raise RuntimeError("immutable evaluator lock is absent")
    if sha256(evaluator_lock) != expected_hash:
        raise RuntimeError("immutable evaluator lock hash mismatch")
    if evaluator.get("spent_sealed_split_touched") is not False:
        raise RuntimeError("spent-sealed guard failed")

    grid = args.run_dir / "sweep/grid.json"
    validation_status = args.run_dir / "validation/status.json"
    if not completed(validation_status, "validation_decode_and_stability_gate"):
        raise RuntimeError("validation decode and gate are not complete")

    scoring_common = [
        args.python, str(args.project / "scripts/revision/flagship/run_fair_demo_validation_scoring.py"),
        "--project", str(args.project), "--python", args.python,
        "--flagship-root", str(args.flagship_root), "--fair-root", str(args.fair_root),
        "--run-dir", str(args.run_dir), "--grid", str(grid),
        "--evaluator-lock", str(evaluator_lock), "--general-rm-cache", str(args.general_rm_cache),
        "--qwen-judge", str(args.qwen_judge), "--gpt-judge", str(args.gpt_judge),
    ]
    selection_lock = args.run_dir / "validation/results/panel/selection_lock.json"
    if not selection_lock.is_file():
        atomic_json(status_path, {"status": "running", "stage": "validation_locked_scoring_resume",
                                  "evaluator_lock_sha256": expected_hash,
                                  "spent_sealed_split_touched": False,
                                  "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
        run(scoring_common, cwd=args.project, log=logs / "validation_scoring_resume.log")
    selection = json.loads(selection_lock.read_text(encoding="utf-8"))
    if selection.get("status") != "VALIDATION_SELECTION_LOCKED_BEFORE_FRESH_TEST":
        raise RuntimeError("validation selection lock is invalid")
    if selection.get("evaluator_lock_sha256") != expected_hash:
        raise RuntimeError("selection was not made with the immutable evaluator")

    fresh_prereg = args.run_dir / "fresh_test_preregistration"
    fresh_manifest = fresh_prereg / "fresh_test_manifest.json"
    fresh_prompts = fresh_prereg / "fresh_test_prompts.jsonl"
    if not fresh_manifest.is_file():
        atomic_json(status_path, {"status": "running", "stage": "fresh_test_preregistration",
                                  "spent_sealed_split_touched": False,
                                  "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
        run([
            args.python, str(args.project / "scripts/revision/flagship/prepare_fair_demo_fresh_test.py"),
            "--selection-lock", str(selection_lock), "--evaluator-lock", str(evaluator_lock),
            "--avg-precomputed", str(args.flagship_root / "precomputed/avg"),
            "--validation-prompts", str(args.flagship_root / "data/pool_validation.jsonl"),
            "--base-model", str(args.base_model), "--output-dir", str(fresh_prereg),
            "--cache-dir", str(args.flagship_root / "cache/huggingface"),
        ], cwd=args.project, log=logs / "prepare_fresh_test.log")

    fresh_status = args.run_dir / "fresh_test/status.json"
    if not completed(fresh_status, "fresh_decode_and_stability_gate"):
        atomic_json(status_path, {"status": "running", "stage": "fresh_test_decode_gate",
                                  "spent_sealed_split_touched": False,
                                  "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
        run([
            args.python, str(args.project / "scripts/revision/flagship/run_fair_demo_fresh_decode_gate.py"),
            "--project", str(args.project), "--python", args.python,
            "--flagship-root", str(args.flagship_root), "--fair-root", str(args.fair_root),
            "--run-dir", str(args.run_dir), "--selection-lock", str(selection_lock),
            "--fresh-manifest", str(fresh_manifest), "--fresh-prompts", str(fresh_prompts),
            "--base-model", str(args.base_model),
        ], cwd=args.project, log=logs / "fresh_decode_gate.log")

    fresh_scoring = args.run_dir / "fresh_test/scoring_status.json"
    if not completed(fresh_scoring, "fresh_test_scoring_completed"):
        atomic_json(status_path, {"status": "running", "stage": "fresh_test_locked_scoring",
                                  "spent_sealed_split_touched": False,
                                  "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
        run([*scoring_common, "--split", "fresh_test"], cwd=args.project,
            log=logs / "fresh_scoring.log")

    run([args.python, str(args.project / "scripts/revision/flagship/build_fair_demo_report.py"),
         "--run-dir", str(args.run_dir), "--fair-root", str(args.fair_root)],
        cwd=args.project, log=logs / "build_report.log")
    if not (args.run_dir / "wandb_eval_run.json").is_file():
        run([args.python, str(args.project / "scripts/revision/flagship/log_fair_demo_results_wandb.py"),
             "--run-dir", str(args.run_dir)], cwd=args.project, log=logs / "wandb_final_eval.log")
    atomic_json(status_path, {"status": "running", "stage": "verified_public_hf_upload",
                              "spent_sealed_split_touched": False,
                              "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
    run([args.python, str(args.project / "scripts/revision/flagship/upload_fair_demo_selected_to_hf.py"),
         "--selection-lock", str(selection_lock), "--fair-root", str(args.fair_root),
         "--run-dir", str(args.run_dir), "--namespace", "promotion"],
        cwd=args.project, log=logs / "hf_upload.log")
    atomic_json(status_path, {"status": "completed", "stage": "report_built_and_verified",
                              "report": str(args.run_dir / "REPORT.md"),
                              "evaluator_lock_sha256": expected_hash,
                              "spent_sealed_split_touched": False,
                              "completed_at": datetime.now().astimezone().isoformat(timespec="seconds")})


if __name__ == "__main__":
    main()
