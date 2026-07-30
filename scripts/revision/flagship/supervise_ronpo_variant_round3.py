#!/usr/bin/env python3
"""Wait for Round 2 RM selection, then execute and validate the final refinement round."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--flagship-root", type=Path, required=True)
    parser.add_argument("--variant-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--general-rm-cache", type=Path, required=True)
    parser.add_argument("--start-deadline", required=True)
    parser.add_argument("--validation-deadline", required=True)
    args = parser.parse_args()
    selection = args.run_dir / "round2_validation/results/selection_lock.json"
    deadline = datetime.fromisoformat(args.start_deadline)
    while True:
        if datetime.now().astimezone() >= deadline:
            raise RuntimeError("Round 3 start deadline reached before Round 2 validation selection")
        try:
            payload = json.loads(selection.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            time.sleep(20); continue
        if payload.get("status") == "locked_after_validation_rm_scoring_before_any_variant_panel_judgment": break
        time.sleep(20)
    grid = args.run_dir / "round3_grid.json"; lock = args.run_dir / "round3_grid_lock.json"
    run([args.python, str(args.project / "scripts/revision/flagship/build_ronpo_variant_round3_grid.py"),
         "--selection-lock", str(selection), "--evaluator-lock", str(args.run_dir / "evaluator_lock.json"),
         "--round2-grid", str(args.run_dir / "round2_grid.json"),
         "--output-grid", str(grid), "--output-lock", str(lock)],
        args.project, args.run_dir / "round3_grid_build.log")
    round_root = args.variant_root / "round3"
    run([args.python, str(args.project / "scripts/revision/flagship/run_ronpo_variant_search_round.py"),
         "--project", str(args.project), "--python", args.python,
         "--grid", str(grid), "--grid-lock", str(lock), "--target-dataset", str(args.target_dataset),
         "--work", str(round_root), "--base-model", args.base_model],
        args.project, args.run_dir / "round3_driver.log")
    run([args.python, str(args.project / "scripts/revision/flagship/supervise_ronpo_variant_round_validation.py"),
         "--project", str(args.project), "--python", args.python, "--round-id", "round3",
         "--round-root", str(round_root), "--grid", str(grid),
         "--flagship-root", str(args.flagship_root), "--run-dir", str(args.run_dir),
         "--evaluator-lock", str(args.run_dir / "evaluator_lock.json"),
         "--general-rm-cache", str(args.general_rm_cache), "--deadline", args.validation_deadline],
        args.project, args.run_dir / "round3_eval_supervisor.log")
    print(json.dumps({"status": "completed", "round3_selection":
                      str(args.run_dir / "round3_validation/results/selection_lock.json"),
                      "spent_sealed_split_touched": False}, indent=2))


if __name__ == "__main__":
    main()
