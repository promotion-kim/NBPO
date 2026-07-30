#!/usr/bin/env python3
"""Lock the cross-round finalist, run its prospective panel, and finalize artifacts."""

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
    parser.add_argument("--fair-root", type=Path, required=True)
    parser.add_argument("--fair-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--general-rm-cache", type=Path, required=True)
    parser.add_argument("--qwen-judge", type=Path, required=True)
    parser.add_argument("--gpt-judge", type=Path, required=True)
    parser.add_argument("--deadline", required=True)
    args = parser.parse_args()
    selections = [args.run_dir / "round1_validation/results/selection_lock.json",
                  args.run_dir / "round2_validation/results/selection_lock.json",
                  args.run_dir / "round3_validation/results/selection_lock.json"]
    deadline = datetime.fromisoformat(args.deadline)
    while True:
        if datetime.now().astimezone() >= deadline:
            raise RuntimeError("final-confirmation launch deadline reached before Round 2 selection")
        good = True
        for path in selections:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                good = good and payload.get("status") == "locked_after_validation_rm_scoring_before_any_variant_panel_judgment"
            except (OSError, json.JSONDecodeError):
                good = False
        if good: break
        time.sleep(20)
    fresh_prompts = args.fair_run / "fresh_test_preregistration/fresh_test_prompts.jsonl"
    final_lock = args.run_dir / "final_variant_set_lock.json"
    run([args.python, str(args.project / "scripts/revision/flagship/lock_ronpo_variant_finalist.py"),
         "--selection", str(selections[0]), "--selection", str(selections[1]),
         "--selection", str(selections[2]),
         "--evaluator-lock", str(args.run_dir / "evaluator_lock.json"),
         "--fresh-prompts", str(fresh_prompts), "--output", str(final_lock)],
        args.project, args.run_dir / "final_lock.log")
    confirmation = args.run_dir / "final_confirmation/fresh_test"
    run([args.python, str(args.project / "scripts/revision/flagship/run_ronpo_variant_final_decode_gate.py"),
         "--project", str(args.project), "--python", args.python,
         "--flagship-root", str(args.flagship_root), "--fair-run", str(args.fair_run),
         "--final-lock", str(final_lock), "--fresh-prompts", str(fresh_prompts),
         "--work", str(confirmation)], args.project, args.run_dir / "final_decode_gate.log")
    # This invocation starts at reward scoring and judge inference.  It does not
    # decode and cannot change the validation-locked finalist.
    run([args.python, str(args.project / "scripts/revision/flagship/run_fair_demo_validation_scoring.py"),
         "--project", str(args.project), "--python", args.python,
         "--flagship-root", str(args.flagship_root), "--fair-root", str(args.fair_root),
         "--run-dir", str(args.run_dir / "final_confirmation"),
         "--grid", str(args.run_dir / "round3_grid.json"), "--split", "fresh_test",
         "--evaluator-lock", str(args.run_dir / "evaluator_lock.json"),
         "--general-rm-cache", str(args.general_rm_cache),
         "--qwen-judge", str(args.qwen_judge), "--gpt-judge", str(args.gpt_judge)],
        args.project, args.run_dir / "final_scoring.log")
    run([args.python, str(args.project / "scripts/revision/flagship/finalize_ronpo_variant_search.py"),
         "--run-dir", str(args.run_dir), "--final-lock", str(final_lock),
         "--candidate-panel", str(confirmation / "results/panel/panel_summary.json"),
         "--calibration-panel", str(args.run_dir / "baseline_calibration/results/panel/panel_summary.json"),
         "--gates", str(confirmation / "stability_gates/summary.json"),
         "--reward-summary", str(confirmation / "results/rewards/reward_summary.json"),
         "--main-tex", str(args.project / "ronpo_aaai/main_v3.tex"),
         "--main-pdf", str(args.project / "ronpo_aaai/main_v3.pdf")],
        args.project, args.run_dir / "finalize.log")
    print(json.dumps({"status": "completed", "summary": str(args.run_dir / "summary.json"),
                      "spent_sealed_split_touched": False}, indent=2))


if __name__ == "__main__":
    main()
