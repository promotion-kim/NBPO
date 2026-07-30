#!/usr/bin/env python3
"""Lock one validation-selected finalist before any new variant panel judgment."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", action="append", type=Path, required=True)
    parser.add_argument("--evaluator-lock", type=Path, required=True)
    parser.add_argument("--fresh-prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    finalists = []
    all_attempts = []
    inputs = {}
    for path in args.selection:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "locked_after_validation_rm_scoring_before_any_variant_panel_judgment":
            raise RuntimeError(f"selection is not properly locked: {path}")
        if payload.get("panel_judgments_used_for_selection") is not False:
            raise RuntimeError("panel leakage into variant selection")
        inputs[str(path)] = sha256(path)
        all_attempts.extend(payload["per_candidate_selected_checkpoint"])
        finalists.extend(row for row in payload["per_candidate_selected_checkpoint"]
                         if row.get("status") == "selected_on_validation_locked_rm_metric")
    if not finalists:
        raise RuntimeError("no S3-eligible validation finalist")
    selected = sorted(finalists, key=lambda row: (-float(row["selection_metric"]),
                                                  int(row["step"]), row["candidate_id"]))[0]
    lock = {
        "status": "FINAL_VARIANT_SET_LOCKED_BEFORE_FRESH_PANEL",
        "locked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_split": "existing prompt-disjoint 128-prompt validation",
        "selection_metric": "mean_prompt_worst_standardized_delta",
        "selected_variant": selected,
        "all_validation_selected_candidates": finalists,
        "all_attempted_variants": all_attempts,
        "selection_lock_sha256": inputs,
        "evaluator_lock_sha256": sha256(args.evaluator_lock),
        "fresh_prompt_sha256": sha256(args.fresh_prompts),
        "fresh_prompt_count": 1024,
        "fresh_test_context": "Previously opened for frozen-baseline calibration; this variant was locked without any panel judgment on these prompts.",
        "baseline_calibration_results_known_before_search": True,
        "variant_panel_judgments_seen_before_lock": False,
        "primary_success_rule": {
            "beat": "point estimate >= 0.5 and paired delta-vs-base CI lower bound > 0",
            "match": "point estimate >= 0.5 and paired delta-vs-base CI includes zero",
        },
        "spent_sealed_split_touched": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
