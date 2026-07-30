#!/usr/bin/env python3
"""Fail-closed gates and aligned merge for the locked fixed-test generations."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, default=647)
    args = parser.parse_args()
    lock = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_AFTER_VALIDATION_BEFORE_FIXED647_DECODE":
        raise RuntimeError("selection is not locked")
    models = [lock["base"], *lock["selected"]]
    base_file = args.work / "generations/base/output_42.json"
    gates = []; gate_dir = args.work / "stability_gates"; logs = args.work / "logs"
    gate_dir.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    for row in models[1:]:
        candidate = args.work / "generations" / row["candidate_id"] / "output_42.json"
        output = gate_dir / f"{row['candidate_id']}.json"
        command = [args.python, str(args.project / "scripts/revision/flagship/stability_gate_corrected.py"),
                   "--base", str(base_file), "--candidate", str(candidate), "--output", str(output),
                   "--expected-records", str(args.expected_records), "--min-length-ratio", "0.33",
                   "--max-length-ratio", "2.0", "--max-repeat-run", "20"]
        result = subprocess.run(command, cwd=args.project, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False)
        (logs / f"gate_{row['candidate_id']}.log").write_text(result.stdout, encoding="utf-8")
        if result.returncode not in {0, 4} or not output.is_file():
            raise RuntimeError(f"gate runner error: {row['candidate_id']}, rc={result.returncode}")
        value = json.loads(output.read_text(encoding="utf-8"))
        gates.append({"id": row["candidate_id"], "method": row["method"], "passed": value["passed"],
                      "status": value["status"], "checks": value["checks"],
                      "candidate": value["candidate"],
                      "candidate_base_mean_word_ratio": value["candidate_base_mean_word_ratio"]})
    eligible = [row["id"] for row in gates if row["passed"]]
    failed = [row["id"] for row in gates if not row["passed"]]
    atomic_json(gate_dir / "summary.json", {
        "status": "completed_fail_closed", "detector": "corrected_nonempty_paired_span_v1",
        "thresholds": {"records": args.expected_records, "empty": 0, "think_leaks": 0,
                       "mean_word_ratio": [0.33, 2.0], "max_repeat_run": 20},
        "eligible_candidates": eligible, "failed_candidates": failed, "rows": gates,
        "reward_scores_consulted": False, "spent_sealed_split_touched": False})
    merge = [args.python, "-m", "mnpo_scripts.merge_model_generations", "--generations",
             f"base={base_file}"]
    merge += [f"{candidate_id}={args.work / 'generations' / candidate_id / 'output_42.json'}"
              for candidate_id in eligible]
    merged = args.work / "merged_generations.json"; merge += ["--output_file", str(merged)]
    with (logs / "merge.log").open("w", encoding="utf-8") as handle:
        subprocess.run(merge, cwd=args.project, stdout=handle, stderr=subprocess.STDOUT, check=True)
    rows = json.loads(merged.read_text(encoding="utf-8"))
    if len(rows) != args.expected_records:
        raise RuntimeError("merged fixed test has the wrong prompt count")
    atomic_json(args.work / "preparation_status.json", {
        "status": "ready_for_locked_reward_scoring", "prompt_count": len(rows),
        "eligible_candidates": eligible, "failed_candidates": failed,
        "spent_sealed_split_touched": False})
    print(json.dumps({"eligible": eligible, "failed": failed, "merged": str(merged)}, indent=2))


if __name__ == "__main__":
    main()
