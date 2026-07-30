#!/usr/bin/env python3
"""Outcome-blind validation preparation for the locked Stage-1 OS comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_rows(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise RuntimeError(f"not a JSON list: {path}")
    return value


def prompt_hash(rows: list[dict]) -> str:
    payload = "\n".join(str(row["prompt"]) for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--base-generation", type=Path, required=True)
    args = parser.parse_args()

    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    if grid.get("status") != "LOCKED_BEFORE_VALIDATION_LOCAL_RM_RANKING":
        raise RuntimeError("candidate grid was not locked before ranking")
    if grid.get("spent_sealed_split_touched") is not False:
        raise RuntimeError("invalid sealed-split audit marker")
    base_rows = load_rows(args.base_generation)
    if len(base_rows) != 128:
        raise RuntimeError("base validation generation must have exactly 128 records")
    base_prompts = [str(row["prompt"]) for row in base_rows]
    common_hash = prompt_hash(base_rows)
    prompt_file = args.work / "validation/validation_prompts.jsonl"
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text("".join(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n"
                                   for prompt in base_prompts), encoding="utf-8")

    availability = []
    available = []
    for candidate in grid["candidates"]:
        generation = Path(candidate["validation_generation"])
        model_path = str(candidate["model_path"])
        local_model = Path(model_path).exists() if not model_path.startswith("promotion/") else None
        row = {
            "id": candidate["id"], "method": candidate["method"],
            "generation": str(generation), "generation_available": generation.is_file(),
            "model_path": model_path, "local_model_available": local_model,
        }
        if generation.is_file():
            records = load_rows(generation)
            row["record_count"] = len(records)
            row["prompt_sequence_sha256"] = prompt_hash(records)
            row["prompt_aligned"] = [str(value["prompt"]) for value in records] == base_prompts
            if len(records) != 128 or not row["prompt_aligned"]:
                raise RuntimeError(f"unaligned validation generation: {candidate['id']}")
            available.append(candidate)
        else:
            row["exclusion_reason"] = "pre-locked validation generation is unavailable; no post-lock decode"
        availability.append(row)

    counts: dict[str, int] = {}
    for candidate in available:
        counts[candidate["method"]] = counts.get(candidate["method"], 0) + 1
    expected_counts = {str(key): int(value) for key, value in grid["candidate_count_by_method"].items()}
    if counts != expected_counts:
        raise RuntimeError(f"available grid differs from the pre-locked counts: {counts} != {expected_counts}")
    atomic_json(args.work / "sweep/candidate_availability.json", {
        "status": "verified_before_reward_scoring", "created_at": datetime.now().astimezone().isoformat(),
        "common_prompt_sequence_sha256": common_hash, "available_count_by_method": counts,
        "rows": availability, "spent_sealed_split_touched": False,
    })

    gate_dir = args.work / "validation/stability_gates"
    log_dir = args.work / "validation/logs"
    gate_dir.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)
    gates = []
    for candidate in available:
        output = gate_dir / f"{candidate['id']}.json"
        command = [args.python, str(args.project / "scripts/revision/flagship/stability_gate_corrected.py"),
                   "--base", str(args.base_generation), "--candidate", candidate["validation_generation"],
                   "--output", str(output), "--expected-records", "128", "--min-length-ratio", "0.33",
                   "--max-length-ratio", "2.0", "--max-repeat-run", "20"]
        result = subprocess.run(command, cwd=args.project, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False)
        (log_dir / f"gate_{candidate['id']}.log").write_text(result.stdout, encoding="utf-8")
        if result.returncode not in {0, 4} or not output.is_file():
            raise RuntimeError(f"gate execution error for {candidate['id']}: rc={result.returncode}")
        payload = json.loads(output.read_text(encoding="utf-8"))
        gates.append({"id": candidate["id"], "method": candidate["method"],
                      "passed": payload["passed"], "status": payload["status"],
                      "checks": payload["checks"], "candidate": payload["candidate"],
                      "candidate_base_mean_word_ratio": payload["candidate_base_mean_word_ratio"]})
    eligible = [row["id"] for row in gates if row["passed"]]
    failed = [row["id"] for row in gates if not row["passed"]]
    gate_summary = {
        "status": "completed_fail_closed", "detector": "corrected_nonempty_paired_span_v1",
        "thresholds": {"records": 128, "empty": 0, "think_leaks": 0,
                       "mean_word_ratio": [0.33, 2.0], "max_repeat_run": 20},
        "eligible_candidates": eligible, "failed_candidates": failed, "rows": gates,
        "reward_scores_consulted": False, "spent_sealed_split_touched": False,
    }
    atomic_json(gate_dir / "summary.json", gate_summary)

    by_id = {row["id"]: row for row in available}
    merge = [args.python, "-m", "mnpo_scripts.merge_model_generations", "--generations",
             f"base={args.base_generation}"]
    merge += [f"{candidate_id}={by_id[candidate_id]['validation_generation']}" for candidate_id in eligible]
    merged = args.work / "validation/merged_generations.json"
    merge += ["--output_file", str(merged)]
    with (log_dir / "merge.log").open("w", encoding="utf-8") as handle:
        subprocess.run(merge, cwd=args.project, stdout=handle, stderr=subprocess.STDOUT, check=True)
    merged_rows = load_rows(merged)
    expected_names = ["base", *eligible]
    if len(merged_rows) != 128 or any(row["response_model_names"] != expected_names for row in merged_rows):
        raise RuntimeError("merged validation alignment failed")
    atomic_json(args.work / "validation/preparation_status.json", {
        "status": "ready_for_locked_reward_scoring", "prompt_count": 128,
        "prompt_sequence_sha256": common_hash, "model_order": expected_names,
        "validation_prompt_file": str(prompt_file),
        "gate_summary": str(gate_dir / "summary.json"), "spent_sealed_split_touched": False,
    })
    print(json.dumps({"eligible": eligible, "failed": failed, "merged": str(merged)}, indent=2))


if __name__ == "__main__":
    main()
