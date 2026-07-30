#!/usr/bin/env python3
"""Lock validation-selected Stage-1 checkpoints before the fixed 647 test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def resolve_path(row: dict, hf_cache: Path) -> str:
    value = str(row["model_path"])
    if not value.startswith("promotion/"):
        if not Path(value).is_dir():
            raise RuntimeError(f"selected local checkpoint is missing: {value}")
        return value
    repo, revision = value.split("@", 1)
    snapshot = hf_cache / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    if not snapshot.is_dir():
        raise RuntimeError(f"selected frozen snapshot is missing: {snapshot}")
    return str(snapshot)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--metric-lock", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--base-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    gates = json.loads(args.gates.read_text(encoding="utf-8"))
    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    lock = json.loads(args.metric_lock.read_text(encoding="utf-8"))
    if summary.get("split") != "frozen_128_prompt_validation" or summary.get("status") != "completed":
        raise RuntimeError("validation summary is not complete")
    test_sha = hashlib.sha256(args.test_file.read_bytes()).hexdigest()
    if test_sha != lock["stage1_test"]["sha256"]:
        raise RuntimeError("fixed 647 test hash differs from preregistration")
    if sum(bool(line.strip()) for line in args.test_file.open(encoding="utf-8")) != 647:
        raise RuntimeError("fixed test does not have 647 records")
    if not args.base_snapshot.is_dir():
        raise RuntimeError("exact base snapshot is missing")
    selected = []
    for row in summary["selected_candidate_per_method"]:
        selected.append({
            "method": row["method"], "candidate_id": row["model"], "stage": row["stage"],
            "model_path": resolve_path(row, args.hf_cache), "source_model_path": row["model_path"],
            "validation_primary": row["worst_objective_marginal_win_rate"],
            "validation_primary_ci95": row["worst_objective_marginal_win_rate_ci95"],
        })
    failed_by_method = {}
    candidate_method = {row["id"]: row["method"] for row in grid["candidates"]}
    for candidate_id in gates["failed_candidates"]:
        failed_by_method.setdefault(candidate_method[candidate_id], []).append(candidate_id)
    expected_methods = set(grid["candidate_count_by_method"])
    selected_methods = {row["method"] for row in selected}
    terminal = sorted(expected_methods - selected_methods)
    if terminal != ["simpo"]:
        raise RuntimeError(f"unexpected methods with no stable candidate: {terminal}")
    payload = {
        "status": "LOCKED_AFTER_VALIDATION_BEFORE_FIXED647_DECODE",
        "locked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_split": "frozen_128_prompt_validation",
        "selection_metric": lock["primary"], "test_used_for_selection": False,
        "base": {"candidate_id": "base", "method": "base", "stage": 0,
                 "model_path": str(args.base_snapshot),
                 "hf_repo": "Qwen/Qwen3-8B", "hf_revision": args.base_snapshot.name},
        "selected": selected,
        "terminal_failed_methods": {method: failed_by_method[method] for method in terminal},
        "fixed_test": {"path": str(args.test_file), "sha256": test_sha, "prompt_count": 647},
        "spent_sealed_split_touched": False,
    }
    atomic_json(args.output, payload)
    tsv = args.output.with_suffix(".tsv")
    with tsv.open("w", newline="", encoding="utf-8") as handle:
        fields = ["method", "candidate_id", "stage", "model_path", "validation_primary"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader(); writer.writerow({**payload["base"], "validation_primary": 0.5})
        writer.writerows(selected)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
