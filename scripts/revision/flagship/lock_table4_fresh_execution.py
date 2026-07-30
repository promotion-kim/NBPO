#!/usr/bin/env python3
"""Bind the prospectively locked prompts to the validation-selected models before opening."""

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
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--fresh-manifest", type=Path, required=True)
    parser.add_argument("--metric-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    manifest = json.loads(args.fresh_manifest.read_text(encoding="utf-8"))
    metric = json.loads(args.metric_lock.read_text(encoding="utf-8"))
    if selection.get("status") != "VALIDATION_SELECTION_LOCKED_BEFORE_FRESH_TEST":
        raise RuntimeError("validation selection is not locked")
    if manifest.get("status") != "FRESH_TEST_PROMPTS_LOCKED_UNOPENED" or manifest.get("fresh_test_opened") is not False:
        raise RuntimeError("fresh prompts are not locked and unopened")
    if metric.get("status") != "LOCKED_BEFORE_VALIDATION_REAGGREGATION_OR_FRESH_MEASUREMENT":
        raise RuntimeError("metric is not locked")
    if manifest.get("metric_lock_sha256") != sha256(args.metric_lock):
        raise RuntimeError("manifest and metric lock disagree")
    payload = {
        "status": "FRESH_EXECUTION_LOCKED_BEFORE_OPENING",
        "locked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "fresh_manifest_sha256": sha256(args.fresh_manifest),
        "selection_lock_sha256": sha256(args.selection_lock),
        "metric_lock_sha256": sha256(args.metric_lock),
        "selected_ronpo_overall": selection["selected_ronpo_overall"],
        "selected_by_method": selection["selected_by_method"],
        "failed_methods": selection["failed_methods"],
        "fresh_test_opened": False,
        "spent_sealed_split_touched": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    args.output.with_suffix(".sha256").write_text(f"{sha256(args.output)}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
