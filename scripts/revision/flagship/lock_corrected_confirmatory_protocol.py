#!/usr/bin/env python3
"""Freeze the corrected common decode only after all non-headline gates pass."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--protocol-candidate", type=Path, required=True)
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--confirmatory-prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split = json.loads(args.split_manifest.read_text())
    candidate = json.loads(args.protocol_candidate.read_text())
    validation = json.loads(args.validation_summary.read_text())
    selection = json.loads(args.selection_lock.read_text())
    if validation.get("all_passed") is not True:
        raise RuntimeError("cannot freeze corrected decode: protocol validation did not pass all models")
    failed = [name for name, gate in validation.get("models", {}).items()
              if gate.get("passed") is not True]
    if failed:
        raise RuntimeError(f"cannot freeze corrected decode; failed models: {failed}")
    if selection.get("status") != "locked" or selection.get("selected_model_name") != "ronpo_k_only":
        raise RuntimeError("the pre-source-test top-mass selection lock is missing or changed")
    prompt_count = int(split["counts"]["confirmatory_holdout"])
    file_digest = sha256_file(args.confirmatory_prompts)
    if file_digest != split["file_sha256"]["confirmatory_holdout"]:
        raise RuntimeError("confirmatory prompt file does not match split manifest")
    if any(split.get("overlaps", {}).values()):
        raise RuntimeError("confirmatory split manifest reports prompt overlap")

    payload = {
        "schema_version": 1,
        "status": "frozen_after_protocol_validation",
        "frozen_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "decode": candidate["decode"],
        "stability_gate": {"records": prompt_count, "empty": 0, "think_leak": 0,
                           "length_ratio": [0.33, 2.0], "max_repeat_run": 20},
        "protocol_validation": {
            "records": candidate["stability_gate"]["records"],
            "summary_path": str(args.validation_summary),
            "summary_sha256": sha256_file(args.validation_summary),
            "all_models_passed": True,
        },
        "confirmatory_prompt_count": prompt_count,
        "confirmatory_file_sha256": file_digest,
        "confirmatory_prompt_text_sha256": split["prompt_text_sha256"]["confirmatory_holdout"],
        "confirmatory_rule": split["confirmatory_rule"],
        "confirmatory_holdout_opened": False,
        "selection_lock": selection,
        "model_selection_changed_after_source_test": False,
        "prior_source_test": {
            "status": "consumed_and_failed_closed_before_reward_scoring",
            "artifact": "results/p1_sealed_reward_seed42_20260714/status.json",
        },
        "provenance_limitations": split["limitations"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        comparable = dict(payload)
        comparable.pop("frozen_at_kst", None)
        previous.pop("frozen_at_kst", None)
        if previous != comparable:
            raise RuntimeError("a different corrected protocol is already locked")
        return
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
