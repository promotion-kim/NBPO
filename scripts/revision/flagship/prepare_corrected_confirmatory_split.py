#!/usr/bin/env python3
"""Freeze a post-failure confirmatory holdout without consulting model scores.

The original model-selection split was the first 128 lexicographically sorted
validation prompts.  This script reserves every remaining validation prompt as
one confirmatory holdout and builds a score-free protocol-validation set from
the original 128 prompts plus the two prompts that exercised the frozen S3
guards in the consumed source-test run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_generation(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"generation artifact is not a list: {path}")
    return payload


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_digest(prompts: list[str]) -> str:
    return hashlib.sha256("\n".join(prompts).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-validation", type=Path, required=True)
    parser.add_argument("--failed-base-generations", type=Path, required=True)
    parser.add_argument("--failed-dpo-generations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-count", type=int, default=128)
    parser.add_argument("--think-leak-index", type=int, default=394)
    parser.add_argument("--repeat-collapse-index", type=int, default=252)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool = read_jsonl(args.pool_validation)
    by_prompt: dict[str, dict] = {}
    for row in pool:
        prompt = str(row["prompt"])
        if prompt in by_prompt and by_prompt[prompt].get("prompt_id") != row.get("prompt_id"):
            raise RuntimeError("duplicate prompt text has inconsistent prompt IDs")
        by_prompt[prompt] = row
    ordered_prompts = sorted(by_prompt)
    if len(ordered_prompts) <= args.selection_count:
        raise RuntimeError("validation pool does not contain a confirmatory remainder")
    selection_prompts = ordered_prompts[: args.selection_count]
    holdout_prompts = ordered_prompts[args.selection_count :]

    base_rows = read_generation(args.failed_base_generations)
    dpo_rows = read_generation(args.failed_dpo_generations)
    stress_prompts = [
        str(base_rows[args.think_leak_index]["prompt"]),
        str(dpo_rows[args.repeat_collapse_index]["prompt"]),
    ]
    if len(set(stress_prompts)) != 2:
        raise RuntimeError("expected two distinct S3 stress prompts")
    if set(stress_prompts) & set(ordered_prompts):
        raise RuntimeError("old source-test stress prompt overlaps validation pool")

    selection_rows = [
        {"prompt_id": by_prompt[prompt].get("prompt_id"), "prompt": prompt,
         "source": "original_nonsealed_validation_selection"}
        for prompt in selection_prompts
    ]
    stress_rows = [
        {"prompt_id": hashlib.sha256(prompt.encode()).hexdigest(), "prompt": prompt,
         "source": source}
        for prompt, source in zip(
            stress_prompts,
            ("consumed_sealed_think_leak_stress_only", "consumed_sealed_repeat_collapse_stress_only"),
        )
    ]
    protocol_validation_rows = selection_rows + stress_rows
    holdout_rows = [
        {"prompt_id": by_prompt[prompt].get("prompt_id"), "prompt": prompt,
         "source": "unused_nonsealed_validation_remainder"}
        for prompt in holdout_prompts
    ]

    paths = {
        "selection_validation": args.output_dir / "selection_validation_prompts.jsonl",
        "stress": args.output_dir / "s3_stress_prompts.jsonl",
        "protocol_validation": args.output_dir / "protocol_validation_prompts.jsonl",
        "confirmatory_holdout": args.output_dir / "confirmatory_holdout_prompts.jsonl",
    }
    write_jsonl(paths["selection_validation"], selection_rows)
    write_jsonl(paths["stress"], stress_rows)
    write_jsonl(paths["protocol_validation"], protocol_validation_rows)
    write_jsonl(paths["confirmatory_holdout"], holdout_rows)

    sets = {
        "selection_validation": set(selection_prompts),
        "stress": set(stress_prompts),
        "confirmatory_holdout": set(holdout_prompts),
    }
    manifest = {
        "schema_version": 1,
        "created_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "post-failure confirmatory reward evaluation under a newly frozen common decode guard",
        "selection_rule_inherited": "first 128 lexicographically sorted prompts from pool_validation; model selection remains locked to ronpo_k_only",
        "confirmatory_rule": "all lexicographically sorted pool_validation prompts after the first 128",
        "counts": {name: len(values) for name, values in sets.items()},
        "prompt_text_sha256": {
            "selection_validation": prompt_digest(selection_prompts),
            "stress": prompt_digest(stress_prompts),
            "protocol_validation": prompt_digest(selection_prompts + stress_prompts),
            "confirmatory_holdout": prompt_digest(holdout_prompts),
        },
        "file_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "overlaps": {
            "selection_confirmatory": len(sets["selection_validation"] & sets["confirmatory_holdout"]),
            "stress_confirmatory": len(sets["stress"] & sets["confirmatory_holdout"]),
            "stress_selection": len(sets["stress"] & sets["selection_validation"]),
        },
        "old_failure_evidence": {
            "think_leak_generation": str(args.failed_base_generations),
            "think_leak_index": args.think_leak_index,
            "repeat_collapse_generation": str(args.failed_dpo_generations),
            "repeat_collapse_index": args.repeat_collapse_index,
        },
        "holdout_status": "unopened_for_generation_or_scoring",
        "limitations": [
            "The confirmatory holdout comes from the original source-train validation partition, not the consumed source-test partition.",
            "The objective conflict gate had used the score-free response pool containing this partition; no flagship model generations or selection scores from these holdout prompts were used.",
        ],
    }
    if any(manifest["overlaps"].values()):
        raise RuntimeError(f"prompt overlap detected: {manifest['overlaps']}")
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
