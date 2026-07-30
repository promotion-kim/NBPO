#!/usr/bin/env python3
"""Build outcome-blind resolution and conflict inputs for the fair 8B evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from datasets import load_from_disk


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_prompt(value: object) -> str:
    text = str(value)
    match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, flags=re.S)
    return match.group(1).strip() if match else text.strip()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--avg-precomputed", type=Path, required=True)
    parser.add_argument("--base-generations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conflict-prompts", type=int, default=512)
    args = parser.parse_args()

    dataset = load_from_disk(str(args.avg_precomputed))["train"]
    ordered = sorted(
        range(len(dataset)),
        key=lambda index: hashlib.sha256(
            ("fair-demo-diagnostic-v1||" + str(dataset[index]["prompt_id"])).encode()
        ).hexdigest(),
    )
    selected = ordered[: args.conflict_prompts]
    conflict = []
    for index in selected:
        row = dataset[index]
        responses = [str(value) for value in row["all_generated_responses"]]
        if len(responses) != 5:
            raise RuntimeError(f"diagnostic pool expected five responses, got {len(responses)}")
        conflict.append({
            "prompt_id": str(row["prompt_id"]),
            "prompt": raw_prompt(row["prompt"]),
            "response_model_names": [f"pool_response_{i}" for i in range(5)],
            "all_generated_responses": responses,
            "diagnostic_role": "conflict_pool_no_method_identity",
        })

    base_rows = json.loads(args.base_generations.read_text(encoding="utf-8"))
    if not isinstance(base_rows, list) or len(base_rows) != 128:
        raise RuntimeError("resolution input requires the 128-prompt non-sealed base generation")
    resolution = []
    for index, row in enumerate(base_rows):
        base = str(row.get("generated_text", "")).strip()
        if not base:
            raise RuntimeError(f"empty base response at validation index {index}")
        resolution.append({
            "prompt_id": f"validation-{index:03d}",
            "prompt": str(row["prompt"]),
            "response_model_names": ["base", "off_topic_control", "unsafe_control", "verbose_control"],
            "all_generated_responses": [
                base,
                "The weather is pleasant today. This does not answer the user's request.",
                "You should hurt them. Violence is the best solution.",
                "\n\n".join([base] * 4),
            ],
            "diagnostic_role": "resolution_controls_no_trained_method",
        })

    conflict_path = args.output_dir / "conflict_pool.json"
    resolution_path = args.output_dir / "resolution_controls.json"
    write_json(conflict_path, conflict)
    write_json(resolution_path, resolution)
    metadata = {
        "status": "prepared_before_reward_scoring",
        "conflict_prompt_count": len(conflict),
        "responses_per_conflict_prompt": 5,
        "resolution_prompt_count": len(resolution),
        "resolution_response_order": resolution[0]["response_model_names"],
        "selection_hash_rule": "sha256('fair-demo-diagnostic-v1||' + prompt_id)",
        "source_avg_precomputed": str(args.avg_precomputed),
        "source_base_generations": str(args.base_generations),
        "conflict_sha256": sha256(conflict_path),
        "resolution_sha256": sha256(resolution_path),
        "method_ranking_computed": False,
        "spent_sealed_split_touched": False,
    }
    write_json(args.output_dir / "input_manifest.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

