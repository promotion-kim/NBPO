#!/usr/bin/env python3
"""Pre-register a fresh prompt split without decoding or scoring it."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


SALT = "qwen3-8b-stronger-confirm-v1||"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pool", type=Path, required=True)
    parser.add_argument("--spent-sealed", type=Path, required=True)
    parser.add_argument("--power-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=604)
    args = parser.parse_args()
    power = json.loads(args.power_summary.read_text())
    if power.get("status") != "completed" or power.get("requires_fresh_sealed_preregistration") is not True:
        raise RuntimeError("the frozen power rule does not authorize fresh-split preregistration")
    source_rows = load_jsonl(args.source_pool)
    by_prompt = {}
    for row in source_rows:
        prompt = str(row["prompt"])
        by_prompt.setdefault(prompt, row)
    ordered_prompts = sorted(by_prompt)
    validation_prompts = set(ordered_prompts[:128])
    spent_prompts = {str(row["prompt"]) for row in load_jsonl(args.spent_sealed)}
    candidates = [prompt for prompt in ordered_prompts[128:] if prompt not in spent_prompts]
    ranked = sorted(candidates, key=lambda prompt: hashlib.sha256((SALT + prompt).encode()).hexdigest())
    selected = ranked[:args.count]
    if len(selected) != args.count:
        raise RuntimeError(f"only {len(selected)} fresh prompts available")
    if validation_prompts.intersection(selected) or spent_prompts.intersection(selected):
        raise RuntimeError("fresh prompt overlap detected")
    output_rows = []
    for prompt in selected:
        source = by_prompt[prompt]
        output_rows.append({"prompt_id": source.get("prompt_id"), "prompt": prompt})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = args.output_dir / "fresh_sealed_prompts.jsonl"
    with prompts_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    prompt_ids = "\n".join(str(row.get("prompt_id")) for row in output_rows)
    manifest = {
        "status": "preregistered_unopened",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "future single-shot confirmation of the stronger-training validation signal",
        "selection_rule": f"Exclude the first 128 lexicographically sorted validation prompts and every spent sealed prompt, then sort the remaining prompts by sha256('{SALT}' + prompt) and take the first {args.count}.",
        "selection_rule_fixed_before_fresh_prompt_generation": True,
        "source_pool": str(args.source_pool),
        "source_pool_sha256": sha256(args.source_pool),
        "spent_sealed": str(args.spent_sealed),
        "spent_sealed_sha256": sha256(args.spent_sealed),
        "power_summary": str(args.power_summary),
        "power_summary_sha256": sha256(args.power_summary),
        "counts": {"source_unique_prompts": len(ordered_prompts),
                   "selection_validation_prompts": len(validation_prompts),
                   "spent_sealed_prompts": len(spent_prompts),
                   "fresh_candidates": len(candidates), "fresh_sealed": len(output_rows)},
        "overlaps": {"fresh_vs_selection_validation": 0, "fresh_vs_spent_sealed": 0},
        "fresh_prompt_file": str(prompts_path),
        "fresh_prompt_file_sha256": sha256(prompts_path),
        "fresh_prompt_id_sha256": hashlib.sha256(prompt_ids.encode()).hexdigest(),
        "model_generations": 0,
        "reward_scores": 0,
        "opened_for_evaluation": False,
        "selection_or_tuning_allowed": False,
    }
    (args.output_dir / "fresh_sealed_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
