#!/usr/bin/env python3
"""Prepare the preregistered prompt-disjoint UltraChat test after validation selection is locked."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime
from pathlib import Path

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def raw_prompt(value: object) -> str:
    text = str(value)
    match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, flags=re.S)
    return match.group(1).strip() if match else text.strip()


def first_user(row: dict) -> str:
    messages = row.get("messages") or row.get("conversation") or row.get("conversations") or []
    for message in messages:
        role = str(message.get("role") or message.get("from") or "").lower()
        if role in {"user", "human"}:
            return str(message.get("content") or message.get("value") or "")
    return ""


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--evaluator-lock", type=Path, required=True)
    parser.add_argument("--avg-precomputed", type=Path, required=True)
    parser.add_argument("--validation-prompts", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    evaluator = json.loads(args.evaluator_lock.read_text(encoding="utf-8"))
    if selection.get("status") != "VALIDATION_SELECTION_LOCKED_BEFORE_FRESH_TEST":
        raise RuntimeError("validation selection is not locked")
    if selection.get("fresh_test_opened") is not False:
        raise RuntimeError("selection lock does not certify an unopened fresh test")
    source = evaluator["fresh_test_source"]
    if source != {"dataset": "HuggingFaceH4/ultrachat_200k",
                  "revision": "8049631c405ae6576f93f445c6b8166f76f5505a", "split": "test_sft"}:
        raise RuntimeError("fresh source differs from evaluator lock")
    planned = int(evaluator["power"]["planned_fresh_test_prompts"])

    excluded_hashes = set()
    frozen = load_from_disk(str(args.avg_precomputed))
    source_pool_records = 0
    for split in frozen:
        for row in frozen[split]:
            value = normalize(raw_prompt(row["prompt"]))
            if value:
                excluded_hashes.add(hashlib.sha256(value.encode()).hexdigest())
                source_pool_records += 1
    validation_records = 0
    with args.validation_prompts.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = normalize(str(json.loads(line)["prompt"]))
            excluded_hashes.add(hashlib.sha256(value.encode()).hexdigest())
            validation_records += 1

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), local_files_only=True)
    dataset = load_dataset(source["dataset"], revision=source["revision"], split=source["split"],
                           cache_dir=str(args.cache_dir))
    unique = {}
    counts = {"source_records": len(dataset), "empty": 0, "token_filter": 0,
              "frozen_pool_overlap": 0, "duplicate": 0}
    for row in dataset:
        prompt = normalize(first_user(row))
        if not prompt:
            counts["empty"] += 1; continue
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        if digest in excluded_hashes:
            counts["frozen_pool_overlap"] += 1; continue
        if digest in unique:
            counts["duplicate"] += 1; continue
        tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if not 16 <= tokens <= 1800:
            counts["token_filter"] += 1; continue
        unique[digest] = {"prompt": prompt, "prompt_sha256": digest, "token_count": tokens}
    ordered = sorted(unique.values(), key=lambda row: hashlib.sha256(
        ("fair-demo-fresh-test-v1||" + row["prompt"]).encode()
    ).hexdigest())
    actual = min(planned, len(ordered))
    if actual < 2:
        raise RuntimeError(f"only {actual} eligible fresh prompts")
    selected = ordered[:actual]
    paired_sd = float(evaluator["power"]["paired_sd"])
    z_alpha, z_power = 1.959963984540054, 0.8416212335729143
    minimum_detectable_effect = (z_alpha + z_power) * paired_sd / math.sqrt(actual)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = args.output_dir / "fresh_test_prompts.jsonl"
    temporary = prompts_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(prompts_path)
    manifest = {
        "status": "FRESH_TEST_PROMPTS_LOCKED_UNOPENED", "locked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source, "selection_lock_sha256": sha256(args.selection_lock),
        "evaluator_lock_sha256": sha256(args.evaluator_lock), "prompt_count": len(selected),
        "prompt_file_sha256": sha256(prompts_path), "selection_hash_rule": "sha256('fair-demo-fresh-test-v1||' + normalized_prompt)",
        "normalization": "NFKC plus whitespace collapse", "token_filter": [16, 1800],
        "exclusion": {
            "frozen_avg_precomputed_all_splits_records": source_pool_records,
            "validation_records": validation_records,
            "unique_excluded_prompt_hashes": len(excluded_hashes),
            "spent_path_read": False,
            "reason": "All splits of the frozen flagship source pool were excluded, covering its training, validation, and held-out source prompts without reading the forbidden spent-sealed path."
        },
        "filter_counts": counts, "fresh_test_opened": False,
        "power": {
            "planned_prompt_count": planned, "eligible_prompt_count": len(ordered),
            "actual_prompt_count": actual, "capped_by_source_pool": actual < planned,
            "minimum_detectable_absolute_effect": minimum_detectable_effect,
            "target_absolute_effect": evaluator["power"]["target_absolute_effect"],
            "alpha_two_sided": evaluator["power"]["alpha_two_sided"],
            "power": evaluator["power"]["power"],
        },
        "spent_sealed_split_touched": False,
    }
    atomic_json(args.output_dir / "fresh_test_manifest.json", manifest)
    (args.output_dir / "fresh_test_manifest.sha256").write_text(
        f"{sha256(args.output_dir / 'fresh_test_manifest.json')}  fresh_test_manifest.json\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
