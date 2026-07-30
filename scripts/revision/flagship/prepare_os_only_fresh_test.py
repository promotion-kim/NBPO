#!/usr/bin/env python3
"""Lock one prompt-disjoint fresh split for the OS-only confirmation before ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer


SOURCE = {"dataset": "HuggingFaceH4/ultrachat_200k",
          "revision": "8049631c405ae6576f93f445c6b8166f76f5505a", "split": "test_sft"}
SALT = "ronpo-os-only-fresh-confirmation-20260716-v1||"


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
        if str(message.get("role") or message.get("from") or "").lower() in {"user", "human"}:
            return str(message.get("content") or message.get("value") or "")
    return ""


def read_prompt(row: dict) -> str:
    for key in ("prompt", "instruction", "question", "input"):
        if key in row and str(row[key]).strip():
            return normalize(raw_prompt(row[key]))
    return normalize(first_user(row))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-lock", type=Path, required=True)
    parser.add_argument("--avg-precomputed", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--source-arrow", type=Path,
                        help="Verified local test_sft Arrow file; avoids any network/cache rebuild.")
    parser.add_argument("--exclude-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-count", type=int, default=128)
    args = parser.parse_args()
    lock = json.loads(args.metric_lock.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_BEFORE_NEW_OS_TRAINING_AND_RANKING":
        raise RuntimeError("OS-only metric is not locked")
    if args.prompt_count != 128:
        raise RuntimeError("fresh prompt count is frozen at 128 for the time-bounded two-judge panel")
    excluded = set()
    frozen_records = 0
    dataset = load_from_disk(str(args.avg_precomputed))
    for split in dataset:
        for row in dataset[split]:
            prompt = normalize(raw_prompt(row["prompt"]))
            if prompt:
                excluded.add(hashlib.sha256(prompt.encode()).hexdigest()); frozen_records += 1
    exclusion_rows = []
    for path in args.exclude_jsonl:
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                prompt = read_prompt(json.loads(line))
                if prompt:
                    excluded.add(hashlib.sha256(prompt.encode()).hexdigest()); count += 1
        exclusion_rows.append({"path": str(path), "sha256": sha256(path), "records": count})
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), local_files_only=True)
    if args.source_arrow:
        if not args.source_arrow.is_file():
            raise RuntimeError("locked UltraChat test_sft Arrow file is missing")
        source = Dataset.from_file(str(args.source_arrow))
        source_artifact = {"path": str(args.source_arrow), "sha256": sha256(args.source_arrow)}
    else:
        source = load_dataset(SOURCE["dataset"], revision=SOURCE["revision"], split=SOURCE["split"],
                              cache_dir=str(args.cache_dir))
        source_artifact = {"cache_dir": str(args.cache_dir)}
    unique = {}
    counts = {"source_records": len(source), "empty": 0, "token_filter": 0,
              "excluded_overlap": 0, "duplicate": 0}
    for row in source:
        prompt = normalize(first_user(row))
        if not prompt:
            counts["empty"] += 1; continue
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        if digest in excluded:
            counts["excluded_overlap"] += 1; continue
        if digest in unique:
            counts["duplicate"] += 1; continue
        tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if not 16 <= tokens <= 1800:
            counts["token_filter"] += 1; continue
        unique[digest] = {"prompt": prompt, "prompt_sha256": digest, "token_count": tokens}
    ordered = sorted(unique.values(), key=lambda row: hashlib.sha256((SALT + row["prompt"]).encode()).hexdigest())
    if len(ordered) < args.prompt_count:
        raise RuntimeError("not enough disjoint fresh prompts")
    selected = ordered[:args.prompt_count]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = args.output_dir / "fresh_test_prompts.jsonl"
    with prompts.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "status": "FRESH_TEST_PROMPTS_LOCKED_UNOPENED_BEFORE_OS_RANKING",
        "locked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": SOURCE, "source_artifact": source_artifact,
        "selection_salt_sha256": hashlib.sha256(SALT.encode()).hexdigest(),
        "prompt_count": len(selected), "prompt_file": str(prompts),
        "prompt_file_sha256": sha256(prompts), "metric_lock_sha256": sha256(args.metric_lock),
        "exclusion": {"frozen_pool_records": frozen_records,
                      "unique_excluded_prompt_hashes": len(excluded),
                      "additional_files": exclusion_rows, "spent_path_read": False},
        "filter_counts": counts, "token_filter": [16, 1800],
        "fresh_test_opened": False, "measured_once": False,
        "spent_sealed_split_touched": False,
    }
    manifest_path = args.output_dir / "fresh_test_manifest.json"
    atomic_json(manifest_path, manifest)
    (args.output_dir / "fresh_test_manifest.sha256").write_text(
        f"{sha256(manifest_path)}  fresh_test_manifest.json\n", encoding="utf-8")
    atomic_json(args.output_dir / "fresh_design_lock.json", {
        "status": "LOCKED_BEFORE_OS_RANKING", "planned_prompt_count": 128,
        "manifest_sha256": sha256(manifest_path), "metric_lock_sha256": sha256(args.metric_lock),
        "local_rm_models": list(lock["objectives"].values()),
        "independent_panel": lock["fresh_confirmation"]["independent_panel"],
        "decode": lock["selection"]["decode"], "one_shot": True,
        "spent_sealed_split_touched": False,
    })
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
