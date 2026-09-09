"""Reconstruct the prescribed SafeRLHF selection from verified preserved rows.

This is an explicit protocol amendment if the nonexistent SafeRLHF 7000/500
manifest cannot be supplied: never substitute the 7000-row UltraFeedback bank.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path

from scripts.experiments.nbpo_repair_20260909.common import (
    digest, file_hash, normalize_prompt, object_hash, read_jsonl, write_json, write_jsonl,
)


def legacy_normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def raw_prompt_lookup(source, preserved):
    """Identify the old subset using every row index and old normalization."""
    expected = {r["row_index"]: r for rows in preserved.values() for r in rows}
    matches = []
    individuals = sorted(source.glob("data/*/train.jsonl"))
    # The pinned dataset card defines default as this ordered concatenation.
    default = [source / "data" / name / "train.jsonl" for name in ("Alpaca-7B", "Alpaca2-7B", "Alpaca3-8B")]
    for paths in [[p] for p in individuals] + [default]:
        raw = {}
        valid, i = True, 0
        for path in paths:
            with path.open() as stream:
                for line in stream:
                    r = json.loads(line)
                    if i in expected:
                        raw[i] = r["prompt"]
                        e = expected[i]
                        valid &= all(legacy_normalize(r[k]) == e[k] for k in ("prompt", "response_0", "response_1"))
                        valid &= all(r[k] == e[k] for k in ("better_response_id", "safer_response_id"))
                    i += 1
        if valid and i == len(expected) and len(raw) == len(expected):
            matches.append((paths, raw))
    if len(matches) != 1:
        raise ValueError(f"Expected one exact historical subset match; got {[str(p) for p, _ in matches]}")
    return matches[0]


def benchmark_prompts(data):
    values = {}
    for filename, keys in (
        ("alpaca_eval.jsonl", ("instruction",)), ("arena_hard.jsonl", ("prompt",)),
        ("ifeval.jsonl", ("prompt",)), ("gsm8k_test.jsonl", ("question",)),
        ("harmbench_behaviors.jsonl", ("Behavior", "ContextString")),
        ("mt_bench_question.jsonl", ("turns",)),
    ):
        texts = []
        for row in read_jsonl(data / filename):
            for key in keys:
                value = row.get(key)
                texts.extend(value if isinstance(value, list) else [value])
        values[filename] = {normalize_prompt(p) for p in texts if isinstance(p, str) and p}
    with (data / "xstest_prompts.csv").open() as stream:
        values["xstest_prompts.csv"] = {normalize_prompt(r["prompt"]) for r in csv.DictReader(stream)}
    if any(not prompts for prompts in values.values()):
        raise ValueError(f"Benchmark prompt extraction failed: {[(k, len(v)) for k, v in values.items()]}")
    return values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--base", default="/work/models/bases/Llama-3.1-8B-Instruct")
    ap.add_argument("--source", type=Path, default=Path("/work/iclr27_table1_v2/preserved/sr_splits"))
    ap.add_argument("--dataset-snapshot", type=Path, default=Path("/work/hf_cache/hub/datasets--PKU-Alignment--PKU-SafeRLHF/snapshots/9421ffafec3fa40a1f1a7d567b4d525079477ecb"))
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    preserved = {s: read_jsonl(args.source / f"saferlhf_{s}.jsonl") for s in ("train", "validation", "test")}
    source_manifest = json.loads((args.source / "split_manifest.json").read_text())
    for split, rows in preserved.items():
        assert file_hash(args.source / f"saferlhf_{split}.jsonl") == source_manifest["splits"][split]["file_sha256"]
    raw_paths, original = raw_prompt_lookup(args.dataset_snapshot, preserved)
    subset = "default" if len(raw_paths) == 3 else raw_paths[0].parent.name
    benchmarks = benchmark_prompts(args.root / "data")
    excluded = set().union(*benchmarks.values())
    # Also match the old normalization, so Unicode/whitespace variants cannot
    # re-enter through the historical grouping convention.
    excluded_legacy = {legacy_normalize(p) for p in excluded}
    source_groups, filters = {}, defaultdict(int)
    for split, rows in preserved.items():
        grouped = {}
        for row in rows:
            text = original[row["row_index"]]
            norm = normalize_prompt(text)
            old_id = row["prompt_sha256"]
            if old_id in grouped:
                continue
            ids = tok.apply_chat_template([{"role": "user", "content": norm}], tokenize=True, add_generation_prompt=True)
            if len(ids) > 1024:
                filters[f"{split}_prompt_over1024"] += 1
                continue
            if norm in excluded or legacy_normalize(norm) in excluded_legacy:
                filters[f"{split}_benchmark_overlap"] += 1
                continue
            grouped[old_id] = {"prompt_id": digest(norm), "prompt": norm, "original_prompt": text,
                "legacy_prompt_sha256": old_id, "source_row_index": row["row_index"],
                "source_split": split, "prompt_token_ids": ids,
                "prompt_token_sha256": object_hash(ids),
                "selection_hash": digest("20260909-repair:" + norm)}
        source_groups[split] = sorted(grouped.values(), key=lambda r: r["selection_hash"])
    legacy_path = Path("/work/iclr27_table1_v2/preserved/smoke_pools_train/pool_manifest.json")
    legacy_ids = set(json.loads(legacy_path.read_text())["prompt_ids"])
    sets = {s: {r["prompt_id"] for r in rows} for s, rows in source_groups.items()}
    assert not (sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"])
    train = source_groups["train"][:2000]
    dev = source_groups["validation"][:500]
    heldout = [r for r in source_groups["test"] if r["legacy_prompt_sha256"] not in legacy_ids][:1000]
    assert len(train) == 2000 and len(dev) == 500
    for label, rows in (("train", train), ("dev", dev), ("test", heldout)):
        for row in rows:
            row["split"] = label
        write_jsonl(args.root / "splits" / f"{label}.jsonl", rows)
    write_jsonl(args.root / "splits" / "eligible_train7000_reconstructed.jsonl", source_groups["train"][:7000])
    write_json(args.root / "splits" / "manifest.json", {
        "dataset": source_manifest["dataset"], "verified_subset": subset,
        "dataset_revision": args.dataset_snapshot.name,
        "raw_sources": [{"path": str(p), "sha256": file_hash(p)} for p in raw_paths],
        "historical_match": "Every row index, normalized prompt and both response strings, and both objective labels match the preserved extraction.",
        "preserved_manifest_sha256": file_hash(args.source / "split_manifest.json"),
        "normalization": "NFC, CRLF/CR to LF, trim boundary whitespace; original text retained",
        "grouping": "retain historical whole-prompt groups and verify NFC cross-split disjointness",
        "selection": "ascending SHA256('20260909-repair:' + normalized_prompt)",
        "protocol_amendment": "SafeRLHF 7000/dev500 manifest unavailable; existing 7000/500/1000 is UltraFeedback. Reconstruct eligible7000 and take first2000 using prescribed hash; dev500 reconstructed from preserved SafeRLHF validation.",
        "counts": {"train": len(train), "dev": len(dev), "test": len(heldout)},
        "eligible_counts": {s: len(r) for s, r in source_groups.items()}, "filters": dict(filters),
        "preference_model_overlap": {"policy_train_in_preference_train": len(train),
            "policy_dev_in_preference_calibration": len(dev), "policy_test_in_preference_train_or_calibration": 0,
            "policy_test_in_preference_test": len(heldout)},
        "fresh_test_claim": False,
        "test_exposure_note": "Held out from preference training/calibration and policy fitting; source test already measured in prior GPM/BT comparison. Not claimed globally untouched or prospective benchmark test.",
        "legacy_policy_overlap": {s: sum(r["legacy_prompt_sha256"] in legacy_ids for r in rows)
                                  for s, rows in (("train", train), ("dev", dev), ("test", heldout))},
        "benchmark_file_hashes": {name: file_hash(args.root / "data" / name) for name in benchmarks},
        "benchmark_unique_prompt_counts": {name: len(prompts) for name, prompts in benchmarks.items()},
        "splits": {s: {"file_sha256": file_hash(args.root / "splits" / f"{s}.jsonl"),
                        "prompt_set_sha256": object_hash(sorted(r["prompt_id"] for r in rows))}
                   for s, rows in (("train", train), ("dev", dev), ("test", heldout))},
    })
    print(json.dumps({"verified_subset": subset, "train": len(train), "dev": len(dev), "test": len(heldout), "filters": dict(filters)}), flush=True)


if __name__ == "__main__":
    main()
