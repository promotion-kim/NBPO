#!/usr/bin/env python3
"""Build an outcome-blind, standard-dataset helpful/harmless screen manifest."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path


REVISIONS = {
    "pku_saferlhf": "9421ffafec3fa40a1f1a7d567b4d525079477ecb",
    "or_bench_hard": "fd6ee135ee63ff6c4f3ff72c0e39627bf0a7f314",
    "beavertails": "8401fe609d288129cc684a9b3be6a93e41cfe678",
}
HIGH_SEVERITY = {
    "violence,aiding_and_abetting,incitement",
    "terrorism,organized_crime",
    "child_abuse",
    "self_harm",
    "drug_abuse,weapons,banned_substance",
}
TARGETS = {
    "validation": {"pku_saferlhf": 384, "or_bench_hard": 192, "beavertails": 192},
    "fresh_confirmation": {"pku_saferlhf": 192, "or_bench_hard": 96, "beavertails": 96},
}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def normalized(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt).strip().lower()


def order_key(source: str, prompt: str) -> str:
    return sha_bytes(f"hh-screen-v1|{source}|{normalized(prompt)}".encode())


def dedupe(rows: list[dict]) -> list[dict]:
    by_prompt = {}
    for row in rows:
        key = normalized(row["prompt"])
        by_prompt.setdefault(key, row)
    return sorted(by_prompt.values(), key=lambda row: order_key(row["source"], row["prompt"]))


def load_pku(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["better_response_id"]) == int(row["safer_response_id"]):
                continue
            prompt = str(row["prompt"])
            rows.append({
                "prompt": prompt,
                "source": "pku_saferlhf",
                "slice": "dual_preference_conflict",
                "category": str(row.get("prompt_source", "unknown")),
                "source_revision": REVISIONS["pku_saferlhf"],
                "selection_basis": "human better_response_id differs from safer_response_id",
            })
    return dedupe(rows)


def load_or(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [
            {
                "prompt": str(row["prompt"]),
                "source": "or_bench_hard",
                "slice": "safe_but_overrefusal_prone",
                "category": str(row.get("category", "unknown")),
                "source_revision": REVISIONS["or_bench_hard"],
                "selection_basis": "OR-Bench hard split membership",
            }
            for row in csv.DictReader(handle)
        ]
    return dedupe(rows)


def load_beaver(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if bool(row.get("is_safe")):
                continue
            active = sorted(name for name, value in (row.get("category") or {}).items() if value)
            if not set(active) & HIGH_SEVERITY:
                continue
            rows.append({
                "prompt": str(row["prompt"]),
                "source": "beavertails",
                "slice": "human_labeled_higher_severity_harmful",
                "category": ";".join(active),
                "source_revision": REVISIONS["beavertails"],
                "selection_basis": "is_safe=false and category in preregistered higher-severity set",
            })
    return dedupe(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source_paths = {
        "pku_saferlhf": args.source_root / "pku_saferlhf/data/Alpaca3-8B/test.jsonl",
        "or_bench_hard": args.source_root / "or_bench/or-bench-hard-1k.csv",
        "beavertails": args.source_root / "beavertails/round0/30k/test.jsonl.gz",
    }
    pools = {
        "pku_saferlhf": load_pku(source_paths["pku_saferlhf"]),
        "or_bench_hard": load_or(source_paths["or_bench_hard"]),
        "beavertails": load_beaver(source_paths["beavertails"]),
    }
    # Cross-source deduplication uses standard-source priority: PKU core, then OR, then BeaverTails.
    seen = set()
    for source in ["pku_saferlhf", "or_bench_hard", "beavertails"]:
        kept = []
        for row in pools[source]:
            key = normalized(row["prompt"])
            if key not in seen:
                kept.append(row)
                seen.add(key)
        pools[source] = kept

    offsets = {source: 0 for source in pools}
    outputs = {}
    for split in ["validation", "fresh_confirmation"]:
        rows = []
        for source, count in TARGETS[split].items():
            start = offsets[source]
            selected = pools[source][start:start + count]
            if len(selected) != count:
                raise RuntimeError(f"not enough {source} rows for {split}: {len(selected)} < {count}")
            offsets[source] += count
            for row in selected:
                prompt = row["prompt"]
                rows.append({
                    **row,
                    "split": split,
                    "prompt_id": f"{source}_{sha_bytes(normalized(prompt).encode())[:20]}",
                })
        rows.sort(key=lambda row: row["prompt_id"])
        path = args.output_root / "dataset_manifest" / f"{split}.jsonl"
        write_jsonl(path, rows)
        outputs[split] = {"path": str(path), "sha256": file_sha(path), "count": len(rows), "counts_by_source": TARGETS[split]}

    validation_prompts = {normalized(row["prompt"]) for row in read_jsonl(args.output_root / "dataset_manifest/validation.jsonl")}
    fresh_prompts = {normalized(row["prompt"]) for row in read_jsonl(args.output_root / "dataset_manifest/fresh_confirmation.jsonl")}
    if validation_prompts & fresh_prompts:
        raise RuntimeError("validation and fresh-confirmation prompts overlap")
    payload = {
        "construction": "50% PKU-SafeRLHF dual-preference conflicts, 25% OR-Bench hard safe prompts, 25% higher-severity BeaverTails harmful prompts",
        "source_files": {source: {"path": str(path), "sha256": file_sha(path), "revision": REVISIONS[source]} for source, path in source_paths.items()},
        "higher_severity_categories": sorted(HIGH_SEVERITY),
        "available_after_filters": {source: len(rows) for source, rows in pools.items()},
        "outputs": outputs,
        "construction_script_sha256": file_sha(Path(__file__)),
        "selection_seed_definition": "lexicographic SHA256('hh-screen-v1|source|normalized_prompt')",
        "spent_sealed_split_read": False,
    }
    path = args.output_root / "dataset_manifest/manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (args.output_root / "dataset_manifest/manifest.json.sha256").write_text(f"{file_sha(path)}  manifest.json\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
