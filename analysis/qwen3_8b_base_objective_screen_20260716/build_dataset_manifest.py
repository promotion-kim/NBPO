#!/usr/bin/env python3
"""Build the outcome-blind standard safety mix for base/objective screening."""

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
    "xstest": "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d",
    "beavertails": "8401fe609d288129cc684a9b3be6a93e41cfe678",
}
HIGH_SEVERITY = {
    "violence,aiding_and_abetting,incitement", "terrorism,organized_crime",
    "child_abuse", "self_harm", "drug_abuse,weapons,banned_substance",
}
TARGETS = {
    "validation": {"pku_saferlhf": 256, "or_bench_hard": 128, "xstest_safe": 64, "xstest_unsafe": 64, "beavertails": 128},
    "fresh_confirmation": {"pku_saferlhf": 128, "or_bench_hard": 64, "xstest_safe": 32, "xstest_unsafe": 32, "beavertails": 64},
}


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def order(row: dict) -> str:
    return sha(f"baseobj-screen-v1|{row['source']}|{norm(row['prompt'])}".encode())


def dedupe(rows: list[dict]) -> list[dict]:
    kept = {}
    for row in rows:
        kept.setdefault(norm(row["prompt"]), row)
    return sorted(kept.values(), key=order)


def load_pku(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if int(item["better_response_id"]) == int(item["safer_response_id"]):
                continue
            rows.append({
                "prompt": str(item["prompt"]), "source": "pku_saferlhf",
                "slice": "dual_preference_conflict", "behavior_label": "safety_tradeoff",
                "category": str(item.get("prompt_source", "unknown")),
                "source_revision": REVISIONS["pku_saferlhf"],
                "selection_basis": "human better_response_id differs from safer_response_id",
            })
    return dedupe(rows)


def load_or(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [{
            "prompt": str(item["prompt"]), "source": "or_bench_hard",
            "slice": "safe_but_overrefusal_prone", "behavior_label": "should_answer",
            "category": str(item.get("category", "unknown")),
            "source_revision": REVISIONS["or_bench_hard"],
            "selection_basis": "OR-Bench hard split membership",
        } for item in csv.DictReader(handle)]
    return dedupe(rows)


def load_xstest(path: Path) -> dict[str, list[dict]]:
    groups = {"xstest_safe": [], "xstest_unsafe": []}
    with path.open(encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            safe = str(item["label"]).strip().lower() == "safe"
            key = "xstest_safe" if safe else "xstest_unsafe"
            groups[key].append({
                "prompt": str(item["prompt"]), "source": "xstest",
                "slice": key, "behavior_label": "should_answer" if safe else "should_refuse",
                "category": str(item["type"]), "source_revision": REVISIONS["xstest"],
                "selection_basis": f"official XSTest label={item['label']}",
            })
    return {key: dedupe(rows) for key, rows in groups.items()}


def load_beaver(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if bool(item.get("is_safe")):
                continue
            active = sorted(name for name, value in (item.get("category") or {}).items() if value)
            if not set(active) & HIGH_SEVERITY:
                continue
            rows.append({
                "prompt": str(item["prompt"]), "source": "beavertails",
                "slice": "higher_severity_harmful", "behavior_label": "should_refuse",
                "category": ";".join(active), "source_revision": REVISIONS["beavertails"],
                "selection_basis": "is_safe=false and category in locked higher-severity set",
            })
    return dedupe(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "pku_saferlhf": args.source_root / "pku_saferlhf/data/Alpaca3-8B/test.jsonl",
        "or_bench_hard": args.source_root / "or_bench/or-bench-hard-1k.csv",
        "xstest": args.source_root / "xstest/xstest_prompts.csv",
        "beavertails": args.source_root / "beavertails/round0/30k/test.jsonl.gz",
    }
    xstest = load_xstest(paths["xstest"])
    pools = {
        "pku_saferlhf": load_pku(paths["pku_saferlhf"]),
        "or_bench_hard": load_or(paths["or_bench_hard"]),
        "xstest_safe": xstest["xstest_safe"], "xstest_unsafe": xstest["xstest_unsafe"],
        "beavertails": load_beaver(paths["beavertails"]),
    }
    seen = set()
    for key in ["pku_saferlhf", "or_bench_hard", "xstest_safe", "xstest_unsafe", "beavertails"]:
        unique = []
        for row in pools[key]:
            marker = norm(row["prompt"])
            if marker not in seen:
                unique.append(row); seen.add(marker)
        pools[key] = unique
    offsets = {key: 0 for key in pools}
    outputs = {}
    for split, target in TARGETS.items():
        rows = []
        for key, count in target.items():
            start = offsets[key]
            selected = pools[key][start:start + count]
            if len(selected) != count:
                raise RuntimeError(f"{key}/{split}: {len(selected)} < {count}")
            offsets[key] += count
            for row in selected:
                rows.append({**row, "split": split, "prompt_id": f"{key}_{sha(norm(row['prompt']).encode())[:20]}"})
        rows.sort(key=lambda row: row["prompt_id"])
        path = args.output_root / "dataset_manifest" / f"{split}.jsonl"
        write_jsonl(path, rows)
        outputs[split] = {"path": str(path), "sha256": sha(path.read_bytes()), "count": len(rows), "counts": target}
    validation = {norm(json.loads(line)["prompt"]) for line in (args.output_root / "dataset_manifest/validation.jsonl").read_text().splitlines() if line}
    fresh = {norm(json.loads(line)["prompt"]) for line in (args.output_root / "dataset_manifest/fresh_confirmation.jsonl").read_text().splitlines() if line}
    if validation & fresh:
        raise RuntimeError("validation/fresh overlap")
    payload = {
        "construction": "40% PKU dual-preference conflict, 20% OR-Bench Hard, 20% XSTest (half safe, half unsafe), 20% higher-severity BeaverTails",
        "source_files": {key: {"path": str(path), "sha256": sha(path.read_bytes()), "revision": REVISIONS[key]} for key, path in paths.items()},
        "higher_severity_categories": sorted(HIGH_SEVERITY),
        "available_after_filter_and_cross_source_dedup": {key: len(rows) for key, rows in pools.items()},
        "outputs": outputs, "ordering": "SHA256(baseobj-screen-v1|source|normalized_prompt)",
        "construction_script_sha256": sha(Path(__file__).read_bytes()),
        "spent_sealed_split_read": False,
    }
    path = args.output_root / "dataset_manifest/manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
