#!/usr/bin/env python3
"""Freeze the exact Table-4 baseline revisions and compatible 647 reuse artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path


TABLE4_NAMES = [
    "base", "ronpo_full_expect", "ronpo_k_only", "ipo", "simpo", "sppo_avg", "inpo_avg",
    "ht_mnpo_helpfulness", "ht_mnpo_safety", "ht_mnpo_conciseness",
]
REUSE = {
    "base": "base",
    "ipo": "frozen_ipo",
    "ht_mnpo_safety": "frozen_ht_mnpo_safety",
    "ht_mnpo_conciseness": "frozen_ht_mnpo_conciseness",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-tsv", type=Path, required=True)
    parser.add_argument("--prior-fixed647", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.models_tsv.open(encoding="utf-8") as handle:
        source = {row["name"]: row for row in csv.DictReader(handle, delimiter="\t")}
    if any(name not in source for name in TABLE4_NAMES):
        raise RuntimeError("models.tsv is missing a Table-4 model")
    rows = []
    for name in TABLE4_NAMES:
        row = source[name]
        if name == "base":
            snapshot = args.hf_cache / "models--Qwen--Qwen3-8B" / "snapshots" / row["revision"]
        else:
            repo_dir = "models--" + row["model"].replace("/", "--")
            snapshot = args.hf_cache / repo_dir / "snapshots" / row["revision"]
        if not snapshot.is_dir():
            raise RuntimeError(f"pinned snapshot unavailable: {name}: {snapshot}")
        item = {"name": name, "method": row["method"], "repo": row["model"],
                "revision": row["revision"], "snapshot": str(snapshot), "stage": 1 if name != "base" else 0,
                "training_frozen": True, "decode_required": name not in REUSE}
        if name in REUSE:
            generation = args.prior_fixed647 / "generations" / REUSE[name] / "output_42.json"
            metadata = generation.parent / "decode_metadata.json"
            if not generation.is_file() or not metadata.is_file():
                raise RuntimeError(f"compatible reuse generation missing: {name}")
            meta = json.loads(metadata.read_text(encoding="utf-8"))
            if int(meta.get("num_prompts", -1)) != 647 or int(meta.get("max_new_tokens", -1)) != 2048:
                raise RuntimeError(f"reuse decode differs from common protocol: {name}")
            if name != "base" and not str(meta.get("model", "")).endswith("/" + row["revision"]):
                raise RuntimeError(f"reuse generation revision mismatch: {name}")
            item.update({"reuse_generation": str(generation), "reuse_generation_sha256": sha256(generation),
                         "reuse_decode_metadata": str(metadata), "reuse_decode_metadata_sha256": sha256(metadata)})
        rows.append(item)
    ledger = {
        "status": "LOCKED_EXACT_TABLE4_BASELINES_NO_RETRAINING",
        "locked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "models_tsv": str(args.models_tsv), "models_tsv_sha256": sha256(args.models_tsv),
        "table4_model_names": TABLE4_NAMES, "rows": rows,
        "dpo_handling": "not in the existing Table-4 row set; its prior fixed-647 generation failed max-repeat-run and is not rescued",
        "baseline_training_launched": False, "spent_sealed_split_touched": False,
    }
    atomic_json(args.output, ledger)
    print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
