#!/usr/bin/env python3
"""Prepare the locked Stage-4 seed-43 evaluation from existing generations.

No decoding occurs here.  The script verifies the completed seed-43 stability
artifacts, stages them under the seed-42 canonical method IDs, builds the shared
score pool, and deterministically shards it for Beaver scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


CANONICAL_TO_SEED43 = {
    "ronpo_os_stage4": "ronpo_os",
    "ronpo_topmass_stage4": "ronpo_topmass",
    "inpo_avg_stage4": "inpo_avg",
    "sppo_avg_stage4": "sppo_avg",
    "simpo_stage4": "simpo",
    "ipo_stage4": "ipo",
    "dpo_stage4": "dpo",
    "ht_mnpo_harmless_stage4": "ht_mnpo_harmless",
    "ht_mnpo_helpfulness_stage4": "ht_mnpo_helpfulness",
}
MODELS = ["base", *CANONICAL_TO_SEED43]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_link(destination: Path, source: Path) -> None:
    if not source.is_file():
        raise RuntimeError(f"missing source artifact: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() == source.resolve():
        return
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"refusing to replace existing staged artifact: {destination}")
    os.symlink(source.resolve(), destination)


def verify_generation(path: Path, expected: int) -> None:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != expected:
        raise RuntimeError(f"{path}: expected {expected} records, found {len(rows)}")
    prompt_ids = [str(row.get("prompt_id", row.get("prompt"))) for row in rows]
    if len(set(prompt_ids)) != expected:
        raise RuntimeError(f"{path}: duplicate prompt IDs")
    if any(not str(row.get("generated_text_raw", "")).strip() for row in rows):
        raise RuntimeError(f"{path}: empty response")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--seed42-root", type=Path, required=True)
    parser.add_argument("--seed43-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, default=1000)
    parser.add_argument("--num-shards", type=int, default=6)
    args = parser.parse_args()

    gate43 = args.seed43_root / "stage4_stability_p8_locked_panel"
    generation_sources = {
        "base": args.seed42_root / "stage4_eval/generations/base/output_42.json",
        **{
            canonical: gate43 / f"generations/{seed43}/output_43.json"
            for canonical, seed43 in CANONICAL_TO_SEED43.items()
        },
    }
    gate_sources = {
        "base": args.seed42_root / "stage4_eval/gates/base.json",
        **{
            canonical: gate43 / f"gates/{seed43}.json"
            for canonical, seed43 in CANONICAL_TO_SEED43.items()
        },
    }

    manifest = {
        "status": "complete",
        "purpose": "Stage-4 seed-43 evaluation reusing existing 1000-prompt gate generations; no decode",
        "canonical_models": MODELS,
        "records": args.expected_records,
        "decode_seed": 42,
        "training_seed": 43,
        "models": {},
        "spent_sealed_split_touched": False,
    }
    for model in MODELS:
        generation, gate = generation_sources[model], gate_sources[model]
        verify_generation(generation, args.expected_records)
        gate_payload = json.loads(gate.read_text(encoding="utf-8"))
        if not gate_payload.get("passed"):
            raise RuntimeError(f"fail-closed gate did not pass for {model}: {gate}")
        staged_generation = args.output / f"generations/{model}/output_42.json"
        staged_gate = args.output / f"gates/{model}.json"
        stage_link(staged_generation, generation)
        stage_link(staged_gate, gate)
        manifest["models"][model] = {
            "generation_source": str(generation),
            "generation_sha256": sha256(generation),
            "gate_source": str(gate),
            "gate_sha256": sha256(gate),
            "gate_passed": True,
        }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    merge = args.project / "analysis/p2_8b_hh_multiobjective_20260717/merge_eval_pool.py"
    subprocess.run(
        [
            sys.executable,
            str(merge),
            "--generation-root", str(args.output / "generations"),
            "--models", ",".join(MODELS),
            "--seed", "42",
            "--expected-records", str(args.expected_records),
            "--gate-root", str(args.output / "gates"),
            "--output", str(args.output / "response_pool.jsonl"),
            "--audit", str(args.output / "pool_audit.json"),
        ],
        check=True,
    )
    shard = args.project / "analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py"
    subprocess.run(
        [
            sys.executable,
            str(shard),
            "split",
            "--input", str(args.output / "response_pool.jsonl"),
            "--output-dir", str(args.output / "shards"),
            "--num-shards", str(args.num_shards),
            "--expected-records", str(args.expected_records),
        ],
        check=True,
    )
    print(json.dumps({"output": str(args.output), "models": MODELS, "records": args.expected_records}, indent=2))


if __name__ == "__main__":
    main()
