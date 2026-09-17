"""Rebuild a UF-4 dataset from pairs already on disk, with correct provenance.

The pairs and the solver artifacts are fine; only dataset_provenance.json carried
a wrong tokenizer_hash, so re-solving would be pure waste. This recomputes the
real tokenizer content hashes the trainer checks against and materializes a new
dataset directory. The original directory and its provenance are left in place
as the record of what failed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/work/uf4_20260910")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--model", default="/work/models/bases/Llama-3.1-8B-Instruct")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from mnpo_scripts.precompute_provenance import tokenizer_content_hashes

    target_dir = ROOT / "targets" / args.targets
    provenance_path = target_dir / "dataset_provenance.json"
    provenance = json.loads(provenance_path.read_text())

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    pad_fallback = tokenizer.pad_token_id is None
    if pad_fallback:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    hashes = tokenizer_content_hashes(tokenizer)

    before = {k: provenance.get(k) for k in ("tokenizer_hash", "chat_template_hash")}
    if provenance.get("chat_template_hash") != hashes["chat_template_hash"]:
        raise ValueError("Chat template hash disagrees; the pool was not generated with this tokenizer")
    provenance.update(hashes)
    provenance["training_pad_token_fallback_to_eos"] = pad_fallback
    provenance["provenance_repair"] = {
        "field": "tokenizer_hash",
        "was": before["tokenizer_hash"],
        "now": hashes["tokenizer_hash"],
        "cause": ("solve_uf4_targets.py filled tokenizer_hash with the chat-template hash to "
                  "satisfy prepare_nbpo_dataset's required-key check. The trainer recomputes "
                  "both hashes and refused the dataset, which is the guard working."),
        "pairs_unchanged": True, "solver_artifacts_unchanged": True}
    repaired = target_dir / "dataset_provenance.repaired.json"
    repaired.write_text(json.dumps(provenance, indent=2) + "\n")

    dataset_out = ROOT / "datasets" / args.dataset_name
    env = dict(os.environ, HF_DATASETS_CACHE=str(target_dir / "arrow_cache_repaired"),
               PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(
        [sys.executable, "-m", "mnpo_scripts.prepare_nbpo_dataset",
         "--train", str(target_dir / "pairs/train.jsonl"),
         "--dev", str(target_dir / "pairs/dev.jsonl"),
         "--output", str(dataset_out), "--provenance", str(repaired)],
        env=env, capture_output=True, text=True)
    if completed.returncode:
        (target_dir / "rematerialization_failure.json").write_text(json.dumps(
            {"returncode": completed.returncode, "stdout": completed.stdout[-4000:],
             "stderr": completed.stderr[-4000:]}, indent=2) + "\n")
        raise RuntimeError("Re-materialization failed; diagnostic preserved")
    manifest = file_hash(dataset_out / "precompute_manifest.json")

    complete_path = target_dir / "complete.json"
    complete = json.loads(complete_path.read_text())
    complete["superseded_dataset_path"] = complete.get("dataset_path")
    complete["superseded_dataset_manifest_sha256"] = complete.get("dataset_manifest_sha256")
    complete["dataset_path"] = str(dataset_out)
    complete["dataset_manifest_sha256"] = manifest
    complete["dataset_provenance_repaired"] = str(repaired)
    complete_path.write_text(json.dumps(complete, indent=2) + "\n")
    print(json.dumps({"targets": args.targets, "dataset": str(dataset_out),
                      "manifest_sha256": manifest,
                      "tokenizer_hash": hashes["tokenizer_hash"][:16]}), flush=True)


if __name__ == "__main__":
    main()
