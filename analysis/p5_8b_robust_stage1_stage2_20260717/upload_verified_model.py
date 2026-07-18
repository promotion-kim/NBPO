#!/usr/bin/env python3
"""Upload one gate-passing P5 final model without persisting credentials.

The caller supplies HF_TOKEN through process environment.  This script refuses
to upload a model without a passed corrected stability gate and a measured
JSON summary, then records the verified public revision in hf_uploads.jsonl.
It intentionally never deletes local data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--method", required=True)
    args = parser.parse_args()
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN must be supplied ephemerally by the parent process")
    model_dir = args.model_dir.resolve()
    experiment = args.experiment.resolve()
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        raise RuntimeError(f"not a final model directory: {model_dir}")
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    if not gate.get("passed", False):
        raise RuntimeError("refusing upload: corrected stability gate did not pass")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    row = next((entry for entry in summary.get("ranking", []) if entry.get("model") == args.model_key), None)
    if row is None:
        raise RuntimeError("refusing upload: model has no measured row in the supplied summary")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(args.repo_id, repo_type="model", private=False, exist_ok=True)
    metadata = {
        "base_model": args.base_model,
        "method": args.method,
        "model_key": args.model_key,
        "source_model_dir": str(model_dir),
        "stability_gate": str(args.gate.resolve()),
        "evaluation_summary": str(args.summary.resolve()),
        "measured_row": row,
        "limitations": "Measured on the fixed 49-prompt P4 validation panel only; this is not a fresh confirmation.",
        "uploaded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "spent_sealed_split_touched": False,
    }
    staging = experiment / "uploads" / args.model_key
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "measured_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    card = "\n".join([
        "---", "license: apache-2.0", "language:", "- en", "tags:", "- preference-optimization", "- research", "---", "",
        f"# {args.method}", "",
        f"Base model: `{args.base_model}`.", "",
        "This is a research checkpoint uploaded only after the corrected generation-stability gate passed.",
        "Its measured result is recorded in `measured_provenance.json`; the evaluation uses a fixed 49-prompt validation panel and is not a fresh confirmation.",
        "", "## Reproducibility", "", "The exact training configuration, stability gate JSON, and measured summary paths are preserved in `measured_provenance.json`.", "",
    ])
    (staging / "README.md").write_text(card, encoding="utf-8")
    # Trainer-generated READMEs can contain local dataset/model paths in their
    # YAML metadata, which the Hub correctly rejects.  Preserve that local
    # artifact but exclude it from the folder upload; the validated research
    # card above is uploaded explicitly in the next request.
    commit = api.upload_folder(
        repo_id=args.repo_id, repo_type="model", folder_path=str(model_dir),
        commit_message=f"Upload verified {args.model_key} final checkpoint",
        ignore_patterns=["README.md", "optimizer.pt", "scheduler.pt", "rng_state*.pth", "checkpoint-*", "wandb/**"],
    )
    api.upload_file(path_or_fileobj=str(staging / "README.md"), path_in_repo="README.md", repo_id=args.repo_id, repo_type="model", commit_message="Add model card")
    api.upload_file(path_or_fileobj=str(staging / "measured_provenance.json"), path_in_repo="measured_provenance.json", repo_id=args.repo_id, repo_type="model", commit_message="Add measured provenance")
    files = api.list_repo_files(args.repo_id, repo_type="model")
    required = {"config.json", "README.md", "measured_provenance.json"}
    has_weight = any(name.endswith(".safetensors") or name.endswith(".bin") for name in files)
    if not required.issubset(files) or not has_weight:
        raise RuntimeError(f"remote verification failed; missing required files: {sorted(required - set(files))}")
    result = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "repo_id": args.repo_id,
        "url": f"https://huggingface.co/{args.repo_id}", "upload_commit": str(commit),
        "verified_required_files": sorted(required), "verified_weight_present": has_weight,
        "model_key": args.model_key, "local_model_dir": str(model_dir), "local_action": "retained_pending_retention_ledger",
    }
    audit = experiment / "hf_uploads.jsonl"
    with audit.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
