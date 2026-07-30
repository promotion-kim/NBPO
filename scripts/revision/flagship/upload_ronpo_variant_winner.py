#!/usr/bin/env python3
"""Upload only a measured base-matching/beating finalist and verify it is public."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--namespace", default="promotion")
    args = parser.parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    lock = json.loads((args.run_dir / "final_variant_set_lock.json").read_text(encoding="utf-8"))
    if not summary.get("upload_authorized") or summary.get("outcome") not in {"BEAT_BASE", "MATCH_BASE"}:
        raise RuntimeError("measured success rule does not authorize upload")
    selected = lock["selected_variant"]
    folder = Path(selected["model_path"])
    weight = folder / "model.safetensors"
    if not weight.is_file() or not (folder / "config.json").is_file():
        raise RuntimeError("selected checkpoint is not minimally reloadable")
    api = HfApi()
    who = api.whoami()["name"]
    if who != args.namespace:
        raise RuntimeError(f"HF account mismatch: expected {args.namespace}, got {who}")
    slug = selected["candidate_id"].replace("_", "-")
    repo_id = f"{args.namespace}/ronpo-qwen3-8b-worstobj-{slug}-s42"
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    card = args.run_dir / "WINNER_MODEL_CARD.md"
    card.write_text(
        "---\nbase_model: Qwen/Qwen3-8B\nlibrary_name: transformers\ntags:\n"
        "- preference-optimization\n- ronpo\n- qwen3\n---\n\n"
        f"# RONPO Qwen3-8B worst-objective variant\n\nValidation-selected candidate "
        f"`{selected['candidate_id']}` at step `{selected['step']}`. The prospective evaluation "
        "lock, final selection, and measured summary are included under `experiment/`.\n",
        encoding="utf-8")
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(folder),
                      ignore_patterns=["trainer_state.json", "training_args.bin", "checkpoint-*/*"],
                      commit_message=f"Upload locked RONPO winner {selected['candidate_id']}")
    api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo_id,
                    repo_type="model", commit_message="Add model card")
    for source, target in [
        (args.run_dir / "evaluator_lock.json", "experiment/evaluator_lock.json"),
        (args.run_dir / "final_variant_set_lock.json", "experiment/final_variant_set_lock.json"),
        (args.run_dir / "summary.json", "experiment/summary.json"),
        (args.run_dir / "REPORT.md", "experiment/REPORT.md"),
    ]:
        api.upload_file(path_or_fileobj=str(source), path_in_repo=target, repo_id=repo_id,
                        repo_type="model", commit_message="Add reproducibility metadata")
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    files = set(api.list_repo_files(repo_id=repo_id, repo_type="model", revision=info.sha))
    AutoConfig.from_pretrained(repo_id, revision=info.sha)
    sibling = next((value for value in info.siblings or [] if value.rfilename == "model.safetensors"), None)
    remote_size = getattr(sibling, "size", None) if sibling is not None else None
    verified = bool(not info.private and "config.json" in files and "model.safetensors" in files
                    and (remote_size in {None, weight.stat().st_size}))
    result = {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
              "candidate_id": selected["candidate_id"], "step": selected["step"],
              "repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}",
              "revision": info.sha, "public": not info.private,
              "minimal_config_reload": True, "local_weight_sha256": sha256(weight),
              "local_weight_bytes": weight.stat().st_size, "remote_reported_bytes": remote_size,
              "verified": verified, "spent_sealed_split_touched": False}
    with (args.run_dir / "hf_uploads.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")
    if not verified:
        raise RuntimeError("public upload verification failed")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
