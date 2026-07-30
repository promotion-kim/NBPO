#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--audit", type=Path, required=True)
    a = p.parse_args()
    root = Path("/NHNHOME/AIPR/sjkim/uf5_inpo_warmstart_robustft_20260724").resolve()
    model = a.model.resolve()
    if root not in model.parents:
        raise RuntimeError("model path outside approved experiment root")
    repo = "promotion/ronpo-gemma2-2b-uf5-inpo-warmstart-s42"
    path = a.arm.lower().replace("_", "-")
    api = HfApi()
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo,
        repo_type="model",
        folder_path=str(model),
        path_in_repo=path,
        ignore_patterns=["checkpoint-*/*", "optimizer.pt", "scheduler.pt", "rng_state.pth", "wandb/*"],
        commit_message=f"Upload {a.arm} final gate-passing checkpoint",
    )
    weight = model / "model.safetensors"
    record = {
        "repo": repo,
        "public": True,
        "path_in_repo": path,
        "commit": commit.oid,
        "local_weight_sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
        "verified": bool(api.repo_info(repo, repo_type="model", revision=commit.oid).sha == commit.oid),
    }
    if not record["verified"]:
        raise RuntimeError("remote revision verification failed")
    a.audit.parent.mkdir(parents=True, exist_ok=True)
    a.audit.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()

