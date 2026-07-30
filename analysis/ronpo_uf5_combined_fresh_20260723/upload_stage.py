#!/usr/bin/env python3
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoConfig, AutoTokenizer

REPO = "promotion/ronpo-gemma2-2b-uf5-combined-fresh-s42"
ROOT = Path("/NHNHOME/AIPR/sjkim/ronpo_uf5_combined_fresh_20260723")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lfs_sha(info):
    lfs = getattr(info, "lfs", None)
    return lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--arm", choices=("combined", "fixed_base"), required=True)
    p.add_argument("--stage", type=int, required=True)
    p.add_argument("--audit", type=Path, required=True)
    a = p.parse_args()
    model = a.model.resolve()
    if ROOT.resolve() not in model.parents:
        raise RuntimeError(f"path outside locked experiment: {model}")
    weight = model / "model.safetensors"
    for name in ("config.json", "model.safetensors", "tokenizer_config.json"):
        if not (model / name).is_file() or not (model / name).stat().st_size:
            raise RuntimeError(f"incomplete model: {name}")
    gate = model.parent / "eval" / "stability_gate.json"
    metadata = {
        "method": "RONPO objective-stratified",
        "arm": a.arm,
        "stage": a.stage,
        "base_model": "google/gemma-2-2b-it",
        "seed": 42,
        "kappa": 0.05,
        "alpha": 1.0,
        "tau": 0.05,
        "learning_rate": 1e-6,
        "max_steps": 1800,
        "checkpoint_rule": "final step",
        "stability_gate": json.loads(gate.read_text()),
        "preregistration": "results/ronpo_uf5_combined_fresh_20260723/PREREG.md",
    }
    (model / "reproducibility.json").write_text(json.dumps(metadata, indent=2) + "\n")
    api = HfApi()
    api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
    card = """---
license: gemma
base_model: google/gemma-2-2b-it
library_name: transformers
tags:
- ronpo
- multi-objective-alignment
---
# RONPO Gemma-2-2B UF5 combined continuation

Public checkpoints from the preregistered seed-42 UF5 continuation. The
`combined` arm uses a heterogeneous ten-response pool and a moving parent
anchor; `fixed_base` uses the identical Stage-5 pool with Base as anchor.
Training uses objective-stratified RONPO with kappa 0.05, alpha 1.0, tau 0.05,
learning rate 1e-6, and 1,800 steps per stage. Every uploaded checkpoint passed
the locked reward-blind stability gate. Reward conclusions require the one-shot
fresh 647-prompt joint evaluation and must not be inferred from checkpoint
availability alone.
"""
    api.upload_file(repo_id=REPO, repo_type="model", path_or_fileobj=card.encode(),
                    path_in_repo="README.md", commit_message="Document preregistered UF5 continuation")
    dest = f"{a.arm}/stage{a.stage}"
    commit = api.upload_folder(
        repo_id=REPO, repo_type="model", folder_path=str(model), path_in_repo=dest,
        commit_message=f"Add {a.arm} stage {a.stage}",
        # Trainer writes a subfolder README whose base_model is the local parent
        # path.  The repository-level card above carries the valid public base.
        ignore_patterns=["README.md", "checkpoint-*", "optimizer.pt", "scheduler.pt", "rng_state.pth", "*.log"],
    )
    info = api.model_info(REPO, files_metadata=True)
    if info.private:
        raise RuntimeError("repository is not public")
    paths = [f"{dest}/{x}" for x in ("config.json", "model.safetensors", "tokenizer_config.json", "reproducibility.json")]
    files = set(api.list_repo_files(REPO, repo_type="model", revision=info.sha))
    if set(paths) - files:
        raise RuntimeError(f"missing remote files: {sorted(set(paths) - files)}")
    remote = api.get_paths_info(REPO, paths, repo_type="model", revision=info.sha, expand=True)
    rw = next(x for x in remote if x.path.endswith("model.safetensors"))
    local_sha = sha256(weight)
    if lfs_sha(rw) != local_sha:
        raise RuntimeError("remote LFS SHA does not match local weight")
    with tempfile.TemporaryDirectory(prefix="uf5_combined_verify_") as cache:
        AutoConfig.from_pretrained(REPO, subfolder=dest, revision=info.sha, cache_dir=cache)
        AutoTokenizer.from_pretrained(REPO, subfolder=dest, revision=info.sha, cache_dir=cache)
    result = {
        "repo": REPO, "public": True, "path_in_repo": dest,
        "upload_commit": getattr(commit, "oid", None) or str(commit),
        "verified_revision": info.sha, "weight_sha256": local_sha, "verified": True,
    }
    a.audit.parent.mkdir(parents=True, exist_ok=True)
    a.audit.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
