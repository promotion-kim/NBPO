#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoConfig, AutoTokenizer


REPO = "promotion/ronpo-gemma2-2b-uf5-anneal-s42"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lfs_sha(info):
    lfs = getattr(info, "lfs", None)
    if isinstance(lfs, dict):
        return lfs.get("sha256")
    return getattr(lfs, "sha256", None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--stage", type=int, required=True)
    p.add_argument("--audit", type=Path, required=True)
    a = p.parse_args()
    dest = f"{a.arm}/stage{a.stage}"
    model = a.model.resolve()
    root = Path("/NHNHOME/AIPR/sjkim/ronpo_uf5_anneal_20260722").resolve()
    if root not in model.parents:
        raise RuntimeError(f"refusing path outside experiment namespace: {model}")
    weight = model / "model.safetensors"
    required_local = [model / "config.json", weight, model / "tokenizer_config.json"]
    if any(not x.is_file() or x.stat().st_size == 0 for x in required_local):
        raise RuntimeError(f"incomplete local model: {[str(x) for x in required_local]}")
    try:
        git_commit = subprocess.check_output(
            ["git", "-C", "/NHNHOME/AIPR/sjkim/MNPO_rev_20260720", "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        git_commit = "unknown_remote_snapshot_not_git"
    repro = {
        "method": "RONPO objective-stratified",
        "arm": a.arm,
        "stage": a.stage,
        "base_model": "google/gemma-2-2b-it",
        "dataset": "UltraFeedback five-head split",
        "seed": 42,
        "kappa": 0.05,
        "alpha": 1.0,
        "tau": 0.05,
        "max_steps": 1800,
        "checkpoint_rule": "final step",
        "source_checkpoint": str(model),
        "git_commit": git_commit,
        "stability_gate": json.loads((model.parent / "eval" / "stability_gate.json").read_text()),
        "known_limitations": "This checkpoint is one preregistered stage; final reward conclusions require the common-batch 586-prompt evaluation.",
    }
    (model / "reproducibility.json").write_text(json.dumps(repro, indent=2) + "\n")
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
# RONPO Gemma-2-2B UF5 annealing experiment

Public checkpoints for the preregistered moving-anchor and stronger-signal arms. Each stage is stored in
`moving_anchor/stageN` or `stronger_signal/stageN`. Training uses objective-stratified RONPO over five ArmoRM
heads on UltraFeedback, seed 42, kappa 0.05, alpha 1.0, tau 0.05, and 1,800 steps. The per-stage
`reproducibility.json` records the exact arm, source revision, and reward-blind stability gate. These are
research artifacts; reward conclusions must use the common-batch 586-prompt evaluation in the accompanying
MNPO results rather than checkpoint selection. After both arms finish, the measured report and JSON summary
are published under `evaluation/` in this repository.
"""
    api.upload_file(
        repo_id=REPO, path_or_fileobj=card.encode(), path_in_repo="README.md",
        repo_type="model", commit_message="Document preregistered UF5 experiment",
    )
    commit = api.upload_folder(
        repo_id=REPO,
        folder_path=str(model),
        path_in_repo=dest,
        commit_message=f"Add {a.arm} Stage {a.stage}",
        ignore_patterns=["README.md", "checkpoint-*", "wandb/**", "*.log", "optimizer.pt", "scheduler.pt", "rng_state.pth"],
    )
    latest = api.model_info(REPO, files_metadata=True)
    revision = latest.sha
    if latest.private:
        raise RuntimeError("repository is not public")
    files = set(api.list_repo_files(repo_id=REPO, repo_type="model", revision=revision))
    required = {
        f"{dest}/config.json", f"{dest}/model.safetensors",
        f"{dest}/tokenizer_config.json", f"{dest}/reproducibility.json",
    }
    missing = sorted(required - files)
    if missing:
        raise RuntimeError(f"remote verification failed: {missing}")
    info = api.get_paths_info(
        repo_id=REPO, paths=sorted(required), repo_type="model", revision=revision, expand=True
    )
    sizes = {x.path: x.size for x in info}
    if any(not sizes.get(path, 0) for path in required):
        raise RuntimeError(f"invalid remote sizes: {sizes}")
    remote_weight = next(x for x in info if x.path == f"{dest}/model.safetensors")
    local_weight_sha = sha256(weight)
    remote_weight_sha = lfs_sha(remote_weight)
    if remote_weight_sha != local_weight_sha:
        raise RuntimeError(f"weight SHA mismatch: local={local_weight_sha} remote={remote_weight_sha}")
    with tempfile.TemporaryDirectory(prefix="uf5_hf_verify_") as cache:
        AutoConfig.from_pretrained(REPO, subfolder=dest, revision=revision, cache_dir=cache)
        AutoTokenizer.from_pretrained(REPO, subfolder=dest, revision=revision, cache_dir=cache)
    result = {
        "repo": REPO,
        "public": True,
        "path_in_repo": dest,
        "upload_commit": getattr(commit, "oid", None) or str(commit),
        "verified_revision": revision,
        "local_path": str(model),
        "local_weight_sha256": local_weight_sha,
        "remote_lfs_weight_sha256": remote_weight_sha,
        "verified_required_sizes": sizes,
        "fresh_remote_config_load": True,
        "fresh_remote_tokenizer_load": True,
        "full_model_reload": "not run; exact local/remote LFS SHA-256 match verified",
        "verified": True,
    }
    a.audit.parent.mkdir(parents=True, exist_ok=True)
    a.audit.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
