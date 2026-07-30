#!/usr/bin/env python3
"""Upload, verify, and prune only this run's model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoConfig, AutoTokenizer

from common import ARMS, SEEDS, gate_path, model_dir


def passed(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("passed") is True and data.get("status") == "passed"
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def upload(api: HfApi, token: str, namespace: str, root: Path, seed: int, stage: int, arm: str, output: Path) -> dict:
    local = model_dir(root, seed, stage, arm).resolve()
    approved = (Path("/NHNHOME/sjkim").resolve(), Path("/NHNHOME/AIPR/sjkim").resolve())
    if not any(root in local.parents for root in approved):
        raise RuntimeError(f"path outside approved namespace: {local}")
    weights = sorted(local.glob("*.safetensors"))
    if not weights or not (local / "config.json").is_file():
        raise RuntimeError(f"incomplete model: {local}")
    repo = f"{namespace}/ronpo-saferlhf-qwen25-7b-s{seed}-{arm.replace('_', '-')}"
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True, token=token)
    card = output / f"README_s{seed}_{arm}.md"
    if not card.exists():
        status = json.loads((local / "job_status.json").read_text(encoding="utf-8"))
        card.write_text(
            "---\nlicense: apache-2.0\nbase_model: Qwen/Qwen2.5-7B-Instruct\n---\n\n"
            f"# SafeRLHF Qwen2.5-7B seed {seed}: {arm}\n\n"
            "Matched-budget RONPO Table-4 replication with Beaver helpfulness and harmlessness objectives. "
            "Each stage uses 900 steps, effective batch 16, and a frozen final-checkpoint rule. "
            f"Stage {stage} passed the locked 1,000-prompt reward-blind stability gate. "
            f"W&B run: {status.get('wandb_url')}. Reward evaluation is recorded separately.\n",
            encoding="utf-8",
        )
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo, repo_type="model", token=token, commit_message="Add model card")
    commit = api.upload_folder(
        repo_id=repo, repo_type="model", folder_path=str(local), path_in_repo=f"stage{stage}",
        allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*", "special_tokens_map.json", "training_args.bin"],
        token=token, commit_message=f"Upload seed {seed} stage {stage}",
    )
    info = api.model_info(repo, revision=commit.oid, files_metadata=True, token=token)
    if info.private:
        raise RuntimeError(f"repository is not public: {repo}")
    remote = {item.rfilename: item for item in info.siblings}
    checked = {}
    for path in weights:
        key = f"stage{stage}/{path.name}"
        item = remote.get(key)
        lfs = getattr(item, "lfs", None) if item else None
        remote_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        local_sha = sha(path)
        if remote_sha != local_sha:
            raise RuntimeError(f"remote LFS hash mismatch: {key}")
        checked[key] = {"sha256": local_sha, "bytes": path.stat().st_size}
    AutoConfig.from_pretrained(repo, subfolder=f"stage{stage}", revision=commit.oid, token=token)
    AutoTokenizer.from_pretrained(repo, subfolder=f"stage{stage}", revision=commit.oid, token=token)
    return {"status": "verified", "repo": repo, "revision": commit.oid, "seed": seed, "stage": stage, "arm": arm, "public": True, "weights": checked, "verified_at": time.time()}


def can_prune(root: Path, seed: int, stage: int, arm: str) -> bool:
    return stage == 4 or passed(gate_path(root, seed, stage + 1, arm)) or terminal(root, seed, arm)


def terminal(root: Path, seed: int, arm: str) -> bool:
    if gate_path(root, seed, 4, arm).is_file():
        return True
    name = f"s{seed}__stage4__{arm}"
    return any((root / "scheduler" / f"{name}.{suffix}.json").is_file() for suffix in ("DONE", "FAILED", "BLOCKED"))


def prune(root: Path, seed: int, stage: int, arm: str) -> dict:
    local = model_dir(root, seed, stage, arm).resolve()
    approved = (Path("/NHNHOME/sjkim").resolve(), Path("/NHNHOME/AIPR/sjkim").resolve())
    if not any(root in local.parents for root in approved):
        raise RuntimeError(f"path outside approved namespace: {local}")
    probe = subprocess.run(["lsof", "+D", str(local)], capture_output=True, text=True) if shutil.which("lsof") else None
    if probe and probe.stdout.strip():
        return {"status": "deferred", "reason": "model path open by a process"}
    deleted, freed = [], 0
    for pattern in ("*.safetensors", "optimizer.pt", "optimizer.bin", "scheduler.pt"):
        for path in local.glob(pattern):
            freed += path.stat().st_size
            deleted.append(str(path))
            path.unlink()
    return {"status": "pruned", "deleted": deleted, "freed_bytes": freed, "at": time.time()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    parser.add_argument("--stages", type=int, nargs="+", choices=(1, 2, 3, 4), default=[4])
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be supplied ephemerally")
    output = args.root / "hf_uploads"
    output.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=token)
    namespace = api.whoami(token=token)["name"]
    while True:
        for seed in args.seeds:
            for arm in args.arms:
                for stage in args.stages:
                    record_path = output / f"s{seed}_stage{stage}_{arm}.json"
                    prune_path = output / f"s{seed}_stage{stage}_{arm}_prune.json"
                    if passed(gate_path(args.root, seed, stage, arm)) and not record_path.exists():
                        try:
                            record = upload(api, token, namespace, args.root, seed, stage, arm, output)
                            record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                        except Exception as exc:
                            with (output / "errors.jsonl").open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps({"seed": seed, "stage": stage, "arm": arm, "error": repr(exc), "at": time.time()}) + "\n")
                    if record_path.exists() and not prune_path.exists() and can_prune(args.root, seed, stage, arm):
                        result = prune(args.root, seed, stage, arm)
                        if result["status"] == "pruned":
                            prune_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if all(
            terminal(args.root, seed, arm)
            and all(
                not passed(gate_path(args.root, seed, stage, arm))
                or ((output / f"s{seed}_stage{stage}_{arm}.json").exists() and (output / f"s{seed}_stage{stage}_{arm}_prune.json").exists())
                for stage in args.stages
            )
            for seed in args.seeds for arm in args.arms
        ):
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
