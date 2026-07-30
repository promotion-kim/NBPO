#!/usr/bin/env python3
"""Public-upload, hash-verify, then safely prune exact model weight files."""

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


ARMS = ["ronpo_os", "ronpo_topmass", "inpo_avg", "sppo_avg", "simpo", "ipo", "dpo", "ht_mnpo_harmless", "ht_mnpo_helpfulness"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def model_dir(root: Path, stage: int, arm: str) -> Path:
    base = root / ("stage12" if stage <= 2 else f"stage{stage}")
    return base / f"stage{stage}/{arm}/train/full"


def gate_path(root: Path, stage: int, arm: str) -> Path:
    base = root / ("stage12" if stage <= 2 else f"stage{stage}")
    return base / f"stage{stage}_stability_p8_locked_panel/gates/{arm}.json"


def passed(path: Path) -> bool:
    try:
        d = json.loads(path.read_text(encoding="utf-8")); return d.get("passed") is True and d.get("status") == "passed"
    except (FileNotFoundError, json.JSONDecodeError): return False


def upload(api: HfApi, token: str, root: Path, stage: int, arm: str, out: Path) -> dict:
    local = model_dir(root, stage, arm).resolve(); approved = Path("/NHNHOME/AIPR/sjkim").resolve()
    if approved not in local.parents: raise RuntimeError(f"path outside approved namespace: {local}")
    weights = sorted(local.glob("*.safetensors"))
    if not weights or not (local / "config.json").is_file(): raise RuntimeError(f"incomplete model: {local}")
    repo = f"promotion-kim/ronpo-saferlhf-llama31-8b-s44-{arm.replace('_', '-')}"
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True, token=token)
    readme = out / f"README_{arm}.md"
    if not readme.exists():
        readme.write_text(f"---\nlicense: llama3.1\nbase_model: meta-llama/Llama-3.1-8B-Instruct\n---\n\n# SafeRLHF seed-44 {arm}\n\nSeed-44 matched-budget SafeRLHF training artifact. Each stage uses 900 steps, effective batch 16, and the Beaver helpfulness/harmlessness objectives. Stage {stage} passed the frozen 1,000-prompt reward-blind stability gate. Reward evaluation is pending.\n", encoding="utf-8")
        api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md", repo_id=repo, repo_type="model", token=token, commit_message="Add model card")
    commit = api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(local), path_in_repo=f"stage{stage}",
                               allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*", "special_tokens_map.json", "training_args.bin"],
                               token=token, commit_message=f"Upload seed-44 stage {stage}")
    info = api.model_info(repo, revision=commit.oid, files_metadata=True, token=token)
    if info.private: raise RuntimeError(f"repo is not public: {repo}")
    remote = {s.rfilename: s for s in info.siblings}
    checked = {}
    for path in weights:
        key = f"stage{stage}/{path.name}"; item = remote.get(key); lfs = getattr(item, "lfs", None) if item else None
        remote_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        local_sha = sha(path)
        if remote_sha != local_sha: raise RuntimeError(f"remote LFS hash mismatch: {key}")
        checked[key] = {"sha256": local_sha, "bytes": path.stat().st_size}
    AutoConfig.from_pretrained(repo, subfolder=f"stage{stage}", revision=commit.oid, token=token)
    AutoTokenizer.from_pretrained(repo, subfolder=f"stage{stage}", revision=commit.oid, token=token)
    return {"status": "verified", "repo": repo, "revision": commit.oid, "stage": stage, "arm": arm,
            "local": str(local), "weight_files": checked, "public": True, "verified_at": time.time()}


def prune(record: dict, root: Path, stage: int, arm: str) -> dict:
    if record.get("status") != "verified": return {"status": "blocked", "reason": "upload not verified"}
    if stage < 4 and not passed(gate_path(root, stage + 1, arm)): return {"status": "deferred", "reason": "child stage not gated"}
    local = model_dir(root, stage, arm).resolve(); approved = Path("/NHNHOME/AIPR/sjkim").resolve()
    if approved not in local.parents: raise RuntimeError(f"path outside approved namespace: {local}")
    lsof = subprocess.run(["lsof", "+D", str(local)], capture_output=True, text=True) if shutil.which("lsof") else None
    if lsof and lsof.stdout.strip(): return {"status": "deferred", "reason": "model path is open by a process"}
    deleted, freed = [], 0
    for path in sorted(local.glob("*.safetensors")):
        freed += path.stat().st_size; deleted.append(str(path)); path.unlink()
    for name in ("optimizer.pt", "optimizer.bin", "scheduler.pt"):
        path = local / name
        if path.is_file(): freed += path.stat().st_size; deleted.append(str(path)); path.unlink()
    return {"status": "pruned", "deleted": deleted, "freed_bytes": freed, "at": time.time()}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); p.add_argument("--poll", type=int, default=60); a = p.parse_args()
    token = os.environ.get("HF_TOKEN");
    if not token: raise RuntimeError("HF_TOKEN is required ephemerally")
    out = a.root / "stage4/hf_uploads"; out.mkdir(parents=True, exist_ok=True); api = HfApi(token=token)
    while True:
        for stage in range(1, 5):
            for arm in ARMS:
                record_path = out / f"stage{stage}_{arm}.json"; prune_path = out / f"stage{stage}_{arm}_prune.json"
                if passed(gate_path(a.root, stage, arm)) and not record_path.exists():
                    try:
                        record = upload(api, token, a.root, stage, arm, out); record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                    except Exception as exc:
                        (out / "upload_errors.jsonl").open("a", encoding="utf-8").write(json.dumps({"stage": stage, "arm": arm, "error": repr(exc), "at": time.time()}) + "\n")
                if record_path.exists() and not prune_path.exists():
                    record = json.loads(record_path.read_text(encoding="utf-8")); result = prune(record, a.root, stage, arm)
                    if result["status"] == "pruned": prune_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        done = all((out / f"stage4_{arm}.json").exists() and (out / f"stage4_{arm}_prune.json").exists() for arm in ARMS)
        if done: break
        time.sleep(a.poll)


if __name__ == "__main__": main()
