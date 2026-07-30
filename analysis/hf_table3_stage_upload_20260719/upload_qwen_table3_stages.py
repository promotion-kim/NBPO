#!/usr/bin/env python3
"""Upload and verify eligible Qwen Table-3 stage chains without pruning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoConfig, AutoTokenizer


MAIN = Path("/NHNHOME/sjkim/qwen25_7b_table4_stage4_20260718")
DIST = Path("/NHNHOME/AIPR/sjkim/qwen25_7b_table4_stage4_20260718_dist")
OUT = MAIN / "hf_table3_stage_upload_20260719"
ARMS = ("ronpo_os", "ht_mnpo_helpfulness", "ht_mnpo_harmless", "simpo", "dpo")


def model_dir(stage: int, arm: str) -> Path:
    if arm == "ht_mnpo_helpfulness" and stage >= 3:
        return DIST / f"seeds/s42/stage{stage}/{arm}/train/full"
    return MAIN / f"seeds/s42/stage{stage}/{arm}/train/full"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_hash(item) -> str | None:
    lfs = getattr(item, "lfs", None)
    if isinstance(lfs, dict):
        return lfs.get("sha256")
    return getattr(lfs, "sha256", None)


def upload_arm(api: HfApi, token: str, namespace: str, arm: str) -> None:
    repo = f"{namespace}/ronpo-saferlhf-qwen25-7b-s42-{arm.replace('_', '-')}"
    record_path = OUT / "records" / f"s42_{arm}.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"repo": repo, "seed": 42, "arm": arm, "public": True, "stages": {}}
    if record_path.is_file():
        record.update(json.loads(record_path.read_text()))
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True, token=token)
    card = OUT / "cards" / f"s42_{arm}.md"
    card.parent.mkdir(parents=True, exist_ok=True)
    card.write_text(
        "---\nlicense: apache-2.0\nbase_model: Qwen/Qwen2.5-7B-Instruct\n---\n\n"
        f"# SafeRLHF Qwen2.5-7B seed 42: {arm}\n\n"
        "Eligible paper Table-3 checkpoints for stages 1 through 4. Training uses Beaver reward "
        "helpfulness and negated Beaver cost harmlessness under the matched stage budget.\n"
    )
    api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo,
                    repo_type="model", token=token, commit_message="Update Table-3 model card")
    for stage in range(1, 5):
        local = model_dir(stage, arm).resolve()
        approved = (Path("/NHNHOME/sjkim").resolve(), Path("/NHNHOME/AIPR/sjkim").resolve())
        if not any(root in local.parents for root in approved):
            raise RuntimeError(f"outside approved namespace: {local}")
        status = json.loads((local / "job_status.json").read_text())
        if status.get("status") != "completed" or status.get("finite_metrics") is False:
            raise RuntimeError(f"training not complete: {local}")
        weights = sorted(local.glob("*.safetensors"))
        if weights:
            api.upload_folder(
                repo_id=repo, repo_type="model", folder_path=str(local), path_in_repo=f"stage{stage}",
                allow_patterns=["*.json", "*.safetensors", "*.model", "*.tiktoken", "tokenizer*",
                                "special_tokens_map.json", "training_args.bin", "config.yaml"],
                token=token, commit_message=f"Upload seed 42 stage {stage}",
            )
        info = api.model_info(repo, files_metadata=True, token=token)
        if info.private:
            raise RuntimeError(f"repository is private: {repo}")
        remote = {item.rfilename: item for item in info.siblings}
        remote_weights = {k: v for k, v in remote.items() if k.startswith(f"stage{stage}/") and k.endswith(".safetensors")}
        if not remote_weights:
            raise RuntimeError(f"no remote stage weights: {repo}/stage{stage}")
        checked = {}
        for path in weights:
            key = f"stage{stage}/{path.name}"
            local_hash = sha256(path)
            if key not in remote or remote_hash(remote[key]) != local_hash:
                raise RuntimeError(f"LFS hash mismatch: {repo}/{key}")
            checked[key] = {"sha256": local_hash, "bytes": path.stat().st_size}
        AutoConfig.from_pretrained(repo, subfolder=f"stage{stage}", revision=info.sha, token=token)
        AutoTokenizer.from_pretrained(repo, subfolder=f"stage{stage}", revision=info.sha, token=token)
        record["stages"][str(stage)] = {
            "status": "verified_public", "revision": info.sha, "local": str(local),
            "wandb_url": status.get("wandb_url"), "local_weight_hashes": checked,
            "remote_weight_files": sorted(remote_weights), "verified_at": time.time(),
        }
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"event": "verified", "repo": repo, "stage": stage, "revision": info.sha}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required ephemerally")
    api = HfApi(token=token)
    namespace = api.whoami(token=token)["name"]
    for index, arm in enumerate(ARMS):
        if index % args.workers != args.worker:
            continue
        try:
            upload_arm(api, token, namespace, arm)
        except Exception as exc:
            errors = OUT / "errors"
            errors.mkdir(parents=True, exist_ok=True)
            (errors / f"s42_{arm}.json").write_text(json.dumps({
                "seed": 42, "arm": arm, "error": repr(exc), "at": time.time()
            }, indent=2) + "\n")
            print(json.dumps({"event": "error", "arm": arm, "error": repr(exc)}), flush=True)


if __name__ == "__main__":
    main()
