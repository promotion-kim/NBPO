#!/usr/bin/env python3
"""Upload and verify the complete Llama Table-3 stage chain without pruning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoConfig, AutoTokenizer


PROJECT = Path("/NHNHOME/AIPR/sjkim/MNPO_rev_20260710")
RESULTS = PROJECT / "results"
OUT = RESULTS / "hf_table3_stage_upload_20260719"
ARMS = (
    "ronpo_os", "ht_mnpo_helpfulness", "ht_mnpo_harmless", "inpo_avg",
    "sppo_avg", "simpo", "ipo", "dpo",
)


def model_dir(seed: int, stage: int, arm: str) -> Path:
    if seed == 42:
        if stage == 1:
            name = "ronpo_os_confirmatory" if arm == "ronpo_os" else arm
            return RESULTS / "p4_8b_saferlhf_table4_20260717/train/w1_900steps" / name
        if stage == 2:
            return RESULTS / "p5_8b_robust_stage1_stage2_20260717" / f"stage2/{arm}_stage2/train/full"
        if stage == 3:
            return RESULTS / "p7_stage3_fresh_default_test_20260717" / f"stage3/{arm}_stage3/train/full"
        return RESULTS / "p8_stage4_fresh_default_test_20260718" / f"stage4/{arm}_stage4/train/full"
    if seed == 43:
        run = {
            1: "p10_saferlhf_training_seed43_20260718",
            2: "p10_saferlhf_training_seed43_20260718",
            3: "p10_saferlhf_stage3_seed43_20260718",
            4: "p10_saferlhf_stage4_seed43_20260718",
        }[stage]
        return RESULTS / run / f"stage{stage}/{arm}/train/full"
    run = "stage12" if stage < 3 else f"stage{stage}"
    return RESULTS / "p13_saferlhf_seed44_20260718" / run / f"stage{stage}/{arm}/train/full"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(local: Path) -> tuple[dict, list[Path]]:
    approved = Path("/NHNHOME/AIPR/sjkim").resolve()
    local = local.resolve()
    if approved not in local.parents:
        raise RuntimeError(f"outside approved namespace: {local}")
    status = json.loads((local / "job_status.json").read_text())
    if status.get("status") != "completed" or status.get("returncode", 0) != 0:
        raise RuntimeError(f"training not complete: {local}")
    if status.get("finite_metrics") is False:
        raise RuntimeError(f"non-finite training metrics: {local}")
    weights = sorted(local.glob("*.safetensors"))
    if not weights or not (local / "config.json").is_file():
        raise RuntimeError(f"incomplete checkpoint: {local}")
    return status, weights


def remote_sha(item) -> str | None:
    lfs = getattr(item, "lfs", None)
    if isinstance(lfs, dict):
        return lfs.get("sha256")
    return getattr(lfs, "sha256", None)


def verify(api: HfApi, repo: str, stage: int, local: Path, token: str) -> tuple[str, dict]:
    info = api.model_info(repo, files_metadata=True, token=token)
    if info.private:
        raise RuntimeError(f"repository is private: {repo}")
    files = {item.rfilename: item for item in info.siblings}
    checked = {}
    for path in sorted(local.glob("*.safetensors")):
        local_hash = sha256(path)
        key = f"stage{stage}/{path.name}"
        if key not in files or remote_sha(files[key]) != local_hash:
            raise RuntimeError(f"LFS hash mismatch: {repo}/{key}")
        checked[key] = {"sha256": local_hash, "bytes": path.stat().st_size}
    AutoConfig.from_pretrained(repo, subfolder=f"stage{stage}", revision=info.sha, token=token)
    AutoTokenizer.from_pretrained(repo, subfolder=f"stage{stage}", revision=info.sha, token=token)
    return info.sha, checked


def upload_repo(api: HfApi, token: str, namespace: str, seed: int, arm: str) -> None:
    repo = f"{namespace}/ronpo-saferlhf-llama31-8b-s{seed}-{arm.replace('_', '-')}"
    record_path = OUT / "records" / f"s{seed}_{arm}.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"repo": repo, "seed": seed, "arm": arm, "public": True, "stages": {}}
    if record_path.is_file():
        record.update(json.loads(record_path.read_text()))
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True, token=token)
    card = OUT / "cards" / f"s{seed}_{arm}.md"
    card.parent.mkdir(parents=True, exist_ok=True)
    card.write_text(
        "---\nlicense: llama3.1\nbase_model: meta-llama/Llama-3.1-8B-Instruct\n---\n\n"
        f"# SafeRLHF Llama-3.1-8B seed {seed}: {arm}\n\n"
        "Paper Table-3 matched-budget checkpoints for stages 1 through 4. Training uses Beaver reward "
        "helpfulness and negated Beaver cost harmlessness, 900 steps per stage, effective batch 16, "
        "and the frozen final-checkpoint rule. See each stage's job_status.json for its W&B run.\n"
    )
    api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo,
                    repo_type="model", token=token, commit_message="Add Table-3 model card")
    for stage in range(1, 5):
        local = model_dir(seed, stage, arm)
        status, weights = validate(local)
        already = record.get("stages", {}).get(str(stage), {}).get("status") == "verified_public"
        if not already:
            api.upload_folder(
                repo_id=repo, repo_type="model", folder_path=str(local), path_in_repo=f"stage{stage}",
                allow_patterns=["*.json", "*.safetensors", "*.model", "*.tiktoken", "tokenizer*",
                                "special_tokens_map.json", "training_args.bin", "config.yaml"],
                token=token, commit_message=f"Upload seed {seed} stage {stage}",
            )
        revision, checked = verify(api, repo, stage, local, token)
        record["stages"][str(stage)] = {
            "status": "verified_public", "revision": revision, "local": str(local.resolve()),
            "wandb_url": status.get("wandb_url"), "weights": checked, "verified_at": time.time(),
        }
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"event": "verified", "repo": repo, "stage": stage, "revision": revision}), flush=True)


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
    tasks = [(seed, arm) for seed in (42, 43, 44) for arm in ARMS]
    for index, (seed, arm) in enumerate(tasks):
        if index % args.workers != args.worker:
            continue
        try:
            upload_repo(api, token, namespace, seed, arm)
        except Exception as exc:
            errors = OUT / "errors"
            errors.mkdir(parents=True, exist_ok=True)
            (errors / f"s{seed}_{arm}.json").write_text(json.dumps({
                "seed": seed, "arm": arm, "error": repr(exc), "at": time.time()
            }, indent=2) + "\n")
            print(json.dumps({"event": "error", "seed": seed, "arm": arm, "error": repr(exc)}), flush=True)


if __name__ == "__main__":
    main()
