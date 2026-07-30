#!/usr/bin/env python3
"""Upload verified final repaired checkpoints publicly, then prune local weights."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoConfig, AutoTokenizer


ROOT = Path("/NHNHOME/AIPR/sjkim/qwen25_7b_baseline_repair_20260719")
ARMS = ("inpo_avg", "ipo", "sppo_avg")


def passed(path: Path) -> bool:
    try:
        data = json.loads(path.read_text())
        return data.get("passed") is True and data.get("status") == "passed"
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be supplied ephemerally")
    api = HfApi(token=token)
    namespace = api.whoami(token=token)["name"]
    output = ROOT / "hf_uploads_repaired"
    output.mkdir(parents=True, exist_ok=True)
    pending = set(ARMS)
    while pending:
        for arm in tuple(pending):
            gate = ROOT / f"seeds/s42/stage4/gates/{arm}.json"
            status = ROOT / f"seeds/s42/stage4/{arm}/train/full/job_status.json"
            if status.is_file() and json.loads(status.read_text()).get("status") == "failed":
                (output / f"{arm}_failed.json").write_text(json.dumps({"status": "training_failed"}, indent=2) + "\n")
                pending.remove(arm)
                continue
            if not passed(gate):
                time.sleep(1)
                continue
            local = (ROOT / f"seeds/s42/stage4/{arm}/train/full").resolve()
            if Path("/NHNHOME/AIPR/sjkim") not in local.parents:
                raise RuntimeError(f"path outside approved namespace: {local}")
            weights = sorted(local.glob("*.safetensors"))
            if not weights or not (local / "config.json").is_file():
                raise RuntimeError(f"incomplete model: {local}")
            repo = f"{namespace}/ronpo-saferlhf-qwen25-7b-s42-{arm.replace('_', '-')}"
            api.create_repo(repo, repo_type="model", private=False, exist_ok=True, token=token)
            run = json.loads(status.read_text())
            card = output / f"README_{arm}.md"
            card.write_text("---\nlicense: apache-2.0\nbase_model: Qwen/Qwen2.5-7B-Instruct\n---\n\n"
                            f"# Qwen2.5-7B SafeRLHF seed 42: {arm}\n\n"
                            "Four-stage matched-budget SafeRLHF alignment checkpoint. The final model passed "
                            "the locked 1,000-prompt reward-blind stability gate. "
                            f"W&B: {run.get('wandb_url')}.\n")
            api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo, token=token)
            commit = api.upload_folder(repo_id=repo, folder_path=str(local), path_in_repo="stage4", repo_type="model",
                                       allow_patterns=["*.json", "*.safetensors", "tokenizer*", "special_tokens_map.json"], token=token)
            info = api.model_info(repo, revision=commit.oid, files_metadata=True, token=token)
            if info.private:
                raise RuntimeError(f"repository is private: {repo}")
            remote = {x.rfilename: x for x in info.siblings}
            verified = {}
            for path in weights:
                item = remote.get(f"stage4/{path.name}")
                lfs = getattr(item, "lfs", None) if item else None
                remote_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
                local_sha = digest(path)
                if remote_sha != local_sha:
                    raise RuntimeError(f"LFS hash mismatch: {path.name}")
                verified[path.name] = local_sha
            AutoConfig.from_pretrained(repo, subfolder="stage4", revision=commit.oid, token=token)
            AutoTokenizer.from_pretrained(repo, subfolder="stage4", revision=commit.oid, token=token)
            record = {"status": "verified_public", "repo": repo, "revision": commit.oid,
                      "weights": verified, "gate": str(gate), "verified_at": time.time()}
            (output / f"{arm}.json").write_text(json.dumps(record, indent=2) + "\n")
            deleted = []
            for path in weights:
                deleted.append({"path": str(path), "bytes": path.stat().st_size})
                path.unlink()
            (output / f"{arm}_prune.json").write_text(json.dumps({"status": "pruned_after_verification", "deleted": deleted}, indent=2) + "\n")
            pending.remove(arm)
        time.sleep(30)


if __name__ == "__main__":
    main()
