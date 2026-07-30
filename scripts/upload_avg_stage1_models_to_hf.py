#!/usr/bin/env python3
from __future__ import annotations

import textwrap
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoConfig, AutoTokenizer


MODELS = [
    {
        "repo_name": "sppo-avg-qwen25-1p5b-stage1",
        "path": "/home/sjkim/mnpo_runs/loki3/out/sppo_s1",
        "description": "SPPO stage-1 Qwen2.5-1.5B-Instruct policy trained with the homogeneous prompt-wise average reward oracle.",
        "eval_label": "sppo",
    },
    {
        "repo_name": "inpo-avg-qwen25-1p5b-stage1",
        "path": "/home/sjkim/mnpo_runs/loki3/out/inpo_s1",
        "description": "INPO stage-1 Qwen2.5-1.5B-Instruct policy trained with the homogeneous prompt-wise average reward oracle.",
        "eval_label": "inpo",
    },
]

ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.bin",
]

IGNORE_PATTERNS = [
    "checkpoint-*",
    "checkpoint-*/**",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state*.pth",
    "global_step*",
    "runs/**",
]

REQUIRED_FILES = {
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "trainer_state.json",
    "training_args.bin",
    "train_results.json",
    "all_results.json",
}


def write_model_card(path: Path, repo_id: str, info: dict[str, str], git_commit: str) -> Path:
    card = textwrap.dedent(
        f"""\
        ---
        license: apache-2.0
        base_model: Qwen/Qwen2.5-1.5B-Instruct
        pipeline_tag: text-generation
        ---

        # {repo_id}

        {info["description"]}

        This repository stores the final inference artifacts used for the RONPO paper
        stage-1 comparison table.

        - Evaluation label: `{info["eval_label"]}`
        - Original local path: `{path}`
        - Source Git commit: `{git_commit}`
        - Evaluation report: `ronpo_stage1_local_rm_eval_report.md`
        - Comparison report: `ronpo_paper_eval_table_analysis.md`

        Optimizer, RNG, and intermediate `checkpoint-*` directories are intentionally
        not uploaded because this public repository is intended for paper evaluation
        and inference reproduction, not optimizer-state resume.
        """
    )
    tmp = Path("/tmp") / f"{repo_id.replace('/', '__')}_README.md"
    tmp.write_text(card, encoding="utf-8")
    return tmp


def verify_local(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = {p.name for p in path.iterdir() if p.is_file()}
    missing = sorted(REQUIRED_FILES - files)
    shards = sorted(p.name for p in path.glob("model-*.safetensors"))
    if missing:
        raise RuntimeError(f"{path} is missing required files: {missing}")
    if not shards:
        raise RuntimeError(f"{path} has no model shard files")


def verify_remote(api: HfApi, repo_id: str) -> str:
    info = api.model_info(repo_id=repo_id, files_metadata=True)
    remote_files = {s.rfilename: s for s in info.siblings}
    missing = sorted(REQUIRED_FILES - set(remote_files))
    shards = sorted(name for name in remote_files if name.startswith("model-") and name.endswith(".safetensors"))
    nested = sorted(name for name in remote_files if name.startswith("checkpoint-"))
    if missing:
        raise RuntimeError(f"{repo_id} is missing required files: {missing}")
    if not shards:
        raise RuntimeError(f"{repo_id} has no model shard files")
    if nested:
        raise RuntimeError(f"{repo_id} unexpectedly contains checkpoint files: {nested[:5]}")

    cache_dir = "/tmp/hf-prune-verify-cache"
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model", cache_dir=cache_dir)
    AutoConfig.from_pretrained(repo_id, cache_dir=cache_dir)
    AutoTokenizer.from_pretrained(repo_id, cache_dir=cache_dir)
    return info.sha


def main() -> None:
    api = HfApi()
    username = api.whoami()["name"]
    git_commit = Path(".git/refs/heads/master")
    commit = git_commit.read_text(encoding="utf-8").strip() if git_commit.exists() else "unknown"

    for info in MODELS:
        path = Path(info["path"]).resolve()
        verify_local(path)
        repo_id = f"{username}/{info['repo_name']}"

        print(f"==> Uploading {path} -> {repo_id}")
        api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)

        card = write_model_card(path, repo_id, info, commit)
        api.upload_file(
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add model card for {info['repo_name']}",
        )
        api.upload_folder(
            folder_path=str(path),
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=ALLOW_PATTERNS,
            ignore_patterns=IGNORE_PATTERNS,
            commit_message=f"Upload final inference artifacts for {info['repo_name']}",
        )
        sha = verify_remote(api, repo_id)
        print(f"Verified {repo_id}@{sha}")


if __name__ == "__main__":
    main()
