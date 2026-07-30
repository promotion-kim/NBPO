#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoConfig, AutoTokenizer

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


MODELS = [
    {
        "repo_name": "htmnpo-skywork-qwen25-1p5b-stage2",
        "path": "/ext_hdd/sjkim/mnpo/ht_stage1_out/qwen2.5-1.5b-instruct_htmnpo_skywork_online_multiobj_stage_2",
        "description": "HT-MNPO stage-2 policy trained from the Skywork homogeneous oracle player.",
        "eval_label": "htmnpo_skywork_s2",
    },
    {
        "repo_name": "htmnpo-athene-qwen25-1p5b-stage2",
        "path": "/ext_hdd/sjkim/mnpo/ht_stage1_out/qwen2.5-1.5b-instruct_htmnpo_athene_online_multiobj_stage_2",
        "description": "HT-MNPO stage-2 policy trained from the Athene homogeneous oracle player.",
        "eval_label": "htmnpo_athene_s2",
    },
    {
        "repo_name": "htmnpo-armorm-qwen25-1p5b-stage2",
        "path": "/ext_hdd/sjkim/mnpo/ht_stage1_out/qwen2.5-1.5b-instruct_htmnpo_armo_online_multiobj_stage_2",
        "description": "HT-MNPO stage-2 policy trained from the ArmoRM homogeneous oracle player.",
        "eval_label": "htmnpo_armorm_s2",
    },
    {
        "repo_name": "ronpo-qwen25-1p5b-stage2-ckpt1400",
        "path": "/ext_hdd/sjkim/mnpo/outputs_ronpo_h200/qwen2.5-1.5b-instruct_ronpo_stage2_relative_lr2e8_od2g2/checkpoint-1400",
        "description": "RONPO resumed stage-2 checkpoint-1400 used in local RM and IFEval comparisons.",
        "eval_label": "ronpo_s2_ckpt1400",
    },
    {
        "repo_name": "ronpo-qwen25-1p5b-stage2-ckpt2457",
        "path": "/ext_hdd/sjkim/mnpo/outputs_ronpo_h200/qwen2.5-1.5b-instruct_ronpo_stage2_relative_lr2e8_od2g2/checkpoint-2457",
        "description": "RONPO resumed stage-2 final checkpoint-2457 used in local RM, IFEval, and GPT judge comparisons.",
        "eval_label": "ronpo_s2_ckpt2457",
    },
    {
        "repo_name": "ronpo-safe-fullatom-qwen25-1p5b-ckpt900",
        "path": "/ext_hdd/sjkim/mnpo/experiments/ronpo_safety_conflict_qwen25_1p5b_20260629/outputs/ronpo-safe-full-s1_seed42/checkpoint-900",
        "description": "Safety-conflict full atom adversary checkpoint-900, the trainer-best checkpoint for the diagnostic ablation run.",
        "eval_label": "ronpo_safe_fullatom_ckpt900",
    },
    {
        "repo_name": "ronpo-safe-fullatom-qwen25-1p5b-ckpt3152",
        "path": "/ext_hdd/sjkim/mnpo/experiments/ronpo_safety_conflict_qwen25_1p5b_20260629/outputs/ronpo-safe-full-s1_seed42/checkpoint-3152",
        "description": "Safety-conflict full atom adversary final checkpoint-3152 used in the final-checkpoint atom ablation table.",
        "eval_label": "ronpo_safe_fullatom_ckpt3152",
    },
    {
        "repo_name": "ronpo-safe-konly-qwen25-1p5b-ckpt1600",
        "path": "/ext_hdd/sjkim/mnpo/experiments/ronpo_safety_conflict_qwen25_1p5b_20260629/outputs/ronpo-safe-konly-s1_seed42/checkpoint-1600",
        "description": "Safety-conflict k-only adversary checkpoint-1600, the trainer-best checkpoint for the diagnostic ablation run.",
        "eval_label": "ronpo_safe_konly_ckpt1600",
    },
    {
        "repo_name": "ronpo-safe-konly-qwen25-1p5b-ckpt2227",
        "path": "/ext_hdd/sjkim/mnpo/experiments/ronpo_safety_conflict_qwen25_1p5b_20260629/outputs/ronpo-safe-konly-s1_seed42/checkpoint-2227",
        "description": "Safety-conflict k-only adversary final checkpoint-2227 used in the final-checkpoint atom ablation table.",
        "eval_label": "ronpo_safe_konly_ckpt2227",
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
    "wandb/**",
    "README.md",
]

REQUIRED_COMMON = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "trainer_state.json",
    "training_args.bin",
}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def verify_local(path: Path) -> None:
    resolved = path.resolve()
    if not str(resolved).startswith("/ext_hdd/sjkim/"):
        raise RuntimeError(f"Refusing non-/ext_hdd/sjkim path: {resolved}")
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    files = {p.name for p in resolved.iterdir() if p.is_file()}
    missing = sorted(REQUIRED_COMMON - files)
    if missing:
        raise RuntimeError(f"{resolved} is missing required files: {missing}")
    has_single = (resolved / "model.safetensors").is_file()
    has_index = (resolved / "model.safetensors.index.json").is_file()
    has_shards = any(p.name.startswith("model-") and p.name.endswith(".safetensors") for p in resolved.iterdir())
    if not (has_single or (has_index and has_shards)):
        raise RuntimeError(f"{resolved} has no model.safetensors or indexed shard files")


def write_model_card(path: Path, repo_id: str, info: dict[str, str], commit: str) -> Path:
    card = textwrap.dedent(
        f"""\
        ---
        license: apache-2.0
        base_model: Qwen/Qwen2.5-1.5B-Instruct
        pipeline_tag: text-generation
        ---

        # {repo_id}

        {info["description"]}

        This repository stores inference artifacts for RONPO paper experiments.

        - Evaluation label: `{info["eval_label"]}`
        - Original local path: `{path}`
        - Source Git commit: `{commit}`
        - Main comparison report: `ronpo_paper_eval_table_analysis.md`

        Optimizer, scheduler, RNG, W&B, and intermediate checkpoint state files are
        intentionally not uploaded. This public repo is for evaluation and inference
        reproduction, not optimizer-state resume.
        """
    )
    tmp = Path("/tmp") / f"{repo_id.replace('/', '__')}_README.md"
    tmp.write_text(card, encoding="utf-8")
    return tmp


def verify_remote(api: HfApi, repo_id: str) -> str:
    info = api.model_info(repo_id=repo_id, files_metadata=True)
    remote_files = {s.rfilename: s for s in info.siblings}
    missing = sorted(REQUIRED_COMMON - set(remote_files))
    if missing:
        raise RuntimeError(f"{repo_id} is missing required files: {missing}")
    has_single = "model.safetensors" in remote_files
    has_index = "model.safetensors.index.json" in remote_files
    has_shards = any(name.startswith("model-") and name.endswith(".safetensors") for name in remote_files)
    if not (has_single or (has_index and has_shards)):
        raise RuntimeError(f"{repo_id} has no model weights")
    nested = sorted(name for name in remote_files if name.startswith("checkpoint-"))
    if nested:
        raise RuntimeError(f"{repo_id} unexpectedly contains nested checkpoints: {nested[:5]}")

    cache_dir = "/tmp/hf-prune-verify-cache"
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model", cache_dir=cache_dir)
    AutoConfig.from_pretrained(repo_id, cache_dir=cache_dir)
    AutoTokenizer.from_pretrained(repo_id, cache_dir=cache_dir)
    return info.sha


def main() -> None:
    api = HfApi()
    username = api.whoami()["name"]
    commit = git_commit()
    print(f"Logged in as: {username}")

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
            commit_message=f"Upload inference artifacts for {info['repo_name']}",
        )
        sha = verify_remote(api, repo_id)
        print(f"Verified {repo_id}@{sha}")


if __name__ == "__main__":
    main()
