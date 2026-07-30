#!/usr/bin/env python3
"""Upload paper-relevant Qwen3-8B revision checkpoints to public HF repos.

This script intentionally uploads final checkpoint roots only. Nested
checkpoint-* directories are skipped to avoid duplicating intermediate state.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoConfig, AutoTokenizer


MODELS = [
    {
        "method": "DPO-avg",
        "repo_id": "promotion/qwen3-8b-dpo-avg-beta0p01-s42",
        "local_path": Path(
            "/ext_hdd/sjkim/mnpo/revision_qwen3_8b/full_iter1/train/"
            "dpo_avg_beta0p01_s42_odin2"
        ),
        "beta": "0.01",
    },
    {
        "method": "DPO-avg",
        "repo_id": "promotion/qwen3-8b-dpo-avg-beta0p05-s42",
        "local_path": Path(
            "/ext_hdd/sjkim/mnpo/revision_qwen3_8b/full_iter1/train/"
            "dpo_avg_beta0p05_s42_odin2"
        ),
        "beta": "0.05",
    },
]

ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "config.yaml",
    "train_results.json",
    "all_results.json",
    "trainer_state.json",
    "training_args.bin",
    "run_status.json",
]

IGNORE_PATTERNS = [
    "checkpoint-*",
    "checkpoint-*/*",
    "wandb",
    "wandb/*",
    "runs",
    "runs/*",
    "*.log",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state*.pth",
]


def build_readme(model: dict) -> str:
    return f"""---
library_name: transformers
base_model: Qwen/Qwen3-8B
license: other
tags:
- ronpo
- mnpo
- dpo
- qwen3
- preference-optimization
datasets:
- UltraFeedback
---

# {model["repo_id"].split("/", 1)[1]}

This is a research checkpoint for the RONPO AAAI revision experiments.

- Method: {model["method"]} on the averaged three-reward oracle
- Base model: `Qwen/Qwen3-8B`, non-thinking mode
- Training seed: 42
- DPO beta: {model["beta"]}
- Data split: the existing MNPO/RONPO UltraFeedback split
- Oracle construction: per-prompt min-max normalization over
  `Skywork/Skywork-Reward-V2-Llama-3.1-8B`,
  `Nexusflow/Athene-RM-8B`, and `RLHFlow/ArmoRM-Llama3-8B-v0.1`,
  followed by an unweighted average
- Local source checkpoint at upload time: `{model["local_path"]}`

Intended use: reproducibility and evaluation for the RONPO research paper.
This model is not intended as a general-purpose production assistant.
"""


def required_files_ok(path: Path) -> None:
    required = [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{path}: missing required files: {missing}")
    shards = sorted(path.glob("model-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"{path}: no safetensors shards found")


def main() -> None:
    api = HfApi()
    output_dir = Path("results/hf_uploads")
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = []

    for model in MODELS:
        local_path = model["local_path"].resolve()
        if not str(local_path).startswith("/ext_hdd/sjkim/"):
            raise RuntimeError(f"Refusing path outside /ext_hdd/sjkim: {local_path}")
        required_files_ok(local_path)

        repo_id = model["repo_id"]
        api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
        upload_info = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(local_path),
            allow_patterns=ALLOW_PATTERNS,
            ignore_patterns=IGNORE_PATTERNS,
            commit_message="Upload Qwen3-8B DPO averaged-oracle revision checkpoint",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            readme_path = Path(tmpdir) / "README.md"
            readme_path.write_text(build_readme(model), encoding="utf-8")
            api.upload_file(
                repo_id=repo_id,
                repo_type="model",
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                commit_message="Add RONPO revision model card",
            )

        info = api.model_info(repo_id=repo_id)
        sibling_names = {s.rfilename for s in info.siblings}
        required_remote = {
            "README.md",
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
        }
        missing_remote = sorted(required_remote - sibling_names)
        if missing_remote:
            raise RuntimeError(f"{repo_id}: missing remote files: {missing_remote}")

        verified_checks = [
            "remote file listing",
            "small-file fresh download",
        ]
        verification_warnings = []

        with tempfile.TemporaryDirectory() as tmp_cache:
            for filename in [
                "config.json",
                "model.safetensors.index.json",
                "tokenizer_config.json",
            ]:
                hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=tmp_cache)
            config_path = hf_hub_download(repo_id=repo_id, filename="config.json", cache_dir=tmp_cache)
            config_data = json.loads(Path(config_path).read_text(encoding="utf-8"))
            if config_data.get("model_type") != "qwen3":
                raise RuntimeError(f"{repo_id}: expected model_type=qwen3, got {config_data.get('model_type')}")
            verified_checks.append("config.json model_type sanity check")
            try:
                AutoConfig.from_pretrained(repo_id, cache_dir=tmp_cache)
                verified_checks.append("AutoConfig.from_pretrained")
            except Exception as exc:  # Local base env can be older than Qwen3 support.
                verification_warnings.append(f"AutoConfig skipped: {type(exc).__name__}: {exc}")
            try:
                AutoTokenizer.from_pretrained(repo_id, cache_dir=tmp_cache)
                verified_checks.append("AutoTokenizer.from_pretrained")
            except Exception as exc:
                verification_warnings.append(f"AutoTokenizer skipped: {type(exc).__name__}: {exc}")

        ledger.append(
            {
                "method": model["method"],
                "repo_id": repo_id,
                "local_path": str(local_path),
                "revision": info.sha,
                "private": getattr(info, "private", None),
                "uploaded_files": sorted(sibling_names),
                "upload_commit": str(upload_info),
                "verified": True,
                "verified_checks": verified_checks,
                "verification_warnings": verification_warnings,
            }
        )

    out_path = output_dir / (
        "qwen3_revision_hf_upload_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + ".json"
    )
    out_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(ledger, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
