#!/usr/bin/env python3
from __future__ import annotations

import textwrap
from pathlib import Path

from huggingface_hub import HfApi


ALLOW_PATTERNS = [
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "trainer_state.json",
    "training_args.bin",
    "train_results.json",
    "all_results.json",
]

REQUIRED_FILES = {
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
}

MODELS = [
    {
        "repo_name": "htmnpo-skywork-qwen25-1p5b-stage1",
        "path": "/ext_hdd/sjkim/mnpo/outputs/qwen2.5-1.5b-instruct_htmnpo_skywork_online_multiobj_stage_1",
        "description": "HT-MNPO stage-1 Qwen2.5-1.5B-Instruct policy trained with the Skywork reward oracle.",
        "eval_label": "htmnpo_skywork",
    },
    {
        "repo_name": "htmnpo-athene-qwen25-1p5b-stage1-ckpt300",
        "path": "/ext_hdd/sjkim/mnpo/outputs/qwen2.5-1.5b-instruct_htmnpo_athene_online_multiobj_stage_1/checkpoint-300",
        "description": "HT-MNPO stage-1 Qwen2.5-1.5B-Instruct checkpoint trained with the Athene reward oracle.",
        "eval_label": "htmnpo_athene",
    },
    {
        "repo_name": "htmnpo-armorm-qwen25-1p5b-stage1",
        "path": "/ext_hdd/sjkim/mnpo/ht_stage1_out/qwen2.5-1.5b-instruct_htmnpo_armo_online_multiobj_stage_1",
        "description": "HT-MNPO stage-1 Qwen2.5-1.5B-Instruct policy trained with the ArmoRM reward oracle.",
        "eval_label": "htmnpo_armorm",
    },
    {
        "repo_name": "ronpo-qwen25-1p5b-stage1-ckpt1100",
        "path": "/ext_hdd/sjkim/mnpo/outputs_ronpo_fair/qwen2.5-1.5b-instruct_ronpo_vs_htmnpo_online_multiobj_stage_1/checkpoint-1100",
        "description": "RONPO stage-1 Qwen2.5-1.5B-Instruct checkpoint used in the local RM evaluation table.",
        "eval_label": "ronpo",
    },
    {
        "repo_name": "ronpo-qwen25-1p5b-stage1-final",
        "path": "/ext_hdd/sjkim/mnpo/outputs_ronpo_fair/qwen2.5-1.5b-instruct_ronpo_vs_htmnpo_online_multiobj_stage_1/checkpoint-1184",
        "description": "Final RONPO stage-1 Qwen2.5-1.5B-Instruct checkpoint. Re-evaluate before replacing the ckpt1100 table value.",
        "eval_label": "ronpo_final",
    },
]


def write_model_card(path: Path, repo_id: str, info: dict[str, str]) -> Path:
    card = textwrap.dedent(
        f"""\
        ---
        license: apache-2.0
        base_model: Qwen/Qwen2.5-1.5B-Instruct
        pipeline_tag: text-generation
        ---

        # {repo_id}

        {info["description"]}

        This repository keeps the inference-ready model artifacts used for RONPO/MNPO paper experiments.
        The original local checkpoint path was:

        `{path}`

        Evaluation label: `{info["eval_label"]}`.

        Optimizer, RNG, and DeepSpeed ZeRO state files are intentionally not uploaded because this public
        repository is intended for paper evaluation and inference reproduction, not optimizer-state resume.
        """
    )
    tmp = Path("/tmp") / f"{repo_id.replace('/', '__')}_README.md"
    tmp.write_text(card, encoding="utf-8")
    return tmp


def main() -> None:
    api = HfApi()
    username = api.whoami()["name"]
    print(f"Logged in as: {username}")

    for info in MODELS:
        path = Path(info["path"])
        repo_id = f"{username}/{info['repo_name']}"
        if not path.is_dir():
            raise FileNotFoundError(path)

        print(f"\n==> Uploading {path} -> {repo_id}")
        api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)

        card = write_model_card(path, repo_id, info)
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
            commit_message=f"Upload inference artifacts for {info['repo_name']}",
        )

        files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
        missing = sorted(REQUIRED_FILES - files)
        if missing:
            raise RuntimeError(f"{repo_id} is missing required files: {missing}")

        uploaded = sorted(f for f in files if f in set(ALLOW_PATTERNS))
        print(f"Verified {repo_id}: {', '.join(uploaded)}")


if __name__ == "__main__":
    main()
