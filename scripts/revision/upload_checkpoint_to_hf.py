#!/usr/bin/env python3
"""Upload one completed Transformers checkpoint directory to a public HF repo."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "pytorch_model-*.bin",
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
    "job_status.json",
    "stability_gate.json",
    "split_manifest.json",
    "objective_protocol.json",
    "resource_profile.json",
    "pair_manifest.json",
    "provenance.json",
    "benchmark_config.json",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-path", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--seed", default="42")
    parser.add_argument("--notes", default="")
    parser.add_argument("--ledger", default="")
    parser.add_argument(
        "--full-model-reload",
        action="store_true",
        help="Anonymously download and instantiate all uploaded weights before declaring success.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def find_weight_files(local_path: Path) -> list[Path]:
    """Return final weight artifacts stored directly in the run directory."""
    patterns = [
        "model.safetensors",
        "model.safetensors.index.json",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "pytorch_model-*.bin",
    ]
    files: dict[str, Path] = {}
    for pattern in patterns:
        for path in local_path.glob(pattern):
            if path.is_file():
                files[path.name] = path
    return [files[name] for name in sorted(files)]


def build_readme(args: argparse.Namespace, local_path: Path) -> str:
    config = read_json(local_path / "run_status.json")
    return f"""---
library_name: transformers
base_model: {args.base_model}
tags:
- ronpo
- mnpo
- preference-optimization
---

# {args.repo_id.split("/", 1)[-1]}

Research checkpoint for the RONPO AAAI revision experiments.

- Method: {args.method}
- Base model: `{args.base_model}`
- Seed: {args.seed}
- Local source at upload time: `{local_path}`
- Uploaded at UTC: {datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
- Run status metadata: `{json.dumps(config, ensure_ascii=False, sort_keys=True) if config else "not found"}`

{args.notes}

Intended use: reproducibility and evaluation for the RONPO paper. This
checkpoint is not intended as a production assistant.
"""


def main() -> None:
    args = parse_args()
    local_path = Path(args.local_path).resolve()
    if not local_path.is_dir():
        raise FileNotFoundError(f"missing checkpoint directory: {local_path}")
    if not (local_path / "config.json").is_file():
        raise FileNotFoundError(f"missing config.json in {local_path}")
    weight_files = find_weight_files(local_path)
    if not weight_files:
        raise FileNotFoundError(f"missing model weights in {local_path}")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=False, exist_ok=True)
    upload_info = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(local_path),
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        commit_message=f"Upload {args.method} checkpoint",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        readme = Path(tmpdir) / "README.md"
        readme.write_text(build_readme(args, local_path), encoding="utf-8")
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="model",
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            commit_message="Add research model card",
        )

    info = api.model_info(args.repo_id, files_metadata=True)
    if getattr(info, "private", None) is not False:
        raise RuntimeError(f"expected public repo, got private={info.private}")

    # `private=False` on an authenticated response is necessary but not quite
    # sufficient evidence that evaluators can fetch the model.  Resolve the
    # same revision again without a token and use only anonymous downloads for
    # the reload smoke test below.
    anonymous_api = HfApi(token=False)
    anonymous_info = anonymous_api.model_info(args.repo_id, files_metadata=True)
    if getattr(anonymous_info, "private", None) is not False:
        raise RuntimeError(
            f"anonymous repo lookup did not confirm public visibility: "
            f"private={anonymous_info.private}"
        )
    if anonymous_info.sha != info.sha:
        raise RuntimeError(
            f"anonymous revision mismatch: authenticated={info.sha} "
            f"anonymous={anonymous_info.sha}"
        )

    remote_by_name = {s.rfilename: s for s in info.siblings}
    remote_files = set(remote_by_name)
    weight_names = {path.name for path in weight_files}
    required = {"README.md", "config.json", *weight_names}
    if (local_path / "tokenizer_config.json").is_file():
        required.add("tokenizer_config.json")
    missing = sorted(required - remote_files)
    if missing:
        raise RuntimeError(f"remote repo missing files: {missing}")

    size_mismatches = []
    for path in weight_files:
        remote_size = getattr(remote_by_name[path.name], "size", None)
        if remote_size is not None and remote_size != path.stat().st_size:
            size_mismatches.append(
                {"file": path.name, "local": path.stat().st_size, "remote": remote_size}
            )
    if size_mismatches:
        raise RuntimeError(f"remote weight size mismatch: {size_mismatches}")

    full_model_reload = "not requested; local checkpoint retained"
    with tempfile.TemporaryDirectory() as cache:
        for filename in sorted(required - {"README.md", *weight_names}):
            hf_hub_download(
                repo_id=args.repo_id,
                filename=filename,
                revision=anonymous_info.sha,
                cache_dir=cache,
                token=False,
            )
        AutoConfig.from_pretrained(
            args.repo_id,
            revision=anonymous_info.sha,
            cache_dir=cache,
            token=False,
        )
        if "tokenizer_config.json" in required:
            AutoTokenizer.from_pretrained(
                args.repo_id,
                revision=anonymous_info.sha,
                cache_dir=cache,
                token=False,
            )
        if args.full_model_reload:
            model = AutoModelForCausalLM.from_pretrained(
                args.repo_id,
                revision=anonymous_info.sha,
                cache_dir=cache,
                token=False,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
            )
            if model.config.model_type != AutoConfig.from_pretrained(
                args.repo_id,
                revision=anonymous_info.sha,
                cache_dir=cache,
                token=False,
            ).model_type:
                raise RuntimeError("full model reload produced a config type mismatch")
            del model
            full_model_reload = "passed anonymous full-weight download and AutoModelForCausalLM reload"

    record = {
        "repo_id": args.repo_id,
        "revision": info.sha,
        "private": getattr(info, "private", None),
        "local_path": str(local_path),
        "method": args.method,
        "seed": args.seed,
        "upload_commit": str(upload_info),
        "verified": True,
        "verified_checks": [
            "public repo visibility",
            "anonymous public repo lookup",
            "remote file listing",
            "remote weight file size",
            "anonymous fresh metadata download",
            "anonymous AutoConfig reload",
            "anonymous AutoTokenizer reload",
        ] + (["anonymous full-weight AutoModelForCausalLM reload"] if args.full_model_reload else []),
        "full_model_reload": full_model_reload,
    }
    print(json.dumps(record, indent=2, ensure_ascii=False))

    if args.ledger:
        ledger_path = Path(args.ledger)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
