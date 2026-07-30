#!/usr/bin/env python3
"""Upload one paper checkpoint subfolder and verify its reload-critical files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


KEEP = {
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "chat_template.jinja", "tokenizer.model",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--path-in-repo", required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--title", required=True)
    a = p.parse_args()
    api = HfApi()
    user = api.whoami()["name"]
    if a.repo.split("/", 1)[0] != user:
        raise RuntimeError(f"authenticated HF namespace is {user}, not {a.repo.split('/', 1)[0]}")
    files = sorted([x for x in a.model.iterdir() if x.is_file() and (x.name in KEEP or x.name.endswith(".safetensors") or x.name == "model.safetensors.index.json")])
    if not any(x.name.endswith(".safetensors") for x in files) or not any(x.name == "config.json" for x in files):
        raise RuntimeError(f"not a reloadable model directory: {a.model}")
    api.create_repo(a.repo, repo_type="model", private=False, exist_ok=True)
    readme = f"---\nlibrary_name: transformers\n---\n\n# {a.title}\n\nPaper checkpoint for the locked RONPO Figure 2/3 continuation experiment. Training used seed 42 and W&B project `promotion-kim/mnpo`. Each stage is stored in a named subfolder. Base-model licensing terms continue to apply. Evaluation artifacts and provenance remain in the accompanying code repository.\n"
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=a.repo, repo_type="model", commit_message="Add model card")
    for file in files:
        api.upload_file(path_or_fileobj=str(file), path_in_repo=f"{a.path_in_repo}/{file.name}", repo_id=a.repo, repo_type="model", commit_message=f"Upload {a.path_in_repo}/{file.name}")
    info = api.repo_info(a.repo, repo_type="model", files_metadata=True)
    remote = {x.rfilename: x.size for x in info.siblings}
    expected = {f"{a.path_in_repo}/{x.name}": x.stat().st_size for x in files}
    missing = [name for name in expected if name not in remote]
    size_mismatch = [name for name, size in expected.items() if remote.get(name) not in (None, size)]
    if missing or size_mismatch or info.private:
        raise RuntimeError(f"upload verification failed: missing={missing}, size_mismatch={size_mismatch}, private={info.private}")
    payload = {"repo": a.repo, "commit": info.sha, "public": not info.private, "path_in_repo": a.path_in_repo,
               "local_model": str(a.model), "files": expected, "verified": True}
    a.audit.parent.mkdir(parents=True, exist_ok=True)
    a.audit.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
