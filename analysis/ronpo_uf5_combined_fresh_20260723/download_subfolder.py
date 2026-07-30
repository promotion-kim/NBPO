#!/usr/bin/env python3
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--subfolder", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=a.repo, revision=a.revision,
        allow_patterns=[f"{a.subfolder}/*"], local_dir=a.output,
    )
    model = a.output / a.subfolder
    for name in ("config.json", "model.safetensors", "tokenizer_config.json"):
        if not (model / name).is_file():
            raise RuntimeError(f"missing {name} in {model}")
    print(model)


if __name__ == "__main__":
    main()
