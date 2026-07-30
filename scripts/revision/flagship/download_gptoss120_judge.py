#!/usr/bin/env python3
"""Download the pinned Transformers-format gpt-oss-120b judge snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = "openai/gpt-oss-120b"
    revision = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
    path = snapshot_download(
        repo_id=repo,
        revision=revision,
        cache_dir=str(args.cache_dir),
        ignore_patterns=["original/*"],
        max_workers=4,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"repo": repo, "revision": revision, "snapshot_path": path}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(path, flush=True)


if __name__ == "__main__":
    main()
