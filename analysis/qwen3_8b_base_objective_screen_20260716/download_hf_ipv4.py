#!/usr/bin/env python3
"""Download a pinned Hugging Face snapshot while forcing IPv4 in this container."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
from pathlib import Path


_original_getaddrinfo = socket.getaddrinfo


def ipv4_getaddrinfo(*args, **kwargs):
    values = [item for item in _original_getaddrinfo(*args, **kwargs) if item[0] == socket.AF_INET]
    if not values:
        raise OSError(f"no IPv4 address for {args[0] if args else 'host'}")
    return values


socket.getaddrinfo = ipv4_getaddrinfo

from huggingface_hub import HfApi, snapshot_download  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    info = HfApi().model_info(args.repo, revision="main", files_metadata=True)
    args.local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo,
        revision=info.sha,
        local_dir=args.local_dir,
        max_workers=16,
        ignore_patterns=["original/*", "*.pth", "*.gguf", "*.bin", "consolidated.safetensors"],
    )
    if not (args.local_dir / "config.json").is_file():
        raise RuntimeError("download completed without config.json")
    files = []
    for path in sorted(args.local_dir.rglob("*")):
        if path.is_file() and ".cache" not in path.parts:
            files.append({"path": str(path.relative_to(args.local_dir)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    safetensors = [row for row in files if row["path"].endswith(".safetensors")]
    payload = {
        "official_repo": args.repo,
        "resolved_revision": info.sha,
        "local_dir": str(args.local_dir),
        "file_count": len(files),
        "safetensor_count": len(safetensors),
        "safetensor_bytes": sum(row["bytes"] for row in safetensors),
        "files": files,
        "transport": "huggingface_official_ipv4_forced",
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ["official_repo", "resolved_revision", "local_dir", "file_count", "safetensor_count", "safetensor_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
