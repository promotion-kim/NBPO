#!/usr/bin/env python3
"""Upload a public, sealed-safe bundle needed to resume the flagship run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


PRECOMPUTED_NAMES = {
    "dataset_dict.json", "dataset_info.json", "precompute_status.json", "state.json"
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path, pattern: str = "*") -> None:
    for path in source.rglob(pattern):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--repo-id", default="promotion/aaai27-ronpo-flagship-resume-20260713")
    parser.add_argument("--sealed-sha256", required=True)
    parser.add_argument("--sealed-size", type=int, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    project = args.project.resolve()
    if "aipr_lab_sjkim_eval" not in root.parts:
        raise RuntimeError(f"refusing non-sjkim experiment root: {root}")
    staging = root / "resume_upload_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    selected: list[tuple[Path, Path]] = []
    for source in (root / "precomputed").rglob("*"):
        if not source.is_file():
            continue
        if source.name.startswith("data-") and source.suffix == ".arrow":
            selected.append((source, Path("precomputed") / source.relative_to(root / "precomputed")))
        elif source.name in PRECOMPUTED_NAMES:
            selected.append((source, Path("precomputed") / source.relative_to(root / "precomputed")))
    for source in (root / "kto_pointwise").glob("*"):
        if source.is_file():
            selected.append((source, Path("kto_pointwise") / source.name))
    for name in ("pool_train.jsonl", "pool_validation.jsonl", "split_manifest.json"):
        source = root / "data" / name
        if not source.is_file():
            raise FileNotFoundError(source)
        selected.append((source, Path("data") / name))

    for source, relative in selected:
        link_or_copy(source, staging / relative)

    metadata = staging / "metadata"
    metadata.mkdir()
    result_source = project / "results/ronpo_flagship_20260712"
    if result_source.is_dir():
        copy_tree(result_source, metadata / "protocol")
    live_hf_ledger = root / "hf_uploads.jsonl"
    if live_hf_ledger.is_file():
        (metadata / "protocol").mkdir(parents=True, exist_ok=True)
        shutil.copy2(live_hf_ledger, metadata / "protocol/hf_uploads.jsonl")
    for name in ("status", "stability"):
        source = root / name
        if source.is_dir():
            copy_tree(source, metadata / name)
    overnight = project / "results/overnight_status.md"
    if overnight.is_file():
        shutil.copy2(overnight, metadata / "overnight_status.md")

    copy_tree(project / "mnpo_scripts", staging / "code/mnpo_scripts")
    copy_tree(project / "scripts/revision", staging / "code/scripts/revision")
    copy_tree(project / "accelerate_configs", staging / "code/accelerate_configs")
    nhn_out = staging / "code/nhn"
    nhn_out.mkdir(parents=True)
    for source in (project / "nhn").glob("*.sh"):
        shutil.copy2(source, nhn_out / source.name)

    environment = subprocess.run(
        [str(Path(os.sys.executable)), "-m", "pip", "freeze"],
        text=True, capture_output=True, check=True,
    ).stdout
    (metadata / "environment.txt").write_text(
        f"python={platform.python_version()}\nplatform={platform.platform()}\n\n{environment}"
    )

    manifest_files = []
    for path in sorted(staging.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(staging).as_posix()
        manifest_files.append({
            "path": relative, "size": path.stat().st_size, "sha256": sha256(path),
        })
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "AAAI-27 RONPO flagship container-independent resume bundle",
        "base_model": "Qwen/Qwen3-8B",
        "reward_model": "RLHFlow/ArmoRM-Llama3-8B-v0.1",
        "sealed_test": {
            "included": False,
            "reason": "excluded from public repo to preserve the sealed evaluation",
            "local_backup": "results/ronpo_flagship_resume_20260713/sealed_test_prompts.jsonl",
            "size": args.sealed_size,
            "sha256": args.sealed_sha256,
        },
        "excluded_rebuildable_data": [
            "upstream model caches", "precompute cache-*.arrow", "precompute tmp* files",
            "pool_gate.jsonl", "W&B local cache",
        ],
        "files": manifest_files,
    }
    (metadata / "resume_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    readme = f"""---
pretty_name: AAAI-27 RONPO Flagship Resume Bundle
license: other
---

# AAAI-27 RONPO flagship resume bundle

This public dataset repository stores the frozen training/precompute artifacts,
protocols, status snapshot, and source snapshot needed to resume the matched
Qwen3-8B flagship experiment in a new container.

The sealed-test prompts are intentionally **not** public. Their local-server
backup has SHA-256 `{args.sealed_sha256}` and size `{args.sealed_size}` bytes;
the public `split_manifest.json` records the split hashes.

Only canonical `data-*.arrow` datasets are retained. Rebuildable Hugging Face
cache files, temporary Arrow files, upstream model caches, local W&B cache, and
`pool_gate.jsonl` are excluded. Model checkpoints are stored separately in the
public `promotion/qwen3-8b-aaai27-flagship-*` model repositories recorded in
`metadata/protocol/hf_uploads.jsonl`.

Download with `huggingface-cli download {args.repo_id} --repo-type dataset` and
restore the directory layout under the new flagship experiment root. Reload
the upstream Qwen3-8B and ArmoRM models from their public repositories.
"""
    (staging / "README.md").write_text(readme)

    status_path = root / "status/resume_bundle_upload.json"
    atomic_json(status_path, {
        "status": "uploading", "repo_id": args.repo_id,
        "selected_files": len(manifest_files),
        "selected_bytes": sum(item["size"] for item in manifest_files),
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_large_folder(
        args.repo_id, str(staging), repo_type="dataset", private=False,
        num_workers=4, print_report=True, print_report_every=60,
    )

    anonymous = HfApi(token=False)
    info = anonymous.repo_info(args.repo_id, repo_type="dataset")
    remote_files = set(anonymous.list_repo_files(args.repo_id, repo_type="dataset"))
    expected = {item["path"] for item in manifest_files} | {
        "README.md", "metadata/resume_manifest.json"
    }
    missing = sorted(expected - remote_files)
    if info.private or missing:
        raise RuntimeError(f"remote verification failed: private={info.private} missing={missing}")
    size_mismatches = []
    expected_sizes = {item["path"]: item["size"] for item in manifest_files}
    expected_sizes["README.md"] = (staging / "README.md").stat().st_size
    expected_sizes["metadata/resume_manifest.json"] = (
        metadata / "resume_manifest.json"
    ).stat().st_size
    paths_info = anonymous.get_paths_info(
        args.repo_id, list(expected_sizes), repo_type="dataset", revision=info.sha,
    )
    remote_sizes = {item.path: item.size for item in paths_info}
    for path, size in expected_sizes.items():
        if remote_sizes.get(path) != size:
            size_mismatches.append({
                "path": path, "local_size": size, "remote_size": remote_sizes.get(path),
            })
    if size_mismatches:
        raise RuntimeError(f"remote size verification failed: {size_mismatches}")
    with tempfile.TemporaryDirectory(prefix="ronpo_resume_verify_") as cache:
        downloaded = hf_hub_download(
            args.repo_id, "metadata/resume_manifest.json", repo_type="dataset",
            revision=info.sha, token=False, cache_dir=cache,
        )
        if sha256(Path(downloaded)) != sha256(metadata / "resume_manifest.json"):
            raise RuntimeError("anonymous manifest download checksum mismatch")

    result = {
        "status": "verified", "repo_id": args.repo_id,
        "url": f"https://huggingface.co/datasets/{args.repo_id}",
        "revision": info.sha, "private": info.private,
        "selected_files": len(manifest_files),
        "selected_bytes": sum(item["size"] for item in manifest_files),
        "remote_file_count": len(remote_files),
        "verified_checks": [
            "anonymous public repo lookup", "remote file listing",
            "remote file sizes", "anonymous fresh manifest download",
            "manifest checksum",
        ],
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    atomic_json(status_path, result)
    api.upload_file(
        path_or_fileobj=str(status_path), path_in_repo="metadata/resume_bundle_upload.json",
        repo_id=args.repo_id, repo_type="dataset",
        commit_message="Record verified resume-bundle upload",
    )
    shutil.rmtree(staging)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
