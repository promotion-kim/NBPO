"""Provenance metadata for precomputed logp artifacts (NBPO validation support).

The precompute artifact historically recorded nothing about how its logps were
produced; a mean/sum reduction mismatch between artifact and trainer was silent
and undetectable after the fact. ``mnpo_scripts.precompute`` now writes a
sidecar ``precompute_meta.json`` next to the saved dataset, and
``run_mnpo`` / ``validate_nbpo_args`` verify it for ``loss_type=nbpo``.

Hashes are of CANONICAL CONTENT, never of directory listings or file bytes, so
two checkpoints of the same model family that tokenize identically compare
equal:

- ``chat_template_hash``: sha256 of the tokenizer's rendered chat-template
  string (empty string when the tokenizer has none);
- ``tokenizer_hash``: sha256 of the sorted-JSON serialization of the
  tokenizer's vocabulary plus its special-token map.

Tokenizer equality is NOT weight equality: two checkpoints of one family share
a tokenizer and differ in every weight. ``checkpoint_fingerprint`` therefore
hashes the content of every weight shard, the model index, the config, and the
inference-relevant tokenizer files, caching per-file digests keyed by
(size, mtime) so a 16 GB checkpoint is read once. That fingerprint binds
history0 to the true proximal centre pi_t, the decode manifest to the candidate
that produced it, and the gate record to both.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

PRECOMPUTE_META_FILENAME = "precompute_meta.json"
FINGERPRINT_CACHE_FILENAME = ".nbpo_fingerprint_cache.json"
FINGERPRINT_SCHEMA = 1
# Files whose content determines inference behaviour of a checkpoint directory.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
_CONFIG_FILES = (
    "config.json", "generation_config.json",
    "model.safetensors.index.json", "pytorch_model.bin.index.json",
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
    "special_tokens_map.json", "chat_template.jinja", "chat_template.json",
    "vocab.json", "merges.txt", "added_tokens.json",
    "MARKER.json",  # dry-run / stub checkpoints carry their identity here
)


def sha256_file_hex(path: str) -> str:
    """sha256 of a file's bytes (pair artifacts, solver artifacts)."""
    return _sha256_path(path)


def _sha256_path(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint_files(directory: str):
    names = []
    for root, _dirs, files in os.walk(directory):
        for fn in files:
            if fn.endswith(_WEIGHT_SUFFIXES) or fn in _CONFIG_FILES:
                names.append(os.path.relpath(os.path.join(root, fn), directory))
    return sorted(names)


def _cache_path(directory: str) -> str:
    primary = os.path.join(directory, FINGERPRINT_CACHE_FILENAME)
    if os.access(directory, os.W_OK):
        return primary
    side = os.path.join(os.path.expanduser("~"), ".cache", "nbpo_fingerprints")
    os.makedirs(side, exist_ok=True)
    return os.path.join(side, _sha256_text(os.path.abspath(directory)) + ".json")


def checkpoint_manifest(directory: str) -> dict:
    """Per-file sha256 of every inference-relevant file in a checkpoint dir (cached)."""
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"checkpoint directory not found: {directory}")
    names = _fingerprint_files(directory)
    if not names:
        raise ValueError(f"no weight/config files found under {directory}")
    cache_file = _cache_path(directory)
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cache = json.load(f).get("files", {})
        except Exception:
            cache = {}
    files, changed = {}, False
    for rel in names:
        st = os.stat(os.path.join(directory, rel))
        key = f"{st.st_size}:{int(st.st_mtime_ns)}"
        entry = cache.get(rel)
        if entry and entry.get("key") == key:
            files[rel] = {"key": key, "sha256": entry["sha256"], "size": st.st_size}
        else:
            files[rel] = {"key": key, "sha256": _sha256_path(os.path.join(directory, rel)),
                          "size": st.st_size}
            changed = True
    if changed or set(cache) != set(files):
        try:
            with open(cache_file, "w") as f:
                json.dump({"schema": FINGERPRINT_SCHEMA, "files": files}, f, indent=1)
        except OSError:
            pass
    return {rel: {"sha256": v["sha256"], "size": v["size"]} for rel, v in files.items()}


def checkpoint_fingerprint(directory: str) -> str:
    """Content fingerprint of a checkpoint: sha256 over sorted (relpath, sha256, size)."""
    manifest = checkpoint_manifest(directory)
    canon = "\n".join(f"{rel}:{v['sha256']}:{v['size']}" for rel, v in sorted(manifest.items()))
    return _sha256_text(f"nbpo-ckpt-v{FINGERPRINT_SCHEMA}\n" + canon)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenizer_content_hashes(tokenizer) -> dict:
    """Canonical (tokenizer_hash, chat_template_hash) for a loaded tokenizer."""
    vocab = tokenizer.get_vocab()
    special = {k: str(v) for k, v in sorted(tokenizer.special_tokens_map.items())}
    tokenizer_hash = _sha256_text(
        json.dumps({"vocab": vocab, "special_tokens": special}, sort_keys=True, ensure_ascii=False)
    )
    chat_template = getattr(tokenizer, "chat_template", None) or ""
    return {
        "tokenizer_hash": tokenizer_hash,
        "chat_template_hash": _sha256_text(str(chat_template)),
    }


def write_precompute_meta(output_dir: str, meta: dict) -> str:
    path = os.path.join(output_dir, PRECOMPUTE_META_FILENAME)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def read_precompute_meta(dataset_dir: str) -> Optional[dict]:
    """Sidecar metadata of a precomputed dataset dir, or None for legacy artifacts."""
    path = os.path.join(dataset_dir, PRECOMPUTE_META_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


PRECOMPUTE_MANIFEST_FILENAME = "precompute_manifest.json"
# Everything a saved DatasetDict consists of. Arrow shards carry the logps
# themselves, so an edited shard changes the training targets while every
# JSON-level check still passes -- they must be hashed.
_DATASET_SUFFIXES = (".arrow", ".json", ".txt", ".parquet")


def dataset_file_manifest(dataset_dir: str) -> dict:
    """Per-file sha256 of every file of a saved precomputed dataset (Arrow included)."""
    dataset_dir = os.path.abspath(dataset_dir)
    files = {}
    for root, _dirs, names in os.walk(dataset_dir):
        for fn in sorted(names):
            if fn in (PRECOMPUTE_META_FILENAME, PRECOMPUTE_MANIFEST_FILENAME):
                continue          # written after, and hashed by, this manifest
            if not fn.endswith(_DATASET_SUFFIXES):
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, dataset_dir)
            files[rel] = {"sha256": _sha256_path(full), "size": os.path.getsize(full)}
    if not files:
        raise ValueError(f"no dataset files found under {dataset_dir}")
    return files


def write_precompute_manifest(dataset_dir: str, splits=None):
    """Write ``precompute_manifest.json``; return ``(path, sha256_of_that_file)``."""
    manifest = {
        "schema": FINGERPRINT_SCHEMA,
        "dataset_dir": os.path.abspath(dataset_dir),
        "splits": sorted(splits or []),
        "files": dataset_file_manifest(dataset_dir),
    }
    path = os.path.join(dataset_dir, PRECOMPUTE_MANIFEST_FILENAME)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return path, _sha256_path(path)


def read_precompute_manifest(dataset_dir: str) -> Optional[dict]:
    path = os.path.join(dataset_dir, PRECOMPUTE_MANIFEST_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def verify_precompute_manifest(dataset_dir: str, expected_manifest_sha256: str = None) -> dict:
    """Re-hash the dataset and compare with its manifest; raises on any drift.

    An expected manifest sha (from the run config or the stage record) is
    checked too, so replacing BOTH a shard and the manifest still fails.
    """
    path = os.path.join(dataset_dir, PRECOMPUTE_MANIFEST_FILENAME)
    manifest = read_precompute_manifest(dataset_dir)
    if manifest is None:
        raise ValueError(
            f"{dataset_dir} has no {PRECOMPUTE_MANIFEST_FILENAME}; re-run "
            "mnpo_scripts.precompute (legacy artifacts cannot be integrity-checked)")
    if expected_manifest_sha256 is not None:
        got = _sha256_path(path)
        if got != expected_manifest_sha256:
            raise ValueError(
                f"precompute manifest sha256 {got[:12]} != expected "
                f"{expected_manifest_sha256[:12]} (the dataset artifact is not the one "
                "this run config was written for)")
    recorded = manifest.get("files") or {}
    current = dataset_file_manifest(dataset_dir)
    changed = sorted(k for k in recorded if k in current
                     and current[k]["sha256"] != recorded[k]["sha256"])
    missing = sorted(set(recorded) - set(current))
    added = sorted(set(current) - set(recorded))
    if changed or missing or added:
        raise ValueError(
            f"precomputed dataset {dataset_dir} does not match its manifest: "
            f"modified={changed[:5]} missing={missing[:5]} added={added[:5]}")
    return manifest
