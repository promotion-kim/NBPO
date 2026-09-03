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
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

PRECOMPUTE_META_FILENAME = "precompute_meta.json"


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
