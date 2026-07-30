#!/usr/bin/env python3
"""Build deterministic, prompt-disjoint HH-RLHF manifests for covariance."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from datasets import load_dataset


def last_user(text: str) -> str:
    prefix = text.rsplit("\n\nAssistant:", 1)[0]
    return prefix.rsplit("\n\nHuman:", 1)[-1].strip()


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def select(rows, n: int, salt: str, banned: set[str]):
    candidates = {}
    for i, row in enumerate(rows):
        prompt = last_user(row["chosen"])
        key = hashlib.sha256(norm(prompt).encode()).hexdigest()
        if prompt and key not in banned and key not in candidates:
            candidates[key] = (i, prompt)
    ranked = sorted(candidates.items(), key=lambda x: hashlib.sha256(
        f"{salt}|{x[0]}".encode()).hexdigest())
    picked = ranked[:n]
    out = [{
        "prompt_id": f"{salt}-{j:04d}",
        "prompt": prompt,
        "source": "Anthropic/hh-rlhf",
        "source_row": i,
        "behavior_label": "unknown",
        "normalized_prompt_sha256": key,
    } for j, (key, (i, prompt)) in enumerate(picked)]
    return out, {key for key, _ in picked}, len(candidates)


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--train-n", type=int, default=512)
    ap.add_argument("--test-n", type=int, default=256)
    args = ap.parse_args()
    revision = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"
    ds = load_dataset("Anthropic/hh-rlhf", revision=revision)
    train, train_keys, n_train_unique = select(
        ds["train"], args.train_n, "cov-train", set())
    test, test_keys, n_test_unique = select(
        ds["test"], args.test_n, "cov-test", train_keys)
    if len(train) != args.train_n or len(test) != args.test_n:
        raise RuntimeError(f"insufficient prompts: train={len(train)} test={len(test)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "test.jsonl", test)
    meta = {
        "dataset": "Anthropic/hh-rlhf",
        "revision": revision,
        "selection": "SHA-256 rank of normalized last user turn",
        "train_count": len(train),
        "test_count": len(test),
        "train_test_overlap": len(train_keys & test_keys),
        "source_unique": {"train": n_train_unique, "test": n_test_unique},
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
