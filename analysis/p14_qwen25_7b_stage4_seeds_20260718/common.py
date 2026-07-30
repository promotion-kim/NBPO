"""Locked Qwen2.5-7B SafeRLHF Stage-1-to-4 experiment constants."""

from __future__ import annotations

import hashlib
from pathlib import Path


BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
BASE_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
REWARD_ID = "PKU-Alignment/beaver-7b-v1.0-reward"
REWARD_REVISION = "375cd6a9f0d7e339d2199b05ba129a4a8906596d"
COST_ID = "PKU-Alignment/beaver-7b-v1.0-cost"
COST_REVISION = "c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28"
SEEDS = (43, 44)
ARMS = {
    "ronpo_os": ("ronpo", "target_os"),
    "inpo_avg": ("inpo", None),
    "sppo_avg": ("sppo", None),
    "simpo": ("simpo", None),
    "ipo": ("ipo", None),
    "dpo": ("dpo", None),
    "ht_mnpo_harmless": ("ht_mnpo", "ht_target"),
    "ht_mnpo_helpfulness": ("ht_mnpo", "ht_target_helpfulness"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def seed_root(root: Path, seed: int) -> Path:
    return root / "seeds" / f"s{seed}"


def stage_root(root: Path, seed: int, stage: int) -> Path:
    return seed_root(root, seed) / f"stage{stage}"


def model_dir(root: Path, seed: int, stage: int, arm: str) -> Path:
    return stage_root(root, seed, stage) / arm / "train" / "full"


def gate_path(root: Path, seed: int, stage: int, arm: str) -> Path:
    return stage_root(root, seed, stage) / "gates" / f"{arm}.json"
