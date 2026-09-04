#!/usr/bin/env python3
"""Run the PRODUCTION mnpo_scripts.precompute -- not a stub of it.

The audit forbids a precompute stub that omits production behaviour, in
particular the empty-split handling this fixture exists to exercise. This module
only fixes up ``sys.argv`` and calls ``mnpo_scripts.precompute.main`` directly,
so dataset loading, split construction, chat templating, the logp loop, the
provenance sidecar and the dataset manifest all run for real against the tiny
CPU checkpoint the fixture builds.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def ensure_tiny_model(path: Path) -> Path:
    """A real (tiny) causal LM + tokenizer at ``path``; built once, then cached.

    Built by the fixture BEFORE the stage runs: adding weight files to the parent
    directory afterwards would change its content fingerprint mid-run and break
    the history0 binding the pipeline is supposed to enforce.
    """
    path = Path(path)
    if (path / "config.json").exists():
        return path
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    path.mkdir(parents=True, exist_ok=True)
    cfg = GPT2Config(vocab_size=50257, n_positions=256, n_embd=32, n_layer=2, n_head=2)
    GPT2LMHeadModel(cfg).save_pretrained(path)
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.save_pretrained(path)
    return path


def main() -> None:
    sys.argv = ["precompute.py"] + sys.argv[1:]
    from mnpo_scripts.precompute import main as real_main

    real_main()


if __name__ == "__main__":
    main()
