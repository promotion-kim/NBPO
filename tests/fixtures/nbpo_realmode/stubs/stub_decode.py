#!/usr/bin/env python3
"""Stub decode for the real-mode test: asserts it received the EXACT candidate
checkpoint the stage produced (recorded by stub_train in MARKER.json), then
delegates to the real synchronous decoder with the mock backend, which writes
the fingerprint-bound manifest the gate verifies."""
import json, sys
from pathlib import Path
from scripts.nbpo import decode_candidate as dc

argv = sys.argv[1:]
ckpt = Path(argv[argv.index("--model-checkpoint") + 1])
marker = json.loads((ckpt / "MARKER.json").read_text())
assert marker.get("stub_train") is True, f"decode got a checkpoint not written by stub_train: {ckpt}"
(ckpt.parent / "decode_received_checkpoint.txt").write_text(str(ckpt.resolve()))
sys.argv = ["decode_candidate.py"] + argv + ["--backend", "mock"]
dc.main()
