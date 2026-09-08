#!/usr/bin/env python3
"""Dump the per-row realized log-ratio change ``h`` and its row key.

Small enough to move off the pod, so the join against the preference tensor can
happen where the tensor lives. Only the join key and two floats travel: no
response text, no logps beyond what ``h`` needs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--precomputed", type=Path, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from datasets import load_from_disk
    d = load_from_disk(str(args.precomputed))[args.split].to_dict()
    h = (np.asarray(d["reference_chosen_logps"], float)
         - np.asarray(d["reference_rejected_logps"], float)
         - np.asarray(d["history0_chosen_logps"], float)
         + np.asarray(d["history0_rejected_logps"], float))
    np.savez_compressed(
        args.out, h=h,
        z=np.asarray(d["nbpo_weighted_z"], float),
        prompt_id=np.asarray(d["prompt_id"], dtype=object),
        chosen=np.asarray(d["chosen_response_id"], dtype=object),
        rejected=np.asarray(d["rejected_response_id"], dtype=object))
    print(f"wrote {args.out} rows={len(h)} h_rms={float(np.sqrt((h**2).mean())):.4f}")


if __name__ == "__main__":
    main()
