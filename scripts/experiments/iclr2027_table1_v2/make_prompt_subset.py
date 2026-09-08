#!/usr/bin/env python3
"""Cut a precomputed NBPO dataset down to its first N training prompts.

The generalization question is whether the regression fails because it has too
little compute per pair or because per-prompt targets simply do not transfer.
Holding gradient updates fixed and varying the number of TRAINING PROMPTS
separates the two: every arm gets the same compute, sees the same held-out set,
and differs only in how many prompts its target came from.

The log-probability columns are not recomputed -- they are a property of the
frozen pi_t and the frozen response pool, not of which rows we keep -- so the
parent's precompute sidecar is carried over unchanged and the subset records its
own provenance beside it rather than pretending to be an independent precompute.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-prompts", type=int, required=True)
    args = ap.parse_args()

    from datasets import load_from_disk

    dd = load_from_disk(str(args.parent))
    train = dd["train"]
    ids = sorted(set(train["prompt_id"]))[: args.n_prompts]
    keep = set(ids)
    sub = train.filter(lambda r: r["prompt_id"] in keep, num_proc=8)
    dd["train"] = sub
    args.out.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(str(args.out))

    for name in ("precompute_meta.json", "precompute_manifest.json"):
        src = args.parent / name
        if src.exists():
            shutil.copy2(src, args.out / name)
    (args.out / "subset_provenance.json").write_text(json.dumps({
        "parent_precomputed": str(args.parent),
        "n_prompts_kept": len(ids),
        "n_train_rows": sub.num_rows,
        "parent_train_rows": train.num_rows,
        "selection": "first N prompt ids in sorted order -- deterministic, not sampled",
        "note": ("log-probability columns are inherited unchanged from the parent "
                 "precompute; this is a row filter, not a new precompute, and the "
                 "parent's sidecar is what the trainer's provenance check binds to"),
    }, indent=2) + "\n")
    print(f"{args.n_prompts:>4} prompts -> {sub.num_rows:>6} rows  ({args.out})")


if __name__ == "__main__":
    main()
