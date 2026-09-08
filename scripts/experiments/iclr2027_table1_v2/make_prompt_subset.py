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

    src = args.parent / "precompute_meta.json"
    if src.exists():
        shutil.copy2(src, args.out / "precompute_meta.json")
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

    # The parent's manifest describes the parent's shards, so copying it would
    # make the trainer's integrity check fail -- correctly. Re-hash the subset
    # instead and point its meta at the new manifest, so the check still has
    # something true to verify rather than something disabled.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from mnpo_scripts.precompute_provenance import write_precompute_manifest

    _, manifest_sha = write_precompute_manifest(str(args.out), splits=list(dd.keys()))
    meta_path = args.out / "precompute_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["precompute_manifest_sha256"] = manifest_sha
        meta["split_sizes"] = {k: v.num_rows for k, v in dd.items()}
        meta["derived_from"] = {
            "parent_precomputed": str(args.parent),
            "operation": f"prompt-wise row filter to {len(ids)} prompts",
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{args.n_prompts:>4} prompts -> {sub.num_rows:>6} rows  "
          f"manifest {manifest_sha[:12]}  ({args.out})")


if __name__ == "__main__":
    main()
