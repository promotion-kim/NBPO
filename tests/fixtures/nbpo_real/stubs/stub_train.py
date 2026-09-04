#!/usr/bin/env python3
"""The ONLY stubbed step: the neural weight update.

It still does the production-relevant work around that update -- parses the
materialized run_config with run_mnpo's own dataclasses, runs the real
``validate_nbpo_args`` against the real precompute sidecar, dataset columns and
dataset manifest (so a stale artifact fails here exactly as it would in
training), and only then writes the candidate checkpoint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from datasets import load_from_disk  # noqa: E402

from mnpo_scripts.precompute_provenance import (  # noqa: E402
    checkpoint_fingerprint,
    read_precompute_meta,
)
from scripts.nbpo.run_nbpo_stage import parse_run_config  # noqa: E402


def main() -> None:
    run_config = Path(sys.argv[1])
    model_args, data_args, training_args = parse_run_config(run_config)
    dataset_path = list(data_args.dataset_mixer)[0]
    ds = load_from_disk(dataset_path)
    splits = sorted(ds.keys())
    if splits != sorted(data_args.dataset_splits):
        raise SystemExit(f"run_config names splits {sorted(data_args.dataset_splits)} but the "
                         f"artifact has {splits}")
    from mnpo_scripts.mnpo_trainer import validate_nbpo_args

    validate_nbpo_args(
        training_args,
        dataset_columns=list(ds["train"].features),
        precompute_meta=read_precompute_meta(dataset_path),
        expected_parent_fingerprint=checkpoint_fingerprint(model_args.model_name_or_path),
        dataset_dir=dataset_path,
    )
    out = Path(training_args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Copy the parent's loadable files so the candidate is a real checkpoint,
    # then stamp it so its fingerprint differs from the parent's.
    import shutil

    for f in Path(model_args.model_name_or_path).iterdir():
        if f.is_file() and f.name != "MARKER.json" and not f.name.startswith("."):
            shutil.copy(f, out / f.name)
    (out / "MARKER.json").write_text(json.dumps(
        {"stub_train": True, "trained_from": model_args.model_name_or_path,
         "splits": splits, "run_config": str(run_config)}))
    print(f"[stub_train] candidate written to {out} (splits={splits})")


if __name__ == "__main__":
    main()
