#!/usr/bin/env python3
"""Stub for `mnpo_scripts.precompute` in the real-mode orchestration test.

Writes a valid DatasetDict (train/test) with the columns loss_type=nbpo needs
and a provenance sidecar whose history0 fingerprint is the CONTENT fingerprint
of the parent checkpoint it was given -- exactly what run_mnpo verifies.
"""
import argparse, json, sys
from pathlib import Path
from datasets import Dataset, DatasetDict
from mnpo_scripts.precompute_provenance import (checkpoint_fingerprint, sha256_file_hex,
                                                write_precompute_manifest,
                                                write_precompute_meta)
from mnpo_scripts.pair_tokenization import (tokenization_config,
                                            tokenization_config_hash)

ap = argparse.ArgumentParser()
ap.add_argument("--parent", required=True)
ap.add_argument("--pairs-dir", required=True)
ap.add_argument("--output-dir", required=True)
ap.add_argument("--solver-dir", required=True)
a = ap.parse_args()

def load(split):
    rows = [json.loads(l) for l in Path(a.pairs_dir, f"pairs_{split}.jsonl").read_text().splitlines() if l.strip()]
    n = len(rows)
    return Dataset.from_dict({
        "prompt": [r["prompt"] for r in rows], "chosen": [r["chosen"] for r in rows],
        "rejected": [r["rejected"] for r in rows],
        "nbpo_weighted_z": [float(r["nbpo_weighted_z"]) for r in rows],
        "reference_chosen_logps": [-10.0] * n, "reference_rejected_logps": [-11.0] * n,
        "history0_chosen_logps": [-10.0] * n, "history0_rejected_logps": [-11.0] * n,
    })
ds = DatasetDict({"train": load("train")})
if Path(a.pairs_dir, "pairs_test.jsonl").exists() and Path(a.pairs_dir, "pairs_test.jsonl").stat().st_size:
    ds["test"] = load("test")
out = Path(a.output_dir); ds.save_to_disk(str(out))
fp = checkpoint_fingerprint(a.parent)
# Real precompute records how it tokenized, because Eq. (22) subtracts logps that
# must come from the same token ids the trainer scores. The stub declares the
# same fields (with the production defaults) so the orchestration checks run.
tok_cfg = tokenization_config(max_length=2048, max_prompt_length=1024,
                              truncation_mode="keep_end")
manifest_path, manifest_sha = write_precompute_manifest(str(out), splits=sorted(ds.keys()))
write_precompute_meta(str(out), {
    "logp_reduction": "sum", "tokenizer_hash": "stub-tok", "chat_template_hash": "stub-chat",
    "model_fingerprint": fp, "reference_fingerprint": fp, "history_fingerprints": [fp],
    "history_paths": [a.parent],
    "pair_artifact_sha256": sha256_file_hex(str(Path(a.pairs_dir, "pairs_train.jsonl"))),
    "solver_artifact_sha256": sha256_file_hex(str(Path(a.solver_dir, "solution.json"))),
    "dataset_splits": sorted(ds.keys()),
    "split_sizes": {k: len(v) for k, v in ds.items()},
    "precompute_manifest_sha256": manifest_sha,
    **tok_cfg,
    "tokenization_config_sha256": tokenization_config_hash(tok_cfg),
    "stub": True,
})
print(json.dumps({"stub_precompute": str(out), "history_fingerprints": [fp]}))
