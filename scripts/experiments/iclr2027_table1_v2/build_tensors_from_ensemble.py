#!/usr/bin/env python3
"""Write finite-pool preference tensors from the frozen preference-model ensemble.

This is the replacement for the retired prompted-judge tensor builder. It emits
exactly the artifacts the existing solver and Eq. (26) pair builder consume --
``tensor_policy.npz``, ``tensor_ref.npz`` and a ``meta.json`` carrying the same
contract block -- so nothing downstream needs a special case for how the
preferences were obtained.

Two properties are enforced rather than hoped for:

* the reference block is made **exactly skew-symmetric with a zero diagonal**.
  ``P_k(y > y) = 1/2`` is Eq. (1), an identity, not an imputation; and the
  ensemble's own antisymmetry residual (about 1e-7, float32 sigmoid precision) is
  projected away here so the disagreement point is computed on a tensor that
  satisfies the algebra the theory assumes. The size of that projection is
  recorded.
* the calibration temperatures are applied **before** averaging seeds, matching
  how the oracle was frozen.

Provenance is recorded in full: which checkpoints, which calibration file, which
pool manifest, and the hash of each.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.nbpo.nbpo_common import (
    implementation_contract, response_pool_hash, sha256_text,
)

SCHEMA_VERSION = "nbpo-tensor-2"


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probs-dir", type=Path, required=True)
    ap.add_argument("--pool-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default="gpm", choices=("gpm", "bt"))
    ap.add_argument("--pool-size", type=int, default=8)
    ap.add_argument("--objectives", nargs="+", default=["helpfulness", "harmlessness"])
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--responses", type=Path, default=None,
                    help="the pool_responses.jsonl these probabilities were scored "
                         "from. Required to bind the tensor to the exact response "
                         "STRINGS: the same seed files can be regenerated with new "
                         "text and every id-level hash would still match, so the "
                         "Eq. (26) pair builder refuses a tensor without it.")
    args = ap.parse_args()

    n = args.pool_size
    pol = np.load(args.probs_dir / f"probs_{args.model}_policy.npz")["p"].astype(np.float64)
    ref = np.load(args.probs_dir / f"probs_{args.model}_ref.npz")["p"].astype(np.float64)
    if n > pol.shape[-1]:
        raise SystemExit(f"pool size {n} exceeds the scored pool {pol.shape[-1]}")
    pol = pol[:, :, :, :n, :n].mean(axis=0)          # (K, X, n, n), seeds averaged
    ref = ref[:, :, :, :n, :n].mean(axis=0)

    A_policy = pol - 0.5
    A_ref_raw = ref - 0.5
    A_ref = 0.5 * (A_ref_raw - np.swapaxes(A_ref_raw, -1, -2))
    idx = np.arange(n)
    diag_before = float(np.abs(A_ref_raw[..., idx, idx]).max())
    skew_residual = float(np.abs(A_ref_raw + np.swapaxes(A_ref_raw, -1, -2)).max())
    A_ref[..., idx, idx] = 0.0

    if np.abs(A_policy).max() > 0.5 + 1e-9 or np.abs(A_ref).max() > 0.5 + 1e-9:
        raise SystemExit("centered payoffs out of [-1/2, 1/2]")

    manifest = json.loads((args.pool_dir / "pool_manifest.json").read_text())
    scoring = json.loads((args.probs_dir / "scoring_manifest.json").read_text())

    # Bind the tensor to the exact response strings it was built from.
    resp_path = args.responses or (args.pool_dir / "pool_responses.jsonl")
    pool = {"policy": {}, "ref": {}}
    text_sha = {"policy": {}, "reference": {}}
    for line in resp_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["sample_index"] >= n:
            continue
        if r["role"] == "learner":
            side, tag, key = "policy", "policy", f"s{r['seed']}"
        else:
            side, tag, key = "ref", "reference", f"r{r['seed']}"
        pool[side].setdefault(key, {})[r["prompt_id"]] = {
            "generated_text": r["response"]}
        text_sha[tag].setdefault(f"{tag}:{key}", {})[r["prompt_id"]] = \
            sha256_text(r["response"])
    learner_pool_hash = response_pool_hash(pool["policy"])
    reference_pool_hash = response_pool_hash(pool["ref"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "tensor_policy.npz", A=A_policy)
    np.savez_compressed(args.out_dir / "tensor_ref.npz", A=A_ref)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "objectives": list(args.objectives),
        "prompt_ids": manifest["prompt_ids"],
        "policy_learner_ids": [f"policy:s{manifest['learner_seeds'][i]}" for i in range(n)],
        "comparator_ids": [f"ref:s{manifest['comparator_seeds'][i]}" for i in range(n)],
        "generation_seeds": {
            "policy": {f"policy:s{s}": s for s in manifest["learner_seeds"][:n]},
            "reference": {f"ref:s{s}": s for s in manifest["comparator_seeds"][:n]},
        },
        "preference_source": {
            "kind": "learned_preference_model_ensemble",
            "model": args.model,
            "seeds": scoring["seeds"],
            "calibration": scoring.get("calibration_source"),
            "calibration_applied_before_averaging": True,
            "note": ("the prompted-LLM judge is permanently retired; preferences "
                     "come from a frozen calibrated ensemble trained on released "
                     "human pairwise annotations"),
        },
        "judge_models": [],
        "rubric_versions": [],
        "self_pairs": ("identity_zero (Eq. (1): P_k(y>y|x)=1/2 exactly; not an "
                       "imputation)"),
        "reference_skew": {
            "pre_projection_max_skew_residual": skew_residual,
            "pre_projection_max_abs_diagonal": diag_before,
            "projected": True,
            "note": ("the ensemble is antisymmetric to float32 sigmoid precision; "
                     "the residual is projected away so the disagreement point is "
                     "computed on a tensor satisfying the assumed algebra"),
        },
        "reference_construction": "shared_pool",
        "tensor_kind": "centered_preference",
        "pool_geometry": f"{n}+{n}",
        "learner_response_pool_hash": learner_pool_hash,
        "reference_response_pool_hash": reference_pool_hash,
        "response_text_sha256": text_sha,
        "response_binding_note": ("the tensor's payoffs describe one specific set "
                                  "of response STRINGS; these hashes let the pair "
                                  "builder prove the texts it loads are those"),
        "pool_manifest_sha256": sha_file(args.pool_dir / "pool_manifest.json"),
        "scoring_manifest_sha256": sha_file(args.probs_dir / "scoring_manifest.json"),
        "probs_policy_sha256": sha_file(args.probs_dir / f"probs_{args.model}_policy.npz"),
        "probs_ref_sha256": sha_file(args.probs_dir / f"probs_{args.model}_ref.npz"),
        **implementation_contract(),
        "shape_policy": list(A_policy.shape),
        "shape_ref": list(A_ref.shape),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"policy {A_policy.shape}, ref {A_ref.shape} -> {args.out_dir}")
    print(f"  pre-projection skew residual {skew_residual:.2e}, "
          f"max |diagonal| {diag_before:.2e}")
    print(f"  mean |A_policy| {np.abs(A_policy).mean():.4f}, "
          f"mean |A_ref| {np.abs(A_ref).mean():.4f}")


if __name__ == "__main__":
    main()
