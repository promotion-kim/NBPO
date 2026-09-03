#!/usr/bin/env python3
"""Build the centered preference tensors for the NBPO finite-pool solver.

Aggregates the judged rows of ``judge_pairwise_matrix.py`` into two float64
tensors (saved as a versioned artifact directory):

- ``tensor_policy.npz``: ``A_policy[k, x, i, j] = P_k(y_i > z_j | x) - 1/2``
  (Eq. (2) ``eq:centered``) for current-policy learners ``i`` vs reference
  comparators ``j``;
- ``tensor_ref.npz``: the reference-as-learner tensor used for the
  disagreement point ``d_k = V_{k,beta_k}(mu)`` (Eq. (10)) -- ``d`` is never
  replaced by ``g_k(mu, mu) = 0``.

Each semantic comparison must have BOTH presentation orders judged valid; the
two are converted to P(learner > comparator) and swap-averaged. A missing or
single-order cell is a hard error naming the cell (never imputed as 0.5),
unless ``--allow-single-order-ablation`` is passed for an explicitly labeled
ablation (recorded in the metadata). Reference self-comparisons are the
definitional tie ``P_k(y > y | x) = 1/2`` (Eq. (1)) and are stored as exactly
``A = 0`` -- an identity, not an imputation.

No scalar reward-model scores and no Bradley--Terry reconstruction enter this
path.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.nbpo.nbpo_common import (
    SCHEMA_VERSION,
    load_response_files,
    read_jsonl,
    sha256_file,
    write_json,
)


def aggregate_cells(rows, allow_single_order=False):
    """(prompt, obj, pool, lid, cid) -> centered swap-averaged payoff; errors on gaps."""
    by_cell = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if not r.get("valid"):
            continue  # invalid rows are not completed work
        cell = (r["prompt_id"], r["objective"], r.get("learner_pool", "policy"),
                r["learner_response_id"], r["comparator_response_id"])
        by_cell[cell][r["presentation_order"]].append(float(r["policy_win"]))
    payoff, single_order_cells = {}, []
    for cell, orders in by_cell.items():
        means = {o: sum(v) / len(v) for o, v in orders.items()}
        if len(means) < 2:
            if not allow_single_order:
                single_order_cells.append(cell)
                continue
            p = next(iter(means.values()))
        else:
            p = 0.5 * (means["learner_first"] + means["comparator_first"])
        payoff[cell] = p - 0.5
    if single_order_cells:
        sample = single_order_cells[:10]
        raise RuntimeError(
            f"{len(single_order_cells)} cells have only one presentation order judged "
            f"(both are required outside a labeled ablation); first: {sample}"
        )
    return payoff


def fill_tensor(payoff, objectives, prompt_ids, learner_ids, comparator_ids, pool):
    """Dense (K, X, I, J) tensor; ref self-pairs are the Eq. (1) identity A = 0."""
    K, X, I, J = len(objectives), len(prompt_ids), len(learner_ids), len(comparator_ids)
    A = np.full((K, X, I, J), np.nan, dtype=np.float64)
    for k, obj in enumerate(objectives):
        for x, pid in enumerate(prompt_ids):
            for i, lid in enumerate(learner_ids):
                for j, cid in enumerate(comparator_ids):
                    if pool == "reference" and lid == cid:
                        A[k, x, i, j] = 0.0
                        continue
                    val = payoff.get((pid, obj, pool, lid, cid))
                    if val is not None:
                        A[k, x, i, j] = val
    if np.isnan(A).any():
        missing = list(zip(*np.nonzero(np.isnan(A))))[:10]
        named = [
            (objectives[k], prompt_ids[x], learner_ids[i], comparator_ids[j])
            for k, x, i, j in missing
        ]
        raise RuntimeError(
            f"preference matrix for pool {pool!r} is incomplete: "
            f"{int(np.isnan(A).sum())} missing swap-averaged cells; first: {named} "
            "-- missing comparisons are never imputed as ties"
        )
    return A


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--policy-files", nargs="+", required=True, help="seed=path.json")
    ap.add_argument("--reference-files", nargs="+", required=True, help="seed=path.json")
    ap.add_argument("--objectives", required=True, help="comma-separated, fixes the k order")
    ap.add_argument("--objectives-config", type=Path, required=True,
                    help="rubric YAML (hashed into the artifact for provenance)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--allow-single-order-ablation", action="store_true")
    ap.add_argument("--max-prompts", type=int, default=0)
    args = ap.parse_args()

    objectives = [o.strip() for o in args.objectives.split(",") if o.strip()]
    policy = load_response_files(args.policy_files)
    reference = load_response_files(args.reference_files)
    prompt_ids = sorted(set.intersection(*[set(m) for m in policy.values()],
                                         *[set(m) for m in reference.values()]))
    if args.max_prompts:
        prompt_ids = prompt_ids[:args.max_prompts]
    policy_ids = [f"policy:{s}" for s in sorted(policy)]
    ref_ids = [f"ref:{s}" for s in sorted(reference)]

    rows = read_jsonl(args.verdicts)
    payoff = aggregate_cells(rows, allow_single_order=args.allow_single_order_ablation)
    A_policy = fill_tensor(payoff, objectives, prompt_ids, policy_ids, ref_ids, "policy")
    A_ref = fill_tensor(payoff, objectives, prompt_ids, ref_ids, ref_ids, "reference")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "tensor_policy.npz", A=A_policy)
    np.savez_compressed(args.out_dir / "tensor_ref.npz", A=A_ref)
    judge_models = sorted({r.get("judge_model") for r in rows if r.get("valid")})
    rubric_versions = sorted({r.get("rubric_version") for r in rows if r.get("valid")})
    meta = {
        "schema_version": SCHEMA_VERSION,
        "objectives": objectives,
        "prompt_ids": prompt_ids,
        "policy_learner_ids": policy_ids,
        "comparator_ids": ref_ids,
        "generation_seeds": {
            "policy": {f"policy:{s}": s for s in sorted(policy)},
            "reference": {f"ref:{s}": s for s in sorted(reference)},
        },
        "judge_models": judge_models,
        "rubric_versions": rubric_versions,
        "rubric_config_hash": sha256_file(args.objectives_config),
        "verdicts_hash": sha256_file(args.verdicts),
        "both_orders_required": not args.allow_single_order_ablation,
        "single_order_ablation": bool(args.allow_single_order_ablation),
        "self_pairs": "identity_zero (Eq. (1): P_k(y>y|x)=1/2 exactly; not an imputation)",
        "shape_policy": list(A_policy.shape),
        "shape_ref": list(A_ref.shape),
    }
    write_json(args.out_dir / "meta.json", meta)
    print(f"[build_preference_tensor] policy {A_policy.shape}, ref {A_ref.shape} -> {args.out_dir}")


if __name__ == "__main__":
    main()
