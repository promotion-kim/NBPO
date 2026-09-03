#!/usr/bin/env python3
"""Build NBPO regression pairs with objective-specific opponents (Section 5.2).

For each prompt: all six unordered pairs from the four learner responses
(the full-support proposal ``xi`` of Eq. (26)). For EACH pair and EACH
objective ``k`` one opponent ``z_k`` is sampled from ``nu_update`` of the
solver artifact (the opponent that generated the policy) and SHARED by ``y``
and ``y'`` of that row; other rows of the same prompt and other objectives draw
independently (``opponent_sampling_scope: pair_objective``). Target modes:

- ``sampled`` (manuscript default, Eq. (24) ``eq:binary-target``):
  ``B_k ~ Bernoulli(P_k(y > z_k))``, ``B'_k ~ Bernoulli(P_k(y' > z_k))``,
  ``Z_k = B_k - B'_k``;
- ``rao_blackwell`` (variance-reduced alternative, NOT in the manuscript):
  ``Z_k = P_k(y > z_k) - P_k(y' > z_k)`` (the conditional mean, Eq. (25)).

The stored target is the UNSCALED weighted sum
``nbpo_weighted_z = sum_k lambda_k Z_k`` with RAW lambda from the dual solve --
eta is applied exactly once, by the trainer (Eq. (26)). Pair orientation is
canonical (learner index order); ``flip_pair_row`` flips every ``Z_k`` and the
aggregate consistently and is unit-tested.

The builder is aggregation-agnostic: pointing ``--solver-dir`` at a matched
control's artifact (utilitarian / max-min weights in place of lambda) yields
control training arms through the identical path.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from scripts.nbpo.nbpo_common import load_response_files, sha256_file, write_json

TARGET_MODES = ("sampled", "rao_blackwell")


def flip_pair_row(row: dict) -> dict:
    """Swap the pair orientation, flipping every Z_k and the aggregate target."""
    flipped = dict(row)
    flipped["chosen"], flipped["rejected"] = row["rejected"], row["chosen"]
    flipped["chosen_response_id"], flipped["rejected_response_id"] = (
        row["rejected_response_id"], row["chosen_response_id"])
    flipped["nbpo_z"] = {k: -v for k, v in row["nbpo_z"].items()}
    flipped["nbpo_weighted_z"] = -row["nbpo_weighted_z"]
    return flipped


def build_rows(prompt_ids, objectives, A_policy, nu, lam, betas, policy, ref_seed_of,
               rng, target_mode, meta_ids, provenance):
    """One row per (prompt, unordered learner pair); deterministic given the RNG."""
    K = len(objectives)
    I = A_policy.shape[2]
    policy_ids = meta_ids["policy_learner_ids"]
    comparator_ids = meta_ids["comparator_ids"]
    rows = []
    for x, pid in enumerate(prompt_ids):
        seed0 = policy_ids[0].split(":", 1)[1]
        prompt_text = str(policy[seed0][pid]["prompt"])
        for i1, i2 in itertools.combinations(range(I), 2):
            z, opp = {}, {}
            for k, obj in enumerate(objectives):
                # Eq. (26): draw (y, y') first, THEN one z_k ~ nu*_k for this pair and
                # objective. The same z_k serves both y and y' of the row; other rows
                # of the same prompt and other objectives draw independently.
                j = int(rng.choice(len(comparator_ids), p=nu[k, x]))
                p1 = float(A_policy[k, x, i1, j]) + 0.5
                p2 = float(A_policy[k, x, i2, j]) + 0.5
                if target_mode == "sampled":
                    b1 = float(rng.random() < p1)
                    b2 = float(rng.random() < p2)
                    z[obj] = b1 - b2
                else:  # rao_blackwell
                    z[obj] = p1 - p2
                opp[obj] = comparator_ids[j]
            id1, id2 = policy_ids[i1], policy_ids[i2]
            rows.append({
                "prompt_id": pid,
                "prompt": prompt_text,
                "chosen": str(policy[id1.split(":", 1)[1]][pid]["generated_text"]),
                "rejected": str(policy[id2.split(":", 1)[1]][pid]["generated_text"]),
                "chosen_response_id": id1,
                "rejected_response_id": id2,
                "nbpo_z": z,
                "nbpo_weighted_z": float(sum(lam[k] * z[obj] for k, obj in enumerate(objectives))),
                "lambda_raw": {obj: float(lam[k]) for k, obj in enumerate(objectives)},
                "opponent_response_id": opp,
                "opponent_beta": {obj: float(betas[k]) for k, obj in enumerate(objectives)},
                "opponent_sampling_scope": "pair_objective",
                "target_mode": target_mode,
                **provenance,
            })
    return rows


def build_pairs_from_artifacts(tensor_dir: Path, solver_dir: Path, policy_file_specs,
                               out_dir: Path, target_mode: str = "sampled", seed: int = 42,
                               stage: int = 0, test_prompts: int = 0,
                               split_salt: str = "nbpo-v1") -> dict:
    """Importable core of the CLI (also used by run_nbpo_stage)."""
    if target_mode not in TARGET_MODES:
        raise ValueError(f"target_mode must be one of {TARGET_MODES}, got {target_mode!r}")
    meta = json.loads((tensor_dir / "meta.json").read_text())
    A_policy = np.load(tensor_dir / "tensor_policy.npz")["A"]
    solution = json.loads((solver_dir / "solution.json").read_text())
    # The regression opponent is nu_update (the opponent that generated the
    # policy through Eq. (21)), never nu_final_policy. Verify both files against
    # the hashes the solver recorded so a swapped or stale file cannot pass.
    recorded = solution.get("artifact_hashes") or {}
    for fname in ("nu_update.npz", "nu_final_policy.npz"):
        f = solver_dir / fname
        if not f.exists():
            raise FileNotFoundError(f"solver artifact lacks {fname}; re-run solve_nbpo_dual")
        if fname not in recorded:
            raise ValueError(f"solution.json records no hash for {fname}; refusing to use it")
        got = sha256_file(f)
        if got != recorded[fname]:
            raise ValueError(
                f"{fname} hash {got[:12]} != recorded {recorded[fname][:12]}: the solver "
                "artifact was modified or its opponent files were swapped"
            )
    nu = np.load(solver_dir / "nu_update.npz")["nu"]
    objectives = meta["objectives"]
    lam = np.asarray(solution["lambda_raw"], dtype=np.float64)
    betas = np.asarray(solution["config"]["beta"], dtype=np.float64)
    if nu.shape[:2] != (len(objectives), len(meta["prompt_ids"])):
        raise ValueError("solver nu does not match the tensor artifact's objectives/prompts")

    policy = load_response_files(policy_file_specs)
    provenance = {
        "outer_stage": int(stage),
        "dual_checkpoint": solution["config"]["M"],
        "aggregation": solution["aggregation"],
        "solver_hash": sha256_file(solver_dir / "solution.json"),
        "tensor_policy_hash": solution["input_hashes"]["tensor_policy.npz"],
    }
    rng = np.random.default_rng(seed)
    rows = build_rows(meta["prompt_ids"], objectives, A_policy, nu, lam, betas, policy,
                      None, rng, target_mode, meta, provenance)

    test_ids = set()
    if test_prompts:
        test_ids = set(sorted(
            meta["prompt_ids"],
            key=lambda p: hashlib.sha256(f"{split_salt}|{p}".encode()).hexdigest(),
        )[:test_prompts])
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "test": 0}
    handles = {split: (out_dir / f"pairs_{split}.jsonl").open("w") for split in counts}
    try:
        for row in rows:
            split = "test" if row["prompt_id"] in test_ids else "train"
            handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for h in handles.values():
            h.close()
    summary = {
        "target_mode": target_mode,
        "seed": int(seed),
        "pairs": counts,
        "objectives": objectives,
        "lambda_raw": [float(v) for v in lam],
        "aggregation": solution["aggregation"],
        "note": "nbpo_weighted_z is unscaled: eta is applied exactly once, in the trainer",
        "opponent_sampling_scope": "pair_objective",
        "opponent_source": "nu_update.npz",
        "opponent_source_hash": recorded["nu_update.npz"],
        **provenance,
    }
    write_json(out_dir / "pairs_summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tensor-dir", type=Path, required=True)
    ap.add_argument("--solver-dir", type=Path, required=True)
    ap.add_argument("--policy-files", nargs="+", required=True, help="seed=path.json")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--target-mode", choices=list(TARGET_MODES), default="sampled",
                    help="'sampled' is the manuscript construction (Eq. (24)) and the "
                         "reproduction default; 'rao_blackwell' is a labeled variant")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--test-prompts", type=int, default=0)
    ap.add_argument("--split-salt", default="nbpo-v1")
    args = ap.parse_args()
    summary = build_pairs_from_artifacts(
        args.tensor_dir, args.solver_dir, args.policy_files, args.out_dir,
        target_mode=args.target_mode, seed=args.seed, stage=args.stage,
        test_prompts=args.test_prompts, split_salt=args.split_salt,
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
