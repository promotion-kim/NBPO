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

from scripts.nbpo.nbpo_common import (
    implementation_contract,
    load_response_files,
    response_pool_hash,
    sha256_file,
    sha256_text,
    write_json,
)

TARGET_MODES = ("sampled", "rao_blackwell")


def _array_sha256(arr) -> str:
    """sha256 of an array's exact float64 bytes -- matches the solver's hash."""
    import hashlib

    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(arr), dtype=np.float64).tobytes()).hexdigest()


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
                # Response IDS are not response TEXT: these pin the exact strings
                # this row was built from, so a later pool swap is detectable.
                "chosen_text_sha256": sha256_text(
                    str(policy[id1.split(":", 1)[1]][pid]["generated_text"])),
                "rejected_text_sha256": sha256_text(
                    str(policy[id2.split(":", 1)[1]][pid]["generated_text"])),
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


def verify_solver_input_chain(tensor_dir: Path, solver_dir: Path, solution: dict,
                              reproduction_mode_required: bool = True) -> dict:
    """Prove the tensors on disk ARE the solver's inputs, and the opponents its outputs.

    Verifying only the ``nu`` hashes left the biggest hole open: nothing showed
    that ``tensor_policy.npz`` / ``tensor_ref.npz`` / ``meta.json`` currently in
    ``tensor_dir`` are the files the dual was solved against. A tensor swapped
    for one of the same shape produced targets from lambda that never saw it.

    Every file is hashed HERE, from its current bytes, and compared with what
    the solver recorded. Nothing is copied from the solution into the pair
    artifact without being independently recomputed first.
    """
    tensor_dir, solver_dir = Path(tensor_dir), Path(solver_dir)
    recorded_inputs = solution.get("input_hashes") or {}
    recorded_artifacts = solution.get("artifact_hashes") or {}
    problems, verified = [], {}
    for fname in ("tensor_policy.npz", "tensor_ref.npz", "meta.json"):
        f = tensor_dir / fname
        if not f.exists():
            problems.append(f"tensor artifact lacks {fname}")
            continue
        if fname not in recorded_inputs:
            problems.append(f"solution.json records no input hash for {fname}")
            continue
        got = sha256_file(f)                       # recomputed, never copied
        if got != recorded_inputs[fname]:
            problems.append(
                f"{fname} hash {got[:12]} != solver input {recorded_inputs[fname][:12]}: "
                "these tensors are NOT the ones the dual was solved against")
            continue
        verified[fname] = got
    for fname in ("nu_update.npz", "nu_final_policy.npz", "update_source_pi.npz"):
        f = solver_dir / fname
        if not f.exists():
            problems.append(f"solver artifact lacks {fname}; re-run solve_nbpo_dual")
            continue
        if fname not in recorded_artifacts:
            problems.append(f"solution.json records no hash for {fname}; refusing to use it")
            continue
        got = sha256_file(f)
        if got != recorded_artifacts[fname]:
            problems.append(
                f"{fname} hash {got[:12]} != recorded {recorded_artifacts[fname][:12]}: the "
                "solver artifact was modified or its opponent files were swapped")
            continue
        verified[fname] = got
    # The two opponents may legitimately be IDENTICAL -- a zero-payoff game, a
    # symmetric one, or a policy already at the fixed point all give nu at pi_t
    # equal to nu at the final policy. Byte inequality is therefore not a valid
    # integrity signal; the artifacts are distinguished by declared metadata
    # instead, and both hashes are still verified above.
    kinds = solution.get("opponent_artifacts") or {}
    if kinds:
        # The Eq. (26) opponent may come from the proximal centre, a warm-start
        # iterate or a later fixed-point iterate -- the released R=1 dual solve
        # warm-starts, so proximal_centre is NOT the only valid answer. What is
        # checked is that the declared source is a real one, that the named
        # policy artifact hashes to the declared value, and that the two
        # opponents have not traded roles.
        valid_sources = {"proximal_centre", "warm_start_iterate", "fixed_point_iterate"}
        expected_use = {"nu_update.npz": "eq26_target",
                        "nu_final_policy.npz": "diagnostics"}
        for fname, want_use in expected_use.items():
            declared = kinds.get(fname) or {}
            if declared.get("used_for") != want_use:
                problems.append(
                    f"{fname} declares used_for={declared.get('used_for')!r}, expected "
                    f"{want_use!r}: the Eq. (26) opponent and the diagnostic one have "
                    "been swapped")
            if declared.get("artifact_kind") != "regularized_opponent":
                problems.append(f"{fname} declares artifact_kind="
                                f"{declared.get('artifact_kind')!r}")
        update_meta = kinds.get("nu_update.npz") or {}
        source_kind = update_meta.get("source_policy")
        if source_kind not in valid_sources:
            problems.append(
                f"nu_update.npz declares source_policy={source_kind!r}, not one of "
                f"{sorted(valid_sources)}")
        artifact_name = update_meta.get("source_policy_artifact")
        declared_hash = update_meta.get("source_policy_hash")
        if artifact_name is None or declared_hash is None:
            problems.append(
                "nu_update.npz names no source_policy_artifact/source_policy_hash; the "
                "policy it was built from cannot be identified")
        else:
            src = solver_dir / artifact_name
            if not src.exists():
                problems.append(f"nu_update source policy artifact missing: {src}")
            else:
                arr = np.load(src)
                key = "pi" if "pi" in arr else arr.files[0]
                got = _array_sha256(arr[key])
                if got != declared_hash:
                    problems.append(
                        f"{artifact_name} content hash {got[:12]} != declared "
                        f"source_policy_hash {declared_hash[:12]}: the recorded source "
                        "policy is not the one on disk")
                # A declared proximal_centre must genuinely be the proximal centre.
                if source_kind == "proximal_centre" and artifact_name != "update_source_pi.npz":
                    problems.append(
                        "proximal_centre is declared but the source artifact is "
                        f"{artifact_name!r}")
    elif reproduction_mode_required:
        problems.append(
            "solution.json declares no opponent_artifacts metadata; nu_update and "
            "nu_final_policy cannot be told apart by role (re-run solve_nbpo_dual)")
    if problems:
        raise ValueError("solver -> pair artifact hash chain broken: " + "; ".join(problems))
    return verified


def verify_response_pool_against_tensor(meta: dict, policy, reproduction_mode: bool,
                                        learner_manifest_sha256=None) -> dict:
    """Prove the loaded response texts ARE the pool the tensor was built from.

    The tensor's payoffs -- and therefore lambda and nu -- describe one specific
    set of response STRINGS. This function loaded them again from
    ``policy_file_specs``, which only guaranteed the same response IDS: the same
    seed files can be regenerated with new text and every previous hash still
    matched, so the pair rows would carry text that no judged comparison ever
    saw. The current files are re-hashed here and compared with what the tensor
    recorded, string by string.
    """
    recorded_pool = meta.get("learner_response_pool_hash")
    recorded_texts = (meta.get("response_text_sha256") or {}).get("policy")
    if recorded_pool is None or recorded_texts is None:
        if reproduction_mode:
            raise ValueError(
                "tensor meta.json records no learner_response_pool_hash/response_text_sha256; "
                "the pair texts cannot be shown to be the pool the tensor was built from "
                "(rebuild the tensor with the current build_preference_tensor)")
        return {"verified": False, "reason": "legacy tensor artifact"}
    current_pool = response_pool_hash(policy)
    problems = []
    if current_pool != recorded_pool:
        problems.append(
            f"learner response pool hash {current_pool[:12]} != tensor "
            f"{recorded_pool[:12]}: these are not the responses that were judged")
    for rid, per_prompt in sorted(recorded_texts.items()):
        seed = rid.split(":", 1)[1]
        if seed not in policy:
            problems.append(f"seed {seed} missing from the loaded response files")
            continue
        for pid, want in sorted(per_prompt.items()):
            row = policy[seed].get(pid)
            if row is None:
                problems.append(f"{rid}/{pid} missing from the loaded response files")
                continue
            got = sha256_text(str(row.get("generated_text", "")))
            if got != want:
                problems.append(
                    f"{rid}/{pid} response text {got[:12]} != tensor {want[:12]}")
    if (learner_manifest_sha256 is not None
            and meta.get("learner_manifest_sha256") is not None
            and learner_manifest_sha256 != meta["learner_manifest_sha256"]):
        problems.append(
            f"learner manifest {learner_manifest_sha256[:12]} != the one the tensor was "
            f"built from {meta['learner_manifest_sha256'][:12]}")
    if problems:
        raise ValueError("pair texts are not the tensor's response pool: "
                         + "; ".join(problems[:8])
                         + (f" (+{len(problems) - 8} more)" if len(problems) > 8 else ""))
    return {"verified": True, "learner_response_pool_hash": current_pool,
            "learner_manifest_sha256": meta.get("learner_manifest_sha256")}


def build_pairs_from_artifacts(tensor_dir: Path, solver_dir: Path, policy_file_specs,
                               out_dir: Path, target_mode: str = "sampled", seed: int = 42,
                               stage: int = 0, test_prompts: int = 0,
                               split_salt: str = "nbpo-v1",
                               reproduction_mode: bool = True,
                               learner_manifest_sha256=None) -> dict:
    """Importable core of the CLI (also used by run_nbpo_stage)."""
    if target_mode not in TARGET_MODES:
        raise ValueError(f"target_mode must be one of {TARGET_MODES}, got {target_mode!r}")
    solution = json.loads((solver_dir / "solution.json").read_text())
    verified = verify_solver_input_chain(tensor_dir, solver_dir, solution,
                                        reproduction_mode_required=reproduction_mode)
    meta = json.loads((tensor_dir / "meta.json").read_text())
    A_policy = np.load(tensor_dir / "tensor_policy.npz")["A"]
    # The regression opponent is nu_update (the opponent that generated the
    # policy through Eq. (21)), never nu_final_policy.
    nu = np.load(solver_dir / "nu_update.npz")["nu"]
    objectives = meta["objectives"]
    lam = np.asarray(solution["lambda_raw"], dtype=np.float64)
    betas = np.asarray(solution["config"]["beta"], dtype=np.float64)
    if nu.shape[:2] != (len(objectives), len(meta["prompt_ids"])):
        raise ValueError("solver nu does not match the tensor artifact's objectives/prompts")

    policy = load_response_files(policy_file_specs, reproduction_mode=reproduction_mode)
    pool_check = verify_response_pool_against_tensor(
        meta, policy, reproduction_mode, learner_manifest_sha256)
    provenance = {
        "outer_stage": int(stage),
        "dual_checkpoint": solution["config"]["M"],
        "aggregation": solution["aggregation"],
        # Every hash below was recomputed from the current bytes by
        # verify_solver_input_chain, not read out of solution.json.
        "solver_hash": sha256_file(solver_dir / "solution.json"),
        "tensor_response_pool_hash": meta.get("learner_response_pool_hash"),
        "learner_manifest_sha256": pool_check.get("learner_manifest_sha256"),
        "tensor_policy_hash": verified["tensor_policy.npz"],
        "tensor_ref_hash": verified["tensor_ref.npz"],
        "tensor_meta_hash": verified["meta.json"],
        "nu_update_hash": verified["nu_update.npz"],
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
        "opponent_source_hash": verified["nu_update.npz"],
        "rng_seed": int(seed),
        "verified_input_hashes": verified,
        **implementation_contract(
            dual_iterations=solution["config"].get("M"),
            fixed_point_steps=solution["config"].get("R")),
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
    ap.add_argument("--no-reproduction-mode", action="store_true",
                    help="accept a legacy tensor artifact that does not record its "
                         "response-pool text hashes (the pair texts then cannot be "
                         "proven to be the judged pool)")
    ap.add_argument("--learner-manifest-sha256", default=None,
                    help="expected learner-pool manifest hash; compared with the one the "
                         "tensor was built from")
    args = ap.parse_args()
    summary = build_pairs_from_artifacts(
        args.tensor_dir, args.solver_dir, args.policy_files, args.out_dir,
        target_mode=args.target_mode, seed=args.seed, stage=args.stage,
        test_prompts=args.test_prompts, split_salt=args.split_salt,
        reproduction_mode=not args.no_reproduction_mode,
        learner_manifest_sha256=args.learner_manifest_sha256,
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
