# Generated copy -- do not edit by hand.
# Copied from /work/uf4_20260910/code/solve_uf4_targets.py
# source sha256 2ff97b1b496099c83e41155cd5c60a551125768e8ca2aa3170531be12905acb9
# change: OBJECTIVES -> item0..item3
# Reason: in PROSPER's PSC setting the four objectives of a prompt's game
# are four of that prompt's OWN checklist items. The slot names are shared
# so the fixed-K solver contract holds; the criterion text behind each slot
# is prompt-specific and recorded in panel/pros_chosen_items.json. Item k of
# two different prompts is never pooled as one objective.
"""Solve the UF-4 finite-pool teacher and realize its all-pair training data.

One outer stage. The dual weights are fit once on policy_train and are GLOBAL
across prompts; the finite-pool problem is then solved directly per prompt under
those fixed weights. Solving per prompt does not turn the global bargaining
objective into a per-prompt Nash product: the objective is still the one Nash
problem, and the per-prompt solve is how its inner map is evaluated.

policy_dev reuses the train weights unchanged, so dev is a held-out measurement
of the same teacher rather than a second fit.

The certificate records the unprojected stationarity residual, the dual box
activity and the per-objective surplus side by side. A projected residual of
zero at an active box constraint is not evidence that the original problem was
solved, and it is not reported as such. Surplus is never epsilon-clipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mnpo_scripts.nbpo_core import uniform_policy
from mnpo_scripts.nbpo_generic import solve_finite_pool, validate_finite_pool_solution
from mnpo_scripts.nbpo_representations import (
    AdaptiveGameRepresentation, BTRewardRepresentation, FixedReferenceRepresentation,
)
from scripts.nbpo.build_nbpo_pairs import build_rows, load_canonical_artifact
from scripts.nbpo.solve_nbpo_dual import write_generic_solution_artifact

ROOT = Path("/work/uf4_20260910")
OBJECTIVES = ("item0", "item1", "item2", "item3")
POOL = 8


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


CANONICAL_DECIMALS = 10


def quantize_canonical_row(row):
    """Round the solver masses to the loader's precision and rebuild the target.

    prepare_nbpo_dataset reads these rows through a JSON parser that keeps ten
    decimal places, then checks target == log(w_a/c_a) - log(w_b/c_b) to 1e-9.
    Rounding the masses first and deriving the target from the rounded values
    makes the row consistent under that parse instead of only before it. A
    per-prompt Nash solve concentrates mass, so without this the identity
    survives in memory and fails after the parse.
    """
    import math

    for key in ("nbpo_weight_a", "nbpo_weight_b"):
        if key in row:
            row[key] = round(float(row[key]), CANONICAL_DECIMALS)
    if row.get("target_mode") == "canonical_logratio" and "nbpo_weight_a" in row:
        wa, wb = float(row["nbpo_weight_a"]), float(row["nbpo_weight_b"])
        ca = float(row.get("nbpo_center_a", 1.0 / POOL))
        cb = float(row.get("nbpo_center_b", 1.0 / POOL))
        if min(wa, wb) > 0:
            row["nbpo_logratio_target"] = round(
                math.log(wa / ca) - math.log(wb / cb), CANONICAL_DECIMALS)
            row["canonical_target_quantized_decimals"] = CANONICAL_DECIMALS
    return row


def load_scores(score_root, shards):
    """prompt_id -> (A_policy[K,8,8], A_ref[K,8,8], r_bt[K,2,8]), hash-verified.

    r_bt is the frozen BT head's scalar reward for the eight learner and eight
    comparator occurrences. It is loaded for every representation but only the
    bt_reward representation reads it, so nothing else changes.
    """
    scores, manifests = {}, []
    for shard in range(shards):
        directory = Path(score_root) / f"shard{shard}"
        complete = json.loads((directory / f"complete_shard{shard}.json").read_text())
        manifests.append({"shard": shard, "gpm_teacher": complete["gpm_teacher"],
                          "bt_teacher": complete["bt_teacher"],
                          "reference_construction": complete["reference_construction"]})
        for path in sorted(directory.glob("chunk*.npz")):
            meta = json.loads(path.with_suffix("").with_suffix(".manifest.json").read_text()) \
                if path.with_suffix("").with_suffix(".manifest.json").exists() else \
                json.loads((directory / (path.stem + ".manifest.json")).read_text())
            if file_hash(path) != meta["sha256"]:
                raise ValueError(f"Score chunk hash mismatch: {path}")
            arrays = np.load(path, allow_pickle=True)
            pids = [str(p) for p in arrays["prompt_ids"]]
            A, Aref, Rbt = arrays["A_policy"], arrays["A_ref"], arrays["r_bt"]
            if A.shape[0] != len(OBJECTIVES) or A.shape[2:] != (POOL, POOL):
                raise ValueError(f"Unexpected score tensor shape in {path}: {A.shape}")
            if Rbt.shape[0] != len(OBJECTIVES) or Rbt.shape[2:] != (2, POOL):
                raise ValueError(f"Unexpected BT reward shape in {path}: {Rbt.shape}")
            for index, pid in enumerate(pids):
                if pid in scores:
                    raise ValueError(f"Duplicate scored prompt {pid}")
                scores[pid] = (A[:, index], Aref[:, index], Rbt[:, index])
    identities = {object_hash(m["gpm_teacher"]) for m in manifests}
    if len(identities) != 1:
        raise ValueError("Score shards disagree on the frozen GPM teacher")
    return scores, manifests


def load_pool(pool_root, shards):
    """prompt_id -> {role: {index: event}}, from the immutable pool chunks."""
    pool = {}
    settings = None
    split_provenance = []
    # "shard" is the shard index; splits_dir and split_files_sha256 record WHICH
    # prompts a shard drew, not HOW they were sampled. This pool is assembled
    # from two prompt batches under byte-identical sampling settings, so those
    # three are collected rather than compared. Everything else, including any
    # field added later, still has to match exactly.
    PROVENANCE_KEYS = ("shard", "splits_dir", "split_files_sha256")
    for shard in range(shards):
        directory = Path(pool_root) / f"shard{shard}"
        shard_settings = json.loads((directory / "settings.json").read_text())
        comparable = {k: v for k, v in shard_settings.items()
                      if k not in PROVENANCE_KEYS}
        split_provenance.append({k: shard_settings.get(k) for k in PROVENANCE_KEYS})
        if settings is None:
            settings = comparable
        elif settings != comparable:
            differing = sorted(k for k in set(settings) | set(comparable)
                               if settings.get(k) != comparable.get(k))
            raise ValueError("Pool shards were generated under different sampling "
                             "settings; differing fields: %s" % differing)
        for path in sorted(directory.glob("chunk*.jsonl")):
            manifest = json.loads((directory / (path.stem + ".manifest.json")).read_text())
            if file_hash(path) != manifest["sha256"]:
                raise ValueError(f"Pool chunk hash mismatch: {path}")
            with path.open() as stream:
                for line in stream:
                    event = json.loads(line)
                    pool.setdefault(event["prompt_id"], {}).setdefault(
                        event["role"], {})[event["sample_index"]] = event
    settings = dict(settings or {}, shard_prompt_provenance=split_provenance)
    return pool, settings


def certificate_record(result, solution_certificate):
    """Everything needed to tell 'solved' from 'at a bound' from 'unresolved'."""
    weights = result.weights.numpy()
    return {"certified": bool(solution_certificate.get("certified")),
            "independent_stationarity_inf": solution_certificate.get("independent_stationarity_inf"),
            "unprojected_kkt_residual": result.kkt_residual,
            "projected_kkt_residual": result.projected_kkt_residual,
            "lambda_at_lower_bound": result.lambda_at_lower_bound,
            "lambda_at_upper_bound": result.lambda_at_upper_bound,
            "box_active": bool(result.lambda_at_lower_bound or result.lambda_at_upper_bound),
            "fixed_point_residual": result.fixed_point_residual,
            "control_residual": result.control_residual,
            "lambda_raw": [float(v) for v in weights],
            "lambda_l1": float(weights.sum()),
            "surplus_per_objective": {name: float(v) for name, v
                                      in zip(OBJECTIVES, result.surplus)},
            "min_surplus": float(result.surplus.min()),
            "all_surplus_positive": bool((result.surplus > 0).all()),
            "V": [float(v) for v in result.V], "d": [float(v) for v in result.d],
            "reading": ("A projected residual of zero while the dual box is active does not "
                        "certify the original problem; read box_active with "
                        "unprojected_kkt_residual before calling this solved.")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", default="v1")
    ap.add_argument("--train-scores", default=str(ROOT / "scores/v1"))
    ap.add_argument("--train-pool", default=str(ROOT / "pools/v1"))
    ap.add_argument("--dev-scores", default=str(ROOT / "scores/dev_v1"))
    ap.add_argument("--dev-pool", default=str(ROOT / "pools/dev_v1"))
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--aggregation", default="nash",
                    choices=("nash", "utilitarian", "absolute_maxmin", "kalai_smorodinsky"))
    ap.add_argument("--weight-l1", type=float,
                    help="Matched L1 norm for a non-Nash rule. Must come from the UF-4 Nash "
                         "dual, never from the SafeRLHF panel.")
    ap.add_argument("--weight-l1-from", type=Path,
                    help="Read the matched norm from this UF-4 Nash targets/<name>/complete.json. "
                         "Safer than typing the number: it cannot pick up the SafeRLHF panel's.")
    ap.add_argument("--representation", default="adaptive_game",
                    choices=("adaptive_game", "fixed_reference", "bt_reward"),
                    help="fixed_reference freezes the comparator at mu: the beta -> infinity "
                         "limit, where q carries no learner dependence and the proximal solve "
                         "is exact in one map. bt_reward replaces the game value by a frozen "
                         "scalar Bradley-Terry reward per objective, so there is no opponent "
                         "at all; it changes the preference representation, not the rule.")
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=min(32, len(os.sched_getaffinity(0))))
    ap.add_argument("--max-dual-calls", type=int, default=200)
    ap.add_argument("--probability-floor", type=float, default=1e-12)
    ap.add_argument("--skip-dataset", action="store_true")
    args = ap.parse_args()
    if args.weight_l1_from is not None:
        if args.weight_l1 is not None:
            raise ValueError("Give --weight-l1 or --weight-l1-from, not both")
        source = json.loads(args.weight_l1_from.read_text())
        if source.get("aggregation") != "nash":
            raise ValueError("--weight-l1-from must point at a Nash solution")
        args.weight_l1 = float(source["splits"]["train"]["certificate"]["lambda_l1"])
        print(json.dumps({"matched_weight_l1": args.weight_l1,
                          "from": str(args.weight_l1_from)}), flush=True)
    if args.aggregation == "nash":
        if args.weight_l1 is not None:
            raise ValueError("The Nash dual picks its own weight scale")
    elif args.weight_l1 is None or not args.weight_l1 > 0:
        raise ValueError("A non-Nash rule needs a positive --weight-l1 from the UF-4 Nash dual")

    torch.set_num_threads(1)
    start = time.monotonic()
    out = ROOT / "targets" / args.out_name
    out.mkdir(parents=True, exist_ok=False)
    split_dir = ROOT / "splits" / args.splits

    sources = {"train": (args.train_scores, args.train_pool, "policy_train"),
               "dev": (args.dev_scores, args.dev_pool, "policy_dev")}
    shared_meta, training, outputs = None, None, {}
    # The BT reward scale is fit once, on the policy_train comparator pool, and
    # reused unchanged on dev -- the same discipline the dual weights follow, so
    # dev stays a held-out measurement rather than a second fit.
    bt_norm = None
    for split, (score_root, pool_root, split_file) in sources.items():
        split_start = time.monotonic()
        scores, score_manifests = load_scores(score_root, args.shards)
        pool, pool_settings = load_pool(pool_root, args.shards)
        # Read with the file iterator, never str.splitlines(). JSON does not escape
        # U+2028/U+2029/U+0085, splitlines() breaks on them, and 31 of the 10,000
        # policy_train instructions contain U+2028.
        with (split_dir / f"{split_file}.jsonl").open() as stream:
            pids = [json.loads(line)["prompt_id"] for line in stream if line.strip()]
        missing = [pid for pid in pids if pid not in scores or pid not in pool]
        if missing:
            raise ValueError(f"{split}: {len(missing)} prompts have no scores or no pool")
        for pid in pids:
            roles = pool[pid]
            if sorted(roles) != ["comparator", "learner"] or any(
                    sorted(roles[r]) != list(range(POOL)) for r in roles):
                raise ValueError(f"{split}: incomplete pool for {pid}")

        A = np.stack([scores[pid][0] for pid in pids], axis=1)
        Aref = np.stack([scores[pid][1] for pid in pids], axis=1)
        skew = float(np.abs(Aref + np.swapaxes(Aref, -1, -2)).max())
        if skew > 0:
            raise ValueError(f"{split}: shared-pool reference tensor is not skew symmetric ({skew})")

        Rbt = np.stack([scores[pid][2] for pid in pids], axis=1)     # (K, X, 2, POOL)
        if args.representation == "bt_reward":
            r_learner_raw, r_ref_raw = Rbt[:, :, 0, :], Rbt[:, :, 1, :]
            if bt_norm is None:
                centre = r_ref_raw.reshape(len(OBJECTIVES), -1).mean(axis=1)
                scale = r_ref_raw.reshape(len(OBJECTIVES), -1).std(axis=1, ddof=1)
                if not np.all(scale > 0):
                    raise ValueError("A BT objective has zero spread on the reference pool")
                bt_norm = {"objectives": list(OBJECTIVES),
                           "mu_k_ref": [float(x) for x in centre],
                           "sigma_k_ref": [float(x) for x in scale],
                           "fitted_on": "policy_train comparator pool, all 8 occurrences",
                           "reused_on": "policy_dev, never refit"}
            centre = np.asarray(bt_norm["mu_k_ref"])[:, None, None]
            scale = np.asarray(bt_norm["sigma_k_ref"])[:, None, None]
            r_learner = (r_learner_raw - centre) / scale
            r_reference = (r_ref_raw - centre) / scale

        tensor_dir = out / split / "tensor"
        tensor_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tensor_dir / "tensor_policy.npz", A=A)
        np.savez_compressed(tensor_dir / "tensor_ref.npz", A=Aref)
        if shared_meta is None:
            # The trainer recomputes tokenizer_content_hashes on the tokenizer it
            # loads and refuses a dataset whose provenance disagrees. Compute the
            # real pair here, with the same pad-token fallback the trainer applies,
            # instead of putting a neighbouring hash in the field.
            from transformers import AutoTokenizer
            from mnpo_scripts.precompute_provenance import tokenizer_content_hashes
            tokenizer = AutoTokenizer.from_pretrained(pool_settings["model"],
                                                      local_files_only=True)
            pad_fallback = tokenizer.pad_token_id is None
            if pad_fallback:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            token_hashes = tokenizer_content_hashes(tokenizer)
            if token_hashes["chat_template_hash"] != pool_settings["chat_template_sha256"]:
                raise ValueError("Pool chat template does not match the training tokenizer's")
            shared_meta = {"model_revision": pool_settings["model_revision"],
                           **token_hashes,
                           "training_pad_token_fallback_to_eos": pad_fallback,
                           "pool_settings_sha256": object_hash(pool_settings),
                           "teacher_manifest_sha256": object_hash(score_manifests),
                           "gpm_teacher": score_manifests[0]["gpm_teacher"],
                           "bt_teacher": score_manifests[0]["bt_teacher"]}
        meta = {"prompt_ids": pids, "objectives": list(OBJECTIVES),
                "reference_construction": "shared_pool",
                "policy_learner_ids": [f"policy:{i}" for i in range(POOL)],
                "comparator_ids": [f"ref:{j}" for j in range(POOL)],
                "split": split, **shared_meta}
        write_json(tensor_dir / "meta.json", meta)

        if args.representation == "adaptive_game":
            rep = AdaptiveGameRepresentation(
                torch.from_numpy(A), torch.from_numpy(Aref),
                uniform_policy(len(pids), POOL),
                torch.full((len(OBJECTIVES),), args.beta, dtype=torch.float64),
                reference_construction="shared_pool")
        elif args.representation == "fixed_reference":
            rep = FixedReferenceRepresentation(
                torch.from_numpy(A), torch.from_numpy(Aref),
                uniform_policy(len(pids), POOL),
                reference_construction="shared_pool")
        elif args.representation == "bt_reward":
            rep = BTRewardRepresentation(
                torch.from_numpy(r_learner), torch.from_numpy(r_reference),
                uniform_policy(len(pids), POOL), normalization=bt_norm)
        else:
            raise ValueError(f"Unhandled representation {args.representation!r}")
        print(json.dumps({"phase": "solve", "split": split, "prompts": len(pids),
                          "objectives": len(OBJECTIVES), "workers": args.workers,
                          "dual_fit": training is None}), flush=True)
        result = None
        try:
            result = solve_finite_pool(
                rep, args.aggregation, eta=args.eta, inner_solver="exact", dual_solver="root",
                dual_tol=1e-10, M=args.max_dual_calls, inner_workers=args.workers,
                probability_floor=args.probability_floor,
                weight_l1=(None if training is not None else args.weight_l1),
                fixed_weights=None if training is None else training.weights, log_every=10)
            certificate = validate_finite_pool_solution(result)
        except Exception as error:                          # noqa: BLE001
            failure = {"status": "infeasible_or_unresolved", "error": str(error)[:2000],
                       "split": split, "seconds": time.monotonic() - split_start,
                       "usable_for_training": False}
            if result is not None:
                failure["uncertified"] = certificate_record(
                    result, validate_finite_pool_solution(result, require_optimality=False))
                np.savez_compressed(out / split / "uncertified_policy.npz", pi=result.pi.numpy())
            write_json(out / split / "solver_unresolved.json", failure)
            raise

        record = certificate_record(result, certificate)
        solver_dir = out / split / "solver"
        extra = {"split": split, **shared_meta,
                 "lambda_source": ("policy_train only" if split == "train"
                                   else "fixed policy_train solution"),
                 "beta": (args.beta if args.representation == "adaptive_game" else None),
                 "representation": args.representation, "eta": args.eta,
                 "bt_reward_normalization": bt_norm,
                 "dual_weights_scope": "global across prompts",
                 "certificate_reading": record["reading"]}
        if training is not None:
            extra["training_solution_sha256"] = file_hash(out / "train/solver/solution.json")
        solution = write_generic_solution_artifact(
            solver_dir, result, meta,
            {name: file_hash(tensor_dir / name) for name in
             ("tensor_policy.npz", "tensor_ref.npz", "meta.json")},
            tensor_dir, 0, False, extra)
        if training is None:
            training = result
        canonical = load_canonical_artifact(solver_dir, solution, expected_prompt_ids=pids,
                                            expected_representation=args.representation,
                                            expected_aggregation=args.aggregation)
        solver_hash = file_hash(solver_dir / "solution.json")
        betas = (np.full(len(OBJECTIVES), args.beta)
                 if args.representation == "adaptive_game" else None)

        def pair_rows():
            for x, pid in enumerate(pids):
                learners = {str(i): {pid: {**pool[pid]["learner"][i],
                                           "generated_text": pool[pid]["learner"][i]["response"]}}
                            for i in range(POOL)}
                rows = build_rows(
                    [pid], list(OBJECTIVES), A[:, x:x + 1], result.nu_update.numpy()[:, x:x + 1],
                    result.weights.numpy(), betas, learners, None,
                    np.random.default_rng(42), "canonical_logratio", meta,
                    {"solver_artifact_sha256": solver_hash, "solver_hash": solver_hash,
                     "target_artifact_hash": solution["artifact_hashes"]["target_log_ratio.npz"],
                     "representation": args.representation, "aggregation": args.aggregation,
                     "split": split, "panel": "UF-4", **shared_meta},
                    canonical_data={key: values[x:x + 1] for key, values in canonical.items()})
                for row in rows:
                    a, b = row["chosen_candidate_index"], row["rejected_candidate_index"]
                    row["chosen_response_id"] = pool[pid]["learner"][a]["candidate_id"]
                    row["rejected_response_id"] = pool[pid]["learner"][b]["candidate_id"]
                    yield row

        pair_path = out / "pairs" / f"{split}.jsonl"
        write_jsonl(pair_path, pair_rows())
        capped = sum(pool[pid]["learner"][i]["capped_horizon"]
                     for pid in pids for i in range(POOL))
        outputs[split] = {"n_prompts": len(pids), "n_pairs": len(pids) * 28,
                          "pairs_path": str(pair_path), "pairs_sha256": file_hash(pair_path),
                          "solver_solution_sha256": solver_hash,
                          "certificate": record,
                          "learner_capped_fraction": capped / (len(pids) * POOL),
                          "reference_skew_residual": skew,
                          "seconds": time.monotonic() - split_start}
        write_json(out / split / "complete.json", outputs[split])
        print(json.dumps({"split": split, "n_pairs": len(pids) * 28,
                          "seconds": round(outputs[split]["seconds"], 1),
                          "lambda_raw": record["lambda_raw"],
                          "surplus": record["surplus_per_objective"],
                          "min_surplus": record["min_surplus"],
                          "box_active": record["box_active"],
                          "unprojected_kkt": record["unprojected_kkt_residual"]}), flush=True)

    provenance = {**shared_meta, "objectives": list(OBJECTIVES), "panel": "UF-4",
                  "aggregation": args.aggregation, "beta": args.beta, "eta": args.eta,
                  "weight_l1": args.weight_l1, "splits": outputs,
                  "train_pool_sha256": object_hash(sorted(outputs)),
                  "dev_pool_sha256": object_hash(sorted(outputs))}
    write_json(out / "dataset_provenance.json", provenance)

    dataset_manifest = None
    if not args.skip_dataset:
        dataset_out = ROOT / "datasets" / args.out_name
        env = dict(os.environ, HF_DATASETS_CACHE=str(out / "arrow_cache"),
                   PYTHONDONTWRITEBYTECODE="1")
        completed = subprocess.run(
            [sys.executable, "-m", "mnpo_scripts.prepare_nbpo_dataset",
             "--train", str(out / "pairs/train.jsonl"), "--dev", str(out / "pairs/dev.jsonl"),
             "--output", str(dataset_out), "--provenance", str(out / "dataset_provenance.json")],
            env=env, capture_output=True, text=True)
        if completed.returncode:
            write_json(out / "dataset_materialization_failure.json",
                       {"returncode": completed.returncode, "stdout": completed.stdout[-4000:],
                        "stderr": completed.stderr[-4000:]})
            raise RuntimeError("Dataset materialization failed; diagnostic preserved")
        dataset_manifest = file_hash(dataset_out / "precompute_manifest.json")

    write_json(out / "complete.json", {
        "splits": outputs, "aggregation": args.aggregation, "beta": args.beta, "eta": args.eta,
        "weight_l1": args.weight_l1,
        "dataset_path": None if args.skip_dataset else str(ROOT / "datasets" / args.out_name),
        "dataset_manifest_sha256": dataset_manifest,
        "seconds": time.monotonic() - start, "source_sha256": file_hash(__file__)})
    print(json.dumps({"targets": str(out), "dataset_manifest_sha256": dataset_manifest,
                      "seconds": round(time.monotonic() - start, 1)}), flush=True)


if __name__ == "__main__":
    main()
