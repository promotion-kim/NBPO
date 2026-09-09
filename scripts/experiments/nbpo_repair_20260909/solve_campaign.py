"""Certify one train-only Nash teacher and realize its shared immutable all-pair data."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from mnpo_scripts.nbpo_core import uniform_policy
from mnpo_scripts.nbpo_generic import solve_finite_pool, validate_finite_pool_solution
from mnpo_scripts.nbpo_representations import AdaptiveGameRepresentation
from scripts.nbpo.build_nbpo_pairs import build_rows, load_canonical_artifact
from scripts.nbpo.solve_nbpo_dual import write_generic_solution_artifact
from scripts.experiments.nbpo_repair_20260909.common import (
    digest, file_hash, object_hash, read_jsonl, write_json, write_jsonl,
)

OBJECTIVES = ["helpfulness", "harmlessness"]
ROLES = ("learner", "comparator", "reference_learner")
REFUSAL_DIAGNOSTIC = re.compile(
    r"^\s*(?:i(?:'m| am) sorry|i (?:cannot|can't|won't|am unable)|sorry[,\s]|as an ai)", re.I)


def load_campaign_inputs(root, shards):
    """Verify every score/pool shard before exposing ordered prompt occurrences."""
    root = Path(root)
    split_manifest = json.loads((root / "splits/manifest.json").read_text())
    splits = {split: read_jsonl(root / "splits" / f"{split}.jsonl")
              for split in ("train", "dev", "test")}
    for split, rows in splits.items():
        if file_hash(root / "splits" / f"{split}.jsonl") != split_manifest["splits"][split]["file_sha256"]:
            raise ValueError(f"Changed split artifact: {split}")
        if len(rows) != split_manifest["counts"][split]:
            raise ValueError(f"Wrong split count: {split}")
    expected = {row["prompt_id"]: row for rows in splits.values() for row in rows}
    if len(expected) != sum(map(len, splits.values())):
        raise ValueError("Prompt groups intersect across policy splits")
    learners, scores, shard_manifests = {}, {}, []
    teacher_identity = None
    pool_hashes_by_split = defaultdict(dict)
    for shard in range(shards):
        folder = root / "scores" / f"shard{shard}"
        settings_path = root / "pools" / f"shard{shard}" / "settings.json"
        settings = json.loads(settings_path.read_text())
        required_settings = {"temperature": 1., "top_p": 1., "top_k": -1,
                             "repetition_penalty": 1., "presence_penalty": 0., "frequency_penalty": 0.,
                             "max_tokens": 1024, "max_prompt_tokens": 1024, "max_model_len": 2048,
                             "occurrence_mass": .125, "duplicate_occurrences_retained": True}
        if any(settings.get(key) != value for key, value in required_settings.items()):
            raise ValueError("Generation settings do not match the fixed raw-policy protocol")
        if settings["inventory_sha256"] != file_hash(root / "provenance/inventory.json") or settings["split_manifest_sha256"] != file_hash(root / "splits/manifest.json"):
            raise ValueError("Generation settings have changed inventory/split provenance")
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest["objectives"] != OBJECTIVES or manifest["seeds"] != [41, 42, 43]:
            raise ValueError("Unexpected teacher objective order or ensemble members")
        if not manifest.get("reference_tensor_not_symmetrized") or manifest.get("reference_construction") != "independent_reference_learner8_against_comparator8":
            raise ValueError("Reference-as-learner tensor must use its independent8 against actual comparator8")
        if manifest.get("calibration_applied_before_seed_average") is not True:
            raise ValueError("Frozen calibration must precede ensemble averaging")
        identity = object_hash({"models": manifest["teacher_models"],
                                "calibration_sha256": manifest["calibration_sha256"],
                                "max_teacher_tokens": manifest["max_teacher_tokens"]})
        if teacher_identity is not None and identity != teacher_identity:
            raise ValueError("Teacher checkpoint/calibration/context mismatch between shards")
        teacher_identity = identity
        score_file = folder / "probabilities.npz"
        if file_hash(score_file) != manifest["probabilities_sha256"]:
            raise ValueError("Score tensor hash mismatch")
        arrays = np.load(score_file)
        pids = manifest["prompt_ids"]
        pid_set = set(pids)
        shape = (3, 2, len(pids), 8, 8)
        for key in ("policy", "reference"):
            if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                raise ValueError("Invalid score tensor shape/nonfinite values")
            if np.any(arrays[key] < 0) or np.any(arrays[key] > 1):
                raise ValueError("Preference probabilities outside [0,1]")
        # Each checkpoint's calibrated probability is averaged once. The
        # independent reference tensor is never antisymmetrized or zeroed.
        policy, reference = arrays["policy"].mean(0) - .5, arrays["reference"].mean(0) - .5
        for index, pid in enumerate(pids):
            if pid in scores or pid not in expected:
                raise ValueError("Duplicate or unexpected scored prompt")
            if manifest["prompt_splits"][pid] != expected[pid]["split"]:
                raise ValueError("Scored prompt split mismatch")
            scores[pid] = (policy[:, index], reference[:, index])
        seen = defaultdict(set)
        for raw_path, expected_hash in manifest["pool_files"].items():
            path = Path(raw_path)
            if not path.exists():
                path = root / "pools" / f"shard{shard}" / path.name
            if file_hash(path) != expected_hash:
                raise ValueError("Scorer pool input hash no longer matches token events")
            rows = read_jsonl(path)
            for event in rows:
                pid, role, index = event["prompt_id"], event["role"], event["sample_index"]
                if pid not in pid_set or pid not in expected:
                    raise ValueError("Unexpected pool prompt")
                key = (role, index)
                if key in seen[pid]:
                    raise ValueError("Duplicate pool occurrence")
                seen[pid].add(key)
                source = expected[pid]
                if event["prompt"] != source["prompt"] or event["prompt_token_ids"] != source["prompt_token_ids"]:
                    raise ValueError("Pool event differs from fixed prompt text/token context")
                if event["split"] != source["split"] or event["candidate_id"] != f"{pid}:{role}:{index}":
                    raise ValueError("Pool occurrence identity mismatch")
                expected_seed = int(digest("20260909-repair-pool:" + event["candidate_id"])[:16], 16) % (2**63-1)
                if event["seed"] != expected_seed:
                    raise ValueError("Pool occurrence does not use its prescribed independent RNG stream")
                token_fields = {k: event[k] for k in ("input_ids", "attention_mask", "labels")}
                if object_hash(token_fields) != event["token_event_sha256"] or digest(event["response"]) != event["response_sha256"]:
                    raise ValueError("Pool candidate token/text hash mismatch")
                pool_hashes_by_split[event["split"]][event["candidate_id"]] = object_hash({
                    "token_event_sha256": event["token_event_sha256"], "response_sha256": event["response_sha256"],
                    "seed": event["seed"], "finish_reason": event["finish_reason"]})
                if role == "learner":
                    learners[(pid, index)] = event
        wanted_occurrences = {(role, i) for role in ROLES for i in range(8)}
        if set(seen) != set(pids) or any(values != wanted_occurrences for values in seen.values()):
            raise ValueError("Each scored prompt requires exactly 8+8+8 independent occurrences")
        shard_manifests.append({"shard": shard, "sha256": file_hash(folder / "manifest.json"),
                                "generation_settings_sha256": file_hash(settings_path),
                                "manifest": manifest})
    if set(scores) != set(expected) or len(learners) != len(expected) * 8:
        raise ValueError("Incomplete scored pool campaign")
    return splits, learners, scores, shard_manifests, {
        split: object_hash(records) for split, records in pool_hashes_by_split.items()}


def teacher_diagnostics(result, prompt_ids, learners):
    p = result.pi.numpy()
    g = result.target_log_ratio.numpy()
    lengths = np.array([[learners[(pid, i)]["n_tokens"] for i in range(8)] for pid in prompt_ids])
    refusals = np.array([[bool(REFUSAL_DIAGNOSTIC.search(learners[(pid, i)]["response"]))
                         for i in range(8)] for pid in prompt_ids])
    entropy = -(p * np.log(p)).sum(-1)
    ess, maximum = 1 / (p * p).sum(-1), p.max(-1)
    upper = np.triu_indices(8, 1)
    pair_targets = (g[:, :, None] - g[:, None, :])[:, upper[0], upper[1]]
    quantiles = [0., .01, .05, .25, .5, .75, .95, .99, 1.]
    per_prompt = [{"prompt_id": pid, "entropy": float(entropy[x]), "ess": float(ess[x]),
                   "max_mass": float(maximum[x]), "p_star": p[x].tolist(), "g": g[x].tolist(),
                   "response_tokens": lengths[x].tolist(),
                   "refusal_keyword_diagnostic": refusals[x].tolist()}
                  for x, pid in enumerate(prompt_ids)]
    return {"n_prompts": len(prompt_ids), "entropy_mean": float(entropy.mean()),
            "ess_mean": float(ess.mean()), "ess_min": float(ess.min()),
            "maximum_mass_mean": float(maximum.mean()), "maximum_mass_max": float(maximum.max()),
            "canonical_g_rms": float(np.sqrt((g*g).mean())),
            "pair_target_rms": float(np.sqrt((pair_targets*pair_targets).mean())),
            "quantile_levels": quantiles, "g_quantiles": np.quantile(g, quantiles).tolist(),
            "pair_target_quantiles": np.quantile(pair_targets, quantiles).tolist(),
            "response_length_unweighted_mean": float(lengths.mean()),
            "response_length_teacher_weighted_mean": float((p*lengths).sum(-1).mean()),
            "refusal_keyword_unweighted_fraction": float(refusals.mean()),
            "refusal_keyword_teacher_mass": float((p*refusals).sum(-1).mean()),
            "refusal_measure_status": "diagnostic keyword heuristic; not a semantic refusal evaluation"}, per_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--workers", type=int, default=min(32, len(os.sched_getaffinity(0))))
    ap.add_argument("--max-dual-calls", type=int, default=200)
    ap.add_argument("--probability-floor", type=float, default=1e-12)
    ap.add_argument("--output-name", default="nash_repair_v1")
    args = ap.parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError("Use 1..32 CPU solver workers")
    torch.set_num_threads(1)
    start = time.monotonic()
    splits, learners, scores, score_manifests, pool_hashes = load_campaign_inputs(args.root, args.shards)
    if len(splits["train"]) != 2000 or len(splits["dev"]) != 500:
        raise ValueError("Primary protocol requires train2000/dev500")
    out = args.root / "teachers" / args.output_name
    out.mkdir(parents=True, exist_ok=False)
    inventory = json.loads((args.root / "provenance/inventory.json").read_text())
    tokenizer_assets = {Path(row["path"]).name: row["sha256"] for row in inventory["assets"]
                        if Path(row["path"]).name.startswith("tokenizer") or Path(row["path"]).name == "special_tokens_map.json"}
    if not tokenizer_assets:
        raise ValueError("Inventory does not pin tokenizer assets")
    tokenizer_config_path = next(Path(row["path"]) for row in inventory["assets"]
                                 if Path(row["path"]).name == "tokenizer_config.json")
    from transformers import AutoTokenizer
    from mnpo_scripts.precompute_provenance import tokenizer_content_hashes
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config_path.parent, local_files_only=True)
    raw_tokenizer_hashes = tokenizer_content_hashes(tokenizer)
    # Match alignment.get_tokenizer's batching-only pad-token fallback. This
    # does not retokenize or append any token to the immutable sampled events.
    pad_token_fallback = tokenizer.pad_token_id is None
    if pad_token_fallback:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer_hashes = tokenizer_content_hashes(tokenizer)
    shared = {"model_revision": inventory["base_revision"], **tokenizer_hashes,
              "tokenizer_files_manifest_sha256": object_hash(tokenizer_assets),
              "generation_tokenizer_content_hashes": raw_tokenizer_hashes,
              "training_pad_token_fallback_to_eos": pad_token_fallback,
              "teacher_manifest_sha256": object_hash(score_manifests),
              "inventory_sha256": file_hash(args.root / "provenance/inventory.json"),
              "split_manifest_sha256": file_hash(args.root / "splits/manifest.json")}
    shared["model_weights_manifest_sha256"] = object_hash({Path(row["path"]).name: row["sha256"]
        for row in inventory["assets"] if Path(row["path"]).suffix == ".safetensors"})
    if file_hash(tokenizer_config_path) != tokenizer_assets["tokenizer_config.json"]:
        raise ValueError("Tokenizer config changed after the pinned inventory")
    write_json(out / "inputs.json", {**shared, "score_manifests": score_manifests,
                                     "pool_hashes": pool_hashes, "source_sha256": file_hash(__file__)})
    training = None
    outputs = {}
    for split, prompts in splits.items():
        if not prompts:
            outputs[split] = {"n_prompts": 0, "status": "no_eligible_prompts"}
            continue
        split_start = time.monotonic()
        pids = [p["prompt_id"] for p in prompts]
        A = np.stack([scores[pid][0] for pid in pids], axis=1)
        Aref = np.stack([scores[pid][1] for pid in pids], axis=1)
        tensor_dir = out / split / "tensor"
        tensor_dir.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(tensor_dir / "tensor_policy.npz", A=A)
        np.savez_compressed(tensor_dir / "tensor_ref.npz", A=Aref)
        meta = {"prompt_ids": pids, "objectives": OBJECTIVES,
                "reference_construction": "independent_samples",
                "reference_tensor_not_symmetrized": True,
                "policy_learner_ids": [f"policy:{i}" for i in range(8)],
                "comparator_ids": [f"ref:{i}" for i in range(8)],
                "pool_sha256": pool_hashes[split], **shared}
        write_json(tensor_dir / "meta.json", meta)
        rep = AdaptiveGameRepresentation(torch.from_numpy(A), torch.from_numpy(Aref),
                                         uniform_policy(len(pids), 8), torch.full((2,), .25, dtype=torch.float64),
                                         reference_construction="independent_samples")
        print(json.dumps({"phase": "solve", "split": split, "prompts": len(pids),
                          "workers": args.workers, "dual_fit": training is None}), flush=True)
        result = None
        try:
            result = solve_finite_pool(rep, "nash", eta=1., inner_solver="exact", dual_solver="root",
                                       dual_tol=1e-10, M=args.max_dual_calls, inner_workers=args.workers,
                                       probability_floor=args.probability_floor,
                                       fixed_weights=None if training is None else training.weights,
                                       log_every=10)
            certificate = validate_finite_pool_solution(result)
        except Exception as error:
            failure = {"status": "infeasible_or_unresolved", "error": str(error),
                       "split": split, "seconds": time.monotonic()-split_start,
                       "usable_for_training": False}
            if result is not None:
                failure.update({"lambda_raw": result.weights.tolist(), "surplus": result.surplus.tolist(),
                                "certificate": validate_finite_pool_solution(result, require_optimality=False)})
                np.savez_compressed(out / split / "uncertified_policy.npz", pi=result.pi.numpy())
            write_json(out / split / "solver_unresolved.json", failure)
            raise
        solver_dir = out / split / "solver"
        extra = {"split": split, "pool_sha256": pool_hashes[split], **shared,
                 "lambda_source": "train2000 only" if split == "train" else "fixed train2000 solution"}
        if training is not None:
            extra["training_solution_sha256"] = file_hash(out / "train/solver/solution.json")
        solution = write_generic_solution_artifact(
            solver_dir, result, meta, {name: file_hash(tensor_dir/name) for name in
                                      ("tensor_policy.npz", "tensor_ref.npz", "meta.json")},
            tensor_dir, 0, False, extra)
        if training is None:
            training = result
        canonical = load_canonical_artifact(solver_dir, solution, expected_prompt_ids=pids,
                                             expected_representation="adaptive_game", expected_aggregation="nash")
        solver_hash = file_hash(solver_dir / "solution.json")

        def pair_rows():
            for x, pid in enumerate(pids):
                policy = {str(i): {pid: {**learners[(pid, i)], "generated_text": learners[(pid, i)]["response"]}}
                          for i in range(8)}
                rows = build_rows([pid], OBJECTIVES, A[:, x:x+1], result.nu_update.numpy()[:, x:x+1],
                    result.weights.numpy(), np.array([.25, .25]), policy, None,
                    np.random.default_rng(42), "canonical_logratio", meta,
                    {"solver_artifact_sha256": solver_hash, "solver_hash": solver_hash,
                     "target_artifact_hash": solution["artifact_hashes"]["target_log_ratio.npz"],
                     "pool_sha256": pool_hashes[split], "representation": "adaptive_game",
                     "aggregation": "nash", "split": split, **shared},
                    canonical_data={key: values[x:x+1] for key, values in canonical.items()})
                for row in rows:
                    a, b = row["chosen_candidate_index"], row["rejected_candidate_index"]
                    row["chosen_response_id"] = learners[(pid, a)]["candidate_id"]
                    row["rejected_response_id"] = learners[(pid, b)]["candidate_id"]
                    yield row

        pair_path = out / "pairs" / f"{split}.jsonl"
        write_jsonl(pair_path, pair_rows())
        stats, per_prompt = teacher_diagnostics(result, pids, learners)
        write_json(out / split / "teacher_diagnostics.json", stats)
        write_jsonl(out / split / "teacher_per_prompt.jsonl", per_prompt)
        outputs[split] = {"n_prompts": len(pids), "n_pairs": len(pids)*28,
                          "pairs_path": str(pair_path), "pairs_sha256": file_hash(pair_path),
                          "solver_solution_sha256": solver_hash, "pool_sha256": pool_hashes[split],
                          "certificate": certificate, "seconds": time.monotonic()-split_start}
        write_json(out / split / "complete.json", outputs[split])
        print(json.dumps({"split": split, "n_pairs": len(pids)*28, "seconds": outputs[split]["seconds"],
                          "stationarity": certificate["independent_stationarity_inf"],
                          "lambda": result.weights.tolist()}), flush=True)
    write_json(out / "dataset_provenance.json", {**shared,
        "train_pool_sha256": pool_hashes["train"], "dev_pool_sha256": pool_hashes["dev"],
        "test_pool_sha256": pool_hashes.get("test"), "solver_splits": outputs})
    write_json(out / "complete.json", {"splits": outputs, "seconds": time.monotonic()-start,
                                       "source_sha256": file_hash(__file__)})


if __name__ == "__main__":
    main()
