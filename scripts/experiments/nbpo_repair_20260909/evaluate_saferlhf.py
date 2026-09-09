"""Frozen-GPM SafeRLHF evaluation of saved greedy responses, not policy acceptance.

The learner for each prompt is the point mass on its one generated response.
Its adaptive opponent ranges over the same eight saved comparator occurrences
used by training. Original disagreement uses the independently sampled eight
reference-as-learner occurrences, whose calibrated scores are reused verbatim.
Only prompt-sampling uncertainty conditional on these fixed pools is estimated.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from mnpo_scripts.nbpo_core import compute_margins, compute_regularized_game_value, uniform_policy
from scripts.experiments.nbpo_repair_20260909.common import (
    digest, file_hash, object_hash, read_jsonl, write_json, write_jsonl,
)
from scripts.experiments.nbpo_repair_20260909.evaluate_responses import (
    BOOTSTRAP_SEED, assert_matched_protocol, load_responses,
)

OBJECTIVES = ("helpfulness", "harmlessness")
SEEDS = (41, 42, 43)
BETA = .25


def response_uid(prompt_id):
    return "saferlhf:" + str(prompt_id)


def bind_responses(rows, prompts):
    """Align by exact UID and text, never file order or a shortened hash."""
    expected = {response_uid(p["prompt_id"]): p for p in prompts}
    if len(expected) != len(prompts) or set(rows) - set(expected):
        raise ValueError("Duplicate planned or unexpected SafeRLHF response UID")
    aligned = []
    for p in prompts:
        row = rows.get(response_uid(p["prompt_id"]))
        if row is not None:
            if row["prompt"] != p["prompt"] or row.get("prompt_sha256") != digest(p["prompt"]):
                raise ValueError("SafeRLHF response prompt/text hash mismatch")
            if row.get("meta", {}).get("split") != p["split"]:
                raise ValueError("SafeRLHF response split mismatch")
            if row.get("prompt_token_ids") != p["prompt_token_ids"]:
                raise ValueError("Greedy and comparator prompts have different token contexts")
        aligned.append(row)
    return aligned


def generated_game_values(calibrated_probabilities, *, beta=BETA):
    """(K,X,J) ensemble probabilities -> (X,K) deterministic-policy values."""
    probabilities = np.asarray(calibrated_probabilities, dtype=np.float64)
    if probabilities.ndim != 3 or probabilities.shape[0] != 2 or probabilities.shape[2] != 8:
        raise ValueError("Expected (2, prompts, 8) calibrated ensemble probabilities")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Invalid calibrated probabilities")
    margins = torch.from_numpy(probabilities - .5)
    values = compute_regularized_game_value(
        margins, uniform_policy(probabilities.shape[1], 8),
        torch.full((2,), beta, dtype=torch.float64), form="softmin", per_prompt=True)
    return values.numpy().T


def reference_disagreement_values(reference_probabilities, *, beta=BETA):
    """Reuse independent8 reference learner scores; average learners before softmin."""
    probabilities = np.asarray(reference_probabilities, dtype=np.float64)
    if probabilities.ndim != 4 or probabilities.shape[0] != 2 or probabilities.shape[2:] != (8, 8):
        raise ValueError("Expected (2, prompts, 8, 8) reference probabilities")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Invalid independent-reference probabilities")
    Aref = torch.from_numpy(probabilities - .5)
    margins = compute_margins(Aref, uniform_policy(probabilities.shape[1], 8))
    values = compute_regularized_game_value(
        margins, uniform_policy(probabilities.shape[1], 8),
        torch.full((2,), beta, dtype=torch.float64), form="softmin", per_prompt=True)
    return values.numpy().T


def paired_prompt_bootstrap(values, disagreement, base_values=None, *,
                            seed=BOOTSTRAP_SEED, repetitions=2000):
    """Jointly resample prompt rows for V,d,s and candidate-minus-base vectors."""
    values, disagreement = np.asarray(values, dtype=np.float64), np.asarray(disagreement, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or disagreement.shape != values.shape:
        raise ValueError("V and original disagreement must be aligned (prompts,2) arrays")
    if repetitions < 1 or not np.isfinite(values).all() or not np.isfinite(disagreement).all():
        raise ValueError("Bootstrap requires finite values and positive repetition count")
    arrays = {"V": values, "d_original_reference": disagreement, "s_original_reference": values-disagreement}
    if base_values is not None:
        base_values = np.asarray(base_values, dtype=np.float64)
        if base_values.shape != values.shape or not np.isfinite(base_values).all():
            raise ValueError("Base values must have the same aligned prompt rows")
        arrays.update(V_base_greedy=base_values, delta_vs_base_greedy=values-base_values)
    n = len(values)
    if not n:
        return {"n_prompts": 0, "objectives": list(OBJECTIVES), "metrics": {
            name: {"estimate": None, "ci95": None} for name in arrays}}
    rng = np.random.default_rng(seed)
    draws = {key: [] for key in arrays}
    for start in range(0, repetitions, 100):
        indices = rng.integers(n, size=(min(100, repetitions-start), n))
        for key, array in arrays.items():
            draws[key].append(array[indices].mean(1))
    return {"n_prompts": n, "objectives": list(OBJECTIVES),
            "bootstrap_seed": seed, "bootstrap_repetitions": repetitions,
            "uncertainty_scope": "paired prompt sampling conditional on frozen pools and teacher; not training-seed, pool-sampling, or teacher-estimation uncertainty",
            "metrics": {key: {"estimate": array.mean(0).tolist(),
                               "ci95": np.quantile(np.concatenate(draws[key]), [.025, .975], axis=0).T.tolist()}
                        for key, array in arrays.items()}}


def load_fixed_game(root, prompts, shards):
    """Recover exact comparator occurrences and stored independent-reference scores."""
    expected = {p["prompt_id"]: p for p in prompts}
    references, comparators, source_hashes = {}, {}, []
    common_teacher = None
    for shard in range(shards):
        folder = root / "scores" / f"shard{shard}"
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["objectives"] != list(OBJECTIVES) or manifest["seeds"] != list(SEEDS):
            raise ValueError("Unexpected frozen GPM objective/seed order")
        if not manifest.get("reference_tensor_not_symmetrized") or manifest["reference_construction"] != "independent_reference_learner8_against_comparator8":
            raise ValueError("Original disagreement must retain independent reference learner support")
        if manifest.get("calibration_applied_before_seed_average") is not True or manifest["max_teacher_tokens"] != 384:
            raise ValueError("Frozen calibration/context contract mismatch")
        teacher = {key: manifest[key] for key in ("teacher_models", "calibration_sha256",
            "teacher_tokenizer_hashes", "encoder_config_sha256", "max_teacher_tokens")}
        if common_teacher is not None and teacher != common_teacher:
            raise ValueError("Frozen teacher identities differ between source shards")
        common_teacher = teacher
        tensor_path = folder / "probabilities.npz"
        if file_hash(tensor_path) != manifest["probabilities_sha256"]:
            raise ValueError("Original reference score tensor hash mismatch")
        with np.load(tensor_path) as arrays:
            reference = arrays["reference"]
            if reference.shape != (3, 2, len(manifest["prompt_ids"]), 8, 8):
                raise ValueError("Original reference score tensor shape mismatch")
            reference = reference.mean(0)
        for x, pid in enumerate(manifest["prompt_ids"]):
            if pid in expected:
                if pid in references or manifest["prompt_splits"][pid] != expected[pid]["split"]:
                    raise ValueError("Duplicate or incorrectly assigned held-out score prompt")
                references[pid] = reference[:, x]
        used_pool_files = {}
        for raw_path, sha in manifest["pool_files"].items():
            path = Path(raw_path)
            if not path.exists():
                path = root / "pools" / f"shard{shard}" / path.name
            if file_hash(path) != sha:
                raise ValueError("Fixed comparator pool file hash mismatch")
            used_pool_files[str(path)] = sha
            for row in read_jsonl(path):
                pid = row["prompt_id"]
                if pid not in expected or row["role"] != "comparator":
                    continue
                index = row["sample_index"]
                if index not in range(8) or (pid, index) in comparators:
                    raise ValueError("Duplicate/invalid comparator occurrence")
                if row["prompt"] != expected[pid]["prompt"] or row["prompt_token_ids"] != expected[pid]["prompt_token_ids"]:
                    raise ValueError("Comparator has different text/token conditioning context")
                if row["candidate_id"] != f"{pid}:comparator:{index}" or digest(row["response"]) != row["response_sha256"]:
                    raise ValueError("Comparator response identity/hash mismatch")
                if object_hash({k: row[k] for k in ("input_ids", "attention_mask", "labels")}) != row["token_event_sha256"]:
                    raise ValueError("Comparator sampled token event hash mismatch")
                comparators[(pid, index)] = row
        source_hashes.append({"shard": shard, "score_manifest_sha256": file_hash(manifest_path),
                               "probabilities_sha256": manifest["probabilities_sha256"],
                               "score_source_sha256": manifest["source_sha256"], "pool_files": used_pool_files})
    if set(references) != set(expected) or set(comparators) != {(pid, i) for pid in expected for i in range(8)}:
        raise ValueError("Incomplete fixed comparator/reference coverage")
    Aref = np.stack([references[p["prompt_id"]] for p in prompts], axis=1)
    fixed_identity = {"source_artifacts": source_hashes,
                      "comparator_occurrences_sha256": object_hash({row["candidate_id"]: {
                          "token_sha256": row["token_event_sha256"], "response_sha256": row["response_sha256"]}
                          for row in comparators.values()}),
                      "reference_ensemble_probabilities_sha256": digest(np.ascontiguousarray(Aref).tobytes()),
                      "prompt_order_sha256": object_hash([p["prompt_id"] for p in prompts]),
                      "teacher": common_teacher}
    return comparators, reference_disagreement_values(Aref), fixed_identity


def calibrated_head_probabilities(model, left, comparators, objective, temperature):
    """Same FP32 head -> sigmoid -> FP64 clipping/calibration path as score_pool."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid frozen GPM calibration temperature")
    count = len(left)
    lhs = left.unsqueeze(1).expand(count, 8, -1).reshape(count*8, -1)
    rhs = comparators.unsqueeze(0).expand(count, 8, -1).reshape(count*8, -1)
    raw = torch.sigmoid(model.logit(lhs, rhs, objective)).double().cpu().numpy().reshape(count, 8)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite frozen GPM head probabilities")
    raw = np.clip(raw, 1e-9, 1-1e-9)
    logit = np.log(raw) - np.log1p(-raw)
    return 1 / (1 + np.exp(-logit / temperature))


@torch.inference_mode()
def score_generated(args, prompts, rows_by_label, comparators, fixed_identity):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from mnpo_scripts.gpm import AntiSymmetricGPM
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Production scoring requires the explicitly assigned CUDA device")
    teacher = fixed_identity["teacher"]
    calibration_path = args.ensemble / "saferlhf_ensemble.json"
    if file_hash(calibration_path) != teacher["calibration_sha256"]:
        raise ValueError("Evaluation calibration differs from the frozen pool teacher")
    if file_hash(args.encoder / "config.json") != teacher["encoder_config_sha256"]:
        raise ValueError("Evaluation encoder configuration differs from the frozen pool teacher")
    calibration = json.loads(calibration_path.read_text())["calibration"]["gpm"]
    probabilities = {label: np.full((3, 2, len(prompts), 8), np.nan) for label in rows_by_label}
    failures, truncations, timings = [], [], {}
    for si, seed in enumerate(SEEDS):
        started = time.monotonic()
        checkpoint = args.ensemble / f"ckpt_gpm_seed{seed}"
        recorded = next(record for record in teacher["teacher_models"] if record["seed"] == seed)
        if file_hash(checkpoint / "model.pt") != recorded["checkpoint_sha256"]:
            raise ValueError("Evaluation GPM checkpoint differs from frozen teacher")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        hashes = {name: file_hash(checkpoint / name) for name in teacher["teacher_tokenizer_hashes"]}
        if hashes != teacher["teacher_tokenizer_hashes"] or tokenizer.padding_side != "right":
            raise ValueError("Frozen GPM tokenizer/right-padding contract differs")
        blob = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=False)
        config = AutoConfig.from_pretrained(args.encoder, local_files_only=True)
        if blob["kind"] != "gpm" or config.model_type != "roberta" or config.hidden_size != blob["hidden"] or config.max_position_embeddings < 386:
            raise ValueError("Frozen encoder/head configuration mismatch")
        encoder = AutoModel.from_config(config)
        model = AntiSymmetricGPM(encoder, blob["hidden"], 2, width=blob["width"], dropout=0.)
        model.load_state_dict(blob["state_dict"], strict=True)
        model = model.to(device).eval()
        for x, prompt in enumerate(prompts):
            labels = [label for label, rows in rows_by_label.items() if rows[x] is not None]
            if not labels:
                continue
            candidate_texts = [rows_by_label[label][x]["output"] for label in labels]
            comparator_texts = [comparators[(prompt["prompt_id"], i)]["response"] for i in range(8)]
            texts = candidate_texts + comparator_texts
            try:
                batch = tokenizer([prompt["prompt"]]*len(texts), texts, padding=True,
                    truncation="longest_first", max_length=384, return_tensors="pt")
                if si == 0:
                    prompt_length = len(tokenizer(prompt["prompt"], add_special_tokens=False)["input_ids"])
                    response_lengths = [len(v) for v in tokenizer(texts, add_special_tokens=False)["input_ids"]]
                    for index, label in enumerate(labels):
                        sequence_ids = batch.sequence_ids(index)
                        truncations.append({"uid": response_uid(prompt["prompt_id"]), "label": label,
                            "prompt_truncated": sequence_ids.count(0) < prompt_length,
                            "response_truncated": sequence_ids.count(1) < response_lengths[index],
                            "teacher_prompt_tokens": prompt_length, "teacher_response_tokens": response_lengths[index],
                            "teacher_prompt_tokens_kept": sequence_ids.count(0), "teacher_response_tokens_kept": sequence_ids.count(1)})
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    encoded = model.encode(batch["input_ids"].to(device), batch["attention_mask"].to(device)).float()
                for k, name in enumerate(OBJECTIVES):
                    calibrated = calibrated_head_probabilities(model, encoded[:len(labels)], encoded[len(labels):],
                        k, float(calibration[str(seed)][name]["temperature"]))
                    for index, label in enumerate(labels):
                        probabilities[label][si, k, x] = calibrated[index]
            except Exception as error:
                for label in labels:
                    failures.append({"uid": response_uid(prompt["prompt_id"]), "label": label,
                                     "teacher_seed": seed, "error": f"{type(error).__name__}: {error}"})
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    raise
            if (x+1) % 100 == 0:
                print(json.dumps({"teacher_seed": seed, "prompts": x+1,
                                  "seconds": time.monotonic()-started}), flush=True)
        timings[str(seed)] = time.monotonic()-started
        del model, encoder, blob
        torch.cuda.empty_cache()
    return probabilities, failures, truncations, timings


def summarize_label(label, prompts, rows, probabilities, disagreement, base_probabilities,
                    *, repetitions=2000):
    valid = np.isfinite(probabilities).all(axis=(0, 1, 3))
    base_valid = np.isfinite(base_probabilities).all(axis=(0, 1, 3))
    values, base_values = np.full((len(prompts), 2), np.nan), np.full((len(prompts), 2), np.nan)
    if valid.any():
        values[valid] = generated_game_values(probabilities[:, :, valid].mean(0))
    if base_valid.any():
        base_values[base_valid] = generated_game_values(base_probabilities[:, :, base_valid].mean(0))
    records, reports = [], {}
    for x, prompt in enumerate(prompts):
        record = {"uid": response_uid(prompt["prompt_id"]), "prompt_id": prompt["prompt_id"],
                  "split": prompt["split"], "label": label, "d_original_reference": disagreement[x].tolist(),
                  "status": "ok" if valid[x] else "missing_response" if rows[x] is None else "teacher_failure"}
        if valid[x]:
            record.update(V=values[x].tolist(), s_original_reference=(values[x]-disagreement[x]).tolist())
        if valid[x] and base_valid[x]:
            record.update(V_base_greedy=base_values[x].tolist(), delta_vs_base_greedy=(values[x]-base_values[x]).tolist())
        if rows[x] is not None:
            record.update(n_output_tokens=rows[x]["n_output_tokens"], finish_reason=rows[x]["finish_reason"])
        records.append(record)
    for split in ("dev", "test"):
        planned = np.array([p["split"] == split for p in prompts])
        scored, paired = planned & valid, planned & valid & base_valid
        marginal = paired_prompt_bootstrap(values[scored], disagreement[scored], repetitions=repetitions)
        pair_report = paired_prompt_bootstrap(values[paired], disagreement[paired], base_values[paired], repetitions=repetitions)
        generated = sum(row is not None and is_planned for row, is_planned in zip(rows, planned))
        reports[split] = {"n_planned": int(planned.sum()), "n_generated": int(generated),
            "n_scored": int(scored.sum()), "n_failed": int(planned.sum()-scored.sum()),
            "n_paired_vs_base": int(paired.sum()), "complete": bool(np.array_equal(planned, scored)),
            "original_reference_comparison": marginal, "paired_greedy_base_comparison": pair_report,
            "original_disagreement_all_planned": paired_prompt_bootstrap(
                disagreement[planned], disagreement[planned], repetitions=repetitions)["metrics"]["d_original_reference"],
            "fresh_test_claim": False,
            "acceptance_estimate": {
                "scope": "finite-comparator, frozen-teacher values of the deterministic greedy decoder",
                "all_empirical_original_surpluses_positive": bool((values[scored]-disagreement[scored]).mean(0).min() > 0) if scored.any() else None,
                "algorithm1_accepted_next_stage_policy": False,
                "status": "not_certified_for_stochastic_neural_policy",
                "reason": "Greedy point-mass responses are not unbiased samples from the raw stochastic trained policy. This estimate neither certifies Algorithm1 acceptance nor replaces the finite-pool solver certificate."}}
    return records, reports


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--base-label", default="base")
    ap.add_argument("--base-score-dir", type=Path, help="Reuse a prior base label's probabilities/scoring_manifest directory")
    ap.add_argument("--ensemble", type=Path, default=Path("/work/iclr27_table1_v2/models/saferlhf_ensemble_ckpt"))
    ap.add_argument("--encoder", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--device", default="cuda:0", help="Use only the GPU explicitly assigned by the caller")
    ap.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = ap.parse_args()
    prompts = [p for split in ("dev", "test") for p in read_jsonl(args.root / "splits" / f"{split}.jsonl")]
    split_manifest = json.loads((args.root / "splits/manifest.json").read_text())
    for split in ("dev", "test"):
        if file_hash(args.root / "splits" / f"{split}.jsonl") != split_manifest["splits"][split]["file_sha256"]:
            raise ValueError("Evaluation split file no longer matches its manifest")
    labels = list(dict.fromkeys([args.base_label, *args.labels]))
    rows_by_label, provenance = {}, {}
    for label in labels:
        rows, _, provenance[label] = load_responses(args.root, label, "saferlhf")
        rows_by_label[label] = bind_responses(rows, prompts)
        settings = provenance[label]["generation_settings"]
        if settings.get("temperature") != 0 or settings.get("top_p") != 1 or not settings.get("native_user_only"):
            raise ValueError("This evaluator is scoped to the fixed native-template greedy response protocol")
        assert_matched_protocol(provenance[args.base_label]["generation_settings"], settings)
        if any(row is not None and row.get("model_sha256") != object_hash(settings["model_weights"])
               for row in rows_by_label[label]):
            raise ValueError("Response model hash does not match its exported checkpoint manifest")
    comparators, disagreement, fixed_identity = load_fixed_game(args.root, prompts, args.shards)
    fixed_hash = object_hash(fixed_identity)
    cached_base = None
    cached_base_lineage = None
    if args.base_score_dir:
        cache = json.loads((args.base_score_dir / "scoring_manifest.json").read_text())
        if (cache["fixed_game_sha256"] != fixed_hash or cache["response_provenance"] != provenance[args.base_label]
                or cache["prompt_ids"] != [p["prompt_id"] for p in prompts]):
            raise ValueError("Cached base probabilities have different fixed game or responses")
        path = args.base_score_dir / "probabilities.npz"
        if file_hash(path) != cache["probabilities_sha256"]:
            raise ValueError("Cached base probability hash mismatch")
        cached_base = np.load(path)["probabilities"]
        if cached_base.shape != (3, 2, len(prompts), 8):
            raise ValueError("Cached base probability shape mismatch")
        cached_base_lineage = {"scoring_manifest_sha256": file_hash(args.base_score_dir / "scoring_manifest.json"),
                               "scoring_source_sha256": cache["source_sha256"],
                               "probabilities_sha256": cache["probabilities_sha256"],
                               "source_directory": str(args.base_score_dir)}
    args.out.mkdir(parents=True, exist_ok=False)
    score_rows = {label: rows for label, rows in rows_by_label.items()
                  if cached_base is None or label != args.base_label}
    probabilities, failures, truncations, timings = (score_generated(
        args, prompts, score_rows, comparators, fixed_identity) if score_rows else ({}, [], [], {}))
    if cached_base is not None:
        probabilities[args.base_label] = cached_base
    write_json(args.out / "fixed_game.json", {**fixed_identity, "sha256": fixed_hash,
        "beta": [.25, .25], "comparator_occurrence_mass": [.125]*8,
        "reference_construction": "independent8 reference learner versus fixed8 comparator",
        "disagreement_reused_from_original_scores": True, "split_manifest_sha256": file_hash(args.root / "splits/manifest.json")})
    write_jsonl(args.out / "failures.jsonl", failures)
    write_jsonl(args.out / "teacher_truncation.jsonl", truncations)
    for label in labels:
        folder = args.out / label
        folder.mkdir()
        np.savez_compressed(folder / "probabilities.npz", probabilities=probabilities[label])
        write_json(folder / "scoring_manifest.json", {"label": label, "prompt_ids": [p["prompt_id"] for p in prompts],
            "fixed_game_sha256": fixed_hash, "response_provenance": provenance[label],
            "probabilities_sha256": file_hash(folder / "probabilities.npz"),
            "seeds": list(SEEDS), "objectives": list(OBJECTIVES), "source_sha256": file_hash(__file__),
            "pool_scorer_source_sha256_current": file_hash(Path(__file__).with_name("score_pool.py")),
            "encoder_precision": "BF16 autocast; encoded representations cast to FP32",
            "head_and_calibration_precision": "FP32 head/sigmoid, then FP64 clipping/logit/temperature/sigmoid exactly as pool scorer",
            "encoder_batch": "available generated labels followed by fixed8 comparators; right padded; longest_first384",
            "base_cache_reused": bool(cached_base is not None and label == args.base_label),
            "cached_probability_source": cached_base_lineage if label == args.base_label else None})
        records, reports = summarize_label(label, prompts, rows_by_label[label], probabilities[label], disagreement,
                                           probabilities[args.base_label], repetitions=args.bootstrap_repetitions)
        write_jsonl(folder / "per_prompt.jsonl", records)
        truncation_records = [row for row in truncations if row["label"] == label]
        lengths = [row["n_output_tokens"] for row in rows_by_label[label] if row is not None]
        truncation_report = {"n_evaluated": len(truncation_records),
            "prompt_truncated_fraction": float(np.mean([r["prompt_truncated"] for r in truncation_records])) if truncation_records else None,
            "response_truncated_fraction": float(np.mean([r["response_truncated"] for r in truncation_records])) if truncation_records else None,
            "status": "measured" if truncation_records else "not_recomputed_for_cached_base" if cached_base is not None and label == args.base_label else "unavailable"}
        write_json(folder / "report.json", {"label": label, "objectives": list(OBJECTIVES), "splits": reports,
            "response_provenance": provenance[label], "fixed_game_sha256": fixed_hash,
            "scope": "actual saved greedy generation against frozen finite comparator pools; separate from finite-pool solver certificate",
            "test_exposure": split_manifest.get("test_exposure_note"), "fresh_test_claim": False,
            "reference_interpretation": "s uses original independently sampled raw-reference d; delta_vs_base_greedy uses actual saved greedy base V. These are different reference policies.",
            "teacher_truncation": truncation_report,
            "response_length": {"mean_tokens": float(np.mean(lengths)) if lengths else None,
                                "median_tokens": float(np.median(lengths)) if lengths else None,
                                "max_length_hits": sum(row is not None and row["finish_reason"] == "length" for row in rows_by_label[label])},
            "cached_probability_source": cached_base_lineage if label == args.base_label else None,
            "paid_judge_api_calls": 0})
    write_json(args.out / "evaluation_complete.json", {"labels": labels, "seconds_per_teacher_seed": timings,
        "failures": len(failures), "bootstrap_seed": BOOTSTRAP_SEED,
        "source_sha256": file_hash(__file__), "paid_judge_api_calls": 0,
        "algorithm1_accepted_next_stage_policy": False})


if __name__ == "__main__":
    main()
