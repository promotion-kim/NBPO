"""CPU-only preparation of optional Game-utilitarian-WBC-L1matched artifacts.

No training launch is authorized by this script. All outputs use a separate
namespace. Equal RAW weights inherit their L1 scale from train-only Nash once;
dev/test never fit weights, and no Nash dual certificate applies to this control.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mnpo_scripts.nbpo_core import uniform_policy
from mnpo_scripts.nbpo_generic import solve_finite_pool, validate_finite_pool_solution
from mnpo_scripts.nbpo_representations import AdaptiveGameRepresentation
from scripts.nbpo.build_nbpo_pairs import build_rows, load_canonical_artifact
from scripts.nbpo.solve_nbpo_dual import write_generic_solution_artifact
from scripts.experiments.nbpo_repair_20260909.common import file_hash, object_hash, write_json, write_jsonl
from scripts.experiments.nbpo_repair_20260909.solve_campaign import OBJECTIVES, load_campaign_inputs, teacher_diagnostics

ARM = "Game-utilitarian-WBC-L1matched"


def derive_raw_control_weights(solution, expected_prompt_ids):
    """No dev/test or neural/outcome information enters the equal-weight scale."""
    cfg, certificate = solution["config"], solution["certificate"]
    if (solution["aggregation"] != "nash" or solution["representation"] != "adaptive_game"
            or solution["split"] != "train" or solution["prompt_ids"] != expected_prompt_ids
            or cfg["fixed_weights"] is not False or cfg["dual_fit_scope"] != "training"
            or certificate["certified"] is not True or certificate["nash_dual_scope"] != "training"
            or cfg["inner_solver"] != "exact" or solution["eta"] != 1.
            or cfg["beta"] != [.25, .25]
            or cfg["representation"]["reference_construction"] != "independent_samples"):
        raise ValueError("L1 scale must come only from the certified primary train Nash solution")
    original = np.asarray(solution["aggregation_weights_raw"], dtype=np.float64)
    if (original.shape != (2,) or not np.isfinite(original).all() or np.any(original <= 0)
            or not np.array_equal(original, np.asarray(solution["lambda_raw"], dtype=np.float64))):
        raise ValueError("Expected two consistent positive RAW Nash multipliers")
    total = float(original.sum())
    weights = np.full(2, total/2., dtype=np.float64)
    return weights, {"arm": ARM, "train_nash_lambda_raw": original.tolist(), "train_nash_lambda_l1": total,
        "control_weights_raw": weights.tolist(), "weight_sum": float(weights.sum()),
        "weight_construction": "L=sum(train-only Nash lambda) once; equal raw weights=(L/2,L/2)",
        "simplex_normalized": False, "scale_fit_scope": "source train2000 Nash artifact only",
        "weights_refit_on_dev_or_test": False, "nash_dual_certificate_status": "not_applicable",
        "interpretation": "Matched-total-weight aggregation control, not a claim against every utilitarian method"}


def numpy_independent_crosscheck(A, Aref, pi, target, weights, *, eta=1., beta=.25):
    """Independent NumPy log-sum-exp value/Q/centered-stationarity recomputation."""
    A, Aref, pi, target, weights = (np.asarray(v, dtype=np.float64) for v in (A, Aref, pi, target, weights))
    if A.ndim != 4 or Aref.shape != A.shape or pi.shape != A.shape[1:3] or target.shape != pi.shape:
        raise ValueError("Cross-check tensor/policy shape mismatch")
    if not all(np.isfinite(v).all() for v in (A, Aref, pi, target, weights)) or np.any(pi <= 0):
        raise ValueError("Cross-check requires finite full support")
    if weights.shape != (A.shape[0],) or abs(eta-1.) > 1e-12 or abs(beta-.25) > 1e-12:
        raise ValueError("Cross-check primary raw-weight/eta/beta contract mismatch")

    def softmin(margins):
        z = -margins/beta
        top = z.max(-1, keepdims=True)
        logmeanexp = top[..., 0] + np.log(np.exp(z-top).mean(-1))
        return -beta*logmeanexp

    margins = np.einsum("xi,kxij->kxj", pi, A)
    logits = -margins/beta
    nu = np.exp(logits-logits.max(-1, keepdims=True))
    nu /= nu.sum(-1, keepdims=True)
    Q = np.einsum("kxij,kxj->kxi", A, nu)
    g = np.log(pi)+np.log(pi.shape[1])
    b = g-eta*np.einsum("k,kxi->xi", weights, Q)
    centered = b-(pi*b).sum(-1, keepdims=True)
    V = softmin(margins).mean(-1)
    d = softmin(Aref.mean(2)).mean(-1)
    result = {"implementation": "Independent NumPy, no production value/opponent/stationarity routines",
        "canonical_serialization_error": float(np.abs(g-target).max()),
        "normalization_error": float(np.abs(pi.sum(-1)-1.).max()),
        "independent_stationarity_inf": float(np.abs(centered).max()),
        "V": V.tolist(), "d": d.tolist(), "surplus": (V-d).tolist(),
        "weights_raw": weights.tolist(), "weight_sum": float(weights.sum()),
        "nash_dual_certificate_status": "not_applicable"}
    if (result["canonical_serialization_error"] >= 1e-9 or result["normalization_error"] >= 1e-10
            or result["independent_stationarity_inf"] >= 1e-4):
        raise ValueError("Independent utilitarian teacher cross-check failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-nash", default="nash_repair_v2")
    parser.add_argument("--namespace", default="utilitarian_l1matched_v1")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or Path(args.namespace).name != args.namespace or args.namespace == args.source_nash:
        raise ValueError("Use 1..32 CPU workers and a separate output namespace")
    torch.set_num_threads(1)
    start = time.monotonic()
    source = args.root / "teachers" / args.source_nash
    out, dataset_out = args.root / "teachers" / args.namespace, args.root / "datasets" / args.namespace
    if out.exists() or dataset_out.exists():
        raise FileExistsError("Optional teacher/dataset namespace already exists; original outputs are immutable")
    splits, learners, scores, score_manifests, pool_hashes = load_campaign_inputs(args.root, 4)
    if len(splits["train"]) != 2000 or len(splits["dev"]) != 500:
        raise ValueError("Primary fixed train2000/dev500 must be retained")
    input_path, train_solution_path = source / "inputs.json", source / "train/solver/solution.json"
    original_inputs, nash_solution = (json.loads(path.read_text()) for path in (input_path, train_solution_path))
    pids_train = [p["prompt_id"] for p in splits["train"]]
    weights, control = derive_raw_control_weights(nash_solution, pids_train)
    load_canonical_artifact(source / "train/solver", nash_solution, expected_prompt_ids=pids_train,
        expected_representation="adaptive_game", expected_aggregation="nash")
    if (pool_hashes != original_inputs["pool_hashes"]
            or object_hash(score_manifests) != original_inputs["teacher_manifest_sha256"]
            or original_inputs["teacher_manifest_sha256"] != nash_solution["teacher_manifest_sha256"]):
        raise ValueError("Optional control and primary Nash must use identical pools and frozen preference teacher")
    shared = {key: value for key, value in original_inputs.items()
              if key not in ("score_manifests", "pool_hashes", "source_sha256")}
    control.update(source_nash_inputs_sha256=file_hash(input_path),
                   source_train_nash_solution_sha256=file_hash(train_solution_path),
                   source_train_nash_pi_star_sha256=nash_solution["artifact_hashes"]["pi_star.npz"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "inputs.json", {**shared, "score_manifests": score_manifests, "pool_hashes": pool_hashes,
        "control": control, "source_sha256": file_hash(__file__),
        "primary_files_modified": False, "optional_training_authorized": False})
    outputs = {}
    raw = torch.from_numpy(weights.copy())
    for split, prompts in splits.items():
        if not prompts:
            outputs[split] = {"n_prompts": 0, "status": "no_eligible_prompts"}
            continue
        split_start = time.monotonic()
        pids = [p["prompt_id"] for p in prompts]
        source_solution_path = source / split / "solver/solution.json"
        source_solution = json.loads(source_solution_path.read_text())
        A = np.stack([scores[pid][0] for pid in pids], axis=1)
        Aref = np.stack([scores[pid][1] for pid in pids], axis=1)
        original_tensor = source / split / "tensor"
        if (not np.array_equal(A, np.load(original_tensor / "tensor_policy.npz")["A"])
                or not np.array_equal(Aref, np.load(original_tensor / "tensor_ref.npz")["A"])):
            raise ValueError("Original Nash tensors differ from the shared frozen scores")
        tensor_dir = out / split / "tensor"
        tensor_dir.mkdir(parents=True, exist_ok=False)
        tensor_hashes = {}
        for filename in ("tensor_policy.npz", "tensor_ref.npz", "meta.json"):
            actual_hash = file_hash(original_tensor / filename)
            if actual_hash != source_solution["input_hashes"][filename]:
                raise ValueError("Source Nash tensor hash changed")
            shutil.copyfile(original_tensor / filename, tensor_dir / filename)
            tensor_hashes[filename] = actual_hash
        meta = json.loads((tensor_dir / "meta.json").read_text())
        if meta["prompt_ids"] != pids or meta["pool_sha256"] != pool_hashes[split]:
            raise ValueError("Source tensor prompt/pool binding mismatch")
        rep = AdaptiveGameRepresentation(torch.from_numpy(A), torch.from_numpy(Aref), uniform_policy(len(pids), 8),
            torch.full((2,), .25, dtype=torch.float64), reference_construction="independent_samples")
        print(json.dumps({"split": split, "phase": "utilitarian_cpu_solve", "weights_raw": weights.tolist(),
                          "workers": args.workers}), flush=True)
        result = solve_finite_pool(rep, "utilitarian", eta=1., weights=raw, inner_solver="exact", inner_workers=args.workers)
        # Generic metadata calls any non-fixed call "training". There is no
        # dual fit in this control, on any split; state its actual scope.
        result.config["dual_fit_scope"] = "not_applicable_no_dual_fit"
        result.config["control_weight_source"] = "fixed_source_train_Nash_L1"
        if not torch.equal(result.weights, raw):
            raise ValueError("Solver changed or simplex-normalized the raw control weights")
        certificate = validate_finite_pool_solution(result)
        if certificate["nash_dual_scope"] != "not_applicable":
            raise ValueError("Utilitarian control must not claim a Nash dual certificate")
        crosscheck = numpy_independent_crosscheck(A, Aref, result.pi.numpy(), result.target_log_ratio.numpy(), weights)
        for field, actual in (("V", result.V), ("d", result.d), ("surplus", result.surplus)):
            error = float(np.max(np.abs(np.asarray(crosscheck[field])-actual.numpy())))
            crosscheck[field+"_agreement_error"] = error
            if error >= 1e-9:
                raise ValueError("Independent NumPy and production game values disagree")
        solver_dir = out / split / "solver"
        solution = write_generic_solution_artifact(solver_dir, result, meta, tensor_hashes, tensor_dir, 0, False,
            {"split": split, "pool_sha256": pool_hashes[split], **shared, "control": control,
             "lambda_source": "Not Nash multipliers: equal RAW aggregation weights from source train Nash L1",
             "nash_dual_certificate_status": "not_applicable", "optional_training_authorized": False})
        canonical = load_canonical_artifact(solver_dir, solution, expected_prompt_ids=pids,
            expected_representation="adaptive_game", expected_aggregation="utilitarian")
        solver_hash = file_hash(solver_dir / "solution.json")

        def pairs():
            for x, pid in enumerate(pids):
                policy = {str(i): {pid: {**learners[(pid, i)], "generated_text": learners[(pid, i)]["response"]}}
                          for i in range(8)}
                for row in build_rows([pid], OBJECTIVES, A[:, x:x+1], result.nu_update.numpy()[:, x:x+1], weights,
                    np.array([.25, .25]), policy, None, np.random.default_rng(42), "canonical_logratio", meta,
                    {"solver_artifact_sha256": solver_hash, "solver_hash": solver_hash,
                     "target_artifact_hash": solution["artifact_hashes"]["target_log_ratio.npz"],
                     "pool_sha256": pool_hashes[split], "representation": "adaptive_game", "aggregation": "utilitarian",
                     "split": split, "control_scale_source_sha256": control["source_train_nash_solution_sha256"], **shared},
                    canonical_data={key: value[x:x+1] for key, value in canonical.items()}):
                    row["chosen_response_id"] = learners[(pid, row["chosen_candidate_index"])]["candidate_id"]
                    row["rejected_response_id"] = learners[(pid, row["rejected_candidate_index"])]["candidate_id"]
                    yield row

        pair_path = out / "pairs" / f"{split}.jsonl"
        write_jsonl(pair_path, pairs())
        stats, per_prompt = teacher_diagnostics(result, pids, learners)
        write_json(out / split / "teacher_diagnostics.json", stats)
        write_jsonl(out / split / "teacher_per_prompt.jsonl", per_prompt)
        write_json(out / split / "independent_numpy_crosscheck.json", crosscheck)
        outputs[split] = {"n_prompts": len(pids), "n_pairs": len(pids)*28, "pairs_path": str(pair_path),
            "pairs_sha256": file_hash(pair_path), "solver_solution_sha256": solver_hash, "pool_sha256": pool_hashes[split],
            "certificate": certificate, "nash_dual_certificate_status": "not_applicable",
            "independent_crosscheck_sha256": file_hash(out / split / "independent_numpy_crosscheck.json"),
            "seconds": time.monotonic()-split_start}
        write_json(out / split / "complete.json", outputs[split])
        print(json.dumps({"split": split, "phase": "complete", "n_pairs": len(pids)*28,
                          "stationarity": crosscheck["independent_stationarity_inf"], "seconds": outputs[split]["seconds"]}), flush=True)
    provenance_path = out / "dataset_provenance.json"
    write_json(provenance_path, {**shared, "train_pool_sha256": pool_hashes["train"],
        "dev_pool_sha256": pool_hashes["dev"], "test_pool_sha256": pool_hashes.get("test"),
        "solver_splits": outputs, "control": control})
    env = dict(os.environ, HF_DATASETS_CACHE=str(out / "arrow_cache"), PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run([sys.executable, "-m", "mnpo_scripts.prepare_nbpo_dataset", "--train", str(out / "pairs/train.jsonl"),
        "--dev", str(out / "pairs/dev.jsonl"), "--output", str(dataset_out), "--provenance", str(provenance_path)],
        env=env, capture_output=True, text=True)
    if completed.returncode:
        write_json(out / "dataset_materialization_failure.json", {"returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr, "optional_training_authorized": False})
        raise RuntimeError("Optional dataset materialization failed; diagnostic preserved without changing primary files")
    dataset_manifest = file_hash(dataset_out / "precompute_manifest.json")
    compatibility = {"status": "prepared_only_not_training_authorization", "arm": ARM,
        "trainer_validation": "Canonical target/mass/center/token contract supports utilitarian aggregation; no trainer change needed",
        "required_config_changes_if_later_authorized": {"loss_type": "nbpo_wbc", "dataset_mixer": {str(dataset_out): 1.0},
            "nbpo_expected_dataset_manifest_sha256": dataset_manifest, "output_dir": "new optional arm directory only",
            "run_name": ARM},
        "retain_primary_fields": ["model_revision", "model_name_or_path", "nbpo_target_mode", "nbpo_target_units",
            "nbpo_eta_already_included", "nbpo_logp_reduction", "max_length", "max_prompt_length", "seed",
            "learning_rate", "optim", "lr_scheduler_type", "max_steps", "gradient_accumulation_steps"],
        "dataset_splits": ["train", "dev"], "test_not_training_eval": True,
        "optional_training_authorized": False, "no_existing_primary_config_modified": True}
    write_json(out / "main_config_compatibility.json", compatibility)
    write_json(out / "complete.json", {"splits": outputs, "control": control, "dataset_path": str(dataset_out),
        "dataset_manifest_sha256": dataset_manifest, "dataset_prepare_stdout": completed.stdout,
        "seconds": time.monotonic()-start, "source_sha256": file_hash(__file__),
        "optional_training_authorized": False, "gpu_model_forwards": 0})
    print(json.dumps({"teacher": str(out), "dataset": str(dataset_out), "manifest_sha256": dataset_manifest,
                      "weights_raw": weights.tolist(), "optional_training_authorized": False}), flush=True)


if __name__ == "__main__":
    main()
