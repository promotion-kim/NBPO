#!/usr/bin/env python3
"""One outer stage of NBPO Algorithm 1 -- the finite-pool NBPO realization -- with
the checkpoint gate and end-to-end provenance binding.

Pipeline (the finite-pool math never loads an LLM):

1.  load the stage config and the versioned objective rubrics (missing or
    unavailable rubrics fail loudly -- nothing is invented);
2.  enforce that the training, held-out monitoring, and final-evaluation
    prompt sets are pairwise DISJOINT (hard error);
3.  judge the training preference matrix with the TRAINING judge
    (policy-vs-reference, plus unordered reference pairs for d_k), both
    presentation orders, retries, loud completeness failure, backend
    cardinality check;
4.  build the centered tensors (A_ref exactly skew-symmetric, zero diagonal),
    run the projected dual solve of Eq. (27) on the frozen pool, and write the
    solver artifact (raw lambda, V, d, s, inverse-surplus and projected KKT
    residuals, fixed-point and extra-map residuals, nu_update / nu_final_policy
    with hashes);
5.  build the pair targets (Eq. (24); z_k drawn per pair and objective from
    nu_update) -- the ONE-SHOT neural realization that follows is fit once
    from these frozen-pool targets, it is not re-solved per dual step;
6.  real mode: precompute logps (sum reduction) for the pairs, verify the
    sidecar binds history0 to the parent checkpoint's content fingerprint,
    materialize a run_mnpo YAML, parse it with run_mnpo's own argument
    dataclasses BEFORE any model loads, and train into a temporary candidate
    directory;
7.  decode the candidate on the monitoring prompts SYNCHRONOUSLY through
    ``scripts/nbpo/decode_candidate.py``; the resulting manifest must bind to
    the candidate fingerprint, the exact monitoring prompt set, every
    configured seed, and every response file hash -- otherwise the stage
    aborts; every candidate and reference seed file must carry EXACTLY the
    monitoring prompt set;
8.  judge the monitoring matrix with the MONITORING judge (never silently the
    training judge) and evaluate with the game-value evaluator (Nash welfare
    null under any nonpositive surplus, never clamped);
9.  gate (Algorithm 1 lines 11-15): ``min_k s_hat_k <= 0`` -> REJECT, retain
    ``pi_t``, write a rejection record; else promote the candidate into a
    versioned checkpoint directory and atomically repoint ``pi_{t+1}``.

``--dry-run`` exercises steps 1-5 and 8-9 with the mock judge and a marker
checkpoint, taking the monitoring responses from the explicitly named
``dry_run.candidate_policy_files`` fixture section (manifest binding is
skipped there and recorded as skipped). Real mode never reads that section.

Disclosures: ``solver.R = 1`` is the manuscript's practical fixed-point
approximation (its residual is reported); the neural policy is realized once
after the frozen-pool dual, not inside it.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from mnpo_scripts.nbpo_core import (
    uniform_policy,
    validate_centered_preference_tensor,
    validate_reference_tensor,
)
from mnpo_scripts.nbpo_solver import solve_nbpo_dual, solve_weighted_policy
from mnpo_scripts.precompute_provenance import (
    checkpoint_fingerprint,
    read_precompute_meta,
    sha256_file_hex,
)
from scripts.nbpo.build_nbpo_pairs import build_pairs_from_artifacts
from scripts.nbpo.build_preference_tensor import (
    aggregate_cells,
    fill_policy_tensor,
    fill_reference_tensor,
)
from scripts.nbpo.eval_game_value import evaluate_game_value
from scripts.nbpo.judge_pairwise_matrix import (
    MockJudge,
    VllmJudge,
    build_task_grid,
    rubric_identity_hash,
    run_judging,
)
from scripts.nbpo.nbpo_common import (
    check_pool_cardinality,
    check_reproduction_invariants,
    implementation_contract,
    jsonl_has_records,
    judge_config_hash,
    load_objective_rubrics,
    load_response_files,
    normalize_judge_config,
    read_jsonl,
    response_pool_hash,
    sha256_file,
    sha256_text,
    verdict_cache_path,
    write_json,
)
from scripts.nbpo.response_manifest import (
    check_stage_zero_identity,
    load_prompt_rows,
    pool_manifest_summary,
    prompt_text_hashes,
    verify_manifest,
)

JUDGE_ROLES = ("training", "monitoring", "final_eval")
NBPO_TRAINER_INVARIANTS = {
    "loss_type": "nbpo",
    "logp_reduction": "sum",
    "max_history_t": 1,
    "history_weights": [1.0],
    "reference_anchor_weight": 0.0,
    "preference_sft_weight": 0.0,
    "nbpo_target_column": "nbpo_weighted_z",
}


def _step(name: str, detail: str = "") -> None:
    print(f"[nbpo-stage] step={name} status=ok{' detail=' + detail if detail else ''}", flush=True)


def _resolve(base: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (base / q)


def _specs(base: Path, mapping: dict) -> list:
    return [f"{seed}={_resolve(base, path)}" for seed, path in sorted(mapping.items())]


# Backward-compatible name; the content fingerprint now hashes every weight shard.
def hash_checkpoint_dir(path: Path) -> str:
    return checkpoint_fingerprint(str(path))


# --------------------------------------------------------------------------- #
# prompt sets
# --------------------------------------------------------------------------- #
def load_prompt_set(path: Path):
    """Prompt ids from a json list of ids or a jsonl of {prompt_id, prompt}; returns (ids, rows)."""
    text = path.read_text()
    if path.suffix == ".json":
        ids = [str(x) for x in json.loads(text)]
        return set(ids), None
    rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    ids = {str(r.get("prompt_id", i)) for i, r in enumerate(rows)}
    return ids, rows


def check_disjoint(train_ids, monitoring_ids, final_ids) -> None:
    """Pairwise-disjoint prompt sets; overlap is exactly the leak class the audit found."""
    overlaps = {
        "train∩monitoring": sorted(set(train_ids) & set(monitoring_ids)),
        "train∩final_eval": sorted(set(train_ids) & set(final_ids)),
        "monitoring∩final_eval": sorted(set(monitoring_ids) & set(final_ids)),
    }
    bad = {k: v for k, v in overlaps.items() if v}
    if bad:
        raise ValueError(
            "prompt sets must be pairwise disjoint (monitoring is never reused for "
            f"final reporting): overlapping ids {bad}"
        )


def check_prompt_set_equality(actual, expected, what: str) -> None:
    """``actual == expected`` exactly; reports both missing and unexpected ids."""
    actual, expected = set(map(str, actual)), set(map(str, expected))
    if actual != expected:
        raise ValueError(
            f"{what}: prompt set differs from the monitoring set -- "
            f"missing={sorted(expected - actual)[:10]} (n={len(expected - actual)}), "
            f"unexpected={sorted(actual - expected)[:10]} (n={len(actual - expected)})"
        )


def check_seed_files_cover_prompts(specs, expected_ids, what: str) -> None:
    """Every seed file independently carries exactly the expected prompt ids."""
    files = load_response_files(specs)
    for seed, rows in files.items():
        check_prompt_set_equality(rows.keys(), expected_ids, f"{what} seed {seed}")


# --------------------------------------------------------------------------- #
# judges
# --------------------------------------------------------------------------- #
def judge_config(cfg: dict, role: str, allow_legacy_flat: bool = False) -> dict:
    """``judge.<role>`` (training / monitoring / final_eval).

    The flat legacy ``judge:`` block gives all three roles one identity, which
    is exactly how monitoring metadata ended up standing in for the training and
    final-eval provenance. It is rejected unless the config explicitly opts in
    with ``allow_legacy_flat_judge_config: true``.
    """
    j = cfg.get("judge") or {}
    if role in j and isinstance(j[role], dict):
        return dict(j[role])
    if "backend" in j:  # flat legacy form
        if not allow_legacy_flat:
            raise ValueError(
                f"config uses the flat legacy `judge:` block, which cannot distinguish the "
                f"{role!r} role from the others. Split it into judge.training / "
                "judge.monitoring / judge.final_eval, or set "
                "allow_legacy_flat_judge_config: true to accept one shared identity.")
        return dict(j)
    raise ValueError(f"config needs judge.{role} (backend, model_path, ...)")


def make_backend(judge_cfg: dict, objectives_config: Path, objectives, identity: dict):
    """Construct the backend and hand it the EFFECTIVE decoding config to apply."""
    if judge_cfg["backend"] == "mock":
        return MockJudge(invalid_rate=float(judge_cfg.get("mock_invalid_rate", 0.0)),
                         judge_config=identity)
    loaded = load_objective_rubrics(objectives_config, objectives)
    return VllmJudge(judge_cfg["model_path"], loaded["rubrics"],
                     max_model_len=int(judge_cfg.get("max_model_len", 8192)),
                     tensor_parallel_size=int(judge_cfg.get("tensor_parallel_size", 1)),
                     batch_size=int(judge_cfg.get("batch_size", 512)),
                     judge_config=identity,
                     chat_template_kwargs=identity.get("chat_template_kwargs"))


def check_disjoint_by_text(named_prompt_files: dict) -> dict:
    """Pairwise disjointness by prompt_id AND by normalized prompt TEXT.

    Checking ids alone lets the same question appear in training and validation
    under two different ids, which is leakage that every id-level check passes.
    Text is normalized (collapsed whitespace, casefolded) before hashing so
    cosmetic differences do not hide a duplicate.
    """
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", str(s)).strip().casefold()

    ids, texts = {}, {}
    for name, rows in named_prompt_files.items():
        ids[name] = {str(pid) for pid, _ in rows}
        texts[name] = {sha256_text(norm(txt)): str(pid) for pid, txt in rows}
    problems = []
    names = sorted(named_prompt_files)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared_ids = ids[a] & ids[b]
            if shared_ids:
                problems.append(f"{a} and {b} share prompt ids {sorted(shared_ids)[:5]}")
            shared_text = set(texts[a]) & set(texts[b])
            if shared_text:
                examples = [(texts[a][h], texts[b][h]) for h in sorted(shared_text)[:5]]
                problems.append(
                    f"{a} and {b} share prompt TEXT under different ids {examples}")
    if problems:
        raise ValueError("prompt sets must be disjoint by id AND by text: "
                         + "; ".join(problems))
    return {name: {"n_prompts": len(ids[name]),
                   "n_distinct_texts": len(texts[name])} for name in names}


def build_validation_pairs(cfg, base, workdir, objectives, objectives_config, judge_cfg,
                           solver_solution, beta, stage, reproduction_mode, allow_partial,
                           mu_fingerprint, parent_fingerprint, pairs_dir):
    """Build pairs_test.jsonl from EXTERNAL validation prompts and their own responses.

    ``targets.test_prompts`` holds prompts out of the pre-generated training
    response pool -- a split AFTER generation, not a validation set. When
    ``validation:`` is configured, the validation prompts get their own learner
    and comparator pools (manifest-bound to pi_t and mu), their own judged
    tensor, and their own Eq. (26) targets. lambda is NOT re-solved: the dual
    weights are a property of the training problem, so the training solution's
    raw lambda is applied to the validation tensor and only the opponent nu is
    recomputed there, exactly as the training targets do at the proximal centre.
    """
    vcfg = cfg["validation"]
    v_prompts_path = _resolve(base, vcfg["prompts_file"])
    v_ids, _ = load_prompt_set(v_prompts_path)
    v_rows = load_prompt_rows(v_prompts_path)
    v_text_hashes = prompt_text_hashes(v_rows)
    gen_ctk = (cfg.get("generation") or {}).get("chat_template_kwargs") or {}
    bindings = {}
    for key, role, want_fp in (("learner_manifest", "learner_pool", parent_fingerprint),
                               ("reference_manifest", "reference_comparator_pool",
                                mu_fingerprint)):
        if key not in vcfg:
            raise ValueError(f"validation requires validation.{key}")
        bindings[role] = verify_manifest(
            _resolve(base, vcfg[key]), expected_role=role, expected_fingerprint=want_fp,
            expected_prompt_ids=v_ids, expected_prompt_text_sha256=v_text_hashes,
            expected_prompts_file=v_prompts_path,
            expected_chat_template_kwargs=gen_ctk,
            allow_partial_prompt_intersection=allow_partial)
    v_policy_specs = [f"{s}={p}" for s, p in
                      sorted(bindings["learner_pool"]["seed_files"].items())]
    v_ref_specs = [f"{s}={p}" for s, p in
                   sorted(bindings["reference_comparator_pool"]["seed_files"].items())]
    v_tensor_dir, n_cells = judge_and_build_tensors(
        v_policy_specs, v_ref_specs, objectives, objectives_config, judge_cfg,
        workdir, tag="validation", role="training", reproduction_mode=reproduction_mode,
        pool_bindings=bindings)

    # nu at the proximal centre on the VALIDATION tensor, with the TRAINING lambda.
    A_val = torch.from_numpy(np.load(v_tensor_dir / "tensor_policy.npz")["A"])
    A_val_ref = torch.from_numpy(np.load(v_tensor_dir / "tensor_ref.npz")["A"])
    mu_val = uniform_policy(A_val.shape[1], A_val.shape[3])
    pi_t_val = uniform_policy(A_val.shape[1], A_val.shape[2])
    lam = torch.tensor(solver_solution["lambda_raw"], dtype=torch.float64)
    sol = solve_weighted_policy(A_val, mu_val, pi_t_val, lam, beta,
                                float((solver_solution["config"] or {}).get("eta", 1.0)),
                                R=int((solver_solution["config"] or {}).get("R", 1)))
    v_solver_dir = workdir / "solver_validation"
    v_solver_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(v_solver_dir / "nu_update.npz", nu=sol.nu_update.numpy())
    np.savez_compressed(v_solver_dir / "nu_final_policy.npz", nu=sol.nu_final_policy.numpy())
    np.savez_compressed(v_solver_dir / "pi_star.npz", pi=sol.pi.numpy())
    v_solution = {
        **implementation_contract(
            dual_iterations=(solver_solution["config"] or {}).get("M"),
            fixed_point_steps=(solver_solution["config"] or {}).get("R")),
        "aggregation": solver_solution["aggregation"],
        "stage": int(stage),
        "objectives": objectives,
        "lambda_raw": list(solver_solution["lambda_raw"]),
        "lambda_source": "training dual solve (validation does NOT re-solve the dual)",
        "config": dict(solver_solution["config"] or {}),
        "input_hashes": {n: sha256_file(v_tensor_dir / n)
                         for n in ("tensor_policy.npz", "tensor_ref.npz", "meta.json")},
        "artifact_hashes": {n: sha256_file(v_solver_dir / n)
                            for n in ("nu_update.npz", "nu_final_policy.npz", "pi_star.npz")},
        "opponent_artifacts": {
            "nu_update.npz": {"artifact_kind": "regularized_opponent",
                              "source_policy": "proximal_centre",
                              "source_fixed_point_iteration": 0,
                              "used_for": "eq26_target"},
            "nu_final_policy.npz": {"artifact_kind": "regularized_opponent",
                                    "source_policy": "final_policy",
                                    "source_fixed_point_iteration": int(
                                        (solver_solution["config"] or {}).get("R", 1)),
                                    "used_for": "diagnostics"},
        },
    }
    write_json(v_solver_dir / "solution.json", v_solution)
    v_pairs_dir = workdir / "pairs_validation"
    v_summary = build_pairs_from_artifacts(
        v_tensor_dir, v_solver_dir, v_policy_specs, v_pairs_dir,
        target_mode=(cfg.get("targets") or {}).get("mode", "sampled"),
        seed=int((cfg.get("targets") or {}).get("seed", 42)), stage=stage,
        reproduction_mode=reproduction_mode,
        learner_manifest_sha256=bindings["learner_pool"]["manifest_sha256"])
    # pairs_test.jsonl comes EXCLUSIVELY from the validation artifacts.
    src = v_pairs_dir / "pairs_train.jsonl"
    dst = pairs_dir / "pairs_test.jsonl"
    dst.write_text(src.read_text())
    got = {json.loads(l)["prompt_id"] for l in dst.read_text().splitlines() if l.strip()}
    if got != set(map(str, v_ids)):
        raise ValueError(
            "validation pair prompt ids do not equal the validation prompts file: "
            f"missing={sorted(set(map(str, v_ids)) - got)[:5]} "
            f"unexpected={sorted(got - set(map(str, v_ids)))[:5]}")
    return {"prompts_file": str(v_prompts_path),
            "prompts_file_sha256": sha256_file_hex(str(v_prompts_path)),
            "n_prompts": len(v_ids), "n_pairs": len(dst.read_text().splitlines()),
            "tensor_dir": str(v_tensor_dir), "judged_cells": n_cells,
            "manifests": {r: pool_manifest_summary(b) for r, b in bindings.items()},
            "lambda_source": v_solution["lambda_source"],
            "summary": v_summary}


def build_judge_provenance(judges: dict, artifacts: dict, objectives_config, objectives) -> dict:
    """Per-role judge provenance, each from that role's own artifact.

    A role whose artifact does not exist -- final evaluation, when it has not
    been run -- is recorded as ``configured_not_executed`` with its configured
    identity and NO verdict or task-set hash. Nothing is borrowed from another
    role, so a configured evaluation can never read as a completed one.
    """
    loaded = load_objective_rubrics(objectives_config, objectives)
    rubric_sha = rubric_identity_hash(loaded["rubrics"])
    out = {}
    for role in JUDGE_ROLES:
        cfg = judges.get(role) or {}
        identity = normalize_judge_config(cfg, role, rubric_sha256=rubric_sha,
                                          max_retries=cfg.get("max_retries", 2))
        meta = artifacts.get(role)
        if meta is None:
            out[role] = {
                "status": "configured_not_executed",
                "model": cfg.get("model_path"),
                "model_revision": cfg.get("model_revision"),
                "backend": cfg.get("backend"),
                "decoding": {k: identity[k] for k in
                             ("temperature", "top_p", "top_k", "max_tokens",
                              "retry_temperature", "max_retries")},
                "rubric_content_hash": rubric_sha,
                "judge_config_hash": judge_config_hash(identity),
                "verdict_artifact_hash": None,
                "task_set_hash": None,
            }
            continue
        out[role] = {
            "status": "executed",
            "model": meta.get("judge_models", [cfg.get("model_path")])[0],
            "model_revision": (meta.get("judge_effective_config") or {}).get("judge_revision"),
            "backend": meta.get("judge_backend"),
            "decoding": meta.get("judge_decoding"),
            "rubric_versions": meta.get("rubric_versions"),
            "rubric_config_hash": meta.get("rubric_config_hash"),
            "rubric_content_hash": meta.get("rubric_content_hash"),
            "judge_config_hash": meta.get("judge_config_hash"),
            "verdict_artifact_hash": meta.get("verdicts_hash"),
            "task_set_hash": meta.get("response_pool_hash"),
        }
    return out


def judge_and_build_tensors(policy_specs, reference_specs, objectives, objectives_config,
                            judge_cfg, out_dir: Path, tag: str, role: str = None,
                            reproduction_mode: bool = True, pool_bindings: dict = None):
    """Judge a matrix and build the tensor artifact, with content-addressed caching.

    The verdict cache lives at ``verdicts/<pool_hash>/<judge_config_hash>/`` and
    every task id is the sha256 of the exact texts plus the effective judge
    settings, so a new candidate never inherits an older one's verdicts and two
    judge configurations never share a file.
    """
    role = role or tag
    policy = load_response_files(policy_specs, reproduction_mode=reproduction_mode)
    reference = load_response_files(reference_specs, reproduction_mode=reproduction_mode)
    cardinality = check_pool_cardinality(policy, reference, reproduction_mode)
    tasks = build_task_grid(policy, reference, objectives, include_reference_learner=True)
    loaded = load_objective_rubrics(objectives_config, objectives)
    identity = normalize_judge_config(judge_cfg, role,
                                      rubric_sha256=rubric_identity_hash(loaded["rubrics"]),
                                      max_retries=judge_cfg.get("max_retries", 2))
    if reproduction_mode:
        # A configured kwarg the tokenizer rejects must fail, not be dropped.
        identity["strict_chat_template_kwargs"] = True
    backend = make_backend(judge_cfg, objectives_config, objectives, identity)
    # Render ONE task first, then freeze: effective_config() read before any render
    # reports the REQUESTED kwargs, so a silently-dropped one would be baked into
    # the cache identity as if it had been applied.
    identity = {**identity, **backend.preflight(tasks[0])}
    pool_hash = response_pool_hash(policy, reference)
    cfg_hash = judge_config_hash(identity)
    verdicts = verdict_cache_path(out_dir, pool_hash, cfg_hash)
    n = run_judging(tasks, backend, verdicts, judge_model=judge_cfg["model_path"],
                    rubric_version=loaded["version"],
                    max_retries=int(judge_cfg.get("max_retries", 2)),
                    judge_identity=identity)
    after = {**identity, **backend.effective_config()}
    if judge_config_hash(after) != cfg_hash:
        raise RuntimeError(
            "the judge's effective configuration changed during judging: the verdict cache "
            "key no longer describes what was applied")
    rows = read_jsonl(verdicts)
    payoff = aggregate_cells(rows)
    prompt_ids = sorted(set.intersection(*[set(m) for m in policy.values()],
                                         *[set(m) for m in reference.values()]))
    policy_ids = [f"policy:{s}" for s in sorted(policy)]
    ref_ids = [f"ref:{s}" for s in sorted(reference)]
    A_policy = fill_policy_tensor(payoff, objectives, prompt_ids, policy_ids, ref_ids)
    A_ref, skew_stats = fill_reference_tensor(payoff, objectives, prompt_ids, ref_ids)
    validate_centered_preference_tensor(torch.from_numpy(A_policy), "A_policy")
    # judge_and_build_tensors always builds the reference tensor from ONE response
    # set (unordered pairs of the comparator seeds), so the construction is
    # shared_pool and exact skew symmetry is a hard requirement.
    validate_reference_tensor(torch.from_numpy(A_ref), "A_ref", "shared_pool")
    tensor_dir = out_dir / f"tensor_{tag}"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tensor_dir / "tensor_policy.npz", A=A_policy)
    np.savez_compressed(tensor_dir / "tensor_ref.npz", A=A_ref)
    write_json(tensor_dir / "meta.json", {
        "schema_version": 1,
        **implementation_contract(),
        "objectives": objectives,
        "prompt_ids": prompt_ids,
        "policy_learner_ids": policy_ids,
        "comparator_ids": ref_ids,
        "generation_seeds": {"policy": {f"policy:{s}": s for s in sorted(policy)},
                             "reference": {f"ref:{s}": s for s in sorted(reference)}},
        "judge_role": role,
        "judge_models": [judge_cfg["model_path"]],
        "judge_backend": judge_cfg["backend"],
        "judge_effective_config": identity,
        "judge_config_hash": cfg_hash,
        "judge_effective_chat_template_kwargs": identity.get("chat_template_kwargs"),
        "renderer_schema_version": identity.get("renderer_schema_version"),
        "judge_decoding": {k: identity.get(k) for k in
                           ("temperature", "top_p", "top_k", "max_tokens",
                            "retry_temperature", "max_retries")},
        "rubric_versions": [loaded["version"]],
        "rubric_config_hash": sha256_file(objectives_config),
        "rubric_content_hash": identity["rubric_sha256"],
        "response_pool_hash": pool_hash,
        # Which response texts this tensor -- and therefore lambda and nu --
        # describe. build_nbpo_pairs re-hashes the files it loads against these,
        # so pair texts cannot come from a pool the tensor never saw.
        "learner_response_pool_hash": response_pool_hash(policy),
        "reference_response_pool_hash": response_pool_hash(reference),
        "learner_manifest_sha256": ((pool_bindings or {}).get("learner_pool") or {}).get(
            "manifest_sha256"),
        "reference_manifest_sha256": (
            (pool_bindings or {}).get("reference_comparator_pool") or {}).get("manifest_sha256"),
        "response_text_sha256": {
            "policy": {f"policy:{s}": {pid: sha256_text(str(r.get("generated_text", "")))
                                       for pid, r in sorted(rows.items())}
                       for s, rows in sorted(policy.items())},
            "reference": {f"ref:{s}": {pid: sha256_text(str(r.get("generated_text", "")))
                                       for pid, r in sorted(rows.items())}
                          for s, rows in sorted(reference.items())},
        },
        "prompt_text_sha256": {pid: sha256_text(str(next(iter(policy.values()))[pid]["prompt"]))
                               for pid in prompt_ids},
        "verdicts_path": str(verdicts),
        "verdicts_hash": sha256_file(verdicts),
        "both_orders_required": True,
        "self_pairs": "identity_zero",
        "reference_skew": skew_stats,
        "reference_construction": "shared_pool",
        "tensor_kind": "centered_preference",
        "cardinality": cardinality,
        "shape_policy": list(A_policy.shape),
        "shape_ref": list(A_ref.shape),
    })
    return tensor_dir, n


# --------------------------------------------------------------------------- #
# run_mnpo config materialization
# --------------------------------------------------------------------------- #
def materialize_run_config(stage_cfg: dict, parent_checkpoint, precomputed_dataset_dir,
                           candidate_output_dir, output_yaml, dataset_splits=None,
                           expected_hashes: dict = None) -> dict:
    """Write the run_mnpo YAML for the candidate fit and validate it with run_mnpo's own parser.

    Field names follow the released run_mnpo configs (training_configs/ronpo/*.yaml,
    the eta_x20 arm): ``model_name_or_path``, ``dataset_mixer``, ``dataset_splits``,
    trainer fields from ``stage_cfg["trainer"]``, and ``output_dir``. The NBPO
    invariants are forced last so a stage config cannot silently override them.
    The YAML is parsed with ``H4ArgumentParser((ModelArguments, DataArguments,
    MNPOConfig))`` -- the exact parser ``mnpo_scripts.run_mnpo`` uses -- and
    ``validate_nbpo_args`` is applied, all before any model is loaded.
    """
    trainer = dict(stage_cfg.get("trainer") or {})
    run = {
        "model_name_or_path": str(parent_checkpoint),
        "dataset_mixer": {str(precomputed_dataset_dir): 1.0},
        # dataset_splits must describe what was actually SAVED. Declaring "test"
        # for an artifact that has no test split makes run_mnpo's loader fail on
        # a missing split; the precompute step only builds one when the pair
        # builder actually held prompts out.
        "dataset_splits": list(dataset_splits or ["train"]),
    }
    run.update(trainer)
    run.update(NBPO_TRAINER_INVARIANTS)
    # Bind this training run to the exact artifacts of THIS stage. Without these
    # the trainer only sees the sidecar, which travels with the dataset and so
    # cannot tell a stale-but-self-consistent artifact from the right one.
    for key, value in (expected_hashes or {}).items():
        if value is not None:
            run[key] = value
    run["output_dir"] = str(candidate_output_dir)
    run.setdefault("run_name", f"nbpo-stage{stage_cfg.get('stage', 0)}")
    run.setdefault("save_strategy", "no")
    run.setdefault("save_only_model", True)
    run.setdefault("report_to", [])
    output_yaml = Path(output_yaml)
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(yaml.safe_dump(run, sort_keys=False))
    parse_run_config(output_yaml)
    return run


def parse_run_config(path):
    """Parse a run_mnpo YAML with run_mnpo's own dataclasses; raises before model loading."""
    from alignment import DataArguments, H4ArgumentParser, ModelArguments
    from mnpo_scripts.mnpo_config import MNPOConfig
    from mnpo_scripts.mnpo_trainer import validate_nbpo_args

    parser = H4ArgumentParser((ModelArguments, DataArguments, MNPOConfig))
    model_args, data_args, training_args = parser.parse_yaml_file(os.path.abspath(str(path)))
    if str(training_args.loss_type).lower() != "nbpo":
        raise ValueError(f"materialized run config has loss_type={training_args.loss_type!r}")
    validate_nbpo_args(training_args)   # config-level invariants
    if not data_args.dataset_mixer:
        raise ValueError("materialized run config has an empty dataset_mixer")
    return model_args, data_args, training_args


# --------------------------------------------------------------------------- #
# manifest binding
# --------------------------------------------------------------------------- #
def verify_decode_manifest(manifest_path: Path, candidate_dir: Path, monitoring_ids,
                           required_seeds, prompts_file=None, prompt_text_sha256=None,
                           decode_params: dict = None,
                           chat_template_kwargs: dict = None) -> dict:
    """Bind monitoring responses to the candidate checkpoint; returns seed -> path.

    Delegates to the shared response-pool verifier, so the candidate pool is
    checked exactly like the training pools: content fingerprint of the
    generating checkpoint, sha256 of the prompt FILE, per-prompt text hashes
    (ids alone do not identify a prompt set), exact -- not subset -- seed
    equality, the decode parameters actually used, the chat-template kwargs
    actually applied, and the sha256 of every response file. Relative response
    paths resolve against the manifest's own directory.
    """
    actual_fp = checkpoint_fingerprint(str(candidate_dir))
    binding = verify_manifest(
        manifest_path, expected_role="monitoring_candidate_pool",
        expected_fingerprint=actual_fp,
        expected_prompt_ids=monitoring_ids,
        expected_prompt_text_sha256=prompt_text_sha256,
        expected_prompts_file=prompts_file,
        required_seeds=[str(s) for s in required_seeds],
        expected_decode_params=decode_params,
        expected_chat_template_kwargs=chat_template_kwargs,
    )
    man = binding["manifest"]
    return {
        "seed_files": {f"s{k}" if not str(k).startswith("s") else str(k): v
                       for k, v in binding["seed_files"].items()},
        "candidate_fingerprint": actual_fp,
        "manifest_sha256": binding["manifest_sha256"],
        "prompts_file_sha256": man.get("prompt_file_sha256"),
        "decode_params": man.get("decode_params"),
        "chat_template_kwargs": man.get("chat_template_kwargs"),
    }


# --------------------------------------------------------------------------- #
# gate + promotion
# --------------------------------------------------------------------------- #
def promote_candidate(candidate_dir: Path, promote_to: Path, stage: int, fingerprint: str) -> dict:
    """Versioned, atomic promotion: move the candidate into
    ``<promote_to>.versions/stage<t>_<fp12>`` and atomically repoint the
    ``promote_to`` symlink. A pre-existing real directory at ``promote_to`` is
    renamed aside first; nothing is ever partially overwritten.
    """
    promote_to = Path(promote_to)
    versions = promote_to.parent / (promote_to.name + ".versions")
    versions.mkdir(parents=True, exist_ok=True)
    target = versions / f"stage{stage}_{fingerprint[:12]}"
    if target.exists():
        raise FileExistsError(f"versioned checkpoint already exists: {target}")
    os.replace(candidate_dir, target)                      # same filesystem, atomic
    if promote_to.exists() or promote_to.is_symlink():
        if promote_to.is_symlink():
            promote_to.unlink()
        else:
            aside = promote_to.parent / f"{promote_to.name}.prev_{int(time.time())}"
            os.replace(promote_to, aside)
    tmp_link = promote_to.parent / f".{promote_to.name}.link.tmp"
    if tmp_link.is_symlink() or tmp_link.exists():
        tmp_link.unlink()
    os.symlink(os.path.relpath(target, promote_to.parent), tmp_link)
    os.replace(tmp_link, promote_to)                        # atomic symlink swap
    record = {"stage": stage, "fingerprint": fingerprint, "versioned_dir": str(target),
              "symlink": str(promote_to), "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_json(versions / "PROMOTION.json", record)
    return record


def apply_gate(min_surplus: float, parent_dir: Path, candidate_dir: Path,
               promote_to: Path, stage: int = 0, fingerprint: str = None) -> dict:
    """Algorithm 1 lines 11-15: reject on any nonpositive held-out surplus, else promote."""
    if min_surplus <= 0:
        return {"accepted": False, "promoted_path": str(parent_dir),
                "reason": f"min held-out surplus {min_surplus:.6f} <= 0; "
                          "stage flagged empirically infeasible, pi_t retained"}
    fp = fingerprint or checkpoint_fingerprint(str(candidate_dir))
    rec = promote_candidate(candidate_dir, promote_to, stage, fp)
    return {"accepted": True, "promoted_path": rec["versioned_dir"], "symlink": str(promote_to),
            "reason": f"min held-out surplus {min_surplus:.6f} > 0", "promotion": rec}


# --------------------------------------------------------------------------- #
# stage
# --------------------------------------------------------------------------- #
def _run(cmd: str, name: str) -> None:
    print(f"[nbpo-stage] running commands.{name}: {cmd}", flush=True)
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    env.setdefault("MKL_THREADING_LAYER", "GNU")   # torch+numpy in a fresh interpreter
    subprocess.run(cmd, shell=True, check=True, env=env)


def run_stage(config_path: Path, workdir: Path, dry_run: bool) -> dict:
    base = config_path.parent
    cfg = yaml.safe_load(config_path.read_text())
    workdir.mkdir(parents=True, exist_ok=True)
    stage = int(cfg.get("stage", 0))
    objectives = list(cfg["objectives"]["names"])
    objectives_config = _resolve(base, cfg["objectives"]["config"])
    reproduction_mode = bool(cfg.get("reproduction_mode", True))
    allow_flat_judge = bool(cfg.get("allow_legacy_flat_judge_config", False))
    allow_partial = bool(cfg.get("allow_partial_prompt_intersection", False))
    if reproduction_mode and allow_partial:
        raise ValueError("allow_partial_prompt_intersection is diagnostic-only and cannot be "
                         "combined with reproduction_mode")
    # Every invariant the manuscript numbers depend on, asserted before any work.
    invariants = check_reproduction_invariants(
        cfg.get("solver") or {},
        {**(cfg.get("trainer") or {}), **NBPO_TRAINER_INVARIANTS},
    ) if reproduction_mode else {"reproduction_mode": False}
    _step("reproduction_mode", f"{reproduction_mode} invariants={invariants}")

    # 1. Rubrics
    load_objective_rubrics(objectives_config, objectives)
    _step("rubrics", f"{len(objectives)} objectives from {objectives_config.name}")

    # 2. Disjointness of train / monitoring / final-eval prompts.
    gate_cfg = cfg.get("gate") or {}
    gate_base_cfg = workdir if dry_run else base
    parent_model_path = _resolve(gate_base_cfg, gate_cfg["parent_model"])
    mu_model_path = _resolve(gate_base_cfg, gate_cfg.get("reference_model",
                                                         gate_cfg["parent_model"]))
    pool_manifests = cfg.get("response_manifests") or {}
    pool_bindings = {}
    if pool_manifests:
        # Training pools bound to the checkpoints that produced them: the learner
        # pool to the proximal centre pi_t, the comparator pool to the reference
        # policy mu. Bare file paths could not express either binding.
        parent_fp_cfg = checkpoint_fingerprint(str(parent_model_path))
        mu_fp_cfg = checkpoint_fingerprint(str(mu_model_path))
        check_stage_zero_identity(parent_fp_cfg, mu_fp_cfg, stage)
        # The chat-template kwargs the pools were decoded under must match the
        # config: a Qwen pool decoded WITHOUT enable_thinking:false contains
        # reasoning traces, which is a different pool of text entirely.
        gen_ctk = (cfg.get("generation") or {}).get("chat_template_kwargs") or {}
        for role, want_fp in (("learner_pool", parent_fp_cfg),
                              ("reference_comparator_pool", mu_fp_cfg)):
            if role not in pool_manifests:
                raise ValueError(f"response_manifests must declare {role!r}")
            pool_bindings[role] = verify_manifest(
                _resolve(base, pool_manifests[role]), expected_role=role,
                expected_fingerprint=want_fp,
                expected_chat_template_kwargs=gen_ctk,
                allow_partial_prompt_intersection=allow_partial)
        policy_specs = [f"{s}={p}" for s, p in
                        sorted(pool_bindings["learner_pool"]["seed_files"].items())]
        reference_specs = [f"{s}={p}" for s, p in
                           sorted(pool_bindings["reference_comparator_pool"]["seed_files"].items())]
        _step("response_manifests",
              f"learner bound to pi_t {parent_fp_cfg[:12]}, comparator to mu {mu_fp_cfg[:12]}")
    elif reproduction_mode:
        raise ValueError(
            "reproduction_mode requires response_manifests.learner_pool and "
            "response_manifests.reference_comparator_pool: bare response file paths "
            "cannot bind the pools to pi_t and mu")
    else:
        policy_specs = _specs(base, cfg["responses"]["policy_files"])
        reference_specs = _specs(base, cfg["responses"]["reference_files"])
    train_pools = load_response_files(policy_specs, reproduction_mode=reproduction_mode)
    ref_pools = load_response_files(reference_specs, reproduction_mode=reproduction_mode)
    if reproduction_mode:
        # Exact equality, never an intersection: an intersection silently drops
        # prompts that one seed failed to decode and changes the game solved.
        all_ids = sorted(set().union(*[set(m) for m in train_pools.values()],
                                     *[set(m) for m in ref_pools.values()]))
        for label, pools in (("learner", train_pools), ("reference", ref_pools)):
            for seed, rows in pools.items():
                check_prompt_set_equality(rows.keys(), all_ids, f"{label} seed {seed}")
        train_ids = set(all_ids)
    else:
        train_ids = set.intersection(*[set(m) for m in train_pools.values()])
    mon_prompts_path = _resolve(base, cfg["monitoring"]["prompts_file"])
    monitoring_ids, _ = load_prompt_set(mon_prompts_path)
    monitoring_text_hashes = prompt_text_hashes(load_prompt_rows(mon_prompts_path))
    final_prompts_path = _resolve(base, cfg["final_eval"]["prompts_file"])
    final_ids, _ = load_prompt_set(final_prompts_path)
    check_disjoint(train_ids, monitoring_ids, final_ids)
    # ... and by prompt TEXT, so the same question under two ids is caught too.
    train_prompt_rows = [(pid, next(iter(train_pools.values()))[pid]["prompt"])
                         for pid in sorted(train_ids)]
    named = {"train": train_prompt_rows,
             "monitoring": load_prompt_rows(mon_prompts_path),
             "final_eval": load_prompt_rows(final_prompts_path)}
    if cfg.get("validation"):
        named["validation"] = load_prompt_rows(_resolve(base, cfg["validation"]["prompts_file"]))
    disjointness = check_disjoint_by_text(named)
    _step("disjointness", f"{ {k: v['n_prompts'] for k, v in disjointness.items()} } "
                          "(disjoint by prompt id AND normalized prompt text)")

    # 3-4. Training judge -> tensors -> dual solve.
    judges = {}
    for role in JUDGE_ROLES:
        jc = judge_config(cfg, role, allow_legacy_flat=allow_flat_judge)
        if dry_run and jc["backend"] != "mock":
            jc["backend"] = "mock"          # --dry-run never loads an LLM
        judges[role] = jc
    tensor_dir, n_cells = judge_and_build_tensors(
        policy_specs, reference_specs, objectives, objectives_config, judges["training"],
        workdir, tag="train", role="training", reproduction_mode=reproduction_mode,
        pool_bindings=pool_bindings)
    _step("judge_train", f"{n_cells} cells (both orders, retries enforced, "
                         f"judge={judges['training']['model_path']})")
    _step("tensor_train", str(tensor_dir))

    scfg = cfg["solver"]
    tensor_meta_construction = json.loads(
        (tensor_dir / "meta.json").read_text()).get("reference_construction", "shared_pool")
    A_policy = torch.from_numpy(np.load(tensor_dir / "tensor_policy.npz")["A"])
    A_ref = torch.from_numpy(np.load(tensor_dir / "tensor_ref.npz")["A"])
    mu = uniform_policy(A_policy.shape[1], A_policy.shape[3])
    beta = torch.tensor(scfg["opponent_betas"], dtype=torch.float64)
    lambda_init = None
    warm = scfg.get("warm_start_lambda")
    if warm:
        prev = json.loads(_resolve(base, warm).read_text())
        lambda_init = torch.tensor(prev["lambda_raw"], dtype=torch.float64)
    res = solve_nbpo_dual(
        A_policy, A_ref, mu, beta,
        eta=float(scfg["eta"]), gamma=scfg["gamma"], M=int(scfg["M"]), R=int(scfg["R"]),
        lambda_box=tuple(scfg.get("lambda_box", (1e-3, 1e3))),
        lambda_init=lambda_init, aggregation=scfg.get("aggregation", "nash"),
        reference_construction=tensor_meta_construction,
        damping=float(scfg.get("damping", 0.0)),
    )
    from scripts.nbpo.solve_nbpo_dual import write_solution_artifact
    tensor_meta = json.loads((tensor_dir / "meta.json").read_text())
    hashes = {name: sha256_file(tensor_dir / name)
              for name in ("tensor_policy.npz", "tensor_ref.npz", "meta.json")}
    solver_dir = workdir / "solver"
    solution = write_solution_artifact(solver_dir, res, tensor_meta, hashes, tensor_dir,
                                       stage, lambda_warm_started=lambda_init is not None)
    _step("solve_dual", f"lambda={solution['lambda_raw']} "
                        f"inv_surplus_res={solution['inverse_surplus_residual']} "
                        f"projected_kkt={solution['projected_kkt_residual']} R={scfg['R']}"
                        + (" (R=1: disclosed approximation)" if int(scfg["R"]) == 1 else ""))

    # 5. Pair targets from nu_update (hash-verified inside the builder).
    tcfg = cfg.get("targets", {})
    pairs_dir = workdir / "pairs"
    summary = build_pairs_from_artifacts(
        tensor_dir, solver_dir, policy_specs, pairs_dir,
        target_mode=tcfg.get("mode", "sampled"), seed=int(tcfg.get("seed", 42)), stage=stage,
        reproduction_mode=reproduction_mode,
        learner_manifest_sha256=(pool_bindings.get("learner_pool") or {}).get("manifest_sha256"),
        # Prompt-level hold-out: whole prompts move to the test split, so no
        # prompt's pairs straddle it.
        test_prompts=int(tcfg.get("test_prompts", 0) or 0),
        split_salt=str(tcfg.get("split_salt", "nbpo-v1")))
    _step("build_pairs", f"{summary['pairs']} target_mode={summary['target_mode']} "
                         f"opponent_scope={summary['opponent_sampling_scope']} "
                         f"pool={str(summary.get('tensor_response_pool_hash'))[:12]}")

    if reproduction_mode:
        check_reproduction_invariants(cfg.get("solver") or {},
                                      {**(cfg.get("trainer") or {}), **NBPO_TRAINER_INVARIANTS},
                                      target_summary=summary)
        _step("target_invariants", "eta applied once (target unscaled); "
                                   f"opponent_scope={summary['opponent_sampling_scope']}")

    # 6. Candidate fit (real) or marker (dry run).
    gate_base = workdir if dry_run else base
    promote_to = _resolve(gate_base, cfg["gate"]["promote_to"])
    parent_dir = _resolve(gate_base, cfg["gate"]["parent_model"])
    candidate_dir = promote_to.parent / (promote_to.name + ".candidate")
    precomputed = workdir / "precomputed"
    run_config = workdir / "run_config.yaml"
    mon = cfg["monitoring"]
    decode_cfg = dict(mon.get("decode") or {})
    # Monitoring decoding inherits the panel's generation kwargs unless it
    # overrides them, so a Qwen candidate cannot be monitored in thinking mode.
    decode_cfg.setdefault("chat_template_kwargs",
                          (cfg.get("generation") or {}).get("chat_template_kwargs") or {})
    seeds = [int(s) for s in decode_cfg.get("seeds", [42, 43, 44, 45])]
    decode_out = workdir / "candidate_monitoring"
    decode_manifest = workdir / "candidate_monitoring_manifest.json"
    binding = None
    validation_record = None
    if dry_run:
        candidate_dir.mkdir(parents=True, exist_ok=True)
        parent_dir.mkdir(parents=True, exist_ok=True)
        write_json(candidate_dir / "MARKER.json",
                   {"dry_run": True, "stage": stage, "solver_hash": summary["solver_hash"]})
        if not (parent_dir / "MARKER.json").exists():
            write_json(parent_dir / "MARKER.json", {"dry_run": True, "role": "parent pi_t"})
        parent_fp = checkpoint_fingerprint(str(parent_dir))
        _step("train", "skipped under --dry-run (marker candidate written; no LLM loaded)")
        dr = cfg.get("dry_run") or {}
        if "candidate_policy_files" not in dr:
            raise ValueError("--dry-run needs dry_run.candidate_policy_files (fixture responses)")
        candidate_specs = _specs(base, dr["candidate_policy_files"])
        _step("decode", "dry-run fixture responses from dry_run.candidate_policy_files "
                        "(manifest binding skipped and recorded as skipped)")
    else:
        cmds = cfg.get("commands") or {}
        for key in ("precompute", "train", "decode_candidate"):
            if key not in cmds:
                raise ValueError(f"real mode requires commands.{key} in the stage config")
        if not parent_dir.is_dir():
            raise FileNotFoundError(f"gate.parent_model is not a directory: {parent_dir}")
        parent_fp = checkpoint_fingerprint(str(parent_dir))
        # Prompt-level validation, when requested, comes from an explicit
        # prompt-disjoint file -- never from a pair-level slice of the training
        # rows, which would leak a prompt across the split.
        data_cfg = cfg.get("data") or {}
        if data_cfg.get("validation_prompt_file"):
            # This used to only ID-check a file that never defined the validation
            # pairs -- those still came from targets.test_prompts. The external
            # validation path is `validation:`; the old key is refused rather
            # than kept as a claim the code does not honour.
            raise ValueError(
                "data.validation_prompt_file is not implemented and never defined the "
                "validation pairs. Use the `validation:` section (prompts_file + "
                "learner_manifest + reference_manifest), which generates the test split "
                "from those prompts, or targets.test_prompts, which is a prompt-wise "
                "holdout WITHIN the pre-generated training response pool.")
        test_pairs = pairs_dir / "pairs_test.jsonl"
        if cfg.get("validation"):
            if int((cfg.get("targets") or {}).get("test_prompts", 0) or 0):
                raise ValueError(
                    "targets.test_prompts and validation: are mutually exclusive -- the "
                    "first holds prompts out of the training pool, the second builds the "
                    "test split from external validation prompts")
            validation_record = build_validation_pairs(
                cfg, base, workdir, objectives, objectives_config, judges["training"],
                solution, beta, stage, reproduction_mode, allow_partial,
                mu_fingerprint=checkpoint_fingerprint(str(mu_model_path)),
                parent_fingerprint=parent_fp, pairs_dir=pairs_dir)
            _step("validation", f"{validation_record['n_pairs']} pairs from "
                                f"{validation_record['n_prompts']} external validation prompts "
                                "(lambda from the training solve; nu recomputed there)")
        has_test = jsonl_has_records(test_pairs)
        if data_cfg.get("require_nonempty_validation") and not has_test:
            raise ValueError(
                f"data.require_nonempty_validation is set but {test_pairs} holds no rows "
                "(targets.test_prompts is 0, or the split salt selected nothing)")
        dataset_splits = ["train", "test"] if has_test else ["train"]
        fmt = dict(workdir=workdir, pairs_dir=pairs_dir, candidate=candidate_dir,
                   parent=parent_dir, precomputed=precomputed, solver_dir=solver_dir,
                   run_config=run_config, monitoring_prompts=mon_prompts_path,
                   decode_out=decode_out, decode_manifest=decode_manifest,
                   seeds=",".join(str(s) for s in seeds),
                   temperature=decode_cfg.get("temperature", 0.9),
                   top_p=decode_cfg.get("top_p", 0.95),
                   max_new_tokens=decode_cfg.get("max_new_tokens", 512),
                   chat_template_kwargs=json.dumps(decode_cfg.get("chat_template_kwargs") or {}),
                   # build_nbpo_pairs always writes pairs_test.jsonl; passing an
                   # EMPTY one to --test_dir used to reach torch.cat([]) inside
                   # precompute. The flag is emitted only when the file has rows.
                   test_dir_flag=(f"--test_dir {test_pairs}" if has_test else ""))
        _run(cmds["precompute"].format(**fmt), "precompute")
        meta = read_precompute_meta(str(precomputed))
        if meta is None:
            raise RuntimeError(f"precompute wrote no {precomputed}/precompute_meta.json")
        if str(meta.get("logp_reduction")).lower() != "sum":
            raise RuntimeError(f"precompute sidecar reduction={meta.get('logp_reduction')!r}; nbpo needs 'sum'")
        if meta.get("history_fingerprints") != [parent_fp]:
            raise RuntimeError(
                "precompute sidecar does not bind history0 to the parent checkpoint: "
                f"history_fingerprints={meta.get('history_fingerprints')} parent={parent_fp}")
        saved_splits = meta.get("dataset_splits")
        if saved_splits is not None and sorted(saved_splits) != sorted(dataset_splits):
            raise RuntimeError(
                f"precompute saved splits {sorted(saved_splits)} but the stage expected "
                f"{sorted(dataset_splits)}; the run config would name a split that does not exist")
        if meta.get("tokenization_config_sha256") is None:
            raise RuntimeError(
                "precompute artifact records no tokenization_config_sha256; pi_t's logps "
                "cannot be shown to come from the same token ids the trainer will score")
        _step("precompute", f"reduction=sum history0 bound to parent {parent_fp[:12]} "
                            f"splits={sorted(dataset_splits)} "
                            f"tokenization={meta['tokenization_config_sha256'][:12]}")
        expected_hashes = {
            "nbpo_expected_pair_artifact_sha256": sha256_file_hex(str(pairs_dir / "pairs_train.jsonl")),
            "nbpo_expected_solver_artifact_sha256": sha256_file_hex(str(solver_dir / "solution.json")),
            "nbpo_expected_parent_checkpoint_fingerprint": parent_fp,
            "nbpo_expected_precompute_manifest_sha256": meta.get("precompute_manifest_sha256"),
            # Eq. (22) is only meaningful if pi_t (offline) and pi (online) score
            # identical token ids; this pins the tokenization that produced them.
            "nbpo_expected_tokenization_config_sha256": meta.get("tokenization_config_sha256"),
        }
        materialize_run_config(cfg, parent_dir, precomputed, candidate_dir, run_config,
                               dataset_splits=dataset_splits, expected_hashes=expected_hashes)
        _step("run_config", f"{run_config} parsed by run_mnpo's argument dataclasses; "
                            f"expected pair/solver/parent/manifest hashes embedded")
        if candidate_dir.exists():
            raise FileExistsError(f"stale candidate directory exists: {candidate_dir}")
        _run(cmds["train"].format(**fmt), "train")
        if not candidate_dir.is_dir():
            raise RuntimeError(f"training produced no candidate directory at {candidate_dir}")
        _step("train", str(candidate_dir))
        _run(cmds["decode_candidate"].format(**fmt), "decode_candidate")
        binding = verify_decode_manifest(
            decode_manifest, candidate_dir, monitoring_ids, seeds,
            prompts_file=mon_prompts_path, prompt_text_sha256=monitoring_text_hashes,
            decode_params={"temperature": decode_cfg.get("temperature", 0.9),
                           "top_p": decode_cfg.get("top_p", 0.95),
                           "max_new_tokens": decode_cfg.get("max_new_tokens", 512)},
            chat_template_kwargs=decode_cfg.get("chat_template_kwargs") or {})
        candidate_specs = [f"{k}={v}" for k, v in sorted(binding["seed_files"].items())]
        _step("decode", f"manifest bound to candidate {binding['candidate_fingerprint'][:12]}; "
                        f"{len(candidate_specs)} seeds")

    # 7. Monitoring comparators bound to mu, and every seed file carrying exactly
    #    the monitoring prompt set.
    mon_ref_binding = None
    if mon.get("reference_manifest"):
        mu_fp = checkpoint_fingerprint(str(mu_model_path))
        mon_ref_binding = verify_manifest(
            _resolve(base, mon["reference_manifest"]),
            expected_role="monitoring_reference_comparator_pool",
            expected_fingerprint=mu_fp,
            expected_prompt_ids=monitoring_ids,
            expected_prompt_text_sha256=monitoring_text_hashes,
            expected_prompts_file=mon_prompts_path,
            required_seeds=[str(s) for s in sorted(mon["reference_manifest_seeds"])]
            if mon.get("reference_manifest_seeds") else None,
            expected_decode_params=mon.get("reference_decode_params"),
            expected_chat_template_kwargs=(cfg.get("generation") or {}).get(
                "chat_template_kwargs") or {},
            allow_partial_prompt_intersection=allow_partial)
        mon_ref_specs = [f"{s}={p}" for s, p in sorted(mon_ref_binding["seed_files"].items())]
        pool_bindings["monitoring_reference_comparator_pool"] = mon_ref_binding
        _step("monitoring_reference_manifest",
              f"comparators bound to mu {mu_fp[:12]}; {len(mon_ref_specs)} seeds")
    elif reproduction_mode:
        raise ValueError(
            "reproduction_mode requires monitoring.reference_manifest: bare "
            "monitoring.reference_files cannot bind the comparators to mu, so the "
            "Algorithm-1 gate could score the candidate against stale or "
            "differently-generated comparators")
    else:
        mon_ref_specs = _specs(base, mon["reference_files"])
    check_seed_files_cover_prompts(candidate_specs, monitoring_ids, "monitoring candidate")
    check_seed_files_cover_prompts(mon_ref_specs, monitoring_ids, "monitoring reference")
    _step("monitoring_prompt_sets", f"{len(candidate_specs)} candidate + {len(mon_ref_specs)} "
                                    f"reference seed files == monitoring set ({len(monitoring_ids)})")

    # 8. Monitoring judge + game-value evaluation.
    mon_tensor_dir, mon_cells = judge_and_build_tensors(
        candidate_specs, mon_ref_specs, objectives, objectives_config, judges["monitoring"],
        workdir, tag="monitoring", role="monitoring", reproduction_mode=reproduction_mode)
    mon_meta = json.loads((mon_tensor_dir / "meta.json").read_text())
    mon_eval = evaluate_game_value(
        torch.from_numpy(np.load(mon_tensor_dir / "tensor_policy.npz")["A"]),
        torch.from_numpy(np.load(mon_tensor_dir / "tensor_ref.npz")["A"]),
        beta)
    _step("eval_monitoring", f"min_surplus={mon_eval['min_surplus']:.6f} "
                             f"nash_defined={mon_eval['nash_welfare_defined']} "
                             f"judge={judges['monitoring']['model_path']}")

    # 9. Gate.
    cand_fp = checkpoint_fingerprint(str(candidate_dir))
    if binding is not None and binding["candidate_fingerprint"] != cand_fp:
        raise RuntimeError("candidate changed between decode and gate")
    gate = apply_gate(mon_eval["min_surplus"], parent_dir, candidate_dir, promote_to,
                      stage=stage, fingerprint=cand_fp)
    record = {
        "stage": stage,
        "accepted": gate["accepted"],
        "reason": gate["reason"],
        "candidate_fingerprint": cand_fp,
        "parent_fingerprint": parent_fp,
        "lineage": {"parent": str(parent_dir), "candidate": str(candidate_dir),
                    "promoted": gate.get("promoted_path"), "symlink": gate.get("symlink")},
        "decode_manifest_binding": (binding if binding is not None
                                    else "skipped (dry_run.candidate_policy_files fixture)"),
        "monitoring": mon_eval,
        "monitoring_prompts_file_sha256": sha256_file_hex(str(mon_prompts_path)),
        "monitoring_reference_manifest_sha256": (mon_ref_binding["manifest_sha256"]
                                                 if mon_ref_binding else None),
        "monitoring_reference_mu_fingerprint": (
            mon_ref_binding["manifest"]["checkpoint_fingerprint"] if mon_ref_binding else None),
        "mu_fingerprint": checkpoint_fingerprint(str(mu_model_path)),
        "comparator_pool_hash": sha256_file(mon_tensor_dir / "tensor_ref.npz"),
        "monitoring_verdicts_hash": mon_meta["verdicts_hash"],
        # One block per role, each from ITS OWN artifact. Copying the monitoring
        # metadata into all three (which is what the previous version did) made a
        # configured-but-never-run final evaluation look completed.
        "judges": build_judge_provenance(judges, {"training": tensor_meta,
                                                  "monitoring": mon_meta}, objectives_config,
                                         objectives),
        "training_verdicts_hash": tensor_meta["verdicts_hash"],
        "solver": {k: solution[k] for k in ("lambda_raw", "V", "d", "surplus",
                                            "inverse_surplus_residual", "projected_kkt_residual",
                                            "lambda_at_lower_bound", "lambda_at_upper_bound",
                                            "fixed_point_residual", "extra_map_residual")},
        "R": int(scfg["R"]), "R_is_approximation": int(scfg["R"]) == 1,
        "realization": "one-shot neural fit after the frozen-pool dual solve",
        **implementation_contract(dual_iterations=int(scfg["M"]), fixed_point_steps=int(scfg["R"])),
        "reproduction_mode": reproduction_mode,
        "reproduction_invariants": invariants,
        "prompt_set_disjointness": disjointness,
        "validation": validation_record if validation_record is not None else {
            "mode": ("prompt-wise holdout within the pre-generated training response pool"
                     if int((cfg.get("targets") or {}).get("test_prompts", 0) or 0)
                     else "none"),
            "test_prompts": int((cfg.get("targets") or {}).get("test_prompts", 0) or 0),
            "note": ("targets.test_prompts splits AFTER response generation; it is not a "
                     "held-out set generated independently. Use the validation: section "
                     "for that."),
        },
        "response_pool_manifests": {role: pool_manifest_summary(b)
                                    for role, b in pool_bindings.items()},
        "dry_run": dry_run,
        "monitoring_batch_disjoint_from_final_eval": True,
    }
    write_json(workdir / ("rejection_record.json" if not gate["accepted"]
                          else "stage_record.json"), record)
    _step("gate", ("ACCEPTED -> " + str(gate["promoted_path"])) if gate["accepted"]
          else "REJECTED (pi_t retained; rejection record written)")
    _step("complete", f"stage={stage} accepted={gate['accepted']}")
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, default=None,
                    help="artifact directory (default: <config dir>/work)")
    ap.add_argument("--dry-run", action="store_true",
                    help="mock judge, marker checkpoint, dry_run fixture responses; no LLM")
    args = ap.parse_args()
    workdir = args.workdir if args.workdir is not None else args.config.parent / "work"
    record = run_stage(args.config, workdir, dry_run=args.dry_run)
    print(json.dumps({"accepted": record["accepted"], "stage": record["stage"],
                      "min_surplus": record["monitoring"]["min_surplus"]}, indent=1))


if __name__ == "__main__":
    main()
