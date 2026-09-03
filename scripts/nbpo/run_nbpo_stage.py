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
from mnpo_scripts.nbpo_solver import solve_nbpo_dual
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
from scripts.nbpo.judge_pairwise_matrix import MockJudge, VllmJudge, build_task_grid, run_judging
from scripts.nbpo.nbpo_common import (
    load_objective_rubrics,
    load_response_files,
    read_jsonl,
    sha256_file,
    write_json,
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
def judge_config(cfg: dict, role: str) -> dict:
    """``judge.<role>`` (training / monitoring / final_eval); a flat legacy ``judge:`` block is rejected in real mode."""
    j = cfg.get("judge") or {}
    if role in j and isinstance(j[role], dict):
        return dict(j[role])
    if "backend" in j:  # flat legacy form
        return dict(j)
    raise ValueError(f"config needs judge.{role} (backend, model_path, ...)")


def make_backend(judge_cfg: dict, objectives_config: Path, objectives):
    if judge_cfg["backend"] == "mock":
        return MockJudge(invalid_rate=float(judge_cfg.get("mock_invalid_rate", 0.0)))
    loaded = load_objective_rubrics(objectives_config, objectives)
    return VllmJudge(judge_cfg["model_path"], loaded["rubrics"],
                     max_model_len=int(judge_cfg.get("max_model_len", 8192)),
                     tensor_parallel_size=int(judge_cfg.get("tensor_parallel_size", 1)),
                     batch_size=int(judge_cfg.get("batch_size", 512)))


def judge_and_build_tensors(policy_specs, reference_specs, objectives, objectives_config,
                            judge_cfg, out_dir: Path, tag: str):
    """Judge a matrix (mock or vllm) and build the tensor artifact in one go."""
    policy = load_response_files(policy_specs)
    reference = load_response_files(reference_specs)
    tasks = build_task_grid(policy, reference, objectives, include_reference_learner=True)
    verdicts = out_dir / f"verdicts_{tag}.jsonl"
    backend = make_backend(judge_cfg, objectives_config, objectives)
    rubric_version = yaml.safe_load(Path(objectives_config).read_text()).get("version")
    n = run_judging(tasks, backend, verdicts, judge_model=judge_cfg["model_path"],
                    rubric_version=rubric_version, max_retries=int(judge_cfg.get("max_retries", 2)))
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
        "objectives": objectives,
        "prompt_ids": prompt_ids,
        "policy_learner_ids": policy_ids,
        "comparator_ids": ref_ids,
        "generation_seeds": {"policy": {f"policy:{s}": s for s in sorted(policy)},
                             "reference": {f"ref:{s}": s for s in sorted(reference)}},
        "judge_role": tag,
        "judge_models": [judge_cfg["model_path"]],
        "judge_backend": judge_cfg["backend"],
        "judge_decoding": judge_cfg.get("decoding", {"temperature": 0.0, "max_tokens": 512,
                                                      "retry_temperature": 0.3}),
        "rubric_versions": [rubric_version],
        "rubric_config_hash": sha256_file(objectives_config),
        "verdicts_hash": sha256_file(verdicts),
        "both_orders_required": True,
        "self_pairs": "identity_zero",
        "reference_skew": skew_stats,
        "reference_construction": "shared_pool",
        "tensor_kind": "centered_preference",
        "shape_policy": list(A_policy.shape),
        "shape_ref": list(A_ref.shape),
    })
    return tensor_dir, n


# --------------------------------------------------------------------------- #
# run_mnpo config materialization
# --------------------------------------------------------------------------- #
def materialize_run_config(stage_cfg: dict, parent_checkpoint, precomputed_dataset_dir,
                           candidate_output_dir, output_yaml) -> dict:
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
        "dataset_splits": ["train", "test"],
    }
    run.update(trainer)
    run.update(NBPO_TRAINER_INVARIANTS)
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
                           required_seeds) -> dict:
    """Bind monitoring responses to the candidate checkpoint; returns seed -> path.

    Hard errors: fingerprint != current candidate directory; manifest prompt ids
    != monitoring set (missing and unexpected are both reported); any configured
    seed absent; any response file whose prompt set differs; any response file
    whose sha256 differs from the manifest (stale or edited file).
    """
    man = json.loads(Path(manifest_path).read_text())
    actual_fp = checkpoint_fingerprint(str(candidate_dir))
    if man.get("candidate_fingerprint") != actual_fp:
        raise ValueError(
            "decode manifest fingerprint does not match the candidate checkpoint: "
            f"manifest={str(man.get('candidate_fingerprint'))[:12]} candidate={actual_fp[:12]}"
        )
    check_prompt_set_equality(man.get("prompt_ids", []), monitoring_ids, "decode manifest")
    have = {str(s) for s in man.get("seeds", [])}
    need = {str(s) for s in required_seeds}
    if not need <= have:
        raise ValueError(f"decode manifest lacks configured seeds: missing {sorted(need - have)}")
    outputs = man.get("outputs") or {}
    paths = {}
    for s in sorted(need, key=int):
        key = f"s{s}"
        if key not in outputs:
            raise ValueError(f"decode manifest has no output entry for seed {s}")
        entry = outputs[key]
        p = Path(entry["path"])
        if not p.exists():
            raise ValueError(f"decode manifest output missing on disk: {p}")
        got = sha256_file_hex(str(p))
        if got != entry.get("sha256"):
            raise ValueError(
                f"response file {p} hash {got[:12]} != manifest {str(entry.get('sha256'))[:12]} "
                "(stale or edited after decoding)"
            )
        rows = json.loads(p.read_text())
        check_prompt_set_equality([r["prompt_id"] for r in rows], monitoring_ids,
                                  f"candidate seed {key}")
        paths[key] = str(p)
    return {"seed_files": paths, "candidate_fingerprint": actual_fp,
            "manifest_sha256": sha256_file_hex(str(manifest_path)),
            "prompts_file_sha256": man.get("prompts_file_sha256"),
            "decode_params": man.get("decode_params")}


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

    # 1. Rubrics
    load_objective_rubrics(objectives_config, objectives)
    _step("rubrics", f"{len(objectives)} objectives from {objectives_config.name}")

    # 2. Disjointness of train / monitoring / final-eval prompts.
    policy_specs = _specs(base, cfg["responses"]["policy_files"])
    reference_specs = _specs(base, cfg["responses"]["reference_files"])
    train_ids = set.intersection(*[set(m) for m in load_response_files(policy_specs).values()])
    mon_prompts_path = _resolve(base, cfg["monitoring"]["prompts_file"])
    monitoring_ids, _ = load_prompt_set(mon_prompts_path)
    final_ids, _ = load_prompt_set(_resolve(base, cfg["final_eval"]["prompts_file"]))
    check_disjoint(train_ids, monitoring_ids, final_ids)
    _step("disjointness", f"train={len(train_ids)} monitoring={len(monitoring_ids)} "
                          f"final={len(final_ids)}")

    # 3-4. Training judge -> tensors -> dual solve.
    judges = {}
    for role in JUDGE_ROLES:
        jc = judge_config(cfg, role)
        if dry_run and jc["backend"] != "mock":
            jc["backend"] = "mock"          # --dry-run never loads an LLM
        judges[role] = jc
    tensor_dir, n_cells = judge_and_build_tensors(
        policy_specs, reference_specs, objectives, objectives_config, judges["training"],
        workdir, tag="train")
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
        target_mode=tcfg.get("mode", "sampled"), seed=int(tcfg.get("seed", 42)), stage=stage)
    _step("build_pairs", f"{summary['pairs']} target_mode={summary['target_mode']} "
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
    seeds = [int(s) for s in decode_cfg.get("seeds", [42, 43, 44, 45])]
    decode_out = workdir / "candidate_monitoring"
    decode_manifest = workdir / "candidate_monitoring_manifest.json"
    binding = None
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
        fmt = dict(workdir=workdir, pairs_dir=pairs_dir, candidate=candidate_dir,
                   parent=parent_dir, precomputed=precomputed, solver_dir=solver_dir,
                   run_config=run_config, monitoring_prompts=mon_prompts_path,
                   decode_out=decode_out, decode_manifest=decode_manifest,
                   seeds=",".join(str(s) for s in seeds),
                   temperature=decode_cfg.get("temperature", 0.9),
                   top_p=decode_cfg.get("top_p", 0.95),
                   max_new_tokens=decode_cfg.get("max_new_tokens", 512))
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
        _step("precompute", f"reduction=sum history0 bound to parent {parent_fp[:12]}")
        materialize_run_config(cfg, parent_dir, precomputed, candidate_dir, run_config)
        _step("run_config", f"{run_config} parsed by run_mnpo's argument dataclasses")
        if candidate_dir.exists():
            raise FileExistsError(f"stale candidate directory exists: {candidate_dir}")
        _run(cmds["train"].format(**fmt), "train")
        if not candidate_dir.is_dir():
            raise RuntimeError(f"training produced no candidate directory at {candidate_dir}")
        _step("train", str(candidate_dir))
        _run(cmds["decode_candidate"].format(**fmt), "decode_candidate")
        binding = verify_decode_manifest(decode_manifest, candidate_dir, monitoring_ids, seeds)
        candidate_specs = [f"{k}={v}" for k, v in sorted(binding["seed_files"].items())]
        _step("decode", f"manifest bound to candidate {binding['candidate_fingerprint'][:12]}; "
                        f"{len(candidate_specs)} seeds")

    # 7. Every candidate AND reference seed file carries exactly the monitoring prompt set.
    mon_ref_specs = _specs(base, mon["reference_files"])
    check_seed_files_cover_prompts(candidate_specs, monitoring_ids, "monitoring candidate")
    check_seed_files_cover_prompts(mon_ref_specs, monitoring_ids, "monitoring reference")
    _step("monitoring_prompt_sets", f"{len(candidate_specs)} candidate + {len(mon_ref_specs)} "
                                    f"reference seed files == monitoring set ({len(monitoring_ids)})")

    # 8. Monitoring judge + game-value evaluation.
    mon_tensor_dir, mon_cells = judge_and_build_tensors(
        candidate_specs, mon_ref_specs, objectives, objectives_config, judges["monitoring"],
        workdir, tag="monitoring")
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
        "comparator_pool_hash": sha256_file(mon_tensor_dir / "tensor_ref.npz"),
        "monitoring_verdicts_hash": mon_meta["verdicts_hash"],
        "judges": {role: {"model": judges[role]["model_path"], "backend": judges[role]["backend"],
                          "rubric_versions": mon_meta["rubric_versions"],
                          "rubric_config_hash": mon_meta["rubric_config_hash"],
                          "decoding": mon_meta["judge_decoding"]} for role in JUDGE_ROLES},
        "training_verdicts_hash": tensor_meta["verdicts_hash"],
        "solver": {k: solution[k] for k in ("lambda_raw", "V", "d", "surplus",
                                            "inverse_surplus_residual", "projected_kkt_residual",
                                            "lambda_at_lower_bound", "lambda_at_upper_bound",
                                            "fixed_point_residual", "extra_map_residual")},
        "R": int(scfg["R"]), "R_is_approximation": int(scfg["R"]) == 1,
        "realization": "one-shot neural fit after the frozen-pool dual solve",
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
