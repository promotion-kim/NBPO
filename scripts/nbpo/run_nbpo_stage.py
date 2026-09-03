#!/usr/bin/env python3
"""One outer stage of NBPO Algorithm 1, end to end, with the checkpoint gate.

Pipeline (all in-process; the finite-pool math never loads an LLM):

1.  load the stage config and the versioned objective rubrics (missing or
    unavailable rubrics fail loudly -- nothing is invented);
2.  enforce that the training, held-out monitoring, and final-evaluation
    prompt sets are pairwise DISJOINT (hard error, not a convention);
3.  judge the training preference matrix (policy-vs-reference AND
    reference-vs-reference for the disagreement point d_k) in both
    presentation orders, with retries and a loud completeness failure;
4.  build the centered tensors, run the projected dual solve of Eq. (27) on
    the frozen pool, and write the solver artifact (raw lambda, V, d, s, KKT
    residual, fixed-point residual);
5.  build the pair targets (Eq. (24), sampled Bernoulli by default);
6.  train the neural candidate into a TEMPORARY candidate directory
    (real mode; skipped under --dry-run, which stands in a marker checkpoint);
7.  judge + evaluate the candidate on the held-out MONITORING batch with the
    game-value evaluator (never the fixed-reference one, never clamped);
8.  gate (Algorithm 1 lines 11-15): if ``min_k s_hat_k <= 0`` the candidate is
    REJECTED, ``pi_t`` is retained and a rejection record is written;
    otherwise the candidate is atomically promoted to ``pi_{t+1}``
    (``os.replace``; the temp dir lives under the same parent as the target).

The monitoring batch is never reused for final reporting: the config must name
a disjoint ``final_eval.prompts_file`` (the manuscript leaves the monitoring
batch size blank, so this config field is required and has no default).

``--dry-run`` runs steps 1-5, 7 and 8 with the deterministic mock judge and no
LLM anywhere, emitting one structured ``[nbpo-stage] step=... status=ok`` line
per step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from mnpo_scripts.nbpo_solver import solve_nbpo_dual
from scripts.nbpo.build_nbpo_pairs import build_pairs_from_artifacts
from scripts.nbpo.build_preference_tensor import aggregate_cells, fill_tensor
from scripts.nbpo.eval_game_value import evaluate_game_value
from scripts.nbpo.judge_pairwise_matrix import MockJudge, VllmJudge, build_task_grid, run_judging
from scripts.nbpo.nbpo_common import (
    load_objective_rubrics,
    load_response_files,
    read_jsonl,
    sha256_file,
    write_json,
)


def _step(name: str, detail: str = "") -> None:
    print(f"[nbpo-stage] step={name} status=ok{' detail=' + detail if detail else ''}", flush=True)


def _resolve(base: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (base / q)


def _specs(base: Path, mapping: dict) -> list:
    return [f"{seed}={_resolve(base, path)}" for seed, path in sorted(mapping.items())]


def hash_checkpoint_dir(path: Path) -> str:
    """Content hash of a checkpoint directory (sorted relpath + size + config bytes)."""
    parts = []
    for p in sorted(path.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(path))
            parts.append(f"{rel}:{p.stat().st_size}")
            if p.name in ("config.json", "MARKER.json"):
                parts.append(p.read_text())
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


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


def judge_and_build_tensors(policy_specs, reference_specs, objectives, objectives_config,
                            judge_cfg, out_dir: Path, tag: str):
    """Judge a matrix (mock or vllm) and build the tensor artifact in one go."""
    policy = load_response_files(policy_specs)
    reference = load_response_files(reference_specs)
    tasks = build_task_grid(policy, reference, objectives, include_reference_learner=True)
    verdicts = out_dir / f"verdicts_{tag}.jsonl"
    if judge_cfg["backend"] == "mock":
        backend = MockJudge(invalid_rate=float(judge_cfg.get("mock_invalid_rate", 0.0)))
    else:
        loaded = load_objective_rubrics(objectives_config, objectives)
        backend = VllmJudge(judge_cfg["model_path"], loaded["rubrics"],
                            max_model_len=int(judge_cfg.get("max_model_len", 8192)),
                            tensor_parallel_size=int(judge_cfg.get("tensor_parallel_size", 1)),
                            batch_size=int(judge_cfg.get("batch_size", 512)))
    rubric_version = yaml.safe_load(Path(objectives_config).read_text()).get("version")
    n = run_judging(tasks, backend, verdicts, judge_model=judge_cfg["model_path"],
                    rubric_version=rubric_version, max_retries=int(judge_cfg.get("max_retries", 2)))
    rows = read_jsonl(verdicts)
    payoff = aggregate_cells(rows)
    prompt_ids = sorted(set.intersection(*[set(m) for m in policy.values()],
                                         *[set(m) for m in reference.values()]))
    policy_ids = [f"policy:{s}" for s in sorted(policy)]
    ref_ids = [f"ref:{s}" for s in sorted(reference)]
    A_policy = fill_tensor(payoff, objectives, prompt_ids, policy_ids, ref_ids, "policy")
    A_ref = fill_tensor(payoff, objectives, prompt_ids, ref_ids, ref_ids, "reference")
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
        "judge_models": [judge_cfg["model_path"]],
        "rubric_versions": [rubric_version],
        "verdicts_hash": sha256_file(verdicts),
        "both_orders_required": True,
        "self_pairs": "identity_zero",
        "shape_policy": list(A_policy.shape),
        "shape_ref": list(A_ref.shape),
    })
    return tensor_dir, n


def apply_gate(min_surplus: float, parent_dir: Path, candidate_dir: Path,
               promote_to: Path) -> dict:
    """Algorithm 1 lines 11-15: reject on any nonpositive held-out surplus, else promote."""
    if min_surplus <= 0:
        return {"accepted": False, "promoted_path": str(parent_dir),
                "reason": f"min held-out surplus {min_surplus:.6f} <= 0; "
                          "stage flagged empirically infeasible, pi_t retained"}
    promote_to.parent.mkdir(parents=True, exist_ok=True)
    os.replace(candidate_dir, promote_to)  # atomic: same-parent temp dir
    return {"accepted": True, "promoted_path": str(promote_to),
            "reason": f"min held-out surplus {min_surplus:.6f} > 0"}


def run_stage(config_path: Path, workdir: Path, dry_run: bool) -> dict:
    base = config_path.parent
    cfg = yaml.safe_load(config_path.read_text())
    workdir.mkdir(parents=True, exist_ok=True)
    stage = int(cfg.get("stage", 0))
    objectives = list(cfg["objectives"]["names"])
    objectives_config = _resolve(base, cfg["objectives"]["config"])

    # 1. Rubrics: fail loudly before any compute if the exact prompts are unavailable.
    load_objective_rubrics(objectives_config, objectives)
    _step("rubrics", f"{len(objectives)} objectives from {objectives_config.name}")

    # 2. Disjointness of train / monitoring / final-eval prompts (enforced, not documented).
    policy_specs = _specs(base, cfg["responses"]["policy_files"])
    reference_specs = _specs(base, cfg["responses"]["reference_files"])
    train_ids = set.intersection(*[set(m) for m in load_response_files(policy_specs).values()])
    monitoring_ids = set(json.loads(_resolve(base, cfg["monitoring"]["prompts_file"]).read_text()))
    final_ids = set(json.loads(_resolve(base, cfg["final_eval"]["prompts_file"]).read_text()))
    check_disjoint(train_ids, monitoring_ids, final_ids)
    _step("disjointness", f"train={len(train_ids)} monitoring={len(monitoring_ids)} "
                          f"final={len(final_ids)}")

    # 3-4. Judge the training matrix and build tensors (policy-vs-ref + ref-vs-ref for d).
    judge_cfg = dict(cfg["judge"])
    if dry_run and judge_cfg["backend"] != "mock":
        judge_cfg["backend"] = "mock"  # --dry-run never loads an LLM
    tensor_dir, n_cells = judge_and_build_tensors(
        policy_specs, reference_specs, objectives, objectives_config, judge_cfg,
        workdir, tag="train")
    _step("judge_train", f"{n_cells} cells (both orders, retries enforced)")
    _step("tensor_train", str(tensor_dir))

    # Dual solve on the frozen pool (Eq. (27); R=1 is the disclosed approximation).
    scfg = cfg["solver"]
    A_policy = torch.from_numpy(np.load(tensor_dir / "tensor_policy.npz")["A"])
    A_ref = torch.from_numpy(np.load(tensor_dir / "tensor_ref.npz")["A"])
    from mnpo_scripts.nbpo_core import uniform_policy
    mu = uniform_policy(A_policy.shape[1], A_policy.shape[3])
    beta = torch.tensor(scfg["opponent_betas"], dtype=torch.float64)
    lambda_init = None
    warm = cfg["solver"].get("warm_start_lambda")
    if warm:
        prev = json.loads(_resolve(base, warm).read_text())
        lambda_init = torch.tensor(prev["lambda_raw"], dtype=torch.float64)
    res = solve_nbpo_dual(
        A_policy, A_ref, mu, beta,
        eta=float(scfg["eta"]), gamma=scfg["gamma"], M=int(scfg["M"]), R=int(scfg["R"]),
        lambda_box=tuple(scfg.get("lambda_box", (1e-3, 1e3))),
        lambda_init=lambda_init, aggregation=scfg.get("aggregation", "nash"),
        damping=float(scfg.get("damping", 0.0)),
    )
    from scripts.nbpo.solve_nbpo_dual import write_solution_artifact
    tensor_meta = json.loads((tensor_dir / "meta.json").read_text())
    hashes = {name: sha256_file(tensor_dir / name)
              for name in ("tensor_policy.npz", "tensor_ref.npz", "meta.json")}
    solver_dir = workdir / "solver"
    solution = write_solution_artifact(solver_dir, res, tensor_meta, hashes, tensor_dir,
                                       stage, lambda_warm_started=lambda_init is not None)
    _step("solve_dual", f"lambda={solution['lambda_raw']} kkt={solution['kkt_residual']}")

    # 5. Pair targets (Eq. (24)): unscaled sum_k lambda_k Z_k; eta applied by the trainer.
    tcfg = cfg.get("targets", {})
    pairs_dir = workdir / "pairs"
    summary = build_pairs_from_artifacts(
        tensor_dir, solver_dir, policy_specs, pairs_dir,
        target_mode=tcfg.get("mode", "sampled"), seed=int(tcfg.get("seed", 42)), stage=stage)
    _step("build_pairs", f"{summary['pairs']} target_mode={summary['target_mode']}")

    # 6. Neural training into a TEMPORARY candidate dir (same parent as the promote target).
    promote_to = _resolve(base if not dry_run else workdir, cfg["gate"]["promote_to"])
    parent_dir = _resolve(base if not dry_run else workdir, cfg["gate"]["parent_model"])
    candidate_dir = promote_to.parent / (promote_to.name + ".candidate")
    if dry_run:
        candidate_dir.mkdir(parents=True, exist_ok=True)
        parent_dir.mkdir(parents=True, exist_ok=True)
        write_json(candidate_dir / "MARKER.json",
                   {"dry_run": True, "stage": stage, "solver_hash": summary["solver_hash"]})
        _step("train", "skipped under --dry-run (marker candidate written; no LLM loaded)")
    else:
        train_cmds = cfg.get("commands", {})
        for key in ("precompute", "train", "decode_candidate"):
            if key not in train_cmds:
                raise ValueError(
                    f"real mode requires commands.{key} in the stage config "
                    "(see training_configs/nbpo/*.yaml for templates)"
                )
        for key in ("precompute", "train", "decode_candidate"):
            cmd = train_cmds[key].format(workdir=workdir, pairs_dir=pairs_dir,
                                         candidate=candidate_dir, parent=parent_dir)
            print(f"[nbpo-stage] running commands.{key}: {cmd}", flush=True)
            subprocess.run(cmd, shell=True, check=True)
        _step("train", str(candidate_dir))

    # 7. Held-out monitoring evaluation with the game-value evaluator.
    mon = cfg["monitoring"]
    mon_tensor_dir, mon_cells = judge_and_build_tensors(
        _specs(base, mon["candidate_policy_files"]), _specs(base, mon["reference_files"]),
        objectives, objectives_config, judge_cfg, workdir, tag="monitoring")
    mon_meta = json.loads((mon_tensor_dir / "meta.json").read_text())
    unexpected = set(mon_meta["prompt_ids"]) - monitoring_ids
    if unexpected:
        raise ValueError(f"monitoring responses cover prompts outside the monitoring set: "
                         f"{sorted(unexpected)[:5]}")
    mon_eval = evaluate_game_value(
        torch.from_numpy(np.load(mon_tensor_dir / "tensor_policy.npz")["A"]),
        torch.from_numpy(np.load(mon_tensor_dir / "tensor_ref.npz")["A"]),
        beta)
    _step("eval_monitoring", f"min_surplus={mon_eval['min_surplus']:.6f} "
                             f"nash_defined={mon_eval['nash_welfare_defined']}")

    # 8. Gate (Algorithm 1 lines 11-15).
    gate = apply_gate(mon_eval["min_surplus"], parent_dir, candidate_dir, promote_to)
    record = {
        "stage": stage,
        "accepted": gate["accepted"],
        "reason": gate["reason"],
        "candidate_hash": hash_checkpoint_dir(candidate_dir if not gate["accepted"]
                                              else promote_to),
        "parent_hash": hash_checkpoint_dir(parent_dir) if parent_dir.exists() else None,
        "monitoring": mon_eval,
        "comparator_pool_hash": sha256_file(mon_tensor_dir / "tensor_ref.npz"),
        "judge_model": judge_cfg["model_path"],
        "rubric_versions": mon_meta["rubric_versions"],
        "solver": {k: solution[k] for k in ("lambda_raw", "V", "d", "surplus",
                                            "kkt_residual", "fixed_point_residual")},
        "dry_run": dry_run,
        "monitoring_batch_disjoint_from_final_eval": True,
    }
    write_json(workdir / ("rejection_record.json" if not gate["accepted"]
                          else "stage_record.json"), record)
    _step("gate", ("ACCEPTED -> " + gate["promoted_path"]) if gate["accepted"]
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
                    help="run opponent construction, game values, dual updates, pair "
                         "targets, evaluation, and gate logic with the mock judge and "
                         "no LLM")
    args = ap.parse_args()
    workdir = args.workdir if args.workdir is not None else args.config.parent / "work"
    record = run_stage(args.config, workdir, dry_run=args.dry_run)
    print(json.dumps({"accepted": record["accepted"], "stage": record["stage"],
                      "min_surplus": record["monitoring"]["min_surplus"]}, indent=1))


if __name__ == "__main__":
    main()
