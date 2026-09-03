"""Pair-target and judging-matrix tests (scripts/nbpo).

Covers spec tests 11 (shared opponent), 12 (all six pairs), 13 (orientation
flip), 14 (missing judgments are errors, never ties), plus target-mode
recording, the eta-is-not-applied-in-the-builder contract, and the retry loop.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mnpo_scripts.nbpo_core import uniform_policy
from mnpo_scripts.nbpo_solver import solve_nbpo_dual
from scripts.nbpo.build_nbpo_pairs import build_pairs_from_artifacts, flip_pair_row
from scripts.nbpo.build_preference_tensor import aggregate_cells, fill_tensor
from scripts.nbpo.judge_pairwise_matrix import MockJudge, build_task_grid, run_judging
from scripts.nbpo.nbpo_common import read_jsonl, sha256_file, write_json
from scripts.nbpo.solve_nbpo_dual import write_solution_artifact

PROMPTS = [f"p{i}" for i in range(3)]


def _write_responses(tmp: Path, tag: str, seeds):
    specs = []
    for s in seeds:
        rows = [{"prompt_id": p, "prompt": f"prompt {p}", "seed": s,
                 "generated_text": f"[{tag}:{s}] response for {p}"} for p in PROMPTS]
        path = tmp / f"{tag}_{s}.json"
        path.write_text(json.dumps(rows))
        specs.append(f"{s}={path}")
    return specs


def _build_artifacts(tmp: Path, target_mode="sampled", invalid_rate=0.0):
    policy_specs = _write_responses(tmp, "policy", ["s0", "s1", "s2", "s3"])
    ref_specs = _write_responses(tmp, "ref", ["r0", "r1", "r2", "r3"])
    from scripts.nbpo.nbpo_common import load_response_files

    policy = load_response_files(policy_specs)
    reference = load_response_files(ref_specs)
    tasks = build_task_grid(policy, reference, ["clarity", "brevity"], True)
    verdicts = tmp / "verdicts.jsonl"
    run_judging(tasks, MockJudge(invalid_rate=invalid_rate), verdicts,
                judge_model="mock", rubric_version=1, max_retries=3)
    rows = read_jsonl(verdicts)
    payoff = aggregate_cells(rows)
    objectives = ["clarity", "brevity"]
    policy_ids = [f"policy:{s}" for s in sorted(policy)]
    ref_ids = [f"ref:{s}" for s in sorted(reference)]
    A_policy = fill_tensor(payoff, objectives, PROMPTS, policy_ids, ref_ids, "policy")
    A_ref = fill_tensor(payoff, objectives, PROMPTS, ref_ids, ref_ids, "reference")
    tensor_dir = tmp / "tensor"
    tensor_dir.mkdir()
    np.savez_compressed(tensor_dir / "tensor_policy.npz", A=A_policy)
    np.savez_compressed(tensor_dir / "tensor_ref.npz", A=A_ref)
    write_json(tensor_dir / "meta.json", {
        "schema_version": 1, "objectives": objectives, "prompt_ids": PROMPTS,
        "policy_learner_ids": policy_ids, "comparator_ids": ref_ids,
        "judge_models": ["mock"], "rubric_versions": [1],
        "verdicts_hash": sha256_file(verdicts),
    })
    res = solve_nbpo_dual(torch.from_numpy(A_policy), torch.from_numpy(A_ref),
                          uniform_policy(len(PROMPTS), 4),
                          torch.tensor([0.25, 0.25], dtype=torch.float64),
                          eta=1.0, gamma=0.3, M=300, R=1, damping=0.3)
    solver_dir = tmp / "solver"
    hashes = {n: sha256_file(tensor_dir / n)
              for n in ("tensor_policy.npz", "tensor_ref.npz", "meta.json")}
    write_solution_artifact(solver_dir, res, json.loads((tensor_dir / "meta.json").read_text()),
                            hashes, tensor_dir, 0, lambda_warm_started=False)
    out_dir = tmp / "pairs"
    build_pairs_from_artifacts(tensor_dir, solver_dir, policy_specs, out_dir,
                               target_mode=target_mode, seed=7, stage=0)
    return read_jsonl(out_dir / "pairs_train.jsonl"), json.loads(
        (solver_dir / "solution.json").read_text())


def test_all_six_unordered_pairs_and_shared_opponent(tmp_path):
    rows, _ = _build_artifacts(tmp_path)
    by_prompt = {}
    for r in rows:
        by_prompt.setdefault(r["prompt_id"], []).append(r)
    for pid, prows in by_prompt.items():
        # 12: four learner responses -> exactly six unordered pairs
        assert len(prows) == 6
        pairs = {(r["chosen_response_id"], r["rejected_response_id"]) for r in prows}
        assert len(pairs) == 6
        assert all(a < b for a, b in pairs)  # canonical orientation
        # 11: within an objective, every pair of this prompt shares the same z_k;
        # objectives may select different opponents
        for obj in ("clarity", "brevity"):
            assert len({r["opponent_response_id"][obj] for r in prows}) == 1


def test_orientation_flip_negates_every_target(tmp_path):
    rows, _ = _build_artifacts(tmp_path)
    row = rows[0]
    flipped = flip_pair_row(row)
    assert flipped["chosen"] == row["rejected"] and flipped["rejected"] == row["chosen"]
    for obj, z in row["nbpo_z"].items():
        assert flipped["nbpo_z"][obj] == -z
    assert flipped["nbpo_weighted_z"] == -row["nbpo_weighted_z"]
    back = flip_pair_row(flipped)
    assert back["nbpo_weighted_z"] == row["nbpo_weighted_z"]
    assert back["chosen"] == row["chosen"]


def test_weighted_target_is_unscaled_by_eta_and_uses_raw_lambda(tmp_path):
    rows, solution = _build_artifacts(tmp_path)
    lam = dict(zip(solution["objectives"], solution["lambda_raw"]))
    for r in rows:
        expected = sum(lam[obj] * z for obj, z in r["nbpo_z"].items())
        assert abs(r["nbpo_weighted_z"] - expected) < 1e-9
        assert r["lambda_raw"] == {obj: pytest.approx(lam[obj]) for obj in lam}
        assert r["target_mode"] == "sampled"
        assert r["outer_stage"] == 0 and "solver_hash" in r and "dual_checkpoint" in r


def test_sampled_and_rao_blackwell_modes(tmp_path):
    (tmp_path / "s").mkdir()
    (tmp_path / "rb").mkdir()
    rows_s, _ = _build_artifacts(tmp_path / "s", target_mode="sampled")
    rows_rb, _ = _build_artifacts(tmp_path / "rb", target_mode="rao_blackwell")
    # sampled: Z_k = B_k - B'_k is integer-valued in {-1, 0, 1}
    for r in rows_s:
        for z in r["nbpo_z"].values():
            assert z in (-1.0, 0.0, 1.0)
    # rao_blackwell: Z_k = P(y>z_k) - P(y'>z_k), generally fractional; mode recorded
    assert all(r["target_mode"] == "rao_blackwell" for r in rows_rb)
    assert any(abs(z) not in (0.0, 1.0) for r in rows_rb for z in r["nbpo_z"].values())


def test_invalid_judgments_retry_then_fail_loudly(tmp_path):
    policy_specs = _write_responses(tmp_path, "policy", ["s0", "s1"])
    ref_specs = _write_responses(tmp_path, "ref", ["r0", "r1"])
    from scripts.nbpo.nbpo_common import load_response_files

    policy = load_response_files(policy_specs)
    reference = load_response_files(ref_specs)
    tasks = build_task_grid(policy, reference, ["clarity"], True)
    # moderate invalid rate: first pass leaves gaps, retries close them
    out = tmp_path / "v_ok.jsonl"
    run_judging(tasks, MockJudge(invalid_rate=0.3), out, "mock", 1, max_retries=6)
    rows = read_jsonl(out)
    assert any(r["attempt"] > 0 for r in rows), "retries should have been exercised"
    assert any(not r["valid"] for r in rows), "invalid rows are recorded, not hidden"
    valid_cells = {(r["prompt_id"], r["objective"], r["learner_pool"],
                    r["learner_response_id"], r["comparator_response_id"],
                    r["presentation_order"]) for r in rows if r["valid"]}
    assert len(valid_cells) == len(tasks)
    # invalid_rate=1.0: nothing ever validates -> loud failure, not 0.5 imputation
    with pytest.raises(RuntimeError, match="incomplete"):
        run_judging(tasks, MockJudge(invalid_rate=1.0), tmp_path / "v_bad.jsonl",
                    "mock", 1, max_retries=2)


def test_single_order_cells_are_errors_not_ties():
    rows = [
        {"prompt_id": "p0", "objective": "clarity", "learner_pool": "policy",
         "learner_response_id": "policy:s0", "comparator_response_id": "ref:r0",
         "presentation_order": "learner_first", "policy_win": 1.0, "valid": True},
        # comparator_first order missing for the same semantic comparison
    ]
    with pytest.raises(RuntimeError, match="one presentation order"):
        aggregate_cells(rows)
    # explicit labeled ablation is the only way through
    payoff = aggregate_cells(rows, allow_single_order=True)
    assert payoff[("p0", "clarity", "policy", "policy:s0", "ref:r0")] == 0.5


def test_incomplete_tensor_raises_and_never_imputes():
    payoff = {("p0", "clarity", "policy", "policy:s0", "ref:r0"): 0.1}
    with pytest.raises(RuntimeError, match="never imputed"):
        fill_tensor(payoff, ["clarity"], ["p0"], ["policy:s0", "policy:s1"],
                    ["ref:r0"], "policy")
