"""Pair-target and judging-matrix tests (scripts/nbpo).

Covers shared opponent inside one row/objective, per-pair/objective opponent
sampling with frequencies matching nu_update, all six pairs, orientation flip,
missing judgments as errors (never ties), judge backend cardinality, exact skew
symmetry / zero diagonal of the reference tensor, swapped-opponent-file
rejection, target-mode recording, the eta-is-not-applied-in-the-builder
contract, and the retry loop.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mnpo_scripts.nbpo_core import uniform_policy
from mnpo_scripts.nbpo_solver import solve_nbpo_dual
from scripts.nbpo.build_nbpo_pairs import build_pairs_from_artifacts, flip_pair_row
from scripts.nbpo.build_preference_tensor import (aggregate_cells, fill_reference_tensor,
                                                  fill_tensor)
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
    A_ref, _skew = fill_reference_tensor(payoff, objectives, PROMPTS, ref_ids)
    tensor_dir = tmp / "tensor"
    tensor_dir.mkdir()
    np.savez_compressed(tensor_dir / "tensor_policy.npz", A=A_policy)
    np.savez_compressed(tensor_dir / "tensor_ref.npz", A=A_ref)
    from scripts.nbpo.nbpo_common import response_pool_hash, sha256_text

    write_json(tensor_dir / "meta.json", {
        "schema_version": 1, "objectives": objectives, "prompt_ids": PROMPTS,
        "policy_learner_ids": policy_ids, "comparator_ids": ref_ids,
        "judge_models": ["mock"], "rubric_versions": [1],
        "verdicts_hash": sha256_file(verdicts),
        # Which response TEXTS this tensor describes: the pair builder re-hashes
        # the files it loads against these, so the fixture exercises that check.
        "learner_response_pool_hash": response_pool_hash(policy),
        "reference_response_pool_hash": response_pool_hash(reference),
        "response_text_sha256": {
            "policy": {f"policy:{s}": {pid: sha256_text(str(r["generated_text"]))
                                       for pid, r in sorted(rows.items())}
                       for s, rows in sorted(policy.items())},
            "reference": {f"ref:{s}": {pid: sha256_text(str(r["generated_text"]))
                                       for pid, r in sorted(rows.items())}
                          for s, rows in sorted(reference.items())},
        },
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


def test_all_six_unordered_pairs(tmp_path):
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


def test_shared_opponent_within_row(tmp_path):
    # 11: inside one row and objective, y and y' are compared against the SAME z_k.
    # rao_blackwell exposes it exactly: Z_k = P(y>z_k) - P(y'>z_k) for one shared j.
    rows, _ = _build_artifacts(tmp_path, target_mode="rao_blackwell")
    A = np.load(tmp_path / "tensor" / "tensor_policy.npz")["A"]
    meta = json.loads((tmp_path / "tensor" / "meta.json").read_text())
    pidx = {p: i for i, p in enumerate(meta["prompt_ids"])}
    cidx = {c: j for j, c in enumerate(meta["comparator_ids"])}
    lidx = {l: i for i, l in enumerate(meta["policy_learner_ids"])}
    for r in rows:
        assert r["opponent_sampling_scope"] == "pair_objective"
        x = pidx[r["prompt_id"]]
        i1, i2 = lidx[r["chosen_response_id"]], lidx[r["rejected_response_id"]]
        for k, obj in enumerate(meta["objectives"]):
            j = cidx[r["opponent_response_id"][obj]]          # one z_k per row/objective
            expected = (A[k, x, i1, j] + 0.5) - (A[k, x, i2, j] + 0.5)
            assert abs(r["nbpo_z"][obj] - expected) < 1e-12


def test_opponent_sampled_per_pair_objective(tmp_path):
    # Eq. (26) draws z_k AFTER (y, y'): different rows of a prompt may draw
    # different z_k, objectives may differ, and the empirical draw frequencies
    # over repeated builds match nu_update.
    rows, _ = _build_artifacts(tmp_path)
    by_prompt = {}
    for r in rows:
        by_prompt.setdefault(r["prompt_id"], []).append(r)
    varies = any(len({r["opponent_response_id"][obj] for r in prows}) > 1
                 for prows in by_prompt.values() for obj in ("clarity", "brevity"))
    assert varies, "per-pair sampling must let rows of one prompt draw different opponents"
    assert any(r["opponent_response_id"]["clarity"] != r["opponent_response_id"]["brevity"]
               for r in rows), "objectives may select different opponents"
    # frequency check: rebuild with many seeds and compare draw rates to nu_update
    nu = np.load(tmp_path / "solver" / "nu_update.npz")["nu"]
    meta = json.loads((tmp_path / "tensor" / "meta.json").read_text())
    counts = np.zeros_like(nu)
    n_builds = 40
    specs = [f"s{i}={tmp_path}/policy_s{i}.json" for i in range(4)]
    for seed in range(n_builds):
        build_pairs_from_artifacts(tmp_path / "tensor", tmp_path / "solver", specs,
                                   tmp_path / f"pairs_freq{seed}", target_mode="sampled",
                                   seed=1000 + seed, stage=0)
        for r in read_jsonl(tmp_path / f"pairs_freq{seed}" / "pairs_train.jsonl"):
            x = meta["prompt_ids"].index(r["prompt_id"])
            for k, obj in enumerate(meta["objectives"]):
                counts[k, x, meta["comparator_ids"].index(r["opponent_response_id"][obj])] += 1
    freq = counts / counts.sum(axis=-1, keepdims=True)   # 40 builds x 6 pairs = 240 draws per cell
    assert np.abs(freq - nu).max() < 0.09, np.abs(freq - nu).max()


def test_solver_artifact_rejects_swapped_nu_file(tmp_path):
    _build_artifacts(tmp_path)
    solver = tmp_path / "solver"
    a, b = solver / "nu_update.npz", solver / "nu_final_policy.npz"
    tmp = solver / "swap.tmp"
    a.rename(tmp); b.rename(a); tmp.rename(b)          # swap the two opponent files
    specs = [f"s{i}={tmp_path}/policy_s{i}.json" for i in range(4)]
    with pytest.raises(ValueError, match="swapped|hash"):
        build_pairs_from_artifacts(tmp_path / "tensor", solver, specs, tmp_path / "pairs_swapped",
                                   target_mode="sampled", seed=1, stage=0)


def test_reference_tensor_is_exactly_skew_symmetric(tmp_path):
    _build_artifacts(tmp_path)
    A_ref = np.load(tmp_path / "tensor" / "tensor_ref.npz")["A"]
    assert np.abs(A_ref + np.swapaxes(A_ref, -1, -2)).max() == 0.0    # exact, not approximate
    assert (A_ref != 0).any()                                            # nontrivial
    # legacy verdicts with both ordered directions judged inconsistently are
    # projected, the residual reported, and the result is still exactly skew
    payoff = {("p0", "clarity", "reference", "ref:r0", "ref:r1"): 0.20,
              ("p0", "clarity", "reference", "ref:r1", "ref:r0"): -0.10}
    A, stats = fill_reference_tensor(payoff, ["clarity"], ["p0"], ["ref:r0", "ref:r1"])
    assert stats["skew_projection_applied"] is True
    assert abs(stats["skew_residual_pre_projection_max"] - 0.10) < 1e-12
    assert abs(A[0, 0, 0, 1] - 0.15) < 1e-15 and A[0, 0, 1, 0] == -A[0, 0, 0, 1]
    assert np.abs(A + np.swapaxes(A, -1, -2)).max() == 0.0


def test_reference_tensor_diagonal_is_zero(tmp_path):
    _build_artifacts(tmp_path)
    A_ref = np.load(tmp_path / "tensor" / "tensor_ref.npz")["A"]
    assert (np.diagonal(A_ref, axis1=-2, axis2=-1) == 0.0).all()
    from mnpo_scripts.nbpo_core import validate_reference_tensor
    bad = A_ref.copy(); bad[0, 0, 1, 1] = 1e-9
    with pytest.raises(ValueError, match="diagonal"):
        validate_reference_tensor(torch.from_numpy(bad))
    bad2 = A_ref.copy(); bad2[0, 0, 0, 1] += 1e-9
    with pytest.raises(ValueError, match="skew"):
        validate_reference_tensor(torch.from_numpy(bad2))


def test_judge_short_output_fails_loudly(tmp_path):
    policy_specs = _write_responses(tmp_path, "policy", ["s0", "s1"])
    ref_specs = _write_responses(tmp_path, "ref", ["r0", "r1"])
    from scripts.nbpo.nbpo_common import load_response_files

    tasks = build_task_grid(load_response_files(policy_specs), load_response_files(ref_specs),
                            ["clarity"], True)

    class ShortJudge:
        def judge(self, pending, attempt):
            return [1.0] * (len(pending) - 1)              # one verdict short

    with pytest.raises(RuntimeError, match="returned"):
        run_judging(tasks, ShortJudge(), tmp_path / "short.jsonl", "mock", 1, max_retries=0)


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


def test_reference_construction_must_be_declared_and_is_enforced():
    # shared_pool: one response set on both sides -> exact skew symmetry is a
    # hard requirement. independent_samples: two independent mu draws -> the two
    # supports are different response sets, skew symmetry does not hold and must
    # not be imposed, but the caller has to SAY so; nothing is inferred.
    from mnpo_scripts.nbpo_core import (reference_skew_residual, validate_reference_tensor)

    g = torch.Generator().manual_seed(3)
    S = torch.rand(1, 2, 4, 4, generator=g, dtype=torch.float64) * 0.4 - 0.2
    shared = (S - S.transpose(-1, -2)) / 2                      # exactly skew, zero diagonal
    validate_reference_tensor(shared, construction="shared_pool")
    assert reference_skew_residual(shared) == 0.0
    indep = torch.rand(1, 2, 4, 4, generator=g, dtype=torch.float64) * 0.4 - 0.2
    assert reference_skew_residual(indep) > 1e-3
    with pytest.raises(ValueError, match="diagonal"):
        validate_reference_tensor(indep, construction="shared_pool")
    zero_diag = indep.clone()
    zero_diag.diagonal(dim1=-2, dim2=-1).zero_()          # isolate the skew check
    with pytest.raises(ValueError, match="skew"):
        validate_reference_tensor(zero_diag, construction="shared_pool")
    validate_reference_tensor(indep, construction="independent_samples")   # accepted, declared
    with pytest.raises(ValueError, match="construction must be one of"):
        validate_reference_tensor(indep, construction="whatever")
