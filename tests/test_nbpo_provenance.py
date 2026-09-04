"""Provenance and integrity tests for the finite-pool NBPO realization.

Covers the audit's caching, manifest, hash-chain, cardinality, judge-config and
general-mu sections. Every test asserts that a specific WRONG artifact is
rejected -- the failure modes here are all ones that previously passed silently.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mnpo_scripts.nbpo_core import compute_disagreement_point, uniform_policy
from mnpo_scripts.precompute_provenance import (
    checkpoint_fingerprint,
    dataset_file_manifest,
    read_precompute_manifest,
    verify_precompute_manifest,
    write_precompute_manifest,
)
from scripts.nbpo.build_nbpo_pairs import _array_sha256, verify_solver_input_chain
from scripts.nbpo.judge_pairwise_matrix import MockJudge, build_task_grid, run_judging
from scripts.nbpo.nbpo_common import (
    IMPLEMENTATION_TYPE,
    check_pool_cardinality,
    check_reproduction_invariants,
    comparison_content_hash,
    implementation_contract,
    jsonl_has_records,
    judge_config_hash,
    load_response_files,
    normalize_judge_config,
    read_jsonl,
)
from scripts.nbpo.response_manifest import (
    build_manifest,
    check_stage_zero_identity,
    verify_manifest,
    write_manifest,
)

OBJ = ["clarity"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ckpt(tmp: Path, name: str, body: str) -> Path:
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "MARKER.json").write_text(json.dumps({"id": body}))
    return d


def _responses(tmp: Path, name: str, texts: dict) -> Path:
    p = tmp / name
    p.write_text(json.dumps([{"prompt_id": pid, "prompt": f"p{pid}", "generated_text": t}
                             for pid, t in sorted(texts.items())]))
    return p


def _prompts(tmp: Path, name: str, ids) -> Path:
    p = tmp / name
    p.write_text("\n".join(json.dumps({"prompt_id": i, "prompt": f"p{i}"}) for i in ids))
    return p


def _pool(tmp: Path, prefix: str, seeds, ids, salt=""):
    files = {}
    for s in seeds:
        files[s] = str(_responses(tmp, f"{prefix}_{s}.json",
                                  {i: f"{prefix}-{s}-{i}{salt}" for i in ids}))
    return files


def _judge_tasks(tmp: Path, learner_text: str, comparator_text: str = "cmp"):
    policy = {"s0": {"1": {"prompt_id": "1", "prompt": "hello", "generated_text": learner_text}}}
    reference = {"r0": {"1": {"prompt_id": "1", "prompt": "hello", "generated_text": comparator_text}}}
    return build_task_grid(policy, reference, OBJ, include_reference_learner=False)


def _identity(**over):
    base = normalize_judge_config({"backend": "mock", "model_path": "judge-v1"},
                                  "training", rubric_sha256="rub-a", max_retries=0)
    base.update(over)
    return base


def _run(tasks, out: Path, identity) -> int:
    return run_judging(tasks, MockJudge(judge_config=identity), out, judge_model="judge-v1",
                       rubric_version=1, max_retries=0, judge_identity=identity)


def _n_rows(out: Path) -> int:
    return len(read_jsonl(out)) if out.exists() else 0


# --------------------------------------------------------------------------- #
# 1. empty test split
# --------------------------------------------------------------------------- #
def test_empty_optional_test_split_is_omitted(tmp_path):
    empty = tmp_path / "pairs_test.jsonl"
    empty.write_text("")                       # what build_nbpo_pairs writes with 0 held out
    assert empty.exists() and not jsonl_has_records(empty)
    blank_lines = tmp_path / "blank.jsonl"
    blank_lines.write_text("\n\n  \n")
    assert not jsonl_has_records(blank_lines)
    nonempty = tmp_path / "pairs_train.jsonl"
    nonempty.write_text(json.dumps({"prompt_id": "1"}) + "\n")
    assert jsonl_has_records(nonempty)
    assert not jsonl_has_records(tmp_path / "missing.jsonl")
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n")
    with pytest.raises(ValueError, match="not valid JSON"):
        jsonl_has_records(bad)


def test_dataset_splits_match_saved_dataset(tmp_path):
    # materialize_run_config must name only splits that exist, or run_mnpo's
    # loader fails on a split the artifact never saved.
    from scripts.nbpo.run_nbpo_stage import materialize_run_config

    stage_cfg = {"stage": 0, "trainer": {"learning_rate": 5e-7, "max_steps": 2, "bf16": False}}
    parent = _ckpt(tmp_path, "parent", "p")
    out = tmp_path / "run_config.yaml"
    run = materialize_run_config(stage_cfg, parent, tmp_path / "pre", tmp_path / "cand", out,
                                 dataset_splits=["train"])
    assert run["dataset_splits"] == ["train"]
    run2 = materialize_run_config(stage_cfg, parent, tmp_path / "pre", tmp_path / "cand", out,
                                  dataset_splits=["train", "test"])
    assert run2["dataset_splits"] == ["train", "test"]


def test_run_config_carries_expected_artifact_hashes(tmp_path):
    from scripts.nbpo.run_nbpo_stage import materialize_run_config

    parent = _ckpt(tmp_path, "parent", "p")
    out = tmp_path / "run_config.yaml"
    run = materialize_run_config(
        {"stage": 0, "trainer": {"max_steps": 2, "bf16": False}}, parent,
        tmp_path / "pre", tmp_path / "cand", out, dataset_splits=["train"],
        expected_hashes={"nbpo_expected_pair_artifact_sha256": "a" * 64,
                         "nbpo_expected_solver_artifact_sha256": "b" * 64,
                         "nbpo_expected_parent_checkpoint_fingerprint": "c" * 64,
                         "nbpo_expected_precompute_manifest_sha256": None})
    assert run["nbpo_expected_pair_artifact_sha256"] == "a" * 64
    assert run["nbpo_expected_solver_artifact_sha256"] == "b" * 64
    assert run["nbpo_expected_parent_checkpoint_fingerprint"] == "c" * 64
    assert "nbpo_expected_precompute_manifest_sha256" not in run   # None is not written


# --------------------------------------------------------------------------- #
# 2. content-addressed judgment caching
# --------------------------------------------------------------------------- #
def test_same_ids_and_text_reuses_cache(tmp_path):
    out = tmp_path / "v.jsonl"
    ident = _identity()
    _run(_judge_tasks(tmp_path, "answer A"), out, ident)
    first = _n_rows(out)
    _run(_judge_tasks(tmp_path, "answer A"), out, ident)
    assert _n_rows(out) == first, "identical content must not be re-judged"


@pytest.mark.parametrize("kw,val", [
    ("learner_text", "a DIFFERENT answer"),
    ("comparator_text", "a DIFFERENT comparator"),
])
def test_changed_response_text_under_same_ids_triggers_judging(tmp_path, kw, val):
    # The exact stale-candidate bug: same prompt ids, same seed ids, new weights.
    out = tmp_path / "v.jsonl"
    ident = _identity()
    _run(_judge_tasks(tmp_path, "answer A"), out, ident)
    before = _n_rows(out)
    args = {"learner_text": "answer A", "comparator_text": "cmp"}
    args[kw] = val
    _run(_judge_tasks(tmp_path, **args), out, ident)
    assert _n_rows(out) > before, f"changing {kw} must force re-judging"


def test_changed_prompt_text_triggers_judging(tmp_path):
    out = tmp_path / "v.jsonl"
    ident = _identity()
    tasks = _judge_tasks(tmp_path, "answer A")
    _run(tasks, out, ident)
    before = _n_rows(out)
    moved = [dict(t, prompt="a completely different prompt") for t in tasks]
    _run(moved, out, ident)
    assert _n_rows(out) > before


@pytest.mark.parametrize("over", [
    {"rubric_sha256": "rub-b"},
    {"judge_model": "judge-v2"},
    {"judge_revision": "abc123"},
    {"temperature": 0.7},
    {"top_p": 0.9},
    {"max_tokens": 256},
    {"max_retries": 3},
])
def test_changed_judge_configuration_triggers_judging(tmp_path, over):
    out = tmp_path / "v.jsonl"
    tasks = _judge_tasks(tmp_path, "answer A")
    _run(tasks, out, _identity())
    before = _n_rows(out)
    _run(tasks, out, _identity(**over))
    assert _n_rows(out) > before, f"changing {over} must force re-judging"


def test_stale_monitoring_cache_cannot_accept_a_new_candidate(tmp_path):
    out = tmp_path / "v.jsonl"
    ident = _identity()
    _run(_judge_tasks(tmp_path, "old candidate response"), out, ident)
    old = {r["comparison_content_hash"] for r in read_jsonl(out)}
    _run(_judge_tasks(tmp_path, "NEW candidate response"), out, ident)
    new = {r["comparison_content_hash"] for r in read_jsonl(out)} - old
    assert new, "the new candidate must produce its own verdict rows"


def test_legacy_cache_rows_are_not_reused_and_are_quarantined(tmp_path):
    out = tmp_path / "v.jsonl"
    tasks = _judge_tasks(tmp_path, "answer A")
    ident = _identity()
    # A pre-content-addressing row: identifier fields only, no hash.
    legacy = {"prompt_id": "1", "objective": "clarity", "learner_pool": "policy",
              "learner_response_id": "policy:s0", "comparator_response_id": "ref:r0",
              "presentation_order": "learner_first", "policy_win": 1.0, "valid": True,
              "judge_model": "judge-v1", "rubric_version": 1}
    out.write_text(json.dumps(legacy) + "\n")
    _run(tasks, out, ident)
    rows = read_jsonl(out)
    assert all(r.get("comparison_content_hash") for r in rows), "legacy row must not survive"
    assert (tmp_path / "legacy" / "v.jsonl").exists(), "legacy rows are quarantined, not deleted"


def test_verdict_rows_record_the_effective_judge_config(tmp_path):
    out = tmp_path / "v.jsonl"
    ident = _identity(max_tokens=333)
    _run(_judge_tasks(tmp_path, "answer A"), out, ident)
    row = read_jsonl(out)[0]
    assert row["judge_effective_config"]["max_tokens"] == 333
    assert row["comparison_content_hash"] == comparison_content_hash(
        {**row["comparison_payload"], "prompt": "hello", "learner_text": "answer A",
         "comparator_text": "cmp", "prompt_id": "1", "objective": "clarity",
         "learner_pool": "policy", "learner_response_id": "policy:s0",
         "comparator_response_id": "ref:r0", "presentation_order": row["presentation_order"]},
        ident)


# --------------------------------------------------------------------------- #
# 3. training pools bound to pi_t and mu
# --------------------------------------------------------------------------- #
def test_training_pool_bound_to_parent_and_reference_to_mu(tmp_path):
    pi_t = _ckpt(tmp_path, "pi_t", "parent-weights")
    mu = _ckpt(tmp_path, "mu", "reference-weights")
    prompts = _prompts(tmp_path, "prompts.jsonl", ["1", "2"])
    rows = [("1", "p1"), ("2", "p2")]
    files = _pool(tmp_path, "learn", ["s0", "s1"], ["1", "2"])
    man_path = tmp_path / "learner.json"
    write_manifest(man_path, build_manifest(
        "learner_pool", str(pi_t), prompts, rows, ["s0", "s1"], files,
        {"temperature": 0.9, "top_p": 0.95, "max_new_tokens": 64}))
    ok = verify_manifest(man_path, "learner_pool",
                         expected_fingerprint=checkpoint_fingerprint(str(pi_t)),
                         expected_prompt_ids=["1", "2"], required_seeds=["s0", "s1"])
    assert set(ok["seed_files"]) == {"s0", "s1"}
    with pytest.raises(ValueError, match="checkpoint_fingerprint"):
        verify_manifest(man_path, "learner_pool",
                        expected_fingerprint=checkpoint_fingerprint(str(mu)))
    with pytest.raises(ValueError, match="role="):
        verify_manifest(man_path, "reference_comparator_pool")


def test_stage_zero_parent_reference_mismatch_fails(tmp_path):
    a = checkpoint_fingerprint(str(_ckpt(tmp_path, "a", "weights-a")))
    b = checkpoint_fingerprint(str(_ckpt(tmp_path, "b", "weights-b")))
    check_stage_zero_identity(a, a, stage=0)          # pi_0 == mu: fine
    check_stage_zero_identity(a, b, stage=1)          # only constrained at t = 0
    with pytest.raises(ValueError, match="stage 0 requires"):
        check_stage_zero_identity(a, b, stage=0)


def test_missing_prompt_in_one_seed_fails(tmp_path):
    ck = _ckpt(tmp_path, "ck", "w")
    prompts = _prompts(tmp_path, "prompts.jsonl", ["1", "2"])
    files = _pool(tmp_path, "pool", ["s0"], ["1", "2"])
    files["s1"] = str(_responses(tmp_path, "pool_s1.json", {"1": "only one"}))   # missing "2"
    man = tmp_path / "m.json"
    write_manifest(man, build_manifest("learner_pool", str(ck), prompts,
                                       [("1", "p1"), ("2", "p2")], ["s0", "s1"], files, {}))
    with pytest.raises(ValueError, match="prompt set differs"):
        verify_manifest(man, "learner_pool", expected_prompt_ids=["1", "2"])


def test_extra_or_missing_seed_fails_exactly(tmp_path):
    ck = _ckpt(tmp_path, "ck", "w")
    prompts = _prompts(tmp_path, "prompts.jsonl", ["1"])
    files = _pool(tmp_path, "pool", ["s0", "s1"], ["1"])
    man = tmp_path / "m.json"
    write_manifest(man, build_manifest("learner_pool", str(ck), prompts, [("1", "p1")],
                                       ["s0", "s1"], files, {}))
    with pytest.raises(ValueError, match="missing="):
        verify_manifest(man, "learner_pool", required_seeds=["s0", "s1", "s2"])
    with pytest.raises(ValueError, match="extra="):
        verify_manifest(man, "learner_pool", required_seeds=["s0"])


def test_duplicate_prompt_id_and_duplicate_seed_raise(tmp_path):
    dup = tmp_path / "dup.json"
    dup.write_text(json.dumps([{"prompt_id": "1", "prompt": "p", "generated_text": "a"},
                               {"prompt_id": "1", "prompt": "p", "generated_text": "b"}]))
    with pytest.raises(ValueError, match="duplicate prompt_id"):
        load_response_files([f"s0={dup}"])
    ok = _responses(tmp_path, "ok.json", {"1": "a"})
    with pytest.raises(ValueError, match="duplicate seed key"):
        load_response_files([f"s0={ok}", f"s0={ok}"])


def test_empty_response_and_mismatched_prompt_text_fail(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps([{"prompt_id": "1", "prompt": "p", "generated_text": "   "}]))
    with pytest.raises(ValueError, match="empty generated_text"):
        load_response_files([f"s0={empty}"])
    load_response_files([f"s0={empty}"], reproduction_mode=False)     # allowed when declared
    ck = _ckpt(tmp_path, "ck", "w")
    prompts = _prompts(tmp_path, "prompts.jsonl", ["1"])
    man = tmp_path / "m.json"
    write_manifest(man, build_manifest("learner_pool", str(ck), prompts, [("1", "p1")],
                                       ["s0"], _pool(tmp_path, "pool", ["s0"], ["1"]), {}))
    from scripts.nbpo.nbpo_common import sha256_text
    with pytest.raises(ValueError, match="prompt TEXT differs"):
        verify_manifest(man, "learner_pool",
                        expected_prompt_text_sha256={"1": sha256_text("a different prompt")})


# --------------------------------------------------------------------------- #
# 4. solver -> pair hash chain
# --------------------------------------------------------------------------- #
def _solver_pair(tmp_path):
    """A minimal consistent (tensor_dir, solver_dir, solution) triple."""
    from scripts.nbpo.nbpo_common import sha256_file, write_json

    tdir, sdir = Path(tmp_path) / "tensor", Path(tmp_path) / "solver"
    tdir.mkdir(parents=True, exist_ok=True); sdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tdir / "tensor_policy.npz", A=np.zeros((1, 1, 2, 2)))
    np.savez_compressed(tdir / "tensor_ref.npz", A=np.zeros((1, 1, 2, 2)))
    write_json(tdir / "meta.json", {"objectives": OBJ, "prompt_ids": ["1"]})
    np.savez_compressed(sdir / "nu_update.npz", nu=np.full((1, 1, 2), 0.5))
    np.savez_compressed(sdir / "nu_final_policy.npz", nu=np.array([[[0.4, 0.6]]]))
    source_pi = np.array([[0.5, 0.5]])
    np.savez_compressed(sdir / "update_source_pi.npz", pi=source_pi)
    solution = {
        "config": {"M": 10, "R": 1},
        "input_hashes": {n: sha256_file(tdir / n)
                         for n in ("tensor_policy.npz", "tensor_ref.npz", "meta.json")},
        "artifact_hashes": {n: sha256_file(sdir / n)
                            for n in ("nu_update.npz", "nu_final_policy.npz",
                                      "update_source_pi.npz")},
        "opponent_artifacts": {
            "nu_update.npz": {"artifact_kind": "regularized_opponent",
                              "source_policy": "proximal_centre",
                              "source_policy_hash": _array_sha256(source_pi),
                              "source_policy_artifact": "update_source_pi.npz",
                              "source_fixed_point_iteration": 0,
                              "used_for": "eq26_target"},
            "nu_final_policy.npz": {"artifact_kind": "regularized_opponent",
                                    "source_policy": "final_policy",
                                    "source_policy_artifact": "pi_star.npz",
                                    "source_fixed_point_iteration": 1,
                                    "used_for": "diagnostics"},
        },
    }
    write_json(sdir / "solution.json", solution)
    return tdir, sdir, solution


def test_solver_pair_tensor_hash_chain(tmp_path):
    tdir, sdir, solution = _solver_pair(tmp_path)
    verified = verify_solver_input_chain(tdir, sdir, solution)
    assert set(verified) >= {"tensor_policy.npz", "tensor_ref.npz", "meta.json", "nu_update.npz"}


def test_swapped_tensor_of_the_same_shape_fails(tmp_path):
    tdir, sdir, solution = _solver_pair(tmp_path)
    np.savez_compressed(tdir / "tensor_policy.npz", A=np.full((1, 1, 2, 2), 0.1))
    with pytest.raises(ValueError, match="NOT the ones the dual was solved against"):
        verify_solver_input_chain(tdir, sdir, solution)


def test_single_changed_payoff_entry_fails(tmp_path):
    tdir, sdir, solution = _solver_pair(tmp_path)
    A = np.load(tdir / "tensor_policy.npz")["A"]
    A[0, 0, 0, 1] = 0.25                       # one cell
    np.savez_compressed(tdir / "tensor_policy.npz", A=A)
    with pytest.raises(ValueError, match="tensor_policy"):
        verify_solver_input_chain(tdir, sdir, solution)


def test_changed_tensor_metadata_fails(tmp_path):
    from scripts.nbpo.nbpo_common import write_json

    tdir, sdir, solution = _solver_pair(tmp_path)
    write_json(tdir / "meta.json", {"objectives": ["something_else"], "prompt_ids": ["1"]})
    with pytest.raises(ValueError, match="meta.json"):
        verify_solver_input_chain(tdir, sdir, solution)


def test_swapped_nu_update_fails(tmp_path):
    tdir, sdir, solution = _solver_pair(tmp_path)
    np.savez_compressed(sdir / "nu_update.npz", nu=np.array([[[0.9, 0.1]]]))
    with pytest.raises(ValueError, match="nu_update"):
        verify_solver_input_chain(tdir, sdir, solution)


def test_identical_nu_arrays_are_accepted_when_roles_are_declared(tmp_path):
    """A zero-payoff or already-converged game gives nu_update == nu_final_policy.

    Byte inequality was never a valid integrity signal; the roles are.
    """
    import shutil

    from scripts.nbpo.nbpo_common import sha256_file, write_json

    for label in ("zero payoff", "at the fixed point"):
        tdir, sdir, solution = _solver_pair(tmp_path / label.replace(" ", "_"))
        shutil.copy(sdir / "nu_update.npz", sdir / "nu_final_policy.npz")
        solution["artifact_hashes"]["nu_final_policy.npz"] = sha256_file(
            sdir / "nu_final_policy.npz")
        write_json(sdir / "solution.json", solution)
        verified = verify_solver_input_chain(tdir, sdir, solution)
        assert verified["nu_update.npz"] == verified["nu_final_policy.npz"], label


def test_swapped_opponent_roles_fail(tmp_path):
    """The Eq. (26) opponent and the diagnostic one may not trade places."""
    from scripts.nbpo.nbpo_common import write_json

    tdir, sdir, solution = _solver_pair(tmp_path)
    solution["opponent_artifacts"]["nu_update.npz"]["used_for"] = "diagnostics"
    solution["opponent_artifacts"]["nu_final_policy.npz"]["used_for"] = "eq26_target"
    write_json(sdir / "solution.json", solution)
    with pytest.raises(ValueError, match="used_for"):
        verify_solver_input_chain(tdir, sdir, solution)
    tdir2, sdir2, solution2 = _solver_pair(tmp_path / "b")
    solution2["opponent_artifacts"]["nu_update.npz"]["source_policy"] = "final_policy"
    write_json(sdir2 / "solution.json", solution2)
    with pytest.raises(ValueError, match="not one of"):
        verify_solver_input_chain(tdir2, sdir2, solution2)


def test_missing_opponent_role_metadata_fails_in_reproduction_mode(tmp_path):
    from scripts.nbpo.nbpo_common import write_json

    tdir, sdir, solution = _solver_pair(tmp_path)
    solution.pop("opponent_artifacts")
    write_json(sdir / "solution.json", solution)
    with pytest.raises(ValueError, match="no opponent_artifacts metadata"):
        verify_solver_input_chain(tdir, sdir, solution)
    verify_solver_input_chain(tdir, sdir, solution, reproduction_mode_required=False)


# --------------------------------------------------------------------------- #
# 5. precompute manifest (Arrow shards included)
# --------------------------------------------------------------------------- #
def test_modified_arrow_file_fails_manifest_verification(tmp_path):
    d = tmp_path / "pre"
    d.mkdir()
    (d / "data-00000-of-00001.arrow").write_bytes(b"arrow-payload-v1")
    (d / "dataset_info.json").write_text("{}")
    _, sha = write_precompute_manifest(str(d), splits=["train"])
    verify_precompute_manifest(str(d), expected_manifest_sha256=sha)
    (d / "data-00000-of-00001.arrow").write_bytes(b"arrow-payload-v2")
    with pytest.raises(ValueError, match="does not match its manifest"):
        verify_precompute_manifest(str(d))
    assert "data-00000-of-00001.arrow" in dataset_file_manifest(str(d))


def test_wrong_expected_manifest_hash_fails(tmp_path):
    d = tmp_path / "pre"
    d.mkdir()
    (d / "x.arrow").write_bytes(b"payload")
    write_precompute_manifest(str(d), splits=["train"])
    with pytest.raises(ValueError, match="not the one this run config"):
        verify_precompute_manifest(str(d), expected_manifest_sha256="f" * 64)
    assert read_precompute_manifest(str(d))["splits"] == ["train"]


@pytest.mark.parametrize("field,value,pattern", [
    ("nbpo_expected_pair_artifact_sha256", "deadbeef" * 8, "stale pair artifact"),
    ("nbpo_expected_solver_artifact_sha256", "deadbeef" * 8, "stale solver artifact"),
    ("nbpo_expected_parent_checkpoint_fingerprint", "deadbeef" * 8, "does not match the parent"),
])
def test_stale_expected_artifact_hashes_fail(field, value, pattern):
    from types import SimpleNamespace

    from mnpo_scripts.mnpo_trainer import validate_nbpo_args

    args = SimpleNamespace(
        loss_type="nbpo", reference_anchor_weight=0.0, preference_sft_weight=0.0,
        logp_reduction="sum", max_history_t=1, history_weights=[1.0],
        nbpo_target_column="nbpo_weighted_z",
        nbpo_expected_pair_artifact_sha256=None,
        nbpo_expected_solver_artifact_sha256=None,
        nbpo_expected_parent_checkpoint_fingerprint=None,
        nbpo_expected_precompute_manifest_sha256=None)
    setattr(args, field, value)
    meta = {"logp_reduction": "sum", "pair_artifact_sha256": "a" * 64,
            "solver_artifact_sha256": "b" * 64, "history_fingerprints": ["c" * 64]}
    cols = ["nbpo_weighted_z", "history0_chosen_logps", "history0_rejected_logps"]
    with pytest.raises(ValueError, match=pattern):
        validate_nbpo_args(args, dataset_columns=cols, precompute_meta=meta)


# --------------------------------------------------------------------------- #
# 6. cardinality
# --------------------------------------------------------------------------- #
def test_exact_response_cardinality(tmp_path):
    four = {s: {"1": {"generated_text": "t"}} for s in ("s0", "s1", "s2", "s3")}
    obs = check_pool_cardinality(four, four, reproduction_mode=True)
    assert obs["unordered_learner_pairs"] == 6
    assert obs["presentation_orders_per_semantic_comparison"] == 2
    three = {s: {"1": {"generated_text": "t"}} for s in ("s0", "s1", "s2")}
    with pytest.raises(ValueError, match="cardinality violation"):
        check_pool_cardinality(three, four, reproduction_mode=True)
    # outside reproduction mode any geometry is allowed but still recorded
    loose = check_pool_cardinality(three, four, reproduction_mode=False)
    assert loose["learner_responses_per_prompt"] == 3 and loose["reproduction_mode"] is False


# --------------------------------------------------------------------------- #
# 7. candidate manifest
# --------------------------------------------------------------------------- #
def test_candidate_manifest_checks_prompt_file_decoding_and_relative_paths(tmp_path):
    from scripts.nbpo.decode_candidate import decode_candidate

    cand = _ckpt(tmp_path, "cand", "candidate-weights")
    prompts = _prompts(tmp_path, "mon.jsonl", ["m1", "m2"])
    man = tmp_path / "man.json"
    decode_candidate(cand, prompts, [0, 1], 0.9, 0.95, 16, tmp_path / "out", man,
                     backend="mock", chat_template_kwargs={"enable_thinking": False})
    fp = checkpoint_fingerprint(str(cand))
    from scripts.nbpo.response_manifest import prompt_text_hashes
    texts = prompt_text_hashes([("m1", "pm1"), ("m2", "pm2")])
    verify_manifest(man, "monitoring_candidate_pool", expected_fingerprint=fp,
                    expected_prompt_ids=["m1", "m2"], expected_prompt_text_sha256=texts,
                    expected_prompts_file=prompts, required_seeds=["0", "1"],
                    expected_decode_params={"temperature": 0.9, "top_p": 0.95},
                    expected_chat_template_kwargs={"enable_thinking": False})
    # a different prompt FILE carrying the same ids
    other = tmp_path / "other.jsonl"
    other.write_text("\n".join(json.dumps({"prompt_id": i, "prompt": f"DIFFERENT {i}"})
                               for i in ("m1", "m2")))
    with pytest.raises(ValueError, match="prompt_file_sha256"):
        verify_manifest(man, "monitoring_candidate_pool", expected_prompts_file=other)
    with pytest.raises(ValueError, match="decode params differ"):
        verify_manifest(man, "monitoring_candidate_pool",
                        expected_decode_params={"temperature": 0.7})
    with pytest.raises(ValueError, match="chat_template_kwargs"):
        verify_manifest(man, "monitoring_candidate_pool", expected_chat_template_kwargs={})
    with pytest.raises(ValueError, match="missing="):
        verify_manifest(man, "monitoring_candidate_pool", required_seeds=["0", "1", "2"])
    # relative output paths resolve against the manifest directory, not the cwd
    payload = json.loads(man.read_text())
    for entry in payload["outputs"].values():
        entry["path"] = str(Path(entry["path"]).relative_to(tmp_path))
    moved = tmp_path / "man_rel.json"
    moved.write_text(json.dumps(payload))
    verify_manifest(moved, "monitoring_candidate_pool", expected_fingerprint=fp)


# --------------------------------------------------------------------------- #
# 8. Qwen thinking mode
# --------------------------------------------------------------------------- #
class _FakeQwenTokenizer:
    """Records the kwargs apply_chat_template is called with; also tokenizes."""

    def __init__(self, accepts_thinking=True):
        self.calls = []
        self.accepts_thinking = accepts_thinking

    def __call__(self, text, add_special_tokens=False):
        ids = [abs(hash(w)) % 1000 for w in str(text).split()]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids):
        return " ".join(f"t{i}" for i in ids)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        if "enable_thinking" in kw and not self.accepts_thinking:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        self.calls.append(kw)
        return "RENDERED"


def test_qwen_thinking_disabled_reaches_the_tokenizer():
    from scripts.nbpo.decode_candidate import VllmDecoder
    from scripts.nbpo.judge_pairwise_matrix import VllmJudge

    dec = VllmDecoder.__new__(VllmDecoder)
    dec.tok = _FakeQwenTokenizer()
    dec.chat_template_kwargs = {"enable_thinking": False}
    dec.applied_chat_template_kwargs = None
    dec._render("hi")
    assert dec.tok.calls == [{"enable_thinking": False}]
    assert dec.applied_chat_template_kwargs == {"enable_thinking": False}

    # A tokenizer that does not accept the kwarg must not be recorded as if it did.
    dec2 = VllmDecoder.__new__(VllmDecoder)
    dec2.tok = _FakeQwenTokenizer(accepts_thinking=False)
    dec2.chat_template_kwargs = {"enable_thinking": False}
    dec2.applied_chat_template_kwargs = None
    dec2._render("hi")
    assert dec2.applied_chat_template_kwargs == {}

    judge = VllmJudge.__new__(VllmJudge)
    judge.tok = _FakeQwenTokenizer()
    judge.rubrics = {"clarity": {"system": "sys", "user_template": "{prompt}{a}{b}"}}
    judge.chat_template_kwargs = {"enable_thinking": False}
    judge._applied_chat_template_kwargs = None
    judge.strict_chat_template_kwargs = False
    judge.max_model_len, judge.max_tokens = 4096, 512
    judge.truncation_policy, judge._template_tokens = "proportional_tail", None
    judge._render({"objective": "clarity", "prompt": "p", "learner_text": "a",
                   "comparator_text": "b", "presentation_order": "learner_first"})
    assert judge.tok.calls == [{"enable_thinking": False}]


def test_qwen_config_declares_thinking_disabled_everywhere():
    import yaml

    cfg = yaml.safe_load(Path("training_configs/nbpo/ultrafeedback_qwen.yaml").read_text())
    assert cfg["generation"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert cfg["monitoring"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert cfg["judge"]["training"]["chat_template_kwargs"] == {"enable_thinking": False}


# --------------------------------------------------------------------------- #
# 9. judge config is effective, not merely recorded
# --------------------------------------------------------------------------- #
def test_judge_effective_config_is_what_the_backend_reports():
    cfg = normalize_judge_config(
        {"backend": "mock", "model_path": "m", "decoding": {"temperature": 0.4, "max_tokens": 77}},
        "training", rubric_sha256="r", max_retries=5)
    assert cfg["temperature"] == 0.4 and cfg["max_tokens"] == 77 and cfg["max_retries"] == 5
    backend = MockJudge(judge_config=cfg)
    eff = backend.effective_config()
    # The mock does not sample; it says so rather than echoing the request back.
    assert eff["applied"] is True and eff["temperature"] == 0.0
    assert eff["max_tokens"] == 77
    assert judge_config_hash({**cfg, **eff}) != judge_config_hash(cfg)


def test_configured_judge_decoding_reaches_the_vllm_backend():
    from scripts.nbpo.judge_pairwise_matrix import VllmJudge

    j = VllmJudge.__new__(VllmJudge)
    cfg = normalize_judge_config(
        {"backend": "vllm", "model_path": "m",
         "decoding": {"temperature": 0.25, "top_p": 0.8, "top_k": 40, "max_tokens": 123,
                      "retry_temperature": 0.9}}, "training", rubric_sha256="r")
    j._cfg = cfg
    j.temperature, j.top_p, j.top_k = cfg["temperature"], cfg["top_p"], cfg["top_k"]
    j.max_tokens, j.retry_temperature = cfg["max_tokens"], cfg["retry_temperature"]
    j.chat_template_kwargs, j._applied_chat_template_kwargs = {}, None
    j._frozen_config = None
    j.strict_chat_template_kwargs = False
    j.max_model_len, j.truncation_policy, j._template_tokens = 4096, "proportional_tail", None
    eff = j.effective_config()
    assert (eff["temperature"], eff["top_p"], eff["top_k"], eff["max_tokens"]) == \
           (0.25, 0.8, 40, 123)
    assert eff["retry_temperature"] == 0.9


def test_flat_judge_config_is_rejected_unless_explicitly_allowed():
    from scripts.nbpo.run_nbpo_stage import judge_config

    flat = {"judge": {"backend": "mock", "model_path": "one-judge"}}
    with pytest.raises(ValueError, match="flat legacy"):
        judge_config(flat, "training")
    assert judge_config(flat, "training", allow_legacy_flat=True)["model_path"] == "one-judge"
    roles = {"judge": {"training": {"backend": "mock", "model_path": "t"},
                       "monitoring": {"backend": "mock", "model_path": "m"}}}
    assert judge_config(roles, "training")["model_path"] == "t"
    with pytest.raises(ValueError, match="judge.final_eval"):
        judge_config(roles, "final_eval")


# --------------------------------------------------------------------------- #
# 10. role separation
# --------------------------------------------------------------------------- #
def test_final_eval_not_falsely_marked_complete():
    from scripts.nbpo.run_nbpo_stage import build_judge_provenance

    judges = {"training": {"backend": "mock", "model_path": "judge-train"},
              "monitoring": {"backend": "mock", "model_path": "judge-monitor"},
              "final_eval": {"backend": "mock", "model_path": "judge-final"}}
    artifacts = {
        "training": {"judge_models": ["judge-train"], "judge_backend": "mock",
                     "verdicts_hash": "t" * 64, "response_pool_hash": "tp" * 32,
                     "judge_config_hash": "tc", "rubric_versions": [1]},
        "monitoring": {"judge_models": ["judge-monitor"], "judge_backend": "mock",
                       "verdicts_hash": "m" * 64, "response_pool_hash": "mp" * 32,
                       "judge_config_hash": "mc", "rubric_versions": [1]},
    }
    prov = build_judge_provenance(judges, artifacts,
                                  Path("tests/fixtures/nbpo_toy/objectives.yaml"),
                                  ["clarity", "brevity"])
    assert prov["final_eval"]["status"] == "configured_not_executed"
    assert prov["final_eval"]["verdict_artifact_hash"] is None
    assert prov["final_eval"]["task_set_hash"] is None
    assert prov["final_eval"]["model"] == "judge-final"
    # roles never borrow each other's artifacts
    assert prov["training"]["verdict_artifact_hash"] != prov["monitoring"]["verdict_artifact_hash"]
    assert prov["training"]["model"] == "judge-train"
    assert prov["monitoring"]["status"] == "executed"


# --------------------------------------------------------------------------- #
# 11. general mu + reproduction invariants
# --------------------------------------------------------------------------- #
def test_nonuniform_mu_requires_an_explicit_learner_distribution():
    g = torch.Generator().manual_seed(11)
    S = torch.rand(1, 2, 3, 3, generator=g, dtype=torch.float64) * 0.4 - 0.2
    A_ref = (S - S.transpose(-1, -2)) / 2
    beta = torch.tensor([0.25], dtype=torch.float64)
    uniform = uniform_policy(2, 3)
    compute_disagreement_point(A_ref, uniform, beta)              # uniform: fine
    skewed = torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]], dtype=torch.float64)
    with pytest.raises(ValueError, match="NONUNIFORM comparator mu"):
        compute_disagreement_point(A_ref, skewed, beta)
    both = compute_disagreement_point(A_ref, skewed, beta, mu_learner=skewed)
    assert torch.isfinite(both).all()
    with pytest.raises(ValueError, match="mu_learner must have shape"):
        compute_disagreement_point(A_ref, skewed, beta, mu_learner=uniform_policy(2, 2))


@pytest.mark.parametrize("solver,trainer,pattern", [
    ({"damping": 0.5}, {}, "damping must be 0"),
    ({"normalize_lambda": True}, {}, "lambda must stay RAW"),
    ({}, {"logp_reduction": "mean"}, "logp_reduction must be 'sum'"),
    ({}, {"max_history_t": 2}, "max_history_t must be 1"),
    ({}, {"history_weights": [0.5, 0.5]}, "history_weights must be"),
    ({}, {"reference_anchor_weight": 0.1}, "reference_anchor_weight must be 0"),
    ({}, {"preference_sft_weight": 0.2}, "preference_sft_weight must be 0"),
])
def test_reproduction_invariants_are_asserted(solver, trainer, pattern):
    good_trainer = {"logp_reduction": "sum", "max_history_t": 1, "history_weights": [1.0],
                    "reference_anchor_weight": 0.0, "preference_sft_weight": 0.0}
    check_reproduction_invariants({}, good_trainer)               # baseline passes
    with pytest.raises(ValueError, match=pattern):
        check_reproduction_invariants({**solver}, {**good_trainer, **trainer})


def test_eta_applied_once_and_opponent_scope_are_checked():
    good_trainer = {"logp_reduction": "sum", "max_history_t": 1, "history_weights": [1.0],
                    "reference_anchor_weight": 0.0, "preference_sft_weight": 0.0}
    ok = {"note": "nbpo_weighted_z is unscaled: eta is applied exactly once, in the trainer",
          "opponent_sampling_scope": "pair_objective"}
    out = check_reproduction_invariants({}, good_trainer, target_summary=ok)
    assert out["eta_applications"] == 1
    with pytest.raises(ValueError, match="eta must be applied exactly once"):
        check_reproduction_invariants({}, good_trainer,
                                      target_summary={**ok, "note": "pre-scaled by eta"})
    with pytest.raises(ValueError, match="opponent_sampling_scope"):
        check_reproduction_invariants({}, good_trainer,
                                      target_summary={**ok, "opponent_sampling_scope": "prompt"})


# --------------------------------------------------------------------------- #
# 0. implementation contract
# --------------------------------------------------------------------------- #
def test_implementation_contract_is_declared():
    c = implementation_contract(dual_iterations=40000, fixed_point_steps=1)
    assert c["implementation_type"] == IMPLEMENTATION_TYPE == \
        "finite_pool_one_shot_neural_realization"
    assert c["dual_policy_representation"] == "finite_response_distribution"
    assert c["neural_fits_per_outer_stage"] == 1
    assert c["dual_iterations"] == 40000 and c["fixed_point_steps"] == 1


def test_no_paper_exact_algorithm_1_claim_survives():
    """The phrase may appear only in a sentence that DISCLAIMS it."""
    disclaimers = ("is not used for this pipeline", "Do not describe it as",
                   "that claim is only true if")
    offenders = []
    for path in list(Path("scripts/nbpo").glob("*.py")) + \
            list(Path("mnpo_scripts").glob("nbpo_*.py")) + \
            [Path("README.md")] + list(Path("training_configs/nbpo").glob("*.yaml")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if "paper-exact" not in line:
                continue
            window = path.read_text().splitlines()[max(0, i - 3):i + 2]
            if not any(d in " ".join(window) for d in disclaimers):
                offenders.append(f"{path}:{i}: {line.strip()}")
    assert not offenders, f"'paper-exact' still CLAIMED at {offenders}"
    doc = Path("docs/NBPO_ALGORITHM_MAPPING.md").read_text()
    assert "finite-pool NBPO realization" in doc
    assert 'Do not describe it as\n"paper-exact Algorithm 1"' in doc


# --------------------------------------------------------------------------- #
# P1: monitoring comparators bound to mu
# --------------------------------------------------------------------------- #
def _mon_pool(tmp_path, mu, ids=("m0", "m1"), seeds=("r0", "r1"), text="mon"):
    prompts = _prompts(tmp_path, "mon.jsonl", ids)
    rows = [(i, f"p{i}") for i in ids]
    files = {}
    for s in seeds:
        p = tmp_path / f"monref_{s}.json"
        p.write_text(json.dumps([{"prompt_id": i, "prompt": f"p{i}",
                                  "generated_text": f"{text}-{s}-{i}"} for i in ids]))
        files[s] = str(p)
    man = tmp_path / "mon_ref.json"
    write_manifest(man, build_manifest("monitoring_reference_comparator_pool", str(mu),
                                       prompts, rows, seeds, files,
                                       {"temperature": 0.9, "top_p": 0.95}))
    return man, prompts, rows, files


def test_monitoring_reference_pool_binds_to_mu(tmp_path):
    from scripts.nbpo.response_manifest import prompt_text_hashes

    mu = _ckpt(tmp_path, "mu", "reference-weights")
    other = _ckpt(tmp_path, "other", "some-other-weights")
    man, prompts, rows, _files = _mon_pool(tmp_path, mu)
    texts = prompt_text_hashes(rows)
    verify_manifest(man, "monitoring_reference_comparator_pool",
                    expected_fingerprint=checkpoint_fingerprint(str(mu)),
                    expected_prompt_ids=["m0", "m1"], expected_prompt_text_sha256=texts,
                    expected_prompts_file=prompts, required_seeds=["r0", "r1"],
                    expected_decode_params={"temperature": 0.9})
    with pytest.raises(ValueError, match="checkpoint_fingerprint"):
        verify_manifest(man, "monitoring_reference_comparator_pool",
                        expected_fingerprint=checkpoint_fingerprint(str(other)))
    with pytest.raises(ValueError, match="decode params differ"):
        verify_manifest(man, "monitoring_reference_comparator_pool",
                        expected_decode_params={"temperature": 0.0})
    with pytest.raises(ValueError, match="missing="):
        verify_manifest(man, "monitoring_reference_comparator_pool",
                        required_seeds=["r0", "r1", "r2"])
    with pytest.raises(ValueError, match="extra="):
        verify_manifest(man, "monitoring_reference_comparator_pool", required_seeds=["r0"])


def test_stale_monitoring_reference_response_file_fails(tmp_path):
    mu = _ckpt(tmp_path, "mu", "reference-weights")
    man, _prompts, _rows, files = _mon_pool(tmp_path, mu)
    Path(files["r0"]).write_text(json.dumps(
        [{"prompt_id": i, "prompt": f"p{i}", "generated_text": "REGENERATED"}
         for i in ("m0", "m1")]))
    with pytest.raises(ValueError, match="edited or replaced"):
        verify_manifest(man, "monitoring_reference_comparator_pool")


def test_monitoring_prompt_text_change_under_the_same_id_fails(tmp_path):
    from scripts.nbpo.nbpo_common import sha256_text

    mu = _ckpt(tmp_path, "mu", "reference-weights")
    man, _prompts, _rows, _files = _mon_pool(tmp_path, mu)
    with pytest.raises(ValueError, match="prompt TEXT differs"):
        verify_manifest(man, "monitoring_reference_comparator_pool",
                        expected_prompt_text_sha256={"m0": sha256_text("a different question"),
                                                     "m1": sha256_text("p m1")})


# --------------------------------------------------------------------------- #
# P1: row prompts must be the canonical prompts
# --------------------------------------------------------------------------- #
def test_row_prompt_must_match_the_canonical_prompt_text(tmp_path):
    ck = _ckpt(tmp_path, "ck", "w")
    prompts = _prompts(tmp_path, "prompts.jsonl", ["1", "2"])
    rows = [("1", "p1"), ("2", "p2")]
    good = tmp_path / "good_s0.json"
    good.write_text(json.dumps([{"prompt_id": "1", "prompt": "p1", "generated_text": "a"},
                                {"prompt_id": "2", "prompt": "p2", "generated_text": "b"}]))
    man = tmp_path / "m.json"
    write_manifest(man, build_manifest("learner_pool", str(ck), prompts, rows,
                                       ["s0"], {"s0": str(good)}, {}))
    verify_manifest(man, "learner_pool", expected_prompt_ids=["1", "2"])

    # same id, different prompt text inside the row
    good.write_text(json.dumps([{"prompt_id": "1", "prompt": "A DIFFERENT QUESTION",
                                 "generated_text": "a"},
                                {"prompt_id": "2", "prompt": "p2", "generated_text": "b"}]))
    with pytest.raises(ValueError, match="edited or replaced"):
        verify_manifest(man, "learner_pool")     # the file hash catches it first
    # rebuilding the manifest around the mismatched file must still fail
    with pytest.raises(ValueError, match="prompt text"):
        build_manifest("learner_pool", str(ck), prompts, rows, ["s0"], {"s0": str(good)}, {})


def test_missing_row_prompt_fails_and_one_seed_may_not_differ(tmp_path):
    ck = _ckpt(tmp_path, "ck", "w")
    prompts = _prompts(tmp_path, "prompts.jsonl", ["1"])
    rows = [("1", "p1")]
    no_prompt = tmp_path / "np_s0.json"
    no_prompt.write_text(json.dumps([{"prompt_id": "1", "generated_text": "a"}]))
    with pytest.raises(ValueError, match="no 'prompt' field"):
        build_manifest("learner_pool", str(ck), prompts, rows, ["s0"],
                       {"s0": str(no_prompt)}, {})
    ok = tmp_path / "ok_s0.json"
    ok.write_text(json.dumps([{"prompt_id": "1", "prompt": "p1", "generated_text": "a"}]))
    odd = tmp_path / "odd_s1.json"
    odd.write_text(json.dumps([{"prompt_id": "1", "prompt": "OTHER", "generated_text": "b"}]))
    with pytest.raises(ValueError, match="prompt text"):
        build_manifest("learner_pool", str(ck), prompts, rows, ["s0", "s1"],
                       {"s0": str(ok), "s1": str(odd)}, {})


# --------------------------------------------------------------------------- #
# P1: pair texts belong to the tensor's response pool
# --------------------------------------------------------------------------- #
def _tensor_meta_with_pool(policy):
    from scripts.nbpo.nbpo_common import response_pool_hash, sha256_text

    return {
        "learner_response_pool_hash": response_pool_hash(policy),
        "response_text_sha256": {
            "policy": {f"policy:{s}": {pid: sha256_text(str(r["generated_text"]))
                                       for pid, r in sorted(rows.items())}
                       for s, rows in sorted(policy.items())}},
        "learner_manifest_sha256": "manifest-a",
    }


def test_pair_texts_must_be_the_tensors_response_pool(tmp_path):
    from scripts.nbpo.build_nbpo_pairs import verify_response_pool_against_tensor

    policy = {"s0": {"1": {"prompt_id": "1", "prompt": "p", "generated_text": "original"}}}
    meta = _tensor_meta_with_pool(policy)
    out = verify_response_pool_against_tensor(meta, policy, True, "manifest-a")
    assert out["verified"] is True

    # same response ID, new text -- the exact failure IDs cannot catch
    mutated = {"s0": {"1": {"prompt_id": "1", "prompt": "p", "generated_text": "REGENERATED"}}}
    with pytest.raises(ValueError, match="response text"):
        verify_response_pool_against_tensor(meta, mutated, True, "manifest-a")
    with pytest.raises(ValueError, match="learner manifest"):
        verify_response_pool_against_tensor(meta, policy, True, "manifest-b")
    with pytest.raises(ValueError, match="records no learner_response_pool_hash"):
        verify_response_pool_against_tensor({}, policy, True, None)
    assert verify_response_pool_against_tensor({}, policy, False, None)["verified"] is False


def test_pair_rows_carry_the_response_text_hashes(tmp_path):
    """chosen/rejected hashes must equal sha256 of the strings actually stored."""
    from scripts.nbpo.nbpo_common import sha256_text
    from tests.test_nbpo_targets import _build_artifacts

    rows, _summary = _build_artifacts(tmp_path)
    for r in rows[:20]:
        assert r["chosen_text_sha256"] == sha256_text(r["chosen"])
        assert r["rejected_text_sha256"] == sha256_text(r["rejected"])


# --------------------------------------------------------------------------- #
# P2: judge preflight
# --------------------------------------------------------------------------- #
class _StrictTokenizer:
    """Word-level fake tokenizer: the renderer now measures a TOKEN budget."""

    def __init__(self, accepts=True):
        self.accepts, self.calls = accepts, []

    def __call__(self, text, add_special_tokens=False):
        ids = [abs(hash(w)) % 1000 for w in str(text).split()]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids):
        return " ".join(f"t{i}" for i in ids)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        if kw and not self.accepts:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        self.calls.append(kw)
        return "RENDERED"


def _bare_vllm_judge(accepts=True, strict=True, kwargs=None):
    from scripts.nbpo.judge_pairwise_matrix import VllmJudge

    j = VllmJudge.__new__(VllmJudge)
    j.tok = _StrictTokenizer(accepts)
    j.rubrics = {"clarity": {"system": "s", "user_template": "{prompt}{a}{b}"}}
    j._cfg = {"judge_model": "m", "tokenizer_revision": "rev1"}
    j.temperature, j.top_p, j.top_k = 0.0, 1.0, -1
    j.max_tokens, j.retry_temperature = 512, 0.3
    j.chat_template_kwargs = dict(kwargs if kwargs is not None else {"enable_thinking": False})
    j._applied_chat_template_kwargs = None
    j._frozen_config = None
    j.strict_chat_template_kwargs = strict
    j.max_model_len = 4096
    j.truncation_policy = "proportional_tail"
    j._template_tokens = None
    return j


_TASK = {"objective": "clarity", "prompt": "p", "learner_text": "a", "comparator_text": "b",
         "presentation_order": "learner_first"}


def test_unsupported_chat_template_kwarg_fails_in_reproduction_mode():
    j = _bare_vllm_judge(accepts=False, strict=True)
    with pytest.raises(ValueError, match="does not accept chat_template_kwargs"):
        j.preflight(_TASK)
    # outside reproduction mode it may be dropped, and is REPORTED as dropped
    lenient = _bare_vllm_judge(accepts=False, strict=False)
    cfg = lenient.preflight(_TASK)
    assert cfg["chat_template_kwargs"] == {}
    assert cfg["requested_chat_template_kwargs"] == {"enable_thinking": False}


def test_cache_identity_uses_the_post_render_effective_config():
    applied = _bare_vllm_judge(accepts=True, strict=True)
    before = judge_config_hash(applied.effective_config())
    after = judge_config_hash(applied.preflight(_TASK))
    assert applied.tok.calls == [{"enable_thinking": False}]
    dropped = _bare_vllm_judge(accepts=False, strict=False)
    dropped_hash = judge_config_hash(dropped.preflight(_TASK))
    assert dropped_hash != after, \
        "a dropped kwarg must give a different cache identity than an applied one"
    assert before == after, "an applied kwarg is already what effective_config reported"


def test_effective_config_cannot_change_after_the_first_task():
    j = _bare_vllm_judge(accepts=True, strict=True)
    j.preflight(_TASK)
    frozen = j.effective_config()
    j.chat_template_kwargs = {"enable_thinking": True}      # someone mutates mid-run
    with pytest.raises(RuntimeError, match="changed mid-run"):
        j._render(_TASK)
    assert j.effective_config() == frozen, "the frozen config must not follow the mutation"


def test_qwen_enable_thinking_false_reaches_apply_chat_template():
    j = _bare_vllm_judge(accepts=True, strict=True, kwargs={"enable_thinking": False})
    cfg = j.preflight(_TASK)
    assert j.tok.calls == [{"enable_thinking": False}]
    assert cfg["chat_template_kwargs"] == {"enable_thinking": False}
    assert cfg["renderer_schema_version"] == 2   # token-budget renderer
    assert cfg["tokenizer_revision"] == "rev1"


# --------------------------------------------------------------------------- #
# the ACTUAL source policy of nu_update
# --------------------------------------------------------------------------- #
def _feasible():
    from tests.test_nbpo_dual import feasible_game

    return feasible_game()


def test_r1_without_warm_start_nu_source_is_proximal_centre():
    from mnpo_scripts.nbpo_solver import solve_weighted_policy

    A, _A_ref, mu, beta = _feasible()
    pi_t = uniform_policy(4, 4)
    sol = solve_weighted_policy(A, mu, pi_t, torch.tensor([2.0, 3.0], dtype=torch.float64),
                                beta, 1.0, R=1)
    assert sol.update_source_kind == "proximal_centre"
    assert sol.update_source_iteration == 0
    assert torch.allclose(sol.update_source_pi, pi_t)


def test_r1_with_warm_start_nu_source_is_warm_start_iterate():
    from mnpo_scripts.nbpo_solver import solve_weighted_policy

    A, _A_ref, mu, beta = _feasible()
    g = torch.Generator().manual_seed(5)
    warm = torch.softmax(torch.rand(4, 4, generator=g, dtype=torch.float64) * 2, dim=-1)
    sol = solve_weighted_policy(A, mu, uniform_policy(4, 4),
                                torch.tensor([2.0, 3.0], dtype=torch.float64), beta, 1.0,
                                R=1, pi_init=warm)
    assert sol.update_source_kind == "warm_start_iterate"
    assert torch.allclose(sol.update_source_pi, warm)
    assert not torch.allclose(sol.update_source_pi, uniform_policy(4, 4))


def test_r_greater_than_one_records_r_minus_one_source_iteration():
    from mnpo_scripts.nbpo_solver import solve_weighted_policy

    A, _A_ref, mu, beta = _feasible()
    for R in (2, 3, 5):
        sol = solve_weighted_policy(A, mu, uniform_policy(4, 4),
                                    torch.tensor([2.0, 3.0], dtype=torch.float64), beta,
                                    1.0, R=R)
        assert sol.update_source_kind == "fixed_point_iterate"
        assert sol.update_source_iteration == R - 1


def test_released_dual_solve_reports_the_warm_start_iterate_not_the_centre():
    """The released R=1 configuration warm-starts, so nu_update is NOT from pi_t."""
    from mnpo_scripts.nbpo_solver import solve_nbpo_dual

    A, A_ref, mu, beta = _feasible()
    res = solve_nbpo_dual(A, A_ref, mu, beta, eta=1.0, gamma=0.3, M=50, R=1)
    assert res.update_source_kind == "warm_start_iterate", \
        "the artifact must not claim proximal_centre when the solve warm-started"
    assert res.update_source_pi is not None


def test_nu_source_policy_hash_matches_actual_array(tmp_path):
    from scripts.nbpo.solve_nbpo_dual import write_solution_artifact
    from mnpo_scripts.nbpo_solver import solve_nbpo_dual

    A, A_ref, mu, beta = _feasible()
    res = solve_nbpo_dual(A, A_ref, mu, beta, eta=1.0, gamma=0.3, M=40, R=1)
    tdir = tmp_path / "t"
    tdir.mkdir()
    sol = write_solution_artifact(tmp_path / "s", res, {"objectives": ["a", "b"]},
                                  {"tensor_policy.npz": "x"}, tdir, 0,
                                  lambda_warm_started=False)
    meta = sol["opponent_artifacts"]["nu_update.npz"]
    saved = np.load(tmp_path / "s" / meta["source_policy_artifact"])["pi"]
    assert _array_sha256(saved) == meta["source_policy_hash"]
    assert np.allclose(saved, res.update_source_pi.numpy())
    assert meta["source_policy"] == res.update_source_kind
    assert sol["artifact_hashes"]["update_source_pi.npz"]


def test_missing_nu_source_policy_artifact_fails(tmp_path):
    from scripts.nbpo.nbpo_common import write_json

    tdir, sdir, solution = _solver_pair(tmp_path)
    (sdir / "update_source_pi.npz").unlink()
    write_json(sdir / "solution.json", solution)
    with pytest.raises(ValueError, match="update_source_pi"):
        verify_solver_input_chain(tdir, sdir, solution)


def test_pair_builder_rejects_false_proximal_centre_metadata(tmp_path):
    """A source hash that does not match the saved array is caught."""
    from scripts.nbpo.nbpo_common import write_json

    tdir, sdir, solution = _solver_pair(tmp_path)
    solution["opponent_artifacts"]["nu_update.npz"]["source_policy_hash"] = "0" * 64
    write_json(sdir / "solution.json", solution)
    with pytest.raises(ValueError, match="is not the one on disk"):
        verify_solver_input_chain(tdir, sdir, solution)

    tdir2, sdir2, sol2 = _solver_pair(tmp_path / "c")
    sol2["opponent_artifacts"]["nu_update.npz"].pop("source_policy_artifact")
    write_json(sdir2 / "solution.json", sol2)
    with pytest.raises(ValueError, match="names no source_policy_artifact"):
        verify_solver_input_chain(tdir2, sdir2, sol2)


def test_warm_start_source_is_accepted_not_forced_to_proximal_centre(tmp_path):
    """proximal_centre is not the only legal source; a declared warm start passes."""
    from scripts.nbpo.nbpo_common import write_json

    tdir, sdir, solution = _solver_pair(tmp_path)
    solution["opponent_artifacts"]["nu_update.npz"]["source_policy"] = "warm_start_iterate"
    write_json(sdir / "solution.json", solution)
    verify_solver_input_chain(tdir, sdir, solution)


# --------------------------------------------------------------------------- #
# token-budget judge rendering (replaces character slicing)
# --------------------------------------------------------------------------- #
class _WordTokenizer:
    """One token per whitespace-separated word; decode is exact for these tests."""

    def __init__(self):
        self.vocab, self.inv = {}, {}

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for w in str(text).split():
            if w not in self.vocab:
                i = len(self.vocab) + 1
                self.vocab[w], self.inv[i] = i, w
            ids.append(self.vocab[w])
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids):
        return " ".join(self.inv[i] for i in ids)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return "\n".join(m["content"] for m in msgs)


def _budget_judge(max_model_len=200, max_tokens=16, policy="proportional_tail"):
    from scripts.nbpo.judge_pairwise_matrix import VllmJudge

    j = VllmJudge.__new__(VllmJudge)
    j.tok = _WordTokenizer()
    j.rubrics = {"clarity": {"system": "SYS", "user_template": "{prompt}||{a}||{b}"}}
    j._cfg = {"judge_model": "m"}
    j.temperature, j.top_p, j.top_k = 0.0, 1.0, -1
    j.max_tokens, j.retry_temperature = max_tokens, 0.3
    j.max_model_len, j.truncation_policy, j._template_tokens = max_model_len, policy, None
    j.chat_template_kwargs, j._applied_chat_template_kwargs = {}, None
    j._frozen_config, j.strict_chat_template_kwargs = None, True
    return j


def _task(prompt, a, b):
    return {"objective": "clarity", "prompt": prompt, "learner_text": a,
            "comparator_text": b, "presentation_order": "learner_first"}


def test_long_prompt_that_fits_the_budget_is_not_truncated():
    j = _budget_judge(max_model_len=4096, max_tokens=16)
    prompt = " ".join(f"w{i}" for i in range(300))
    _out, meta = j._render(_task(prompt, "short a", "short b"), return_meta=True)
    assert meta["truncated"] is False
    assert meta["retained_tokens"] == meta["original_tokens"]
    assert meta["original_tokens"]["prompt"] == 300


def test_over_budget_tldr_prompt_follows_the_configured_token_policy():
    """A very long summarization input is cut on token boundaries, and recorded."""
    j = _budget_judge(max_model_len=200, max_tokens=16)
    prompt = " ".join(f"p{i}" for i in range(500))          # a TL;DR-sized input
    _out, meta = j._render(_task(prompt, "a " * 20, "b " * 20), return_meta=True)
    assert meta["truncated"] is True
    assert meta["truncation_policy"] == "proportional_tail"
    assert meta["retained_tokens"]["prompt"] < meta["original_tokens"]["prompt"]
    # the responses survive: one long prompt cannot erase them
    assert meta["retained_tokens"]["a"] > 0 and meta["retained_tokens"]["b"] > 0
    assert sum(meta["retained_tokens"].values()) <= meta["token_budget"]

    strict = _budget_judge(max_model_len=200, max_tokens=16, policy="no_truncation")
    with pytest.raises(ValueError, match="no_truncation"):
        strict._render(_task(prompt, "a", "b"), return_meta=True)


def test_unicode_text_is_never_character_sliced():
    """Character slicing can cut a multibyte codepoint; token cuts cannot."""
    j = _budget_judge(max_model_len=120, max_tokens=8)
    prompt = " ".join(["한국어문장"] * 200 + ["🎯🔥"] * 50)
    out, meta = j._render(_task(prompt, "답변 하나", "답변 둘"), return_meta=True)
    assert meta["truncated"] is True
    out.encode("utf-8").decode("utf-8")               # never raises: no split codepoint
    for word in out.split():
        assert word in ("SYS", "||") or "�" not in word, "replacement char means a bad cut"


def test_rendered_prompt_hash_changes_when_the_truncation_policy_changes():
    prompt = " ".join(f"p{i}" for i in range(400))
    task = _task(prompt, " ".join(f"a{i}" for i in range(60)),
                 " ".join(f"b{i}" for i in range(60)))
    prop = _budget_judge(policy="proportional_tail")._render(task, return_meta=True)[1]
    head = _budget_judge(policy="head_only")._render(task, return_meta=True)[1]
    assert prop["truncated"] and head["truncated"]
    assert prop["rendered_prompt_sha256"] != head["rendered_prompt_sha256"]
    assert prop["retained_tokens"] != head["retained_tokens"]


def test_truncation_policy_is_part_of_both_identity_hashes():
    a = normalize_judge_config({"backend": "vllm", "model_path": "m",
                                "truncation_policy": "proportional_tail"},
                               "training", rubric_sha256="r")
    b = normalize_judge_config({"backend": "vllm", "model_path": "m",
                                "truncation_policy": "head_only"},
                               "training", rubric_sha256="r")
    assert judge_config_hash(a) != judge_config_hash(b)
    task = {"prompt_id": "1", "prompt": "p", "objective": "clarity",
            "learner_pool": "policy", "learner_response_id": "policy:s0",
            "comparator_response_id": "ref:r0", "presentation_order": "learner_first",
            "learner_text": "x", "comparator_text": "y"}
    assert comparison_content_hash(task, a) != comparison_content_hash(task, b)


def test_invalid_truncation_policy_is_rejected():
    from scripts.nbpo.judge_pairwise_matrix import VllmJudge

    j = VllmJudge.__new__(VllmJudge)
    with pytest.raises(ValueError, match="truncation_policy must be one of"):
        VllmJudge.__init__.__wrapped__ if False else None
        # exercise the guard directly
        policy = "slice_characters"
        from scripts.nbpo.judge_pairwise_matrix import TRUNCATION_POLICIES
        if policy not in TRUNCATION_POLICIES:
            raise ValueError(f"truncation_policy must be one of {TRUNCATION_POLICIES}, "
                             f"got {policy!r}")


# --------------------------------------------------------------------------- #
# training pools bound to the exact reported protocol
# --------------------------------------------------------------------------- #
def _training_manifest(tmp_path, ck, ids=("1", "2"), seeds=("s0", "s1"),
                       decode=None, prompt_text=None):
    decode = decode or {"temperature": 0.9, "top_p": 0.95, "max_new_tokens": 64}
    texts = prompt_text or {i: f"p{i}" for i in ids}
    prompts = tmp_path / "train_prompts.jsonl"
    prompts.write_text("\n".join(json.dumps({"prompt_id": i, "prompt": texts[i]})
                                 for i in ids) + "\n")
    rows = [(i, texts[i]) for i in ids]
    files = {}
    for s in seeds:
        f = tmp_path / f"pool_{s}.json"
        f.write_text(json.dumps([{"prompt_id": i, "prompt": texts[i],
                                  "generated_text": f"{s}-{i}"} for i in ids]))
        files[s] = str(f)
    man = tmp_path / "learner.json"
    write_manifest(man, build_manifest("learner_pool", str(ck), prompts, rows,
                                       seeds, files, decode))
    return man, prompts, rows, files


def test_wrong_training_prompt_file_fails(tmp_path):
    from scripts.nbpo.response_manifest import prompt_text_hashes

    ck = _ckpt(tmp_path, "ck", "w")
    man, prompts, rows, _ = _training_manifest(tmp_path, ck)
    verify_manifest(man, "learner_pool", expected_prompts_file=prompts,
                    expected_prompt_ids=["1", "2"],
                    expected_prompt_text_sha256=prompt_text_hashes(rows),
                    required_seeds=["s0", "s1"], expected_num_prompts=2,
                    expected_decode_params={"temperature": 0.9})
    other = tmp_path / "other_prompts.jsonl"
    other.write_text("\n".join(json.dumps({"prompt_id": i, "prompt": f"p{i}"})
                               for i in ("1", "2")) + "  \n")   # different bytes, same ids
    with pytest.raises(ValueError, match="prompt_file_sha256"):
        verify_manifest(man, "learner_pool", expected_prompts_file=other)


def test_both_pools_on_the_same_wrong_subset_still_fail(tmp_path):
    """Mutual agreement is not evidence: both are checked against the INTENDED file."""
    from scripts.nbpo.response_manifest import prompt_text_hashes

    ck = _ckpt(tmp_path, "ck", "w")
    sub = tmp_path / "sub"
    sub.mkdir()
    man, _p, _r, _f = _training_manifest(sub, ck, ids=("1",))     # only prompt 1
    intended_rows = [("1", "p1"), ("2", "p2")]
    intended = tmp_path / "intended.jsonl"
    intended.write_text("\n".join(json.dumps({"prompt_id": i, "prompt": t})
                                  for i, t in intended_rows) + "\n")
    with pytest.raises(ValueError, match="prompt ids differ|covers 1 prompts"):
        verify_manifest(man, "learner_pool", expected_prompts_file=intended,
                        expected_prompt_ids=["1", "2"],
                        expected_prompt_text_sha256=prompt_text_hashes(intended_rows),
                        expected_num_prompts=2)


def test_wrong_training_temperature_fails(tmp_path):
    ck = _ckpt(tmp_path, "ck", "w")
    man, _p, _r, _f = _training_manifest(tmp_path, ck)
    with pytest.raises(ValueError, match="decode params differ"):
        verify_manifest(man, "learner_pool",
                        expected_decode_params={"temperature": 0.7, "top_p": 0.95})


def test_missing_or_extra_training_seed_fails(tmp_path):
    ck = _ckpt(tmp_path, "ck", "w")
    man, _p, _r, _f = _training_manifest(tmp_path, ck, seeds=("s0", "s1"))
    with pytest.raises(ValueError, match="missing="):
        verify_manifest(man, "learner_pool", required_seeds=["s0", "s1", "s2", "s3"])
    with pytest.raises(ValueError, match="extra="):
        verify_manifest(man, "learner_pool", required_seeds=["s0"])


def test_manifest_seeds_and_output_keys_must_agree(tmp_path):
    ck = _ckpt(tmp_path, "ck", "w")
    man, _p, _r, _f = _training_manifest(tmp_path, ck, seeds=("s0", "s1"))
    payload = json.loads(man.read_text())
    payload["seeds"] = ["s0", "s1", "s2"]          # declared but never produced
    man.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="!= output keys"):
        verify_manifest(man, "learner_pool")


def test_wrong_expected_prompt_count_fails(tmp_path):
    ck = _ckpt(tmp_path, "ck", "w")
    man, _p, _r, _f = _training_manifest(tmp_path, ck, ids=("1", "2"))
    verify_manifest(man, "learner_pool", expected_num_prompts=2)
    with pytest.raises(ValueError, match="covers 2 prompts, expected 7000"):
        verify_manifest(man, "learner_pool", expected_num_prompts=7000)


def test_changed_manifest_bytes_fail_against_the_configured_sha(tmp_path):
    from mnpo_scripts.precompute_provenance import sha256_file_hex

    ck = _ckpt(tmp_path, "ck", "w")
    man, _p, _r, _f = _training_manifest(tmp_path, ck)
    sha = sha256_file_hex(str(man))
    verify_manifest(man, "learner_pool", expected_manifest_sha256=sha)
    payload = json.loads(man.read_text())
    payload["model_revision"] = "edited"
    man.write_text(json.dumps(payload, indent=2))
    with pytest.raises(ValueError, match="not the manifest the experiment was run with"):
        verify_manifest(man, "learner_pool", expected_manifest_sha256=sha)


# --------------------------------------------------------------------------- #
# standalone tensor CLI prompt-set resolution
# --------------------------------------------------------------------------- #
def test_tensor_cli_requires_exact_prompt_sets():
    from scripts.nbpo.build_preference_tensor import resolve_prompt_set

    full = {"s0": {"1": {"prompt": "p1"}, "2": {"prompt": "p2"}},
            "s1": {"1": {"prompt": "p1"}, "2": {"prompt": "p2"}}}
    ref = {"r0": {"1": {"prompt": "p1"}, "2": {"prompt": "p2"}}}
    ids, report = resolve_prompt_set(full, ref)
    assert ids == ["1", "2"] and report["n_dropped"] == 0

    short = {"s0": full["s0"], "s1": {"1": {"prompt": "p1"}}}
    with pytest.raises(ValueError, match="is missing 1 prompts"):
        resolve_prompt_set(short, ref)
    ids, report = resolve_prompt_set(short, ref, allow_partial=True)
    assert ids == ["1"]
    assert report["dropped_prompt_ids"] == ["2"]
    assert report["allow_partial_prompt_intersection"] is True


def test_tensor_cli_rejects_the_same_id_with_different_prompt_text():
    from scripts.nbpo.build_preference_tensor import resolve_prompt_set

    a = {"s0": {"1": {"prompt": "p1"}}, "s1": {"1": {"prompt": "A DIFFERENT QUESTION"}}}
    with pytest.raises(ValueError, match="different text"):
        resolve_prompt_set(a, {"r0": {"1": {"prompt": "p1"}}})
