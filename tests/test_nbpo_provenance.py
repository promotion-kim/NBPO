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
from scripts.nbpo.build_nbpo_pairs import verify_solver_input_chain
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

    tdir, sdir = tmp_path / "tensor", tmp_path / "solver"
    tdir.mkdir(); sdir.mkdir()
    np.savez_compressed(tdir / "tensor_policy.npz", A=np.zeros((1, 1, 2, 2)))
    np.savez_compressed(tdir / "tensor_ref.npz", A=np.zeros((1, 1, 2, 2)))
    write_json(tdir / "meta.json", {"objectives": OBJ, "prompt_ids": ["1"]})
    np.savez_compressed(sdir / "nu_update.npz", nu=np.full((1, 1, 2), 0.5))
    np.savez_compressed(sdir / "nu_final_policy.npz", nu=np.array([[[0.4, 0.6]]]))
    solution = {
        "config": {"M": 10, "R": 1},
        "input_hashes": {n: sha256_file(tdir / n)
                         for n in ("tensor_policy.npz", "tensor_ref.npz", "meta.json")},
        "artifact_hashes": {n: sha256_file(sdir / n)
                            for n in ("nu_update.npz", "nu_final_policy.npz")},
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


def test_nu_final_policy_cannot_stand_in_for_the_target_opponent(tmp_path):
    import shutil
    from scripts.nbpo.nbpo_common import sha256_file, write_json

    tdir, sdir, solution = _solver_pair(tmp_path)
    shutil.copy(sdir / "nu_final_policy.npz", sdir / "nu_update.npz")
    solution["artifact_hashes"]["nu_update.npz"] = sha256_file(sdir / "nu_update.npz")
    write_json(sdir / "solution.json", solution)
    with pytest.raises(ValueError, match="byte-identical at R=1"):
        verify_solver_input_chain(tdir, sdir, solution)


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
    """Records the kwargs apply_chat_template is called with."""

    def __init__(self, accepts_thinking=True):
        self.calls = []
        self.accepts_thinking = accepts_thinking

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
