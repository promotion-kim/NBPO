"""End-to-end integration over the PRODUCTION modules (audit section 13).

Everything except the neural weight update runs the real code: the real judge
task construction, the real preference-tensor builder, the real finite-pool
solver, the real pair builder, the real ``mnpo_scripts.precompute`` (with a tiny
CPU model, not a stub that changes empty-split behaviour), the real run_config
materialization and parsing, the real manifest and provenance verification, the
real monitoring tensor construction, and the real gate.

CPU only, no LLM: the judges are the deterministic mock backend and the
candidate is decoded by the mock decoder, both of which still exercise the
production caching, manifest and evaluation paths.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "nbpo_real"

pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="nbpo_real fixture missing")


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env["MKL_THREADING_LAYER"] = "GNU"          # torch + numpy in a fresh interpreter
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def _stage_config(tmp_path: Path, **overrides) -> Path:
    """Copy the fixture into tmp, build the tiny checkpoint, refresh the manifests.

    The checkpoint is built here rather than committed: weights are not fixture
    material. Because the pool manifests bind to its content fingerprint, they
    are rewritten after it exists -- a fingerprint captured before the weights
    landed would be a different checkpoint.
    """
    work = tmp_path / "fixture"
    shutil.copytree(FIXTURE, work)
    sys.path.insert(0, str(REPO))
    from tests.fixtures.nbpo_real.make_fixture import main as build_fixture

    build_fixture(work)
    cfg = yaml.safe_load((work / "config.yaml").read_text())
    cfg["gate"]["parent_model"] = str(work / "assets" / "checkpoints" / "mu")
    cfg["gate"]["promote_to"] = str(tmp_path / "pi_next")
    cfg["gate"]["reference_model"] = str(work / "assets" / "checkpoints" / "mu")
    cfg.update(overrides)
    out = work / "config_filled.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def _run_stage(config: Path, workdir: Path):
    return subprocess.run(
        [sys.executable, "-m", "scripts.nbpo.run_nbpo_stage",
         "--config", str(config), "--workdir", str(workdir)],
        cwd=REPO, env=_env(), capture_output=True, text=True, timeout=1800)


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    """One full production run, reused by the assertions below."""
    tmp = tmp_path_factory.mktemp("nbpo_real")
    cfg = _stage_config(tmp)
    work = tmp / "work"
    proc = _run_stage(cfg, work)
    if proc.returncode != 0:
        pytest.fail(f"stage failed rc={proc.returncode}\n"
                    f"--- stdout ---\n{proc.stdout[-6000:]}\n"
                    f"--- stderr ---\n{proc.stderr[-6000:]}")
    record_path = next(p for p in (work / "stage_record.json", work / "rejection_record.json")
                       if p.exists())
    return {"tmp": tmp, "config": cfg, "fixture": cfg.parent, "work": work, "proc": proc,
            "record": json.loads(record_path.read_text())}


# --------------------------------------------------------------------------- #
# the pipeline runs, over production modules
# --------------------------------------------------------------------------- #
def test_production_pipeline_completes_every_step(staged):
    out = staged["proc"].stdout
    for step in ("rubrics", "reproduction_mode", "response_manifests", "disjointness",
                 "judge_train", "tensor_train", "solve_dual", "build_pairs",
                 "target_invariants", "validation", "precompute", "run_config", "train",
                 "decode", "monitoring_reference_manifest", "monitoring_prompt_sets",
                 "eval_monitoring", "gate", "complete"):
        assert f"step={step} status=ok" in out, f"missing step={step}\n{out[-3000:]}"


def test_real_precompute_train_only(tmp_path):
    """With no validation section and no holdout, the artifact is train-only.

    build_nbpo_pairs still writes an EMPTY pairs_test.jsonl, so the stage must
    omit --test_dir entirely: passing that empty file is what used to reach
    torch.cat([]).
    """
    cfg_path = _stage_config(tmp_path)
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg.pop("validation", None)                 # no external validation
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    work = tmp_path / "work_trainonly"
    proc = _run_stage(cfg_path, work)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    assert (work / "pairs" / "pairs_test.jsonl").exists()
    assert (work / "pairs" / "pairs_test.jsonl").stat().st_size == 0
    meta = json.loads((work / "precomputed" / "precompute_meta.json").read_text())
    assert meta["dataset_splits"] == ["train"]
    assert meta["logp_reduction"] == "sum"
    assert meta["split_sizes"]["train"] > 0
    run_cfg = yaml.safe_load((work / "run_config.yaml").read_text())
    assert run_cfg["dataset_splits"] == ["train"], "run config must not name a missing split"
    from datasets import load_from_disk
    assert sorted(load_from_disk(str(work / "precomputed")).keys()) == ["train"]


def test_requested_empty_split_fails_before_torch_cat(tmp_path):
    """Asking the production precompute for an empty split is a NAMED error."""
    empty = tmp_path / "pairs_test.jsonl"
    empty.write_text("")
    train = tmp_path / "pairs_train.jsonl"
    train.write_text(json.dumps({"prompt": "p", "chosen": "a", "rejected": "b",
                                 "nbpo_weighted_z": 0.1}) + "\n")
    model = tmp_path / "tiny"
    sys.path.insert(0, str(FIXTURE / "stubs"))
    from tests.fixtures.nbpo_real.stubs.real_precompute import ensure_tiny_model
    ensure_tiny_model(model)
    proc = subprocess.run(
        [sys.executable, "-m", "mnpo_scripts.precompute",
         "--model_name_or_path", str(model), "--ref_model", str(model),
         "--train_dir", str(train), "--test_dir", str(empty),
         "--output_dir", str(tmp_path / "out"), "--logp_reduction", "sum",
         "--apply_chat_template", "false", "--max_length", "64",
         "--max_prompt_length", "32", "--report_to", "none"],
        cwd=REPO, env=_env(), capture_output=True, text=True, timeout=900)
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "contains no records" in combined, combined[-2000:]
    assert "torch.cat" not in combined.split("contains no records")[-1]


def test_real_precompute_nonempty_validation(staged):
    """The EXTERNAL validation prompts define the test split, exactly.

    Not a slice of the training pool: the validation prompts have their own
    manifest-bound learner and comparator pools, their own judged tensor, and
    their own Eq. (26) targets.
    """
    work = staged["work"]
    assert (work / "pairs" / "pairs_test.jsonl").stat().st_size > 0
    meta = json.loads((work / "precomputed" / "precompute_meta.json").read_text())
    assert sorted(meta["dataset_splits"]) == ["test", "train"]
    assert meta["split_sizes"]["test"] > 0
    from datasets import load_from_disk
    ds = load_from_disk(str(work / "precomputed"))
    assert sorted(ds.keys()) == ["test", "train"]

    val_ids = {json.loads(l)["prompt_id"] for l in
               (staged["fixture"] / "assets" / "validation_prompts.jsonl").read_text().splitlines()
               if l.strip()}
    test_p = {r["prompt_id"] for r in map(json.loads,
              (work / "pairs" / "pairs_test.jsonl").read_text().splitlines())}
    assert test_p == val_ids, "the validation prompts file must define the test pairs exactly"
    train_p = {r["prompt_id"] for r in map(json.loads,
               (work / "pairs" / "pairs_train.jsonl").read_text().splitlines())}
    assert not (train_p & test_p), "a training prompt may not appear in the validation pairs"
    rec = staged["record"]["validation"]
    assert rec["n_prompts"] == len(val_ids)
    assert "training" in rec["lambda_source"], rec["lambda_source"]
    assert set(rec["manifests"]) == {"learner_pool", "reference_comparator_pool"}


def test_same_prompt_text_under_a_different_id_is_rejected(tmp_path):
    """Leakage that every id-level check passes: same question, new id."""
    cfg_path = _stage_config(tmp_path)
    assets = cfg_path.parent / "assets"
    train_first = json.loads(
        (assets / "train_prompts.jsonl").read_text().splitlines()[0])
    val_path = assets / "validation_prompts.jsonl"
    rows = [json.loads(l) for l in val_path.read_text().splitlines() if l.strip()]
    rows[0]["prompt"] = train_first["prompt"]          # same text, different id
    val_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    proc = _run_stage(cfg_path, tmp_path / "work_leak")
    assert proc.returncode != 0
    assert "share prompt TEXT under different ids" in proc.stdout + proc.stderr


def test_legacy_validation_prompt_file_key_is_refused(tmp_path):
    """The old key never defined the validation pairs; it is refused, not honoured."""
    cfg_path = _stage_config(tmp_path)
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg.pop("validation", None)
    cfg["data"] = {"split_unit": "prompt",
                   "validation_prompt_file": "assets/validation_prompts.jsonl"}
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    proc = _run_stage(cfg_path, tmp_path / "work_legacykey")
    assert proc.returncode != 0
    assert "never defined the validation pairs" in proc.stdout + proc.stderr


# --------------------------------------------------------------------------- #
# provenance chains, end to end
# --------------------------------------------------------------------------- #
def test_solver_pair_precompute_trainer_hash_chain(staged):
    """One unbroken chain: tensors -> solver -> pairs -> precompute -> run_config."""
    from mnpo_scripts.precompute_provenance import sha256_file_hex

    work = staged["work"]
    solution = json.loads((work / "solver" / "solution.json").read_text())
    summary = json.loads((work / "pairs" / "pairs_summary.json").read_text())
    meta = json.loads((work / "precomputed" / "precompute_meta.json").read_text())
    run_cfg = yaml.safe_load((work / "run_config.yaml").read_text())

    # solver -> pairs: the pair artifact re-hashed the tensors itself
    for name in ("tensor_policy.npz", "tensor_ref.npz", "meta.json"):
        assert summary["verified_input_hashes"][name] == solution["input_hashes"][name]
    assert summary["opponent_source_hash"] == solution["artifact_hashes"]["nu_update.npz"]
    assert summary["solver_hash"] == sha256_file_hex(str(work / "solver" / "solution.json"))
    # pairs -> precompute
    assert meta["pair_artifact_sha256"] == sha256_file_hex(
        str(work / "pairs" / "pairs_train.jsonl"))
    assert meta["solver_artifact_sha256"] == summary["solver_hash"]
    # precompute -> run_config (what the trainer validates against)
    assert run_cfg["nbpo_expected_pair_artifact_sha256"] == meta["pair_artifact_sha256"]
    assert run_cfg["nbpo_expected_solver_artifact_sha256"] == meta["solver_artifact_sha256"]
    assert run_cfg["nbpo_expected_parent_checkpoint_fingerprint"] == \
        meta["history_fingerprints"][0]
    assert run_cfg["nbpo_expected_precompute_manifest_sha256"] == \
        meta["precompute_manifest_sha256"]
    # and the dataset manifest actually covers the Arrow shards
    manifest = json.loads((work / "precomputed" / "precompute_manifest.json").read_text())
    assert any(f.endswith(".arrow") for f in manifest["files"]), manifest["files"].keys()


def test_precompute_pair_solver_hash_chain_rejects_a_stale_pair_artifact(staged, tmp_path):
    """Editing the pair file after precompute must stop training, not be ignored."""
    work = staged["work"]
    stale = tmp_path / "stale"
    shutil.copytree(work / "precomputed", stale)
    meta = json.loads((stale / "precompute_meta.json").read_text())
    meta["pair_artifact_sha256"] = "0" * 64
    (stale / "precompute_meta.json").write_text(json.dumps(meta))
    from types import SimpleNamespace

    from mnpo_scripts.mnpo_trainer import validate_nbpo_args

    run_cfg = yaml.safe_load((work / "run_config.yaml").read_text())
    args = SimpleNamespace(loss_type="nbpo", reference_anchor_weight=0.0,
                           preference_sft_weight=0.0, logp_reduction="sum",
                           max_history_t=1, history_weights=[1.0],
                           nbpo_target_column="nbpo_weighted_z",
                           **{k: run_cfg[k] for k in run_cfg if k.startswith("nbpo_expected_")})
    from datasets import load_from_disk

    cols = list(load_from_disk(str(stale))["train"].features)
    with pytest.raises(ValueError, match="stale pair artifact"):
        validate_nbpo_args(args, dataset_columns=cols,
                           precompute_meta=json.loads(
                               (stale / "precompute_meta.json").read_text()))


def test_training_pool_bound_to_parent_and_reference_pool_to_mu(staged):
    record = staged["record"]
    pools = record["response_pool_manifests"]
    assert set(pools) == {"learner_pool", "reference_comparator_pool",
                          "monitoring_reference_comparator_pool"}
    parent_fp = record["parent_fingerprint"]
    assert pools["learner_pool"]["checkpoint_fingerprint"] == parent_fp
    # stage 0: mu IS pi_0, so every pool binds to the same weights
    assert pools["reference_comparator_pool"]["checkpoint_fingerprint"] == parent_fp
    assert pools["monitoring_reference_comparator_pool"]["checkpoint_fingerprint"] == parent_fp
    assert record["monitoring_reference_mu_fingerprint"] == record["mu_fingerprint"]
    assert record["monitoring_reference_manifest_sha256"]


def test_wrong_learner_checkpoint_manifest_fails(staged, tmp_path):
    cfg_path = _stage_config(tmp_path)
    work_dir = cfg_path.parent
    other = work_dir / "assets" / "checkpoints" / "other"
    other.mkdir(parents=True, exist_ok=True)
    (other / "MARKER.json").write_text(json.dumps({"identity": "NOT mu"}))
    man_path = work_dir / "assets" / "learner_pool.json"
    man = json.loads(man_path.read_text())
    from mnpo_scripts.precompute_provenance import checkpoint_fingerprint

    man["checkpoint_fingerprint"] = checkpoint_fingerprint(str(other))
    man_path.write_text(json.dumps(man))
    proc = _run_stage(cfg_path, tmp_path / "work_bad")
    assert proc.returncode != 0
    assert "checkpoint_fingerprint" in proc.stdout + proc.stderr


def test_exact_response_cardinality_is_recorded(staged):
    work = staged["work"]
    meta = json.loads((work / "tensor_train" / "meta.json").read_text())
    card = meta["cardinality"]
    assert card["learner_responses_per_prompt"] == 4
    assert card["reference_comparators_per_prompt"] == 4
    assert card["unordered_learner_pairs"] == 6
    assert card["presentation_orders_per_semantic_comparison"] == 2
    assert card["reproduction_mode"] is True
    summary = json.loads((work / "pairs" / "pairs_summary.json").read_text())
    n_prompts = len(meta["prompt_ids"])
    assert summary["pairs"]["train"] == 6 * n_prompts


# --------------------------------------------------------------------------- #
# judging: content addressing under real orchestration
# --------------------------------------------------------------------------- #
def test_stale_judgment_cache_rejected_when_candidate_text_changes(staged, tmp_path):
    """Re-run with DIFFERENT candidate text under the SAME response ids.

    The monitoring verdicts must be recomputed, not inherited: this is the
    failure that would have let the gate score the previous candidate.
    """
    from scripts.nbpo.judge_pairwise_matrix import MockJudge, build_task_grid, run_judging
    from scripts.nbpo.nbpo_common import (
        normalize_judge_config,
        read_jsonl,
        response_pool_hash,
        verdict_cache_path,
    )

    work = staged["work"]
    caches = list((work / "verdicts").rglob("verdicts.jsonl"))
    assert caches, "verdict cache must be versioned by pool and judge-config hash"
    rows = read_jsonl(caches[0])
    assert all(r.get("comparison_content_hash") for r in rows)
    assert all(r.get("judge_effective_config") for r in rows)

    def pools(text):
        pol = {"s0": {"m00": {"prompt_id": "m00", "prompt": "prompt m00",
                              "generated_text": text}}}
        ref = {"r0": {"m00": {"prompt_id": "m00", "prompt": "prompt m00",
                              "generated_text": "comparator"}}}
        return pol, ref

    ident = normalize_judge_config({"backend": "mock", "model_path": "mock-judge-monitor-v1"},
                                   "monitoring", rubric_sha256="r", max_retries=0)
    out = tmp_path / "mon_cache.jsonl"
    for text in ("OLD candidate answer", "NEW candidate answer"):
        pol, ref = pools(text)
        tasks = build_task_grid(pol, ref, ["clarity"], include_reference_learner=False)
        run_judging(tasks, MockJudge(judge_config=ident), out,
                    judge_model="mock-judge-monitor-v1", rubric_version=1,
                    max_retries=0, judge_identity=ident)
    rows = read_jsonl(out)
    # Two texts x two presentation orders, and no key shared between the texts.
    by_text = {}
    for r in rows:
        by_text.setdefault(r["comparison_payload"]["learner_text_sha256"], set()).add(
            r["comparison_content_hash"])
    assert len(by_text) == 2, "the two candidate texts must hash differently"
    old_keys, new_keys = by_text.values()
    assert not (old_keys & new_keys), "no cache key may be shared across candidates"
    assert len(rows) == 4, "both orders judged for both candidates"
    # and the cache path itself separates the two pools
    old_pool, ref_pool = pools("OLD candidate answer")
    new_pool, _ = pools("NEW candidate answer")
    assert verdict_cache_path(tmp_path, response_pool_hash(old_pool, ref_pool), "c") != \
        verdict_cache_path(tmp_path, response_pool_hash(new_pool, ref_pool), "c")


def test_judge_effective_config_recorded_per_role(staged):
    record = staged["record"]
    judges = record["judges"]
    assert judges["training"]["model"] == "mock-judge-train-v1"
    assert judges["monitoring"]["model"] == "mock-judge-monitor-v1"
    assert judges["training"]["status"] == judges["monitoring"]["status"] == "executed"
    assert judges["training"]["verdict_artifact_hash"] != \
        judges["monitoring"]["verdict_artifact_hash"]
    assert judges["training"]["task_set_hash"] != judges["monitoring"]["task_set_hash"]


def test_final_eval_not_falsely_marked_complete(staged):
    fe = staged["record"]["judges"]["final_eval"]
    assert fe["status"] == "configured_not_executed"
    assert fe["verdict_artifact_hash"] is None and fe["task_set_hash"] is None
    assert fe["model"] == "mock-judge-final-v1"


# --------------------------------------------------------------------------- #
# contract + gate
# --------------------------------------------------------------------------- #
def test_every_artifact_declares_the_implementation_contract(staged):
    work = staged["work"]
    for path in (work / "solver" / "solution.json", work / "pairs" / "pairs_summary.json",
                 work / "tensor_train" / "meta.json"):
        art = json.loads(path.read_text())
        assert art["implementation_type"] == "finite_pool_one_shot_neural_realization", path
        assert art["dual_policy_representation"] == "finite_response_distribution"
        assert art["neural_fits_per_outer_stage"] == 1
    record = staged["record"]
    assert record["implementation_type"] == "finite_pool_one_shot_neural_realization"
    assert record["dual_iterations"] == 60 and record["fixed_point_steps"] == 1
    assert record["reproduction_mode"] is True


def test_gate_decision_and_promotion_or_revert(staged):
    record = staged["record"]
    work = staged["work"]
    assert record["candidate_fingerprint"] != record["parent_fingerprint"], \
        "the candidate must differ from pi_t"
    assert "min_surplus" in record["monitoring"]
    if record["accepted"]:
        promoted = Path(record["lineage"]["promoted"])
        assert promoted.is_dir()
        assert Path(record["lineage"]["symlink"]).resolve() == promoted.resolve()
        assert (work / "stage_record.json").exists()
    else:
        # revert: pi_t retained, and the reason names the nonpositive surplus
        assert record["lineage"]["promoted"] == str(Path(record["lineage"]["parent"]))
        assert "<= 0" in record["reason"]
        assert (work / "rejection_record.json").exists()
