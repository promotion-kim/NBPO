"""Real-mode end-to-end tests for run_nbpo_stage (no --dry-run, no LLM).

Stub executables stand in for the GPU steps -- stub precompute writes a valid
dataset + provenance sidecar, stub train writes a checkpoint with a new content
fingerprint, stub decode asserts it received exactly that checkpoint and then
runs the real synchronous decoder with the mock backend -- while the
materialization, parsing, manifest-binding, prompt-set and gate logic are the
production code paths.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import save_file

from mnpo_scripts.mnpo_trainer import validate_nbpo_args
from mnpo_scripts.precompute_provenance import checkpoint_fingerprint
from scripts.nbpo.decode_candidate import decode_candidate
from scripts.nbpo.run_nbpo_stage import (
    check_seed_files_cover_prompts,
    materialize_run_config,
    parse_run_config,
    verify_decode_manifest,
)

REPO = Path(__file__).parents[1]
FIX = REPO / "tests" / "fixtures" / "nbpo_realmode"
TOY = REPO / "tests" / "fixtures" / "nbpo_toy"
# MKL_THREADING_LAYER=GNU: the conda torch/numpy pair otherwise refuses to load in
# subprocesses ("MKL_THREADING_LAYER=INTEL is incompatible with libgomp").
ENV = {**os.environ, "PYTHONPATH": str(REPO), "MKL_THREADING_LAYER": "GNU"}


def make_checkpoint(path: Path, fill: float = 0.0) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"model_type": "stub", "hidden_size": 4}))
    (path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "stub"}))
    save_file({"w": torch.full((4, 4), fill, dtype=torch.float32)}, str(path / "model.safetensors"))
    return path


def stage_config(tmp: Path) -> Path:
    cfg = yaml.safe_load((FIX / "config.yaml").read_text())
    cfg["gate"]["parent_model"] = str(tmp / "ckpt" / "pi_t")
    cfg["gate"]["promote_to"] = str(tmp / "ckpt" / "pi_next")
    # relative fixture paths ("../nbpo_toy/...") resolve against the config's
    # directory, so place the config one level below a symlink to the toy fixture
    (tmp / "cfg").mkdir(exist_ok=True)
    link = tmp / "nbpo_toy"
    if not link.exists():
        link.symlink_to(TOY, target_is_directory=True)
    out = tmp / "cfg" / "config.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


@pytest.fixture(scope="module")
def real_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("realmode")
    make_checkpoint(tmp / "ckpt" / "pi_t", fill=0.0)
    cfg = stage_config(tmp)
    workdir = tmp / "work"
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.nbpo.run_nbpo_stage", "--config", str(cfg),
         "--workdir", str(workdir)],                      # NO --dry-run
        cwd=REPO, env=ENV, capture_output=True, text=True, timeout=900)
    return {"tmp": tmp, "workdir": workdir, "proc": proc, "config": cfg}


def test_real_mode_stub_orchestration_completes(real_run):
    proc = real_run["proc"]
    assert proc.returncode == 0, proc.stderr[-3000:]
    for step in ("precompute", "run_config", "train", "decode", "monitoring_prompt_sets",
                 "eval_monitoring", "gate", "complete"):
        assert f"step={step} status=ok" in proc.stdout, (step, proc.stdout[-2000:])
    assert "dry-run" not in proc.stdout.lower()
    rec = json.loads(next(real_run["workdir"].glob("*_record.json")).read_text())
    assert rec["dry_run"] is False
    assert rec["decode_manifest_binding"] != "skipped (dry_run.candidate_policy_files fixture)"
    assert rec["judges"]["monitoring"]["model"] == "mock-judge-monitor-v1"
    assert rec["judges"]["training"]["model"] == "mock-judge-train-v1"


def test_real_mode_materializes_parseable_run_config(real_run, tmp_path):
    run_cfg = real_run["workdir"] / "run_config.yaml"
    assert run_cfg.exists()
    d = yaml.safe_load(run_cfg.read_text())
    assert d["loss_type"] == "nbpo" and d["logp_reduction"] == "sum"
    assert d["max_history_t"] == 1 and d["history_weights"] == [1.0]
    assert d["reference_anchor_weight"] == 0.0 and d["preference_sft_weight"] == 0.0
    assert d["model_name_or_path"] == str(real_run["tmp"] / "ckpt" / "pi_t")
    assert list(d["dataset_mixer"]) == [str(real_run["workdir"] / "precomputed")]
    assert d["output_dir"].endswith("pi_next.candidate")
    model_args, data_args, training_args = parse_run_config(run_cfg)   # run_mnpo's parser
    assert training_args.loss_type == "nbpo" and training_args.optim == "adafactor"
    # a stage config cannot override the invariants, and an unknown field is rejected
    cfg = yaml.safe_load(real_run["config"].read_text())
    cfg["trainer"]["reference_anchor_weight"] = 0.05
    out = materialize_run_config(cfg, "/p", "/d", "/c", tmp_path / "rc.yaml")
    assert out["reference_anchor_weight"] == 0.0
    cfg["trainer"]["no_such_field_xyz"] = 1
    with pytest.raises(Exception):
        materialize_run_config(cfg, "/p", "/d", "/c", tmp_path / "rc_bad.yaml")


def test_real_mode_uses_new_candidate_checkpoint_for_decode(real_run):
    tmp, wd = real_run["tmp"], real_run["workdir"]
    received = (tmp / "ckpt" / "decode_received_checkpoint.txt").read_text().strip()
    assert received == str((tmp / "ckpt" / "pi_next.candidate").resolve())
    man = json.loads((wd / "candidate_monitoring_manifest.json").read_text())
    rec = json.loads(next(wd.glob("*_record.json")).read_text())
    assert man["candidate_fingerprint"] == rec["candidate_fingerprint"]
    parent_fp = checkpoint_fingerprint(str(tmp / "ckpt" / "pi_t"))
    assert rec["parent_fingerprint"] == parent_fp
    assert rec["candidate_fingerprint"] != parent_fp        # a genuinely new checkpoint
    # promotion (if accepted) is versioned and symlinked; parent untouched either way
    if rec["accepted"]:
        link = tmp / "ckpt" / "pi_next"
        assert link.is_symlink()
        assert checkpoint_fingerprint(str(link.resolve())) == rec["candidate_fingerprint"]
        assert (tmp / "ckpt" / "pi_next.versions" / "PROMOTION.json").exists()
    assert checkpoint_fingerprint(str(tmp / "ckpt" / "pi_t")) == parent_fp


def test_decode_is_synchronous(tmp_path):
    ck = make_checkpoint(tmp_path / "ck", fill=1.0)
    man = decode_candidate(ck, TOY / "assets" / "monitoring_prompts.jsonl", [0, 1, 2, 3],
                           0.9, 0.95, 32, tmp_path / "out", tmp_path / "man.json", backend="mock")
    assert man["synchronous"] is True
    for s in (0, 1, 2, 3):
        p = Path(man["outputs"][f"s{s}"]["path"])
        assert p.exists() and len(json.loads(p.read_text())) == man["n_prompts"]
    assert man["candidate_fingerprint"] == checkpoint_fingerprint(str(ck))
    assert man["prompt_ids"] == sorted(man["prompt_ids"])
    verify_decode_manifest(tmp_path / "man.json", ck, man["prompt_ids"], [0, 1, 2, 3])


def test_gate_rejects_stale_response_manifest(tmp_path):
    ck = make_checkpoint(tmp_path / "ck")
    man = decode_candidate(ck, TOY / "assets" / "monitoring_prompts.jsonl", [0, 1],
                           0.9, 0.95, 32, tmp_path / "out", tmp_path / "man.json", backend="mock")
    p = Path(man["outputs"]["s1"]["path"])
    rows = json.loads(p.read_text()); rows[0]["generated_text"] += " edited"
    p.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="edited or replaced"):
        verify_decode_manifest(tmp_path / "man.json", ck, man["prompt_ids"], [0, 1])


def test_gate_rejects_wrong_checkpoint_fingerprint(tmp_path):
    a = make_checkpoint(tmp_path / "a", fill=0.0)
    b = make_checkpoint(tmp_path / "b", fill=2.0)      # same size, different weights
    man = decode_candidate(a, TOY / "assets" / "monitoring_prompts.jsonl", [0],
                           0.9, 0.95, 32, tmp_path / "out", tmp_path / "man.json", backend="mock")
    with pytest.raises(ValueError, match="fingerprint"):
        verify_decode_manifest(tmp_path / "man.json", b, man["prompt_ids"], [0])


def test_gate_rejects_missing_monitoring_prompts(tmp_path):
    ck = make_checkpoint(tmp_path / "ck")
    man = decode_candidate(ck, TOY / "assets" / "monitoring_prompts.jsonl", [0],
                           0.9, 0.95, 32, tmp_path / "out", tmp_path / "man.json", backend="mock")
    with pytest.raises(ValueError, match="missing=\\['m99'\\]"):
        verify_decode_manifest(tmp_path / "man.json", ck, man["prompt_ids"] + ["m99"], [0])
    with pytest.raises(ValueError, match="unexpected=\\['m01'\\]"):
        verify_decode_manifest(tmp_path / "man.json", ck, ["m00"], [0])
    with pytest.raises(ValueError, match="seeds must match exactly"):
        verify_decode_manifest(tmp_path / "man.json", ck, man["prompt_ids"], [0, 7])


def test_every_seed_has_exact_monitoring_prompt_set(tmp_path):
    good = TOY / "assets" / "mon_ref_r0.json"
    rows = json.loads(good.read_text())
    (tmp_path / "short.json").write_text(json.dumps(rows[:-1]))     # one prompt missing
    ids = {r["prompt_id"] for r in rows}
    check_seed_files_cover_prompts([f"r0={good}"], ids, "reference")
    with pytest.raises(ValueError, match="seed r1.*missing"):
        check_seed_files_cover_prompts([f"r0={good}", f"r1={tmp_path / 'short.json'}"], ids,
                                       "reference")


def test_wrong_history0_checkpoint_fails():
    from types import SimpleNamespace

    args = SimpleNamespace(loss_type="nbpo", reference_anchor_weight=0.0, preference_sft_weight=0.0,
                           logp_reduction="sum", max_history_t=1, history_weights=[1.0], weights=None,
                           nbpo_target_column="nbpo_weighted_z")
    cols = ["nbpo_weighted_z", "history0_chosen_logps", "history0_rejected_logps"]
    meta = {"logp_reduction": "sum", "tokenizer_hash": "t", "chat_template_hash": "c",
            "history_fingerprints": ["ckpt-A"]}
    validate_nbpo_args(args, dataset_columns=cols, precompute_meta=meta, tokenizer_hash="t",
                       chat_template_hash="c", expected_parent_fingerprint="ckpt-A")
    with pytest.raises(ValueError, match="proximal centre"):
        validate_nbpo_args(args, dataset_columns=cols, precompute_meta=meta, tokenizer_hash="t",
                           chat_template_hash="c", expected_parent_fingerprint="ckpt-B")
