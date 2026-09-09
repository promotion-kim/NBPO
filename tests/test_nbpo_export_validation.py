"""Tiny CPU models exercise export checks without touching campaign models."""
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

from scripts.experiments.nbpo_repair_20260909 import validate_export as v


@pytest.fixture
def tiny(tmp_path):
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, bos_token_id=1, eos_token_id=2,
        tie_word_embeddings=False, attention_dropout=0.)
    torch.manual_seed(42)
    model = LlamaForCausalLM._from_config(config, torch_dtype=torch.bfloat16,
                                        attn_implementation="sdpa")
    base, export = tmp_path / "base", tmp_path / "export"
    model.save_pretrained(base, safe_serialization=True)
    with torch.no_grad():
        model.model.layers[0].self_attn.q_proj.weight[0, 0].add_(.1)
    model.save_pretrained(export, safe_serialization=True)
    return config, model, base, export


def test_controller_selection_rejects_wrong_step_arm_and_duplicates():
    record = {"arm": "wbc", "global_step": 1750}
    assert v.controller_arm(record, "wbc") == record
    assert v.controller_arm({"arms": [record]}, "wbc") == record
    with pytest.raises(ValueError, match="exactly one"):
        v.controller_arm({"arms": [record, record]}, "wbc")
    with pytest.raises(ValueError, match="exactly one"):
        v.controller_arm(record, "mse")
    with pytest.raises(ValueError, match="fixed1750"):
        v.controller_arm({**record, "global_step": 20}, "wbc")


def test_controller_hash_attestation_rejects_stale_or_modified_export(tmp_path):
    export = tmp_path / "arms/wbc_primary_v1"
    job = tmp_path / "jobs/wbc_primary_v1"
    export.mkdir(parents=True)
    job.mkdir(parents=True)
    v.write_json(export / "trainer_state.json", {"global_step": 1750})
    v.write_json(job / "exit.json", {"exit_code": 0})
    (export / "model.safetensors").write_bytes(b"example-only")
    record = {"arm": "wbc", "global_step": 1750,
        "trainer_state_sha256": v.file_hash(export / "trainer_state.json"),
        "job_exit_sha256": v.file_hash(job / "exit.json"),
        "weight_sha256": {"model.safetensors": v.file_hash(export / "model.safetensors")}}
    manifest = tmp_path / "controller.json"
    v.write_json(manifest, {"arms": [record]})
    assert v.verify_controller_export(tmp_path, "wbc", manifest)[0] == export
    (export / "model.safetensors").write_bytes(b"modified")
    with pytest.raises(ValueError, match="weights differ"):
        v.verify_controller_export(tmp_path, "wbc", manifest)


def test_full_tensor_schema_checks_keys_shapes_dtype_and_complete_index(tiny):
    config, _, _, export = tiny
    index = v.tensor_inventory(export)
    report = v.check_complete_tensor_schema(config, index, require_8b=False)
    assert report["missing_keys"] == report["unexpected_keys"] == []
    assert report["serialized_dtype"] == "BF16"
    missing = dict(index)
    missing.pop("lm_head.weight")
    with pytest.raises(ValueError, match="Incomplete full-policy"):
        v.check_complete_tensor_schema(config, missing, require_8b=False)
    mismatch = {k: dict(val) for k, val in index.items()}
    mismatch["lm_head.weight"]["shape"] = [1, 1]
    with pytest.raises(ValueError, match="mismatched"):
        v.check_complete_tensor_schema(config, mismatch, require_8b=False)
    mismatch["lm_head.weight"] = {**index["lm_head.weight"], "dtype": "F32"}
    with pytest.raises(ValueError, match="must be BF16"):
        v.check_complete_tensor_schema(config, mismatch, require_8b=False)
    v.write_json(export / "model.safetensors.index.json", {"weight_map": {"lm_head.weight": "wrong.safetensors"}})
    with pytest.raises(ValueError, match="index does not exactly match"):
        v.tensor_inventory(export)


def test_strict_loading_and_saved_loaded_equality_with_diagnostic_base_delta(tiny):
    config, _, base, export = tiny
    loaded, info = v.strict_hf_load(export, device="cpu")
    assert all(not values for values in info.values())
    assert not any(p.requires_grad for p in loaded.parameters())
    rows = v.compare_saved_parameters(loaded, export, base, v.tensor_inventory(export), v.tensor_inventory(base))
    assert all(row["entire_saved_vs_loaded_exact"] for row in rows)
    assert rows[0]["sample_n_changed"] == 1 and rows[0]["sample_delta_l2"] > 0
    assert rows[-1]["sample_n_changed"] == 0  # zero difference is not an outcome gate
    with torch.no_grad():
        loaded.model.norm.weight[0].add_(1)
    with pytest.raises(ValueError, match="differs from serialized"):
        v.compare_saved_parameters(loaded, export, base, v.tensor_inventory(export), v.tensor_inventory(base))


def test_strict_hf_loader_rejects_missing_weight_instead_of_random_init(tiny):
    _, model, _, export = tiny
    state = {key: value.detach().contiguous() for key, value in model.state_dict().items()
             if key != "lm_head.weight"}
    save_file(state, export / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(ValueError, match="strict load"):
        v.strict_hf_load(export, device="cpu")


def test_rope_requires_exact_original_fp32_not_rounded_recast(tiny):
    config, _, _, export = tiny
    loaded, _ = v.strict_hf_load(export, device="cpu")
    rows = v.check_rotary_buffers(loaded, config)
    assert rows and rows[0]["exact_pinned_config_match"]
    rotary = loaded.model.rotary_emb
    rotary.inv_freq = rotary.inv_freq.bfloat16().float()
    with pytest.raises(ValueError, match="exact pinned-config FP32"):
        v.check_rotary_buffers(loaded, config)


def test_structure_rejects_changed_rope_or_small_model_as_8b(tiny):
    config, _, _, _ = tiny
    assert v.check_config(config, config, require_8b=False)["model_type"] == "llama"
    with pytest.raises(ValueError, match="full Llama3.1-8B"):
        v.check_config(config, config)
    altered = LlamaConfig(**config.to_dict())
    altered.rope_theta *= 2
    with pytest.raises(ValueError, match="RoPE config differs"):
        v.check_config(config, altered, require_8b=False)


@pytest.mark.parametrize("ids,ends,limit,expected", [([4, 2], [2, 3], 4, ("terminal", 2)),
    ([4, 5, 6], [2, 3], 3, ("max_new_tokens", None)), ([3], [2, 3], 1, ("terminal", 3))])
def test_generation_native_terminal_and_length_contract(ids, ends, limit, expected):
    assert v.classify_stop(ids, ends, limit) == expected


@pytest.mark.parametrize("ids", [[], [2, 4], [4], [4, 5, 6, 7, 8]])
def test_generation_rejects_wrong_stop_and_continuation_after_eos(ids):
    with pytest.raises(ValueError):
        v.classify_stop(ids, [2, 3], 4)


def test_native_tokenizer_pad_fallback_and_unchanged_prompt_events():
    class Tokenizer:
        eos_token_id = 2
        chat_template = "native-template"
        backend_tokenizer = SimpleNamespace(to_str=lambda: '{"model":{"merges":[]},"padding":null}')
        def __init__(self, pad=None):
            self.pad_token_id = pad
        @property
        def special_tokens_map(self):
            return {"eos_token": "EOS", "pad_token": "EOS" if self.pad_token_id == 2 else "other"}
        def get_vocab(self):
            return {"EOS": 2, "prompt": 3}
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": True, "add_generation_prompt": True}
            assert len(messages) == 1 and messages[0]["role"] == "user"
            return [1, 3, 4]
    generation = {"eos_token_id": [2, 5]}
    contract = v.compare_tokenizers(Tokenizer(), Tokenizer(2), generation, generation)
    assert contract["raw_pad_token_ids"] == {"base": None, "export": 2}
    assert contract["terminal_ids"] == [2, 5]
    assert all(p["native_input_ids"] == [1, 3, 4] for p in contract["probes"])
    with pytest.raises(ValueError, match="terminal IDs differ"):
        v.compare_tokenizers(Tokenizer(), Tokenizer(2), generation, {"eos_token_id": [2]})
    changed = Tokenizer(2)
    changed.backend_tokenizer = SimpleNamespace(to_str=lambda: '{"model":{"merges":["changed"]}}')
    with pytest.raises(ValueError, match="algorithm/merges/normalizer"):
        v.compare_tokenizers(Tokenizer(), changed, generation, generation)


def test_cpu_repeated_forward_and_greedy_raw_token_smoke(tiny, monkeypatch):
    _, _, _, export = tiny
    loaded, _ = v.strict_hf_load(export, device="cpu")
    tokenizer = SimpleNamespace(pad_token_id=2, bos_token_id=1,
        decode=lambda ids, skip_special_tokens: " ".join(map(str, ids)))
    contract = {"terminal_ids": [2, 3], "probes": [
        {"prompt": "arbitrary short probe", "native_input_ids": [1, 4, 5]}]}
    monkeypatch.setattr(v, "MAX_NEW_TOKENS", 3)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = v.smoke_forward(loaded, tokenizer, contract, device="cpu")
    finally:
        torch.set_num_threads(previous)
    assert result["cases"][0]["repeat_logits_exact"]
    assert result["cases"][0]["repeat_greedy_token_ids_exact"]
    assert result["cases"][0]["n_output_tokens"] <= 3
    json.dumps(result, allow_nan=False)
