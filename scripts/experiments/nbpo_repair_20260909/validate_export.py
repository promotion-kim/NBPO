"""Read-only, post-controller full-policy export validation; no training changes.

Metadata mode uses CPU only. Full mode requires explicit allocation of exactly
one visible GPU and the pinned training Transformers 4.45.2 stack. The short
harmless probes are transport/loading checks, never benchmark/quality scores.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

from scripts.experiments.nbpo_repair_20260909.common import (
    digest, file_hash, object_hash, write_json,
)

BASE_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
BASE_CONFIG_SHA256 = "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e"
EXPECTED_STEP = 1750
TRANSFORMERS_VERSION = "4.45.2"
SMOKE_PROMPTS = (
    "Name two colors a paper kite might have. Keep the answer short.",
    "Write one plain sentence about a mug beside a window.",
)
MAX_NEW_TOKENS = 64
SEED = 20260909
STRUCTURAL_KEYS = (
    "model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "vocab_size", "hidden_act",
    "max_position_embeddings", "rms_norm_eps", "rope_theta", "rope_scaling",
    "tie_word_embeddings", "attention_bias", "attention_dropout", "mlp_bias",
    "bos_token_id", "eos_token_id",
)


def controller_arm(manifest, arm):
    records = manifest.get("arms", [manifest])
    selected = [r for r in records if r.get("arm") == arm]
    if len(selected) != 1:
        raise ValueError("Controller must contain exactly one matching arm")
    record = selected[0]
    if record.get("global_step") != EXPECTED_STEP:
        raise ValueError("Controller does not attest fixed1750 export")
    return record


def verify_controller_export(root, arm, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    record = controller_arm(manifest, arm)
    export = root / "arms" / f"{arm}_primary_v1"
    if not export.resolve().is_relative_to(root.resolve()):
        raise ValueError("Export resolves outside task root")
    state_path = export / "trainer_state.json"
    state = json.loads(state_path.read_text())
    if state.get("global_step") != EXPECTED_STEP or file_hash(state_path) != record["trainer_state_sha256"]:
        raise ValueError("Trainer state differs from controller fixed1750 attestation")
    exit_path = root / "jobs" / f"{arm}_primary_v1" / "exit.json"
    if file_hash(exit_path) != record["job_exit_sha256"]:
        raise ValueError("Job exit differs from controller attestation")
    if json.loads(exit_path.read_text()).get("exit_code") != 0:
        raise ValueError("Training job did not finish successfully")
    paths = sorted(export.glob("*.safetensors"))
    observed = {p.name: file_hash(p) for p in paths}
    if not observed or observed != record["weight_sha256"]:
        raise ValueError("Export weights differ from controller hashes")
    if (export / "adapter_config.json").exists() or any("adapter" in p.name for p in paths):
        raise ValueError("An adapter cannot stand in for a full exported policy")
    return export, {"controller_manifest_sha256": file_hash(manifest_path),
        "controller_record": record, "weight_sha256": observed,
        "trainer_state_sha256": file_hash(state_path), "global_step": state["global_step"]}


def verify_base_inventory(base, inventory_path):
    inventory = json.loads(inventory_path.read_text())
    if inventory.get("base_revision") != BASE_REVISION:
        raise ValueError("Inventory base revision is not the pinned campaign base")
    assets = {Path(r["path"]).name: r for r in inventory["assets"]
              if Path(r["path"]).parent.name == "Llama-3.1-8B-Instruct"}
    required = {p.name for p in base.glob("*.safetensors")} | {
        "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "model.safetensors.index.json"}
    if not required <= set(assets) or not any(n.endswith(".safetensors") for n in required):
        raise ValueError("Pinned base inventory lacks required weights/tokenizer/config files")
    hashes = {}
    for name in sorted(required):
        observed = file_hash(base / name)
        if observed != assets[name]["sha256"]:
            raise ValueError(f"Base file differs from captured inventory: {name}")
        hashes[name] = observed
    if hashes["config.json"] != BASE_CONFIG_SHA256:
        raise ValueError("Base config is not the pinned Llama3.1-8B config")
    return {"revision": BASE_REVISION, "inventory_sha256": file_hash(inventory_path), "files_sha256": hashes}


def check_config(base_config, export_config, *, require_8b=True):
    wrong = [k for k in STRUCTURAL_KEYS if getattr(base_config, k, None) != getattr(export_config, k, None)]
    if wrong:
        raise ValueError(f"Export/base architecture or RoPE config differs: {wrong}")
    if getattr(export_config, "quantization_config", None) is not None:
        raise ValueError("Quantized exports are outside this BF16 full-policy contract")
    if require_8b:
        expected = {"model_type": "llama", "hidden_size": 4096, "intermediate_size": 14336,
            "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
            "vocab_size": 128256}
        if any(getattr(export_config, key, None) != val for key, val in expected.items()):
            raise ValueError("Export is not the full Llama3.1-8B architecture")
    return {k: getattr(export_config, k, None) for k in STRUCTURAL_KEYS}


def tensor_inventory(directory):
    from safetensors import safe_open
    result, file_keys = {}, {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            file_keys[path.name] = list(handle.keys())
            for key in handle.keys():
                if key in result:
                    raise ValueError(f"Duplicate serialized tensor key: {key}")
                view = handle.get_slice(key)
                result[key] = {"file": path.name, "shape": list(view.get_shape()), "dtype": view.get_dtype()}
    if not result:
        raise ValueError("No serialized full-policy tensors")
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        declared = json.loads(index_path.read_text())["weight_map"]
        if declared != {key: val["file"] for key, val in result.items()}:
            raise ValueError("Safetensors index does not exactly match actual tensor locations")
    elif len(file_keys) != 1:
        raise ValueError("Sharded model requires a complete safetensors index")
    return result


def check_complete_tensor_schema(config, serialized, *, require_8b=True):
    import torch
    from transformers import AutoModelForCausalLM
    with torch.device("meta"):
        skeleton = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16,
                                                  attn_implementation="sdpa")
    expected = {key: list(value.shape) for key, value in skeleton.state_dict().items()}
    missing, unexpected = sorted(set(expected)-set(serialized)), sorted(set(serialized)-set(expected))
    mismatched = {key: {"saved": serialized[key]["shape"], "expected": expected[key]}
                  for key in set(expected) & set(serialized) if serialized[key]["shape"] != expected[key]}
    if missing or unexpected or mismatched:
        raise ValueError(f"Incomplete full-policy tensor schema: missing={missing}, unexpected={unexpected}, mismatched={mismatched}")
    if {v["dtype"] for v in serialized.values()} != {"BF16"}:
        raise ValueError("Saved full-policy tensors must be BF16, with no implicit load-time downcast")
    count = sum(math.prod(shape) for shape in expected.values())
    if require_8b and not 8_000_000_000 < count < 8_100_000_000:
        raise ValueError("Serialized parameter count is not the complete Llama8B policy")
    del skeleton
    return {"n_tensors": len(expected), "serialized_parameter_count": count,
        "missing_keys": missing, "unexpected_keys": unexpected, "mismatched_shapes": mismatched,
        "serialized_dtype": "BF16"}


def terminal_ids(config):
    values = config.get("eos_token_id")
    values = [values] if isinstance(values, int) else values
    if (not isinstance(values, list) or not values or any(type(i) is not int or i < 0 for i in values)
            or len(values) != len(set(values))):
        raise ValueError("Terminal IDs must be explicit, unique nonnegative integers")
    return values


def compare_tokenizers(base_tokenizer, export_tokenizer, base_generation, export_generation):
    # Saving the training tokenizer materializes pad=eos when base had no pad.
    # Record this batching-only fallback explicitly, not a native-event change.
    from mnpo_scripts.precompute_provenance import tokenizer_content_hashes
    raw_padding = {"base": base_tokenizer.pad_token_id, "export": export_tokenizer.pad_token_id}
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    if export_tokenizer.pad_token_id is None:
        export_tokenizer.pad_token_id = export_tokenizer.eos_token_id
    hashes = tokenizer_content_hashes(base_tokenizer)
    if hashes != tokenizer_content_hashes(export_tokenizer):
        raise ValueError("Export tokenizer vocabulary/special tokens/native chat template differs from base")
    if not base_tokenizer.chat_template or base_tokenizer.chat_template != export_tokenizer.chat_template:
        raise ValueError("Native chat template is absent or changed")
    backend_hashes = []
    for tokenizer in (base_tokenizer, export_tokenizer):
        if not hasattr(tokenizer, "backend_tokenizer"):
            raise ValueError("Native Llama tokenizer validation requires its full fast-tokenizer backend")
        backend = json.loads(tokenizer.backend_tokenizer.to_str())
        # Runtime padding/truncation state is not the tokenizer algorithm.
        for key in ("padding", "truncation"):
            backend.pop(key, None)
        backend_hashes.append(object_hash(backend))
    if backend_hashes[0] != backend_hashes[1]:
        raise ValueError("Export tokenizer algorithm/merges/normalizer differs from base")
    ends = terminal_ids(base_generation)
    if ends != terminal_ids(export_generation):
        raise ValueError("Export/base generation terminal IDs differ")
    probes = []
    for prompt in SMOKE_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        base_ids = base_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        exported_ids = export_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if base_ids != exported_ids:
            raise ValueError("Native prompt tokenization differs after export")
        probes.append({"prompt": prompt, "prompt_sha256": digest(prompt), "messages": messages,
            "native_input_ids": base_ids, "native_input_ids_sha256": object_hash(base_ids)})
    return {**hashes, "tokenizer_backend_core_sha256": backend_hashes[0], "raw_pad_token_ids": raw_padding,
        "effective_pad_token_id": export_tokenizer.pad_token_id, "terminal_ids": ends,
        "native_user_only": True, "add_generation_prompt": True, "probes": probes}


def strict_hf_load(path, *, device):
    import torch
    from transformers import AutoModelForCausalLM
    model, info = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True, trust_remote_code=False,
        low_cpu_mem_usage=True, use_safetensors=True, output_loading_info=True)
    bad = {key: info.get(key, []) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
    # Transformers can suppress architecture-specific missing-key warnings
    # (including lm_head). Independently inspect serialized keys and meta state.
    saved_keys = set(tensor_inventory(path))
    state = model.state_dict()
    bad["missing_keys"] = sorted(set(bad["missing_keys"]) | (set(state)-saved_keys))
    bad["unexpected_keys"] = sorted(set(bad["unexpected_keys"]) | (saved_keys-set(state)))
    bad["unmaterialized_keys"] = [key for key, value in state.items() if value.is_meta]
    if any(bad.values()):
        raise ValueError(f"HF strict load did not consume the complete exact checkpoint: {bad}")
    if hasattr(model, "peft_config") or getattr(model, "_hf_peft_config_loaded", False):
        raise ValueError("Unexpected PEFT/adapter loader path")
    if {p.dtype for p in model.parameters()} != {torch.bfloat16}:
        raise ValueError("Loaded policy parameters are not uniformly BF16")
    return model.to(device=device).eval().requires_grad_(False), bad


def check_rotary_buffers(model, pinned_config):
    import torch
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    expected = LlamaRotaryEmbedding(config=pinned_config, device="cpu")
    rows = []
    for name, module in model.named_modules():
        if not isinstance(module, LlamaRotaryEmbedding):
            continue
        observed = module.inv_freq.detach().cpu()
        exact = observed.dtype == torch.float32 and torch.equal(observed, expected.inv_freq)
        original = getattr(module, "original_inv_freq", None)
        original_exact = (isinstance(original, torch.Tensor) and original.dtype == torch.float32
                          and torch.equal(original.detach().cpu(), expected.inv_freq))
        scaling_exact = module.attention_scaling == expected.attention_scaling
        if not exact or not original_exact or not scaling_exact:
            raise ValueError(f"Export reload RoPE differs from exact pinned-config FP32 values: {name}")
        rows.append({"module": name, "dtype": str(observed.dtype), "exact_pinned_config_match": exact,
            "original_inv_freq_exact": original_exact, "values": observed.tolist(),
            "values_sha256": digest(observed.contiguous().view(torch.uint8).numpy().tobytes()),
            "attention_scaling": float(module.attention_scaling)})
    if not rows:
        raise ValueError("No Llama RoPE buffers found")
    return rows


def compare_saved_parameters(model, export, base, export_index, base_index):
    import torch
    from safetensors import safe_open
    layers = model.config.num_hidden_layers
    selected = [f"model.layers.{i}.self_attn.q_proj.weight" for i in sorted({0, layers//2, layers-1})]
    selected += ["model.norm.weight"]
    parameters = dict(model.named_parameters())
    rows = []
    for key in selected:
        with safe_open(export / export_index[key]["file"], framework="pt", device="cpu") as handle:
            saved = handle.get_tensor(key)
        loaded = parameters[key].detach().cpu()
        if saved.dtype != loaded.dtype or not torch.equal(saved, loaded):
            raise ValueError(f"Loaded parameter differs from serialized tensor: {key}")
        if not torch.isfinite(loaded).all():
            raise ValueError(f"Nonfinite selected exported parameter: {key}")
        with safe_open(base / base_index[key]["file"], framework="pt", device="cpu") as handle:
            original = handle.get_tensor(key)
        # Uniform deterministic locations, chosen before results, no search for
        # changed entries. BF16 rounding may make small training updates vanish.
        ids = torch.linspace(0, saved.numel()-1, min(257, saved.numel())).long().unique()
        a, b = saved.reshape(-1)[ids], original.reshape(-1)[ids]
        delta = a.double() - b.double()
        rows.append({"parameter": key, "shape": list(saved.shape), "dtype": str(saved.dtype),
            "entire_saved_vs_loaded_exact": True, "n_full_tensor_values_compared": saved.numel(),
            "sample_flat_indices": ids.tolist(), "export_sample_values": a.float().tolist(),
            "base_sample_values": b.float().tolist(), "sample_n_changed": int((a != b).sum()),
            "sample_delta_l2": float(delta.norm()), "sample_delta_max_abs": float(delta.abs().max()),
            "difference_is_diagnostic_not_acceptance_gate": True})
    return rows


def classify_stop(ids, ends, max_tokens):
    if not ids or len(ids) > max_tokens:
        raise ValueError("Greedy smoke returned an invalid token count")
    terminal_positions = [i for i, value in enumerate(ids) if value in ends]
    if terminal_positions:
        if terminal_positions != [len(ids)-1]:
            raise ValueError("Generated tokens continued after a configured terminal")
        return "terminal", ids[-1]
    if len(ids) != max_tokens:
        raise ValueError("Generation stopped before max length without a configured terminal")
    return "max_new_tokens", None


def smoke_forward(model, tokenizer, token_contract, *, device):
    import torch
    from transformers import GenerationConfig
    ends = token_contract["terminal_ids"]
    config = GenerationConfig(do_sample=False, num_beams=1, use_cache=True,
        max_new_tokens=MAX_NEW_TOKENS, eos_token_id=ends,
        pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
        repetition_penalty=1.0, no_repeat_ngram_size=0, forced_bos_token_id=None,
        forced_eos_token_id=None, suppress_tokens=None, begin_suppress_tokens=None)
    rows = []
    with torch.inference_mode():
        for probe in token_contract["probes"]:
            inputs = torch.tensor([probe["native_input_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(inputs)
            first = model(input_ids=inputs, attention_mask=mask, use_cache=False).logits
            second = model(input_ids=inputs, attention_mask=mask, use_cache=False).logits
            if not torch.isfinite(first).all() or not torch.isfinite(second).all():
                raise ValueError("Nonfinite repeated-forward logits")
            error = float((first.float()-second.float()).abs().max())
            if not torch.equal(first, second):
                raise ValueError(f"Identical HF repeated forwards are not bitwise deterministic: {error}")
            generated = []
            for _ in range(2):
                full = model.generate(input_ids=inputs, attention_mask=mask, generation_config=config)
                if not torch.equal(full[:, :inputs.shape[1]], inputs):
                    raise ValueError("Greedy output changed native input token prefix")
                generated.append(full[0, inputs.shape[1]:].tolist())
            if generated[0] != generated[1]:
                raise ValueError("Identical greedy smoke repeats produced different raw token IDs")
            ids = generated[0]
            finish, terminal = classify_stop(ids, ends, MAX_NEW_TOKENS)
            text = tokenizer.decode(ids, skip_special_tokens=True)
            rows.append({**probe, "output_token_ids": ids, "output_token_ids_sha256": object_hash(ids),
                "output_text": text, "output_text_sha256": digest(text),
                "output_with_special_tokens": tokenizer.decode(ids, skip_special_tokens=False),
                "n_output_tokens": len(ids), "finish_reason": finish, "terminal_token_id": terminal,
                "empty_decoded_text_diagnostic": not text.strip(), "repeat_logits_max_abs": error,
                "repeat_logits_exact": True, "repeat_greedy_token_ids_exact": True,
                "logit_dtype": str(first.dtype), "input_shape": list(inputs.shape)})
    return {"cases": rows, "generation_config": config.to_dict(),
        "scope": "arbitrary harmless loading/stop smoke; not evaluation quality, safety, or HF-vLLM logit equivalence"}


def run(args):
    import transformers
    import torch
    from transformers import AutoConfig, AutoTokenizer
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise ValueError("Use pinned deps_train Transformers4.45.2 for this HF reload probe")
    if args.stage == "full" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise ValueError("Full probe requires exactly one explicitly allocated visible CUDA GPU")
    started = time.monotonic()
    export, controller = verify_controller_export(args.root, args.arm, args.controller_manifest)
    base = args.base
    base_evidence = verify_base_inventory(base, args.inventory)
    base_config = AutoConfig.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    config = AutoConfig.from_pretrained(export, local_files_only=True, trust_remote_code=False)
    structure = check_config(base_config, config)
    export_index, base_index = tensor_inventory(export), tensor_inventory(base)
    schema = check_complete_tensor_schema(config, export_index)
    base_tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(export, local_files_only=True, trust_remote_code=False)
    contract = compare_tokenizers(base_tokenizer, tokenizer,
        json.loads((base / "generation_config.json").read_text()),
        json.loads((export / "generation_config.json").read_text()))
    report = {"arm": args.arm, "stage": args.stage, "export_path": str(export),
        "controller": controller, "pinned_base": base_evidence, "architecture": structure,
        "tensor_schema": schema, "tokenizer_and_native_inputs": contract,
        "source_sha256": file_hash(__file__), "input_selection": "two fixed harmless prompts, not drawn from evaluation subsets",
        "runtime": {"transformers": transformers.__version__, "torch": torch.__version__,
            "forward_dtype": "bfloat16", "attention_implementation": "sdpa", "seed": SEED},
        "full_hf_loading_executed": False, "gpu_forward_executed": False}
    if args.stage == "full":
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
        model, load_info = strict_hf_load(export, device="cuda:0")
        report["hf_loading_info"] = load_info
        report["full_hf_loading_executed"] = True
        report["rotary_before_forward"] = check_rotary_buffers(model, base_config)
        report["parameter_comparison"] = compare_saved_parameters(model, export, base, export_index, base_index)
        report["smoke"] = smoke_forward(model, tokenizer, contract, device="cuda:0")
        report["rotary_after_forward"] = check_rotary_buffers(model, base_config)
        report["gpu_forward_executed"] = True
        report["runtime"].update(gpu_name=torch.cuda.get_device_name(0),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cudnn_sdpa_enabled=torch.backends.cuda.cudnn_sdp_enabled(),
            flash_sdpa_enabled=torch.backends.cuda.flash_sdp_enabled(),
            memory_efficient_sdpa_enabled=torch.backends.cuda.mem_efficient_sdp_enabled())
    report.update(passed=True, elapsed_seconds=time.monotonic()-started,
        remaining_checks=[] if args.stage == "full" else ["strict HF reload", "saved/loaded parameter equality",
            "base parameter sample differences", "exact FP32 RoPE", "repeated forward and greedy native-token smoke"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("wbc", "mse"), required=True)
    parser.add_argument("--controller-manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path("/work/models/bases/Llama-3.1-8B-Instruct"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("metadata", "full"), default="metadata")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(args.root.resolve()) or args.output.resolve() == args.root.resolve():
        raise ValueError("Use a new child output directory inside the task root")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "source.json", {"filename": Path(__file__).name,
        "source_sha256": file_hash(__file__), "source_text": Path(__file__).read_text(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}})
    try:
        report = run(args)
    except Exception as exc:
        write_json(args.output / "validation.json", {"passed": False, "stage": args.stage, "arm": args.arm,
            "error_type": type(exc).__name__, "error": str(exc), "source_sha256": file_hash(__file__)})
        raise
    write_json(args.output / "validation.json", report)
    write_json(args.output / "manifest.json", {"validation_sha256": file_hash(args.output / "validation.json"),
        "source_sha256": file_hash(__file__), "source_artifact_sha256": file_hash(args.output / "source.json"),
        "arm": args.arm, "stage": args.stage})
    print(json.dumps({"passed": report["passed"], "stage": args.stage,
        "seconds": report["elapsed_seconds"], "remaining_checks": report["remaining_checks"]}), flush=True)


if __name__ == "__main__":
    main()
