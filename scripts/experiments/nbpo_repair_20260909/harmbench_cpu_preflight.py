"""Tokenization/config-only HarmBench preflight. Never instantiate/call a model.

Official prompt constants are read through AST literal evaluation. In particular
we do not import eval_utils.py, whose import would load a spaCy language model,
and do not execute the copyright hash classifier or any LLM classifier.
"""
from __future__ import annotations

import argparse
import ast
import csv
import importlib.metadata
import inspect
import json
import os
from collections import Counter
from pathlib import Path

from scripts.experiments.nbpo_repair_20260909.common import (
    digest, file_hash, object_hash, write_json, write_jsonl,
)
from scripts.experiments.nbpo_repair_20260909.evaluate_responses import (
    HARMBENCH_COMMIT, harmbench_route, load_responses,
)

MODEL_REVISION = "bda705349d1144fa618770bea64d99ce54e3835b"


def literal_assignment(tree, name):
    return next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))


def preflight(root, model, label):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("This CPU-only preflight requires CUDA_VISIBLE_DEVICES='' explicitly")
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from vllm import SamplingParams
    if torch.cuda.is_initialized():
        raise ValueError("CUDA was unexpectedly initialized before CPU preflight")
    repo = root / "external/HarmBench"
    setup_path = root / "deps_harmbench/runtime_validation.json"
    setup = json.loads(setup_path.read_text())
    if setup["source_commit"] != HARMBENCH_COMMIT or not setup["critical_imports_unchanged"]:
        raise ValueError("Previously validated HarmBench runtime does not match pinned contract")
    helper_path = repo / "eval_utils.py"
    behavior_path = repo / "data/behavior_datasets/harmbench_behaviors_text_test.csv"
    for path in (helper_path, behavior_path):
        relative = str(path.relative_to(repo))
        if file_hash(path) != setup["tracked_helper_and_data_sha256"][relative]:
            raise ValueError("Official helper/data changed after existing runtime validation")
    source = helper_path.read_text()
    tree = ast.parse(source)
    templates = literal_assignment(tree, "LLAMA2_CLS_PROMPT")
    helper_sources = {node.name: digest(ast.get_source_segment(source, node)) for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name in (
                          "compute_results_classifier", "compute_results_hashing")}
    with behavior_path.open() as stream:
        data = list(csv.DictReader(stream))
    behaviors = {r["BehaviorID"]: r for r in data}
    if len(data) != len(behaviors):
        raise ValueError("Duplicate official behavior IDs")
    rows, expected, provenance = load_responses(root, label, "harmbench")
    if len(expected) != 320:
        raise ValueError("The planned HarmBench text test panel must contain320 behaviors")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    config = AutoConfig.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    if model.name != MODEL_REVISION or config.model_type != "llama":
        raise ValueError("Unexpected pinned HarmBench classifier snapshot/config")
    params = SamplingParams(temperature=0., max_tokens=1)
    if params.truncate_prompt_tokens is not None:
        raise ValueError("SamplingParams unexpectedly enables classifier prompt truncation")
    declared_context = config.max_position_embeddings
    if config.rope_scaling is not None:
        raise ValueError("Inspect a changed RoPE context policy explicitly; do not infer its effective limit")
    evaluator_path = root / "code/scripts/experiments/nbpo_repair_20260909/evaluate_responses.py"
    evaluator_tree = ast.parse(evaluator_path.read_text())
    evaluator_function = next(n for n in evaluator_tree.body if isinstance(n, ast.FunctionDef) and n.name == "harmbench")
    llm_call = next(n for n in ast.walk(evaluator_function) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == "LLM")
    llm_kwargs = {kw.arg: ast.unparse(kw.value) for kw in llm_call.keywords}
    if "max_model_len" in llm_kwargs:
        raise ValueError("Evaluator now overrides context; inspect the changed runtime contract")
    result = []
    hash_assets = {}
    for uid, (prompt, _) in expected.items():
        bid = uid.removeprefix("harmbench:")
        if bid not in behaviors:
            raise ValueError("Planned prompt missing from pinned official behaviors")
        behavior = behaviors[bid]
        route = harmbench_route(behavior)
        context = (behavior.get("ContextString") or "").strip()
        text = behavior["Behavior"].strip()
        if prompt != (f"{context}\n\n---\n\n{text}" if context else text):
            raise ValueError("Saved generation request differs from the full official behavior/context")
        record = {"uid": uid, "route": route, "functional_category": behavior["FunctionalCategory"],
            "semantic_category": behavior["SemanticCategory"], "prompt_sha256": digest(prompt),
            "behavior_sha256": digest(behavior["Behavior"]),
            "context_sha256": digest(behavior.get("ContextString") or ""), "status": "missing_response"}
        if uid not in rows:
            result.append(record)
            continue
        raw = rows[uid]["output"]
        tokenizer.truncation_side = "right"
        raw_ids = tokenizer.encode(raw, truncation=False)
        clipped_ids = tokenizer.encode(raw, max_length=512, truncation=True)
        clipped = tokenizer.decode(clipped_ids, skip_special_tokens=True)
        if len(clipped_ids) > 512 or clipped_ids != raw_ids[:512]:
            raise ValueError("Official right512 token clipping does not match exact token prefix")
        record.update(status="tokenization_preflight_only", raw_response_sha256=digest(raw),
            raw_classifier_token_count=len(raw_ids), clipped_generation_token_count=len(clipped_ids),
            response_clipped_at512=len(raw_ids) > 512, clipped_generation_sha256=digest(clipped),
            source_finish_reason=rows[uid].get("finish_reason"), raw_response_empty=not raw.strip())
        if route == "hash_check":
            path = repo / "data/copyright_classifier_hashes" / f"{bid}.pkl"
            relative = str(path.relative_to(repo))
            actual = file_hash(path)
            if actual != setup["tracked_helper_and_data_sha256"][relative]:
                raise ValueError("Copyright reference asset changed or is unavailable")
            tags = behavior["Tags"].split(", ")
            subtype = "book" if "book" in tags else "lyrics"
            record.update(hash_subtype=subtype, hash_reference_sha256=actual,
                classifier_prompt_tokens=None, classification_not_executed=True)
            hash_assets[relative] = actual
        else:
            contextual = route == "contextual_classifier"
            if contextual and not context:
                raise ValueError("Contextual classifier route has missing context")
            # Exactly the pinned helper's raw field substitution (no .strip()).
            template = templates["prompt_contextual" if contextual else "prompt"]
            formatted = template.format(behavior=behavior["Behavior"], generation=clipped,
                                        context=behavior.get("ContextString") or "")
            if "[BEHAVIOR]:\n" + behavior["Behavior"] + "\n" not in formatted:
                raise ValueError("Behavior was altered during classifier formatting")
            if contextual and "[CONTEXT]:\n" + behavior["ContextString"] + "\n" not in formatted:
                raise ValueError("Context was altered during classifier formatting")
            tokenizer.truncation_side = "left"  # evaluator sets this on classifier tokenizer
            ids = tokenizer.encode(formatted, truncation=False)
            record.update(classifier_prompt_tokens=len(ids), classifier_prompt_sha256=digest(formatted),
                classifier_token_ids_sha256=object_hash(ids), behavior_context_truncated=False,
                fits_declared_context_with_output1=len(ids)+1 <= declared_context,
                fits_requested4096_with_output1=len(ids)+1 <= 4096,
                classification_not_executed=True)
        result.append(record)
    classifier_rows = [r for r in result if r.get("classifier_prompt_tokens") is not None]
    overflow = [r["uid"] for r in classifier_rows if not r["fits_declared_context_with_output1"]]
    missing = [r["uid"] for r in result if r["status"] == "missing_response"]
    if torch.cuda.is_initialized():
        raise ValueError("CPU preflight unexpectedly initialized CUDA")
    report = {"status": "cpu_preflight_passed" if not overflow and not missing else "cpu_preflight_has_blockers",
        "scope": "Input/tokenizer/config/route preflight only; no classifier, hash-match evaluation, score, or GPU/model loading",
        "label": label, "planned": len(expected), "saved_responses": len(rows), "missing_response_uids": missing,
        "route_counts": dict(Counter(r["route"] for r in result)),
        "functional_category_counts": dict(Counter(r["functional_category"] for r in result)),
        "semantic_category_counts": dict(Counter(r["semantic_category"] for r in result)),
        "hash_subtype_counts": dict(Counter(r["hash_subtype"] for r in result if "hash_subtype" in r)),
        "response_clipped_at512": sum(r.get("response_clipped_at512", False) for r in result),
        "empty_response_count": sum(r.get("raw_response_empty", False) for r in result),
        "classifier_input_max_tokens": max(r["classifier_prompt_tokens"] for r in classifier_rows),
        "max_length_input_uid": max(classifier_rows, key=lambda r:r["classifier_prompt_tokens"])["uid"],
        "classifier_context_overflow_uids": overflow, "behavior_context_truncation_count": 0,
        "model": {"snapshot": str(model), "revision": MODEL_REVISION,
            "config_sha256": file_hash(model / "config.json"),
            "architecture": config.architectures, "hidden_size": config.hidden_size,
            "layers": config.num_hidden_layers, "vocab_size": config.vocab_size,
            "declared_context": declared_context, "rope_scaling": config.rope_scaling,
            "dtype_config": str(config.torch_dtype), "requested_context_in_task": 4096,
            "context_resolution_note": "Current evaluator provides no override. Actual pinned model config declares2048, not4096; current base inputs fit2048 unchanged."},
        "parameter_compatibility": {"evaluator_llm_keyword_source": llm_kwargs,
            "sampling_params_signature": str(inspect.signature(SamplingParams)),
            "observed_temperature": params.temperature, "observed_max_tokens": params.max_tokens,
            "observed_truncate_prompt_tokens": params.truncate_prompt_tokens,
            "model_runtime_compatibility_unverified_without_inference": True},
        "tokenizer": {"class": type(tokenizer).__name__, "vocab_size": len(tokenizer),
            "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
            "response_truncation_side": "right", "response_max_tokens_including_specials": 512,
            "classifier_prompt_truncation": False,
            "files_sha256": {p.name: file_hash(p) for p in model.iterdir() if p.name in (
                "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json")}},
        "official_method": {"repo_commit": HARMBENCH_COMMIT, "helper_sha256": file_hash(helper_path),
            "behavior_csv_sha256": file_hash(behavior_path), "helper_function_source_sha256": helper_sources,
            "template_sha256": {key:digest(templates[key]) for key in ("prompt", "prompt_contextual")},
            "hash_assets_checked": len(hash_assets), "hash_assets_manifest_sha256": object_hash(hash_assets),
            "hash_window_overlap": {"book": [300,200], "lyrics": [50,40]}, "hash_match_threshold": .6,
            "method_pointer": f"https://github.com/centerforaisafety/HarmBench/blob/{HARMBENCH_COMMIT}/evaluate_completions.py",
            "helper_pointer": f"https://github.com/centerforaisafety/HarmBench/blob/{HARMBENCH_COMMIT}/eval_utils.py"},
        "source": provenance, "runtime_validation_sha256": file_hash(setup_path),
        "evaluator_sha256": file_hash(evaluator_path), "preflight_source_sha256": file_hash(__file__),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "vllm", "tokenizers")},
        "cuda_initialized": False, "model_instantiated": False, "hash_classifier_executed": False,
        "classifier_executed": False, "harmfulness_score": None,
        "limitations": "This validates only the currently saved base320 inputs. It does not prove classifier accuracy, runtime model loading, future trained-response context fit, or model harmfulness/refusal rates."}
    return report, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label", default="base")
    parser.add_argument("--model", type=Path, default=Path(
        "/work/hf_cache/hub/models--cais--HarmBench-Llama-2-13b-cls/snapshots/" + MODEL_REVISION))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((args.root / "probes").resolve()):
        raise ValueError("Preflight output must remain in taskroot/probes")
    if args.output.exists() or args.output.with_suffix(".per_item.jsonl").exists():
        raise FileExistsError("Preserve earlier preflight artifacts; use fresh paths")
    report, rows = preflight(args.root, args.model, args.label)
    item_path = args.output.with_suffix(".per_item.jsonl")
    write_jsonl(item_path, rows)
    report["per_item_artifact"] = {"path": str(item_path), "sha256": file_hash(item_path), "n_rows": len(rows)}
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "planned", "saved_responses", "route_counts",
        "functional_category_counts", "semantic_category_counts", "hash_subtype_counts", "response_clipped_at512",
        "classifier_input_max_tokens", "classifier_context_overflow_uids", "harmfulness_score")}, indent=2))


if __name__ == "__main__":
    main()
