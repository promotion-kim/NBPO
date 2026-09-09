"""Read-only replay of legacy pair tokenization on the two preserved datasets.

No model weights are loaded and no dataset mapping/cache is written. The only
output is the new campaign provenance report; old checkpoints, datasets, source,
pair targets, and reference caches remain untouched.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from scripts.experiments.nbpo_repair_20260909.common import digest, file_hash, object_hash, write_json

OLD_ROOT = Path("/work/iclr27_table1_v2")
RUN_CONFIGS = {
    "canonN700": OLD_ROOT / "smoke/canon/arms/canonN700_run_config.yaml",
    "RB1200": OLD_ROOT / "smoke/train3/arms/DIAGrb_steps1200_run_config.yaml",
}


def load_legacy_helper(path):
    """Import the exact preserved file without importing the old trainer stack."""
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("nbpo_preserved_pair_tokenization_audit", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CachedTokenizer:
    """Memoize identical text encodings; the old helper copies their token lists."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.encode = lru_cache(maxsize=4096)(self._encode)

    def _encode(self, text, add_special_tokens):
        return self.tokenizer(text, add_special_tokens=add_special_tokens)

    def __call__(self, text, add_special_tokens=False):
        return self.encode(text, add_special_tokens)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


def audit_rows(rows, tokenizer, helper, tokenization, *, progress_label=None):
    """Measure event/context variability for identical prompt/candidate text."""
    tokenizer = CachedTokenizer(tokenizer)
    candidates = {}
    prompt_id_texts, candidate_id_texts = defaultdict(set), defaultdict(set)
    identity_modes, target_modes, evidence = Counter(), Counter(), Counter()
    errors, prompt_ids = [], set()
    n_rows = 0
    for row_index, row in enumerate(rows):
        n_rows += 1
        target_modes[str(row.get("target_mode", "unavailable"))] += 1
        prompt = row["prompt"]
        if not all(isinstance(row.get(key), str) for key in ("prompt", "chosen", "rejected")):
            errors.append({"row_index": row_index, "error": "saved prompt/response fields are not strings"})
            continue
        prompt_sha = digest(prompt)
        prompt_id = str(row.get("prompt_id") or row.get("uid") or "content:"+prompt_sha)
        prompt_ids.add(prompt_id)
        prompt_id_texts[prompt_id].add(prompt_sha)
        try:
            event = helper.tokenize_preference_pair(tokenizer, prompt, row["chosen"], row["rejected"], **tokenization)
            raw = {side: helper.build_tokenized_answer(tokenizer, prompt, row[side]) for side in ("chosen", "rejected")}
            common_split = min(raw[side]["split_idx"] for side in raw)
        except Exception as error:
            errors.append({"row_index": row_index, "prompt_id": prompt_id,
                           "error": f"{type(error).__name__}: {error}"})
            continue
        for side, partner_side in (("chosen", "rejected"), ("rejected", "chosen")):
            response_sha = digest(row[side])
            response_id = row.get(side+"_response_id") or row.get(side+"_candidate_id")
            mode = "explicit_candidate_id" if response_id is not None else "response_content_hash"
            identity_modes[mode] += 1
            candidate_id = str(response_id if response_id is not None else "content:"+response_sha)
            partner_id = str(row.get(partner_side+"_response_id") or row.get(partner_side+"_candidate_id") or
                             "content:"+digest(row[partner_side]))
            candidate_id_texts[(prompt_id, candidate_id)].add(response_sha)
            # Retain exact text as part of the key. Reused IDs with changed text
            # are identity conflicts, not evidence of partner-dependent tokenization.
            key = (prompt_id, prompt_sha, candidate_id, response_sha)
            block = candidates.setdefault(key, {"variants": {}, "contexts": set(), "responses": set(), "masks": set()})
            tokens = {name: event[side+"_"+name] for name in ("input_ids", "attention_mask", "labels")}
            event_sha = object_hash(tokens)
            pad = tokenization["label_pad_token_id"]
            boundary = next((i for i, label in enumerate(tokens["labels"]) if label != pad), len(tokens["labels"]))
            context_sha = object_hash({"input_ids": tokens["input_ids"][:boundary],
                                       "attention_mask": tokens["attention_mask"][:boundary]})
            response_event_sha = object_hash({name: values[boundary:] for name, values in tokens.items()})
            mask_sha = object_hash({"attention_mask": tokens["attention_mask"],
                                    "response_label_mask": [value != pad for value in tokens["labels"]]})
            block["contexts"].add(context_sha)
            block["responses"].add(response_event_sha)
            block["masks"].add(mask_sha)
            variant = block["variants"].setdefault(event_sha, {"count": 0, "rows": [], "partners": set(),
                "example": {"row_index": row_index, "side": side, "partner_response_id": partner_id,
                            "context_sha256": context_sha, "response_event_sha256": response_event_sha,
                            "mask_sha256": mask_sha, "prompt_tokens": boundary,
                            "response_tokens": len(tokens["input_ids"])-boundary}, "cached_logps": {}})
            variant["count"] += 1
            variant["rows"].append(row_index)
            variant["partners"].add(partner_id)
            for column in ("reference", "history0"):
                value = row.get(f"{column}_{side}_logps")
                if value is not None and math.isfinite(float(value)):
                    values = variant["cached_logps"].setdefault(column, [])
                    values.append(float(value))
            before = helper.split_at(raw[side], common_split)
            response_before = before["input_ids"]
            eos = tokenizer.eos_token_id
            appended = eos is not None and (not response_before or response_before[-1] != eos)
            after_length = len(response_before) + int(appended)
            final_response = tokens["input_ids"][boundary:]
            bos_added = tokenizer.bos_token_id is not None and (
                not before["prompt_input_ids"] or before["prompt_input_ids"][0] != tokenizer.bos_token_id)
            evidence["side_occurrences"] += 1
            evidence["helper_appends_eos"] += int(appended)
            evidence["helper_added_eos_survives"] += int(appended and len(final_response) == after_length)
            evidence["response_truncated_by_training_tokenizer"] += int(len(final_response) < after_length)
            evidence["prompt_truncated_by_training_tokenizer"] += int(boundary < common_split+int(bos_added))
            evidence["partner_forces_boundary_backoff"] += int(common_split < raw[side]["split_idx"])
            evidence["training_sequence_at_max_length"] += int(len(tokens["input_ids"]) == tokenization["max_length"])
            evidence["raw_sampled_token_ids_available"] += int(any(side+"_"+name in row for name in
                ("response_token_ids", "sampled_token_ids", "raw_token_ids")))
            capped = row.get(side+"_capped_horizon")
            finish = row.get(side+"_finish_reason")
            evidence["generation_capped_status_available"] += int(capped is not None or finish is not None)
            evidence["generation_capped_explicit_true"] += int(capped is True or finish == "length")
        if progress_label and n_rows % 5000 == 0:
            print(json.dumps({"run_split": progress_label, "rows": n_rows, "unique_candidates": len(candidates)}), flush=True)
    variable = {key: block for key, block in candidates.items() if len(block["variants"]) > 1}
    affected_rows, nonmodal_rows, affected_prompts = set(), set(), set()
    nonmodal_occurrences = 0
    examples = []
    for key, block in variable.items():
        variants = sorted(block["variants"].items(), key=lambda pair: (-pair[1]["count"], pair[0]))
        affected_prompts.add(key[0])
        for index, (_, variant) in enumerate(variants):
            affected_rows.update(variant["rows"])
            if index:
                nonmodal_rows.update(variant["rows"])
                nonmodal_occurrences += variant["count"]
        if len(examples) < 30:
            examples.append({"prompt_id": key[0], "prompt_text_sha256": key[1],
                "candidate_id": key[2], "response_text_sha256": key[3],
                "variants": [{"event_sha256": sha, "occurrences": variant["count"],
                              "distinct_partners": len(variant["partners"]), **variant["example"]}
                             for sha, variant in variants]})
    cached_ranges = {}
    for column in ("reference", "history0"):
        ranges = [max(values)-min(values) for block in candidates.values() if len(block["variants"]) == 1
                  for variant in block["variants"].values()
                  if len(values := variant["cached_logps"].get(column, [])) > 1]
        cached_ranges[column] = {"stable_event_candidates_repeated": len(ranges),
            "range_gt_1e_4": sum(value > 1e-4 for value in ranges),
            "max_logp_range": max(ranges, default=None),
            "median_logp_range": statistics.median(ranges) if ranges else None,
            "interpretation": "Stored repeated-candidate logp variation conditional on identical replayed tokens. This alone does not identify the numerical cause or actual initialization h."}
    result = {"rows": n_rows, "unique_prompt_groups": len(prompt_ids), "unique_candidate_text_events": len(candidates),
        "rows_per_prompt": n_rows/len(prompt_ids) if prompt_ids else None,
        "candidate_identity_mode_side_counts": dict(identity_modes), "target_mode_counts": dict(target_modes),
        "prompt_ids_with_multiple_texts": sum(len(values)>1 for values in prompt_id_texts.values()),
        "candidate_ids_with_multiple_response_texts": sum(len(values)>1 for values in candidate_id_texts.values()),
        "variable_event_candidates": len(variable),
        "variable_prompt_context_candidates": sum(len(block["contexts"])>1 for block in candidates.values()),
        "variable_response_event_candidates": sum(len(block["responses"])>1 for block in candidates.values()),
        "variable_label_or_attention_mask_candidates": sum(len(block["masks"])>1 for block in candidates.values()),
        "affected_prompt_groups": len(affected_prompts), "rows_touching_variable_candidates": len(affected_rows),
        "affected_row_fraction": len(affected_rows)/n_rows if n_rows else None,
        "nonmodal_side_occurrences": nonmodal_occurrences, "rows_with_nonmodal_events": len(nonmodal_rows),
        "errors": errors[:30], "error_count": len(errors), "eos_and_truncation_evidence": dict(evidence),
        "cached_logp_repeat_diagnostics": cached_ranges, "affected_candidate_examples": examples,
        "sampled_terminal_event_limitations": "Saved text has already been chat-templated. EOS/EOT visible here may have been appended by that template. Without original sampled token IDs and stop reasons, helper EOS additions and training truncation cannot establish an original generation cap/early EOS event.",
        "replay_event_digest": object_hash({"|".join(key): sorted((sha, variant["count"]) for sha, variant in block["variants"].items())
                                             for key, block in candidates.items()})}
    result["replay_status"] = "unresolved_errors" if errors else "affected_rows_present" if variable else "inactive_on_replayed_rows"
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("/work/nbpo_repair_20260909"))
    args = ap.parse_args()
    if args.root.resolve() != Path("/work/nbpo_repair_20260909"):
        raise ValueError("This audit writes only to the explicitly authorized repair provenance directory")
    output = args.root / "provenance/legacy_tokenization_audit_v1.json"
    if output.exists():
        raise FileExistsError(output)
    import yaml
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    from mnpo_scripts.precompute_provenance import tokenizer_content_hashes, verify_precompute_manifest
    helper_path = OLD_ROOT / "code_reffix/mnpo_scripts/pair_tokenization.py"
    helper = load_legacy_helper(helper_path)
    report = {"source_sha256": file_hash(__file__), "legacy_helper_path": str(helper_path),
              "legacy_helper_sha256": file_hash(helper_path), "gpu_model_forwards": 0,
              "method": "Replay every saved pair with exact old helper; compare full input/attention/label event hashes for the same prompt and candidate text across its partners. All hashes are of unpadded events.",
              "scope_limitation": "Code-level reproduction and exact saved-dataset lineage are measured. Replaying declared inputs does not independently recover an unavailable historical runtime binary or raw sampled event.",
              "runs": {}}
    started = time.monotonic()
    for name, config_path in RUN_CONFIGS.items():
        config = yaml.safe_load(config_path.read_text())
        dataset_paths = list(config["dataset_mixer"])
        if len(dataset_paths) != 1:
            raise ValueError("Legacy audit expects one explicitly named preserved dataset")
        dataset_path = Path(dataset_paths[0])
        expected_path = OLD_ROOT / ("smoke/canon/precomputed" if name == "canonN700" else "smoke/train3/precomputed")
        if dataset_path.resolve() != expected_path:
            raise ValueError("Run config points outside the authorized preserved dataset")
        meta_path = dataset_path / "precompute_meta.json"
        meta = json.loads(meta_path.read_text())
        manifest_check = verify_precompute_manifest(str(dataset_path),
            expected_manifest_sha256=config["nbpo_expected_precompute_manifest_sha256"])
        tokenizer_path = Path(meta["tokenizer_source"])
        if tokenizer_path != Path("/work/models/bases/Llama-3.1-8B-Instruct"):
            raise ValueError("Unexpected historical tokenizer source")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        hashes = tokenizer_content_hashes(tokenizer)
        tokenization = {key: meta[key] for key in ("max_length", "max_prompt_length", "truncation_mode", "label_pad_token_id")}
        config_matches = all(config.get(key, value) == value for key, value in tokenization.items())
        tok_cfg = helper.tokenization_config(**tokenization)
        tok_cfg_hash = helper.tokenization_config_hash(tok_cfg)
        provenance_matches = (all(hashes[key] == meta[key] for key in hashes)
            and config_matches and tok_cfg_hash == meta["tokenization_config_sha256"]
            and tok_cfg_hash == config["nbpo_expected_tokenization_config_sha256"])
        run = {"config_path": str(config_path), "config_sha256": file_hash(config_path),
            "dataset_path": str(dataset_path), "precompute_meta_sha256": file_hash(meta_path),
            "precompute_manifest_sha256": file_hash(dataset_path / "precompute_manifest.json"),
            "dataset_manifest_verified": bool(manifest_check), "tokenization": tokenization,
            "tokenizer_content_hashes": hashes, "tokenization_provenance_matches": provenance_matches,
            "declared_training_splits": config["dataset_splits"],
            "run_settings": {key: config.get(key) for key in ("run_name", "optim", "learning_rate", "max_steps",
                "per_device_train_batch_size", "gradient_accumulation_steps", "eta", "loss_type", "nbpo_target_column")},
            "target_solver_sha256": meta.get("solver_artifact_sha256"),
            "pair_artifact_sha256": meta.get("pair_artifact_sha256"), "splits": {}}
        dataset = load_from_disk(str(dataset_path))
        for split, rows in dataset.items():
            run["splits"][split] = audit_rows(rows, tokenizer, helper, tokenization, progress_label=name+"/"+split)
            run["splits"][split]["selected_by_training_config"] = split in config["dataset_splits"]
            run["splits"][split]["saved_columns"] = rows.column_names
        report["runs"][name] = run
        print(json.dumps({"run": name, "provenance_matches": provenance_matches,
            "splits": {split: {key: value[key] for key in ("rows", "unique_prompt_groups", "variable_event_candidates", "affected_row_fraction", "replay_status")}
                       for split, value in run["splits"].items()}}), flush=True)
    report["seconds"] = time.monotonic()-started
    write_json(output, report)
    print(json.dumps({"report": str(output), "sha256": file_hash(output), "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
