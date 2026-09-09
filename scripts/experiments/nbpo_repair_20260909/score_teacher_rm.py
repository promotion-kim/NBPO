"""Independent scalar-RM diagnostic of the frozen train2000 finite-pool teacher.

Score mode is GPU work only when explicitly invoked. Aggregate mode is CPU-only.
No model/teacher fitting, calibration, outcome-based selection, or safety claim.
Completed chunks resume only under identical source, model, and input hashes.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.experiments.nbpo_repair_20260909.common import (
    digest, file_hash, object_hash, read_jsonl, write_json, write_jsonl,
)
from scripts.experiments.nbpo_repair_20260909.evaluate_responses import (
    BOOTSTRAP_SEED, KEYWORD_REFUSAL, SKYWORK_CARD, bootstrap_ratio, skywork_inputs,
)

METRIC_SCOPE = ("Independent frozen Skywork scalar overall-preference proxy on the original train candidate pool; "
                "not separate helpfulness/harmlessness, not judge-calibrated, not a causal effect, "
                "not held-out generalization or a generated neural-policy acceptance certificate")


def atomic_json(path, data):
    """Publish an immutable complete chunk; interruption leaves no partial target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".teacher-rm-", dir=path.parent) as temporary:
        staged = Path(temporary) / "complete.json"
        write_json(staged, data)
        os.link(staged, path)  # Atomic, same filesystem, and refuses replacement.


def assigned_prompt_indices(n_prompts, shard, num_shards):
    if n_prompts < 0 or num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError("Invalid deterministic prompt shard")
    return list(range(shard, n_prompts, num_shards))


def tied_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Ranks require a finite vector")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.
        start = stop
    return ranks


def rank_correlation(a, b):
    a, b = tied_ranks(a), tied_ranks(b)
    if a.shape != b.shape:
        raise ValueError("Rank vectors must be aligned")
    a, b = a-a.mean(), b-b.mean()
    den = float(np.linalg.norm(a)*np.linalg.norm(b))
    return None if den == 0 else float(np.clip(np.dot(a, b)/den, -1., 1.))


def prompt_statistics(p_star, rewards, lengths, refusals, capped):
    p, r, lengths = (np.asarray(v, dtype=np.float64) for v in (p_star, rewards, lengths))
    refusals, capped = (np.asarray(v, dtype=np.float64) for v in (refusals, capped))
    if any(v.shape != (8,) for v in (p, r, lengths, refusals, capped)):
        raise ValueError("Every prompt must retain all eight candidate occurrences")
    if not all(np.isfinite(v).all() for v in (p, r, lengths, refusals, capped)):
        raise ValueError("Do not silently renormalize incomplete/nonfinite prompt scores")
    if np.any(p <= 0) or abs(p.sum()-1.) > 1e-10 or np.any(lengths < 0):
        raise ValueError("Invalid teacher probability/length")
    if any(np.any((v != 0) & (v != 1)) for v in (refusals, capped)):
        raise ValueError("Diagnostic flags must be boolean")
    rm_top, teacher_top = r == r.max(), p == p.max()
    weighted, uniform = float(p@r), float(r.mean())
    return {"rm_teacher_weighted": weighted, "rm_uniform": uniform,
        "rm_paired_delta_teacher_minus_uniform": weighted-uniform,
        "spearman_p_star_vs_rm": rank_correlation(p, r),
        "rm_highest_score": float(r.max()),
        "teacher_mass_on_rm_highest_tie_set": float(p@rm_top),
        "uniform_mass_on_rm_highest_tie_set": float(rm_top.mean()),
        "rm_mean_on_teacher_max_mass_tie_set": float(r[teacher_top].mean()),
        "rm_best_minus_teacher_max_mass_score": float(r.max()-r[teacher_top].mean()),
        "teacher_max_mass_tie_set_fraction_also_rm_highest": float(rm_top[teacher_top].mean()),
        "teacher_max_mass_intersects_rm_highest": bool(np.any(rm_top & teacher_top)),
        "rm_highest_tie_count": int(rm_top.sum()), "teacher_max_mass_tie_count": int(teacher_top.sum()),
        "length_teacher_weighted": float(p@lengths), "length_uniform": float(lengths.mean()),
        "length_paired_delta_teacher_minus_uniform": float(p@lengths-lengths.mean()),
        "spearman_rm_vs_length": rank_correlation(r, lengths),
        "refusal_keyword_teacher_mass": float(p@refusals), "refusal_keyword_uniform_mass": float(refusals.mean()),
        "refusal_keyword_paired_mass_delta": float(p@refusals-refusals.mean()),
        "capped_generation_teacher_mass": float(p@capped), "capped_generation_uniform_mass": float(capped.mean())}


def verify_event(row, prompt, index):
    pid = prompt["prompt_id"]
    if (row["role"] != "learner" or row["sample_index"] != index
            or row["candidate_id"] != f"{pid}:learner:{index}" or row["split"] != "train"):
        raise ValueError("Wrong candidate occurrence identity")
    if row["prompt"] != prompt["prompt"] or row["prompt_token_ids"] != prompt["prompt_token_ids"]:
        raise ValueError("Changed full prompt/context event")
    if digest(row["response"]) != row["response_sha256"]:
        raise ValueError("Changed response text")
    if object_hash({key: row[key] for key in ("input_ids", "attention_mask", "labels")}) != row["token_event_sha256"]:
        raise ValueError("Changed original sampled token event")
    expected_seed = int(digest("20260909-repair-pool:"+row["candidate_id"])[:16], 16) % (2**63-1)
    if row["seed"] != expected_seed or row["n_tokens"] != len(row["response_token_ids"]):
        raise ValueError("Changed candidate RNG or length metadata")
    if row["input_ids"] != prompt["prompt_token_ids"]+row["response_token_ids"]:
        raise ValueError("Sampled response/context concatenation mismatch")


def load_inputs(root, teacher_name, pool_shards=4):
    """Bind teacher weights to split order, all original pool occurrences, and GPM inputs."""
    from scripts.nbpo.build_nbpo_pairs import load_canonical_artifact
    train_path, split_path = root / "splits/train.jsonl", root / "splits/manifest.json"
    prompts = read_jsonl(train_path)
    split_manifest = json.loads(split_path.read_text())
    if (len(prompts) != 2000 or split_manifest["counts"]["train"] != 2000
            or file_hash(train_path) != split_manifest["splits"]["train"]["file_sha256"]):
        raise ValueError("This diagnostic requires the original train2000 split")
    ids = [p["prompt_id"] for p in prompts]
    if len(set(ids)) != len(ids) or any(p["split"] != "train" for p in prompts):
        raise ValueError("Duplicate/wrong-split train prompts")
    by_id = dict(zip(ids, prompts))
    teacher = root / "teachers" / teacher_name
    inputs_path, solution_path = teacher / "inputs.json", teacher / "train/solver/solution.json"
    inputs, solution = (json.loads(p.read_text()) for p in (inputs_path, solution_path))
    if (solution["split"] != "train" or solution["eta"] != 1.
            or inputs["split_manifest_sha256"] != file_hash(split_path)
            or object_hash(inputs["score_manifests"]) != solution["teacher_manifest_sha256"]):
        raise ValueError("Wrong frozen teacher split/provenance")
    if (solution["config"]["beta"] != [.25, .25] or solution["config"]["fixed_weights"] is not False
            or solution["config"]["dual_fit_scope"] != "training"
            or solution["config"]["representation"]["reference_construction"] != "independent_samples"):
        raise ValueError("Wrong primary adaptive-game train-only teacher definition")
    for filename, expected_hash in solution["input_hashes"].items():
        if file_hash(teacher / "train/tensor" / filename) != expected_hash:
            raise ValueError("Changed teacher input tensor/meta")
    canonical = load_canonical_artifact(teacher / "train/solver", solution,
        expected_prompt_ids=ids, expected_representation="adaptive_game", expected_aggregation="nash")
    if canonical["p_star"].shape != (2000, 8) or not np.allclose(canonical["p_t"], .125, atol=1e-12, rtol=0):
        raise ValueError("Teacher must retain uniform IID-eight center")
    events, all_occurrences, source_hashes = {}, {}, []
    if len(inputs["score_manifests"]) != pool_shards:
        raise ValueError("Wrong original score-shard count")
    for shard, original in enumerate(inputs["score_manifests"]):
        manifest_path = root / "scores" / f"shard{shard}" / "manifest.json"
        if original["shard"] != shard or file_hash(manifest_path) != original["sha256"]:
            raise ValueError("Changed frozen GPM score manifest")
        manifest = json.loads(manifest_path.read_text())
        used_paths = {}
        for raw_path, expected_hash in manifest["pool_files"].items():
            path = root / "pools" / f"shard{shard}" / Path(raw_path).name
            if file_hash(path) != expected_hash:
                raise ValueError("Changed original pool file")
            used_paths[str(path)] = expected_hash
            for row in read_jsonl(path):
                pid = row["prompt_id"]
                if pid not in by_id:
                    continue
                cid = row["candidate_id"]
                if cid in all_occurrences:
                    raise ValueError("Duplicate original train occurrence")
                all_occurrences[cid] = object_hash({key: row[key] for key in
                    ("token_event_sha256", "response_sha256", "seed", "finish_reason")})
                if row["role"] == "learner":
                    index = row["sample_index"]
                    if index not in range(8):
                        raise ValueError("Invalid learner occurrence index")
                    verify_event(row, by_id[pid], index)
                    events[(pid, index)] = row
        source_hashes.append({"shard": shard, "manifest_sha256": file_hash(manifest_path), "pool_files": used_paths})
    required = {f"{pid}:{role}:{i}" for pid in ids for role in ("learner", "reference_learner", "comparator") for i in range(8)}
    pool_hash = object_hash(all_occurrences)
    if (set(all_occurrences) != required or len(events) != 16000
            or pool_hash != solution["pool_sha256"] or pool_hash != inputs["pool_hashes"]["train"]):
        raise ValueError("Teacher and original train pool occurrence bindings differ")
    identity = {"train_split_sha256": file_hash(train_path), "split_manifest_sha256": file_hash(split_path),
        "prompt_order_sha256": object_hash(ids), "teacher_inputs_sha256": file_hash(inputs_path),
        "teacher_solution_sha256": file_hash(solution_path), "teacher_pi_star_sha256": solution["artifact_hashes"]["pi_star.npz"],
        "teacher_target_sha256": solution["artifact_hashes"]["target_log_ratio.npz"],
        "pool_sha256": pool_hash, "source_pool_artifacts": source_hashes}
    return prompts, events, canonical["p_star"], identity


def rm_identity(rm):
    weights = sorted(rm.glob("*.safetensors"))
    if not weights:
        raise ValueError("Pinned local Skywork safetensors snapshot is required")
    files = weights + [p for p in sorted(rm.iterdir()) if p.is_file() and p.suffix in (".json", ".jinja", ".model", ".txt")]
    if not (rm / "tokenizer.json").is_file() or not (rm / "config.json").is_file():
        raise ValueError("Pinned RM tokenizer/config required")
    return {"path": str(rm.resolve()), "model_card": SKYWORK_CARD,
            "files_sha256": {p.name: file_hash(p) for p in files}}


def candidate_binding(event, mass):
    return {"prompt_id": event["prompt_id"], "candidate_id": event["candidate_id"],
        "sample_index": event["sample_index"], "response_sha256": event["response_sha256"],
        "prompt_sha256": digest(event["prompt"]), "token_event_sha256": event["token_event_sha256"],
        "p_star": float(mass), "response_tokens": event["n_tokens"],
        "refusal_keyword_diagnostic": bool(KEYWORD_REFUSAL.search(event["response"])),
        "capped_generation": event["finish_reason"] == "length"}


def verify_chunk(chunk, protocol_hash, expected):
    if chunk.get("protocol_sha256") != protocol_hash or object_hash(chunk["records"]) != chunk["records_sha256"]:
        raise ValueError("Changed chunk protocol/record hash")
    records = chunk["records"]
    if len(records) != len(expected):
        raise ValueError("Chunk does not cover its complete assigned candidate list")
    for record, binding in zip(records, expected):
        if any(record.get(key) != value for key, value in binding.items()):
            raise ValueError("Chunk candidate identity/order/teacher mass differs")
        score = record.get("rm_score")
        if record.get("status") == "ok":
            if (score is None or not math.isfinite(score) or not 0 < record.get("n_rm_tokens", 0) <= 16384
                    or len(record.get("rm_input_sha256", "")) != 64):
                raise ValueError("Invalid successful cached RM score")
        elif record.get("status") != "score_failure" or score is not None or not record.get("failure_reason"):
            raise ValueError("Failed RM candidates must not carry usable scores")
    return records


def planned_chunks(prompts, events, masses, shard, num_shards, chunk_prompts):
    indices = assigned_prompt_indices(len(prompts), shard, num_shards)
    for start in range(0, len(indices), chunk_prompts):
        group = indices[start:start+chunk_prompts]
        candidates = [events[(prompts[x]["prompt_id"], i)] for x in group for i in range(8)]
        expected = [candidate_binding(events[(prompts[x]["prompt_id"], i)], masses[x, i]) for x in group for i in range(8)]
        yield start//chunk_prompts, candidates, expected


def score_shard(args, prompts, events, masses, protocol):
    protocol_hash = object_hash(protocol)
    shard_dir = args.out / f"shard{args.shard}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    settings_path = shard_dir / "settings.json"
    settings = {"protocol": protocol, "protocol_sha256": protocol_hash, "shard": args.shard}
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise ValueError("Resume requires identical frozen inputs/RM/source/protocol")
    else:
        atomic_json(settings_path, settings)
    pending, chunk_hashes = [], {}
    for index, candidates, expected in planned_chunks(prompts, events, masses, args.shard, args.num_shards, args.chunk_prompts):
        path = shard_dir / f"chunk{index:05d}.json"
        if path.exists():
            verify_chunk(json.loads(path.read_text()), protocol_hash, expected)
            chunk_hashes[path.name] = file_hash(path)
        else:
            pending.append((path, candidates, expected))
    if pending:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.rm, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.rm, local_files_only=True,
            torch_dtype=torch.bfloat16, num_labels=1).to(args.device).eval()
        model.config.use_cache = False
        runtime = {"package_versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
                   "device": args.device, "dtype": "bfloat16"}
        if runtime["package_versions"] != protocol["runtime_package_versions"]:
            raise ValueError("Runtime package versions differ from frozen chunk protocol")
        for path, candidates, expected in pending:
            records = []
            for event, binding in zip(candidates, expected):
                record = {**binding, "rm_score": None, "status": "score_failure"}
                try:
                    ids, removed_bos = skywork_inputs(tok, event["prompt"], event["response"])
                    n_tokens = int(ids["input_ids"].shape[1])
                    record.update(n_rm_tokens=n_tokens, removed_template_leading_bos=removed_bos,
                                  rm_input_sha256=object_hash({key: value.tolist() for key, value in ids.items()}))
                    if n_tokens > 16384:
                        record["failure_reason"] = "over_16384_tokens_no_truncation"
                    else:
                        with torch.no_grad():
                            score = float(model(**{key: value.to(args.device) for key, value in ids.items()}).logits[0, 0])
                        if math.isfinite(score):
                            record.update(rm_score=score, status="ok")
                        else:
                            record["failure_reason"] = "nonfinite_score"
                except RuntimeError as error:
                    record["failure_reason"] = type(error).__name__
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                records.append(record)
            chunk = {"protocol_sha256": protocol_hash, "runtime": runtime, "records": records,
                     "records_sha256": object_hash(records)}
            verify_chunk(chunk, protocol_hash, expected)
            atomic_json(path, chunk)
            chunk_hashes[path.name] = file_hash(path)
            print(json.dumps({"chunk": str(path), "n_candidates": len(records),
                "n_failed": sum(row["status"] != "ok" for row in records)}), flush=True)
    manifest_path = shard_dir / "manifest.json"
    manifest = {"settings_sha256": file_hash(settings_path), "protocol_sha256": protocol_hash,
                "chunks_sha256": chunk_hashes, "shard": args.shard}
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing shard manifest differs from resumed chunks")
    else:
        atomic_json(manifest_path, manifest)


def summarize(records, planned_ids, repetitions):
    grouped = {}
    for record in records:
        pid, index = record["prompt_id"], record["sample_index"]
        if pid not in planned_ids or index not in range(8) or index in grouped.setdefault(pid, {}):
            raise ValueError("Duplicate/unplanned diagnostic candidate")
        grouped[pid][index] = record
    per_prompt = []
    for pid in planned_ids:
        rows = grouped.get(pid, {})
        valid = len(rows) == 8 and all(row["status"] == "ok" and row.get("rm_score") is not None
            and math.isfinite(row["rm_score"]) for row in rows.values())
        result = {"prompt_id": pid, "status": "ok" if valid else "incomplete_candidate_scores",
                  "n_candidates_present": len(rows), "n_candidates_ok": sum(row["status"] == "ok" for row in rows.values())}
        if valid:
            ordered = [rows[i] for i in range(8)]
            result.update(prompt_statistics(*[[r[key] for r in ordered] for key in
                ("p_star", "rm_score", "response_tokens", "refusal_keyword_diagnostic", "capped_generation")]))
        per_prompt.append(result)
    valid = [r for r in per_prompt if r["status"] == "ok"]
    keys = sorted(set(valid[0]) - {"prompt_id", "status", "n_candidates_present", "n_candidates_ok"}) if valid else []
    metrics = {}
    for key in keys:
        values = [r[key] for r in valid if r[key] is not None]
        metrics[key] = {**bootstrap_ratio(values, repetitions=repetitions),
                        "n_undefined_among_complete_prompts": len(valid)-len(values)}
    return per_prompt, {"metric_scope": METRIC_SCOPE, "n_planned_prompts": len(planned_ids),
        "n_planned_candidates": len(planned_ids)*8, "n_present_candidates": len(records),
        "n_complete_prompts": len(valid), "n_incomplete_prompts": len(planned_ids)-len(valid),
        "complete_coverage": len(valid) == len(planned_ids),
        "incomplete_policy": "Exclude the whole prompt from paired means if any of its eight scores fail; never renormalize teacher mass over successes; report coverage explicitly",
        "failure_reasons": dict(Counter(r.get("failure_reason", "unspecified") for r in records if r["status"] != "ok")),
        "metrics": metrics,
        "bootstrap_note": "Identical seeded prompt resamples for all fully defined metrics; teacher-minus-uniform is paired within each prompt; conditional on the frozen training pool/teacher/RM, not training-seed uncertainty",
        "tie_rule": "Exact scalar equality; top mass summed over all RM ties; RM at teacher maximum averaged uniformly over its tied maximizers; constant Spearman vectors undefined, not zero",
        "refusal_note": "Lexical initial-phrase heuristic only, not semantic refusal or safety evaluation"}


def aggregate(args, prompts, events, masses, protocol):
    protocol_hash, records, manifests = object_hash(protocol), [], {}
    for shard in range(args.num_shards):
        folder = args.out / f"shard{shard}"
        manifest_path, settings_path = folder / "manifest.json", folder / "settings.json"
        manifest = json.loads(manifest_path.read_text())
        settings = json.loads(settings_path.read_text())
        if (manifest["protocol_sha256"] != protocol_hash or settings != {"protocol": protocol, "protocol_sha256": protocol_hash, "shard": shard}
                or file_hash(settings_path) != manifest["settings_sha256"] or manifest["shard"] != shard):
            raise ValueError("Aggregate shard protocol/settings mismatch")
        expected_files = set()
        for index, _, expected in planned_chunks(prompts, events, masses, shard, args.num_shards, args.chunk_prompts):
            path = folder / f"chunk{index:05d}.json"
            expected_files.add(path.name)
            if file_hash(path) != manifest["chunks_sha256"].get(path.name):
                raise ValueError("Changed/incomplete completed chunk")
            records.extend(verify_chunk(json.loads(path.read_text()), protocol_hash, expected))
        if expected_files != set(manifest["chunks_sha256"]):
            raise ValueError("Unexpected shard chunks")
        manifests[str(shard)] = file_hash(manifest_path)
    per_prompt, report = summarize(records, [p["prompt_id"] for p in prompts], args.bootstrap_repetitions)
    destination = args.out / "aggregate"
    destination.mkdir(parents=True, exist_ok=False)
    write_jsonl(destination / "per_prompt.jsonl", per_prompt)
    write_json(destination / "report.json", {**report, "protocol": protocol, "protocol_sha256": protocol_hash,
        "shard_manifest_sha256": manifests, "per_prompt_sha256": file_hash(destination / "per_prompt.jsonl")})
    print(json.dumps({"report": str(destination / "report.json"), "complete_coverage": report["complete_coverage"],
                      "n_complete_prompts": report["n_complete_prompts"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("score", "aggregate"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--teacher-name", default="nash_repair_v2")
    parser.add_argument("--rm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--chunk-prompts", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    if args.num_shards not in range(1, 5) or args.chunk_prompts < 1 or args.bootstrap_repetitions < 1:
        raise ValueError("Use 1..4 shards and positive chunk/bootstrap counts")
    if args.mode == "score" and (args.shard is None or not 0 <= args.shard < args.num_shards):
        raise ValueError("Score mode requires --shard in range")
    prompts, events, masses, lineage = load_inputs(args.root, args.teacher_name)
    from scripts.experiments.nbpo_repair_20260909 import evaluate_responses
    protocol = {"source_sha256": file_hash(__file__), "skywork_tokenization_source_sha256": file_hash(evaluate_responses.__file__),
        "inputs": lineage, "rm": rm_identity(args.rm), "num_shards": args.num_shards, "chunk_prompts": args.chunk_prompts,
        "runtime_package_versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
        "metric_scope": METRIC_SCOPE, "max_rm_tokens": 16384, "truncation": False,
        "rm_dtype": "bfloat16", "rm_calibration": "none", "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repetitions": args.bootstrap_repetitions, "candidate_selection": "All original train2000 learner occurrences, eight per prompt; no label/outcome filtering"}
    if args.mode == "score":
        score_shard(args, prompts, events, masses, protocol)
    else:
        aggregate(args, prompts, events, masses, protocol)


if __name__ == "__main__":
    main()
