"""Read-only audit of the completed, fixed-1750 MSE/WBC primary comparison.

Reads existing artifacts only. Writes reports into one new exclusive directory.
Never imports a training/GPU library, resumes a run, selects a checkpoint, or
changes a horizon. Token exposure is not FLOPs; allocated GPU-hours are not
hardware-active GPU-hours. Missing optional diagnostics are not failed evidence.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path

import yaml

STEPS, RANKS, EVAL_EVERY, DEV_PAIRS = 1750, 4, 250, 14000
FP32, BF16 = "torch.float32", "torch.bfloat16"
COUNTERS = ("policy_forward_tokens", "policy_response_tokens", "reference_forward_tokens", "reference_response_tokens")
ALLOWED_CONFIG_DIFFERENCES = {"loss_type", "nbpo_online_reference", "nbpo_verify_reference_initialization", "output_dir", "run_name"}
CORE_REQUIRED = {"mnpo_scripts/run_mnpo.py", "mnpo_scripts/mnpo_trainer.py", "mnpo_scripts/nbpo_runtime.py",
                 "mnpo_scripts/response_logps.py", "mnpo_scripts/pair_tokenization.py", "scripts/simpo_trainer.py",
                 "scripts/experiments/nbpo_repair_20260909/train_job.py"}
DS_NAME = "training_configs/nbpo/repair_20260909/deepspeed_zero2.json"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def strict_json(text):
    def reject(value):
        raise ValueError(f"Nonfinite JSON numeric literal: {value}")
    return json.loads(text, parse_constant=reject)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def describe(values):
    values = list(values)
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "median": statistics.median(values) if values else None}


class Audit:
    def __init__(self):
        self.errors, self.missing, self.evidence = [], [], {}

    def require(self, condition, code, detail):
        if not condition:
            self.errors.append({"code": code, "detail": detail})
        return bool(condition)

    def absent(self, detail):
        if detail not in self.missing:
            self.missing.append(detail)

    def file(self, path, expected=None):
        actual = sha256(path)
        self.evidence[str(path)] = actual
        if expected is not None:
            self.require(actual == expected, "artifact_hash", str(path))
        return actual

    def json(self, path, expected=None):
        self.file(path, expected)
        return strict_json(Path(path).read_text())


def core_source(name):
    return name.startswith(("mnpo_scripts/", "alignment/")) or name in CORE_REQUIRED or name == DS_NAME


def archived_config(audit, job, launch):
    """Validate archived bytes, not mutable current checkout/evaluation scripts."""
    audit.file(job / "source.zip", launch["source_archive_sha256"])
    with zipfile.ZipFile(job / "source.zip") as archive:
        names = archive.namelist()
        audit.require(len(names) == len(set(names)), "duplicate_archive_member", str(job))
        expected = set(launch["source_hashes"]) | {"resolved_run_config.yaml"}
        audit.require(set(names) == expected, "archive_inventory", str(job))
        for name, expected_hash in launch["source_hashes"].items():
            audit.require(digest(archive.read(name)) == expected_hash, "archived_source_hash", f"{job}:{name}")
        config_bytes = archive.read("resolved_run_config.yaml")
        audit.require(digest(config_bytes) == launch["config_sha256"], "archived_config_hash", str(job))
        ds = json.loads(archive.read(DS_NAME))
        audit.require(ds["zero_optimization"]["stage"] == 2 and ds["bf16"]["enabled"] is True
                      and ds["fp16"]["enabled"] is False and ds["optimizer"]["type"] == "AdamW"
                      and ds["optimizer"]["params"]["torch_adam"] is True,
                      "deepspeed_contract", str(job))
    audit.require(CORE_REQUIRED <= set(launch["source_hashes"]), "missing_core_source", str(job))
    config = yaml.safe_load(config_bytes)
    json.dumps(config, allow_nan=False)  # Reject nonfinite/non-serializable frozen settings.
    return config


def validate_config(audit, arm, cfg, frozen):
    required = {"max_steps": STEPS, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
        "eval_steps": EVAL_EVERY, "save_steps": EVAL_EVERY, "eval_strategy": "steps", "save_strategy": "steps",
        "load_best_model_at_end": False, "logging_steps": 1, "seed": 42, "data_seed": 42,
        "use_peft": False, "bf16": True, "torch_dtype": "bfloat16", "optim": "adamw_torch",
        "max_grad_norm": 1., "disable_dropout": True, "train_split": "train", "eval_split": "dev",
        "nbpo_require_fp32_optimizer": True, "nbpo_require_immutable_tokens": True,
        "nbpo_target_mode": "canonical_logratio", "nbpo_target_units": "final_logratio_change",
        "nbpo_eta_already_included": True, "logp_reduction": "sum", "nbpo_num_candidates": 8,
        "nbpo_eval_online_reference": True, "nbpo_reference_init_atol": 1e-4,
        "loss_type": "nbpo" if arm == "mse" else "nbpo_wbc",
        "nbpo_online_reference": arm == "mse", "nbpo_verify_reference_initialization": arm == "mse"}
    for key, value in required.items():
        audit.require(cfg.get(key) == value, "frozen_training_contract", f"{arm}:{key}: {cfg.get(key)!r} != {value!r}")
    audit.require(cfg.get("nbpo_profile_updates", 0) == 0 and not cfg.get("resume_from_checkpoint"),
                  "nonprimary_run", arm)
    audit.require(cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"] * RANKS == frozen["global_batch_pairs"],
                  "global_batch", arm)


def validate_state(audit, arm, state):
    audit.require(state["global_step"] == state["max_steps"] == STEPS, "fixed_final_horizon", arm)
    audit.require(state.get("best_model_checkpoint") is None, "best_checkpoint_selection", arm)
    train = [row for row in state["log_history"] if "loss" in row]
    audit.require([row["step"] for row in train] == list(range(1, STEPS + 1)), "training_log_steps", arm)
    evaluations = [row for row in state["log_history"] if "eval_loss" in row]
    audit.require([row["step"] for row in evaluations] == list(range(EVAL_EVERY, STEPS + 1, EVAL_EVERY)),
                  "dev_evaluation_schedule", arm)
    for row in evaluations:
        audit.require(row.get("eval_nbpo/pair_rows") == DEV_PAIRS, "dev_pair_coverage", f"{arm}:step{row['step']}")
        audit.require(finite(row.get("eval_loss")), "nonfinite_dev_loss", f"{arm}:step{row['step']}")
    return {"global_step": state["global_step"], "training_loss_log_rows": len(train),
        "dev_evaluations": [{"step": r["step"], "pair_rows": r.get("eval_nbpo/pair_rows"),
                             "runtime_seconds": r.get("eval_runtime")} for r in evaluations],
        "selection": "Final step 1750, no best-checkpoint selection",
        "hf_total_flos_not_used": state.get("total_flos"), "hf_num_input_tokens_seen_not_used": state.get("num_input_tokens_seen")}


def validate_runtime(audit, arm, ranks, clip):
    offsets, final_tokens = {}, {}
    for rank, rows in ranks.items():
        audit.require([row["step"] for row in rows] == list(range(1, STEPS + 1)), "runtime_update_sequence", f"{arm}:rank{rank}")
        prior = {split: dict.fromkeys(COUNTERS, 0) for split in ("train", "eval")}
        offset = None
        for row in rows:
            where = f"{arm}:rank{rank}:step{row['step']}"
            audit.require(row.get("master_dtypes") == row.get("adam_state_dtypes") == [FP32]
                          and row.get("loss_dtype") == FP32 and row.get("forward_parameter_dtype") == BF16,
                          "observed_precision", where)
            logps = row.get("logp_dtypes", {})
            audit.require(all(logps.get(k) == FP32 for k in ("log_softmax", "selected_logp", "sequence_logp")),
                          "observed_logp_precision", where)
            audit.require(row.get("rotary_buffer_restorations", 0) > 0, "rotary_precision_guard", where)
            counts = row["cumulative_forward_tokens_this_rank"]
            for split in ("train", "eval"):
                current = counts.get(split, dict.fromkeys(COUNTERS, 0))
                for key in COUNTERS:
                    value = current[key]
                    audit.require(type(value) is int and value >= prior[split][key], "token_counter_monotonicity", f"{where}:{split}:{key}")
                audit.require(current["policy_response_tokens"] <= current["policy_forward_tokens"], "response_token_bound", where)
                if split == "train":
                    audit.require(current["policy_forward_tokens"] > prior[split]["policy_forward_tokens"], "empty_training_update", where)
                    this_offset = tuple(current[f"reference_{kind}_tokens"] - current[f"policy_{kind}_tokens"] for kind in ("forward", "response"))
                    if arm == "wbc":
                        audit.require(current["reference_forward_tokens"] == current["reference_response_tokens"] == 0,
                                      "wbc_training_reference_forward", where)
                    else:
                        if offset is None:
                            offset = this_offset
                            audit.require(0 < offset[1] <= offset[0] <= current["policy_forward_tokens"], "mse_startup_reference_repeat", where)
                        audit.require(this_offset == offset, "mse_reference_constant_startup_offset", where)
                else:
                    audit.require(current["reference_forward_tokens"] == current["policy_forward_tokens"]
                                  and current["reference_response_tokens"] == current["policy_response_tokens"],
                                  "eval_reference_forward", where)
                prior[split] = current
            norm = row.get("preclip_grad_norm")
            if norm is None:
                audit.absent(f"{arm}: preclip norm missing at one or more updates")
            else:
                audit.require(finite(norm) and norm >= 0, "preclip_norm_nonfinite", where)
                audit.require(row.get("clipped") == (norm > clip), "clipping_flag", where)
        offsets[str(rank)] = offset
        final_tokens[str(rank)] = prior
    global_tokens = {split: {key: sum(value[split][key] for value in final_tokens.values())
                            for key in COUNTERS} for split in ("train", "eval")}
    return {"per_rank_final_observed_cumulative_tokens": final_tokens,
            "global_last_observed_cumulative_tokens": global_tokens,
            "mse_startup_reference_extra_tokens_per_rank": offsets if arm == "mse" else None}


def compare_exposure(audit, runs):
    checks = 0
    for rank in range(RANKS):
        a, b = runs["wbc"][rank], runs["mse"][rank]
        for wa, mb in zip(a, b):
            audit.require(wa["step"] == mb["step"], "paired_runtime_step", str(rank))
            for split in ("train", "eval"):
                ca = wa["cumulative_forward_tokens_this_rank"].get(split, {})
                cb = mb["cumulative_forward_tokens_this_rank"].get(split, {})
                for key in ("policy_forward_tokens", "policy_response_tokens"):
                    audit.require(ca.get(key, 0) == cb.get(key, 0), "policy_token_exposure_mismatch",
                                  f"rank{rank}:step{wa['step']}:{split}:{key}")
                    checks += 1
    return {"per_step_rank_counter_equalities_checked": checks,
            "scope": "Exact observed non-padding policy input and labeled response token exposure, not FLOPs or full data-order identity. Counters exclude padded-token compute and activation-checkpoint recomputation during backward.",
            "reference_difference": "MSE training includes one frozen reference forward per batch plus one startup repeat per rank; WBC training has none. Both evaluate with a frozen reference."}


def runtime_diagnostics(audit, arm, ranks):
    rows0 = ranks[0]
    norms = [r["preclip_grad_norm"] for r in rows0 if finite(r.get("preclip_grad_norm"))]
    clipped = [r["clipped"] for r in rows0 if isinstance(r.get("clipped"), bool)]
    times, updates, memory = [], [], []
    for rank, rows in ranks.items():
        first = [r for r in rows if 1 <= r["step"] <= 20]
        for row in first:
            for field in ("seconds_per_update_including_instrumentation", "master_update_l2", "peak_allocated_bytes", "peak_reserved_bytes"):
                if not finite(row.get(field)):
                    audit.absent(f"{arm}: first20 {field} unavailable on at least one rank/update")
                else:
                    audit.require(row[field] >= 0, "negative_runtime_diagnostic", f"{arm}:{rank}:{row['step']}:{field}")
            if rank == 0 and finite(row.get("master_update_l2")):
                updates.append(row["master_update_l2"])
            if finite(row.get("peak_allocated_bytes")):
                memory.append(row["peak_allocated_bytes"])
            if rank == 0 and finite(row.get("seconds_per_update_including_instrumentation")):
                times.append(row["seconds_per_update_including_instrumentation"])
    slowest = []
    for step in range(1, 21):
        at_step = [next((r for r in ranks[rank] if r["step"] == step), {}) for rank in range(RANKS)]
        vals = [r.get("seconds_per_update_including_instrumentation") for r in at_step]
        if all(finite(v) for v in vals):
            slowest.append(max(vals))
        l2 = [r.get("master_update_l2") for r in at_step]
        if all(finite(v) for v in l2):
            audit.require(all(math.isclose(v, l2[0], rel_tol=1e-9, abs_tol=1e-12) for v in l2), "global_master_norm_rank_agreement", f"{arm}:step{step}")
    return {"preclip_global_grad_norm_rank0": describe(norms),
        "clipped_fraction_rank0": statistics.mean(clipped) if clipped else None, "clipping_observed_updates": len(clipped),
        "first20_rank0_callback_seconds": describe(times), "first20_slowest_rank_callback_seconds": describe(slowest),
        "first20_global_master_update_l2": describe(updates), "first20_max_allocated_bytes_all_ranks": max(memory) if memory else None,
        "interpretation": "Master update L2 is computed from local ZeRO-2 FP32 partitions then sum-reduced in squared norm across ranks; every rank records the same GLOBAL norm, so never sum these norms again. Update/timing/peak-memory diagnostics exist only for first20. Callback seconds include instrumentation and exclude later evaluation/checkpoint work. Preclip norms are rank0 observations of DeepSpeed global norm; clipping is the declared threshold indicator, not a separately observed postclip norm."}


def reference_probes(audit, output, cfg, linkage):
    reports = []
    for rank in range(RANKS):
        path = output / f"reference_init_rank{rank}.json"
        if not path.exists():
            audit.absent(f"mse: initial reference probe missing on rank{rank}; identity not independently reconstructable from weights alone")
            continue
        probe = audit.json(path)
        audit.require(probe["reference_frozen"] is True and probe["absolute_tolerance"] == cfg["nbpo_reference_init_atol"],
                      "initial_reference_contract", str(path))
        for field in ("sequence_max_abs", "sequence_rms", "pair_h_rms", "repeated_reference_max_abs"):
            audit.require(finite(probe.get(field)) and 0 <= probe[field] <= cfg["nbpo_reference_init_atol"],
                          "initial_reference_identity", f"rank{rank}:{field}")
        reports.append({"rank": rank, "sha256": audit.evidence[str(path)], **probe})
    return {"reports": reports, "linkage": linkage,
        "scope": "Files from this completed arm's output directory, bound here to launch/source/dataset/model revision and runtime startup-repeat counters; the legacy probe file itself lacks token-event IDs or an embedded launch hash, so no stronger event-level cryptographic claim is made."}


def compare_training(root):
    root, audit = Path(root), Audit()
    report = {"schema": "nbpo_training_comparison_v1", "arms": {}, "comparison": {}, "root": str(root),
              "read_only_training_artifacts": True, "scorer_source_sha256": sha256(__file__)}
    try:
        complete = audit.json(root / "controllers/primary_chain_v1/complete.json")
        frozen = audit.json(root / "protocols/primary_horizon_frozen_v1.json")
        audit.require({r["arm"] for r in complete["arms"]} == {"wbc", "mse"} and len(complete["arms"]) == 2,
                      "completed_arm_inventory", "Exactly two completed primary arms required")
        for key, value in {"optimizer_updates_each": STEPS, "global_batch_pairs": 32, "pair_rows": 56000,
                           "pair_row_epochs_each": 1., "policy_seed": 42, "data_seed": 42}.items():
            audit.require(frozen[key] == value, "frozen_protocol", key)
        configs, launches, runs = {}, {}, {}
        entries = {r["arm"]: r for r in complete["arms"]}
        for arm in ("wbc", "mse"):
            job, output = root / "jobs" / f"{arm}_primary_v1", root / "arms" / f"{arm}_primary_v1"
            entry = entries[arm]
            launch = audit.json(job / "launch.json")
            launches[arm] = launch
            exit_record = audit.json(job / "exit.json", entry["job_exit_sha256"])
            state = audit.json(output / "trainer_state.json", entry["trainer_state_sha256"])
            audit.require(exit_record["exit_code"] == 0 and exit_record["gpu_count"] == RANKS,
                          "job_exit", arm)
            audit.require(entry["global_step"] == STEPS, "controller_horizon", arm)
            audit.file(job / "stdout.log", exit_record["log_sha256"])
            cfg = configs[arm] = archived_config(audit, job, launch)
            audit.file(root / "configs" / f"{arm}_primary_v1.yaml", frozen["configs"][arm]["sha256"])
            audit.require(launch["config_sha256"] == frozen["configs"][arm]["sha256"], "launch_frozen_config", arm)
            recorded_root = str(Path(launch["cwd"]).parent)
            expected_config = f"{recorded_root}/configs/{arm}_primary_v1.yaml"
            audit.require(launch["config"] == frozen["configs"][arm]["path"] == expected_config
                          and cfg["output_dir"] == f"{recorded_root}/arms/{arm}_primary_v1"
                          and cfg["run_name"] == f"{arm}_primary_v1", "launch_output_paths", arm)
            command = ["python3", "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=4", "-m", "mnpo_scripts.run_mnpo", expected_config]
            audit.require(launch["command"] == command and launch["environment_overrides"].get("CUDA_VISIBLE_DEVICES") == "0,1,2,3",
                          "launch_command", arm)
            audit.require(dt.datetime.fromisoformat(frozen["frozen_utc"]) <= dt.datetime.fromisoformat(launch["started_utc"]), "config_frozen_before_launch", arm)
            validate_config(audit, arm, cfg, frozen)
            audit.require(cfg.get("dataset_mixer") == {f"{recorded_root}/datasets/nash_repair_v2": 1.0}, "training_dataset_binding", arm)
            audit.require(cfg.get("deepspeed") == f"{recorded_root}/code/{DS_NAME}", "loaded_deepspeed_binding", arm)
            audit.file(root / "datasets/nash_repair_v2/precompute_manifest.json", cfg["nbpo_expected_dataset_manifest_sha256"])
            audit.file(root / "teachers/nash_repair_v2/train/solver/solution.json", cfg["nbpo_expected_solver_artifact_sha256"])
            weights = {p.name for p in output.glob("*.safetensors")}
            audit.require(bool(weights) and weights == set(entry["weight_sha256"]), "final_export_inventory", arm)
            for name, expected_hash in entry["weight_sha256"].items():
                audit.file(output / name, expected_hash)
            audit.file(output / "config.json")
            ranks = {}
            for rank in range(RANKS):
                path = output / f"runtime_rank{rank}.jsonl"
                audit.file(path)
                ranks[rank] = [strict_json(line) for line in path.read_text().splitlines() if line.strip()]
            runs[arm] = ranks
            seconds = exit_record["seconds"]
            audit.require(finite(seconds) and seconds > 0 and math.isclose(seconds, entry["runtime_seconds"], abs_tol=1e-6)
                          and math.isclose(RANKS * seconds / 3600, entry["gpu_hours_allocated"], abs_tol=1e-9), "job_wall_accounting", arm)
            details = {**validate_state(audit, arm, state), **validate_runtime(audit, arm, ranks, cfg["max_grad_norm"]),
                "runtime_diagnostics": runtime_diagnostics(audit, arm, ranks), "job_wall_seconds": seconds,
                "allocated_gpu_hours": RANKS * seconds / 3600,
                "wall_scope": "Persistent job process wall time including startup, training, dev evaluation, and final export; four allocated GPUs, not measured GPU-active utilization",
                "launch_sha256": audit.evidence[str(job / "launch.json")], "config_sha256": launch["config_sha256"]}
            if arm == "mse":
                details["initial_reference_probe"] = reference_probes(audit, output, cfg,
                    {"launch_sha256": details["launch_sha256"], "model_revision": cfg.get("model_revision"),
                     "dataset_manifest_sha256": cfg["nbpo_expected_dataset_manifest_sha256"],
                     "trainer_source_sha256": launch["source_hashes"]["mnpo_scripts/mnpo_trainer.py"]})
            report["arms"][arm] = details
        all_keys = set(configs["wbc"]) | set(configs["mse"])
        differences = {key: {arm: configs[arm].get(key) for arm in configs} for key in sorted(all_keys)
                       if configs["wbc"].get(key) != configs["mse"].get(key)}
        audit.require(set(differences) <= ALLOWED_CONFIG_DIFFERENCES, "undeclared_config_difference", str(sorted(set(differences) - ALLOWED_CONFIG_DIFFERENCES)))
        a, b = launches["wbc"], launches["mse"]
        audit.require(a["cwd"] == b["cwd"] and a["environment_overrides"] == b["environment_overrides"], "launch_environment_difference", "Matched environment required")
        source_a = {k: v for k, v in a["source_hashes"].items() if core_source(k)}
        source_b = {k: v for k, v in b["source_hashes"].items() if core_source(k)}
        audit.require(source_a == source_b, "training_source_difference", "Archived training, alignment, SimPO, and loaded DeepSpeed config must match")
        source_diff = sorted(k for k in set(a["source_hashes"]) | set(b["source_hashes"])
                             if a["source_hashes"].get(k) != b["source_hashes"].get(k))
        report["comparison"] = {**compare_exposure(audit, runs), "declared_config_differences": differences,
            "nontraining_source_differences": [k for k in source_diff if not core_source(k)],
            "training_source_files_compared": len(source_a), "allocated_gpu_hours_total": sum(v["allocated_gpu_hours"] for v in report["arms"].values())}
        audit.absent("Final step1750 dev-evaluation token counters are not logged after evaluation: callbacks run before eval. Seven 14,000-pair evaluations are verified from trainer_state, but exact final-eval token totals cannot be reconstructed here.")
        audit.absent("No per-update timing/master-update norms beyond first20, no postclip norm, and no measured exact FLOPs or hardware-active GPU-hours; HF headline samples/sec and total_flos are not substitutes.")
        report["precision_scope"] = "Every callback checks all observed ZeRO master partitions and all Adam first/second-moment state dtypes. BF16 forward is evidenced by frozen configuration and the recorded first model-parameter dtype, not a trace of every intermediate activation; log-softmax, selected/sequence logp, and loss are observed FP32."
    except (OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        audit.require(False, "missing_or_malformed_contract_artifact", f"{type(exc).__name__}: {exc}")
    report.update(contract_passed=not audit.errors, status="passed" if not audit.errors else "failed",
                  errors=audit.errors, missing_diagnostics=audit.missing, evidence_sha256=audit.evidence)
    return report


def markdown(report):
    lines = ["# Fixed-horizon MSE/WBC training audit", "", f"Contract: **{report['status']}**. Read-only artifact comparison; no jobs launched or selected.", "",
             "| Arm | Updates | Job wall (s) | Allocated GPU-hours | Clipped fraction | First20 callback mean (s) |", "|---|---:|---:|---:|---:|---:|"]
    for arm, row in report["arms"].items():
        d = row["runtime_diagnostics"]
        lines.append(f"| {arm} | {row['global_step']} | {row['job_wall_seconds']:.3f} | {row['allocated_gpu_hours']:.6f} | {d['clipped_fraction_rank0']} | {d['first20_rank0_callback_seconds']['mean']} |")
    lines += ["", "Matched policy token exposure is not equal FLOPs. MSE has extra frozen-reference training forwards. Callback timing/norms concern first20 only; master L2 is already global after squared-norm reduction across ZeRO partitions and must not be summed over ranks. GPU-hours multiply job wall by four, not GPU utilization.", "", "## Contract failures", ""]
    lines += [f"- {r['code']}: {r['detail']}" for r in report["errors"]] or ["None."]
    lines += ["", "## Missing diagnostics / interpretation limits", ""]
    lines += [f"- {r}" for r in report["missing_diagnostics"]]
    lines += ["", "All exact counts, allowed configuration differences, and source hashes are in `training_comparison.json`. No training-seed uncertainty or downstream quality claim is made.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New exclusive report namespace; existing directories are refused")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Report namespace already exists; no artifact is overwritten")
    report = compare_training(args.root)
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "training_comparison.json").open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with (args.out / "training_comparison.md").open("x") as handle:
        handle.write(markdown(report))
    print(json.dumps({"status": report["status"], "report": str(args.out / "training_comparison.json"), "errors": len(report["errors"])}))
    raise SystemExit(0 if report["contract_passed"] else 2)


if __name__ == "__main__":
    main()
