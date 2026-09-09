"""Freeze matched main configs only after both corrected20-update profiles pass."""
import argparse
import datetime
import json
import statistics
import subprocess
from pathlib import Path

from scripts.experiments.nbpo_repair_20260909.common import file_hash, read_jsonl, write_json


def validate_profiles(root):
    reports, records = {}, {}
    for arm in ("wbc", "mse"):
        job = root / "jobs" / f"profile_{arm}_v2"
        if json.loads((job / "exit.json").read_text())["exit_code"] != 0:
            raise ValueError(f"{arm} profile did not complete successfully")
        directory = root / "profiles" / f"{arm}_v2"
        records[arm] = {rank: read_jsonl(directory / f"runtime_rank{rank}.jsonl") for rank in range(4)}
        for rank, rows in records[arm].items():
            if [r["step"] for r in rows] != list(range(1, 21)):
                raise ValueError("Exactly20 measured optimizer updates perrank required")
            for row in rows:
                if (row["master_dtypes"] != ["torch.float32"] or row["adam_state_dtypes"] != ["torch.float32"]
                        or row["loss_dtype"] != "torch.float32" or row["forward_parameter_dtype"] != "torch.bfloat16"
                        or row["rotary_buffer_restorations"] < 1):
                    raise ValueError("Observed numerical precision contract is not satisfied")
            tokens = rows[-1]["cumulative_forward_tokens_this_rank"]["train"]
            if (arm == "wbc" and tokens["reference_forward_tokens"] != 0) or (arm == "mse" and tokens["reference_forward_tokens"] <= 0):
                raise ValueError("Training reference-forward contract mismatch")
            if arm == "mse":
                init = json.loads((directory / f"reference_init_rank{rank}.json").read_text())
                if any(init[key] > 1e-4 for key in ("sequence_max_abs", "pair_h_rms", "repeated_reference_max_abs")):
                    raise ValueError("Actual distributed initial reference identity failed")
        reports[arm] = {"mean_update_seconds_including_instrumentation": statistics.mean(
            r["seconds_per_update_including_instrumentation"] for r in records[arm][0]),
            "maximum_allocated_bytes_all_ranks": max(r["peak_allocated_bytes"] for v in records[arm].values() for r in v),
            "clipped_fraction": statistics.mean(r["clipped"] for r in records[arm][0]),
            "runtime_file_sha256": {str(rank): file_hash(directory / f"runtime_rank{rank}.jsonl") for rank in range(4)},
            "profile_config_sha256": file_hash(root / "configs" / f"profile_{arm}_v2.yaml")}
    for rank in range(4):
        a = records["wbc"][rank][-1]["cumulative_forward_tokens_this_rank"]["train"]
        b = records["mse"][rank][-1]["cumulative_forward_tokens_this_rank"]["train"]
        for key in ("policy_forward_tokens", "policy_response_tokens"):
            if a[key] != b[key]:
                raise ValueError("Actual first20 policy token exposure differs between arms")
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    reports = validate_profiles(args.root)
    now = datetime.datetime.now(datetime.timezone.utc)
    deadline = datetime.datetime.fromisoformat("2026-09-10T09:00:00+09:00")
    steps = 1750
    train_seconds = steps * sum(v["mean_update_seconds_including_instrumentation"] for v in reports.values())
    reserve = 6 * 3600  # fourhours for dev/I/O headroom plus final twohours generation/judging/reporting
    if train_seconds + reserve > (deadline - now).total_seconds():
        raise ValueError("Planned1750 plus conservative reserve no longer fits; explicitly amend matched horizon before main outcomes")
    configs = {}
    for arm, loss in (("wbc", "nbpo_wbc"), ("mse", "nbpo")):
        config = args.root / "configs" / f"{arm}_primary_v1.yaml"
        command = ["python3", "-m", "mnpo_scripts.materialize_nbpo_train",
            "--dataset", str(args.root / "datasets/nash_repair_v2"),
            "--output-dir", str(args.root / "arms" / f"{arm}_primary_v1"),
            "--config-output", str(config), "--loss", loss, "--steps", str(steps)]
        subprocess.run(command, check=True)
        configs[arm] = {"path": str(config), "sha256": file_hash(config)}
    write_json(args.root / "protocols/primary_horizon_frozen_v1.json", {
        "frozen_utc": now.isoformat(), "deadline_kst": deadline.isoformat(),
        "optimizer_updates_each": steps, "global_batch_pairs": 32, "pair_rows": 56000,
        "pair_row_epochs_each": 1.0, "policy_seed": 42, "data_seed": 42,
        "selection": "Fixed final1750; no benchmark selection or regression stop gate",
        "loss_comparison_scope": "Same policy token exposure, not equal FLOPs; frozen reference forward only for MSEtrain",
        "nMSE_definition_new": "mean((h-target)^2)/mean(target^2), zero-predictor-relative; legacy reports used targetvariance",
        "profiles": reports, "first20_policy_token_exposure_equal_all_ranks": True,
        "estimated_training_only_seconds": train_seconds, "dev_io_evaluation_report_reserve_seconds": reserve,
        "estimated_finish_with_reserve_utc": (now + datetime.timedelta(seconds=train_seconds+reserve)).isoformat(),
        "configs": configs, "source_sha256": file_hash(__file__),
        "note": "Profile Trainer headline samples/stepspersecond uses planned1750 despite callbackstop20; only actual callback timing is used here"})


if __name__ == "__main__":
    main()
