#!/usr/bin/env python3
"""Freeze the OS-only stabilization grid and analysis before new training/ranking."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path


RMS = {
    "skywork": {"model": "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
                "revision": "cba2f842f3f1af2f1b2f0d35e794d789976390c5"},
    "athene": {"model": "Nexusflow/Athene-RM-8B",
               "revision": "cdf428f7b52a323b6cf4e9803e5bcba9f1fb5a59"},
    "armo": {"model": "RLHFlow/ArmoRM-Llama3-8B-v0.1",
             "revision": "eb2676d20da2f2d41082289d23c59b9f7427f955"},
}
SCHEDULE = ["target_os_k0p05", "target_os_k0p02", "target_os_k0p01",
            "target_os_k0p007", "target_os_k0p005"]
BOUNDARIES = [0, 180, 360, 540, 720]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def count_records(path: Path) -> int:
    if path.suffix == ".jsonl":
        return sum(bool(line.strip()) for line in path.open(encoding="utf-8"))
    value = json.loads(path.read_text(encoding="utf-8"))
    return len(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models-tsv", type=Path, required=True)
    parser.add_argument("--fixed647", type=Path, required=True)
    parser.add_argument("--stability-gate", type=Path, required=True)
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--remote-work", type=Path, required=True)
    parser.add_argument("--prior-metric-lock", type=Path, required=True)
    parser.add_argument("--prior-fixed647-summary", type=Path, required=True)
    parser.add_argument("--prior-fixed647-gates", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    if count_records(args.fixed647) != 647:
        raise RuntimeError("fixed held-out file is not exactly 647 records")
    prior = json.loads(args.prior_metric_lock.read_text(encoding="utf-8"))
    if prior.get("primary", {}).get("name") != "min_local_rm_marginal_win_rate_vs_base":
        raise RuntimeError("prior locked evaluator/primary differs")
    gate_hash = sha256(args.stability_gate)
    with args.models_tsv.open(encoding="utf-8") as handle:
        models = list(csv.DictReader(handle, delimiter="\t"))
    baseline_rows = [row for row in models if row["name"] not in {"base", "ronpo_full_expect", "ronpo_k_only"}]

    candidates = [
        {"id": "os_harden_a030_sft030_lr025_w020", "learning_rate": 2.5e-8,
         "warmup_ratio": 0.20, "reference_anchor_weight": 0.30,
         "preference_sft_weight": 0.03},
        {"id": "os_harden_a040_sft030_lr025_w025", "learning_rate": 2.5e-8,
         "warmup_ratio": 0.25, "reference_anchor_weight": 0.40,
         "preference_sft_weight": 0.03},
        {"id": "os_harden_a050_sft050_lr025_w020", "learning_rate": 2.5e-8,
         "warmup_ratio": 0.20, "reference_anchor_weight": 0.50,
         "preference_sft_weight": 0.05},
        {"id": "os_harden_a040_sft020_lr0125_w030", "learning_rate": 1.25e-8,
         "warmup_ratio": 0.30, "reference_anchor_weight": 0.40,
         "preference_sft_weight": 0.02},
    ]
    for row in candidates:
        row.update({
            "ronpo_alpha": 0.15, "ronpo_tau": 0.1, "eta": 0.0075,
            "ronpo_target_column": SCHEDULE[0],
            "ronpo_target_schedule_columns": SCHEDULE,
            "ronpo_target_schedule_boundaries": BOUNDARIES,
            "method": "ronpo_os", "stage": 1, "source": "os_only_stability_hardening_grid",
            "model_path": str(args.remote_work / "candidates" / row["id"]),
            "theory_note": "Stability-only hardening: stronger base trust region, nonzero preference SFT, lower peak LR, longer warmup, and frozen soft-to-hard OS schedule.",
        })
    grid = {
        "schema_version": 1,
        "status": "frozen_before_round_launch_and_ranking",
        "locked_at": now,
        "scope": "RONPO-OS only; no baseline training",
        "common": {
            "optimizer_steps": 900, "effective_batch_size": 16,
            "per_device_train_batch_size": 1, "gradient_accumulation_steps": 16,
            "gradient_checkpointing": True, "bf16": True, "attn_implementation": "sdpa",
            "cudnn_sdpa": False, "seed": 42, "save_steps": 100,
            "save_total_limit": 10,
        },
        "base_model": args.base_model,
        "target_dataset": str(args.target_dataset),
        "target_dataset_dataset_dict_sha256": sha256(args.target_dataset / "dataset_dict.json"),
        "candidates": candidates,
        "spent_sealed_split_touched": False,
    }
    grid_path = out / "sweep/os_grid.json"
    atomic_json(grid_path, grid)
    grid_lock = {
        "status": "locked_before_os_only_training_and_ranking", "locked_at": now,
        "grid_sha256": sha256(grid_path), "stability_gate_sha256": gate_hash,
        "spent_sealed_split_touched": False,
    }
    atomic_json(out / "sweep/os_grid_lock.json", grid_lock)

    metric = {
        "schema_version": 1,
        "status": "LOCKED_BEFORE_NEW_OS_TRAINING_AND_RANKING",
        "locked_at": now, "seed": 42, "stage": 1, "objectives": RMS,
        "primary": prior["primary"],
        "primary_reason": "This reuses the evaluator and primary locked prospectively for the immediately preceding Stage-1 OS comparison; it is not changed for the OS-only repair.",
        "checkpoint_gate": {
            "script": str(args.stability_gate), "sha256": gate_hash,
            "split": {"path": str(args.fixed647), "sha256": sha256(args.fixed647), "records": 647},
            "decode": {"seed": 42, "temperature": 0.7, "top_p": 0.9,
                       "max_new_tokens": 4096, "dtype": "bfloat16", "enable_thinking": False},
            "thresholds": {"records": 647, "empty": 0, "nonempty_paired_think_spans": 0,
                           "mean_word_ratio": [0.33, 2.0], "max_repeat_run": 20},
            "rule": "Every saved 100-step checkpoint is gated. Gate outputs are frozen before any reward score is computed.",
        },
        "selection": {
            "split_role": "fixed 647-prompt held-out set is used as validation for OS checkpoint selection in this OS-only repair",
            "decode": {"seed": 42, "temperature": 0.7, "top_p": 0.9,
                       "max_new_tokens": 2048, "dtype": "bfloat16", "enable_thinking": False},
            "eligible": "only checkpoints passing the independent 4096-token full-647 stability gate",
            "robust_step_preference": "First restrict to passing checkpoints whose available immediate 100-step neighbors also pass and that have at least one neighbor; if none exist, use all passing checkpoints and mark the selection isolated.",
            "ranking": "highest primary; exact ties break by earlier step then lexical config id",
        },
        "bootstrap": prior["bootstrap"],
        "secondary": [
            "per-RM raw paired deltas versus base with CIs",
            "per-RM marginal win rates with CIs",
            "mean marginal win rate and cross-objective spread",
            "mean prompt-level worst and min-objective-mean after per-prompt min-max normalization over the complete evaluated pool",
        ],
        "baseline_policy": {
            "training": "none",
            "models_tsv": str(args.models_tsv), "models_tsv_sha256": sha256(args.models_tsv),
            "rows": baseline_rows,
            "prior_fixed647_summary": str(args.prior_fixed647_summary),
            "prior_fixed647_summary_sha256": sha256(args.prior_fixed647_summary),
            "prior_fixed647_gates": str(args.prior_fixed647_gates),
            "prior_fixed647_gates_sha256": sha256(args.prior_fixed647_gates),
            "reuse_rule": "Reuse exact prior 647 generations/scores when model revision and decode match. Decode a frozen baseline only when no compatible generation exists. Never train a baseline.",
        },
        "fresh_confirmation": {
            "measured_once_after_selection_lock": True,
            "prompt_disjoint": True,
            "spent_604_split_forbidden": True,
            "local_rm_primary": "same primary and evaluator as above",
            "independent_panel": prior["independent_confirmation"],
            "interpretation": "The independent panel is reported as confirmatory evidence. It is not used to alter the selected OS checkpoint or evaluator.",
        },
        "decision": {
            "PASS": "RONPO-OS is strictly highest among all eligible models on the one-shot fresh local-RM primary.",
            "PARTIAL": "RONPO-OS is strictly highest among eligible trained methods but below base on the one-shot fresh local-RM primary.",
            "FAIL": "Any eligible trained baseline ties or exceeds RONPO-OS, or no OS checkpoint passes the 647 gate.",
            "paper_and_upload": "Only PASS or PARTIAL permits a public OS upload and a Table-4 rebuild. The independent-panel result and overlapping intervals must be scoped honestly.",
        },
        "table_rule": "Never append a fresh OS row to the spent-604 table. On PASS/PARTIAL regenerate every displayed baseline and OS row on the same fresh split and protocol.",
        "spent_sealed_split_touched": False,
    }
    metric_path = out / "metric_lock.json"
    atomic_json(metric_path, metric)
    (out / "metric_lock.sha256").write_text(f"{sha256(metric_path)}  metric_lock.json\n", encoding="utf-8")
    gate_spec = {"script": str(args.stability_gate), "sha256": gate_hash,
                 "configuration": metric["checkpoint_gate"], "reward_blind": True,
                 "unchanged_script": True, "spent_sealed_split_touched": False}
    gate_spec_path = out / "stability_gate_spec.json"
    atomic_json(gate_spec_path, gate_spec)
    (out / "stability_gate_spec.sha256").write_text(
        f"{sha256(gate_spec_path)}  stability_gate_spec.json\n", encoding="utf-8")
    prereg = f"""# Qwen3-8B RONPO-OS-only stabilization preregistration

Locked at `{now}` before the new OS training runs and before any new OS reward ranking.

Only RONPO-OS is trained. The public Stage-1 baseline revisions in `models.tsv` are frozen and receive no additional tuning. The four OS recipes vary only the predeclared stability controls: reference anchoring, preference-SFT anchoring, peak learning rate, and warmup. All share the same 900-step budget, seed 42, base KL anchor, Stage-1 pool, and frozen soft-to-hard OS target schedule.

Every 100-step checkpoint is decoded on all 647 held-out prompts with the frozen non-thinking 4096-token gate decode and tested by the unchanged reward-blind detector. Gate decisions are locked before reward scoring. Performance is then measured under the common 2048-token protocol. Selection first prefers a passing step whose available adjacent 100-step checkpoints also pass, then maximizes the already-locked minimum local-RM marginal win rate against base. Exact ties prefer the earlier step and then the lexical configuration id.

The fixed 647 prompts are OS validation in this repair. They are not presented as a fresh test. After the OS checkpoint is locked, one new prompt-disjoint split is measured once. A Table-4 change is allowed only under the PASS or PARTIAL rule in `metric_lock.json`, and every row must be regenerated on that same fresh split. The prior 604-prompt sealed directory is never read.
"""
    (out / "PREREG.md").write_text(prereg, encoding="utf-8")
    atomic_json(out / "prereg_lock.json", {
        "status": "LOCKED_BEFORE_OS_TRAINING_AND_RANKING", "locked_at": now,
        "metric_lock_sha256": sha256(metric_path), "os_grid_sha256": sha256(grid_path),
        "stability_gate_spec_sha256": sha256(gate_spec_path),
        "prereg_sha256": sha256(out / "PREREG.md"), "spent_sealed_split_touched": False,
    })
    print(json.dumps({"metric_lock_sha256": sha256(metric_path),
                      "os_grid_sha256": sha256(grid_path),
                      "stability_gate_sha256": gate_hash,
                      "candidates": [row["id"] for row in candidates]}, indent=2))


if __name__ == "__main__":
    main()
