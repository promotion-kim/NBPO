#!/usr/bin/env python3
"""Create the prospective Stage-1 OS metric lock and symmetric candidate grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path


RMS = {
    "skywork": {"model": "Skywork/Skywork-Reward-V2-Llama-3.1-8B", "revision": "cba2f842f3f1af2f1b2f0d35e794d789976390c5"},
    "athene": {"model": "Nexusflow/Athene-RM-8B", "revision": "cdf428f7b52a323b6cf4e9803e5bcba9f1fb5a59"},
    "armo": {"model": "RLHFlow/ArmoRM-Llama3-8B-v0.1", "revision": "eb2676d20da2f2d41082289d23c59b9f7427f955"},
}
METHODS = (
    "ronpo_os", "ronpo_full_expect", "ronpo_k_only", "dpo", "ipo", "simpo",
    "sppo_avg", "inpo_avg", "ht_mnpo_helpfulness", "ht_mnpo_safety", "ht_mnpo_conciseness",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def count_records(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return sum(bool(line.strip()) for line in text.splitlines())
    value = json.loads(text)
    return len(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fair-grid", type=Path, required=True)
    parser.add_argument("--models-tsv", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--remote-project", type=Path, required=True)
    parser.add_argument("--remote-exp-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    fair = json.loads(args.fair_grid.read_text(encoding="utf-8"))
    if fair.get("budget_rule") != "Exactly two configs per reported method; no extension after validation rankings are visible.":
        raise RuntimeError("fair grid is not the frozen symmetric grid")
    with args.models_tsv.open(encoding="utf-8") as handle:
        frozen_rows = list(csv.DictReader(handle, delimiter="\t"))
    frozen = {row["method"]: row for row in frozen_rows if row["method"] != "base"}

    candidates = [
        {
            "id": "os_r1_s900", "method": "ronpo_os", "stage": 1, "optimizer_steps": 900,
            "source": "ronpo_variant_search_round1_final",
            "model_path": str(args.remote_exp_root / "ronpo_variant_search_20260715/round1/candidates/r1_os_stratified_k002"),
            "validation_generation": str(args.remote_project / "results/p1_8b_ronpo_variant_search_20260715/round1_validation/generations/r1_os_stratified_k002__s900/output_42.json"),
            "config_id": "r1_os_stratified_k002", "target": "objective_stratified",
        },
        {
            "id": "os_r2_s900", "method": "ronpo_os", "stage": 1, "optimizer_steps": 900,
            "source": "ronpo_variant_search_round2_final",
            "model_path": str(args.remote_exp_root / "ronpo_variant_search_20260715/round2/candidates/r2_os_anneal_anchor030"),
            "validation_generation": str(args.remote_project / "results/p1_8b_ronpo_variant_search_20260715/round2_validation/generations/r2_os_anneal_anchor030__s900/output_42.json"),
            "config_id": "r2_os_anneal_anchor030", "target": "objective_stratified_annealed",
        },
    ]
    fair_root = args.remote_exp_root / "fair_demo_20260715"
    for row in fair["candidates"]:
        method = row["method"]
        if method not in METHODS:
            continue
        candidates.append({
            "id": row["id"], "method": method, "stage": 1, "optimizer_steps": 900,
            "source": "frozen_fair_demo_two_config_grid",
            "model_path": str(fair_root / "sweep/candidates" / row["id"]),
            "validation_generation": str(args.remote_project / "results/p1_8b_fair_demo_20260715/validation/generations" / row["id"] / "output_42.json"),
            "config": row,
        })
    frozen_name_for_method = {row["method"]: row["name"] for row in frozen_rows if row["method"] != "base"}
    for method in METHODS:
        if method == "ronpo_os" or method not in frozen:
            continue
        row = frozen[method]
        candidates.append({
            "id": f"frozen_{row['name']}", "method": method, "stage": 1, "optimizer_steps": 900,
            "source": "public_flagship_frozen_checkpoint",
            "model_path": f"{row['model']}@{row['revision']}",
            "hf_repo": row["model"], "hf_revision": row["revision"],
            "validation_generation": str(args.remote_project / "results/p1_validation_reward_seed42_20260714/generations" / frozen_name_for_method[method] / "output_42.json"),
        })
    counts = {method: sum(row["method"] == method for row in candidates) for method in METHODS}
    if counts["ronpo_os"] != 2 or any(counts[method] > 3 for method in METHODS):
        raise RuntimeError(f"asymmetric candidate count: {counts}")
    grid = {
        "schema_version": 1, "status": "LOCKED_BEFORE_VALIDATION_LOCAL_RM_RANKING",
        "locked_at": now, "seed": 42, "stage": 1,
        "common_budget": {"optimizer_steps": 900, "effective_batch_size": 16},
        "candidate_count_by_method": counts,
        "fairness": "RONPO-OS has exactly two final checkpoints. Every non-OS method has at most three candidates (two frozen fair-demo configs plus one earlier public flagship checkpoint), so OS does not receive greater selection intensity.",
        "selection": "S3-passing candidate with highest validation min local-RM marginal win rate versus base; lexical id tie-break.",
        "candidates": candidates,
        "spent_sealed_split_touched": False,
    }
    grid_path = out / "sweep/candidate_grid.json"
    atomic_json(grid_path, grid)

    metric = {
        "schema_version": 1, "status": "LOCKED_BEFORE_ANY_STAGE1_OS_LOCAL_RM_RANKING",
        "locked_at": now, "stage": 1, "seed": 42,
        "objectives": RMS,
        "primary": {
            "name": "min_local_rm_marginal_win_rate_vs_base",
            "definition": "For each prompt and RM, candidate minus base raw score is converted to win=1, exact tie=0.5, loss=0. Average over prompts within each RM, then take the minimum over Skywork, Athene, and ArmoRM.",
            "direction": "higher_is_better", "base_floor": 0.5, "tie_threshold": 0.0,
        },
        "validation_selection": {
            "split": "existing prompt-disjoint 128-prompt validation",
            "rule": "Within method choose the S3-passing candidate with highest primary; lexical candidate id breaks exact ties.",
        },
        "stage1_test": {
            "file": str(args.eval_file.resolve()), "sha256": sha256(args.eval_file),
            "prompt_count": count_records(args.eval_file), "role": "fixed_647_prompt_local_rm_test_not_used_for_selection",
        },
        "decode": {"seed": 42, "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 2048,
                   "dtype": "bfloat16", "enable_thinking": False},
        "bootstrap": {"resamples": 2000, "seed": 42, "unit": "prompt", "paired": True,
                      "interval": "percentile_95", "recompute_min_after_each_resample": True},
        "secondary": [
            "per-RM raw paired deltas versus base with CIs",
            "per-RM marginal win rates with CIs",
            "mean marginal win rate",
            "cross-objective marginal-win-rate spread",
            "mean and minimum per-objective prompt-minmax normalized scores for continuity only",
        ],
        "independent_confirmation": {
            "judges": [
                {"model": "openai/gpt-oss-120b", "revision": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"},
                {"model": "Qwen/Qwen3-32B", "revision": "9216db5781bf21249d130ec9da846c4624c16137"},
            ],
            "position_swap": True, "score": "win=1,tie=0.5,loss=0",
            "primary": "minimum over objectives of per-objective marginal panel win rate versus base",
            "fresh_split": "new prompt-disjoint manifest locked after model selection and measured once",
        },
        "phase_a_decision": {
            "PASS": "Validation-selected RONPO-OS is strictly highest among all eligible methods on the fixed 647-prompt local-RM primary and on the one-shot fresh independent-panel primary, and its fresh panel primary is at least 0.5.",
            "PARTIAL": "RONPO-OS is strictly highest among eligible trained methods on both signals but its fresh panel primary is below 0.5.",
            "FAIL": "Any eligible trained baseline ties or exceeds RONPO-OS on either confirmatory primary, or RONPO-OS fails stability.",
        },
        "stage2_gate": "Phase B may start only after Phase A FAIL and only if every method has an eligible Stage-1 parent and the complete symmetric Stage-2 plus a new one-shot confirmation can finish by 21:00 KST.",
        "spent_sealed_split_touched": False,
    }
    metric_path = out / "metric_lock.json"
    atomic_json(metric_path, metric)
    (out / "metric_lock.sha256").write_text(f"{sha256(metric_path)}  metric_lock.json\n", encoding="utf-8")
    prereg = f"""# Qwen3-8B RONPO-OS Stage-1 and conditional Stage-2 preregistration

Locked at `{now}`, before computing the new Stage-1 OS local-RM ranking.

## Primary

For each of Skywork, Athene, and ArmoRM, compare each response's raw score with the base response on the same prompt. A win is 1, an exact tie is 0.5, and a loss is 0. The prompt mean is computed separately for each RM; the primary is the minimum of those three marginal win rates. The paired prompt bootstrap uses 2,000 resamples and seed 42 and recomputes the minimum in every resample.

Prompt-level min-max normalized reward is retained only as a continuity diagnostic because it depends on the evaluated model pool. Raw paired deltas, all per-RM win rates, their spread, and normalized metrics are reported without selection on them.

## Candidate fairness

RONPO-OS has two eligible 900-step configurations fixed in advance: `r1_os_stratified_k002` and `r2_os_anneal_anchor030`, both at their final step 900. Every other method receives its two-config fair-demo pool plus its earlier frozen flagship checkpoint when present, so OS has no greater checkpoint-selection budget. All candidates must pass the unchanged corrected stability gate on the common 128-prompt validation generation.

## Decision and Stage-2

Models are selected on validation only. The fixed 647-prompt local-RM test is not used for selection. A new disjoint independent-judge split is generated and hashed only after selection and is opened once. PASS, PARTIAL, and FAIL are defined byte-for-byte in `metric_lock.json`. Phase B is allowed only after a Phase-A FAIL and only if every method can advance symmetrically from an eligible Stage-1 parent before the deadline. A partial Stage-2 comparison is never shipped.

The spent 604-prompt sealed split is not read or reused.
"""
    (out / "PREREG.md").write_text(prereg, encoding="utf-8")
    atomic_json(out / "prereg_lock.json", {
        "status": "LOCKED_BEFORE_RANKING", "metric_lock_sha256": sha256(metric_path),
        "candidate_grid_sha256": sha256(grid_path), "prereg_sha256": sha256(out / "PREREG.md"),
        "spent_sealed_split_touched": False,
    })
    print(json.dumps({"metric_lock_sha256": sha256(metric_path), "candidate_grid_sha256": sha256(grid_path),
                      "candidate_count_by_method": counts}, indent=2))


if __name__ == "__main__":
    main()
