#!/usr/bin/env python3
"""Render an evidence-backed report when the sealed run fails closed at S3."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    status = read(args.work / "status.json")
    lock = read(args.work / "selection_lock.json")
    opened = read(args.work / "sealed_opened.json")
    gate_summary = read(args.work / "stability_gates/summary.json")
    wandb_path = args.work / "wandb_failure_run.json"
    wandb = read(wandb_path) if wandb_path.is_file() else None
    if status.get("status") != "failed" or status.get("stage") != "sealed_stability_gates":
        raise RuntimeError("this report is only for a terminal fail-closed S3 run")

    rows = []
    for model, gate in gate_summary["models"].items():
        candidate = gate["candidate"]
        rows.append({
            "model": model,
            "records": int(candidate["records"]),
            "empty_count": int(candidate["empty_count"]),
            "think_leak_count": int(candidate["think_leak_count"]),
            "think_leak_indices": candidate["think_leak_indices"],
            "mean_word_ratio_vs_base": float(gate["candidate_base_mean_word_ratio"]),
            "max_repeat_run": int(candidate["max_repeat_run"]),
            "status": gate["status"],
        })
    diagnostics = {
        "status": "failed_closed",
        "stage": "sealed_stability_gates",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selected_ronpo_variant": lock["selected_ronpo_variant"],
        "selection_locked_at": lock["locked_at"],
        "sealed_opened_at": opened["opened_at"],
        "sealed_sha256": opened["sealed_sha256"],
        "stability_rows": rows,
        "reward_scoring_started": False,
        "bootstrap_started": False,
        "sealed_rank_measured": False,
        "paper_updated": False,
        "wandb_failure_audit": wandb,
        "reason": "At least one frozen stability check failed; the protocol requires fail-closed exclusion before reward scoring.",
    }
    out = args.work / "results"
    out.mkdir(exist_ok=True)
    (out / "failure_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")

    lines = [
        "# P1 sealed reward report — terminal fail-closed",
        "",
        f"Selection was locked on non-sealed validation at `{lock['locked_at']}` and selected "
        f"`{lock['selected_model_name']}` (`{lock['selected_ronpo_variant']}`).",
        f"The 604-prompt sealed test was opened once at `{opened['opened_at']}` with SHA-256 "
        f"`{opened['sealed_sha256']}`.",
        "",
        "All 11 models generated exactly 604 non-empty responses, but the frozen stability gate failed. "
        "The run therefore stopped before ArmoRM scoring, normalization, bootstrap, or ranking. "
        "No sealed reward number exists and the paper was not updated.",
        "",
        "| Model | Records | Empty | Think leaks (indices) | Mean-word ratio | Max repeat | Gate |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        indices = ",".join(str(value) for value in row["think_leak_indices"]) or "--"
        lines.append(
            f"| {row['model']} | {row['records']} | {row['empty_count']} | "
            f"{row['think_leak_count']} ({indices}) | {row['mean_word_ratio_vs_base']:.6f} | "
            f"{row['max_repeat_run']} | {row['status']} |"
        )
    lines.extend([
        "", "## Frozen thresholds", "",
        "- Records: exactly 604",
        "- Empty responses: 0",
        "- `<think>` / `</think>` leakage: 0",
        "- Mean-word ratio vs base: [0.33, 2.0]",
        "- Max consecutive identical word run: <= 20",
        "", "## Outcome", "",
        "- `ranked_sealed_summary.json`: not produced",
        "- `per_objective_scores.csv`: not produced",
        "- W&B sealed reward metric run: not created because no reward metric was measured",
        f"- W&B fail-closed gate audit: `{wandb['wandb_run_id']}` ({wandb['wandb_url']})"
        if wandb else "- W&B fail-closed gate audit: unavailable",
        "- RONPO sealed worst-objective rank: unknown",
        "- Paper table/figure: unchanged",
        "",
        "Primary evidence: `status.json`, `stability_gates/summary.json`, per-model gate JSON, "
        "decode metadata/logs, and the 11 preserved generation JSON files.",
        "",
    ])
    (out / "SEALED_REPORT.md").write_text("\n".join(lines))
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
