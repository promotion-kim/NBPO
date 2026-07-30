#!/usr/bin/env python3
"""Regenerate the P5 Stage-2 training and gate status report from JSON.

This deliberately does not compute or infer reward scores.  A Stage-2 reward
comparison is eligible only when at least one RONPO Stage-2 arm has passed the
unchanged reward-blind stability gate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path


ARMS = [
    "ronpo_os_stage2",
    "ronpo_topmass_stage2",
    "inpo_avg_stage2",
    "simpo_stage2",
    "ipo_stage2",
    "dpo_stage2",
    "sppo_avg_stage2",
    "ht_mnpo_harmless_stage2",
    "ht_mnpo_helpfulness_stage2",
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    rows: list[dict] = []
    for arm in ARMS:
        root = experiment / "stage2" / arm
        job_path = root / "train" / "full" / "job_status.json"
        gate_path = root / "stability_validation" / "gate.json"
        job = load(job_path) if job_path.is_file() else None
        gate = load(gate_path) if gate_path.is_file() else None
        row = {
            "arm": arm,
            "train_status": job.get("status") if job else "missing",
            "steps": job.get("steps") if job else None,
            "effective_batch": job.get("effective_batch") if job else None,
            "seed": job.get("seed") if job else None,
            "finite_metrics": job.get("finite_metrics") if job else None,
            "wandb_run_id": job.get("wandb_run_id") if job else None,
            "wandb_url": job.get("wandb_url") if job else None,
            "gate_status": gate.get("status") if gate else "not_run",
            "gate_passed": gate.get("passed") if gate else False,
            "length_ratio": gate.get("candidate_base_mean_word_ratio") if gate else None,
            "gate_checks": gate.get("checks") if gate else None,
            "job_status_json": str(job_path),
            "gate_json": str(gate_path),
        }
        rows.append(row)
    ronpo = [row for row in rows if row["arm"].startswith("ronpo_")]
    eligible_for_reward_eval = any(row["gate_passed"] for row in ronpo)
    result = {
        "status": "completed",
        "scope": "Stage-2 training plus reward-blind corrected stability gates only. No Stage-2 reward ranking was computed.",
        "primary_reward_evaluation": {
            "status": "not_run" if not eligible_for_reward_eval else "pending",
            "reason": (
                "No RONPO Stage-2 arm passed the unchanged stability gate; a RONPO-versus-baseline "
                "Stage-2 reward comparison would be invalid."
                if not eligible_for_reward_eval else "At least one RONPO Stage-2 arm passed; evaluation remains pending."
            ),
        },
        "arms": rows,
        "spent_sealed_split_touched": False,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    output_json = experiment / "stage2" / "stage2_training_gate_summary.json"
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# P5 Stage-2 training and stability status",
        "",
        result["scope"],
        "",
        "| Arm | Train | Gate | Length ratio | W&B run |",
        "|---|---|---|---:|---|",
    ]
    for row in rows:
        ratio = "—" if row["length_ratio"] is None else f"{row['length_ratio']:.4f}"
        run = "—" if not row["wandb_url"] else f"[{row['wandb_run_id']}]({row['wandb_url']})"
        lines.append(f"| {row['arm']} | {row['train_status']} | {row['gate_status']} | {ratio} | {run} |")
    lines += ["", "## Reward evaluation decision", "", result["primary_reward_evaluation"]["reason"], ""]
    (experiment / "stage2" / "STAGE2_TRAINING_GATE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary": str(output_json), "reward_evaluation": result["primary_reward_evaluation"]}, indent=2))


if __name__ == "__main__":
    main()
