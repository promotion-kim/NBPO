#!/usr/bin/env python3
"""Log only measured confirmatory S3 failures; never synthesize reward metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import wandb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.diagnostics.read_text())
    if payload.get("status") != "failed_closed" or payload.get("reward_scoring_started") is not False:
        raise RuntimeError("expected fail-closed diagnostics with no reward scores")
    identifier = hashlib.sha256(
        f"aaai27|p1-corrected-confirmatory-failed|{payload['prompt_file_sha256']}".encode()
    ).hexdigest()[:12]
    run = wandb.init(
        entity="promotion-kim", project="mnpo", id=identifier, resume="allow",
        name="aaai27-p1-corrected-confirmatory-stability-failed-s42",
        group="ronpo-aaai27-revision", job_type="reward_eval_gate",
        config={
            "stage": payload["stage"], "seed": 42, "paid_api": False,
            "prompt_count": payload["prompt_count"],
            "prompt_file_sha256": payload["prompt_file_sha256"],
            "reward_scoring_started": False, "bootstrap_started": False,
            "source_json": str(args.diagnostics),
        },
    )
    metrics = {"gate/all_passed": 0, "gate/model_count": len(payload["stability_rows"])}
    for row in payload["stability_rows"]:
        prefix = f"gate/{row['model']}"
        metrics.update({
            f"{prefix}/passed": int(row["status"] == "passed"),
            f"{prefix}/records": row["records"],
            f"{prefix}/empty_count": row["empty_count"],
            f"{prefix}/think_leak_count": row["think_leak_count"],
            f"{prefix}/mean_word_ratio_vs_base": row["mean_word_ratio_vs_base"],
            f"{prefix}/max_repeat_run": row["max_repeat_run"],
        })
    run.log(metrics)
    run.summary.update(metrics)
    run.summary["terminal_status"] = "failed_closed"
    run.summary["reward_metrics_measured"] = False
    run.finish()
    output = {
        "status": "completed", "kind": "corrected_confirmatory_stability_failure_audit",
        "reward_metrics_measured": False, "wandb_run_id": identifier,
        "wandb_url": f"https://wandb.ai/promotion-kim/mnpo/runs/{identifier}",
        "source_json": str(args.diagnostics),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
