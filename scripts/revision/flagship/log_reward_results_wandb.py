#!/usr/bin/env python3
"""Log measured reward summaries to the mandatory W&B project."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import wandb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text())
    rows = payload.get("ranked", [])
    identifier = hashlib.sha256(f"aaai27|{args.stage}|seed42".encode()).hexdigest()[:12]
    run = wandb.init(
        entity="promotion-kim", project="mnpo", id=identifier, resume="allow",
        name=f"aaai27-{args.stage}-s42", group="ronpo-aaai27-revision",
        job_type="reward_eval",
        config={"stage": args.stage, "seed": 42, "source_json": str(args.summary),
                "sealed": "sealed" in args.stage, "paid_api": False},
    )
    values = {}
    for row in rows:
        method = row["model"]
        for key in (
            "mean_primary_prompt_worst_norm_score", "mean_primary_prompt_avg_norm_score",
            "mean_objective_norm_score", "min_objective_norm_score",
        ):
            if key in row:
                values[f"{method}/{key}"] = float(row[key])
        rank = row.get("worst_objective_rank", row.get("validation_worst_objective_rank"))
        if rank is not None:
            values[f"{method}/worst_objective_rank"] = int(rank)
    run.log(values)
    run.summary.update(values)
    run.finish()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "status": "completed", "wandb_run_id": identifier,
        "wandb_url": f"https://wandb.ai/promotion-kim/mnpo/runs/{identifier}",
        "stage": args.stage, "source_json": str(args.summary),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
