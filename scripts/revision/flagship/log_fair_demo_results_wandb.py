#!/usr/bin/env python3
"""Log the locked diagnostic and final measured fair-demo results to one reproducible W&B run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import wandb


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    diagnostic = json.loads((args.run_dir / "diagnostics/results/summary.json").read_text())
    official_fresh_path = args.run_dir / "fresh_test/results/panel/panel_summary.json"
    sensitivity_fresh_path = args.run_dir / "fresh_test/results/panel_schema_alias_sensitivity/panel_summary.json"
    fresh_path = official_fresh_path if official_fresh_path.is_file() else sensitivity_fresh_path
    fresh = json.loads(fresh_path.read_text())
    metric_prefix = ("fresh_confirmatory" if summary.get("confirmatory_primary_valid")
                     else "fresh_schema_alias_sensitivity")
    identifier = hashlib.sha256(b"qwen3-8b-fair-demo-final-eval-v1").hexdigest()[:12]
    run = wandb.init(entity="promotion-kim", project="mnpo", id=identifier, resume="allow",
                     name="qwen3-8b-fair-demo-final-eval", group="qwen3-8b-fair-demo-20260715",
                     config={"evaluator_lock_sha256": summary["artifact_hashes"]["evaluator_lock"],
                             "selection_lock_sha256": summary["artifact_hashes"]["selection_lock"],
                             "fresh_manifest_sha256": summary["artifact_hashes"]["fresh_manifest"],
                             "fresh_prompt_count": summary["fresh_prompt_count"],
                             "confirmatory_primary_valid": summary.get("confirmatory_primary_valid", False),
                             "spent_sealed_split_touched": False})
    metrics = {
        "diagnostic/panel_resolution_worst_delta": diagnostic["judge_panel"]["panel_resolution_worst_delta_mean"],
        "diagnostic/inter_judge_agreement": diagnostic["judge_panel"]["inter_judge_exact_label_agreement"],
        f"{metric_prefix}/position_agreement": fresh["position_score_agreement"],
        f"{metric_prefix}/inter_judge_agreement": fresh["inter_judge_mean_position_score_agreement"],
        "fresh/confirmatory_primary_valid": int(summary.get("confirmatory_primary_valid", False)),
        "fresh/strong_claim_supported": int(summary["strong_claim_supported"]),
    }
    if summary.get("selected_ronpo_result"):
        row = summary["selected_ronpo_result"]
        metrics.update({f"{metric_prefix}/selected_ronpo_worst": row["mean_prompt_worst_panel_score"],
                        f"{metric_prefix}/selected_ronpo_delta_vs_base": row["delta_vs_base_tie"]})
    for row in fresh["ranked"]:
        metrics[f"{metric_prefix}/model/{row['model']}/worst"] = row["mean_prompt_worst_panel_score"]
    run.log(metrics)
    artifact = wandb.Artifact("qwen3-8b-fair-demo-measured-results", type="evaluation")
    for path in [summary_path, args.run_dir / "REPORT.md",
                 args.run_dir / "validation/results/panel/panel_summary.json",
                 fresh_path,
                 args.run_dir / "fresh_test/results/rewards/reward_summary.json"]:
        artifact.add_file(str(path), name=str(path.relative_to(args.run_dir)))
    run.log_artifact(artifact)
    url = run.url
    run.finish()
    output = {"status": "completed", "run_id": identifier, "url": url,
              "summary_sha256": sha256(summary_path), "spent_sealed_split_touched": False}
    (args.run_dir / "wandb_eval_run.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
