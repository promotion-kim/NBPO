#!/usr/bin/env python3
"""Lock the seed-42 RONPO choice using validation evidence only.

This program deliberately has no sealed-prompt argument and never reads the
sealed dataset.  Its immutable output is the sole GO signal for the sealed
reward runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path


METRIC = "mean_primary_prompt_worst_norm_score"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_ledger(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {
        row["name"]: {"repo_id": row["model"], "revision": row["revision"],
                      "upload_commit": row.get("upload_commit") or None,
                      "seed": int(row["seed"]) if row.get("seed") else None}
        for row in rows
    }


def ranking_row(summary: dict, model: str) -> dict:
    rows = summary.get("ranked", summary.get("ranking", []))
    found = [row for row in rows if row.get("model") == model]
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one validation row for {model}, found {len(found)}")
    row = found[0]
    value = float(row[METRIC])
    if not math.isfinite(value) or int(row.get("num_prompts", 0)) != 128:
        raise RuntimeError(f"invalid 128-prompt validation measurement for {model}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--models-tsv", type=Path, required=True)
    parser.add_argument("--p3-protocol", type=Path, required=True)
    parser.add_argument("--p3-work", type=Path, required=True)
    parser.add_argument("--ifeval-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise RuntimeError(f"selection is already locked; refusing to overwrite {args.output}")
    validation = read_json(args.validation_summary)
    if validation.get("p1_sealed_test_opened") is not False:
        raise RuntimeError("validation artifact does not certify p1_sealed_test_opened=false")
    protocol = read_json(args.p3_protocol)
    if protocol.get("p1_sealed_test_opened") is not False:
        raise RuntimeError("P3 protocol is not sealed-safe")
    p3_status_path = args.p3_work / "status.json"
    p3_status = read_json(p3_status_path)
    if p3_status.get("p1_sealed_test_opened") is not False:
        raise RuntimeError("P3 status does not certify p1_sealed_test_opened=false")
    terminal = p3_status.get("status") in {"completed", "completed_no_eligible_candidate"}
    selection_ready = (
        p3_status.get("status") == "running"
        and p3_status.get("stage") == "sweep_candidate_ifeval"
        and (args.p3_work / "validation_results/ranked_validation_summary.json").is_file()
        and not p3_status.get("failures")
    )
    if not (terminal or selection_ready):
        raise RuntimeError(
            f"P3 validation selection is not ready: status={p3_status.get('status')} "
            f"stage={p3_status.get('stage')}"
        )

    ledger = model_ledger(args.models_tsv)
    ifeval_payload = read_json(args.ifeval_json)
    ifeval = {row["method"]: float(row["ifeval_prompt_strict_percent"])
              for row in ifeval_payload["rows"]}
    top_mass = ranking_row(validation, "ronpo_k_only")
    options = [{
        "model": "ronpo_k_only",
        "variant": "top-mass",
        "validation_score": float(top_mass[METRIC]),
        "ifeval_percent": ifeval["ronpo_k_only"],
        "s3_passed": True,
        "validation_prompts": 128,
        "model_identity": ledger["ronpo_k_only"],
        "evidence": {"validation_summary": str(args.validation_summary),
                     "models_tsv": str(args.models_tsv),
                     "ifeval_json": str(args.ifeval_json)},
    }]
    excluded = []
    eligible = p3_status.get("eligible", {})
    combined_path = args.p3_work / "validation_results/ranked_validation_summary.json"
    combined = read_json(combined_path) if combined_path.is_file() else None
    selection_metrics_path = args.p3_work / "candidate_selection_metrics.json"
    selection_metrics = read_json(selection_metrics_path) if selection_metrics_path.is_file() else {}
    option_metrics = {row["model"]: row for row in selection_metrics.get("options", [])}
    final_path = args.p3_work.parent / "p1_validation_reward_seed42/final_model_selection.json"
    final_selection = read_json(final_path) if final_path.is_file() else {}

    for candidate in protocol.get("candidates", {}):
        prefixed = f"ronpo_sweep_{candidate}"
        metadata_path = args.p3_work / "candidates" / candidate / "candidate_status.json"
        metadata = eligible.get(candidate)
        if metadata is None and metadata_path.is_file():
            metadata = read_json(metadata_path)
        gate_path = args.p3_work / "candidates" / candidate / "stability/gate.json"
        if not metadata or not gate_path.is_file():
            excluded.append({"candidate": candidate, "reason": "not S3-eligible"})
            continue
        gate = read_json(gate_path)
        if gate.get("passed") is not True or metadata.get("s3_passed") is not True:
            excluded.append({"candidate": candidate, "reason": "frozen S3 failed"})
            continue
        if combined is None:
            excluded.append({"candidate": candidate, "reason": "same-pipeline validation absent"})
            continue
        try:
            row = ranking_row(combined, prefixed)
        except RuntimeError as error:
            excluded.append({"candidate": candidate, "reason": str(error)})
            continue
        metric_row = option_metrics.get(prefixed, {})
        candidate_ifeval = metric_row.get("ifeval_percent")
        if candidate_ifeval is not None:
            candidate_ifeval = float(candidate_ifeval)
            if not math.isfinite(candidate_ifeval):
                candidate_ifeval = None
        if final_selection.get("selected_model_name") == prefixed:
            identity = final_selection.get("selected_model")
        else:
            identity = None
        options.append({
            "model": prefixed,
            "variant": candidate,
            "validation_score": float(row[METRIC]),
            "ifeval_percent": candidate_ifeval,
            "s3_passed": True,
            "validation_prompts": 128,
            "model_identity": identity,
            "evidence": {"validation_summary": str(combined_path),
                         "stability_gate": str(gate_path),
                         "candidate_status": str(metadata_path),
                         "candidate_metrics": str(selection_metrics_path)},
        })

    best_score = max(row["validation_score"] for row in options)
    tied = [row for row in options if abs(row["validation_score"] - best_score) <= 1e-12]
    if len(tied) > 1:
        if any(row["ifeval_percent"] is None for row in tied):
            raise RuntimeError("validation-score tie requires measured IFEval before selection can lock")
        tied.sort(key=lambda row: (-row["ifeval_percent"], row["model"]))
    selected = tied[0]
    options.sort(key=lambda row: (
        -row["validation_score"],
        -(row["ifeval_percent"] if row["ifeval_percent"] is not None else float("-inf")),
        row["model"],
    ))
    if selected["model_identity"] is None:
        raise RuntimeError(
            f"validation winner {selected['model']} is not yet in a verified public repository; "
            "upload it before locking selection"
        )
    payload = {
        "schema_version": 1,
        "status": "locked",
        "locked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_split": "non-sealed validation",
        "selection_metric": METRIC,
        "tie_break": "IFEval prompt-level strict percent",
        "selected_ronpo_variant": selected["variant"],
        "selected_model_name": selected["model"],
        "selected_model": selected["model_identity"],
        "selected_validation_score": selected["validation_score"],
        "selected_ifeval_percent": selected["ifeval_percent"],
        "validation_ranking_used": [
            {
                "model": row["model"],
                "rank": row.get("validation_worst_objective_rank"),
                "mean_primary_prompt_worst_norm_score": float(row[METRIC]),
                "ci95_low": float(row["mean_primary_prompt_worst_norm_score_ci95_low"]),
                "ci95_high": float(row["mean_primary_prompt_worst_norm_score_ci95_high"]),
                "num_prompts": int(row["num_prompts"]),
            }
            for row in validation.get("ranked", validation.get("ranking", []))
        ],
        "eligible_options": options,
        "excluded_p3_candidates": excluded,
        "source_artifacts": {
            "validation_summary": {"path": str(args.validation_summary), "sha256": sha256(args.validation_summary)},
            "p3_status": {"path": str(p3_status_path), "sha256": sha256(p3_status_path)},
            "p3_protocol": {"path": str(args.p3_protocol), "sha256": sha256(args.p3_protocol)},
            "models_tsv": {"path": str(args.models_tsv), "sha256": sha256(args.models_tsv)},
            "ifeval_json": {"path": str(args.ifeval_json), "sha256": sha256(args.ifeval_json)},
        },
        "selection_rule": "Highest eligible 128-prompt validation mean_prompt_worst_norm_score; exact ties broken by measured IFEval. The current top-mass estimator remains selected unless an eligible P3 candidate exceeds it.",
        "sealed_data_consulted_for_selection": False,
        "statement": "Selection was finalized exclusively on non-sealed validation evidence; no sealed prompt or sealed score was consulted.",
        "p1_sealed_test_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
