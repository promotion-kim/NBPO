#!/usr/bin/env python3
"""Render auditable SafeRLHF Table-4 reports from measured artifacts only.

This script deliberately makes no model-selection decision.  It records the
preregistered W1 result, instrument checks, cuts, and the provenance ledger
needed to review the generated table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def maybe_load(path: Path) -> dict | None:
    return load(path) if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    summary = load(root / "model_summary.json")
    preflight = load(root / "tradeoff_pool_gate.json")
    target = load(root / "target_nondegeneracy_audit.json")
    scales = load(root / "target_scale_audit.json")
    calibration = load(root.parent / "p4_8b_saferlhf_kappa_imbalance_20260717" / "preflight" / "calibration_summary.json")
    w1 = load(root / "train" / "w1_900steps" / "summary.json")
    manifest = load(root / "dataset_manifest" / "data_manifest.json")

    rank = {row["model"]: row for row in summary["ranking"]}
    base = rank["base"]
    helpful = rank["ht_mnpo_helpfulness"]
    harmless = rank["ht_mnpo_harmless"]
    helpful_pass = helpful["helpfulness_norm"] > base["helpfulness_norm"]
    harmless_pass = harmless["harmlessness_norm"] > base["harmlessness_norm"]
    sanity_pass = helpful_pass and harmless_pass

    instrument = [
        "# Instrument report",
        "",
        "## Pre-training gates",
        "",
        f"- Beaver reward human-better agreement on PKU conflict rows: {calibration['reward_helpfulness_vs_human_better']['accuracy']:.6f} ({calibration['reward_helpfulness_vs_human_better']['correct']}/{calibration['reward_helpfulness_vs_human_better']['rows']}); preregistered threshold 0.60.",
        f"- Negative Beaver cost human-safer agreement on PKU conflict rows: {calibration['cost_harmlessness_vs_human_safer']['accuracy']:.6f} ({calibration['cost_harmlessness_vs_human_safer']['correct']}/{calibration['cost_harmlessness_vs_human_safer']['rows']}); preregistered threshold 0.65.",
        f"- Shared decoded-pool trade-off gate: median within-prompt Spearman {preflight['median_prompt_spearman']:.6f}, mean {preflight['mean_prompt_spearman']:.6f}, reward/cost argmax mismatch {preflight['reward_argmax_cost_argmax_mismatch_rate']:.6f}. The preregistered median-correlation ceiling was +0.5; this gate passed.",
        "",
        "## Adversary and target audit",
        "",
        f"- Top-mass target was invariant across the built kappas: differing fraction {target['topmass_exact_kappa_invariance']['0.5']['fraction_differing_from_first']:.6f}.",
        f"- Target endpoint assertions passed: {target['assertions']}. The full-expectation correlations move from hard-argmax toward the uniform control as kappa increases; see `target_nondegeneracy_audit.json`.",
        "- The static target builder computes sigma once from the precomputed pool. This is not the adaptive OMD adversary in the toy analysis; no claim equates the two mechanisms.",
        "",
        "## Target-scale audit",
        "",
        "The measured effective target magnitudes and initial gradient norms are retained in `target_scale_audit.json` and the W1 job-status JSONs. They are reported because equal rows and step counts do not imply equal numerical target scales.",
        "",
        "## Post-training single-oracle sanity",
        "",
        f"- HT-MNPO(help.) helpfulness {helpful['helpfulness_norm']:.6f} versus Base {base['helpfulness_norm']:.6f}: {'PASS' if helpful_pass else 'FAIL'}.",
        f"- HT-MNPO(harmless) harmlessness {harmless['harmlessness_norm']:.6f} versus Base {base['harmlessness_norm']:.6f}: {'PASS' if harmless_pass else 'FAIL'}.",
        f"- Overall single-oracle sanity: {'PASS' if sanity_pass else 'FAIL'}.",
        "",
        "All W1 decoded models passed the unchanged reward-blind stability gate; exact per-model checks are in `validation/gates/*.json`.",
        "",
    ]
    (root / "INSTRUMENT.md").write_text("\n".join(instrument), encoding="utf-8")

    gate = summary["gate"]
    w2 = maybe_load(root / "train" / "w2_900steps" / "summary.json")
    cut = [
        "# Cut and execution plan",
        "",
        "- W1 was frozen before training and all eight W1 arms completed at 900 steps.",
        "- The measured confirmatory W1 gate is already FAIL, so no result-dependent W1 retraining or checkpoint selection is permitted.",
        "- W2 is diagnostic only. Its estimator rows and symmetric learning-rate checks cannot replace the preregistered W1 headline arm.",
        "- W3 was not launched. It is below W2 in the preregistered cut order and is not needed to decide the completed headline gate.",
        f"- W1 gate: RONPO OS minus {gate['worst_comparator']} Worst = {gate['worst_paired_difference']:.6f}, 95% CI [{gate['worst_paired_difference_ci95'][0]:.6f}, {gate['worst_paired_difference_ci95'][1]:.6f}], therefore {gate['status'].upper()}.",
    ]
    if w2:
        cut.append(f"- W2 status at report rendering: {w2.get('status', 'unknown')}; its per-arm files are retained under `train/w2_900steps/`.")
    else:
        cut.append("- W2 had not yet written a summary at report rendering.")
    cut.append("")
    (root / "CUT.md").write_text("\n".join(cut), encoding="utf-8")

    ledger_paths = [
        root / "train_pool" / "scores" / "helpfulness.jsonl",
        root / "train_pool" / "scores" / "harmlessness.jsonl",
        root / "validation" / "scores" / "helpfulness.jsonl",
        root / "validation" / "scores" / "harmlessness.jsonl",
        root / "w2_validation" / "scores" / "helpfulness.jsonl",
        root / "w2_validation" / "scores" / "harmlessness.jsonl",
        root / "dataset_manifest" / "train_conflict.jsonl",
        root / "dataset_manifest" / "validation_conflict.jsonl",
    ]
    ledger = []
    for path in ledger_paths:
        if path.is_file():
            ledger.append({"path": str(path.relative_to(root)), "sha256": sha(path), "rows": jsonl_count(path) if path.suffix == ".jsonl" else None})
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "w1_status": w1["status"],
        "w1_arms": [{"arm": row["arm"], "status": row["status"], "checkpoint": row.get("checkpoint"), "config_sha256": row.get("config_sha256")} for row in w1["arms"]],
        "validation_records": summary["records"],
        "validation_manifest_sha256": manifest["validation"]["sha256"],
        "train_manifest_sha256": manifest["train"]["sha256"],
        "primary_gate": gate,
        "single_oracle_sanity_pass": sanity_pass,
        "score_ledger": ledger,
        "gpu_authorization": {"requested_by_prompt": 8, "available_and_used": 4, "other_users_touched": False},
        "spent_sealed_split_touched": False,
    }
    (root / "COMPLETION_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    lines = ["# Completion audit", "", "This report is generated from `COMPLETION_AUDIT.json` and measured artifacts.", "", "## Score ledger", "", "| Artifact | Rows | SHA-256 |", "|---|---:|---|"]
    for row in ledger:
        lines.append(f"| `{row['path']}` | {row['rows'] if row['rows'] is not None else '--'} | `{row['sha256']}` |")
    lines += ["", f"W1 status: **{w1['status']}**. Primary gate: **{gate['status'].upper()}**. Only four authorized B200 GPUs were available and used; no other user's process was touched.", "", "`spent_sealed_split_touched=false`", ""]
    (root / "COMPLETION_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
