#!/usr/bin/env python3
"""Build an outcome-blind audit of the sealed stability-gate correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


METHODS = (
    "base", "ronpo_full_expect", "ronpo_k_only", "dpo", "ipo", "simpo",
    "sppo_avg", "inpo_avg", "ht_mnpo_helpfulness", "ht_mnpo_safety",
    "ht_mnpo_conciseness",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def tag_excerpt(raw: str) -> dict:
    lowered = raw.lower()
    positions = [position for position in (lowered.find("<think>"), lowered.find("</think>")) if position >= 0]
    if not positions:
        return {"first_tag_offset": None, "excerpt": ""}
    position = min(positions)
    return {
        "first_tag_offset": position,
        "excerpt": raw[max(0, position - 160):position + 180],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_dir = args.work / "stability_gates"
    new_dir = args.work / "stability_gates_corrected"

    forbidden = [
        args.work / "scores",
        args.work / "merged_generations.json",
        args.work / "results/model_summary.json",
        args.work / "results/ranked_sealed_summary.json",
        args.work / "results/per_objective_scores.csv",
    ]
    present_reward_artifacts = [str(path) for path in forbidden if path.exists()]
    if present_reward_artifacts:
        raise RuntimeError(
            "reward artifacts existed before the gate correction audit: "
            + ", ".join(present_reward_artifacts)
        )

    old_summary = read_json(old_dir / "summary.json")
    new_summary = read_json(new_dir / "summary.json")
    before_after = {}
    for method in METHODS:
        old = old_summary["models"][method]
        new = new_summary["models"][method]
        before_after[method] = {
            "before": {
                "status": old["status"],
                "checks": old["checks"],
                "think_leak_count": old["candidate"]["think_leak_count"],
                "think_leak_indices": old["candidate"]["think_leak_indices"],
                "max_repeat_run": old["candidate"]["max_repeat_run"],
            },
            "after": {
                "status": new["status"],
                "checks": new["checks"],
                "think_leak_count": new["candidate"]["think_leak_count"],
                "think_leak_indices": new["candidate"]["think_leak_indices"],
                "think_tag_artifacts": new["candidate"]["think_tag_artifacts"],
                "max_repeat_run": new["candidate"]["max_repeat_run"],
                "max_repeat_evidence": new["candidate"]["max_repeat_evidence"],
            },
        }

    index_394 = {}
    for method in METHODS:
        row = read_json(args.work / "generations" / method / "output_42.json")[394]
        raw = str(row.get("generated_text_raw") or row.get("generated_text") or "")
        index_394[method] = {
            "prompt_sha256": hashlib.sha256(str(row.get("prompt", "")).encode()).hexdigest(),
            "raw_generation_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "opening_think_tags": raw.lower().count("<think>"),
            "closing_think_tags": raw.lower().count("</think>"),
            **tag_excerpt(raw),
        }

    dpo_row = read_json(args.work / "generations/dpo/output_42.json")[252]
    dpo_clean = str(dpo_row.get("generated_text") or "")
    dpo_gate = new_summary["models"]["dpo"]["candidate"]
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "decision_timing": "Finalized before any sealed reward scoring or reward artifact existed.",
        "decision_basis": (
            "The correction was decided by inspecting generation text and tag structure only; "
            "no reward score was computed or consulted."
        ),
        "original_rule": (
            "Count a record as leaking if lower(generated_text_raw) contains either "
            "the substring <think> or the substring </think>."
        ),
        "corrected_rule": (
            "Count a record as leaking only if a case-insensitive <think>...</think> "
            "pair contains non-whitespace body text. A lone opening tag, lone closing "
            "tag, or empty paired span is recorded as a template artifact and is not a leak."
        ),
        "unchanged_thresholds": new_summary["thresholds"],
        "reward_artifacts_present_before_correction": present_reward_artifacts,
        "before_after": before_after,
        "evidence": {
            "index_394": {
                "description": "Per-model tag counts and text excerpts for the adversarial coax prompt.",
                "models": index_394,
            },
            "dpo_index_252": {
                "prompt_sha256": hashlib.sha256(str(dpo_row.get("prompt", "")).encode()).hexdigest(),
                "generation_sha256": hashlib.sha256(dpo_clean.encode()).hexdigest(),
                "max_repeat_run": dpo_gate["max_repeat_run"],
                "max_repeat_evidence": dpo_gate["max_repeat_evidence"],
            },
        },
        "original_gate_artifacts_preserved": {
            path.name: sha256(path) for path in sorted(old_dir.glob("*.json"))
        },
        "generation_artifacts": {
            method: {
                "path": str(args.work / "generations" / method / "output_42.json"),
                "sha256": sha256(args.work / "generations" / method / "output_42.json"),
                "records": len(read_json(args.work / "generations" / method / "output_42.json")),
            }
            for method in METHODS
        },
        "go_signal_for_reward_scoring": all(
            new_summary["models"][method]["passed"] is True
            for method in METHODS if method != "dpo"
        ) and new_summary["models"]["dpo"]["passed"] is False,
        "scoring_eligibility": {
            method: new_summary["models"][method]["passed"] for method in METHODS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
