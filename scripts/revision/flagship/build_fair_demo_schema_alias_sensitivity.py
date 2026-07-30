#!/usr/bin/env python3
"""Build a non-confirmatory sensitivity analysis for a documented judge schema alias."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path


OBJECTIVES = ("helpfulness", "safety", "conciseness")
ALLOWED = {"a", "b", "tie"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalize(value: str) -> str:
    lowered = value.strip().lower()
    if lowered not in ALLOWED:
        raise ValueError(lowered)
    return lowered.upper() if lowered != "tie" else "tie"


def security_alias(raw: str) -> dict[str, str] | None:
    for candidate in reversed(re.findall(r"\{[^{}]*\}", raw, flags=re.S)):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if "safety" in value or "security" not in value:
            continue
        try:
            return {
                "helpfulness": normalize(str(value["helpfulness"])),
                "safety": normalize(str(value["security"])),
                "conciseness": normalize(str(value["conciseness"])),
            }
        except (KeyError, ValueError):
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    work = args.run_dir / "fresh_test"
    source = work / "judge/gpt_oss_120b"
    repaired = work / "judge/gpt_oss_120b_schema_alias_sensitivity"
    repaired.mkdir(parents=True, exist_ok=True)
    repaired_tasks = []
    original_rows = 0
    original_invalid = 0
    remaining_invalid = 0
    input_hashes = {}
    output_hashes = {}
    for input_path in sorted(source.glob("shard_*.jsonl")):
        output_path = repaired / input_path.name
        rows = []
        for line in input_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            original_rows += 1
            row["original_valid"] = bool(row.get("valid"))
            row["schema_alias_applied"] = False
            if not row["original_valid"]:
                original_invalid += 1
                parsed = security_alias(str(row.get("raw_judge_output", "")))
                if parsed is None:
                    remaining_invalid += 1
                else:
                    row["parsed"] = parsed
                    row["valid"] = True
                    row["schema_alias_applied"] = True
                    repaired_tasks.append(row["task_id"])
            rows.append(row)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        input_hashes[input_path.name] = sha256(input_path)
        output_hashes[output_path.name] = sha256(output_path)
    if original_rows == 0 or original_invalid == 0:
        raise RuntimeError("no invalid original rows found for the sensitivity analysis")
    if remaining_invalid or len(repaired_tasks) != original_invalid:
        raise RuntimeError(
            f"schema alias repair is incomplete: original_invalid={original_invalid}, "
            f"repaired={len(repaired_tasks)}, remaining={remaining_invalid}"
        )
    audit = {
        "status": "completed_nonconfirmatory_protocol_deviation",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rule": "Only when safety is absent, map a complete JSON key named security to safety; preserve its measured A/B/tie value.",
        "original_rows": original_rows,
        "original_invalid": original_invalid,
        "schema_alias_repairs": len(repaired_tasks),
        "remaining_invalid": remaining_invalid,
        "repaired_task_ids": sorted(repaired_tasks),
        "original_input_sha256": input_hashes,
        "sensitivity_output_sha256": output_hashes,
        "confirmatory_primary_valid": False,
        "sensitivity_only": True,
        "strong_claim_authorized": False,
        "reason": "The evaluator lock required the unchanged parser and 100% validity; this post-lock alias was not preregistered.",
        "spent_sealed_split_touched": False,
    }
    audit_path = work / "judge/schema_alias_sensitivity_audit.json"
    atomic_json(audit_path, audit)
    output = work / "results/panel_schema_alias_sensitivity"
    command = [
        args.python, str(args.project / "scripts/revision/flagship/aggregate_fair_demo_panel.py"),
        "--input", str(work / "judge/input_tasks.jsonl"),
        "--input-lock", str(work / "judge/input_lock.json"),
        "--evaluator-lock", str(args.run_dir / "evaluator_lock.json"),
        "--judge-dir", f"qwen3_32b={work / 'judge/qwen3_32b'}",
        "--judge-dir", f"gpt_oss_120b={repaired}",
        "--gates", str(work / "stability_gates/summary.json"),
        "--output-dir", str(output),
    ]
    log = work / "logs/aggregate_panel_schema_alias_sensitivity.log"
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(command, cwd=args.project, stdout=handle, stderr=subprocess.STDOUT, check=True)
    summary_path = output / "panel_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({
        "confirmatory_primary_valid": False,
        "sensitivity_only": True,
        "strong_claim_authorized": False,
        "protocol_deviation": audit["reason"],
        "schema_alias_sensitivity_audit_sha256": sha256(audit_path),
    })
    atomic_json(summary_path, summary)
    print(json.dumps({"audit": audit, "panel_summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
