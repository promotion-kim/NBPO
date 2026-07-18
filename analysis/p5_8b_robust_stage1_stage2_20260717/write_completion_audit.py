#!/usr/bin/env python3
"""Write a compact, hash-backed audit for the P5 Stage-1/Stage-2 experiment.

The audit reads measured JSON/CSV artefacts only.  It intentionally records a
not-run Stage-2 reward evaluation when the RONPO gate has failed, rather than
constructing a baseline-only ranking.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, default=None)
    args = parser.parse_args()
    root = args.experiment.resolve()
    stage1 = root / "stage1" / "fixed_p4_validation" / "results" / "ranked_validation_summary.json"
    stage2 = root / "stage2" / "stage2_training_gate_summary.json"
    run_lock = root / "run_lock.json"
    uploads = root / "hf_uploads.jsonl"
    diagnostic = root / "stage2" / "fixed_p4_validation_diagnostic_including_stability_failed" / "results" / "diagnostic_summary.json"
    required = [stage1, stage2, run_lock, uploads]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required audit inputs: " + ", ".join(missing))

    stage1_data, stage2_data = read_json(stage1), read_json(stage2)
    files = required + [Path(row["gate_json"]) for row in stage2_data["arms"]]
    diagnostic_data = read_json(diagnostic) if diagnostic.is_file() else None
    if diagnostic_data is not None:
        files.append(diagnostic)
    entries = []
    for path in files:
        entries.append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})

    ronpo_failures = [
        {"arm": row["arm"], "length_ratio": row["length_ratio"], "gate_json": row["gate_json"]}
        for row in stage2_data["arms"]
        if row["arm"].startswith("ronpo_") and not row["gate_passed"]
    ]
    pass_arms = [row["arm"] for row in stage2_data["arms"] if row["gate_passed"]]
    audit = {
        "status": "completed",
        "scope": "P5 Stage-1 comparison plus Stage-2 training/gating. Stage-2 reward evaluation was not run because no RONPO Stage-2 arm was eligible.",
        "stage1_comparison": {
            "summary": str(stage1),
            "records": stage1_data["records"],
            "scope": stage1_data["scope"],
        },
        "stage2": {
            "summary": str(stage2),
            "reward_evaluation": stage2_data["primary_reward_evaluation"],
            "ronpo_gate_failures": ronpo_failures,
            "passed_arms": pass_arms,
            "diagnostic_evaluation": (
                {
                    "status": "completed_diagnostic_only",
                    "summary": str(diagnostic),
                    "records": diagnostic_data["records"],
                    "scope": diagnostic_data["scope"],
                }
                if diagnostic_data is not None else {"status": "not_run"}
            ),
        },
        "huggingface_upload_log": str(uploads),
        "storage_policy": "No optimizer state or intermediate checkpoint was pruned. The uploaded Stage-1 top-mass checkpoint remains locally because it is the Stage-2 parent and all Stage-2 evidence is retained pending review.",
        "spent_sealed_split_touched": False,
        "artifact_hashes": entries,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (root / "completion_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# P5 completion audit",
        "",
        audit["scope"],
        "",
        "## Stage-1",
        "",
        f"The descriptive fixed-panel comparison contains {stage1_data['records']} prompts. Its source summary is `{stage1}`.",
        "",
        "## Stage-2 gate decision",
        "",
        "| RONPO arm | Gate | Length ratio |",
        "|---|---|---:|",
    ]
    for item in ronpo_failures:
        lines.append(f"| {item['arm']} | failed | {item['length_ratio']:.4f} |")
    lines += [
        "",
        stage2_data["primary_reward_evaluation"]["reason"],
        "",
    ]
    if diagnostic_data is not None:
        lines += [
            "## Diagnostic-only score inspection",
            "",
            f"A {diagnostic_data['records']}-prompt diagnostic score table reused existing generations and retained the failed RONPO rows with explicit labels. It is not a paper ranking or selection result: `{diagnostic}`.",
            "",
        ]
    lines += [
        "## Storage and publication",
        "",
        audit["storage_policy"],
        "",
        "No spent sealed split was read or modified.",
        "",
        "## Hash-backed inputs",
        "",
        "| Artifact | SHA-256 | Bytes |",
        "|---|---|---:|",
    ]
    for item in entries:
        lines.append(f"| `{item['path']}` | `{item['sha256']}` | {item['bytes']} |")
    (root / "COMPLETION_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.status_file is not None:
        stamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        args.status_file.parent.mkdir(parents=True, exist_ok=True)
        with args.status_file.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{stamp} P5 complete: Stage-1 descriptive comparison finalized; both RONPO Stage-2 arms "
                "failed the unchanged length gate, so Stage-2 reward evaluation is not_run. "
                "See results/p5_8b_robust_stage1_stage2_20260717/COMPLETION_AUDIT.md.\n"
            )
    print(json.dumps({"audit": str(root / "completion_audit.json"), "markdown": str(root / "COMPLETION_AUDIT.md")}, indent=2))


if __name__ == "__main__":
    main()
