"""Index every artifact a claim in the report or the paper rests on.

Walks the campaign root for the specific files that back a number, records each
one's sha256 and size, and refuses to list a path that does not exist -- so the
index cannot claim provenance for something that was never written.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path("/work/nbpo_repair_20260909")

GROUPS = {
    "protocol": ["resolved_protocol.yaml", "protocols/resolved_protocol_frozen_v1.yaml",
                 "protocols/primary_horizon_frozen_v1.json",
                 "protocols/wbc_short_horizon_prospective_v1.json",
                 "protocols/short_horizon_prospective_v2.json"],
    "provenance": ["provenance/inventory.json", "provenance/legacy_tokenization_audit_v1.json",
                   "splits/manifest.json", "splits/benchmark_counts.json",
                   "datasets/nash_repair_v2/precompute_manifest.json"],
    "teacher": ["teachers/nash_repair_v2/inputs.json",
                "teachers/nash_repair_v2/train/teacher_diagnostics.json",
                "teachers/nash_repair_v2/train/solver/solution.json",
                "teachers/nash_repair_v2/dev/solver/solution.json",
                "teachers/nash_repair_v2/test/solver/solution.json",
                "teachers/utilitarian_l1matched_v1/test/solver/solution.json"],
    "arms": [f"arms/{arm}/trainer_state.json" for arm in
             ("wbc_primary_v1", "mse_primary_v1", "wbc_short_primary_v1", "mse_short_primary_v1")],
    "runtime": [f"arms/{arm}/runtime_rank{rank}.jsonl" for arm in
                ("wbc_primary_v1", "mse_primary_v1") for rank in range(4)],
    "controllers": ["controllers/primary_chain_v1/complete.json",
                    "controllers/evaluation_chain_v1/complete.json",
                    "controllers/evaluation_chain_v1/failure.harmbench_lane_v1.json",
                    "controllers/harmbench_recovery_v1/complete.json",
                    "controllers/followup_chain_v1/complete.json"],
    "analysis": ["analysis_claude/runtime_summary.json", "analysis_claude/capability.json",
                 "analysis_claude/surplus_test.json",
                 "analysis_claude/teacher_length_structure.json",
                 "analysis_claude/engine_reaper.jsonl",
                 "analysis_claude/blind_audit_v1/manifest.json"],
}
GLOBS = {
    "evaluations": "evaluations/*/*.summary.json",
    "saferlhf_reports": "evaluations/*/*/report.json",
    "responses": "responses/*/*.manifest.json",
    "pool_drift": "analysis_claude/pool_drift_*.json",
    "job_exits": "jobs/*/exit.json",
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path),
            "bytes": path.stat().st_size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "result_index.json")
    args = ap.parse_args()
    index, missing = {}, []
    for group, names in GROUPS.items():
        found = []
        for name in names:
            path = ROOT / name
            (found.append(entry(path)) if path.is_file() else missing.append(name))
        index[group] = found
    for group, pattern in GLOBS.items():
        index[group] = [entry(p) for p in sorted(ROOT.glob(pattern))]
    payload = {"schema": "nbpo-repair-result-index-v1",
               "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "root": str(ROOT), "paid_judge_api_calls": 0,
               "files_indexed": sum(len(v) for v in index.values()),
               "declared_but_absent": missing, "groups": index}
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"files_indexed": payload["files_indexed"],
                      "declared_but_absent": missing,
                      "per_group": {k: len(v) for k, v in index.items()}}, indent=2))


if __name__ == "__main__":
    main()
