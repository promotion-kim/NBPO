"""Queue the cross-play aggregation behind the three judged pairs.

CPU only. It runs on whatever pairs exist and reports which are still missing
for the full four-policy bank, so the three-policy sub-matrix is available as
soon as it is measured rather than waiting on competitor selection.
"""
import json, pathlib
R = pathlib.Path("/work/uf4_20260910")
Q = R / "jobs/queue"
spec = {"job_id": "uf4_crossplay_aggregate", "priority": 61, "gpus": 0, "cwd": str(R),
        "env": {"PYTHONPATH": str(R / "code"), "OMP_NUM_THREADS": "4",
                "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"},
        "depends_on": ["uf4_crossplay_judge_base__vs__fixedref_mse_s42",
                       "uf4_crossplay_judge_base__vs__nbpo_mse_s42",
                       "uf4_crossplay_judge_fixedref_mse_s42__vs__nbpo_mse_s42"],
        "command": ["python3", str(R / "code/aggregate_crossplay.py")],
        "timeout_s": 7200,
        "artifacts": [str(R / "analysis/crossplay_summary.json")]}
p = Q / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
payload = json.dumps(spec, indent=2) + "\n"
if p.exists() and p.read_text() != payload:
    raise SystemExit("refusing to change existing spec")
if p.exists():
    print("unchanged", spec["job_id"])
else:
    p.write_text(payload)
    print("queued", spec["job_id"], "deps", len(spec["depends_on"]))
