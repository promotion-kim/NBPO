"""Queue the global game-maxmin aggregation control: solve -> config -> training.

Global game-maxmin changes only the aggregation rule applied to the same
prompt-averaged game values, so it uses the adaptive-game representation
unchanged and takes its multiplier scale from the UF-4 Nash dual by L1 matching
(the same discipline the utilitarian control uses). L1 coefficient matching is
not realized-KL matching, and the manuscript says so.

The solve is CPU-only, so it runs while the 4-GPU training holds the cards.
"""
import json, pathlib
Q = pathlib.Path("/work/uf4_20260910/jobs/queue")
R = "/work/uf4_20260910"
CPU_ENV = {"PYTHONPATH": "/work/nbpo_repair_20260909/deps_train:/work/nbpo_repair_20260909/code:" + R + "/code",
           "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "8",
           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}

jobs = [
    {"job_id": "uf4_solve_maxmin_v1", "priority": 70, "gpus": 0, "depends_on": [],
     "command": ["python3", R + "/code/solve_uf4_targets.py",
                 "--out-name", "maxmin_l1m_v1",
                 "--aggregation", "absolute_maxmin",
                 "--workers", "32",
                 "--weight-l1-from", R + "/targets/nash_v1/complete.json"],
     "artifacts": [R + "/targets/maxmin_l1m_v1/complete.json"]},
    {"job_id": "uf4_make_train_maxmin_s42", "priority": 71, "gpus": 0,
     "depends_on": ["uf4_solve_maxmin_v1"],
     "command": ["python3", R + "/code/make_train_jobs.py",
                 "--targets", "maxmin_l1m_v1", "--arm", "maxmin_mse_s42",
                 "--seed", "42", "--max-steps", "1250",
                 "--priority", "78", "--eval-steps", "250"],
     "artifacts": [R + "/provenance/train_job_maxmin_mse_s42.json"]},
]
for spec in jobs:
    spec.setdefault("cwd", "/work/nbpo_repair_20260909/code")
    spec.setdefault("env", CPU_ENV)
    spec.setdefault("timeout_s", 21600)
    path = Q / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
    payload = json.dumps(spec, indent=2) + "\n"
    if path.exists() and path.read_text() != payload:
        raise SystemExit("refusing to change existing spec: %s" % path)
    if path.exists():
        print("unchanged", spec["job_id"]); continue
    path.write_text(payload)
    print("queued", spec["job_id"], "gpus", spec["gpus"], "deps", spec["depends_on"])
