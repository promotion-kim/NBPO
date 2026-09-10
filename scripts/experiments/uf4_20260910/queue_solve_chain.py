import json, pathlib
Q = pathlib.Path("/work/uf4_20260910/jobs/queue")
R = "/work/uf4_20260910"
CPU_ENV = {"PYTHONPATH": "/work/nbpo_repair_20260909/deps_train:/work/nbpo_repair_20260909/code:" + R + "/code",
           "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "4",
           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
jobs = [
    {"job_id": "uf4_make_train_nbpo_s42", "priority": 51, "gpus": 0,
     "depends_on": ["uf4_solve_nash_v1"],
     "command": ["python3", R + "/code/make_train_jobs.py", "--targets", "nash_v1",
                 "--arm", "nbpo_mse_s42", "--seed", "42", "--max-steps", "1250",
                 "--priority", "60"],
     "artifacts": [R + "/provenance/train_job_nbpo_mse_s42.json"]},
    {"job_id": "uf4_solve_util_v1", "priority": 52, "gpus": 0,
     "depends_on": ["uf4_solve_nash_v1"],
     "command": ["python3", R + "/code/solve_uf4_targets.py", "--out-name", "util_l1matched_v1",
                 "--aggregation", "utilitarian", "--workers", "32",
                 "--weight-l1-from", R + "/targets/nash_v1/complete.json"],
     "artifacts": [R + "/targets/util_l1matched_v1/complete.json"]},
    {"job_id": "uf4_make_train_util_s42", "priority": 53, "gpus": 0,
     "depends_on": ["uf4_solve_util_v1"],
     "command": ["python3", R + "/code/make_train_jobs.py", "--targets", "util_l1matched_v1",
                 "--arm", "util_mse_s42", "--seed", "42", "--max-steps", "1250",
                 "--priority", "61"],
     "artifacts": [R + "/provenance/train_job_util_mse_s42.json"]},
]
for spec in jobs:
    spec.setdefault("cwd", "/work/nbpo_repair_20260909/code")
    spec.setdefault("env", CPU_ENV)
    spec.setdefault("timeout_s", 21600)
    path = Q / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
    path.write_text(json.dumps(spec, indent=2) + "\n")
    print("queued", spec["job_id"], "deps", spec["depends_on"])
