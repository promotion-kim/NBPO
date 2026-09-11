"""Queue BT-RM--Nash, plus a regression solve that proves the patch was additive.

BT-RM--Nash is the matched control that changes the preference representation:
a frozen scalar Bradley-Terry reward per objective, no opponent at all. Like the
fixed-reference control it keeps the Nash rule, so the Nash dual picks its own
multiplier scale and no L1 matching flag is passed.

uf4_regress_nash_check re-solves the published Nash target under the patched
solver and writes its certificate next to the original, so "the change did not
touch the existing path" is checked rather than asserted. It skips dataset
materialization: only the solver numbers are under test.
"""
import json, pathlib
Q = pathlib.Path("/work/uf4_20260910/jobs/queue")
R = "/work/uf4_20260910"
CPU_ENV = {"PYTHONPATH": "/work/nbpo_repair_20260909/deps_train:/work/nbpo_repair_20260909/code:" + R + "/code",
           "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "8",
           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}

jobs = [
    {"job_id": "uf4_regress_nash_check", "priority": 68, "gpus": 0, "depends_on": [],
     "command": ["python3", R + "/code/solve_uf4_targets.py",
                 "--out-name", "regress_nash_check", "--aggregation", "nash",
                 "--workers", "24", "--skip-dataset"],
     "artifacts": [R + "/targets/regress_nash_check/complete.json"]},
    {"job_id": "uf4_solve_btrm_v1", "priority": 69, "gpus": 0, "depends_on": [],
     "command": ["python3", R + "/code/solve_uf4_targets.py",
                 "--out-name", "btrm_nash_v1", "--representation", "bt_reward",
                 "--aggregation", "nash", "--workers", "24"],
     "artifacts": [R + "/targets/btrm_nash_v1/complete.json"]},
    {"job_id": "uf4_make_train_btrm_s42", "priority": 72, "gpus": 0,
     "depends_on": ["uf4_solve_btrm_v1"],
     "command": ["python3", R + "/code/make_train_jobs.py",
                 "--targets", "btrm_nash_v1", "--arm", "btrm_mse_s42",
                 "--seed", "42", "--max-steps", "1250",
                 "--priority", "79", "--eval-steps", "250"],
     "artifacts": [R + "/provenance/train_job_btrm_mse_s42.json"]},
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
