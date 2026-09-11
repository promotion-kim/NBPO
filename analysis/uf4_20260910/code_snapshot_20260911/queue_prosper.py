"""Queue the full PROSPER solve, its dataset, and its training arm.

CPU only, so it runs beside the fixed-reference training. The pilot solved 80
prompts on 16 workers in 89 seconds with every prompt certified and an identity
residual of 5.9e-11, so the full 11,000 is budgeted at a few hours on 48
workers.

Training priority 75 puts it after the fixed-reference seeds, BT-RM and the
seven DPO arms, because those were declared earlier; it is ahead of nothing that
is waiting on it.
"""
import json, pathlib
Q = pathlib.Path("/work/uf4_20260910/jobs/queue")
R = "/work/uf4_20260910"
CPU_ENV = {"PYTHONPATH": "/work/nbpo_repair_20260909/deps_train:/work/nbpo_repair_20260909/code:" + R + "/code",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}

jobs = [
    {"job_id": "uf4_solve_prosper_v1", "priority": 58, "gpus": 0, "depends_on": [],
     "cwd": "/work/nbpo_repair_20260909/code",
     "command": ["python3", R + "/code/solve_prosper_targets.py",
                 "--out-name", "prosper_v1", "--workers", "48"],
     "timeout_s": 43200,
     "artifacts": [R + "/targets/prosper_v1/complete.json"]},
    {"job_id": "uf4_make_train_prosper_s42", "priority": 59, "gpus": 0,
     "depends_on": ["uf4_solve_prosper_v1"],
     "cwd": "/work/nbpo_repair_20260909/code",
     "command": ["python3", R + "/code/make_train_jobs.py",
                 "--targets", "prosper_v1", "--arm", "prosper_mse_s42",
                 "--seed", "42", "--max-steps", "1250",
                 "--priority", "75", "--eval-steps", "250"],
     "timeout_s": 3600,
     "artifacts": [R + "/provenance/train_job_prosper_mse_s42.json"]},
    {"job_id": "uf4_queue_finaleval_prosper_s42", "priority": 60, "gpus": 0,
     "depends_on": ["uf4_make_train_prosper_s42"],
     "cwd": R,
     "command": ["python3", R + "/code/queue_arm_finaleval.py", "--arm", "prosper_mse_s42",
                 "--depends-on", "uf4_train_prosper_mse_s42",
                 "--gen-priority", "62", "--judge-priority", "63"],
     "timeout_s": 3600,
     "artifacts": [R + "/jobs/queue/63_finaleval_judge_prosper_mse_s42.json"]},
]
for spec in jobs:
    spec.setdefault("env", CPU_ENV)
    path = Q / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
    payload = json.dumps(spec, indent=2) + "\n"
    if path.exists() and path.read_text() != payload:
        raise SystemExit("refusing to change existing spec: %s" % path)
    if path.exists():
        print("unchanged", spec["job_id"]); continue
    path.write_text(payload)
    print("queued %-38s prio %2d gpus %d deps %s" % (spec["job_id"], spec["priority"],
                                                     spec["gpus"], spec["depends_on"] or "-"))
