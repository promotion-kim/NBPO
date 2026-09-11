"""Queue fixed-reference Nash seeds 43 and 44, and their evaluations.

The frozen matrix declared one seed for this control, which was right when it was
a secondary comparison. It is no longer secondary: on the common set it is the
stronger policy on all four attributes and the only arm whose minimum-attribute
point estimate clears 0.5, so the paper's central mechanism claim now turns on a
single seed against three for each adaptive-game family. Adding seeds widens the
denominator rather than narrowing it, which is the direction the matrix rules
allow.

They sit behind BT-RM and the main DPO row, so the breadth of the objectives
table still comes first, and ahead of the six specialist DPO corners, which
serve a figure rather than a main-body row.
"""
import json, pathlib
Q = pathlib.Path("/work/uf4_20260910/jobs/queue")
R = "/work/uf4_20260910"
CPU_ENV = {"PYTHONPATH": "/work/nbpo_repair_20260909/deps_train:/work/nbpo_repair_20260909/code:" + R + "/code",
           "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "8",
           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
PLAN = [(43, 81, 90, 91), (44, 82, 92, 93)]

jobs = []
for seed, train_prio, gen_prio, judge_prio in PLAN:
    arm = "fixedref_mse_s%d" % seed
    jobs.append({
        "job_id": "uf4_make_train_fixedref_s%d" % seed, "priority": 64, "gpus": 0,
        "depends_on": [],
        "command": ["python3", R + "/code/make_train_jobs.py",
                    "--targets", "fixedref_nash_v1", "--arm", arm,
                    "--seed", str(seed), "--max-steps", "1250",
                    "--priority", str(train_prio), "--eval-steps", "250"],
        "artifacts": [R + "/provenance/train_job_%s.json" % arm]})
    jobs.append({
        "job_id": "uf4_queue_finaleval_fixedref_s%d" % seed, "priority": 65, "gpus": 0,
        "depends_on": ["uf4_make_train_fixedref_s%d" % seed],
        "command": ["python3", R + "/code/queue_arm_finaleval.py", "--arm", arm,
                    "--depends-on", "uf4_train_%s" % arm,
                    "--gen-priority", str(gen_prio), "--judge-priority", str(judge_prio)],
        "artifacts": [R + "/jobs/queue/%d_finaleval_judge_%s.json" % (judge_prio, arm)]})

for spec in jobs:
    spec.setdefault("cwd", R)
    spec.setdefault("env", CPU_ENV)
    spec.setdefault("timeout_s", 3600)
    path = Q / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
    payload = json.dumps(spec, indent=2) + "\n"
    if path.exists() and path.read_text() != payload:
        raise SystemExit("refusing to change existing spec: %s" % path)
    if path.exists():
        print("unchanged", spec["job_id"]); continue
    path.write_text(payload)
    print("queued %-40s prio %3d" % (spec["job_id"], spec["priority"]))
