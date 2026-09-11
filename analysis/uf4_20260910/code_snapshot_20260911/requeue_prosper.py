"""Requeue the PROSPER chain under fresh ids after the solution-artifact fix.

The solve itself succeeded and is recorded DONE, but its target and dataset
directories have been removed because the rows have to be rebuilt carrying the
hash of the new per-prompt solution artifact. The controller only re-runs a
FAILED or BLOCKED job whose spec changed, never a DONE one, and editing
state.json underneath a running controller races its own flush. New ids are the
clean way to say "this is a different run".

The retired specs are renamed aside so load_specs stops seeing them and cannot
raise on a duplicate id.
"""
import json, pathlib
Q = pathlib.Path("/work/uf4_20260910/jobs/queue")
R = "/work/uf4_20260910"
CPU_ENV = {"PYTHONPATH": "/work/nbpo_repair_20260909/deps_train:/work/nbpo_repair_20260909/code:" + R + "/code",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}

for old_id in ("uf4_solve_prosper_v1", "uf4_make_train_prosper_s42",
               "uf4_queue_finaleval_prosper_s42"):
    for path in sorted(Q.glob("*.json")):
        if json.loads(path.read_text())["job_id"] == old_id:
            path.rename(path.with_suffix(".json.superseded_by_v2"))
            print("retired spec", path.name)

jobs = [
    {"job_id": "uf4_solve_prosper_v2", "priority": 58, "gpus": 0, "depends_on": [],
     "cwd": "/work/nbpo_repair_20260909/code",
     "command": ["python3", R + "/code/solve_prosper_targets.py",
                 "--out-name", "prosper_v1", "--workers", "48"],
     "timeout_s": 43200,
     "note": ("re-solve after adding a hash-bound per-prompt solution artifact at the path "
              "make_train_jobs.py hashes; the first solve was numerically fine but its rows "
              "carried a placeholder solver hash"),
     "artifacts": [R + "/targets/prosper_v1/complete.json"]},
    {"job_id": "uf4_make_train_prosper_s42_v2", "priority": 59, "gpus": 0,
     "depends_on": ["uf4_solve_prosper_v2"],
     "cwd": "/work/nbpo_repair_20260909/code",
     "command": ["python3", R + "/code/make_train_jobs.py",
                 "--targets", "prosper_v1", "--arm", "prosper_mse_s42",
                 "--seed", "42", "--max-steps", "1250",
                 "--priority", "75", "--eval-steps", "250"],
     "timeout_s": 3600,
     "artifacts": [R + "/provenance/train_job_prosper_mse_s42.json"]},
    {"job_id": "uf4_queue_finaleval_prosper_s42_v2", "priority": 60, "gpus": 0,
     "depends_on": ["uf4_make_train_prosper_s42_v2"], "cwd": R,
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
    print("queued", spec["job_id"], "prio", spec["priority"])
