"""Queue fresh make_train jobs so the configs are rewritten against the new datasets.

A DONE job never re-runs in this controller -- only FAILED and BLOCKED jobs do,
and only when their spec hash changes. Bumping the finished make_train specs
therefore did nothing, and since the stale configs had already been deleted,
nothing was left to dispatch. New job ids fix that: they are new work, so they
run once and write the config with the rebuilt dataset's manifest hash.

The v2 jobs also carry the training priority explicitly, so the arms keep the
same order they had before: NBPO 74, fixed-reference 75, utilitarian 76,
maxmin 77, BT-projected 78.
"""
import json
from pathlib import Path

UF = Path("/work/uf4_20260910")
Q = UF / "jobs/queue"
CODE = "/work/uf4_20260910/code"
ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/uf4_20260910/code",
       "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1", "WANDB_MODE": "disabled"}
ARMS = [("safe_nbpo_s42", "safe_nash_v1", 74),
        ("safe_fixedref_s42", "safe_fixedref_v1", 75),
        ("safe_util_s42", "safe_util_v1", 76),
        ("safe_maxmin_s42", "safe_maxmin_v1", 77),
        ("safe_btrm_s42", "safe_btrm_v1", 78)]

existing = {json.loads(p.read_text())["job_id"] for p in Q.glob("*.json")}
queued = []
for arm, targets, priority in ARMS:
    job = "sub_make_train_%s_v2" % arm
    if job in existing:
        print("exists  " + job)
        continue
    spec = {"job_id": job, "priority": 73, "gpus": 0, "depends_on": [],
            "cwd": CODE, "env": ENV, "timeout_s": 43200,
            "command": ["python3", CODE + "/make_train_jobs.py",
                        "--targets", targets, "--arm", arm, "--seed", "42",
                        "--max-steps", "1250", "--priority", str(priority),
                        "--eval-steps", "1250"],
            "artifacts": [str(UF / "provenance" / ("train_job_%s.json" % arm))],
            "note": ("rewrites configs/%s.yaml against the dataset rebuilt under "
                     "deps_train, whose manifest hash is 485c802b1ecfb628 rather than "
                     "the 5d67ac43d9a3a97b the first config pinned" % arm)}
    path = Q / ("%d_%s.json" % (spec["priority"], job))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.replace(path)
    queued.append(job)
print(json.dumps({"queued": queued}, indent=1))
