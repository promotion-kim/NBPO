"""Queue the scalarized-DPO arms: materialize, smoke-test, then train.

uniform is materialized first and alone, because it is the main-body row and
because a bug in the materializer should cost one 8 GB pass rather than seven.
Everything else depends on it succeeding.

The remaining six run in two sequential lanes rather than all at once: they are
CPU jobs, so the controller would dispatch all of them immediately, and seven
concurrent 8 GB json-to-Arrow conversions would compete for the same Lustre
bandwidth the running training's dataloader needs.

A two-update single-GPU smoke run sits at backfill priority. It proves the
config loads, the dataset collates and the online reference initializes, in
about ten minutes on a spare card, instead of discovering a config error when
the first real DPO arm reaches the front of the queue after midnight.
"""
import json, pathlib

Q = pathlib.Path("/work/uf4_20260910/jobs/queue")
R = "/work/uf4_20260910"
CPU_ENV = {"PYTHONPATH": "/work/nbpo_repair_20260909/deps_train:/work/nbpo_repair_20260909/code:" + R + "/code",
           "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "8",
           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false"}
LANES = [["if_only", "truth_only", "honesty_only"],
         ["help_only", "help_heavy", "truth_heavy"]]
TRAIN_PRIORITY = {"uniform": 80, "if_only": 96, "truth_only": 97, "honesty_only": 98,
                  "help_only": 99, "help_heavy": 100, "truth_heavy": 101}

jobs = []


def materialize(weight, priority, deps):
    return {"job_id": f"uf4_materialize_dpo_{weight}", "priority": priority, "gpus": 0,
            "depends_on": deps,
            "command": ["python3", R + "/code/materialize_dpo_dataset.py",
                        "--weight", weight],
            "artifacts": [f"{R}/datasets/dpo_{weight}_v1/dpo_dataset_provenance.json"]}


def make_train(weight, priority, extra=()):
    return {"job_id": f"uf4_make_dpo_train_{weight}", "priority": priority, "gpus": 0,
            "depends_on": [f"uf4_materialize_dpo_{weight}"],
            "command": ["python3", R + "/code/make_dpo_train_jobs.py",
                        "--weight", weight, "--seed", "42",
                        "--priority", str(TRAIN_PRIORITY[weight]), *extra],
            "artifacts": [f"{R}/provenance/train_job_dpo_{weight}_mse_s42.json"]}


jobs.append(materialize("uniform", 66, []))
# the smoke config, and the real uniform arm
jobs.append({"job_id": "uf4_make_dpo_smoke", "priority": 67, "gpus": 0,
             "depends_on": ["uf4_materialize_dpo_uniform"],
             "command": ["python3", R + "/code/make_dpo_train_jobs.py",
                         "--weight", "uniform", "--seed", "42",
                         "--priority", "77", "--gpus", "1", "--smoke"],
             "artifacts": [f"{R}/provenance/train_job_smoke_dpo_uniform.json"]})
jobs.append(make_train("uniform", 68))

priority = 69
for lane in LANES:
    previous = ["uf4_materialize_dpo_uniform"]
    for weight in lane:
        jobs.append(materialize(weight, priority, previous))
        jobs.append(make_train(weight, priority + 1))
        previous = [f"uf4_materialize_dpo_{weight}"]
        priority += 2

for spec in jobs:
    spec.setdefault("cwd", R)
    spec.setdefault("env", CPU_ENV)
    spec.setdefault("timeout_s", 21600)
    path = Q / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
    payload = json.dumps(spec, indent=2) + "\n"
    if path.exists() and path.read_text() != payload:
        raise SystemExit("refusing to change existing spec: %s" % path)
    if path.exists():
        print("unchanged", spec["job_id"]); continue
    path.write_text(payload)
    print("queued %-38s prio %3d deps %s" % (spec["job_id"], spec["priority"],
                                             spec["depends_on"] or "-"))
