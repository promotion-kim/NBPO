"""Queue scoring, solving and training for the Safe policy panel, one seed each.

The chain is declared once with dependencies so it runs unattended:

    labelling shards -> score tensors -> exact targets per method
                     -> make_train_jobs (which queues the 4-GPU training arm)

Every method reads the SAME score tensors, the same pools and the same recipe
anchors; only the aggregation rule and the representation differ. The
utilitarian and maxmin arms take NBPO's dual mass through --weight-l1-from, so
they are not separated by a different total weight on the objectives.

Seeds: this campaign trains seed 42 only, which is what makes the eight-row
table affordable before the deadline. The declared plan of three seeds is not
deleted -- it is simply not run here, and the table's Seeds column will say 1.
"""
import json
from pathlib import Path

Q = Path("/work/uf4_20260910/jobs/queue")
UFCODE = "/work/uf4_20260910/code"
SUBCODE = "/work/sub_20260914/code"
ENV_CPU = {"PYTHONPATH": "/work/pylibs_eval:/work/uf4_20260910/code",
           "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "WANDB_MODE": "disabled"}
ENV_SUB = dict(ENV_CPU, PYTHONPATH="/work/pylibs_eval:/work/sub_20260914/code")

SOLVE_COMMON = ["--splits", "safe_v1",
                "--train-scores", "/work/uf4_20260910/scores/safe_v1",
                "--train-pool", "/work/uf4_20260910/pools/safe_v1",
                "--dev-scores", "/work/uf4_20260910/scores/safe_dev_v1",
                "--dev-pool", "/work/uf4_20260910/pools/safe_dev_v1",
                "--shards", "4", "--workers", "24"]
NASH_WEIGHTS = "/work/uf4_20260910/targets/safe_nash_v1/complete.json"

SPECS = []

# --- score the two splits from the direct judge verdicts ------------------
for tag, pool, out, deps in (
        ("safe_train_v1", "safe_v1", "safe_v1",
         ["sub_label_safe_train_v1_s%d" % s for s in range(4)]),
        ("safe_dev_v1", "safe_dev_v1", "safe_dev_v1",
         ["sub_label_safe_dev_v1_s%d" % s for s in range(4)])):
    SPECS.append({"job_id": "sub_score_%s" % out, "priority": 72, "gpus": 0,
                  "depends_on": deps, "cwd": SUBCODE, "env": ENV_SUB,
                  "command": ["python3", SUBCODE + "/score_safe_pool.py",
                              "--tag", tag, "--judge-shards", "4",
                              "--pool", pool, "--pool-shards", "4",
                              "--out-name", out, "--out-shards", "4"],
                  "artifacts": ["/work/uf4_20260910/scores/%s/score_report.json" % out]})

SCORES = ["sub_score_safe_v1", "sub_score_safe_dev_v1"]

# --- exact targets, one per declared method ------------------------------
# (job suffix, out-name, extra solver flags, depends also on)
METHODS = [
    ("nash", "safe_nash_v1", ["--aggregation", "nash"], []),
    ("fixedref", "safe_fixedref_v1",
     ["--representation", "fixed_reference", "--aggregation", "nash"], []),
    ("btrm", "safe_btrm_v1",
     ["--representation", "bt_reward", "--aggregation", "nash"], []),
    ("util", "safe_util_v1",
     ["--aggregation", "utilitarian", "--weight-l1-from", NASH_WEIGHTS],
     ["sub_solve_nash"]),
    ("maxmin", "safe_maxmin_v1",
     ["--aggregation", "absolute_maxmin", "--weight-l1-from", NASH_WEIGHTS],
     ["sub_solve_nash"]),
]
for suffix, out, extra, extra_deps in METHODS:
    SPECS.append({"job_id": "sub_solve_%s" % suffix, "priority": 73, "gpus": 0,
                  "depends_on": SCORES + extra_deps, "cwd": UFCODE, "env": ENV_CPU,
                  "command": ["python3", UFCODE + "/solve_uf4_targets.py",
                              "--out-name", out] + SOLVE_COMMON + extra,
                  "artifacts": ["/work/uf4_20260910/targets/%s/complete.json" % out]})

SPECS.append({"job_id": "sub_solve_prosper", "priority": 73, "gpus": 0,
              "depends_on": SCORES, "cwd": UFCODE, "env": ENV_CPU,
              "command": ["python3", UFCODE + "/solve_prosper_targets.py",
                          "--out-name", "safe_prosper_v1"] + SOLVE_COMMON,
              "artifacts": ["/work/uf4_20260910/targets/safe_prosper_v1/complete.json"]})

# --- one training arm per method, seed 42 --------------------------------
ARMS = [("nash", "safe_nbpo_s42", "safe_nash_v1", "nbpo", 74),
        ("fixedref", "safe_fixedref_s42", "safe_fixedref_v1", "nbpo", 75),
        ("util", "safe_util_s42", "safe_util_v1", "nbpo", 76),
        ("maxmin", "safe_maxmin_s42", "safe_maxmin_v1", "nbpo", 77),
        ("btrm", "safe_btrm_s42", "safe_btrm_v1", "nbpo", 78),
        ("prosper", "safe_prosper_s42", "safe_prosper_v1", "nbpo", 79)]
for suffix, arm, targets, loss, train_priority in ARMS:
    SPECS.append({"job_id": "sub_make_train_%s" % arm, "priority": 74, "gpus": 0,
                  "depends_on": ["sub_solve_%s" % suffix], "cwd": UFCODE, "env": ENV_CPU,
                  "command": ["python3", UFCODE + "/make_train_jobs.py",
                              "--targets", targets, "--arm", arm, "--seed", "42",
                              "--max-steps", "1250", "--priority", str(train_priority),
                              "--eval-steps", "1250"],
                  "artifacts": ["/work/uf4_20260910/provenance/train_job_%s.json" % arm]})


def main():
    existing = {json.loads(p.read_text())["job_id"] for p in Q.glob("*.json")}
    for spec in SPECS:
        spec = dict(spec)
        spec.setdefault("timeout_s", 86400)
        if spec["job_id"] in existing:
            print("exists  " + spec["job_id"])
            continue
        path = Q / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(spec, indent=2) + "\n")
        tmp.replace(path)
        print("queued  p%-3d %-30s gpus=%d deps=%d"
              % (spec["priority"], spec["job_id"], spec["gpus"], len(spec["depends_on"])))


if __name__ == "__main__":
    main()
