"""Queue the scalarized-DPO baseline: pairs -> dataset -> training arm.

The pair builder is new (the Safe contract puts the pairs on learner-comparator
comparisons and has two objectives), but the materializer and the training-job
writer are the campaign's own scripts and are fully parameterized by the weight
name, so the baseline runs on the same recipe anchors as the mechanism arms.
"""
import json
from pathlib import Path

Q = Path("/work/uf4_20260910/jobs/queue")
UFCODE = "/work/uf4_20260910/code"
SUBCODE = "/work/sub_20260914/code"
DPO_ROOT = "/work/uf4_20260910/dpo/safe_v1"
ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/sub_20260914/code:/work/uf4_20260910/code:"
                     "/work/nbpo_repair_20260909/code",
       "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
       "OMP_NUM_THREADS": "4", "WANDB_MODE": "disabled"}

SPECS = [
    {"job_id": "sub_dpo_pairs_train", "priority": 73, "gpus": 0,
     "depends_on": ["sub_score_safe_v1"],
     "command": ["python3", SUBCODE + "/build_dpo_safe.py", "--split", "train",
                 "--pool", "safe_v1", "--scores", "safe_v1", "--shards", "4",
                 "--out-root", DPO_ROOT + "/pairs",
                 "--calibration", DPO_ROOT + "/calibration.json"],
     "artifacts": [DPO_ROOT + "/pairs/safe_uniform/build_report_train.json"]},
    {"job_id": "sub_dpo_pairs_dev", "priority": 73, "gpus": 0,
     "depends_on": ["sub_score_safe_dev_v1", "sub_dpo_pairs_train"],
     "command": ["python3", SUBCODE + "/build_dpo_safe.py", "--split", "dev",
                 "--pool", "safe_dev_v1", "--scores", "safe_dev_v1", "--shards", "4",
                 "--out-root", DPO_ROOT + "/pairs",
                 "--calibration", DPO_ROOT + "/calibration.json"],
     "artifacts": [DPO_ROOT + "/pairs/safe_uniform/build_report_dev.json"]},
    {"job_id": "sub_dpo_materialize", "priority": 74, "gpus": 0,
     "depends_on": ["sub_dpo_pairs_train", "sub_dpo_pairs_dev"],
     "command": ["python3", UFCODE + "/materialize_dpo_dataset.py",
                 "--weight", "safe_uniform",
                 "--pairs-root", DPO_ROOT + "/pairs",
                 "--calibration", DPO_ROOT + "/calibration.json"],
     "artifacts": ["/work/uf4_20260910/datasets/dpo_safe_uniform_v1/dataset_dict.json"]},
    {"job_id": "sub_make_train_dpo_safe", "priority": 74, "gpus": 0,
     "depends_on": ["sub_dpo_materialize"],
     "command": ["python3", UFCODE + "/make_dpo_train_jobs.py",
                 "--weight", "safe_uniform", "--seed", "42", "--priority", "80",
                 "--eval-steps", "1250", "--dpo-beta", "0.01"],
     "artifacts": ["/work/uf4_20260910/provenance/train_job_dpo_safe_uniform_mse_s42.json"]},
]
existing = {json.loads(p.read_text())["job_id"] for p in Q.glob("*.json")}
for spec in SPECS:
    spec = {**spec, "cwd": SUBCODE, "env": ENV, "timeout_s": 86400}
    if spec["job_id"] in existing:
        print("exists  " + spec["job_id"]); continue
    path = Q / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
    tmp = path.with_suffix(".json.tmp"); tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.replace(path)
    print("queued  p%-3d %-26s deps=%s" % (spec["priority"], spec["job_id"],
                                           ",".join(spec["depends_on"])))
