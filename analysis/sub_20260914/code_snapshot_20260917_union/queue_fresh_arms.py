"""Queue one fresh draw and one judging pass per trained arm.

Each arm's row in tab:stress_results needs a fresh response per test prompt and
16 verdicts against the frozen four-response reference bank. Both jobs depend on
that arm's training job, so they fire as soon as it finishes.

They sit at priority 73, above the 4-GPU trainings at 74-80. That costs about
ten minutes of training time per arm, and buys an incrementally filled table:
the first policy row lands around 02:40 instead of after every arm has trained,
which is when a defect in the evaluation path can still be fixed cheaply.

Seed tags differ per arm, so no two draws share a sampling stream, and the
reference bank is the same frozen one the base row was scored against.
"""
import json
from pathlib import Path

UF = Path("/work/uf4_20260910")
Q = UF / "jobs/queue"
CODE = "/work/sub_20260914/code"
ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/sub_20260914/code", "HF_HUB_OFFLINE": "1",
       "TOKENIZERS_PARALLELISM": "false", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
       "OMP_NUM_THREADS": "4", "WANDB_MODE": "disabled"}

# (row prefix, arm directory, training job id)
ARMS = [
    ("fixed", "safe_fixedref_s42", "uf4_train_safe_fixedref_s42"),
    ("nbpo", "safe_nbpo_s42", "uf4_train_safe_nbpo_s42"),
    ("util", "safe_util_s42", "uf4_train_safe_util_s42"),
    ("maxmin", "safe_maxmin_s42", "uf4_train_safe_maxmin_s42"),
    ("bt", "safe_btrm_s42", "uf4_train_safe_btrm_s42"),
    ("dpo", "dpo_safe_uniform_mse_s42", "uf4_train_dpo_safe_uniform_mse_s42"),
]

existing = {json.loads(p.read_text())["job_id"] for p in Q.glob("*.json")}
queued = []
for prefix, arm, train_job in ARMS:
    gen_job = "sub_gen_fresh_%s" % arm
    judge_job = "sub_judge_fresh_%s" % arm
    specs = [
        {"job_id": gen_job, "priority": 73, "gpus": 1, "depends_on": [train_job],
         "command": ["python3", CODE + "/gen_candidates.py",
                     "--panel", "test1000.jsonl",
                     "--model", str(UF / "arms" / arm),
                     "--model-revision", "local-checkpoint:%s" % arm,
                     "--tag", "test_fresh_%s" % arm,
                     "--seed-tag", "fresh_%s" % arm,
                     "--responses-per-prompt", "1"],
         "artifacts": ["/work/sub_20260914/responses/test_fresh_%s/complete.json" % arm]},
        {"job_id": judge_job, "priority": 73, "gpus": 1, "depends_on": [gen_job],
         "command": ["python3", CODE + "/judge_fresh_safe.py",
                     "--arm-tag", "test_fresh_%s" % arm,
                     "--reference-tag", "test_ref4",
                     "--out-tag", arm],
         "artifacts": ["/work/sub_20260914/fresh_eval/%s/complete.json" % arm]},
    ]
    for spec in specs:
        if spec["job_id"] in existing:
            continue
        spec = {**spec, "cwd": CODE, "env": ENV, "timeout_s": 43200,
                "row_prefix": prefix}
        path = Q / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(spec, indent=2) + "\n")
        tmp.replace(path)
        queued.append(spec["job_id"])

print(json.dumps({"queued": queued, "count": len(queued)}, indent=1))
