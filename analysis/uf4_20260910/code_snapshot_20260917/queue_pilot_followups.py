"""Queue the pilot follow-ups: fit metrics, then fresh generation and judging.

The fit job depends on both pilot arms so it reads two exported checkpoints in
one pass. The fresh rows for the pilots are 200 draws and 6,400 verdicts each,
the same contract as the full-run fresh rows, and they are what decides whether
a better pool fit shows up in new outputs at all.
"""
import json
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
QUEUE = ROOT / "jobs/queue"
DIAG = ROOT / "analysis/diag_20260914"
ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/uf4_20260910/code",
       "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
       "TOKENIZERS_PARALLELISM": "false", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
       "OMP_NUM_THREADS": "4", "WANDB_MODE": "disabled"}
PILOTS = [("diag_proj_sampled_s42", "uf4_diag_proj_sampled"),
          ("diag_proj_all_s42", "uf4_diag_proj_all")]


def write(spec):
    existing = {json.loads(p.read_text())["job_id"] for p in QUEUE.glob("*.json")}
    if spec["job_id"] in existing:
        return "exists  " + spec["job_id"]
    path = QUEUE / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.replace(path)
    return "queued  " + spec["job_id"]


print(write({
    "job_id": "uf4_diag_pilot_fit", "priority": 57, "gpus": 1, "cwd": str(ROOT),
    "env": ENV, "depends_on": [job for _arm, job in PILOTS],
    "command": ["python3", str(ROOT / "code/diag_pilot_table.py")],
    "timeout_s": 10800,
    "artifacts": [str(DIAG / "projection/pilot_table.json")],
}))
for arm, train_job in PILOTS:
    gen = "uf4_diag_fresh_gen_%s" % arm
    print(write({
        "job_id": gen, "priority": 58, "gpus": 1, "cwd": str(ROOT), "env": ENV,
        "depends_on": [train_job],
        "command": ["python3", str(ROOT / "code/diag_gen_fresh.py"),
                    "--arm", arm, "--model", str(ROOT / "arms" / arm)],
        "timeout_s": 5400,
        "artifacts": [str(DIAG / "fresh" / ("complete_%s.json" % arm))],
    }))
    print(write({
        "job_id": "uf4_diag_fresh_judge_%s" % arm, "priority": 59, "gpus": 1,
        "cwd": str(ROOT), "env": ENV, "depends_on": [gen],
        "command": ["python3", str(ROOT / "code/diag_judge_fresh.py"), "--arm", arm],
        "timeout_s": 10800,
        "artifacts": [str(DIAG / "fresh_verdicts" / ("complete_%s.json" % arm))],
    }))
