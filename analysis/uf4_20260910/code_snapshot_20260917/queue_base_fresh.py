"""Queue the optional fourth fresh arm: a new draw from the untrained base.

Section 3.3 declares this row optional and separate, at 200 generated responses
and 6,400 rubric-order verdicts. It is what calibrates the other fresh rows:
without it, a fresh win rate above .5 could be the base's own draw quality
rather than anything training did. Two idle cards and 90 minutes of slack make
it free, and it is recorded as its own manifest row rather than folded into the
three declared arms.
"""
import json
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
QUEUE = ROOT / "jobs/queue"
DIAG = ROOT / "analysis/diag_20260914"
ARM = "base_fresh"
MODEL = "/work/models/bases/Llama-3.1-8B-Instruct"
ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/uf4_20260910/code",
       "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
       "TOKENIZERS_PARALLELISM": "false", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
       "OMP_NUM_THREADS": "4", "WANDB_MODE": "disabled"}


def write(spec):
    existing = {json.loads(p.read_text())["job_id"] for p in QUEUE.glob("*.json")}
    if spec["job_id"] in existing:
        return "exists  " + spec["job_id"]
    path = QUEUE / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.replace(path)
    return "queued  " + spec["job_id"]


gen = "uf4_diag_fresh_gen_%s" % ARM
print(write({
    "job_id": gen, "priority": 50, "gpus": 1, "cwd": str(ROOT), "env": ENV,
    "depends_on": [],
    "command": ["python3", str(ROOT / "code/diag_gen_fresh.py"), "--arm", ARM, "--model", MODEL],
    "timeout_s": 5400,
    "artifacts": [str(DIAG / "fresh" / ("complete_%s.json" % ARM))],
}))
print(write({
    "job_id": "uf4_diag_fresh_judge_%s" % ARM, "priority": 50, "gpus": 1,
    "cwd": str(ROOT), "env": ENV, "depends_on": [gen],
    "command": ["python3", str(ROOT / "code/diag_judge_fresh.py"), "--arm", ARM],
    "timeout_s": 10800,
    "artifacts": [str(DIAG / "fresh_verdicts" / ("complete_%s.json" % ARM))],
}))
