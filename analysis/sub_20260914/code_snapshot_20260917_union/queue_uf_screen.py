"""Queue the UF screening judgments for Table 37's UF row.

Priority 73 puts these ahead of the remaining 4-GPU policy trainings, because
Table 37 was asked for first. Two shards so the pass uses two cards and
finishes in about twenty minutes rather than forty.
"""
import json
from pathlib import Path

Q = Path("/work/uf4_20260910/jobs/queue")
CODE = "/work/sub_20260914/code"
ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/sub_20260914/code", "HF_HUB_OFFLINE": "1",
       "TOKENIZERS_PARALLELISM": "false", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
       "OMP_NUM_THREADS": "4", "WANDB_MODE": "disabled"}
existing = {json.loads(p.read_text())["job_id"] for p in Q.glob("*.json")}
queued = []
for shard in range(2):
    job = "sub_judge_uf_screen_s%d" % shard
    if job in existing:
        continue
    spec = {"job_id": job, "priority": 73, "gpus": 1, "depends_on": [],
            "cwd": CODE, "env": ENV, "timeout_s": 43200,
            "command": ["python3", CODE + "/judge_pairs.py",
                        "--responses", "uf_screen200",
                        "--rubrics", "/work/sub_20260914/contract/rubrics_uf4.json",
                        "--out-tag", "uf_screen200_panelA_s%d" % shard,
                        "--panel", "A", "--draws-per-order", "2",
                        "--shard", str(shard), "--shards", "2"],
            "artifacts": ["/work/sub_20260914/judgments/uf_screen200_panelA_s%d/complete.json" % shard]}
    path = Q / ("%d_%s.json" % (spec["priority"], job))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.replace(path)
    queued.append(job)
print(json.dumps({"queued": queued}, indent=1))
