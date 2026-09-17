"""Queue the frozen reference bank, the base fresh draw and its judging.

The base row of tab:stress_results needs no trained policy, so it can be
measured before any arm exists and it fixes the reference bank every arm will
be judged against. The two draws use different seed tags, so the base's fresh
response is an independent draw rather than a copy of reference occurrence 0.
"""
import json
from pathlib import Path

Q = Path("/work/uf4_20260910/jobs/queue")
CODE = "/work/sub_20260914/code"
ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/sub_20260914/code", "HF_HUB_OFFLINE": "1",
       "TOKENIZERS_PARALLELISM": "false", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
       "OMP_NUM_THREADS": "4", "WANDB_MODE": "disabled"}
SPECS = [
    {"job_id": "sub_gen_ref4_test", "priority": 75, "gpus": 1, "depends_on": [],
     "command": ["python3", CODE + "/gen_candidates.py", "--panel", "test1000.jsonl",
                 "--tag", "test_ref4", "--seed-tag", "ref4",
                 "--responses-per-prompt", "4"],
     "artifacts": ["/work/sub_20260914/responses/test_ref4/complete.json"]},
    {"job_id": "sub_gen_fresh_base_test", "priority": 75, "gpus": 1, "depends_on": [],
     "command": ["python3", CODE + "/gen_candidates.py", "--panel", "test1000.jsonl",
                 "--tag", "test_fresh_base", "--seed-tag", "fresh_base",
                 "--responses-per-prompt", "1"],
     "artifacts": ["/work/sub_20260914/responses/test_fresh_base/complete.json"]},
    {"job_id": "sub_judge_fresh_base", "priority": 76, "gpus": 1,
     "depends_on": ["sub_gen_ref4_test", "sub_gen_fresh_base_test"],
     "command": ["python3", CODE + "/judge_fresh_safe.py",
                 "--arm-tag", "test_fresh_base", "--reference-tag", "test_ref4",
                 "--out-tag", "base_fresh"],
     "artifacts": ["/work/sub_20260914/fresh_eval/base_fresh/complete.json"]},
]
existing = {json.loads(p.read_text())["job_id"] for p in Q.glob("*.json")}
for spec in SPECS:
    spec = {**spec, "cwd": CODE, "env": ENV, "timeout_s": 43200}
    if spec["job_id"] in existing:
        print("exists  " + spec["job_id"]); continue
    path = Q / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
    tmp = path.with_suffix(".json.tmp"); tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.replace(path)
    print("queued  p%-3d %-26s deps=%d" % (spec["priority"], spec["job_id"], len(spec["depends_on"])))
