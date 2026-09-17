import json
from pathlib import Path
Q = Path("/work/uf4_20260910/jobs/queue")
CODE = "/work/sub_20260914/code"
ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/sub_20260914/code", "HF_HUB_OFFLINE": "1",
       "TOKENIZERS_PARALLELISM": "false", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
       "OMP_NUM_THREADS": "4", "WANDB_MODE": "disabled"}
SPECS = [
    # the remaining 190 screening prompts, two shards so both free cards work
    {"job_id": "sub_judge_safe_rest_s0", "priority": 66, "gpus": 1, "depends_on": [],
     "command": ["python3", CODE + "/judge_pairs.py", "--responses", "screen200",
                 "--exclude-prompt-list", "pilot10.jsonl", "--out-tag", "safe_rest_panelA_s0",
                 "--panel", "A", "--draws-per-order", "2", "--shard", "0", "--shards", "2"],
     "artifacts": ["/work/sub_20260914/judgments/safe_rest_panelA_s0/complete.json"]},
    {"job_id": "sub_judge_safe_rest_s1", "priority": 66, "gpus": 1, "depends_on": [],
     "command": ["python3", CODE + "/judge_pairs.py", "--responses", "screen200",
                 "--exclude-prompt-list", "pilot10.jsonl", "--out-tag", "safe_rest_panelA_s1",
                 "--panel", "A", "--draws-per-order", "2", "--shard", "1", "--shards", "2"],
     "artifacts": ["/work/sub_20260914/judgments/safe_rest_panelA_s1/complete.json"]},
    # the preselected 50-prompt confirmation set, five extra draws per order, panel B
    {"job_id": "sub_judge_safe_confirm_s0", "priority": 68, "gpus": 1, "depends_on": [],
     "command": ["python3", CODE + "/judge_pairs.py", "--responses", "screen200",
                 "--prompt-list", "confirm50.jsonl", "--out-tag", "safe_confirm_panelB_s0",
                 "--panel", "B", "--draws-per-order", "5", "--shard", "0", "--shards", "2"],
     "artifacts": ["/work/sub_20260914/judgments/safe_confirm_panelB_s0/complete.json"]},
    {"job_id": "sub_judge_safe_confirm_s1", "priority": 68, "gpus": 1, "depends_on": [],
     "command": ["python3", CODE + "/judge_pairs.py", "--responses", "screen200",
                 "--prompt-list", "confirm50.jsonl", "--out-tag", "safe_confirm_panelB_s1",
                 "--panel", "B", "--draws-per-order", "5", "--shard", "1", "--shards", "2"],
     "artifacts": ["/work/sub_20260914/judgments/safe_confirm_panelB_s1/complete.json"]},
]
existing = {json.loads(p.read_text())["job_id"] for p in Q.glob("*.json")}
for spec in SPECS:
    spec = {**spec, "cwd": CODE, "env": ENV, "timeout_s": 43200}
    if spec["job_id"] in existing:
        print("exists  " + spec["job_id"]); continue
    path = Q / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
    tmp = path.with_suffix(".json.tmp"); tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.replace(path)
    print("queued  p%-3d %s" % (spec["priority"], spec["job_id"]))
