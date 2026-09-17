"""Give the dev pool the same shard count as train, which the solver requires.

solve_uf4_targets.py passes one --shards to both load_pool calls, so a 2-shard
dev pool against a 4-shard train pool would fail at load time. Nothing has run
yet, so the two dev specs are replaced by four and the dev labelling jobs are
repointed at them.
"""
import json
from pathlib import Path

Q = Path("/work/uf4_20260910/jobs/queue")
CODE = "/work/uf4_20260910/code"
SUBCODE = "/work/sub_20260914/code"
removed, added = [], []

for p in sorted(Q.glob("*.json")):
    spec = json.loads(p.read_text())
    if spec["job_id"].startswith("sub_pool_safe_dev_v1_shard"):
        p.unlink(); removed.append(spec["job_id"])
    if spec["job_id"].startswith("sub_label_safe_dev_v1_s"):
        p.unlink(); removed.append(spec["job_id"])

ENV_POOL = {"PYTHONPATH": "/work/pylibs_eval:/work/uf4_20260910/code", "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "OMP_NUM_THREADS": "4", "WANDB_MODE": "disabled"}
ENV_SUB = {"PYTHONPATH": "/work/pylibs_eval:/work/sub_20260914/code", "HF_HUB_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
           "OMP_NUM_THREADS": "4", "WANDB_MODE": "disabled"}

specs = []
for shard in range(4):
    specs.append({"job_id": "sub_pool_safe_dev_v1_shard%d" % shard, "priority": 69,
                  "gpus": 1, "depends_on": [], "cwd": CODE, "env": ENV_POOL,
                  "timeout_s": 43200,
                  "command": ["python3", CODE + "/generate_uf4_pool.py",
                              "--splits", "safe_v1", "--split-names", "policy_dev",
                              "--out-name", "safe_dev_v1",
                              "--shard", str(shard), "--shards", "4"],
                  "artifacts": ["/work/uf4_20260910/pools/safe_dev_v1/shard%d/settings.json" % shard]})
deps = ["sub_pool_safe_dev_v1_shard%d" % s for s in range(4)]
for shard in range(4):
    specs.append({"job_id": "sub_label_safe_dev_v1_s%d" % shard, "priority": 71,
                  "gpus": 1, "depends_on": deps, "cwd": SUBCODE, "env": ENV_SUB,
                  "timeout_s": 86400,
                  "command": ["python3", SUBCODE + "/judge_pool_pairs.py",
                              "--pool", "safe_dev_v1", "--pool-shards", "4",
                              "--out-tag", "safe_dev_v1", "--panel", "A",
                              "--draws-per-order", "2",
                              "--shard", str(shard), "--shards", "4"],
                  "artifacts": ["/work/sub_20260914/pool_judgments/safe_dev_v1/shard%d/complete.json" % shard]})

for spec in specs:
    path = Q / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.replace(path)
    added.append(spec["job_id"])
print(json.dumps({"removed": removed, "added": added}, indent=1))
