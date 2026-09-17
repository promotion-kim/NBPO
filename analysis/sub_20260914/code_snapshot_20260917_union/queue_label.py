"""Queue the policy-panel labelling: 736 verdicts per prompt on the 8Y+8Z pool.

Each shard is one GPU job that depends on the pool it reads, so the controller
starts labelling the moment generation finishes and nothing waits on an
operator. Four train shards and four dev shards means the four cards stay busy
and any single failure re-runs a quarter of one split rather than all of it.
"""
import json
from pathlib import Path

Q = Path("/work/uf4_20260910/jobs/queue")
CODE = "/work/sub_20260914/code"
ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/sub_20260914/code", "HF_HUB_OFFLINE": "1",
       "TOKENIZERS_PARALLELISM": "false", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
       "OMP_NUM_THREADS": "4", "WANDB_MODE": "disabled"}
SPECS = []
for pool, pool_shards, tag, shards, prio in (("safe_v1", 4, "safe_train_v1", 4, 70),
                                             ("safe_dev_v1", 2, "safe_dev_v1", 4, 71)):
    deps = ["sub_pool_%s_shard%d" % (pool, s) for s in range(pool_shards)]
    for shard in range(shards):
        SPECS.append({
            "job_id": "sub_label_%s_s%d" % (tag, shard),
            "priority": prio, "gpus": 1, "depends_on": deps,
            "command": ["python3", CODE + "/judge_pool_pairs.py",
                        "--pool", pool, "--pool-shards", str(pool_shards),
                        "--out-tag", tag, "--panel", "A", "--draws-per-order", "2",
                        "--shard", str(shard), "--shards", str(shards)],
            "artifacts": ["/work/sub_20260914/pool_judgments/%s/shard%d/complete.json"
                          % (tag, shard)]})
existing = {json.loads(p.read_text())["job_id"] for p in Q.glob("*.json")}
for spec in SPECS:
    spec = {**spec, "cwd": CODE, "env": ENV, "timeout_s": 86400}
    if spec["job_id"] in existing:
        print("exists  " + spec["job_id"]); continue
    path = Q / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
    tmp = path.with_suffix(".json.tmp"); tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.replace(path)
    print("queued  p%-3d %-28s deps=%s" % (spec["priority"], spec["job_id"], len(spec["depends_on"])))
