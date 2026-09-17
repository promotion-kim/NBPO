"""Queue the Wild row: eight base responses per prompt, then item-level judging.

Priority 73, alongside the UF judging, because Table 37 was asked for first.
The judging is two shards so it uses two cards and finishes in about half an
hour rather than an hour.
"""
import json
from pathlib import Path

Q = Path("/work/uf4_20260910/jobs/queue")
CODE = "/work/sub_20260914/code"
ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/sub_20260914/code", "HF_HUB_OFFLINE": "1",
       "TOKENIZERS_PARALLELISM": "false", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
       "OMP_NUM_THREADS": "4", "WANDB_MODE": "disabled"}
SPECS = [{"job_id": "sub_gen_wild200", "priority": 73, "gpus": 1, "depends_on": [],
          "command": ["python3", CODE + "/gen_candidates.py",
                      "--panel", "wild200.jsonl", "--tag", "wild200",
                      "--seed-tag", "wild", "--responses-per-prompt", "8"],
          "artifacts": ["/work/sub_20260914/responses/wild200/complete.json"]}]
for shard in range(2):
    SPECS.append({"job_id": "sub_judge_wild_s%d" % shard, "priority": 73, "gpus": 1,
                  "depends_on": ["sub_gen_wild200"],
                  "command": ["python3", CODE + "/judge_wild_items.py",
                              "--out-tag", "wild200_panelA_s%d" % shard,
                              "--panel-name", "A", "--draws-per-order", "2",
                              "--shard", str(shard), "--shards", "2"],
                  "artifacts": ["/work/sub_20260914/wild_judgments/wild200_panelA_s%d/complete.json" % shard]})
existing = {json.loads(p.read_text())["job_id"] for p in Q.glob("*.json")}
queued = []
for spec in SPECS:
    if spec["job_id"] in existing:
        continue
    spec = {**spec, "cwd": CODE, "env": ENV, "timeout_s": 43200}
    path = Q / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.replace(path)
    queued.append(spec["job_id"])
print(json.dumps({"queued": queued}, indent=1))
