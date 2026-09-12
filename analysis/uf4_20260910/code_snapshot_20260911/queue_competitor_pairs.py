"""Complete the cross-play bank with the selected competitor, util_mse_s42.

The dev-only rule picked util_mse_s42 over maxmin_mse_s42 on policy_dev minimum
objective (0.5074 against 0.5005, outside the 0.002 tie band), recorded before
any cross-play response existed. That fixes the fourth bank slot, so its
generation and its three pairs can be queued.

Judging runs at 2048 tokens, matching the three pairs already judged, so every
cell in the matrix shares one token budget. Aggregation is requeued under a new
id to wait on all six pairs.
"""
import json, pathlib
R = pathlib.Path("/work/uf4_20260910")
Q = R / "jobs/queue"
COMP = "util_mse_s42"
BANK = ["base", "fixedref_mse_s42", "nbpo_mse_s42"]
ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/uf4_20260910/code",
       "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
       "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "OMP_NUM_THREADS": "4",
       "WANDB_MODE": "disabled"}


def put(spec):
    p = Q / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
    payload = json.dumps(spec, indent=2) + "\n"
    if p.exists() and p.read_text() != payload:
        raise SystemExit("refusing to change existing spec: %s" % p)
    if p.exists():
        print("unchanged", spec["job_id"]); return False
    p.write_text(payload)
    print("queued", spec["job_id"])
    return True


put({"job_id": "uf4_crossplay_gen_%s" % COMP, "priority": 64, "gpus": 1,
     "cwd": str(R), "env": ENV, "depends_on": [],
     "command": ["python3", str(R / "code/generate_uf4_pool.py"),
                 "--splits", "v1_reproduce", "--split-names", "crossplay_500",
                 "--out-name", "crossplay_%s" % COMP, "--model", str(R / "arms" / COMP),
                 "--shard", "0", "--shards", "1", "--prompts-per-chunk", "25"],
     "timeout_s": 21600,
     "artifacts": [str(R / ("pools/crossplay_%s/shard0/complete_shard0.json" % COMP))]})

judges = []
for other in BANK:
    a, b = sorted([other, COMP])
    tag = "%s__vs__%s" % (a, b)
    jid = "uf4_crossplay_judge_%s_t2048" % tag
    judges.append(jid)
    put({"job_id": jid, "priority": 64, "gpus": 1, "cwd": str(R), "env": ENV,
         "depends_on": ["uf4_crossplay_gen_%s" % COMP] +
                       ([] if other == "base" else []),
         "command": ["python3", str(R / "code/judge_crossplay_pair.py"),
                     "--policy-a", a, "--policy-b", b,
                     "--pool-a", str(R / ("pools/crossplay_%s" % a)),
                     "--pool-b", str(R / ("pools/crossplay_%s" % b)),
                     "--max-tokens", "2048"],
         "timeout_s": 43200,
         "artifacts": [str(R / ("evaluation/crossplay/pairs/%s/complete.json" % tag))]})

existing = ["uf4_crossplay_judge_%s__vs__%s_t2048" % t
            for t in (("base", "fixedref_mse_s42"), ("base", "nbpo_mse_s42"),
                      ("fixedref_mse_s42", "nbpo_mse_s42"))]
put({"job_id": "uf4_crossplay_aggregate2", "priority": 61, "gpus": 0, "cwd": str(R),
     "env": {"PYTHONPATH": str(R / "code"), "OMP_NUM_THREADS": "4",
             "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"},
     "depends_on": existing + judges,
     "command": ["python3", str(R / "code/aggregate_crossplay.py")],
     "timeout_s": 7200,
     "artifacts": [str(R / "analysis/crossplay_summary.json")]})
