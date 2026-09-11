"""Queue the development-only judging that selects the cross-play competitor.

The declared rule is the minimum-objective win rate against the base under the
frozen independent judge on policy_dev -- 1,000 prompts that have never produced
a final-eval number. pools/dev_v1 is already the base sampled on exactly those
prompts under the same contract (same checkpoint and revision, temperature 1.0,
top_p 1.0, 1024 tokens), so it serves as the reference with no new generation and
judge_final_eval is reused unchanged with --base-pool pointed at it.

Output arms are named devsel_<arm> so these results live beside, and can never
be mistaken for, the final-eval numbers.

One representative seed per candidate method: comparing families on a single
shared seed keeps the selection about the method rather than about which family
happened to get more seeds first. Later arms (BT-RM, scalarized DPO, PROSPER)
are queued the same way as they finish training.
"""
import json, pathlib
R = pathlib.Path("/work/uf4_20260910")
Q = R / "jobs/queue"
GEN_ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/uf4_20260910/code",
           "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "OMP_NUM_THREADS": "4",
           "WANDB_MODE": "disabled"}
CANDIDATES = ["util_mse_s42", "maxmin_mse_s42"]

for arm in CANDIDATES:
    gen = {"job_id": "uf4_devsel_gen_%s" % arm, "priority": 64, "gpus": 1,
           "cwd": str(R), "env": GEN_ENV, "depends_on": [],
           "command": ["python3", str(R / "code/generate_uf4_pool.py"),
                       "--splits", "v1", "--split-names", "policy_dev",
                       "--out-name", "devsel_%s" % arm, "--model", str(R / "arms" / arm),
                       "--shard", "0", "--shards", "1", "--prompts-per-chunk", "25"],
           "timeout_s": 21600,
           "artifacts": [str(R / ("pools/devsel_%s/shard0/complete_shard0.json" % arm))]}
    judge = {"job_id": "uf4_devsel_judge_%s" % arm, "priority": 64, "gpus": 1,
             "cwd": str(R), "env": GEN_ENV,
             "depends_on": ["uf4_devsel_gen_%s" % arm],
             "command": ["python3", str(R / "code/judge_final_eval.py"),
                         "--arm", "devsel_%s" % arm,
                         "--policy-pool", str(R / ("pools/devsel_%s" % arm)),
                         "--base-pool", str(R / "pools/dev_v1")],
             "timeout_s": 43200,
             "artifacts": [str(R / ("evaluation/final_eval/devsel_%s/complete.json" % arm))]}
    for spec in (gen, judge):
        p = Q / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
        payload = json.dumps(spec, indent=2) + "\n"
        if p.exists() and p.read_text() != payload:
            raise SystemExit("refusing to change existing spec: %s" % p)
        if p.exists():
            print("unchanged", spec["job_id"]); continue
        p.write_text(payload)
        print("queued %-38s gpus %d deps %s" % (spec["job_id"], spec["gpus"],
                                                spec["depends_on"] or "-"))
