"""Queue global game-maxmin seeds 43 and 44, on the evidence the last cycle produced.

This control is now the only arm leading the objectives table, and it leads on
one seed. The fixed-reference control just showed what that is worth: at one
seed it led both three-seed means on all four attributes and was the only arm
with a minimum above 0.5; at two seeds it holds neither, because its second seed
moved honesty by 0.0126 and instruction following by 0.0112. The maxmin lead is
larger in seed-standard-deviation terms, but it rests on the same single
observation.

They go directly after the fixed-reference seeds and ahead of BT-RM and the DPO
arms, because the claim the paper now has to defend turns on this row.
"""
import json, pathlib
Q = pathlib.Path("/work/uf4_20260910/jobs/queue")
R = "/work/uf4_20260910"
S = json.load(open(R + "/jobs/state.json"))["jobs"]
CPU_ENV = {"PYTHONPATH": "/work/nbpo_repair_20260909/deps_train:/work/nbpo_repair_20260909/code:" + R + "/code",
           "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "8",
           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
# train priorities 66a/66b would collide with btrm at 66, so shift the later arms
SHIFT = {"uf4_train_btrm_mse_s42": 68, "uf4_train_smoke_dpo_uniform": 69,
         "uf4_train_dpo_uniform_mse_s42": 70, "uf4_train_dpo_if_only_mse_s42": 71,
         "uf4_train_dpo_truth_only_mse_s42": 72, "uf4_train_dpo_honesty_only_mse_s42": 73,
         "uf4_train_dpo_help_only_mse_s42": 74, "uf4_train_dpo_help_heavy_mse_s42": 75,
         "uf4_train_dpo_truth_heavy_mse_s42": 76, "uf4_train_prosper_mse_s42": 77}

jobs = []
for seed, train_prio, gen_prio, judge_prio in ((43, 66, 48, 49), (44, 67, 50, 51)):
    arm = "maxmin_mse_s%d" % seed
    jobs.append({"job_id": "uf4_make_train_maxmin_s%d" % seed, "priority": 56, "gpus": 0,
                 "depends_on": [], "cwd": "/work/nbpo_repair_20260909/code",
                 "command": ["python3", R + "/code/make_train_jobs.py",
                             "--targets", "maxmin_l1m_v1", "--arm", arm, "--seed", str(seed),
                             "--max-steps", "1250", "--priority", str(train_prio),
                             "--eval-steps", "250"],
                 "artifacts": [R + "/provenance/train_job_%s.json" % arm]})
    jobs.append({"job_id": "uf4_queue_finaleval_maxmin_s%d" % seed, "priority": 57, "gpus": 0,
                 "depends_on": ["uf4_make_train_maxmin_s%d" % seed], "cwd": R,
                 "command": ["python3", R + "/code/queue_arm_finaleval.py", "--arm", arm,
                             "--depends-on", "uf4_train_" + arm,
                             "--gen-priority", str(gen_prio), "--judge-priority", str(judge_prio)],
                 "artifacts": [R + "/jobs/queue/%d_finaleval_judge_%s.json" % (judge_prio, arm)]})

for spec in jobs:
    spec.setdefault("env", CPU_ENV)
    spec.setdefault("timeout_s", 3600)
    path = Q / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
    payload = json.dumps(spec, indent=2) + "\n"
    if path.exists() and path.read_text() != payload:
        raise SystemExit("refusing to change existing spec: %s" % path)
    if path.exists():
        print("unchanged", spec["job_id"]); continue
    path.write_text(payload)
    print("queued %-36s prio %2d" % (spec["job_id"], spec["priority"]))

for path in sorted(Q.glob("*.json")):
    spec = json.loads(path.read_text())
    jid = spec["job_id"]
    if jid not in SHIFT or spec.get("gpus", 0) == 0:
        continue
    if S.get(jid, {}).get("state") in ("RUNNING", "DONE"):
        print("skip (%s): %s" % (S[jid]["state"], jid)); continue
    if spec.get("priority") == SHIFT[jid]:
        continue
    old = spec.get("priority"); spec["priority"] = SHIFT[jid]
    path.write_text(json.dumps(spec, indent=2) + "\n")
    print("  %-38s %s -> %d" % (jid, old, SHIFT[jid]))
