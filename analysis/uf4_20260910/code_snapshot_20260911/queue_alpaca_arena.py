"""Queue AlpacaEval-2 and Arena-Hard-v2 for the UF arms, at zero API cost.

Both prompt sets are already on disk: data/alpaca_eval.jsonl with 804 prompts,
and data/arena_hard.jsonl with 749 -- 500 hard_prompt plus 250
creative_writing, which is the Arena-Hard-v2 composition. Scoring uses the
pinned local Skywork-Reward-V2-Qwen3-8B snapshot through the verified
evaluate_responses.py --mode skywork path. No paid API is involved.

What this produces is a local reward-model proxy win rate against the common
base, and the code says so itself: it stamps every report with
"Not AlpacaEval LC or official Arena-Hard score". The official numbers need a
GPT-4-class annotator, which is a paid API, so they cannot be produced here. The
column headings and caption must carry that distinction rather than implying a
leaderboard number.

Fresh labels per arm, because generate_eval compares the recorded benchmark list
in settings.json and refuses when it differs -- the same trap that would have
killed the HarmBench batch. assert_matched_protocol only compares decoding and
tokenizer fields, so a fresh label still validates against the existing base
responses, which already contain both benchmarks.

Priority 48/49 puts this immediately behind the running experiment's own
evaluation (maxmin seed 44 generation and judging at 46/47) and ahead of
everything else: the remaining cross-play pairs, BT-RM, the DPO arms and
PROSPER.
"""
import json, pathlib
R = pathlib.Path("/work/uf4_20260910")
REPAIR = pathlib.Path("/work/nbpo_repair_20260909")
Q = R / "jobs/queue"
SKYWORK = ("/work/hf_cache/hub/models--Skywork--Skywork-Reward-V2-Qwen3-8B/"
           "snapshots/6f19fdefb933293d4898bdb59a96f7223d998659")
ENV = {"PYTHONPATH": "/work/pylibs_eval:%s/code" % REPAIR,
       "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
       "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "OMP_NUM_THREADS": "4",
       "WANDB_MODE": "disabled"}
ARMS = ["nbpo_mse_s42", "nbpo_mse_s43", "nbpo_mse_s44",
        "util_mse_s42", "util_mse_s43", "util_mse_s44",
        "fixedref_mse_s42", "fixedref_mse_s43", "fixedref_mse_s44",
        "maxmin_mse_s42", "maxmin_mse_s43"]


def put(spec):
    p = Q / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
    payload = json.dumps(spec, indent=2) + "\n"
    if p.exists() and p.read_text() != payload:
        raise SystemExit("refusing to change existing spec: %s" % p)
    if p.exists():
        print("unchanged", spec["job_id"]); return
    p.write_text(payload)
    print("queued", spec["job_id"])


gen_ids = []
for arm in ARMS:
    jid = "uf4_genab_%s" % arm
    gen_ids.append(jid)
    put({"job_id": jid, "priority": 48, "gpus": 1,
         "cwd": str(REPAIR / "code"), "env": ENV, "depends_on": [],
         "command": ["python3", "-m",
                     "scripts.experiments.nbpo_repair_20260909.generate_eval",
                     "--root", str(REPAIR), "--model", str(R / "arms" / arm),
                     "--label", "uf4ab_%s" % arm,
                     "--benchmarks", "alpaca_eval", "arena_hard"],
         "timeout_s": 43200,
         "artifacts": [str(REPAIR / ("responses/uf4ab_%s/alpaca_eval.jsonl" % arm)),
                       str(REPAIR / ("responses/uf4ab_%s/arena_hard.jsonl" % arm))]})

put({"job_id": "uf4_score_alpaca_arena", "priority": 49, "gpus": 1,
     "cwd": str(REPAIR / "code"), "env": ENV, "depends_on": gen_ids,
     "command": ["python3", "-m",
                 "scripts.experiments.nbpo_repair_20260909.evaluate_responses",
                 "--root", str(REPAIR), "--mode", "skywork",
                 "--labels", *["uf4ab_%s" % a for a in ARMS],
                 "--base-label", "base", "--rm", SKYWORK,
                 "--benchmarks", "alpaca_eval", "arena_hard",
                 "--out", str(R / "analysis/alpaca_arena_uf4")],
     "timeout_s": 86400,
     "artifacts": [str(R / "analysis/alpaca_arena_uf4/evaluation_complete.json")]})
