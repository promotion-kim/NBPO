"""Queue HarmBench on the UF arms, under the protocol already recorded.

tab:general-capability currently carries a base HarmBench value of 0.28125 that
comes from /work/nbpo_repair_20260909/analysis_claude/capability.json -- the
SafeRLHF repair campaign -- while the IFEval and GSM8K cells in the same row are
UF. The appendix says elsewhere that the UF and SafeRLHF panels use different
recorded protocols and are not pooled, so a row that mixes them is exactly what
that sentence warns against. Measuring HarmBench on the UF arms and on the UF
base under one protocol replaces a borrowed number with a measured one.

The protocol is not invented here. It is the one recorded in
/work/nbpo_downstream_local_v1/scores/harmbench.json: the official
cais/HarmBench-Llama-2-13b-cls classifier, the direct-request harmful-completion
rate over the 320 text behaviours, a 512-token classifier budget, and the
explicit disclaimer that with no jailbreak template and no attack search this is
not adversarial robustness. Generation and scoring both go through the verified
evaluate_responses.py --mode harmbench path.

Single-GPU jobs at backfill priority, so they fill judge windows rather than
displacing a training.
"""
import json, pathlib
R = pathlib.Path("/work/uf4_20260910")
REPAIR = pathlib.Path("/work/nbpo_repair_20260909")
Q = R / "jobs/queue"
CLS = ("/work/hf_cache/hub/models--cais--HarmBench-Llama-2-13b-cls/"
       "snapshots/bda705349d1144fa618770bea64d99ce54e3835b")
ENV = {"PYTHONPATH": "/work/pylibs_eval:%s/code" % REPAIR,
       "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
       "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "OMP_NUM_THREADS": "4",
       "WANDB_MODE": "disabled"}
ARMS = ["nbpo_mse_s42", "nbpo_mse_s43", "nbpo_mse_s44",
        "util_mse_s42", "util_mse_s43", "util_mse_s44",
        "fixedref_mse_s42", "fixedref_mse_s43", "fixedref_mse_s44",
        "maxmin_mse_s42"]

gen_ids = []
for arm in ARMS:
    jid = "uf4_genhb_%s" % arm
    gen_ids.append(jid)
    spec = {"job_id": jid, "priority": 64, "gpus": 1,
            "cwd": str(REPAIR / "code"), "env": ENV, "depends_on": [],
            "command": ["python3", "-m",
                        "scripts.experiments.nbpo_repair_20260909.generate_eval",
                        "--root", str(REPAIR), "--model", str(R / "arms" / arm),
                        "--label", "uf4_%s" % arm, "--benchmarks", "harmbench"],
            "timeout_s": 21600,
            "artifacts": [str(REPAIR / ("responses/uf4_%s/harmbench.jsonl" % arm))]}
    p = Q / ("%d_%s.json" % (spec["priority"], jid))
    payload = json.dumps(spec, indent=2) + "\n"
    if p.exists() and p.read_text() != payload:
        raise SystemExit("refusing to change existing spec: %s" % p)
    if p.exists():
        print("unchanged", jid); continue
    p.write_text(payload)
    print("queued", jid)

score = {"job_id": "uf4_score_harmbench", "priority": 65, "gpus": 1,
         "cwd": str(REPAIR / "code"), "env": ENV, "depends_on": gen_ids,
         "command": ["python3", "-m",
                     "scripts.experiments.nbpo_repair_20260909.evaluate_responses",
                     "--root", str(REPAIR), "--mode", "harmbench",
                     "--labels", *["uf4_%s" % a for a in ARMS], "base",
                     "--base-label", "base",
                     "--harmbench-model", CLS,
                     "--harmbench-repo", str(REPAIR / "external/HarmBench"),
                     "--out", str(R / "analysis/harmbench_uf4.json")],
         "timeout_s": 43200,
         "artifacts": [str(R / "analysis/harmbench_uf4.json")]}
p = Q / ("%d_%s.json" % (score["priority"], score["job_id"]))
payload = json.dumps(score, indent=2) + "\n"
if p.exists() and p.read_text() != payload:
    raise SystemExit("refusing to change existing scoring spec")
if p.exists():
    print("unchanged", score["job_id"])
else:
    p.write_text(payload)
    print("queued %s (deps %d gens)" % (score["job_id"], len(gen_ids)))
