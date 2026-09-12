"""Re-queue HarmBench under fresh labels, because the generator compares settings.

generate_eval writes responses/<label>/settings.json including the benchmark
list, and refuses to run when an existing settings.json differs:

    if json.loads(settings_path.read_text()) != settings:
        raise ValueError("Existing response run has different settings")

The uf4_<arm> directories already hold settings recorded with
benchmarks ["ifeval", "gsm8k"], so asking for --benchmarks harmbench under the
same label would have raised on all ten jobs. Passing the stored list instead
would not generate harmbench at all, since that list does not contain it.

A separate label per arm is the clean route: a fresh directory has no settings
to conflict with, and the two runs stay separately auditable rather than one
overwriting the other's provenance. The model, tokenizer, chat template,
decoding and seed are identical either way, so this changes bookkeeping and not
the measurement.
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

# retire the specs that would have raised
retired = 0
for path in sorted(Q.glob("*.json")):
    spec = json.loads(path.read_text())
    if spec["job_id"].startswith("uf4_genhb_") or spec["job_id"] == "uf4_score_harmbench":
        path.rename(path.with_suffix(".json.superseded_label_conflict"))
        retired += 1
print("retired %d specs that would have hit the settings check" % retired)

gen_ids = []
for arm in ARMS:
    jid = "uf4_genhb2_%s" % arm
    gen_ids.append(jid)
    spec = {"job_id": jid, "priority": 64, "gpus": 1,
            "cwd": str(REPAIR / "code"), "env": ENV, "depends_on": [],
            "command": ["python3", "-m",
                        "scripts.experiments.nbpo_repair_20260909.generate_eval",
                        "--root", str(REPAIR), "--model", str(R / "arms" / arm),
                        "--label", "uf4hb_%s" % arm, "--benchmarks", "harmbench"],
            "timeout_s": 21600,
            "artifacts": [str(REPAIR / ("responses/uf4hb_%s/harmbench.jsonl" % arm))]}
    p = Q / ("%d_%s.json" % (spec["priority"], jid))
    payload = json.dumps(spec, indent=2) + "\n"
    if p.exists() and p.read_text() != payload:
        raise SystemExit("refusing to change existing spec: %s" % p)
    if not p.exists():
        p.write_text(payload)
print("queued %d harmbench generations under uf4hb_* labels" % len(gen_ids))

score = {"job_id": "uf4_score_harmbench2", "priority": 65, "gpus": 1,
         "cwd": str(REPAIR / "code"), "env": ENV, "depends_on": gen_ids,
         "command": ["python3", "-m",
                     "scripts.experiments.nbpo_repair_20260909.evaluate_responses",
                     "--root", str(REPAIR), "--mode", "harmbench",
                     "--labels", *["uf4hb_%s" % a for a in ARMS], "base",
                     "--base-label", "base",
                     "--harmbench-model", CLS,
                     "--harmbench-repo", str(REPAIR / "external/HarmBench"),
                     "--out", str(R / "analysis/harmbench_uf4.json")],
         "timeout_s": 43200,
         "artifacts": [str(R / "analysis/harmbench_uf4.json")]}
p = Q / ("%d_%s.json" % (score["priority"], score["job_id"]))
if not p.exists():
    p.write_text(json.dumps(score, indent=2) + "\n")
print("queued", score["job_id"], "deps", len(gen_ids))
