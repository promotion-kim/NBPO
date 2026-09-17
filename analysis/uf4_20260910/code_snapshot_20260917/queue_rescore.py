"""Re-score the four cross-arm panels over every label that now exists.

These scorers need all labels at once, so they cannot be queued per arm: each
new family of seeds obsoletes the previous run. The earlier jobs are DONE and a
DONE job never re-runs, so each pass gets its own id carrying the arm count.

They are single-GPU and short, which is also what the queue needs: with every
remaining READY job a four-GPU training, the controller correctly holds three
cards whenever a judge occupies one, and short work is what absorbs that window.
"""
import json
import glob
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
QUEUE = ROOT / "jobs" / "queue"
REPAIR = "/work/nbpo_repair_20260909"
MOD = "scripts.experiments.nbpo_repair_20260909"
P = 67
RM = ("/work/hf_cache/hub/models--Skywork--Skywork-Reward-V2-Qwen3-8B/"
      "snapshots/6f19fdefb933293d4898bdb59a96f7223d998659")
HB_CLS = ("/work/hf_cache/hub/models--cais--HarmBench-Llama-2-13b-cls/"
          "snapshots/bda705349d1144fa618770bea64d99ce54e3835b")

EVAL_ENV = {"PYTHONPATH": f"/work/pylibs_eval:{REPAIR}/code", "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "OMP_NUM_THREADS": "4", "WANDB_MODE": "disabled"}
HB_ENV = dict(EVAL_ENV, PYTHONPATH=f"{REPAIR}/deps_harmbench/site:/work/pylibs_eval:{REPAIR}/code")
MT_ENV = dict(EVAL_ENV, PYTHONPATH="/work/pylibs_eval:/work/uf4_20260910/code")


def labels(prefix):
    return sorted(Path(p).name for p in glob.glob(f"{REPAIR}/responses/{prefix}*"))


def write(spec):
    existing = {json.loads(p.read_text())["job_id"] for p in QUEUE.glob("*.json")}
    if spec["job_id"] in existing:
        return "exists  " + spec["job_id"]
    (QUEUE / ("%d_%s.json" % (spec["priority"], spec["job_id"]))).write_text(
        json.dumps(spec, indent=2) + "\n")
    return "queued  %s  (%d labels)" % (spec["job_id"], spec.pop("_n"))


ab, mt, xs, hb = labels("uf4ab_"), labels("uf4mt_"), labels("uf4xs_"), labels("uf4hb_")
n = len(ab)
out = []
out.append(write({
    "job_id": f"uf4_score_alpaca_arena_{n}arm", "priority": P, "gpus": 1, "_n": len(ab),
    "cwd": REPAIR + "/code", "env": EVAL_ENV, "depends_on": [],
    "command": ["python3", "-m", MOD + ".evaluate_responses", "--root", REPAIR,
                "--mode", "skywork", "--labels", *ab, "--base-label", "base",
                "--rm", RM, "--benchmarks", "alpaca_eval", "arena_hard",
                "--out", str(ROOT / f"analysis/alpaca_arena_uf4_{n}arm")],
    "timeout_s": 86400,
    "artifacts": [str(ROOT / f"analysis/alpaca_arena_uf4_{n}arm/evaluation_complete.json")]}))
out.append(write({
    "job_id": f"uf4_score_mtbench_{len(mt)}arm", "priority": P, "gpus": 1, "_n": len(mt),
    "cwd": str(ROOT / "code"), "env": MT_ENV, "depends_on": [],
    "command": ["python3", "score_mtbench.py", "--labels", *mt,
                "--base-label", "uf4mt_base",
                "--out", str(ROOT / f"analysis/mtbench_{len(mt)}arm")],
    "timeout_s": 43200,
    "artifacts": [str(ROOT / f"analysis/mtbench_{len(mt)}arm/mtbench_local_scores.json"
                       if False else ROOT / f"analysis/mtbench_{len(mt)}arm/mtbench_scores.json")]}))
out.append(write({
    "job_id": f"uf4_score_xstest_{len(xs)}arm", "priority": P, "gpus": 1, "_n": len(xs),
    "cwd": str(ROOT / "code"), "env": MT_ENV, "depends_on": [],
    "command": ["python3", "score_xstest_local.py", "--labels", *xs,
                "--base-label", "base",
                "--out", str(ROOT / f"analysis/xstest_local_{len(xs)}arm")],
    "timeout_s": 43200,
    "artifacts": [str(ROOT / f"analysis/xstest_local_{len(xs)}arm/xstest_local_scores.json")]}))
out.append(write({
    "job_id": f"uf4_score_harmbench_{len(hb)}arm", "priority": P, "gpus": 1, "_n": len(hb),
    "cwd": REPAIR + "/code", "env": HB_ENV, "depends_on": [],
    "command": ["python3", "-m", MOD + ".evaluate_responses", "--root", REPAIR,
                "--mode", "harmbench", "--labels", *hb, "base", "--base-label", "base",
                "--harmbench-model", HB_CLS,
                "--harmbench-repo", REPAIR + "/external/HarmBench",
                "--out", str(ROOT / f"analysis/harmbench_uf4_{len(hb)}arm.json")],
    "timeout_s": 43200,
    "artifacts": [str(ROOT / f"analysis/harmbench_uf4_{len(hb)}arm.json")]}))
for line in out:
    print(line)
