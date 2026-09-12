"""Queue the Table-3 capability cells that fixed-reference Nash and global game-maxmin never had.

Table 3 stands at 11 measured cells of 36. The gap is not about arms that do
not exist: all six of these checkpoints have been trained and evaluated on the
UF-4 panel for hours. IFEval, GSM8K, ARC-C, HellaSwag and MMLU were simply
never queued for them, and HarmBench was queued for only one maxmin seed, so
the safety row that the manuscript leans on ("no mechanism is exempt") rests on
one seed for one of the four families.

Every spec below copies an existing, proven spec for the same script and only
substitutes the arm: same environments, same labels convention, same artifacts.
The two things checked before writing them were that no `uf4_<arm>` response
directory already exists under a different benchmark list -- which is what
killed the first HarmBench batch -- and that the deterministic scorer is a
0-GPU job, so it never competes for a card.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
QUEUE = ROOT / "jobs" / "queue"
ARMS = ["fixedref_mse_s42", "fixedref_mse_s43", "fixedref_mse_s44",
        "maxmin_mse_s42", "maxmin_mse_s43", "maxmin_mse_s44"]
HB_ARMS = ["maxmin_mse_s43", "maxmin_mse_s44"]          # the two the safety row lacks
ALL_HB = ["nbpo_mse_s42", "nbpo_mse_s43", "nbpo_mse_s44",
          "util_mse_s42", "util_mse_s43", "util_mse_s44",
          "fixedref_mse_s42", "fixedref_mse_s43", "fixedref_mse_s44",
          "maxmin_mse_s42", "maxmin_mse_s43", "maxmin_mse_s44"]
TASKS = ["arc_challenge", "hellaswag", "mmlu"]
PRIORITY = 67          # after the running BT-RM arm's own evaluation, before the next training

BENCH_ENV = {"PYTHONPATH": "/work/nbpo_repair_20260909/code:/work/pylibs_eval:"
                           "/work/pylibs_ifeval:/work/nbpo_downstream_local_v1/code",
             "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
             "TOKENIZERS_PARALLELISM": "false",
             "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "OMP_NUM_THREADS": "4",
             "WANDB_MODE": "disabled", "MNPO_DISABLE_APEX": "1"}
DET_ENV = {"PYTHONPATH": "/work/nbpo_repair_20260909/code:/work/pylibs_eval:"
                         "/work/pylibs_ifeval:/work/nbpo_downstream_local_v1/code",
           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4",
           "WANDB_MODE": "disabled"}
LM_ENV = {"PYTHONPATH": "/work/pylibs_lmeval:/work/pylibs_eval:/work/uf4_20260910/code",
          "HF_HUB_OFFLINE": "0", "TOKENIZERS_PARALLELISM": "false",
          "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "OMP_NUM_THREADS": "4",
          "WANDB_MODE": "disabled", "HF_DATASETS_OFFLINE": "0",
          "HF_HOME": "/work/hf_cache"}
HB_ENV = {"PYTHONPATH": "/work/nbpo_repair_20260909/deps_harmbench/site:/work/pylibs_eval:"
                        "/work/nbpo_repair_20260909/code",
          "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
          "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "OMP_NUM_THREADS": "4",
          "WANDB_MODE": "disabled"}
GEN_ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/nbpo_repair_20260909/code",
           "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "OMP_NUM_THREADS": "4",
           "WANDB_MODE": "disabled"}
REPAIR = "/work/nbpo_repair_20260909"
EVAL_MOD = "scripts.experiments.nbpo_repair_20260909"


def write(spec):
    existing = [p for p in QUEUE.glob("*.json")
                if json.loads(p.read_text())["job_id"] == spec["job_id"]]
    if existing:
        return "exists   " + spec["job_id"]
    path = QUEUE / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
    path.write_text(json.dumps(spec, indent=2) + "\n")
    return "queued   " + path.name


def main():
    out = []
    for arm in ARMS:
        directory = Path(REPAIR) / "responses" / ("uf4_" + arm)
        if directory.exists():
            raise SystemExit("refusing to reuse the label %s: a response run already exists "
                             "there and generate_eval rejects a different benchmark list"
                             % directory.name)
        out.append(write({
            "job_id": "uf4_gen_bench_" + arm, "priority": PRIORITY, "gpus": 1,
            "cwd": REPAIR + "/code", "env": BENCH_ENV, "depends_on": [],
            "command": ["python3", "-m", EVAL_MOD + ".generate_eval", "--root", REPAIR,
                        "--model", str(ROOT / "arms" / arm), "--label", "uf4_" + arm,
                        "--benchmarks", "ifeval", "gsm8k"],
            "timeout_s": 21600,
            "artifacts": ["%s/responses/uf4_%s/%s.jsonl" % (REPAIR, arm, b)
                          for b in ("ifeval", "gsm8k")]}))
        out.append(write({
            "job_id": "uf4_det_" + arm, "priority": PRIORITY, "gpus": 0,
            "cwd": REPAIR + "/code", "env": DET_ENV,
            "depends_on": ["uf4_gen_bench_" + arm],
            "command": ["python3", "-m", EVAL_MOD + ".evaluate_responses", "--root", REPAIR,
                        "--mode", "deterministic", "--labels", "uf4_" + arm,
                        "--base-label", "base", "--benchmarks", "ifeval", "gsm8k",
                        "--out", "%s/evaluations/deterministic_uf4_%s_v1" % (REPAIR, arm)],
            "timeout_s": 7200,
            "artifacts": ["%s/evaluations/deterministic_uf4_%s_v1/evaluation_complete.json"
                          % (REPAIR, arm)]}))
        for task in TASKS:
            out.append(write({
                "job_id": "uf4_cap_%s_%s" % (arm, task), "priority": PRIORITY, "gpus": 1,
                "cwd": str(ROOT), "env": LM_ENV, "depends_on": [],
                "command": ["python3", str(ROOT / "code/run_lm_eval.py"),
                            "--model-path", str(ROOT / "arms" / arm),
                            "--arm", arm, "--task", task],
                "timeout_s": 21600,
                "artifacts": [str(ROOT / "evaluation/capability" / arm / task / "results.json")]}))

    for arm in HB_ARMS:
        out.append(write({
            "job_id": "uf4_genhb3_" + arm, "priority": PRIORITY, "gpus": 1,
            "cwd": REPAIR + "/code", "env": GEN_ENV, "depends_on": [],
            "command": ["python3", "-m", EVAL_MOD + ".generate_eval", "--root", REPAIR,
                        "--model", str(ROOT / "arms" / arm), "--label", "uf4hb_" + arm,
                        "--benchmarks", "harmbench"],
            "timeout_s": 21600,
            "artifacts": ["%s/responses/uf4hb_%s/harmbench.jsonl" % (REPAIR, arm)]}))

    # One scoring job over all twelve arms rather than an increment, so the
    # safety row is a single artifact with one classifier run behind it.
    out.append(write({
        "job_id": "uf4_score_harmbench3", "priority": PRIORITY, "gpus": 1,
        "cwd": REPAIR + "/code", "env": HB_ENV,
        "depends_on": (["uf4_genhb2_" + a for a in ALL_HB if a not in HB_ARMS]
                       + ["uf4_genhb3_" + a for a in HB_ARMS]),
        "command": (["python3", "-m", EVAL_MOD + ".evaluate_responses", "--root", REPAIR,
                     "--mode", "harmbench", "--labels"]
                    + ["uf4hb_" + a for a in ALL_HB] + ["base", "--base-label", "base",
                       "--harmbench-model", "/work/hf_cache/hub/models--cais--HarmBench-"
                       "Llama-2-13b-cls/snapshots/bda705349d1144fa618770bea64d99ce54e3835b",
                       "--harmbench-repo", REPAIR + "/external/HarmBench",
                       "--out", str(ROOT / "analysis/harmbench_uf4_12arm.json")]),
        "timeout_s": 43200,
        "artifacts": [str(ROOT / "analysis/harmbench_uf4_12arm.json")],
        "env_note": "spaCy for HarmBench's eval_utils lives in deps_harmbench/site"}))

    for line in out:
        print(line)
    print("\n%d specs, %d newly queued at priority %d"
          % (len(out), sum(1 for o in out if o.startswith("queued")), PRIORITY))


if __name__ == "__main__":
    main()
