"""Queue one arm's whole Table-3 panel, so judge windows have work to absorb them.

Three of the four cards sat idle for the length of BT-RM s43's judge: the
controller was correctly holding them for the next 4-GPU training, and every
READY job was a 4-GPU training, so there was nothing to backfill with. That is
not a scheduler fault, it is a queue that has run out of small work -- the
capability and arena bands were queued for the twelve arms that existed at the
time, and BT-RM's Table 3 row is still entirely empty.

Queueing each new arm's panel fixes both at once: the row gets filled, and every
future judge window has short single-GPU work to absorb. Priority 67 puts these
ahead of the trainings, which costs the next training a few minutes rather than
leaving three cards at zero for twenty-four.

The cross-arm scorers (AlpacaEval-2/Arena-Hard, MT-Bench, XSTest) are not queued
here: they need every label at once, so they run after a family's seeds are all
present rather than three times over.
"""
import json
import sys
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
QUEUE = ROOT / "jobs" / "queue"
REPAIR = "/work/nbpo_repair_20260909"
MOD = "scripts.experiments.nbpo_repair_20260909"
P = 67

BENCH_ENV = {"PYTHONPATH": f"{REPAIR}/code:/work/pylibs_eval:/work/pylibs_ifeval:"
                           "/work/nbpo_downstream_local_v1/code",
             "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
             "TOKENIZERS_PARALLELISM": "false", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
             "OMP_NUM_THREADS": "4", "WANDB_MODE": "disabled", "MNPO_DISABLE_APEX": "1"}
DET_ENV = {k: v for k, v in BENCH_ENV.items()
           if k not in ("VLLM_WORKER_MULTIPROC_METHOD", "MNPO_DISABLE_APEX")}
LM_ENV = {"PYTHONPATH": "/work/pylibs_lmeval:/work/pylibs_eval:/work/uf4_20260910/code",
          "HF_HUB_OFFLINE": "0", "TOKENIZERS_PARALLELISM": "false",
          "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "OMP_NUM_THREADS": "4",
          "WANDB_MODE": "disabled", "HF_DATASETS_OFFLINE": "0", "HF_HOME": "/work/hf_cache"}
GEN_ENV = {"PYTHONPATH": f"/work/pylibs_eval:{REPAIR}/code", "HF_HUB_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
           "OMP_NUM_THREADS": "4", "WANDB_MODE": "disabled"}
MT_ENV = dict(GEN_ENV, PYTHONPATH="/work/pylibs_eval:/work/uf4_20260910/code")


def write(spec):
    existing = {json.loads(p.read_text())["job_id"] for p in QUEUE.glob("*.json")}
    if spec["job_id"] in existing:
        return "exists  " + spec["job_id"]
    (QUEUE / ("%d_%s.json" % (spec["priority"], spec["job_id"]))).write_text(
        json.dumps(spec, indent=2) + "\n")
    return "queued  " + spec["job_id"]


def panel(arm, depends):
    model = str(ROOT / "arms" / arm)
    out = []
    for label, benches, prefix in (("uf4_" + arm, ["ifeval", "gsm8k"], "gen_bench"),
                                   ("uf4hb_" + arm, ["harmbench"], "genhb3"),
                                   ("uf4ab_" + arm, ["alpaca_eval", "arena_hard"], "genab"),
                                   ("uf4xs_" + arm, ["xstest"], "genxs")):
        directory = Path(REPAIR) / "responses" / label
        if directory.exists():
            out.append("exists  responses/" + label)
            continue
        out.append(write({
            "job_id": f"uf4_{prefix}_{arm}", "priority": P, "gpus": 1,
            "cwd": REPAIR + "/code",
            "env": BENCH_ENV if prefix == "gen_bench" else GEN_ENV,
            "depends_on": depends,
            "command": ["python3", "-m", MOD + ".generate_eval", "--root", REPAIR,
                        "--model", model, "--label", label, "--benchmarks", *benches],
            "timeout_s": 21600,
            "artifacts": ["%s/responses/%s/%s.jsonl" % (REPAIR, label, b) for b in benches]}))
    out.append(write({
        "job_id": "uf4_det_" + arm, "priority": P, "gpus": 0,
        "cwd": REPAIR + "/code", "env": DET_ENV,
        "depends_on": [f"uf4_gen_bench_{arm}"],
        "command": ["python3", "-m", MOD + ".evaluate_responses", "--root", REPAIR,
                    "--mode", "deterministic", "--labels", "uf4_" + arm,
                    "--base-label", "base", "--benchmarks", "ifeval", "gsm8k",
                    "--out", "%s/evaluations/deterministic_uf4_%s_v1" % (REPAIR, arm)],
        "timeout_s": 7200,
        "artifacts": ["%s/evaluations/deterministic_uf4_%s_v1/evaluation_complete.json"
                      % (REPAIR, arm)]}))
    out.append(write({
        "job_id": f"uf4_genmt_{arm}", "priority": P, "gpus": 1,
        "cwd": str(ROOT / "code"), "env": MT_ENV, "depends_on": depends,
        "command": ["python3", "generate_mtbench.py", "--root", REPAIR,
                    "--model", model, "--label", "uf4mt_" + arm],
        "timeout_s": 21600,
        "artifacts": ["%s/responses/uf4mt_%s/mt_bench.jsonl" % (REPAIR, arm)]}))
    for task in ("arc_challenge", "hellaswag", "mmlu"):
        out.append(write({
            "job_id": f"uf4_cap_{arm}_{task}", "priority": P, "gpus": 1,
            "cwd": str(ROOT), "env": LM_ENV, "depends_on": depends,
            "command": ["python3", str(ROOT / "code/run_lm_eval.py"),
                        "--model-path", model, "--arm", arm, "--task", task],
            "timeout_s": 21600,
            "artifacts": [str(ROOT / "evaluation/capability" / arm / task / "results.json")]}))
    return out


if __name__ == "__main__":
    arm = sys.argv[1]
    depends = sys.argv[2:] if len(sys.argv) > 2 else []
    if not (ROOT / "arms" / arm / "config.json").exists() and not depends:
        raise SystemExit("%s has no exported checkpoint and no dependency given" % arm)
    for line in panel(arm, depends):
        print(line)
