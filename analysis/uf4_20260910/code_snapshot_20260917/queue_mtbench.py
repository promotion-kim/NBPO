"""Queue the three new Table-3 columns and put them where the user asked.

Placement. The user asked for AlpacaEval-2 / Arena-Hard-v2 / MT-Bench "after the
current experiment, at the front of the queue". The current experiment is the
maxmin s44 arm, and an arm is not finished until its final evaluation has been
judged -- that is the number the mechanism table reports. So this band sits at
65/66: after every already-trained arm's final evaluation (42-63) and the
cross-play pairs (64), and ahead of every remaining training (68+). Left at 48
the eleven generation jobs would have taken all four cards the moment s44
stopped training and pushed s44's own minimum-objective result back by about
ninety minutes, which is the opposite of what "after the current experiment"
means.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

QUEUE = Path("/work/uf4_20260910/jobs/queue")
GEN_PRIORITY, SCORE_PRIORITY = 65, 66
ARMS = {
    "base": "/work/models/bases/Llama-3.1-8B-Instruct",
    **{a: f"/work/uf4_20260910/arms/{a}" for a in (
        "nbpo_mse_s42", "nbpo_mse_s43", "nbpo_mse_s44",
        "util_mse_s42", "util_mse_s43", "util_mse_s44",
        "fixedref_mse_s42", "fixedref_mse_s43", "fixedref_mse_s44",
        "maxmin_mse_s42", "maxmin_mse_s43", "maxmin_mse_s44")},
}
# The only arm that does not exist on disk yet; its generation waits on the run
# that is producing it rather than being queued blind.
TRAIN_DEP = {"maxmin_mse_s44": ["uf4_train_maxmin_mse_s44"]}
EVAL_ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/nbpo_repair_20260909/code",
            "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "OMP_NUM_THREADS": "4",
            "WANDB_MODE": "disabled"}
MT_ENV = dict(EVAL_ENV, PYTHONPATH="/work/pylibs_eval:/work/uf4_20260910/code")


def write_spec(priority, spec):
    """Replace any spec for this job id, then write it under its new name.

    The old file is removed first: two files carrying one job id make
    ``load_specs`` raise and the controller then schedules nothing that cycle.
    A spec that is briefly absent is harmless -- it is simply not considered
    until the next poll, and recorded state is untouched.
    """
    for old in QUEUE.glob("*.json"):
        if json.loads(old.read_text())["job_id"] == spec["job_id"]:
            os.unlink(old)
    path = QUEUE / f"{priority}_{spec['job_id']}.json"
    tmp = QUEUE / f".{spec['job_id']}.tmp"
    tmp.write_text(json.dumps({"priority": priority, **spec}, indent=2) + "\n")
    os.replace(tmp, path)
    return path.name


def main():
    changed = []

    # 1. Move the already-queued AlpacaEval-2 / Arena-Hard-v2 generation band.
    for path in sorted(QUEUE.glob("*_uf4_genab_*.json")):
        spec = json.loads(path.read_text())
        if spec.get("priority") != GEN_PRIORITY:
            changed.append(("moved", write_spec(GEN_PRIORITY, spec)))

    # 2. The twelfth arm, which did not exist when that band was queued.
    arm = "maxmin_mse_s44"
    changed.append(("added", write_spec(GEN_PRIORITY, {
        "job_id": f"uf4_genab_{arm}", "gpus": 1,
        "cwd": "/work/nbpo_repair_20260909/code", "env": EVAL_ENV,
        "depends_on": TRAIN_DEP[arm],
        "command": ["python3", "-m", "scripts.experiments.nbpo_repair_20260909.generate_eval",
                    "--root", "/work/nbpo_repair_20260909", "--model", ARMS[arm],
                    "--label", f"uf4ab_{arm}", "--benchmarks", "alpaca_eval", "arena_hard"],
        "timeout_s": 43200,
        "artifacts": [f"/work/nbpo_repair_20260909/responses/uf4ab_{arm}/{b}.jsonl"
                      for b in ("alpaca_eval", "arena_hard")]})))

    # 3. MT-Bench generation: base plus every arm, two turns each.
    mt_labels, mt_jobs = [], []
    for name, model in ARMS.items():
        job = f"uf4_genmt_{name}"
        mt_jobs.append(job)
        mt_labels.append(f"uf4mt_{name}")
        changed.append(("added", write_spec(GEN_PRIORITY, {
            "job_id": job, "gpus": 1, "cwd": "/work/uf4_20260910/code", "env": MT_ENV,
            "depends_on": TRAIN_DEP.get(name, []),
            "command": ["python3", "generate_mtbench.py", "--root", "/work/nbpo_repair_20260909",
                        "--model", model, "--label", f"uf4mt_{name}"],
            "timeout_s": 21600,
            "artifacts": [f"/work/nbpo_repair_20260909/responses/uf4mt_{name}/mt_bench.jsonl"]})))

    # 4. Scoring. Both scorers wait for every arm so each column lands complete
    #    rather than as a partial column that would have to be revised.
    arena_labels = [f"uf4ab_{a}" for a in ARMS if a != "base"]
    changed.append(("moved+extended", write_spec(SCORE_PRIORITY, {
        "job_id": "uf4_score_alpaca_arena", "gpus": 1,
        "cwd": "/work/nbpo_repair_20260909/code", "env": EVAL_ENV,
        "depends_on": [f"uf4_genab_{a}" for a in ARMS if a != "base"],
        "command": ["python3", "-m", "scripts.experiments.nbpo_repair_20260909.evaluate_responses",
                    "--root", "/work/nbpo_repair_20260909", "--mode", "skywork",
                    "--labels", *arena_labels, "--base-label", "base",
                    "--rm", "/work/hf_cache/hub/models--Skywork--Skywork-Reward-V2-Qwen3-8B/"
                            "snapshots/6f19fdefb933293d4898bdb59a96f7223d998659",
                    "--benchmarks", "alpaca_eval", "arena_hard",
                    "--out", "/work/uf4_20260910/analysis/alpaca_arena_uf4"],
        "timeout_s": 86400,
        "artifacts": ["/work/uf4_20260910/analysis/alpaca_arena_uf4/evaluation_complete.json"]})))

    changed.append(("added", write_spec(SCORE_PRIORITY, {
        "job_id": "uf4_score_mtbench", "gpus": 1, "cwd": "/work/uf4_20260910/code",
        "env": MT_ENV, "depends_on": mt_jobs,
        "command": ["python3", "score_mtbench.py", "--labels", *mt_labels,
                    "--base-label", "uf4mt_base",
                    "--out", "/work/uf4_20260910/analysis/mtbench"],
        "timeout_s": 43200,
        "artifacts": ["/work/uf4_20260910/analysis/mtbench/mtbench_scores.json"]})))

    for action, name in changed:
        print(f"{action:16s} {name}")
    print(f"\n{len(changed)} specs written at priority {GEN_PRIORITY}/{SCORE_PRIORITY}")


if __name__ == "__main__":
    main()
