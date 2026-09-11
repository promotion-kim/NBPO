"""Aggregate the real GPU-hours each UF-4 method cost, from the queue's own log.

Every job's RUNNING and terminal events are timestamped in logs/queue_report.jsonl
and its device count is in the launch record, so cost is measured rather than
estimated: GPU-hours = wall seconds * devices held. Jobs that held no device are
reported as CPU-hours in a separate column, because app:feedback-cost requires
solver CPU time to be reported separately from GPU-hours.

Stages are attributed the way the appendix defines them:

  teacher    fitting and scoring the frozen preference model, plus generating and
             scoring the candidate pool. This is shared infrastructure: the same
             pool and the same two teacher heads serve every arm, so it is
             reported once as a shared cost and NOT divided among the methods.
             Splitting it would let a method look cheap by arriving late.
  solver     the per-method finite-pool solve. CPU only.
  policy     the neural training run for that arm.
  gen_eval   fresh response generation and the independent judge for that arm.

Unique labels are deliberately left out. The appendix distinguishes unique
semantic comparisons from order-swapped calls, and that convention decides
whether a reversed-order judgement counts once or twice; it is a definition to
declare, not a number to infer from a log, so this script does not guess one.
"""
from __future__ import annotations

import json, re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
ARM_METHOD = {"nbpo_mse": "NBPO", "util_mse": "Game-utilitarian",
              "fixedref_mse": "Fixed-reference Nash", "btrm_mse": "BT-RM--Nash",
              "maxmin_mse": "Global game-maxmin", "prosper_mse": "PROSPER (adapt.)",
              "mopo_mse": "MOPO (adapt.)"}
SOLVE_METHOD = {"uf4_solve_nash_v1": "NBPO", "uf4_solve_util_v1": "Game-utilitarian",
                "uf4_solve_fixedref_v1": "Fixed-reference Nash",
                "uf4_solve_btrm_v1": "BT-RM--Nash",
                "uf4_solve_maxmin_v1": "Global game-maxmin"}
SHARED = ("teacher_", "uf4_pool_", "pool_train", "pool_dev", "uf4_score_", "score_train",
          "score_dev", "uf4_det_", "uf4_audit_", "uf4_regress_", "uf4_build_dpo_pairs")


def parse(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")


def durations():
    """job_id -> (seconds, devices, final state), from the controller's own events."""
    start, out = {}, {}
    for line in (ROOT / "logs/queue_report.jsonl").open():
        e = json.loads(line)
        jid = e["job_id"]
        if e["state"] == "RUNNING":
            start[jid] = e["time"]
            m = re.search(r"devices=\[([^\]]*)\]", e.get("detail", "") or "")
            out.setdefault(jid, {})["devices"] = (
                len([x for x in m.group(1).split(",") if x.strip()]) if m else 0)
        elif e["state"] in ("DONE", "FAILED") and jid in start:
            secs = (parse(e["time"]) - parse(start[jid])).total_seconds()
            rec = out.setdefault(jid, {})
            rec["seconds"] = rec.get("seconds", 0.0) + secs
            rec["state"] = e["state"]
    return {j: r for j, r in out.items() if "seconds" in r}


def classify(job_id):
    """(method, stage) for a job, or (None, None) when it is not a cost we attribute."""
    if any(k in job_id for k in SHARED):
        return "_shared", "teacher"
    if job_id in SOLVE_METHOD:
        return SOLVE_METHOD[job_id], "solver"
    if job_id.startswith("uf4_materialize_dpo_") or job_id.startswith("uf4_make_dpo_train_"):
        return "DPO (uniform / sweep)", "solver"
    body = None
    for prefix in ("uf4_train_", "uf4_finaleval_gen_", "uf4_finaleval_judge_",
                   "uf4_cap_", "uf4_gen_bench_"):
        if job_id.startswith(prefix):
            body = job_id[len(prefix):]
            stage = "policy" if prefix == "uf4_train_" else "gen_eval"
            break
    if body is None:
        return None, None
    if body.startswith("dpo_") or body.startswith("smoke_dpo_"):
        return "DPO (uniform / sweep)", stage
    for arm_prefix, method in ARM_METHOD.items():
        if body.startswith(arm_prefix):
            return method, stage
    if body.startswith("base"):
        return "_shared", "gen_eval"
    return None, None


def main():
    dur = durations()
    cost = defaultdict(lambda: defaultdict(lambda: {"gpu_h": 0.0, "cpu_h": 0.0, "jobs": 0}))
    unattributed = []
    for job_id, rec in sorted(dur.items()):
        method, stage = classify(job_id)
        if method is None:
            unattributed.append(job_id)
            continue
        hours = rec["seconds"] / 3600.0
        devices = rec.get("devices", 0)
        entry = cost[method][stage]
        if devices:
            entry["gpu_h"] += hours * devices
        else:
            entry["cpu_h"] += hours
        entry["jobs"] += 1

    report = {"source": str(ROOT / "logs/queue_report.jsonl"),
              "definition": ("GPU-hours = wall seconds x devices held, from the controller's "
                             "RUNNING and terminal events. Jobs holding no device are counted "
                             "as CPU-hours and reported separately, as the appendix requires."),
              "shared_note": ("the frozen teacher heads, the candidate pool and its scoring "
                              "serve every arm; reported once under _shared and not divided "
                              "among methods"),
              "unique_labels": ("not computed: the appendix distinguishes unique semantic "
                                "comparisons from order-swapped calls, and that convention has "
                                "to be declared rather than inferred from a job log"),
              "tuning_trials": {"DPO (uniform / sweep)": 0,
                                "_note": "dpo_beta fixed at the library default 0.01, no sweep"},
              "methods": {}}
    for method in sorted(cost):
        stages = {s: {k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in d.items()} for s, d in cost[method].items()}
        report["methods"][method] = {
            "stages": stages,
            "gpu_h_total": round(sum(d["gpu_h"] for d in cost[method].values()), 2),
            "cpu_h_total": round(sum(d["cpu_h"] for d in cost[method].values()), 2)}
    report["unattributed_jobs"] = unattributed
    out = ROOT / "analysis/feedback_cost.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    for method, d in sorted(report["methods"].items()):
        parts = " ".join("%s %.2fh" % (s, v["gpu_h"]) for s, v in sorted(d["stages"].items())
                         if v["gpu_h"] > 0)
        print("%-24s GPU %6.2fh  CPU %6.2fh  | %s" % (method, d["gpu_h_total"],
                                                      d["cpu_h_total"], parts))
    print(json.dumps({"written": str(out), "unattributed": len(unattributed)}))


if __name__ == "__main__":
    main()
