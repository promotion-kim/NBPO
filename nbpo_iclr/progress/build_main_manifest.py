#!/usr/bin/env python3
"""Derive the main-body result manifest from the cluster, never from memory.

One row per main-body deliverable: (exhibit_label, method, seed/weight). Each row
carries the checkpoint it needs, the config hash that produced it, the job that
does the work, that job's real state, its progress, the artifact that would prove
it finished, and an ETA built from measured durations of the same job type.

Rules this file follows:
  * DONE means the proving artifact is on disk, not that a job exited zero;
  * a job the queue has never seen is NOT_STARTED, never DONE;
  * a failure superseded by a successful retry is NOT_APPLICABLE with the reason,
    so five historical failures do not read as five live blockers;
  * the denominator is the frozen execution matrix, so a row is never dropped to
    make a percentage look better.
"""
from __future__ import annotations

import json, os, re, subprocess, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

KST = timezone(timedelta(hours=9))
PROG = Path("/home/sjkim/MNPO/nbpo_iclr/progress")
ROOT = "/work/uf4_20260910"
KUBECONFIG = "/home/sjkim/.kube/aipr-kubeconfig.yaml"
POD = ["kubectl", "exec", "-n", "p-aipr", "nbpo-judge", "-c", "main", "--", "bash", "-lc"]

# Measured medians from logs/queue_report.jsonl, in minutes. Used for ETA only.
DURATION = {"solve": 9, "make_train": 1, "train": 184, "gen": 9, "judge": 24,
            "cap": 17, "bench": 3, "dpo_build": 20, "crossplay": 60}


def pod(cmd, timeout=180):
    env = dict(os.environ, KUBECONFIG=KUBECONFIG)
    try:
        r = subprocess.run(POD + [cmd], capture_output=True, text=True, timeout=timeout, env=env)
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def cluster():
    raw = pod(f"cat {ROOT}/jobs/state.json")
    jobs = {}
    try:
        jobs = json.loads(raw)["jobs"] if raw.strip() else {}
    except Exception:
        jobs = {}
    # One find per tree: a glob that matches nothing must not make the whole
    # listing exit non-zero, because pod() then returns "" and every finished
    # job would be scored as missing its artifact.
    listing = pod(
        f"find {ROOT}/evaluation {ROOT}/arms {ROOT}/targets {ROOT}/datasets "
        f"-maxdepth 3 \\( -name complete.json -o -name results.json "
        f"-o -name config.json -o -name dataset_dict.json \\) 2>/dev/null; true")
    present = {p.strip() for p in listing.split() if p.strip()}
    return jobs, present


def progress_of(job_id, entry):
    """Real progress for a running job: optimizer steps for training, else elapsed."""
    if entry.get("state") != "RUNNING":
        return ""
    log = entry.get("log")
    if log and "train" in job_id:
        # Match the arm's own max_steps: the periodic dev pass prints a second
        # tqdm bar, and reporting that one as training progress is wrong.
        arm = job_id[len("uf4_train_"):] if job_id.startswith("uf4_train_") else job_id
        cfg = pod(f"grep -m1 '^max_steps:' {ROOT}/configs/{arm}.yaml 2>/dev/null; true")
        hit = re.search(r"max_steps:\s*(\d+)", cfg)
        if hit:
            total = hit.group(1)
            out = pod(f"tr '\\r' '\\n' < {log} | grep -oE '[0-9]+/{total} \\[' | tail -1")
            step = out.strip().rstrip(" [")
            if step:
                return f"step {step}"
    started = entry.get("started")
    if started:
        t0 = datetime.strptime(started, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return "elapsed %dm" % ((datetime.now(timezone.utc) - t0).total_seconds() // 60)
    return "running"


# ---------------------------------------------------------------- row plan
ARMS = {
    "nbpo": ("nbpo_mse_s%d", [42, 43, 44]),
    "game_utilitarian": ("util_mse_s%d", [42, 43, 44]),
    # amended 2026-09-11 20:22: this control became the pivotal comparison, so
    # the matrix was widened from one seed to three (see execution_matrix.json).
    "fixed_reference_nash": ("fixedref_mse_s%d", [42, 43, 44]),
    "bt_rm_nash": ("btrm_mse_s%d", [42, 43, 44]),
    # amended 2026-09-12 03:45: this control became the only arm leading the
    # objectives table, on one seed, right after the fixed-reference control's
    # one-seed lead dissolved at three seeds. Matrix widened 1 -> 3.
    "game_maxmin": ("maxmin_mse_s%d", [42, 43, 44]),
    "dpo_uniform": ("dpo_uniform_mse_s%d", [42]),
    "prosper_adapt": ("prosper_mse_s%d", [42, 43, 44]),
    "mopo_adapt": ("mopo_mse_s%d", [42, 43, 44]),
}
DPO_WEIGHTS = ["if_only", "truth_only", "honesty_only", "help_only",
               "uniform", "help_heavy", "truth_heavy"]
BENCHMARKS = ["ifeval", "gsm8k", "harmbench", "xstest"]


def rows(jobs, present):
    out = []

    def add(exhibit, method, key, checkpoint, job_id, artifact, dep, kind):
        entry = jobs.get(job_id, {})
        state = entry.get("state")
        if artifact and artifact in present:
            state = "DONE"
        elif state is None:
            state = "NOT_STARTED"
        elif state == "DONE" and artifact and artifact not in present:
            state = "FAILED"          # exited zero without the proving artifact
        out.append({"exhibit_label": exhibit, "method": method, "seed_or_weight": key,
                    "checkpoint_id": checkpoint, "config_hash": entry.get("spec_sha256", ""),
                    "dependency": dep, "job_id": job_id, "state": state,
                    "progress": progress_of(job_id, entry),
                    "artifact_path": artifact, "eta_min": DURATION.get(kind, 0)})

    # Table 1 / Table 5: one independent final evaluation per policy arm
    for method, (pattern, seeds) in ARMS.items():
        for seed in seeds:
            arm = pattern % seed
            add("tab:uf-objectives", method, "seed %d" % seed, arm,
                f"uf4_finaleval_judge_{arm}",
                f"{ROOT}/evaluation/final_eval/{arm}/complete.json",
                f"uf4_train_{arm}", "judge")
            add("tab:uf-objectives", method, "seed %d (train)" % seed, arm,
                f"uf4_train_{arm}", f"{ROOT}/arms/{arm}/config.json", "target solve", "train")
    add("tab:uf-objectives", "base", "reference", "Llama-3.1-8B-Instruct",
        "uf4_pool_finaleval_shard0", f"{ROOT}/arms/_base_is_reference", "-", "gen")
    out[-1]["state"] = "DONE"
    out[-1]["artifact_path"] = "cached base responses (defined reference, win rate 0.5)"

    # Figure 2: the seven declared scalarized-DPO operating points
    for weight in DPO_WEIGHTS:
        arm = f"dpo_{weight}_mse_s42"
        add("fig:uf-tradeoffs", "scalarized_dpo", weight, arm,
            f"uf4_finaleval_judge_{arm}",
            f"{ROOT}/evaluation/final_eval/{arm}/complete.json",
            f"uf4_train_{arm}", "judge")
        add("fig:uf-tradeoffs", "scalarized_dpo", weight + " (train)", arm,
            f"uf4_train_{arm}", f"{ROOT}/arms/{arm}/config.json",
            "uf4_build_dpo_datasets", "train")

    # Table 4: independent cross-play on a frozen comparator bank
    for policy in ["base", "nbpo_mse_s42", "fixedref_mse_s42", "dev_selected_competitor"]:
        add("tab:independent-crossplay", policy, "-", policy,
            f"uf4_crossplay_{policy}", f"{ROOT}/evaluation/crossplay/{policy}/complete.json",
            "uf4_crossplay_bank", "crossplay")

    # Table 5: same checkpoints, fixed capability and safety benchmarks
    for method, (pattern, seeds) in ARMS.items():
        for seed in seeds:
            arm = pattern % seed
            for bench in BENCHMARKS:
                add("tab:general-capability", method, "seed %d / %s" % (seed, bench), arm,
                    f"uf4_bench_{arm}_{bench}",
                    f"{ROOT}/evaluation/capability/{arm}/{bench}/results.json",
                    f"uf4_gen_bench_{arm}", "cap")
    return out


def exists_on_pod(paths):
    """Test these exact paths. The artifact index is narrow by design, so a
    failure record must be checked against the filesystem, not against it."""
    if not paths:
        return set()
    script = "; ".join("[ -e %s ] && echo %s" % (p, p) for p in paths) + "; true"
    return {q.strip() for q in pod(script).split() if q.strip()}


def superseded(jobs, present):
    """Historical failures whose work a later successful retry already delivered."""
    notes = []
    wanted = set()
    for entry in jobs.values():
        if entry.get("state") == "FAILED":
            wanted.update(entry.get("missing_artifacts") or [])
    recovered_all = exists_on_pod(sorted(wanted))
    for job_id, entry in jobs.items():
        if entry.get("state") != "FAILED":
            continue
        missing = entry.get("missing_artifacts") or []
        recovered = [m for m in missing if m in recovered_all]
        if entry.get("failure") and "operator" in str(entry.get("failure")):
            notes.append({"job_id": job_id, "state": "NOT_APPLICABLE",
                          "reason": "stopped by operator; superseded by a later retry"})
        elif missing and len(recovered) == len(missing):
            notes.append({"job_id": job_id, "state": "NOT_APPLICABLE",
                          "reason": "every declared artifact is on disk from a later successful retry"})
        else:
            notes.append({"job_id": job_id, "state": "FAILED",
                          "reason": "no successful retry found; artifact still missing",
                          "missing": missing})
    return notes


def main():
    jobs, present = cluster()
    if not jobs:
        print("cluster unreachable; manifest not rewritten", file=sys.stderr)
        return 1
    table = rows(jobs, present)
    notes = superseded(jobs, present)
    counts = {}
    for r in table:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    payload = {"generated_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
               "execution_matrix": json.loads((PROG / "execution_matrix.json").read_text()),
               "state_counts": counts, "rows": table, "superseded_failures": notes}
    tmp = PROG / "main_results_manifest.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, PROG / "main_results_manifest.json")

    lines = ["# Main-body result manifest",
             "", "Generated %s KST. DONE requires the proving artifact on disk." % payload["generated_kst"],
             "", "State counts: " + ", ".join("%s %d" % kv for kv in sorted(counts.items())), "",
             "| exhibit | method | seed/weight | checkpoint | job | state | progress | ETA (min) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in table:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["exhibit_label"], r["method"], r["seed_or_weight"], r["checkpoint_id"],
            r["job_id"], r["state"], r["progress"] or "-", r["eta_min"]))
    if notes:
        lines += ["", "## Historical failures", "",
                  "| job | state | reason |", "|---|---|---|"]
        for n in notes:
            lines.append("| %s | %s | %s |" % (n["job_id"], n["state"], n["reason"]))
    tmp = PROG / "main_results_manifest.md.tmp"
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, PROG / "main_results_manifest.md")
    print(json.dumps({"rows": len(table), "counts": counts,
                      "superseded": len([n for n in notes if n["state"] == "NOT_APPLICABLE"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
