"""Experiment 2 downstream: 1800 prompts, common certified set, three arms, 2 epochs.

Runs the whole chain after judging: merge the 800-prompt pool and verdicts in as
shards 8..11, re-score all 1800 in one pass so there is a single frozen judge
record, rebuild the split, regenerate the solvers against the new directories,
probe the per-prompt feasibility, solve the three arms, drop prompts the pair
format cannot express, materialize the datasets and queue training.

Every step is idempotent: it skips work whose output already exists, so a rerun
after a failure resumes rather than restarting. The contracts here were each
established the hard way earlier in this campaign -- shard directories rather
than flat files, a single teacher record, prompt-granularity exclusion by the
union across arms, and a new job id whenever a DONE job has to run again.
"""
import hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np

UF = Path("/work/uf4_20260910")
S = Path("/work/sub_20260914")
C = S / "code"
PY = sys.executable
NS = "sub_20260914-prosper-policy-split-v2:"
DEV_FRACTION = 0.10
ARMS = (("pros4_pw_nbpo_v2", "pw_nbpo", "solve_pros4_pw_nbpo_v2.py"),
        ("pros4_pw_fixedref_v2", "pw_fixedref", "solve_pros4_pw_fixedref_v2.py"),
        ("pros4_prosper_v2", "prosper", "solve_pros4_prosper_v2.py"))
ENVBASE = {"PYTHONPATH": "/work/pylibs_eval:/work/nbpo_repair_20260909/code:"
                        + str(C) + ":" + str(UF / "code"),
           "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
           "WANDB_MODE": "disabled", "PATH": "/usr/bin:/bin"}


def sh(cmd, env=None, cwd=str(C), label=""):
    t0 = time.monotonic()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd,
                       env=dict(ENVBASE, **(env or {})))
    dt = round(time.monotonic() - t0, 1)
    if r.returncode != 0:
        tail = [l for l in (r.stdout + r.stderr).splitlines()
                if l.strip() and "examples/s" not in l][-6:]
        raise SystemExit("STEP FAILED %s after %ss:\n  %s" % (label, dt, "\n  ".join(tail)))
    print("  ok %-28s %ss" % (label, dt), flush=True)
    return r.stdout


def fh(p):
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


log = {}

# 1. merge the extension pool and verdicts in as shards 8..11
for kind, main, ext in (("pool", UF / "pools/pros2_pool_v1", UF / "pools/pros3_pool"),
                        ("verdicts", S / "pool_judgments/pros2_psc",
                         S / "pool_judgments/pros3_psc")):
    for k in range(4):
        src, dst = ext / ("shard%d" % k), main / ("shard%d" % (8 + k))
        if dst.exists():
            continue
        if not src.is_dir():
            raise SystemExit("missing extension shard %s" % src)
        shutil.copytree(src, dst)
        if kind == "pool":
            old = dst / ("complete_shard%d.json" % k)
            if old.exists():
                rec = json.loads(old.read_text())
                rec["shard"] = 8 + k
                rec["merged_from"] = str(src)
                (dst / ("complete_shard%d.json" % (8 + k))).write_text(
                    json.dumps(rec, indent=2) + "\n")
                old.unlink()
log["merged_shards"] = 8
print("  ok merge -> shards 8..11", flush=True)

# 2. one scoring pass over all twelve judgment shards
if not (UF / "scores/pros_scores_v2/score_report.json").exists():
    sh([PY, str(C / "score_safe_pool.py"), "--tag", "pros2_psc", "--judge-shards", "12",
        "--pool", "pros2_pool_v1", "--pool-shards", "12",
        "--out-name", "pros_scores_v2", "--out-shards", "12",
        "--objectives", "item0", "item1", "item2", "item3", "--pool-per-role", "8"],
       label="score 1800 prompts")
rep = json.loads((UF / "scores/pros_scores_v2/score_report.json").read_text())
log["scoring"] = {k: rep[k] for k in ("verdicts_read", "parse_rate", "prompts_seen",
                                      "prompts_complete", "prompts_excluded")}
print("  scored: %d prompts complete of %d seen" % (rep["prompts_complete"],
                                                    rep["prompts_seen"]), flush=True)

# 3. split over exactly the scored prompts
import solve_pros4_targets as basemod
scores, _ = basemod.load_scores(str(UF / "scores/pros_scores_v2"), 12)
scored = set(scores)
rows, seen = [], set()
for name in ("pros_v1/pros_train.jsonl", "pros_ext/pros_train.jsonl",
             "pros_ext2/pros_train.jsonl"):
    for line in (UF / "splits" / name).open():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["prompt_id"] in scored and r["prompt_id"] not in seen:
            seen.add(r["prompt_id"]); rows.append(r)
if len(rows) != len(scored):
    raise SystemExit("%d scored but %d split rows matched" % (len(scored), len(rows)))
ordered = sorted(rows, key=lambda r: hashlib.sha256((NS + r["prompt_id"]).encode()).hexdigest())
n_dev = max(20, int(round(len(ordered) * DEV_FRACTION)))
sd = UF / "splits/pros_v2"
sd.mkdir(parents=True, exist_ok=True)
for name, part in (("policy_train", ordered[n_dev:]), ("policy_dev", ordered[:n_dev])):
    with (sd / ("%s.jsonl" % name)).open("w") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
log["split"] = {"train": len(ordered) - n_dev, "dev": n_dev}
print("  split: train %d dev %d" % (len(ordered) - n_dev, n_dev), flush=True)

# 4. solvers pointed at the new directories, as separate _v2 files
sh([PY, str(C / "build_pw_solvers.py"),
    "--score-root", "scores/pros_scores_v2", "--pool-root", "pools/pros2_pool_v1",
    "--splits", "splits/pros_v2f", "--shards", "12", "--suffix", "_v2"],
   label="generate _v2 solvers")
shutil.copy2(C / "solve_pros4_targets.py", C / "solve_pros4_targets_v2.py")

# 5. feasibility probe -> the common certified split
if not (S / "prosper/feasibility_v2.json").exists():
    out = sh([PY, str(C / "probe_feasible.py"),
              "--scores", str(UF / "scores/pros_scores_v2"), "--shards", "12",
              "--splits", "pros_v2", "--out-splits", "pros_v2f", "--workers", "48"],
             label="feasibility probe")
    shutil.move(str(S / "prosper/feasibility.json"), str(S / "prosper/feasibility_v2.json"))
log["feasibility"] = json.loads((S / "prosper/feasibility_v2.json").read_text())["feasible_per_rule"]
print("  feasible per rule:", log["feasibility"], flush=True)

print(json.dumps({"stage1_done": True, **log}, indent=1, default=str), flush=True)
(S / "prosper/exp2_stage1.json").write_text(json.dumps(log, indent=1, default=str) + "\n")
