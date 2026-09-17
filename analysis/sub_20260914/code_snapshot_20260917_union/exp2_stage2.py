"""Experiment 2 stage 2: solve three arms, filter, materialize, queue training.

699 common certified train prompts and 78 dev, against 291 and 31 in the first
round: 2.4 times the panel. PROSPER is again the binding rule at the matched
norm, and the ten-draw ablation showed that spending the budget on judgment
precision instead would have reduced the certified set, so prompts were the
right lever.

Steps are the ones established earlier: solve with --skip-dataset, drop prompts
whose optimal mass the ten-decimal pair format cannot carry using the union
across arms, materialize, then queue training at two epochs over the surviving
pairs. Training is queued above the remaining benchmark jobs because experiment
2 is the one that has to land by the deadline.
"""
import hashlib, json, subprocess, sys, time
from pathlib import Path

UF = Path("/work/uf4_20260910")
S = Path("/work/sub_20260914")
C = S / "code"
D = UF / "datasets"
T = UF / "targets"
PY = sys.executable
DEPS = "/work/nbpo_repair_20260909/deps_train"
ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/nbpo_repair_20260909/code:"
                    + str(C) + ":" + str(UF / "code"),
       "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
       "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
       "WANDB_MODE": "disabled", "PATH": "/usr/bin:/bin"}
# the trainer puts deps_train first, so the dataset has to be WRITTEN by that
# same datasets version: 5.0.1 emits the Json feature type and 3.6.0 rejects it
ENV_DS = dict(ENV, PYTHONPATH=DEPS + ":" + ENV["PYTHONPATH"])
ARMS = (("pros4_pw_nbpo_v2", "solve_pros4_pw_nbpo_v2.py"),
        ("pros4_pw_fixedref_v2", "solve_pros4_pw_fixedref_v2.py"),
        ("pros4_prosper_v2", "solve_pros4_prosper_v2.py"))
MATCHED_L1 = 195.975626      # same matched norm as round one, for comparability
MIN_MASS = 1e-10


def sh(cmd, label, cwd=str(C), env=None):
    t0 = time.monotonic()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd,
                       env=env or ENV)
    if r.returncode != 0:
        tail = [l for l in (r.stdout + r.stderr).splitlines()
                if l.strip() and "examples/s" not in l][-6:]
        raise SystemExit("FAILED %s:\n  %s" % (label, "\n  ".join(tail)))
    print("  ok %-30s %.0fs" % (label, time.monotonic() - t0), flush=True)
    return r.stdout


def fh(p):
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


log = {}
# 1. solve, skipping dataset materialization
for out_name, script in ARMS:
    if (T / out_name / "complete.json").exists():
        print("  already solved:", out_name, flush=True); continue
    if (T / out_name).exists():
        import shutil; shutil.rmtree(T / out_name)
    cmd = [PY, str(C / script), "--out-name", out_name, "--workers", "24",
           "--shards", "12", "--probability-floor", "1e-12", "--skip-dataset"]
    if "prosper" in script:
        cmd += ["--weight-l1", repr(MATCHED_L1)]
    sh(cmd, "solve " + out_name)

# 2. prompt-level representability filter, union across arms
names = [a for a, _ in ARMS]
report = {"min_representable_mass": MIN_MASS, "unit": "prompt", "splits": {}}
for split in ("train", "dev"):
    rows_by_arm, bad, own = {}, set(), {}
    for arm in names:
        rows = [json.loads(l) for l in (T / arm / "pairs" / ("%s.jsonl" % split)).open()
                if l.strip()]
        rows_by_arm[arm] = rows
        o = {r["prompt_id"] for r in rows
             if min(float(r["nbpo_weight_a"]), float(r["nbpo_weight_b"])) < MIN_MASS}
        own[arm] = len(o); bad |= o
    keep_sets, pair_counts = {}, {}
    for arm, rows in rows_by_arm.items():
        kept = [r for r in rows if r["prompt_id"] not in bad]
        with (T / arm / "pairs" / ("%s_final.jsonl" % split)).open("w") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts = {}
        for r in kept:
            counts[r["prompt_id"]] = counts.get(r["prompt_id"], 0) + 1
        short = {p: c for p, c in counts.items() if c != 28}
        if short:
            raise SystemExit("%s/%s: %d prompts lack 28 pairs" % (split, arm, len(short)))
        keep_sets[arm] = set(counts); pair_counts[arm] = len(kept)
    if len({frozenset(v) for v in keep_sets.values()}) != 1:
        raise SystemExit("%s: arms disagree on the surviving prompt set" % split)
    report["splits"][split] = {"excluded_union": len(bad), "excluded_per_arm": own,
                               "prompts": len(next(iter(keep_sets.values()))),
                               "pairs": next(iter(pair_counts.values()))}
(S / "prosper/pair_filter_v2.json").write_text(json.dumps(report, indent=1) + "\n")
log["filter"] = report["splits"]
print("  filter:", json.dumps(report["splits"]), flush=True)

# 3. materialize, and record the dataset in complete.json
for arm in names:
    out = D / arm
    if not (out / "dataset_dict.json").exists():
        sh([PY, "-m", "mnpo_scripts.prepare_nbpo_dataset",
            "--train", str(T / arm / "pairs/train_final.jsonl"),
            "--dev", str(T / arm / "pairs/dev_final.jsonl"),
            "--output", str(out),
            "--provenance", str(T / arm / "dataset_provenance.json")],
           "materialize " + arm, cwd="/work/nbpo_repair_20260909/code",
           env=ENV_DS)
    rec = json.loads((T / arm / "complete.json").read_text())
    if not rec.get("dataset_path"):
        rec.setdefault("_superseded", []).append(
            {"dataset_path": rec.get("dataset_path"),
             "dataset_manifest_sha256": rec.get("dataset_manifest_sha256"),
             "why": "solved with --skip-dataset; materialized behind the filter"})
        rec["dataset_path"] = str(out)
        rec["dataset_manifest_sha256"] = fh(out / "precompute_manifest.json")
        (T / arm / "complete.json").write_text(json.dumps(rec, indent=2) + "\n")

# 4. two epochs over the surviving pairs, queued above the benchmark jobs
pairs = report["splits"]["train"]["pairs"]
steps = max(1, round(2 * pairs / 128))
log["train"] = {"pairs": pairs, "batch": 128, "epochs": 2, "max_steps": steps}
print("  training: %d pairs -> %d steps for 2 epochs" % (pairs, steps), flush=True)
for arm, _ in ARMS:
    short = arm.replace("pros4_", "pros2_").replace("_v2", "")
    sh([PY, str(C / "make_pros_train_jobs.py"),
        "--targets", arm, "--arm", short + "_e2",
        "--seed", "555134", "--max-steps", str(steps), "--eval-steps", str(steps),
        "--priority", "20", "--job-prefix", "pros4",
        "--model", "/work/models/bases/Qwen2.5-7B-Instruct",
        "--model-revision", "a09a35458c702b33eeacc393d103063234e8bc28",
        "--learning-rate", "3.0e-07", "--weight-decay", "1.0e-06",
        "--per-device-batch", "4", "--grad-accum", "8"],
       "queue train " + short)
(S / "prosper/exp2_stage2.json").write_text(json.dumps(log, indent=1) + "\n")
print(json.dumps(log, indent=1), flush=True)
