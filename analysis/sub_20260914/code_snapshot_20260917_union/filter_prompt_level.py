"""Exclude whole prompts, not pairs: the format requires all 28 pairs per prompt.

Dropping individual pairs failed on "Every prompt requires all 28 unordered
pairs of 8 candidates". The canonical all-pair dataset is defined per prompt, so
the unit of exclusion has to be the prompt: if any of a prompt's 28 pairs has an
optimal mass below the ten-decimal representable threshold in ANY arm, that
prompt leaves every arm.

Using the union across arms keeps the three arms on one prompt set, which is
what makes the comparison a comparison.
"""
import json, subprocess, sys
from pathlib import Path

T = Path("/work/uf4_20260910/targets")
D = Path("/work/uf4_20260910/datasets")
S = Path("/work/sub_20260914")
DEPS = "/work/nbpo_repair_20260909/deps_train"
ARMS = ("pros4_pw_nbpo_c", "pros4_pw_fixedref_c", "pros4_prosper")
MIN_MASS = 1e-10

report = {"min_representable_mass": MIN_MASS, "unit_of_exclusion": "prompt", "splits": {}}
for split in ("train", "dev"):
    rows_by_arm, bad_prompts = {}, set()
    per_arm_bad = {}
    for arm in ARMS:
        rows = [json.loads(l) for l in (T / arm / "pairs" / ("%s.jsonl" % split)).open()
                if l.strip()]
        rows_by_arm[arm] = rows
        own = {r["prompt_id"] for r in rows
               if min(float(r["nbpo_weight_a"]), float(r["nbpo_weight_b"])) < MIN_MASS}
        per_arm_bad[arm] = len(own)
        bad_prompts |= own
    per_arm = {}
    prompt_sets, pair_counts = {}, {}
    for arm, rows in rows_by_arm.items():
        kept = [r for r in rows if r["prompt_id"] not in bad_prompts]
        dest = T / arm / "pairs" / ("%s_final.jsonl" % split)
        with dest.open("w") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        pids = {r["prompt_id"] for r in kept}
        # every surviving prompt must still carry all 28 pairs
        counts = {}
        for r in kept:
            counts[r["prompt_id"]] = counts.get(r["prompt_id"], 0) + 1
        short = {p: c for p, c in counts.items() if c != 28}
        if short:
            raise SystemExit("%s/%s: %d prompts do not carry 28 pairs, e.g. %s"
                             % (split, arm, len(short), list(short.items())[:3]))
        per_arm[arm] = {"rows_in": len(rows), "prompts_own_affected": per_arm_bad[arm],
                        "kept_rows": len(kept), "kept_prompts": len(pids)}
        prompt_sets[arm] = pids
        pair_counts[arm] = len(kept)
    if len({frozenset(v) for v in prompt_sets.values()}) != 1:
        raise SystemExit("%s: arms disagree on the surviving prompt set" % split)
    report["splits"][split] = {
        "prompts_excluded_union": len(bad_prompts),
        "prompts_excluded_per_arm": per_arm_bad,
        "surviving_prompts": len(next(iter(prompt_sets.values()))),
        "surviving_pairs": next(iter(pair_counts.values())),
        "identical_across_arms": True, "per_arm": per_arm}

(S / "prosper/pair_filter.json").write_text(json.dumps(report, indent=1) + "\n")
print(json.dumps({k: {kk: v[kk] for kk in ("prompts_excluded_union",
                                            "prompts_excluded_per_arm",
                                            "surviving_prompts", "surviving_pairs")}
                  for k, v in report["splits"].items()}, indent=1))

for arm in ARMS:
    out = D / arm
    if (out / "dataset_dict.json").exists():
        print("dataset exists, skipped:", arm)
        continue
    cmd = [sys.executable, "-m", "mnpo_scripts.prepare_nbpo_dataset",
           "--train", str(T / arm / "pairs/train_final.jsonl"),
           "--dev", str(T / arm / "pairs/dev_final.jsonl"),
           "--output", str(out),
           "--provenance", str(T / arm / "dataset_provenance.json")]
    env = {"PYTHONPATH": DEPS + ":/work/nbpo_repair_20260909/code",
           "HF_DATASETS_CACHE": str(T / arm / "arrow_cache"),
           "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "PYTHONDONTWRITEBYTECODE": "1", "WANDB_MODE": "disabled", "PATH": "/usr/bin:/bin"}
    r = subprocess.run(cmd, cwd="/work/nbpo_repair_20260909/code", env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        tail = [l for l in r.stderr.splitlines() if l.strip() and "examples/s" not in l][-5:]
        print("FAILED %s:\n  %s" % (arm, "\n  ".join(tail)))
        continue
    print("materialized", arm, "|", (r.stdout or "").strip()[-200:])
