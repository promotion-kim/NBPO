"""Drop format-unrepresentable pairs from every arm, then materialize datasets.

The canonical target is log(w_a/c_a) - log(w_b/c_b), and prepare_nbpo_dataset
parses the rows at ten decimals, so a candidate whose optimal mass is below
1e-10 becomes 0.0 and the row is rejected for a non-positive mass. That is a
format limit: the solver is telling us "never produce this response", and the
ten-decimal field cannot carry it.

The affected pairs differ by arm, so removing each arm's own would put the arms
back on different pair sets. The UNION across arms is removed from all three
instead, and the identity of the surviving pair keys is checked before anything
is materialized.

The unordered response pair keys the row, because which side is "chosen"
depends on the arm.
"""
import json, subprocess, sys
from pathlib import Path

T = Path("/work/uf4_20260910/targets")
D = Path("/work/uf4_20260910/datasets")
S = Path("/work/sub_20260914")
DEPS = "/work/nbpo_repair_20260909/deps_train"
ARMS = ("pros4_pw_nbpo_c", "pros4_pw_fixedref_c", "pros4_prosper")
MIN_MASS = 1e-10


def key(row):
    a = int(row["chosen_candidate_index"])
    b = int(row["rejected_candidate_index"])
    return (row["prompt_id"], min(a, b), max(a, b))


report = {"min_representable_mass": MIN_MASS, "splits": {}}
for split in ("train", "dev"):
    rows_by_arm, bad = {}, set()
    for arm in ARMS:
        rows = [json.loads(l) for l in (T / arm / "pairs" / ("%s.jsonl" % split)).open()
                if l.strip()]
        rows_by_arm[arm] = rows
        for r in rows:
            if min(float(r["nbpo_weight_a"]), float(r["nbpo_weight_b"])) < MIN_MASS:
                bad.add(key(r))
    per_arm = {}
    keysets = {}
    for arm, rows in rows_by_arm.items():
        own = sum(1 for r in rows
                  if min(float(r["nbpo_weight_a"]), float(r["nbpo_weight_b"])) < MIN_MASS)
        kept = [r for r in rows if key(r) not in bad]
        dest = T / arm / "pairs" / ("%s_final.jsonl" % split)
        with dest.open("w") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        per_arm[arm] = {"rows_in": len(rows), "own_affected": own,
                        "kept": len(kept), "path": str(dest)}
        keysets[arm] = {key(r) for r in kept}
    sizes = {a: len(k) for a, k in keysets.items()}
    if len(set(sizes.values())) != 1:
        raise SystemExit("%s: arms disagree on surviving pair counts: %s" % (split, sizes))
    first = keysets[ARMS[0]]
    for arm in ARMS[1:]:
        if keysets[arm] != first:
            raise SystemExit("%s: %s has different surviving pair keys" % (split, arm))
    report["splits"][split] = {"union_removed": len(bad), "per_arm": per_arm,
                               "surviving_pairs_identical_across_arms": True,
                               "surviving_pairs": len(first)}

(S / "prosper/pair_filter.json").write_text(json.dumps(report, indent=1) + "\n")
print(json.dumps({k: {"union_removed": v["union_removed"],
                      "surviving_pairs": v["surviving_pairs"],
                      "own_affected": {a: d["own_affected"] for a, d in v["per_arm"].items()}}
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
        tail = [l for l in r.stderr.splitlines() if l.strip() and "examples/s" not in l][-6:]
        print("FAILED %s:\n  %s" % (arm, "\n  ".join(tail)))
        continue
    print("materialized", arm)
