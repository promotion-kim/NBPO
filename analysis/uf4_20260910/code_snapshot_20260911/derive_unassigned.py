"""Recover the 3,240 unassigned eligible prompts, and prove the recovery is exact.

app:crossplay needs held-out prompt IDs never used to fit the preference model,
train a policy, select on development data, or produce a final-eval number.
splits/v1/report.json records exactly such a set -- unassigned_eligible = 3240 --
but the ids themselves were never written out.

Rather than reimplement the eligibility, deduplication and decontamination
pipeline and hope it agrees, this reruns the frozen split builder verbatim under
its own --out-name flag, adding one line that also writes the tail of `order`,
and then checks that all five split files come out with the sha256 recorded in
the original report. If they do, the recomputed order is the same order and the
prompts past the 35,000 assigned are the same prompts. If any hash differs it
emits nothing: a set that is only probably disjoint from training is worthless
here.
"""
import hashlib, json, pathlib, shutil, subprocess, sys

SRC = pathlib.Path("/work/uf4_20260910/code/build_uf4_splits.py")
DERIVED = pathlib.Path("/work/uf4_20260910/code/build_uf4_splits_reproduce.py")
OUT_NAME = "v1_reproduce"
OUT = pathlib.Path("/work/uf4_20260910/splits") / OUT_NAME
ORIGINAL = json.loads(pathlib.Path("/work/uf4_20260910/splits/v1/report.json").read_text())

src = SRC.read_text()
anchor = '''    report["unassigned_eligible"] = len(eligible) - sum(sizes.values())'''
if anchor not in src:
    sys.exit("anchor missing")
# `order` is built only up to sum(sizes), so order[35000:] is empty and
# unassigned_eligible in the report is arithmetic rather than a materialized
# set. The unassigned prompts are `eligible` minus whatever landed in `order`.
# Sorted by prompt_id, which is a sha256 of the normalized text, so taking a
# prefix of it is an unbiased sample rather than a walk through one source.
src = src.replace(anchor, anchor + '''
    _assigned_ids = {row["prompt_id"] for row in order}
    _unassigned = sorted((row for row in eligible
                          if row["prompt_id"] not in _assigned_ids),
                         key=lambda r: r["prompt_id"])
    with (out / "unassigned.jsonl").open("x") as stream:
        for row in _unassigned:
            stream.write(json.dumps(row, ensure_ascii=False) + "\\n")''', 1)
DERIVED.write_text(src)

if OUT.exists():
    shutil.rmtree(OUT)                       # the builder insists on creating it itself
env = {"PYTHONPATH": "/work/nbpo_repair_20260909/deps_train:/work/nbpo_repair_20260909/code:/work/uf4_20260910/code",
       "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
       "PATH": "/usr/bin:/bin:/usr/local/bin", "OMP_NUM_THREADS": "8"}
run = subprocess.run([sys.executable, str(DERIVED), "--out-name", OUT_NAME],
                     capture_output=True, text=True, env=env)
print(run.stdout.strip()[-500:])
if run.returncode:
    sys.exit("builder rerun failed:\n" + run.stderr[-1500:])


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


bad = []
for name, rec in ORIGINAL["splits"].items():
    got = file_hash(OUT / f"{name}.jsonl")
    print("  %-13s %s" % (name, "reproduced " + got[:16] if got == rec["sha256"] else "MISMATCH"))
    if got != rec["sha256"]:
        bad.append(name)
if bad:
    sys.exit("REPRODUCTION FAILED, emitting nothing: %s" % bad)

unassigned = [json.loads(l) for l in (OUT / "unassigned.jsonl").open() if l.strip()]
assigned = set()
for name in ORIGINAL["splits"]:
    with (pathlib.Path("/work/uf4_20260910/splits/v1") / f"{name}.jsonl").open() as f:
        assigned |= {json.loads(l)["prompt_id"] for l in f if l.strip()}
ids = {r["prompt_id"] for r in unassigned}
print(json.dumps({"unassigned": len(unassigned),
                  "recorded_unassigned_eligible": ORIGINAL["unassigned_eligible"],
                  "count_matches_report": len(unassigned) == ORIGINAL["unassigned_eligible"],
                  "distinct_ids": len(ids),
                  "overlap_with_any_assigned_split": len(ids & assigned),
                  "all_five_splits_reproduced": True,
                  "by_source": {s: sum(1 for r in unassigned if r["source"] == s)
                                for s in sorted({r["source"] for r in unassigned})}}, indent=2))
if ids & assigned:
    sys.exit("unassigned set overlaps an assigned split")
