"""Pre-flight every queued 4-GPU training: would it survive step 0?

MOPO failed twice at step zero on pinned-hash mismatches, each costing a
four-GPU slot, so the same hashes are compared on disk before a job gets a card.

The first version of this reported 10 of 14 trainings as broken when none was,
which is worse than no check at all. Two mistakes, both now avoided:

* It checked for a precompute manifest on every arm. run_mnpo only runs the
  provenance block when nbpo_target_mode is canonical_logratio or mopo_rho;
  scalarized DPO sets neither -- it re-establishes its own checks in its
  materializer -- so a missing manifest there is expected. The running DPO arm
  was proof the alarm was false.
* It read the solver hash from pairs/dev.jsonl. Each split carries its OWN
  split's solver solution, and the trainer compares the pin against the TRAIN
  split only. Sampling dev flagged all three PROSPER arms, whose train rows
  match exactly.
"""
import glob
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
PINNED_MODES = ("canonical_logratio", "mopo_rho")


def file_hash(p):
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


state = json.load(open(ROOT / "jobs/state.json"))["jobs"]
rows = []
for p in sorted(glob.glob(str(ROOT / "jobs/queue/*.json"))):
    spec = json.loads(Path(p).read_text())
    if spec.get("gpus") != 4:
        continue
    job = spec["job_id"]
    st = state.get(job, {}).get("state", "-")
    if st == "DONE":
        continue
    cfg_path = next((a for a in spec["command"] if a.endswith(".yaml")), None)
    if not cfg_path or not os.path.exists(cfg_path):
        rows.append((job, st, "NO CONFIG")); continue
    cfg = Path(cfg_path).read_text()

    def get(pat):
        m = re.search(pat, cfg)
        return m.group(1) if m else None

    m = re.search(r"dataset_mixer:\s*\n\s*(\S+): 1\.0", cfg)
    ds = m.group(1) if m else None
    mode = get(r"nbpo_target_mode: (\S+)") or "sampled"
    pinned_ds = get(r"nbpo_expected_dataset_manifest_sha256: (\S+)")
    pinned_sv = get(r"nbpo_expected_solver_artifact_sha256: (\S+)")

    problems = []
    if not ds or not os.path.isdir(ds):
        problems.append("dataset dir missing: %s" % ds)
    elif mode in PINNED_MODES:
        man = os.path.join(ds, "precompute_manifest.json")
        if not os.path.exists(man):
            problems.append("no precompute_manifest.json")
        elif pinned_ds and file_hash(man) != pinned_ds:
            problems.append("manifest hash != pinned")
        tname = Path(ds).name
        # The trainer compares the pin against the TRAIN split only.
        pairs = ROOT / "targets" / tname / "pairs/train.jsonl"
        if pinned_sv and pairs.exists():
            with open(pairs) as f:
                rsv = json.loads(f.readline()).get("solver_artifact_sha256")
            if rsv and rsv != pinned_sv:
                problems.append("train rows' solver hash != pinned")
        sol = ROOT / "targets" / tname / "train/solver/solution.json"
        if pinned_sv and sol.exists() and file_hash(sol) != pinned_sv:
            problems.append("train solution on disk != pinned")
    rows.append((job, st, "OK (%s)" % mode if not problems else "; ".join(problems)))

print("%-34s %-8s %s" % ("job", "state", "verdict"))
bad = sum(1 for _, _, v in rows if not v.startswith("OK"))
for job, st, verdict in sorted(rows):
    print("%-34s %-8s %s%s" % (job, st, verdict, "" if verdict.startswith("OK") else "  <--"))
print("\n%d of %d queued trainings fail the pre-flight" % (bad, len(rows)))
