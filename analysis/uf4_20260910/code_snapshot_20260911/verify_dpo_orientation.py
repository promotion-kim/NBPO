"""Independently re-derive the DPO label for sampled rows and compare."""
import json, hashlib, itertools
from pathlib import Path
import numpy as np

ROOT = Path("/work/uf4_20260910")
OBJ = ("instruction_following", "truthfulness", "honesty", "helpfulness")
cal = json.loads((ROOT / "dpo/v1/calibration_and_counts.json").read_text())
mean = np.asarray(cal["calibration"]["per_objective_mean"])
std = np.asarray(cal["calibration"]["per_objective_std"])
tie = cal["tie_threshold_standardized"]

def file_hash(p):
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""): h.update(c)
    return h.hexdigest()

# BT rewards for dev
scores = {}
for shard in range(4):
    d = ROOT / "scores/dev_v1" / f"shard{shard}"
    for path in sorted(d.glob("chunk*.npz")):
        a = np.load(path, allow_pickle=True)
        for i, pid in enumerate([str(x) for x in a["prompt_ids"]]):
            scores[pid] = a["r_bt"][:, i]

# index the source NBPO pairs by (pid, frozenset(candidate ids)) -> token hashes
src = {}
with (ROOT / "targets/nash_v1/pairs/dev.jsonl").open() as f:
    for line in f:
        r = json.loads(line)
        key = (str(r["prompt_id"]),
               frozenset((int(r["chosen_candidate_index"]), int(r["rejected_candidate_index"]))))
        src[key] = {int(r["chosen_candidate_index"]): r["chosen_token_sha256"],
                    int(r["rejected_candidate_index"]): r["rejected_token_sha256"]}

for name in ("uniform", "help_only", "truth_heavy"):
    w = np.asarray(cal["weights"][name])
    path = ROOT / "dpo/v1/pairs" / name / "dev.jsonl"
    checked = bad_order = bad_token = bad_tie = leftover = 0
    with path.open() as f:
        for n, line in enumerate(f):
            if n % 997:                      # sample ~1/1000 of the rows
                continue
            r = json.loads(line)
            pid = str(r["prompt_id"])
            i, j = int(r["chosen_candidate_index"]), int(r["rejected_candidate_index"])
            z = (scores[pid][:, 0, :] - mean[:, None]) / std[:, None]
            rw = w @ z
            checked += 1
            if not rw[i] - rw[j] > 0:
                bad_order += 1
            if abs(rw[i] - rw[j]) < tie:
                bad_tie += 1
            if abs(abs(rw[i] - rw[j]) - r["dpo_scalarized_gap"]) > 1e-9:
                bad_tie += 1
            toks = src[(pid, frozenset((i, j)))]
            if r["chosen_token_sha256"] != toks[i] or r["rejected_token_sha256"] != toks[j]:
                bad_token += 1
            if any(k.startswith(("nbpo_", "solver_", "opponent_", "lambda_")) for k in r):
                leftover += 1
    print("%-12s checked %4d | wrong order %d | token mismatch %d | tie/gap error %d | nbpo leftovers %d"
          % (name, checked, bad_order, bad_token, bad_tie, leftover))
