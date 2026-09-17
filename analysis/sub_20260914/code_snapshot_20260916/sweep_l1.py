"""How does MaxEntBW certification depend on the multiplier norm?

The probe certified absolute_maxmin on all 697 prompts at L1 = 1.0, and the arm
fails at the matched L1 of 195.98 taken from the per-prompt Nash arm. A large
norm pushes mass toward the probability floor, the floor becomes active and
stationarity of the original problem breaks -- the same mechanism that fails the
global Nash arm. So the certification rate is measured across norms instead of
a value being picked by hand.

Nothing is relaxed: each point is a full solve with the declared tolerances.
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/work/sub_20260914/code")
sys.path.insert(0, "/work/uf4_20260910/code")
from mnpo_scripts.nbpo_core import uniform_policy
from mnpo_scripts.nbpo_generic import solve_finite_pool, validate_finite_pool_solution
from mnpo_scripts.nbpo_representations import AdaptiveGameRepresentation
import solve_pros4_targets as base

UF = Path("/work/uf4_20260910")
POOL, K = 8, 4
_S = {}


def _init(A, Aref, l1):
    _S.update(A=A, Aref=Aref, l1=l1)
    torch.set_num_threads(1)


def _one(x):
    A = np.ascontiguousarray(_S["A"][:, x:x + 1])
    Aref = np.ascontiguousarray(_S["Aref"][:, x:x + 1])
    try:
        rep = AdaptiveGameRepresentation(
            torch.from_numpy(A), torch.from_numpy(Aref), uniform_policy(1, POOL),
            torch.full((K,), 0.25, dtype=torch.float64),
            reference_construction="shared_pool")
        res = solve_finite_pool(rep, "absolute_maxmin", eta=1.0, inner_solver="exact",
                                dual_solver="root", dual_tol=1e-10, M=200,
                                inner_workers=1, probability_floor=1e-12,
                                weight_l1=_S["l1"], log_every=0)
        cert = validate_finite_pool_solution(res)
        ok = bool(cert.get("certified", False)) and \
            abs(float(res.target_log_ratio_check())) <= 1e-9
        pi = res.pi.numpy()
        return ok, float(pi.min()), None if ok else "uncertified"
    except Exception as exc:                      # noqa: BLE001
        return False, float("nan"), str(exc).split(";")[0].strip()[:90]


def main():
    scores, _ = base.load_scores(str(UF / "scores/pros_scores_all"), 8)
    train = {json.loads(l)["prompt_id"]
             for l in (UF / "splits/pros_v1f/policy_train.jsonl").open() if l.strip()}
    pids = sorted(p for p in scores if p in train)
    A = np.stack([scores[p][0] for p in pids], axis=1)
    Aref = np.stack([scores[p][1] for p in pids], axis=1)
    workers = min(48, len(os.sched_getaffinity(0)))
    rows = []
    for l1 in (1.0, 3.0, 10.0, 30.0, 100.0, 195.975626):
        t0 = time.monotonic()
        ok = 0
        mins = []
        reasons = {}
        with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                                 initargs=(A, Aref, l1)) as ex:
            for good, pmin, why in ex.map(_one, range(len(pids)), chunksize=8):
                ok += int(good)
                if np.isfinite(pmin):
                    mins.append(pmin)
                if why:
                    reasons[why] = reasons.get(why, 0) + 1
        row = {"weight_l1": l1, "prompts": len(pids), "certified": ok,
               "certified_fraction": ok / len(pids),
               "median_min_mass": float(np.median(mins)) if mins else None,
               "reasons": reasons, "seconds": round(time.monotonic() - t0, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = Path("/work/sub_20260914/prosper/maxmin_l1_sweep.json")
    out.write_text(json.dumps({"sweep": rows,
                               "note": ("full solves at the declared tolerances; the "
                                        "probability floor stays 1e-12")}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
