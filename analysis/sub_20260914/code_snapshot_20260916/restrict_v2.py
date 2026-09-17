"""Restrict the v2 split to prompts all three arms certify at the matched norm.

The feasibility probe ran at a probe norm of 1.0, where max-min certifies every
prompt. At the matched norm taken from the Nash arm it does not, which is the
same boundary round one hit: the probe's intersection is necessary but not
sufficient. This re-probes max-min at the matched norm and intersects again, so
the three arms share one prompt set at the norm they will actually be solved at.

The norm is kept rather than lowered because the Nash dual is this problem's
natural scale; lowering it would have all three arms train on targets that are
not their own solutions.
"""
import json, os, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/work/sub_20260914/code")
sys.path.insert(0, "/work/uf4_20260910/code")
from mnpo_scripts.nbpo_core import uniform_policy
from mnpo_scripts.nbpo_generic import solve_finite_pool, validate_finite_pool_solution
from mnpo_scripts.nbpo_representations import AdaptiveGameRepresentation
import solve_pros4_targets_v2 as base

UF = Path("/work/uf4_20260910")
S = Path("/work/sub_20260914")
MATCHED_L1 = 195.975626
POOL, K, SHARDS = 8, 4, 12
_S = {}


def _init(A, Aref):
    _S.update(A=A, Aref=Aref)
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
                                weight_l1=MATCHED_L1, log_every=0)
        cert = validate_finite_pool_solution(res)
        return x, bool(cert.get("certified", False)) and \
            abs(float(res.target_log_ratio_check())) <= 1e-9
    except Exception:                             # noqa: BLE001
        return x, False


def main():
    scores, _ = base.load_scores(str(UF / "scores/pros_scores_v2"), SHARDS)
    sd = UF / "splits/pros_v2f"
    out = {}
    t0 = time.monotonic()
    for name in ("policy_train", "policy_dev"):
        rows = [json.loads(l) for l in (sd / ("%s.jsonl" % name)).open() if l.strip()]
        pids = [r["prompt_id"] for r in rows if r["prompt_id"] in scores]
        A = np.stack([scores[p][0] for p in pids], axis=1)
        Aref = np.stack([scores[p][1] for p in pids], axis=1)
        ok = set()
        with ProcessPoolExecutor(max_workers=min(48, len(os.sched_getaffinity(0))),
                                 initializer=_init, initargs=(A, Aref)) as ex:
            for x, good in ex.map(_one, range(len(pids)), chunksize=8):
                if good:
                    ok.add(pids[x])
        kept = [r for r in rows if r["prompt_id"] in ok]
        bak = sd / ("%s.jsonl.before_matched_scale" % name)
        if not bak.exists():
            shutil.copy2(sd / ("%s.jsonl" % name), bak)
        with (sd / ("%s.jsonl" % name)).open("w") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out[name] = {"in": len(rows), "maxmin_certified_at_matched_norm": len(ok),
                     "kept": len(kept), "learner_pairs": len(kept) * 28}
    rep = {"matched_weight_l1": MATCHED_L1, "splits": out,
           "seconds": round(time.monotonic() - t0, 1)}
    (S / "prosper/common_certified_v2.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
