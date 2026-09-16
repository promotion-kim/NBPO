"""Restrict the split to prompts all three arms certify at the matched scale.

PROSPER at the matched multiplier norm (195.975626, the mean per-prompt weight
L1 of the certified NBPO-PW arm) certifies 328 of 384 train prompts. The two
Nash-PW arms certify all 384. Comparing arms on different prompt sets would
confound the aggregation rule with which prompts each arm could solve, so the
split becomes the intersection.

Keeping the matched norm rather than lowering it to 30, where MaxEntBW
certifies everywhere, is deliberate: the Nash dual is the natural scale of this
problem, and forcing a smaller norm would have all three arms train on targets
that are not their own solutions. The cost -- 384 to 328 prompts -- is recorded,
and "MaxEntBW does not certify 14.6% of prompts at the matched scale" is itself
a result rather than something to engineer away.

splits/pros_v1f is updated in place because its meaning is exactly "prompts
every arm can solve"; the previous contents are kept alongside.
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
import solve_pros4_targets as base

UF = Path("/work/uf4_20260910")
S = Path("/work/sub_20260914")
MATCHED_L1 = 195.975626
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
        return x, bool(cert.get("certified", False)) and \
            abs(float(res.target_log_ratio_check())) <= 1e-9
    except Exception:                             # noqa: BLE001
        return x, False


def certified_set(pids, A, Aref):
    ok = set()
    workers = min(48, len(os.sched_getaffinity(0)))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(A, Aref, MATCHED_L1)) as ex:
        for x, good in ex.map(_one, range(len(pids)), chunksize=8):
            if good:
                ok.add(pids[x])
    return ok


def main():
    scores, _ = base.load_scores(str(UF / "scores/pros_scores_all"), 8)
    split_dir = UF / "splits/pros_v1f"
    out = {}
    t0 = time.monotonic()
    for name in ("policy_train", "policy_dev"):
        rows = [json.loads(l) for l in (split_dir / ("%s.jsonl" % name)).open() if l.strip()]
        pids = [r["prompt_id"] for r in rows if r["prompt_id"] in scores]
        A = np.stack([scores[p][0] for p in pids], axis=1)
        Aref = np.stack([scores[p][1] for p in pids], axis=1)
        ok = certified_set(pids, A, Aref)
        kept = [r for r in rows if r["prompt_id"] in ok]
        backup = split_dir / ("%s.jsonl.before_matched_scale" % name)
        if not backup.exists():
            shutil.copy2(split_dir / ("%s.jsonl" % name), backup)
        with (split_dir / ("%s.jsonl" % name)).open("w") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out[name] = {"in": len(rows), "maxmin_certified": len(ok), "kept": len(kept),
                     "learner_pairs": len(kept) * 28}
    report = {"matched_weight_l1": MATCHED_L1, "splits": out,
              "rule": ("intersection: the two Nash-PW arms certify every prompt in the "
                       "previous split, so the binding constraint is MaxEntBW at the "
                       "matched norm"),
              "why_not_lower_the_norm": ("the Nash dual is this problem's natural scale; "
                                         "forcing a smaller norm would make all three arms "
                                         "train on targets that are not their own solutions"),
              "seconds": round(time.monotonic() - t0, 1)}
    (S / "prosper/common_certified.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
