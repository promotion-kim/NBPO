# Generated from probe_feasible_uw1.py by build_panel_solvers.py -- do not
# edit by hand. source sha256 53beb08ef84e4cd9e4a9b6f27b4beba7e5cff72b4e3cad01fab85f9785088f0d
# change: base module -> solve_pros4_targets_uw1cb4. Scores, splits and output
#         split are already flags.
# Copy of probe_feasible.py for the UW panel: independent_samples reference
# construction and the UW base loader (no scalar BT projection).
"""Which prompts can all four arms solve? Restrict the split to those.

A per-prompt Nash bargaining problem needs an interior point: some mixture of
the eight learners must beat the comparator bank on EVERY one of the prompt's
four checklist items at once. Strongly conflicting prompts have none and the
solver refuses, correctly. MaxEntBW needs no interior point, so the per-prompt
arms do not agree on which prompts are solvable.

Rather than teach the validated solvers to swallow a refusal, this probes each
per-prompt rule once, intersects the feasible sets, and writes restricted split
files. All four arms are then compared on one prompt set, and the refusal counts
are reported instead of being hidden. No tolerance is relaxed.

The global NBPO arm is not probed: it solves one problem over all prompts, so it
either certifies as a whole or does not. It runs on the same restricted split so
the four arms share their prompts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/work/sub_20260914/code")
sys.path.insert(0, "/work/uf4_20260910/code")
from mnpo_scripts.nbpo_core import uniform_policy
from mnpo_scripts.nbpo_generic import solve_finite_pool, validate_finite_pool_solution
from mnpo_scripts.nbpo_representations import (AdaptiveGameRepresentation,
                                               FixedReferenceRepresentation)
import solve_pros4_targets_uw1cb4 as base                # the solvers' own score loader

UF = Path("/work/uf4_20260910")
POOL = 8
K = len(base.OBJECTIVES)
RULES = ("pw_nash_adaptive", "pw_nash_fixedref", "pw_absolute_maxmin")
_S = {}


def _init(A, Aref, beta, eta, floor, dual_calls, l1):
    _S.update(A=A, Aref=Aref, beta=beta, eta=eta, floor=floor, M=dual_calls, l1=l1)
    torch.set_num_threads(1)


def _one(x):
    A = np.ascontiguousarray(_S["A"][:, x:x + 1])
    Aref = np.ascontiguousarray(_S["Aref"][:, x:x + 1])
    mu = uniform_policy(1, POOL)
    kw = dict(eta=_S["eta"], inner_solver="exact", dual_solver="root",
              dual_tol=1e-10, M=_S["M"], inner_workers=1,
              probability_floor=_S["floor"], log_every=0)
    out = {}
    for rule in RULES:
        try:
            if rule == "pw_nash_fixedref":
                rep = FixedReferenceRepresentation(
                    torch.from_numpy(A), torch.from_numpy(Aref), mu,
                    reference_construction="independent_samples")
            else:
                rep = AdaptiveGameRepresentation(
                    torch.from_numpy(A), torch.from_numpy(Aref), mu,
                    torch.full((K,), _S["beta"], dtype=torch.float64),
                    reference_construction="independent_samples")
            if rule == "pw_absolute_maxmin":
                res = solve_finite_pool(rep, "absolute_maxmin",
                                        weight_l1=_S["l1"], **kw)
            else:
                res = solve_finite_pool(rep, "nash", **kw)
            cert = validate_finite_pool_solution(res)
            ok = bool(cert.get("certified", False)) and \
                abs(float(res.target_log_ratio_check())) <= 1e-9
            out[rule] = (ok, None if ok else "uncertified_or_identity_residual")
        except Exception as exc:                  # noqa: BLE001 - counted, not hidden
            out[rule] = (False, str(exc).split(";")[0].strip()[:200])
    return x, out


def load_scores(root, shards):
    """Read via the solvers' own loader so the probe sees exactly what they will.

    base.load_scores returns prompt_id -> (A_policy[K,I,I], A_ref[K,I,I],
    r_bt[K,2,I]) with the chunk hashes verified and duplicate prompt ids
    rejected. Stacking on a new axis 1 gives the [K, X, I, I] the
    representations take.
    """
    scores, _ = base.load_scores(str(root), shards)
    pids = sorted(scores)
    if not pids:
        raise SystemExit("no scored prompts under %s" % root)
    A = np.stack([scores[p][0] for p in pids], axis=1)
    Aref = np.stack([scores[p][1] for p in pids], axis=1)
    return pids, np.ascontiguousarray(A), np.ascontiguousarray(Aref)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default=str(UF / "scores/pros_scores_all"),
                    help="unified score directory: one GPM teacher record over "
                         "all judgment shards. pros_scores_v1 is the earlier "
                         "415-prompt run and must not be the default, since a "
                         "missing flag would then silently probe a subset.")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--splits", default="pros_v1")
    ap.add_argument("--out-splits", default="pros_v1f",
                    help="NEW splits directory to write policy_train.jsonl and "
                         "policy_dev.jsonl into. The solvers read those exact file "
                         "names, so the feasible subset must be its own directory; "
                         "the original split stays as the record of what was judged.")
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--probe-weight-l1", type=float, default=1.0,
                    help="maxmin needs a multiplier norm; feasibility of the "
                         "certificate does not depend on its value, and the real "
                         "solve uses the matched norm from the nbpo arm")
    ap.add_argument("--max-dual-calls", type=int, default=200)
    ap.add_argument("--probability-floor", type=float, default=1e-12)
    ap.add_argument("--workers", type=int,
                    default=min(48, len(os.sched_getaffinity(0))))
    args = ap.parse_args()

    pids, A, Aref = load_scores(args.scores, args.shards)
    n = len(pids)
    feasible = {r: np.zeros(n, bool) for r in RULES}
    reasons = {r: {} for r in RULES}
    t0 = time.monotonic()
    print(json.dumps({"phase": "probe", "prompts": n, "workers": args.workers,
                      "rules": list(RULES)}), flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(A, Aref, args.beta, args.eta,
                                       args.probability_floor, args.max_dual_calls,
                                       args.probe_weight_l1)) as ex:
        for x, out in ex.map(_one, range(n), chunksize=8):
            for rule, (ok, why) in out.items():
                feasible[rule][x] = ok
                if not ok:
                    head = (why or "unknown")
                    reasons[rule][head] = reasons[rule].get(head, 0) + 1
            done += 1
            if done % 200 == 0:
                print(json.dumps({"probed": done, "of": n,
                                  "seconds": round(time.monotonic() - t0, 1)}), flush=True)

    common = np.ones(n, bool)
    for r in RULES:
        common &= feasible[r]
    keep = {pids[i] for i in np.where(common)[0]}

    split_dir = UF / "splits" / args.splits
    dest_dir = UF / "splits" / args.out_splits
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name in ("policy_train", "policy_dev"):
        rows = [json.loads(l) for l in (split_dir / ("%s.jsonl" % name)).open() if l.strip()]
        kept = [r for r in rows if r["prompt_id"] in keep]
        dest = dest_dir / ("%s.jsonl" % name)
        with dest.open("w") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        written[name] = {"in": len(rows), "kept": len(kept), "path": str(dest)}
    report = {
        "prompts_probed": n,
        "feasible_per_rule": {r: int(feasible[r].sum()) for r in RULES},
        "feasible_fraction_per_rule": {r: float(feasible[r].mean()) for r in RULES},
        "common_feasible": int(common.sum()),
        "common_fraction": float(common.mean()),
        "refusal_reasons": reasons,
        "splits_written": written,
        "interpretation": ("a refusal is a property of the prompt: per-prompt Nash "
                           "needs a learner mixture that beats the comparator bank on "
                           "all four checklist items at once. MaxEntBW needs no such "
                           "interior point, which is the difference PROSPER's design "
                           "rests on."),
        "tolerances_unchanged": True,
        "seconds": round(time.monotonic() - t0, 1)}
    out = Path("/work/sub_20260914/prosper/feasibility_%s.json" % args.out_splits)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("prompts_probed", "feasible_per_rule", "common_feasible",
                       "common_fraction", "splits_written", "seconds")}, indent=1),
          flush=True)
    # the refusal comes AFTER the report: a probe that finds nothing feasible is
    # exactly when its diagnosis is needed
    if not written["policy_train"]["kept"]:
        raise SystemExit("no train prompt is solvable by all three per-prompt rules; "
                         "see %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
