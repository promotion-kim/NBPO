"""Paired comparison: does ten draws per pair change what the solver can do?

Same 60 prompts, same pool, same responses, same judge prompt. The only
difference is five draws per order instead of one. For each prompt and each
per-prompt rule this records whether the solve certifies, the weight L1, the
minimum surplus and the effective number of active objectives, then reports the
paired change.

The question this answers is narrow and mechanical: Nash's dual weight is the
inverse surplus, so noise in a small surplus is amplified, and the measured
standard error of p_hat at two draws (0.145) is six times the mean minimum
surplus (0.024). If ten draws shrinks the weight tail and raises the
certification rate, the two-draw budget is a real constraint on the Nash arms.
If it does not, the budget is not the bottleneck and the indistinguishability in
the policy table has another cause.
"""
import argparse, json, os, sys, time
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
import solve_pros4_targets as base

UF = Path("/work/uf4_20260910")
S = Path("/work/sub_20260914")
POOL, K = 8, 4
RULES = ("pw_nash_adaptive", "pw_nash_fixedref", "pw_absolute_maxmin")
_S = {}


def _init(A, Aref, l1):
    _S.update(A=A, Aref=Aref, l1=l1)
    torch.set_num_threads(1)


def _one(x):
    A = np.ascontiguousarray(_S["A"][:, x:x + 1])
    Aref = np.ascontiguousarray(_S["Aref"][:, x:x + 1])
    mu = uniform_policy(1, POOL)
    kw = dict(eta=1.0, inner_solver="exact", dual_solver="root", dual_tol=1e-10,
              M=200, inner_workers=1, probability_floor=1e-12, log_every=0)
    out = {}
    for rule in RULES:
        try:
            if rule == "pw_nash_fixedref":
                rep = FixedReferenceRepresentation(torch.from_numpy(A),
                                                   torch.from_numpy(Aref), mu,
                                                   reference_construction="shared_pool")
            else:
                rep = AdaptiveGameRepresentation(
                    torch.from_numpy(A), torch.from_numpy(Aref), mu,
                    torch.full((K,), 0.25, dtype=torch.float64),
                    reference_construction="shared_pool")
            if rule == "pw_absolute_maxmin":
                res = solve_finite_pool(rep, "absolute_maxmin", weight_l1=_S["l1"], **kw)
            else:
                res = solve_finite_pool(rep, "nash", **kw)
            cert = validate_finite_pool_solution(res)
            ok = bool(cert.get("certified", False)) and \
                abs(float(res.target_log_ratio_check())) <= 1e-9
            w = res.weights.numpy().reshape(-1)
            l1 = float(np.abs(w).sum())
            p = w / max(l1, 1e-30)
            eff = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
            surplus = np.asarray(cert.get("surplus", [np.nan]), dtype=np.float64)
            out[rule] = {"certified": ok, "weight_l1": l1,
                         "effective_objectives": eff,
                         "min_surplus": float(np.min(surplus))}
        except Exception as exc:                  # noqa: BLE001
            out[rule] = {"certified": False, "weight_l1": float("nan"),
                         "effective_objectives": float("nan"),
                         "min_surplus": float("nan"),
                         "error": str(exc).split(";")[0].strip()[:90]}
    return x, out


def run(score_root, pids, l1):
    scores, _ = base.load_scores(str(score_root), 8)
    have = [p for p in pids if p in scores]
    if not have:
        raise SystemExit("no ablation prompts in %s" % score_root)
    A = np.stack([scores[p][0] for p in have], axis=1)
    Aref = np.stack([scores[p][1] for p in have], axis=1)
    res = {}
    with ProcessPoolExecutor(max_workers=min(48, len(os.sched_getaffinity(0))),
                             initializer=_init, initargs=(A, Aref, l1)) as ex:
        for x, out in ex.map(_one, range(len(have)), chunksize=4):
            res[have[x]] = out
    return res


def summarise(res, rule):
    rows = [v[rule] for v in res.values()]
    cert = [r for r in rows if r["certified"]]
    l1 = np.array([r["weight_l1"] for r in cert]) if cert else np.array([np.nan])
    eff = np.array([r["effective_objectives"] for r in cert]) if cert else np.array([np.nan])
    ms = np.array([r["min_surplus"] for r in cert]) if cert else np.array([np.nan])
    return {"prompts": len(rows), "certified": len(cert),
            "certified_fraction": round(len(cert) / max(len(rows), 1), 4),
            "weight_l1": {"mean": round(float(np.nanmean(l1)), 3),
                          "median": round(float(np.nanmedian(l1)), 3),
                          "max": round(float(np.nanmax(l1)), 3)},
            "effective_objectives_mean": round(float(np.nanmean(eff)), 4),
            "min_surplus_mean": round(float(np.nanmean(ms)), 5)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--two-draw-scores", default=str(UF / "scores/pros_scores_all"))
    ap.add_argument("--ten-draw-scores", default=str(UF / "scores/pros_abl10_scores"))
    ap.add_argument("--prompt-list", default=str(S / "prosper/ablation_prompts.json"))
    ap.add_argument("--probe-weight-l1", type=float, default=1.0)
    args = ap.parse_args()

    pids = json.loads(Path(args.prompt_list).read_text())
    t0 = time.monotonic()
    two = run(args.two_draw_scores, pids, args.probe_weight_l1)
    ten = run(args.ten_draw_scores, pids, args.probe_weight_l1)
    common = sorted(set(two) & set(ten))
    two = {p: two[p] for p in common}
    ten = {p: ten[p] for p in common}

    report = {"prompts_compared": len(common), "draws_per_pair": {"before": 2, "after": 10},
              "probe_weight_l1": args.probe_weight_l1, "per_rule": {}}
    for rule in RULES:
        b, a = summarise(two, rule), summarise(ten, rule)
        gained = [p for p in common if ten[p][rule]["certified"] and not two[p][rule]["certified"]]
        lost = [p for p in common if two[p][rule]["certified"] and not ten[p][rule]["certified"]]
        report["per_rule"][rule] = {"two_draws": b, "ten_draws": a,
                                    "newly_certified": len(gained),
                                    "lost_certification": len(lost)}
    report["seconds"] = round(time.monotonic() - t0, 1)
    out = S / "prosper/ablation_draws.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
