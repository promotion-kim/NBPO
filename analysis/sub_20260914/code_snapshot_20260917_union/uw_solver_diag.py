"""Recompute the two residual columns of tab:union-diagnostic-template.

The appendix distinguishes them (app:solver-neural, and the UF-4 paragraph that
already reports the same pair): the DUAL residual is the original, unprojected
Nash stationarity  max_k |s_k - 1/lambda_k|, and the INNER residual is the
maximum inner stationarity spread of the strictly concave subproblem. The
finite-pool certificate computes both, so nothing is invented here.

The per-prompt solve is deterministic (exact inner solver, root dual solver), so
this re-solves each prompt and then ASSERTS that the recomputed policy and
multipliers match the stored per_prompt.npz of the arm that was actually
trained. If they did not match, the residuals would belong to a different solve
and the script refuses to print them.
"""
from __future__ import annotations

import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/work/uf4_20260910/code")
sys.path.insert(0, "/work/sub_20260914/code")
import solve_pros4_targets_uw1 as base
from mnpo_scripts.nbpo_core import uniform_policy
from mnpo_scripts.nbpo_generic import solve_finite_pool, validate_finite_pool_solution
from mnpo_scripts.nbpo_representations import (AdaptiveGameRepresentation,
                                               FixedReferenceRepresentation)

ROOT = Path("/work/uf4_20260910")
OBJECTIVES = base.OBJECTIVES
POOL = base.POOL
_S = {}

ARMS = {"nbpo": ("uw1_pw_nbpo", "adaptive"),
        "fixedref": ("uw1_pw_fixedref", "fixed")}


def _init(A, Aref, beta, eta, floor, kind):
    _S.update(A=A, Aref=Aref, beta=beta, eta=eta, floor=floor, kind=kind)
    torch.set_num_threads(1)


def _diag_one(x):
    A = _S["A"][:, x:x + 1]
    Aref = _S["Aref"][:, x:x + 1]
    a = torch.from_numpy(np.ascontiguousarray(A))
    r = torch.from_numpy(np.ascontiguousarray(Aref))
    if _S["kind"] == "adaptive":
        rep = AdaptiveGameRepresentation(
            a, r, uniform_policy(1, POOL),
            torch.full((len(OBJECTIVES),), _S["beta"], dtype=torch.float64),
            reference_construction="independent_samples")
    else:
        rep = FixedReferenceRepresentation(
            a, r, uniform_policy(1, POOL),
            reference_construction="independent_samples")
    res = solve_finite_pool(rep, "nash", eta=_S["eta"], inner_solver="exact",
                            dual_solver="root", dual_tol=1e-10, M=200,
                            inner_workers=1, probability_floor=_S["floor"],
                            log_every=0)
    cert = validate_finite_pool_solution(res)
    return (x,
            res.pi.numpy().astype(np.float64)[0],
            res.weights.numpy().astype(np.float64),
            float(cert["independent_stationarity_inf"]),
            float(cert["dual_unprojected_residual"]),
            float(cert["dual_projected_residual"]),
            float(cert["nash_complementarity_inf"]),
            int(len(cert.get("lambda_at_lower_bound", []))
                + len(cert.get("lambda_at_upper_bound", []))),
            bool(cert["certified"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--probability-floor", type=float, default=1e-12)
    args = ap.parse_args()

    target_name, kind = ARMS[args.arm]
    tdir = ROOT / "targets" / target_name
    out = {"arm": args.arm, "target_dir": str(tdir), "representation": kind,
           "beta": args.beta, "eta": args.eta, "splits": {}}
    started = time.monotonic()

    for split, split_file in (("train", "policy_train"), ("dev", "policy_dev")):
        scores, _ = base.load_scores(str(ROOT / "scores/uw1"), 4)
        with (ROOT / "splits/uw_v1p" / ("%s.jsonl" % split_file)).open() as s:
            pids = [json.loads(l)["prompt_id"] for l in s if l.strip()]
        A = np.stack([scores[p][0] for p in pids], axis=1)
        Aref = np.stack([scores[p][1] for p in pids], axis=1)

        stored = np.load(tdir / ("%s_per_prompt.npz" % split), allow_pickle=True)
        spids = [str(v) for v in stored["prompt_ids"]]
        if spids != pids:
            raise SystemExit("%s: stored prompt order differs from the split file" % split)

        pi = np.zeros((len(pids), POOL))
        w = np.zeros((len(OBJECTIVES), len(pids)))
        inner = np.zeros(len(pids)); dual = np.zeros(len(pids))
        proj = np.zeros(len(pids)); comp = np.zeros(len(pids))
        bounds = np.zeros(len(pids), int); cert = np.zeros(len(pids), bool)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                                 initargs=(A, Aref, args.beta, args.eta,
                                           args.probability_floor, kind)) as ex:
            for r in ex.map(_diag_one, range(len(pids)), chunksize=8):
                x, pi_x, w_x, i_r, d_r, p_r, c_r, nb, ok = r
                pi[x] = pi_x; w[:, x] = w_x
                inner[x] = i_r; dual[x] = d_r; proj[x] = p_r; comp[x] = c_r
                bounds[x] = nb; cert[x] = ok

        # the residuals must belong to the solve that produced the trained arm
        dpi = float(np.abs(pi - stored["pi"]).max())
        dw = float(np.abs(w - stored["weights"]).max())
        if dpi > 1e-9 or dw > 1e-6:
            raise SystemExit("%s: re-solve does not reproduce the stored solution "
                             "(pi %g, weights %g)" % (split, dpi, dw))

        out["splits"][split] = {
            "n_prompts": len(pids),
            "reproduces_stored_solution": {"max_abs_pi_diff": dpi,
                                           "max_abs_weight_diff": dw},
            "inner_stationarity_inf": {"max": float(inner.max()),
                                       "mean": float(inner.mean())},
            "dual_unprojected_residual": {"max": float(dual.max()),
                                          "mean": float(dual.mean())},
            "dual_projected_residual": {"max": float(proj.max())},
            "nash_complementarity_inf": {"max": float(comp.max())},
            "prompts_with_an_active_lambda_bound": int((bounds > 0).sum()),
            "certified_prompts": int(cert.sum()),
        }
        print(json.dumps({split: out["splits"][split]}), flush=True)

    both_inner = max(out["splits"][s]["inner_stationarity_inf"]["max"] for s in out["splits"])
    both_dual = max(out["splits"][s]["dual_unprojected_residual"]["max"] for s in out["splits"])
    out["reported"] = {"inner_residual_max_over_both_splits": both_inner,
                       "dual_residual_max_over_both_splits": both_dual,
                       "definition_inner": ("max over prompts of the certificate's "
                                            "independent_stationarity_inf: the inf-norm "
                                            "spread of the inner stationarity expression"),
                       "definition_dual": ("max over prompts of |s_k - 1/lambda_k|_inf, the "
                                           "ORIGINAL unprojected Nash dual condition, not "
                                           "the box-projected residual")}
    out["seconds"] = round(time.monotonic() - started, 1)
    dst = Path("/work/sub_20260914/diag") / ("uw1_%s_residuals.json" % args.arm)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["reported"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
