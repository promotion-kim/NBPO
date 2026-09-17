"""Recompute the two residual columns of tab:union-diagnostic-template, any panel.

The appendix distinguishes them: the DUAL residual is the original, unprojected
Nash stationarity max_k |s_k - 1/lambda_k|, and the INNER residual is the
maximum inner stationarity spread of the strictly concave subproblem. The
finite-pool certificate computes both, so nothing is invented here.

The per-prompt solve is deterministic (exact inner solver, root dual solver), so
this re-solves each prompt and then ASSERTS that the recomputed policy and
multipliers match the stored per_prompt.npz of the arm that was actually
trained. If they did not match, the residuals would belong to a different solve
and the script refuses to print them.

Generalized from uw_solver_diag.py: the panel, its score directory, its
restricted split, the objective count and the representation are flags, and
`--panel uw1 --arm pw_nbpo` reproduces the UW numbers already in the manuscript.
"""
from __future__ import annotations

import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/work/uf4_20260910/code")
sys.path.insert(0, "/work/sub_20260914/code")
from mnpo_scripts.nbpo_core import uniform_policy
from mnpo_scripts.nbpo_generic import solve_finite_pool, validate_finite_pool_solution
from mnpo_scripts.nbpo_representations import (AdaptiveGameRepresentation,
                                               FixedReferenceRepresentation)

ROOT = Path("/work/uf4_20260910")
POOL = 8
_S = {}


def _init(A, Aref, beta, eta, floor, kind, K):
    _S.update(A=A, Aref=Aref, beta=beta, eta=eta, floor=floor, kind=kind, K=K)
    torch.set_num_threads(1)


def _diag_one(x):
    a = torch.from_numpy(np.ascontiguousarray(_S["A"][:, x:x + 1]))
    r = torch.from_numpy(np.ascontiguousarray(_S["Aref"][:, x:x + 1]))
    if _S["kind"] == "adaptive":
        rep = AdaptiveGameRepresentation(
            a, r, uniform_policy(1, POOL),
            torch.full((_S["K"],), _S["beta"], dtype=torch.float64),
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
    ap.add_argument("--panel", required=True, help="uw1, us1 or ut1")
    ap.add_argument("--arm", required=True, help="pw_nbpo or pw_fixedref")
    ap.add_argument("--scores", required=True)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--splits", required=True, help="restricted split dir, e.g. us_v1p")
    ap.add_argument("--objectives", type=int, required=True)
    ap.add_argument("--representation", default="adaptive",
                    choices=("adaptive", "fixed"))
    ap.add_argument("--target-name", default=None,
                    help="targets/<name>; defaults to <panel>_<arm>")
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--probability-floor", type=float, default=1e-12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # the panel's own score loader, so the tensors are assembled exactly as its
    # solvers assemble them
    base = __import__("solve_pros4_targets_%s" % args.panel)
    if len(base.OBJECTIVES) != args.objectives:
        raise SystemExit("panel %s declares %d objectives, not %d"
                         % (args.panel, len(base.OBJECTIVES), args.objectives))

    target = args.target_name or ("%s_%s" % (args.panel, args.arm))
    tdir = ROOT / "targets" / target
    out = {"panel": args.panel, "arm": args.arm, "target_dir": str(tdir),
           "representation": args.representation, "objectives": args.objectives,
           "beta": args.beta, "eta": args.eta, "splits": {}}
    started = time.monotonic()

    for split, split_file in (("train", "policy_train"), ("dev", "policy_dev")):
        scores, _ = base.load_scores(args.scores, args.shards)
        with (ROOT / "splits" / args.splits / ("%s.jsonl" % split_file)).open() as s:
            pids = [json.loads(l)["prompt_id"] for l in s if l.strip()]
        A = np.stack([scores[p][0] for p in pids], axis=1)
        Aref = np.stack([scores[p][1] for p in pids], axis=1)

        stored = np.load(tdir / ("%s_per_prompt.npz" % split), allow_pickle=True)
        if [str(v) for v in stored["prompt_ids"]] != pids:
            raise SystemExit("%s: stored prompt order differs from the split file" % split)

        n = len(pids)
        pi = np.zeros((n, POOL)); w = np.zeros((args.objectives, n))
        inner = np.zeros(n); dual = np.zeros(n); proj = np.zeros(n)
        comp = np.zeros(n); bounds = np.zeros(n, int); cert = np.zeros(n, bool)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                                 initargs=(A, Aref, args.beta, args.eta,
                                           args.probability_floor,
                                           args.representation, args.objectives)) as ex:
            for r in ex.map(_diag_one, range(n), chunksize=8):
                x, pi_x, w_x, i_r, d_r, p_r, c_r, nb, ok = r
                pi[x] = pi_x; w[:, x] = w_x
                inner[x] = i_r; dual[x] = d_r; proj[x] = p_r; comp[x] = c_r
                bounds[x] = nb; cert[x] = ok

        dpi = float(np.abs(pi - stored["pi"]).max())
        dw = float(np.abs(w - stored["weights"]).max())
        if dpi > 1e-9 or dw > 1e-6:
            raise SystemExit("%s: re-solve does not reproduce the stored solution "
                             "(pi %g, weights %g)" % (split, dpi, dw))

        out["splits"][split] = {
            "n_prompts": n,
            "reproduces_stored_solution": {"max_abs_pi_diff": dpi,
                                           "max_abs_weight_diff": dw},
            "inner_stationarity_inf": {"max": float(inner.max()),
                                       "mean": float(inner.mean())},
            "dual_unprojected_residual": {"max": float(dual.max()),
                                          "mean": float(dual.mean())},
            "dual_projected_residual": {"max": float(proj.max())},
            "nash_complementarity_inf": {"max": float(comp.max())},
            "prompts_with_an_active_lambda_bound": int((bounds > 0).sum()),
            "certified_prompts": int(cert.sum())}
        print(json.dumps({split: out["splits"][split]}), flush=True)

    out["reported"] = {
        "inner_residual_max_over_both_splits":
            max(v["inner_stationarity_inf"]["max"] for v in out["splits"].values()),
        "dual_residual_max_over_both_splits":
            max(v["dual_unprojected_residual"]["max"] for v in out["splits"].values()),
        "definition_inner": ("max over prompts of the certificate's "
                             "independent_stationarity_inf: the inf-norm spread of "
                             "the inner stationarity expression"),
        "definition_dual": ("max over prompts of |s_k - 1/lambda_k|_inf, the ORIGINAL "
                            "unprojected Nash dual condition, not the box-projected "
                            "residual")}
    out["seconds"] = round(time.monotonic() - started, 1)
    dst = Path(args.out or ("/work/sub_20260914/diag/%s_%s_residuals.json"
                            % (args.panel, args.arm)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["reported"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
