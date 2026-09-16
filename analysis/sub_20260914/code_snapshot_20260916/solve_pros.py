"""Per-prompt target solves for the PROSPER-setting campaign: three arms.

  nbpo      AdaptiveGameRepresentation + "nash"              (Nash bargaining)
  fixedref  FixedReferenceRepresentation + "nash"            (fixed reference)
  prosper   AdaptiveGameRepresentation + "absolute_maxmin"   (MaxEntBW)

Each prompt is its own finite game: four responses from the Qwen2.5-7B-Instruct
base and four of that prompt's own checklist items as the objectives. There is
no shared dual, so the problem decomposes across prompts and every prompt gets
its own weights and its own certificate -- the same structure the campaign's
PROSPER solver already used, and the structure this manuscript argues for
(pooling prompts flattens Nash weights to uniform).

prosper is solved at the weight L1 its own nbpo solve produced FOR THAT PROMPT,
so the two arms differ in the aggregation rule and not in the scale of the
target. Uncertified prompts are reported and excluded; tolerances are never
relaxed to make a prompt certify.
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
import solve_safe_targets as base                       # hashing + writers only
from mnpo_scripts.nbpo_core import uniform_policy
from mnpo_scripts.nbpo_generic import (matched_weight_l1, solve_finite_pool,
                                       validate_finite_pool_solution)
from mnpo_scripts.nbpo_representations import (AdaptiveGameRepresentation,
                                               FixedReferenceRepresentation)
from scripts.nbpo.build_nbpo_pairs import build_rows

ROOT = Path("/work/sub_20260914")
POOL = 4
K = 4
OBJECTIVES = tuple("item%d" % i for i in range(K))
_S = {}


def _init(A, Aref, beta, eta, mode, l1, floor, dual_calls):
    _S.update(A=A, Aref=Aref, beta=beta, eta=eta, mode=mode, l1=l1,
              floor=floor, dual_calls=dual_calls)
    torch.set_num_threads(1)


def _rep(x):
    A = np.ascontiguousarray(_S["A"][:, x:x + 1])
    Aref = np.ascontiguousarray(_S["Aref"][:, x:x + 1])
    mu = uniform_policy(1, POOL)
    if _S["mode"] == "fixedref":
        return FixedReferenceRepresentation(torch.from_numpy(A), torch.from_numpy(Aref),
                                            mu, reference_construction="shared_pool")
    return AdaptiveGameRepresentation(
        torch.from_numpy(A), torch.from_numpy(Aref), mu,
        torch.full((K,), _S["beta"], dtype=torch.float64),
        reference_construction="shared_pool")


def _solve_one(x):
    """Solve prompt x alone, or report why it has no certified solution.

    A per-prompt Nash bargaining problem needs an interior point: some response
    mixture must beat the disagreement point on EVERY one of the prompt's four
    checklist items at once. In this self-play game the reference is the uniform
    mixture over the same four responses and the payoff is antisymmetric, so
    d_k = 0 and the requirement is a single pi with pi . (A_k mu) > 0 for all k.
    When the items conflict strongly enough no such pi exists and the solver
    refuses, correctly. That is a property of the prompt, not a bug, so the
    exception is caught here and the prompt is excluded and counted instead of
    killing the whole arm. Tolerances are never relaxed to manufacture a
    solution.
    """
    kw = dict(eta=_S["eta"], inner_solver="exact", dual_solver="root",
              dual_tol=1e-10, inner_workers=1, probability_floor=_S["floor"],
              log_every=0)
    try:
        rep = _rep(x)
        if _S["mode"] == "prosper":
            result = solve_finite_pool(rep, "absolute_maxmin", M=_S["dual_calls"],
                                       weight_l1=float(_S["l1"][x]), **kw)
            l1_out = float(_S["l1"][x])
        else:
            result = solve_finite_pool(rep, "nash", M=_S["dual_calls"], **kw)
            l1_out = float(matched_weight_l1(result))
        cert = validate_finite_pool_solution(result)
        return (x,
                result.pi.numpy().astype(np.float64),
                result.nu_update.numpy().astype(np.float64),
                result.weights.numpy().astype(np.float64),
                result.target_log_ratio.numpy().astype(np.float64),
                float(result.target_log_ratio_check()),
                bool(cert.get("certified", False)),
                float(np.min(np.asarray(cert["surplus"], dtype=np.float64))),
                l1_out, None)
    except Exception as exc:                      # noqa: BLE001 - reported, not hidden
        return (x, None, None, None, None, float("nan"), False, float("nan"),
                float("nan"), str(exc)[:400])


def load_scores(root, shards):
    pids, A, Aref, manifests = [], [], [], []
    for s in range(shards):
        path = Path(root) / ("scores_shard%d.npz" % s)
        if not path.exists():
            continue
        rec = json.loads((Path(root) / ("complete_shard%d.json" % s)).read_text())
        if base.file_hash(path) != rec["sha256"]:
            raise SystemExit("score shard hash mismatch: %s" % path)
        manifests.append(rec)
        z = np.load(path, allow_pickle=True)
        ids = [str(p) for p in z["prompt_ids"]]
        if z["A_policy"].shape[0] != K or z["A_policy"].shape[2:] != (POOL, POOL):
            raise SystemExit("unexpected tensor shape %s" % (z["A_policy"].shape,))
        pids += ids
        A.append(z["A_policy"])
        Aref.append(z["A_ref"])
    if not pids:
        raise SystemExit("no score shards under %s" % root)
    return pids, np.concatenate(A, axis=1), np.concatenate(Aref, axis=1), manifests


def load_responses(tag):
    """{str(i): {pid: event}} in the shape build_rows expects."""
    path = ROOT / "responses" / tag / "responses.jsonl"
    per = {str(i): {} for i in range(POOL)}
    with path.open() as s:
        for line in s:
            if not line.strip():
                continue
            e = json.loads(line)
            i = int(e["response_index"])
            if i >= POOL:
                continue
            per[str(i)][e["prompt_id"]] = {
                "prompt": e["instruction"], "generated_text": e["response"],
                "candidate_id": e["response_id"], "response_sha256": e["response_sha256"],
                "seed": e["seed"], "n_tokens": e["n_tokens"]}
    return per, base.file_hash(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("nbpo", "fixedref", "prosper"))
    ap.add_argument("--scores", default=str(ROOT / "scores/pros_train_v1"))
    ap.add_argument("--responses", default="pros_train2000")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--l1-from", default=None,
                    help="prosper only: the nbpo per-prompt npz whose weight L1 to match")
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--max-dual-calls", type=int, default=200)
    ap.add_argument("--probability-floor", type=float, default=1e-12)
    ap.add_argument("--workers", type=int,
                    default=min(48, len(os.sched_getaffinity(0))))
    args = ap.parse_args()

    out = ROOT / "targets" / args.out_name
    if (out / "complete.json").exists():
        print(json.dumps({"skipped": "already solved", "path": str(out)}))
        return 0
    out.mkdir(parents=True, exist_ok=True)

    pids, A, Aref, manifests = load_scores(args.scores, args.shards)
    policy, responses_sha = load_responses(args.responses)
    missing = [p for p in pids if any(p not in policy[str(i)] for i in range(POOL))]
    if missing:
        raise SystemExit("responses missing for %d scored prompts, e.g. %s"
                         % (len(missing), missing[:3]))

    restricted_to = None
    if args.mode == "prosper":
        if not args.l1_from:
            raise SystemExit("prosper needs --l1-from")
        z = np.load(args.l1_from, allow_pickle=True)
        by_pid = {str(p): float(v) for p, v in zip(z["prompt_ids"], z["weight_l1"])}
        # The reference arm certifies only a subset of prompts, so prosper is
        # solved on exactly that subset rather than erroring on the rest. This is
        # also what matching requires: the arms have to be compared on the same
        # prompts, and the matched weight L1 only exists where the reference arm
        # produced one.
        keep_pids = [p for p in pids if p in by_pid]
        if not keep_pids:
            raise SystemExit("no prompt has a matched L1 in %s" % args.l1_from)
        index = {p: i for i, p in enumerate(pids)}
        sel = np.array([index[p] for p in keep_pids])
        pids = keep_pids
        A = np.ascontiguousarray(A[:, sel])
        Aref = np.ascontiguousarray(Aref[:, sel])
        restricted_to = {"l1_source": args.l1_from,
                         "prompts_in_source": len(by_pid),
                         "prompts_restricted_to": len(pids)}
    l1 = np.zeros(len(pids))
    if args.mode == "prosper":
        l1 = np.array([by_pid[p] for p in pids])

    n = len(pids)
    pi = np.zeros((n, POOL))
    nu = np.zeros((K, n, POOL))
    weights = np.zeros((K, n))
    g = np.zeros((n, POOL))
    identity = np.zeros(n)
    certified = np.zeros(n, bool)
    min_surplus = np.zeros(n)
    l1_out = np.zeros(n)
    reasons = {}
    t0 = time.monotonic()
    print(json.dumps({"phase": "solve", "mode": args.mode, "prompts": n,
                      "workers": args.workers, "per_prompt_weights": True}), flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(A, Aref, args.beta, args.eta, args.mode, l1,
                                       args.probability_floor, args.max_dual_calls)) as ex:
        for res in ex.map(_solve_one, range(n), chunksize=8):
            x, pi_x, nu_x, w_x, g_x, ident, cert, ms, l1v, why = res
            if why is not None:
                reasons[pids[x]] = why
                identity[x] = np.nan
                done += 1
                continue
            pi[x] = pi_x[0]
            nu[:, x] = nu_x[:, 0]
            weights[:, x] = w_x
            g[x] = g_x[0]
            identity[x], certified[x], min_surplus[x], l1_out[x] = ident, cert, ms, l1v
            done += 1
            if done % 500 == 0:
                print(json.dumps({"solved": done, "of": n,
                                  "seconds": round(time.monotonic() - t0, 1),
                                  "certified_so_far": int(certified[:done].sum())}), flush=True)

    finite = np.isfinite(identity)
    keep = np.where(certified & finite & (np.abs(np.nan_to_num(identity)) <= 1e-9))[0]
    reason_counts = {}
    for why in reasons.values():
        head = why.split(";")[0].strip()
        reason_counts[head] = reason_counts.get(head, 0) + 1
    report_uncertified = {
        "prompts_attempted": n,
        "prompts_solver_refused": len(reasons),
        "prompts_certified": int(certified.sum()),
        "prompts_failing_identity": int((finite & (np.abs(np.nan_to_num(identity)) > 1e-9)).sum()),
        "prompts_kept": int(len(keep)),
        "max_identity_residual_kept": float(np.abs(identity[keep]).max()) if len(keep) else None,
        "refusal_reasons": reason_counts,
        "interpretation": ("a refusal means this prompt has no certified solution under "
                           "the declared tolerances -- for nash, no response mixture beats "
                           "the uniform reference on all four items at once. It is a "
                           "property of the prompt and is reported, not worked around."),
        "tolerances_unchanged": True}
    base.write_json(out / "refusals.json",
                    {"reasons": reasons, "summary": report_uncertified})
    if not len(keep):
        base.write_json(out / "failed.json", report_uncertified)
        raise SystemExit("no prompt certified; see %s" % (out / "failed.json"))

    kept_pids = [pids[i] for i in keep]
    per_prompt = out / "per_prompt.npz"
    np.savez_compressed(per_prompt, pi=pi[keep], weights=weights[:, keep], g=g[keep],
                        min_surplus=min_surplus[keep], identity_residual=identity[keep],
                        weight_l1=l1_out[keep],
                        prompt_ids=np.array(kept_pids, dtype=object))

    shared_meta = {"base_model": "/work/models/bases/Qwen2.5-7B-Instruct",
                   "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
                   "responses_tag": args.responses,
                   "responses_sha256": responses_sha,
                   "teacher_manifest_sha256": base.object_hash(manifests),
                   "gpm_teacher": manifests[0]["gpm_teacher"],
                   "bt_teacher": manifests[0]["bt_teacher"],
                   "setting": ("PROSPER section 6.1: WildChecklists, N=4 base responses, "
                               "per-checklist-item judging (PSC)")}
    meta = {"prompt_ids": kept_pids, "objectives": list(OBJECTIVES),
            "reference_construction": "shared_pool",
            "policy_learner_ids": ["policy:%d" % i for i in range(POOL)],
            "comparator_ids": ["ref:%d" % j for j in range(POOL)],
            "split": "train", **shared_meta}
    base.write_json(out / "meta.json", meta)

    aggregation = {"nbpo": "prompt_wise_nash", "fixedref": "prompt_wise_nash_fixed_reference",
                   "prosper": "prompt_wise_absolute_maxmin"}[args.mode]
    representation = {"nbpo": "adaptive_game", "fixedref": "fixed_reference",
                      "prosper": "adaptive_game"}[args.mode]
    solver_dir = out / "solver"
    base.write_json(solver_dir / "solution.json", {
        "target_mode": "canonical_logratio", "target_column": "nbpo_logratio_target",
        "target_units": "final_logratio_change", "eta_already_included": True,
        "representation": representation, "aggregation": aggregation,
        "weights_scope": "per prompt; there is no shared dual",
        "per_prompt_artifact": str(per_prompt),
        "per_prompt_artifact_sha256": base.file_hash(per_prompt),
        "n_prompts": len(kept_pids), "split": "train",
        "weight_l1_mean": float(l1_out[keep].mean()),
        "min_surplus_mean": float(min_surplus[keep].mean()),
        "min_surplus_negative_prompts": int((min_surplus[keep] < 0).sum()),
        "beta": args.beta, "eta": args.eta,
        "certification": report_uncertified,
        "restricted_to": restricted_to,
        "solver_source_sha256": base.file_hash(__file__), **shared_meta})
    solver_hash = base.file_hash(solver_dir / "solution.json")
    provenance = {"solver_artifact_sha256": solver_hash, "solver_hash": solver_hash,
                  "target_artifact_hash": base.file_hash(per_prompt),
                  "representation": representation, "aggregation": aggregation,
                  "split": "train", "panel": "WildChecklists-PROSPER", **shared_meta}
    betas = np.full(K, args.beta)

    def pair_rows():
        for xi, pid in enumerate(kept_pids):
            x = int(keep[xi])
            learners = {str(i): {pid: policy[str(i)][pid]} for i in range(POOL)}
            rows = build_rows([pid], list(OBJECTIVES), A[:, x:x + 1], nu[:, x:x + 1],
                              weights[:, x], betas, learners, None,
                              np.random.default_rng(42), "canonical_logratio",
                              meta, provenance,
                              canonical_data={"g": g[x:x + 1], "p_star": pi[x:x + 1],
                                              "p_t": np.full((1, POOL), 1.0 / POOL)})
            for row in rows:
                row = base.quantize_canonical_row(row)
                a, b = row["chosen_candidate_index"], row["rejected_candidate_index"]
                row["chosen_response_id"] = policy[str(a)][pid]["candidate_id"]
                row["rejected_response_id"] = policy[str(b)][pid]["candidate_id"]
                row["prompt_weights"] = [float(v) for v in weights[:, x]]
                yield row

    pair_path = out / "pairs" / "train.jsonl"
    base.write_jsonl(pair_path, pair_rows())
    base.write_json(out / "complete.json", {
        "mode": args.mode, "aggregation": aggregation, "representation": representation,
        "prompts": len(kept_pids), "pairs_path": str(pair_path),
        "pairs_sha256": base.file_hash(pair_path),
        "certification": report_uncertified,
        "restricted_to": restricted_to,
        "solver_artifact_sha256": solver_hash,
        "per_prompt_artifact_sha256": base.file_hash(per_prompt),
        "seconds": round(time.monotonic() - t0, 1), **shared_meta})
    print(json.dumps({"mode": args.mode, "prompts_kept": len(kept_pids),
                      "certified": int(certified.sum()), "of": n,
                      "pairs": str(pair_path),
                      "seconds": round(time.monotonic() - t0, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
