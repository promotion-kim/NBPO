"""Algorithm 1's acceptance step for the PROMPT-WISE path, evaluated per prompt.

The manuscript's Algorithm 1 does not stop at a certified finite target. It
trains a neural candidate against that target, checks the candidate on held-out
development data, and only then makes it the next policy; a candidate that fails
the check is discarded and the parent is retained. An implementation audit found
that the campaign's prompt-wise training path ran

    job generator -> torch.distributed.run -> run_mnpo -> train() -> save_model()

with nothing between or after it that evaluated, rejected or promoted anything.
The certified solver residual is not a substitute: it certifies the finite target,
not the policy fitted to it.

The existing `scripts/nbpo/eval_game_value.py` cannot be reused here. It reduces
to a scalar minimum over PROMPT-AVERAGED surpluses, which is the right object for
the shared-weight Global Nash control and the wrong one for a rule whose
multipliers are fitted per prompt: two prompts at +.2 and -.1 average to +.05 and
pass, while one of them is a prompt the compromise failed on. This file evaluates
the array and applies the declared policy to it.

What it measures, for each development prompt x and objective k:

  the candidate's distribution over that prompt's eight learner occurrences,
  by importance reweighting on the sampled support,
      p(i) proportional to p_t(i) * exp(log pi_cand(y_i) - log pi_parent(y_i))
  its regularized game value V_{k,beta}(p) against the reference bank, and
  the surplus s_k(x) = V_{k,beta}(p) - d_k(x)
  against the disagreement point d the solver recorded for that prompt.

The declared checks, all three of which must hold:

  finite    every surplus is finite; a NaN or an infinity is a failure, never a
            pass, because `nan <= 0` is False and would otherwise promote
  coverage  the fraction of development prompts whose surplus is positive on
            EVERY objective is at least --coverage-min
  fit       the held-out nMSE of the realized log-ratio against the canonical
            target is at most --nmse-max

On a pass the candidate is promoted to a versioned accepted directory with a
symlink, and the promotion record names the checkpoint, its fingerprint and the
exact artifacts the decision was read from. On a failure nothing is promoted and
the parent path is returned, which is what the caller must then evaluate.

This gate exists from 2026-09-18. It does not apply retroactively: every result
already reported in the manuscript came from an ungated projection, and that is
a fact about those runs, not something a later gate can change. Anything reported
as gated must name a promotion record written by this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/work/uf4_20260910/code")
from diag_neural_pools import load_pool_rows, sequence_logprobs   # noqa: E402

ROOT = Path("/work/uf4_20260910")
POOL = 8


def fingerprint(path: Path) -> str:
    """A content digest of the checkpoint's weight and config files."""
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file() and f.suffix in (".safetensors", ".bin", ".json", ".model"):
            h.update(f.name.encode())
            with f.open("rb") as s:
                for c in iter(lambda: s.read(1 << 22), b""):
                    h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True, help="the trained checkpoint under test")
    ap.add_argument("--parent", required=True,
                    help="the policy retained if the candidate fails; also the "
                         "reference the importance weights are taken against")
    ap.add_argument("--targets", required=True,
                    help="targets/<name>: the per-prompt solve this candidate was fitted to")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--scores", required=True,
                    help="scores/<dir> the solve read; the gate evaluates the SAME game, "
                         "so it goes through the solver's own loader and inherits its "
                         "tensor-role schema check")
    ap.add_argument("--solver-module", required=True,
                    help="the panel's solve_pros4_targets_<panel>.py, whose load_scores "
                         "and OBJECTIVES define the game being gated")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--coverage-min", type=float, default=0.5,
                    help="minimum fraction of development prompts positive on every "
                         "objective; declare it before running, not after reading")
    ap.add_argument("--nmse-max", type=float, default=1.0,
                    help="maximum held-out nMSE; 1.0 is the value of predicting zero")
    ap.add_argument("--nmse-from", default=None,
                    help="trainer_state.json of the candidate, for the recorded "
                         "held-out nMSE; omit to skip the fit check and say so")
    ap.add_argument("--promote-to", default=None,
                    help="symlink to point at the accepted checkpoint; without it "
                         "the gate reports and promotes nothing")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--logprobs", default=None,
                    help="JSON {candidate: {id: logp}, parent: {id: logp}} of sequence "
                         "log-probabilities already computed. Lets the same decision be "
                         "re-derived, and tested, without a forward pass; the models are "
                         "not loaded at all when it is given.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tdir = Path(args.targets)
    npz = tdir / ("%s_per_prompt.npz" % args.split)
    if not npz.exists():
        raise SystemExit(
            "%s has no %s_per_prompt.npz. This gate is defined on a per-prompt solve; a "
            "shared-weight solve has no per-prompt disagreement point and must be gated "
            "with the global evaluator instead." % (tdir, args.split))
    z = np.load(npz, allow_pickle=True)
    pids = [str(p) for p in z["prompt_ids"]]
    recorded_min = np.asarray(z["min_surplus"], dtype=np.float64)

    candidates = {pid: ["%s:learner:%d" % (pid, i) for i in range(POOL)] for pid in pids}
    wanted = {c for v in candidates.values() for c in v}
    order = [c for pid in pids for c in candidates[pid]]
    if args.logprobs:
        rows = None            # no tokenization is needed when the log-probs are given
    else:
        rows = load_pool_rows(args.pool, wanted)
        missing = sorted(wanted - set(rows))
        if missing:
            raise SystemExit("missing cached tokenization for %d occurrences" % len(missing))

    if args.logprobs:
        cached = json.loads(Path(args.logprobs).read_text())
        parent_logp = {k: float(v) for k, v in cached["parent"].items()}
        cand_logp = {k: float(v) for k, v in cached["candidate"].items()}
        for name, d in (("parent", parent_logp), ("candidate", cand_logp)):
            gaps = [c for c in order if c not in d]
            if gaps:
                raise SystemExit("%s log-probabilities are missing %d occurrences"
                                 % (name, len(gaps)))
        logprob_source = args.logprobs
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def score(path: str):
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                local_files_only=True).to(device).eval()
            out = sequence_logprobs(model, rows, order, device, args.batch)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            return dict(zip(order, out))

        parent_logp = score(args.parent)
        cand_logp = score(args.candidate)
        logprob_source = "forward pass on %s" % device

    # the candidate's distribution over each prompt's own eight occurrences
    pt = np.full(POOL, 1.0 / POOL)
    p_cand = np.empty((len(pids), POOL), dtype=np.float64)
    unusable = set()
    for x, pid in enumerate(pids):
        cs = candidates[pid]
        delta = np.array([cand_logp[c] - parent_logp[c] for c in cs], dtype=np.float64)
        if not np.all(np.isfinite(delta)):
            # A prompt whose log-probability is NaN or infinite has no usable
            # distribution. It is recorded as a failure of the finiteness check,
            # not raised: a gate that crashes on a bad value leaves the caller
            # with no decision, and no decision is indistinguishable downstream
            # from a pass.
            unusable.add(pid)
            p_cand[x] = pt
            continue
        logits = np.log(pt) + delta
        logits -= logits.max()
        w = np.exp(logits)
        p_cand[x] = w / w.sum()

    # the surplus is evaluated by the paper's own object rather than a second
    # implementation of Eq. 8 here: same A_policy, same A_ref, same beta, same
    # reference measure as the solve, so a disagreement between the two would be
    # a disagreement about the policy and not about the algebra
    import importlib.util
    spec = importlib.util.spec_from_file_location("gated_solver", args.solver_module)
    solver = importlib.util.module_from_spec(spec)
    sys.modules["gated_solver"] = solver
    spec.loader.exec_module(solver)
    from mnpo_scripts.nbpo_core import uniform_policy
    from mnpo_scripts.nbpo_representations import AdaptiveGameRepresentation

    scores, _ = solver.load_scores(args.scores, 4)
    absent = [pid for pid in pids if pid not in scores]
    if absent:
        raise SystemExit("%d development prompts have no score tensor" % len(absent))
    A = np.stack([scores[pid][0] for pid in pids], axis=1)
    Aref = np.stack([scores[pid][1] for pid in pids], axis=1)
    K = A.shape[0]
    construction = getattr(solver, "REFERENCE_CONSTRUCTION", "shared_pool")
    # ONE REPRESENTATION PER PROMPT. Built over all prompts at once,
    # `game_values` divides by X -- it returns the prompt-averaged value, which is
    # the object the shared-weight control is gated on and the wrong one here: two
    # prompts at +.2 and -.1 average to +.05 and would pass while the compromise
    # failed on one of them. Constructing X=1 at a time keeps the audited
    # implementation of Eq. (8) and Eq. (10) and evaluates it where the rule is
    # defined.
    surplus = np.empty((K, len(pids)), dtype=np.float64)
    beta_vec = torch.full((K,), args.beta, dtype=torch.float64)
    for x in range(len(pids)):
        if pids[x] in unusable:
            surplus[:, x] = np.nan
            continue
        rep = AdaptiveGameRepresentation(
            torch.from_numpy(A[:, x:x + 1]), torch.from_numpy(Aref[:, x:x + 1]),
            uniform_policy(1, POOL), beta_vec,
            reference_construction=construction)
        s = rep.surplus(torch.from_numpy(p_cand[x:x + 1])).detach().cpu().numpy()
        surplus[:, x] = np.asarray(s, dtype=np.float64).reshape(K)

    per_prompt, nonfinite = [], 0
    for x, pid in enumerate(pids):
        s = np.asarray(surplus[:, x], dtype=np.float64)
        ok = pid not in unusable and np.all(np.isfinite(s))
        if not ok:
            nonfinite += 1
        per_prompt.append({
            "prompt_id": pid,
            "surplus": None if pid in unusable else s.tolist(),
            "min_surplus": float(np.min(s)) if ok else None,
            "recorded_target_min_surplus": float(recorded_min[x]),
            "reason": ("candidate log-probability is not finite" if pid in unusable
                       else ("surplus is not finite" if not ok else None))})

    finite_ok = nonfinite == 0
    mins = np.array([r["min_surplus"] if r["min_surplus"] is not None else np.nan
                     for r in per_prompt], dtype=np.float64)
    positive = np.isfinite(mins) & (mins > 0)
    coverage = float(positive.mean())

    nmse, fit_ok, fit_note = None, None, "not checked: --nmse-from was not given"
    if args.nmse_from:
        state = json.loads(Path(args.nmse_from).read_text())
        seen = [e for e in state.get("log_history", []) if "eval_nbpo/nmse" in e]
        if seen:
            nmse = float(seen[-1]["eval_nbpo/nmse"])
            fit_ok = nmse <= args.nmse_max
            fit_note = "last recorded held-out nMSE"
        else:
            fit_note = "no eval_nbpo/nmse in the trainer state"

    checks = {"finite": finite_ok,
              "coverage": coverage >= args.coverage_min,
              "fit": fit_ok}
    accepted = bool(finite_ok and checks["coverage"] and (fit_ok is not False))

    record = {
        "gate": "prompt_wise_local_acceptance",
        "algorithm_1_step": "held-out development check before the candidate becomes pi_{t+1}",
        "decided_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidate": args.candidate,
        "parent": args.parent,
        "targets": str(tdir),
        "pool": args.pool,
        "scores": args.scores,
        "solver_module": args.solver_module,
        "split": args.split,
        "prompts": len(pids),
        "logprob_source": logprob_source,
        "thresholds": {"coverage_min": args.coverage_min, "nmse_max": args.nmse_max,
                       "beta": args.beta},
        "measured": {"coverage": coverage, "nonfinite_prompts": nonfinite,
                     "prompts_with_unusable_logprobs": sorted(unusable),
                     "held_out_nmse": nmse, "nmse_note": fit_note,
                     "min_surplus_quantiles": {
                         q: (float(np.nanquantile(mins, v)) if np.any(np.isfinite(mins)) else None)
                         for q, v in (("p05", .05), ("p50", .5), ("p95", .95))}},
        "checks": checks,
        "accepted": accepted,
        "estimator": ("candidate distribution by importance reweighting on the sampled "
                      "support, p ∝ p_t · exp(log pi_cand − log pi_parent); this is not the "
                      "policy's distribution over all responses"),
        "not_retroactive": ("this gate exists from 2026-09-18; results reported before it "
                            "came from an ungated projection and must not be described as "
                            "gated"),
        "per_prompt": per_prompt,
    }

    if accepted and args.promote_to:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        fp = fingerprint(Path(args.candidate))
        versioned = Path(args.promote_to).parent / ("%s.accepted.%s" % (
            Path(args.promote_to).name, stamp))
        versioned.parent.mkdir(parents=True, exist_ok=True)
        (versioned).mkdir(exist_ok=False)
        (versioned / "ACCEPTED.json").write_text(json.dumps(
            {"checkpoint": args.candidate, "fingerprint": fp,
             "gate_record": args.out, "decided_utc": record["decided_utc"]},
            indent=1) + "\n")
        link = Path(args.promote_to)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(versioned)
        record["promotion"] = {"versioned_dir": str(versioned), "symlink": str(link),
                               "candidate_fingerprint": fp}
        record["policy_after_this_stage"] = args.candidate
    else:
        record["promotion"] = None
        record["policy_after_this_stage"] = args.parent
        if not accepted:
            record["consequence"] = ("the candidate is rejected and the parent is retained; "
                                     "evaluation must be run on the parent, and this stage "
                                     "reports no improvement rather than a smaller one")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in (
        "accepted", "checks", "measured", "policy_after_this_stage")}, indent=1))
    return 0 if accepted else 3


if __name__ == "__main__":
    raise SystemExit(main())
