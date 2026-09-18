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

The predicate is the manuscript's, and it is not a coverage fraction. Appendix H
says, for the NBPO family: "Require every retained local surplus > 1e-8". So the
paper profile is a universal quantifier over the certified development
monitoring subset -- one prompt at or below the tolerance rejects the candidate.
The Global Nash control has a different contract in the same paragraph:
"aggregate surplus > 1e-8, plus dev nMSE <= 1 on certified dev targets", which
is a statement about the prompt-averaged value and carries the fitting check
that the per-prompt rule does not.

Three profiles, and the profile is recorded in the decision:

  paper-nbpo    every local surplus > surplus-tol, on every objective, on every
                prompt of the monitoring subset. No nMSE term: the manuscript
                attaches that one to the control, not to NBPO.
  paper-global  aggregate (prompt-averaged) surplus > surplus-tol on every
                objective, AND a finite dev nMSE in [0, nmse-max].
  coverage      a RELAXED diagnostic: the fraction of prompts positive on every
                objective is at least --coverage-min. It is not the paper
                contract and the record says so. It exists because the fraction
                is the informative number when the universal predicate fails,
                and an earlier version of this file used it as the default,
                which misrepresented the contract.

An earlier version also treated a MISSING fit result as a pass -- an empty
trainer log_history gave fit=null and accepted=true -- and accepted nMSE of
negative infinity. Both are closed: where the profile requires a fit, the value
must be present and finite and in range, and an absent or non-finite value is a
rejection.

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
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

def _pool_helpers():
    """Resolve the pool reader and the log-probability scorer.

    Looks beside this file first, then on the import path, and only then at the
    historical campaign directory. An earlier version inserted the pod path
    unconditionally, so `--help` failed from a clean checkout and the module was
    not usable outside the machine it was written on. The import is deferred to
    call time because the cached-log-probability path needs neither helper.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here, here / "helpers", Path("/work/uf4_20260910/code")):
        if (candidate / "diag_neural_pools.py").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            break
    try:
        from diag_neural_pools import load_pool_rows, sequence_logprobs
    except ImportError as exc:
        raise SystemExit(
            "diag_neural_pools is needed to score a checkpoint and was not found next to "
            "%s, on PYTHONPATH, or at /work/uf4_20260910/code (%s). Either place it beside "
            "this file or pass --logprobs to run from cached log probabilities, which needs "
            "no forward pass." % (here, exc))
    return load_pool_rows, sequence_logprobs

ROOT = Path("/work/uf4_20260910")
POOL = 8


def candidate_problem(path: Path):
    """Why this path cannot be fingerprinted, or None.

    Separated from fingerprint() so the caller can turn it into a recorded
    rejection instead of an exit. An earlier version walked whatever was there
    and returned the sha256 of nothing for a path that did not exist, so a
    cached-log-probability run could write an ACCEPTED record naming a
    checkpoint that was never trained.
    """
    if not path.exists():
        return "candidate path does not exist: %s" % path
    if not [f for f in path.rglob("*")
            if f.is_file() and f.suffix in (".safetensors", ".bin")]:
        return ("%s holds no .safetensors or .bin weights; a fingerprint over an empty "
                "file set would certify nothing" % path)
    return None


def fingerprint(path: Path) -> str:
    """A content digest of the checkpoint's weight and config files.

    Refuses a path that carries no weights. An earlier version walked whatever
    was there and returned the sha256 of nothing for a path that did not exist,
    so a cached-log-probability run could write an ACCEPTED record naming a
    checkpoint that was never trained.
    """
    problem = candidate_problem(path)
    if problem:
        raise SystemExit(problem)
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
    ap.add_argument("--shards", type=int, default=None,
                    help="score shard count; inferred from the shard* directories when "
                         "omitted, and validated as 0..n-1 rather than assumed to be 4")
    ap.add_argument("--beta", type=float, default=None,
                    help="optional cross-check; the beta actually used is read from the "
                         "solved target and a disagreeing value here is refused")
    ap.add_argument("--profile", default="paper-nbpo",
                    choices=("paper-nbpo", "paper-global", "coverage"),
                    help="which acceptance contract to apply; recorded in the decision")
    ap.add_argument("--surplus-tol", type=float, default=1e-8,
                    help="the manuscript's tolerance: a retained local surplus must "
                         "exceed it")
    ap.add_argument("--coverage-min", type=float, default=0.5,
                    help="only for --profile coverage, which is a relaxed diagnostic and "
                         "not the manuscript's predicate")
    ap.add_argument("--nmse-max", type=float, default=1.0,
                    help="maximum held-out nMSE, for the profiles that require a fit")
    ap.add_argument("--assume-beta", type=float, default=None,
                    help="beta to use when the target carries no complete.json; recorded "
                         "as an unverified assumption, never used as a silent default")
    ap.add_argument("--expect-beta", type=float, default=None,
                    help="refuse unless the solved target declares this beta. Without it "
                         "the beta is TAKEN FROM the target and an explicit --beta that "
                         "disagrees is refused rather than silently used.")
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

    # The game is whatever the target was SOLVED against, and beta is part of the
    # game. Taking it from a flag let a beta = .05 target be evaluated at the
    # default .25 and accepted at +.035 where its own beta gives -.146. The
    # solved artifact is now the authority and a disagreeing flag is refused.
    manifest = tdir / "complete.json"
    if manifest.exists():
        complete = json.loads(manifest.read_text())
        if "beta" not in complete:
            raise SystemExit(
                "%s records no beta; the game the candidate was fitted to is then "
                "unidentified and the surplus cannot be computed in it" % manifest)
        solved_beta = float(complete["beta"])
        beta_source = str(manifest)
    elif args.assume_beta is not None:
        # An explicit, recorded assumption for a target that carries no manifest.
        # Deliberately not a default: taking beta from a flag silently is what
        # let a beta = .05 target be judged at .25 and accepted where its own
        # beta rejects it, so the assumption has to be stated and it is written
        # into the decision as unverified.
        complete = {}
        solved_beta = float(args.assume_beta)
        beta_source = "--assume-beta, UNVERIFIED: the target carries no complete.json"
    else:
        raise SystemExit(
            "%s has no complete.json, so the beta this target was solved at is unknown. "
            "The gate will not fall back to a default, because evaluating the surplus in "
            "the wrong game can turn a rejection into an acceptance. Supply "
            "--assume-beta <value> to state the assumption explicitly; it is recorded as "
            "unverified in the decision." % tdir)
    if args.expect_beta is not None and abs(args.expect_beta - solved_beta) > 1e-12:
        raise SystemExit(
            "the target declares beta %.6g and --expect-beta says %.6g; refusing rather "
            "than evaluating a different game than the one that was solved"
            % (solved_beta, args.expect_beta))
    if args.beta is not None and abs(args.beta - solved_beta) > 1e-12:
        raise SystemExit(
            "--beta %.6g disagrees with the beta %.6g this target was solved at; the "
            "surplus would be computed in a game the candidate was never fitted to"
            % (args.beta, solved_beta))
    beta = solved_beta

    # and the target must still be the one the solve certified
    solved_hashes = {}
    for split in ("train", "dev"):
        sc = tdir / split / "complete.json"
        if sc.exists():
            rec = json.loads(sc.read_text())
            solved_hashes[split] = {k: rec.get(k) for k in
                                    ("solver_solution_sha256", "pairs_sha256", "n_prompts")}
    here = solved_hashes.get(args.split, {})
    if here.get("n_prompts") not in (None, len(pids)):
        raise SystemExit(
            "the %s solve recorded %s prompts and the per-prompt array holds %d; the "
            "target has been modified since it was certified"
            % (args.split, here.get("n_prompts"), len(pids)))

    candidates = {pid: ["%s:learner:%d" % (pid, i) for i in range(POOL)] for pid in pids}
    wanted = {c for v in candidates.values() for c in v}
    order = [c for pid in pids for c in candidates[pid]]
    if args.logprobs:
        rows = None            # no tokenization is needed when the log-probs are given
    else:
        load_pool_rows, sequence_logprobs = _pool_helpers()
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

    shards = args.shards
    if shards is None:
        found = sorted(d for d in Path(args.scores).glob("shard*") if d.is_dir())
        if not found:
            raise SystemExit("%s contains no shard* directories" % args.scores)
        indices = sorted(int(d.name[len("shard"):]) for d in found)
        if indices != list(range(len(indices))):
            raise SystemExit(
                "%s has shard directories %s, which are not 0..n-1; the count cannot be "
                "inferred and must be given with --shards" % (args.scores, indices))
        shards = len(indices)
    scores, score_manifests = solver.load_scores(args.scores, shards)
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
    beta_vec = torch.full((K,), beta, dtype=torch.float64)
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

    # the candidate has to be a real checkpoint for any of this to mean anything,
    # and a missing one is reported as a rejection rather than an exit so the
    # caller that reads decisions gets one
    candidate_ok = candidate_problem(Path(args.candidate))
    finite_ok = nonfinite == 0 and candidate_ok is None
    mins = np.array([r["min_surplus"] if r["min_surplus"] is not None else np.nan
                     for r in per_prompt], dtype=np.float64)
    above = np.isfinite(mins) & (mins > args.surplus_tol)
    coverage = float(above.mean())
    # the manuscript's NBPO predicate: EVERY retained local surplus over the
    # tolerance. A coverage fraction is the diagnostic for how badly it fails.
    all_local_positive = bool(finite_ok and above.all() and above.size > 0)
    failing = [r["prompt_id"] for r, ok in zip(per_prompt, above) if not ok]
    # the control's predicate: the prompt-averaged surplus per objective
    aggregate = np.nanmean(surplus, axis=1) if surplus.size else np.array([])
    aggregate_positive = bool(aggregate.size and np.all(np.isfinite(aggregate))
                              and np.all(aggregate > args.surplus_tol))

    nmse, fit_ok, fit_note = None, None, "not requested and not required"
    needs_fit = args.profile == "paper-global"
    if args.nmse_from or needs_fit:
        if not args.nmse_from:
            raise SystemExit("--profile %s requires --nmse-from" % args.profile)
        state = json.loads(Path(args.nmse_from).read_text())
        seen = [e for e in state.get("log_history", []) if "eval_nbpo/nmse" in e]
        if not seen:
            # An absent metric used to leave fit=None, which the old acceptance
            # rule read as "not False" and let through. A requested check that
            # cannot be evaluated is a failure.
            nmse, fit_ok = None, False
            fit_note = ("no eval_nbpo/nmse in the trainer state; a requested fit check "
                        "that cannot be evaluated is a rejection, not a pass")
        else:
            nmse = float(seen[-1]["eval_nbpo/nmse"])
            if not math.isfinite(nmse):
                fit_ok, fit_note = False, "recorded nMSE is not finite"
            elif nmse < 0.0:
                fit_ok = False
                fit_note = "recorded nMSE is negative, which this statistic cannot be"
            else:
                fit_ok = nmse <= args.nmse_max
                fit_note = "last recorded held-out nMSE, finite and in range"

    # A check the caller ASKED for is enforced whatever the profile says. The
    # manuscript attaches nMSE to the control and not to the NBPO family, so the
    # paper-nbpo predicate does not require it -- but silently ignoring a
    # supplied --nmse-from is the same defect as treating a missing one as a
    # pass: the caller reads an acceptance and believes the fit was checked.
    fit_requested = bool(args.nmse_from)

    if args.profile == "paper-nbpo":
        checks = {"finite": finite_ok, "all_local_surplus_above_tol": all_local_positive}
        accepted = bool(finite_ok and all_local_positive)
        predicate = ("every retained local surplus > %g on every objective "
                     "(Appendix H, NBPO family)" % args.surplus_tol)
        if fit_requested:
            checks["fit"] = fit_ok
            accepted = bool(accepted and fit_ok is True)
            predicate += (", and the caller's requested dev nMSE in [0, %g]"
                          % args.nmse_max)
    elif args.profile == "paper-global":
        checks = {"finite": finite_ok, "aggregate_surplus_above_tol": aggregate_positive,
                  "fit": fit_ok}
        accepted = bool(finite_ok and aggregate_positive and fit_ok is True)
        predicate = ("aggregate surplus > %g on every objective and finite dev nMSE in "
                     "[0, %g] (Appendix H, Global Nash control)"
                     % (args.surplus_tol, args.nmse_max))
    else:
        checks = {"finite": finite_ok, "coverage": coverage >= args.coverage_min}
        accepted = bool(finite_ok and checks["coverage"])
        if fit_requested:
            checks["fit"] = fit_ok
            accepted = bool(accepted and fit_ok is True)
        predicate = ("RELAXED DIAGNOSTIC, not the manuscript's contract: at least %g of "
                     "prompts positive on every objective" % args.coverage_min)

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
        "score_shards": shards,
        "score_manifest_schema": [m.get("tensor_role_schema") for m in score_manifests],
        "profile": args.profile,
        "predicate": predicate,
        "is_the_manuscript_contract": args.profile in ("paper-nbpo", "paper-global"),
        "thresholds": {"surplus_tol": args.surplus_tol, "coverage_min": args.coverage_min,
                       "nmse_max": args.nmse_max,
                       "beta": beta, "beta_source": beta_source},
        "solved_target": {"beta": beta, "hashes": solved_hashes},
        "measured": {"coverage": coverage,
                     "candidate_problem": candidate_ok,
                     "prompts_failing_the_local_predicate": len(failing),
                     "failing_prompt_ids": failing[:50],
                     "aggregate_surplus_per_objective": [float(x) for x in aggregate],
                     "nonfinite_prompts": nonfinite,
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
