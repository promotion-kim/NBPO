#!/usr/bin/env python3
"""Deterministic finite-pool NBPO dual solve (Algorithm 1's inner machinery).

Reads a versioned preference-tensor artifact (``build_preference_tensor.py``)
and runs projected dual gradient descent on the raw multipliers
(Eq. (27) ``eq:dual-update``) with the fixed-point weighted-policy solve of
Section 5.2 (Eq. (21), centered at the proximal center), or one of the matched
finite-game controls (utilitarian / absolute max-min / surplus max-min) on the
same tensors and budget.

Scope: every one of the ``M`` dual iterations (4e3--3e5 in the manuscript) is a
cheap tensor computation on the FROZEN finite response pool. The neural policy
is fit afterwards, once, from the targets built by ``build_nbpo_pairs.py`` --
no 8B model is retrained inside this loop.

Outputs (all raw, none normalized or clamped):
``solution.json`` -- raw lambda, V, d, surplus, the inverse-surplus residual
``||s - 1/lambda||_inf`` AND the projected (box-aware) KKT residual with the
active-bound coordinates, fixed-point and one-extra-map residuals, opponent
entropy/ESS of the final policy, the full config (beta, eta, gamma schedule,
M, R, lambda box), input artifact hashes, hashes of the opponent files, and the
iteration history; ``nu_update.npz`` -- the opponent that generated the final
policy (what Eq. (26) pair construction samples from); ``nu_final_policy.npz``
-- ``nu*`` recomputed at the final policy (diagnostics); ``pi_star.npz`` -- the
finite-pool policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mnpo_scripts.nbpo_core import (
    uniform_policy,
    validate_centered_preference_tensor,
    validate_reference_tensor,
)
from mnpo_scripts.nbpo_generic import (
    AGGREGATIONS as GENERIC_AGGREGATIONS,
    KSUndefinedError,
    ks_unregularized_diagnostics,
    matched_weight_l1,
    solve_finite_pool,
)
from mnpo_scripts.nbpo_representations import REPRESENTATIONS, build_representation
from mnpo_scripts.nbpo_solver import AGGREGATIONS, solve_nbpo_dual
from scripts.nbpo.nbpo_common import implementation_contract, sha256_file, write_json


def load_tensor_artifact(tensor_dir: Path):
    meta = json.loads((tensor_dir / "meta.json").read_text())
    A_policy = torch.from_numpy(np.load(tensor_dir / "tensor_policy.npz")["A"])
    A_ref = torch.from_numpy(np.load(tensor_dir / "tensor_ref.npz")["A"])
    hashes = {name: sha256_file(tensor_dir / name)
              for name in ("tensor_policy.npz", "tensor_ref.npz", "meta.json")}
    return meta, A_policy, A_ref, hashes


def parse_gamma(text: str, M: int):
    parts = [float(p) for p in text.split(",") if p.strip()]
    return parts[0] if len(parts) == 1 else parts


def _array_hash(t) -> str:
    """sha256 of an array's exact bytes -- identifies which policy an opponent came from."""
    import hashlib

    arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()


def write_solution_artifact(out_dir: Path, res, tensor_meta: dict, hashes: dict,
                            tensor_dir: Path, stage: int, lambda_warm_started: bool) -> dict:
    """Persist a DualSolveResult as the versioned solver artifact (shared with run_nbpo_stage)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Two opponents, written separately and hashed separately (never confuse them):
    #   nu_update.npz       -- generated the final policy; Eq. (26) targets sample z_k here
    #   nu_final_policy.npz -- nu* recomputed AT the final policy; diagnostics only
    np.savez_compressed(out_dir / "nu_update.npz", nu=res.nu_update.numpy())
    np.savez_compressed(out_dir / "nu_final_policy.npz", nu=res.nu_final_policy.numpy())
    np.savez_compressed(out_dir / "pi_star.npz", pi=res.pi.numpy())
    # The policy nu_update was actually built from, saved so the claim can be
    # checked rather than trusted. solve_nbpo_dual warm-starts the policy iterate
    # across dual iterations, so at R = 1 this is the warm-start iterate, not the
    # proximal centre -- which is what the artifact used to assert.
    if res.update_source_pi is None:
        raise ValueError("solver result carries no update_source_pi; the policy that "
                         "generated nu_update cannot be identified")
    np.savez_compressed(out_dir / "update_source_pi.npz", pi=res.update_source_pi.numpy())
    solution = {
        # What actually ran (audit section 0): the dual below is a frozen finite-pool
        # optimization; the neural policy is realized once, afterwards.
        **implementation_contract(dual_iterations=res.config.get("M"),
                                  fixed_point_steps=res.config.get("R")),
        "aggregation": res.aggregation,
        "stage": int(stage),
        "objectives": tensor_meta.get("objectives"),
        "lambda_raw": [float(v) for v in res.lam],
        "V": [float(v) for v in res.V],
        "d": [float(v) for v in res.d],
        "surplus": [float(v) for v in res.surplus],
        "min_surplus": float(res.surplus.min()),
        "kkt_residual": res.kkt_residual,
        "inverse_surplus_residual": res.kkt_residual,
        "projected_kkt_residual": res.projected_kkt_residual,
        "gamma_ref": res.gamma_ref,
        "lambda_at_lower_bound": res.lambda_at_lower_bound,
        "lambda_at_upper_bound": res.lambda_at_upper_bound,
        "kkt_note": ("lambda_k = 1/s_k is an empirical equality only for coordinates "
                     "strictly inside the box; when a bound is active read "
                     "projected_kkt_residual, not inverse_surplus_residual"),
        "control_residual": res.control_residual,
        "fixed_point_residual": res.fixed_point_residual,
        "extra_map_residual": res.extra_map_residual,
        "opponent_entropy": [float(v) for v in res.opponent_entropy],
        "opponent_ess": [float(v) for v in res.opponent_ess],
        "opponent_diagnostics_from": "nu_final_policy",
        "artifact_hashes": {
            "nu_update.npz": sha256_file(out_dir / "nu_update.npz"),
            "nu_final_policy.npz": sha256_file(out_dir / "nu_final_policy.npz"),
            "pi_star.npz": sha256_file(out_dir / "pi_star.npz"),
            "update_source_pi.npz": sha256_file(out_dir / "update_source_pi.npz"),
        },
        # Semantic roles, declared rather than inferred. The two opponents can be
        # numerically identical in a zero-payoff, symmetric or already-converged
        # game, so their difference is not an integrity signal -- their declared
        # role is.
        "opponent_artifacts": {
            "nu_update.npz": {
                "artifact_kind": "regularized_opponent",
                # Reported, not assumed: proximal_centre / warm_start_iterate /
                # fixed_point_iterate, whichever the solve actually used.
                "source_policy": res.update_source_kind,
                "source_policy_hash": _array_hash(res.update_source_pi),
                "source_policy_artifact": "update_source_pi.npz",
                "source_fixed_point_iteration": int(res.update_source_iteration),
                "used_for": "eq26_target",
            },
            "nu_final_policy.npz": {
                "artifact_kind": "regularized_opponent",
                "source_policy": "final_policy",
                "source_policy_hash": _array_hash(res.pi),
                "source_policy_artifact": "pi_star.npz",
                "source_fixed_point_iteration": int(res.config.get("R", 1)),
                "used_for": "diagnostics",
            },
        },
        "config": res.config,
        "lambda_warm_started": bool(lambda_warm_started),
        "input_hashes": hashes,
        "tensor_dir": str(tensor_dir),
        "history": res.history,
    }
    write_json(out_dir / "solution.json", solution)
    return solution


def build_objective_representation(args, A_policy, A_ref, mu, beta, construction):
    """Instantiate the representation the artifact will declare it used."""
    if args.representation == "adaptive_game":
        return build_representation("adaptive_game", A_policy=A_policy, A_ref=A_ref,
                                    mu=mu, beta=beta,
                                    reference_construction=construction)
    if args.representation == "fixed_reference":
        # beta is irrelevant here by construction (the comparator is mu, which is
        # the beta -> infinity limit), so it is deliberately not passed: a
        # fixed-reference artifact that recorded an opponent temperature would be
        # claiming a knob it does not have.
        return build_representation("fixed_reference", A_policy=A_policy, A_ref=A_ref,
                                    mu=mu, reference_construction=construction)
    if args.reward_table is None:
        raise SystemExit("--representation bt_reward requires --reward-table")
    tab = np.load(args.reward_table)
    for key in ("r_learner", "r_reference"):
        if key not in tab:
            raise SystemExit(f"{args.reward_table} has no array {key!r}")
    norm = {}
    if "normalization" in tab:
        try:
            norm = json.loads(str(tab["normalization"]))
        except Exception:
            norm = {"raw": str(tab["normalization"])}
    if not norm:
        raise SystemExit(
            f"{args.reward_table} carries no `normalization` record. The BT rows are only "
            "comparable once every head is on the frozen reference scale, so a reward table "
            "that cannot say which (mu_ref, sigma_ref) produced it is refused rather than "
            "assumed to be normalized.")
    return build_representation(
        "bt_reward", r_learner=torch.from_numpy(np.asarray(tab["r_learner"])),
        r_reference=torch.from_numpy(np.asarray(tab["r_reference"])), mu=mu,
        normalization=norm)


def write_generic_solution_artifact(out_dir: Path, res, tensor_meta: dict, hashes: dict,
                                    tensor_dir: Path, stage: int,
                                    lambda_warm_started: bool, extra: dict) -> dict:
    """Persist a FinitePoolSolution as the generic finite-pool target artifact.

    Carries everything a downstream consumer needs to rebuild the training target
    without re-solving: the proximal centre, the solved policy, the raw weights,
    the response-level objective scores, the target log-ratios, every residual,
    and the identity check that ties them together.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "nu_update.npz", nu=res.nu_update.numpy())
    np.savez_compressed(out_dir / "nu_final_policy.npz", nu=res.nu_final_policy.numpy())
    np.savez_compressed(out_dir / "pi_star.npz", pi=res.pi.numpy())
    np.savez_compressed(out_dir / "pi_t.npz", pi=res.pi_t.numpy())
    np.savez_compressed(out_dir / "q_update.npz", q=res.q_update.numpy())
    np.savez_compressed(out_dir / "target_log_ratio.npz", h=res.target_log_ratio.numpy())
    np.savez_compressed(out_dir / "update_source_pi.npz", pi=res.update_source_pi.numpy())
    identity = res.target_log_ratio_check()
    solution = {
        **implementation_contract(dual_iterations=res.config.get("M"),
                                  fixed_point_steps=res.config.get("R")),
        "solver_path": "generic_solve_finite_pool",
        "representation": res.representation,
        "aggregation": res.aggregation,
        "stage": int(stage),
        "objectives": tensor_meta.get("objectives"),
        "prompt_ids": tensor_meta.get("prompt_ids"),
        # RAW weights. For nash these are lambda; for the controls they are the
        # rule's weight vector at the matched L1 norm. Never normalized for training.
        "lambda_raw": [float(v) for v in res.weights],
        "aggregation_weights_raw": [float(v) for v in res.weights],
        "V": [float(v) for v in res.V],
        "d": [float(v) for v in res.d],
        "surplus": [float(v) for v in res.surplus],
        "min_surplus": float(res.surplus.min()),
        "kkt_residual": res.kkt_residual,
        "inverse_surplus_residual": res.kkt_residual,
        "projected_kkt_residual": res.projected_kkt_residual,
        "gamma_ref": res.gamma_ref,
        "lambda_at_lower_bound": res.lambda_at_lower_bound,
        "lambda_at_upper_bound": res.lambda_at_upper_bound,
        "control_residual": res.control_residual,
        "fixed_point_residual": res.fixed_point_residual,
        "extra_map_residual": res.extra_map_residual,
        "proximal_kl": res.proximal_kl,
        "opponent_entropy": [float(v) for v in res.opponent_entropy],
        "opponent_ess": [float(v) for v in res.opponent_ess],
        "opponent_diagnostics_from": "nu_final_policy",
        # The identity the pair builder relies on:
        #   [log pi*(y) - log pi_t(y)] - [log pi*(y') - log pi_t(y')]
        #     == eta * sum_k w_k (q_k(y) - q_k(y'))
        # Verified numerically here rather than assumed downstream.
        "target_log_ratio_identity_residual": identity,
        "target_log_ratio_identity_holds": bool(identity < 1e-9),
        "eta": res.eta,
        "eta_applications_in_solver": 1,
        "ks": res.ks,
        "artifact_hashes": {n: sha256_file(out_dir / n) for n in
                            ("nu_update.npz", "nu_final_policy.npz", "pi_star.npz",
                             "pi_t.npz", "q_update.npz", "target_log_ratio.npz",
                             "update_source_pi.npz")},
        "opponent_artifacts": {
            "nu_update.npz": {
                "artifact_kind": "regularized_opponent",
                "source_policy": res.update_source_kind,
                "source_policy_hash": _array_hash(res.update_source_pi),
                "source_policy_artifact": "update_source_pi.npz",
                "source_fixed_point_iteration": int(res.update_source_iteration),
                "used_for": "eq26_target",
            },
            "nu_final_policy.npz": {
                "artifact_kind": "regularized_opponent",
                "source_policy": "final_policy",
                "source_policy_hash": _array_hash(res.pi),
                "source_policy_artifact": "pi_star.npz",
                "used_for": "diagnostics",
            },
        },
        "config": res.config,
        "lambda_warm_started": bool(lambda_warm_started),
        "input_hashes": hashes,
        "tensor_dir": str(tensor_dir),
        "history": res.history,
        **extra,
    }
    if not solution["target_log_ratio_identity_holds"]:
        raise SystemExit(
            f"the solved policy does not reproduce its own weighted-q target "
            f"(residual {identity:.3e}). The Eq. (26) pair target would not be the "
            "target this solve actually optimized; refusing to write a usable artifact.")
    write_json(out_dir / "solution.json", solution)
    return solution


def _print_summary(solution: dict) -> None:
    keys = ("solver_path", "representation", "aggregation", "lambda_raw", "surplus",
            "min_surplus", "inverse_surplus_residual", "projected_kkt_residual",
            "lambda_at_lower_bound", "lambda_at_upper_bound", "control_residual",
            "fixed_point_residual", "extra_map_residual",
            "target_log_ratio_identity_residual", "proximal_kl")
    print(json.dumps({k: solution[k] for k in keys if k in solution}, indent=1))
    if solution.get("ks"):
        ks = solution["ks"]
        print(json.dumps({k: ks[k] for k in
                          ("u_ideal", "rho_star", "normalized_surplus",
                           "individual_rationality_violation", "ideal_point_ir_feasible",
                           "stage1_feasibility_residual", "stage2_pareto_gain",
                           "stage3_kl", "ideal_point_definitions")
                          if k in ks}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tensor-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--beta", required=True,
                    help="opponent temperatures: one value, or comma list per objective "
                         "(this is opponent_betas -- NOT the metrics-only trainer `beta`)")
    ap.add_argument("--eta", type=float, default=1.0, help="proximal coefficient eta_t")
    ap.add_argument("--gamma", required=True,
                    help="dual step size gamma_m: scalar or comma list of length M")
    ap.add_argument("--dual-iterations", "-M", type=int, required=True, dest="M")
    ap.add_argument("--fixed-point-iterations", "-R", type=int, default=1, dest="R",
                    help="R=1 is the manuscript's disclosed practical approximation; the "
                         "fixed-point residual is reported either way")
    ap.add_argument("--lambda-min", type=float, default=1e-3)
    ap.add_argument("--lambda-max", type=float, default=1e3)
    ap.add_argument("--warm-start-lambda", type=Path, default=None,
                    help="solution.json of the previous outer stage (lambda warm start); "
                         "omit at t=0 (lambda initialized to ones)")
    ap.add_argument("--aggregation", choices=list(GENERIC_AGGREGATIONS), default="nash")
    ap.add_argument("--representation", choices=list(REPRESENTATIONS),
                    default="adaptive_game",
                    help="how each objective is valued. adaptive_game is the manuscript's "
                         "finite-temperature game; fixed_reference freezes the comparator at "
                         "mu (the beta -> infinity limit, the matched control that isolates "
                         "the adaptive opponent); bt_reward reads a scalar reward table")
    ap.add_argument("--reward-table", type=Path, default=None,
                    help="npz with r_learner (K,X,I) and r_reference (K,X,J) ALREADY "
                         "normalized on training reference responses; required for "
                         "--representation bt_reward")
    ap.add_argument("--weight-l1", type=float, default=None,
                    help="common L1 weight norm for the matched controls. NBPO keeps its raw "
                         "lambda; pass ||lambda||_1 from the NBPO solve here so every control "
                         "takes the same effective proximal step and the rows differ only in "
                         "the DIRECTION of the weight vector")
    ap.add_argument("--match-weight-l1-to-nash", action="store_true",
                    help="solve the Nash dual first on these same tensors and use its raw "
                         "||lambda||_1 as --weight-l1 (recorded in the artifact)")
    ap.add_argument("--ks-stage1-iterations", type=int, default=2000)
    ap.add_argument("--ks-ideal-iterations", type=int, default=200)
    ap.add_argument("--ks-tau", type=float, default=1e-3)
    ap.add_argument("--ks-tolerance", type=float, default=1e-4)
    ap.add_argument("--ks-unregularized-diagnostics", action="store_true",
                    help="additionally report the spec-literal KS quantities (ideal point, "
                         "rho*, IR violation) at a nearly unregularized step")
    ap.add_argument("--legacy-solver", action="store_true",
                    help="route adaptive_game through the original solve_nbpo_dual code path "
                         "(bitwise identical; kept so a published artifact can be regenerated "
                         "by the exact code that produced it)")
    ap.add_argument("--damping", type=float, default=0.0)
    ap.add_argument("--adversary-step", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--tensor-kind", choices=["centered_preference", "game_utility"],
                    default="centered_preference",
                    help="centered_preference (raw P-1/2, range-checked to [-1/2,1/2]) or "
                         "game_utility (positively rescaled c_k A_k; finite only)")
    args = ap.parse_args()

    meta, A_policy, A_ref, hashes = load_tensor_artifact(args.tensor_dir)
    if args.tensor_kind == "centered_preference":
        validate_centered_preference_tensor(A_policy, "A_policy")
        validate_centered_preference_tensor(A_ref, "A_ref")
    # The reference construction is DECLARED by the artifact, never inferred; a
    # tensor that does not say how it was built is rejected rather than guessed.
    construction = meta.get("reference_construction")
    if construction is None:
        raise ValueError(
            f"{args.tensor_dir}/meta.json declares no reference_construction "
            "('shared_pool' = one response set on both sides, exact skew symmetry "
            "required; 'independent_samples' = two independent mu draws)"
        )
    validate_reference_tensor(A_ref, "A_ref", construction)
    K = A_policy.shape[0]
    beta_vals = [float(b) for b in args.beta.split(",") if b.strip()]
    beta = torch.tensor(beta_vals * K if len(beta_vals) == 1 else beta_vals, dtype=torch.float64)
    if beta.shape != (K,):
        raise ValueError(f"--beta must give 1 or {K} values, got {len(beta_vals)}")
    lambda_init = None
    if args.warm_start_lambda is not None:
        prev = json.loads(args.warm_start_lambda.read_text())
        lambda_init = torch.tensor(prev["lambda_raw"], dtype=torch.float64)

    mu = uniform_policy(A_policy.shape[1], A_policy.shape[3])
    gamma = parse_gamma(args.gamma, args.M)

    # The legacy path is kept verbatim so a published artifact can be regenerated
    # by the exact code that produced it. It only covers adaptive_game, and the
    # generic path is bitwise identical on that configuration (asserted in
    # tests/test_nbpo_generic_solver.py).
    legacy_ok = (args.representation == "adaptive_game"
                 and args.aggregation in AGGREGATIONS)
    if args.legacy_solver:
        if not legacy_ok:
            raise SystemExit(
                f"--legacy-solver cannot run representation={args.representation!r} "
                f"aggregation={args.aggregation!r}; that combination exists only in the "
                "generic path")
        res = solve_nbpo_dual(
            A_policy, A_ref, mu, beta,
            eta=args.eta, gamma=gamma, M=args.M, R=args.R,
            lambda_box=(args.lambda_min, args.lambda_max),
            lambda_init=lambda_init, aggregation=args.aggregation,
            reference_construction=construction,
            damping=args.damping, adversary_step=args.adversary_step,
            log_every=args.log_every,
        )
        solution = write_solution_artifact(
            args.out_dir, res, meta, hashes, args.tensor_dir, args.stage,
            lambda_warm_started=lambda_init is not None,
        )
        solution["solver_path"] = "legacy_solve_nbpo_dual"
        write_json(args.out_dir / "solution.json", solution)
        _print_summary(solution)
        return

    rep = build_objective_representation(args, A_policy, A_ref, mu, beta, construction)

    weight_l1 = args.weight_l1
    matched_note = None
    if args.match_weight_l1_to_nash:
        if weight_l1 is not None:
            raise SystemExit("pass either --weight-l1 or --match-weight-l1-to-nash, not both")
        nash = solve_finite_pool(rep, "nash", eta=args.eta, pi_t=None, R=args.R, M=args.M,
                                 gamma=gamma, lambda_box=(args.lambda_min, args.lambda_max),
                                 lambda_init=lambda_init, damping=args.damping)
        weight_l1 = matched_weight_l1(nash)
        matched_note = {"source": "nash solve on these same tensors",
                        "lambda_raw": [float(v) for v in nash.weights],
                        "weight_l1": weight_l1}
        print(f"[matched step] NBPO ||lambda||_1 = {weight_l1:.6g}; controls solved there")

    ks_kwargs = dict(stage1_iters=args.ks_stage1_iterations,
                     ideal_iters=args.ks_ideal_iterations,
                     tau=args.ks_tau, tolerance=args.ks_tolerance)
    try:
        res = solve_finite_pool(
            rep, args.aggregation, eta=args.eta, pi_t=None, R=args.R, M=args.M,
            gamma=gamma, lambda_box=(args.lambda_min, args.lambda_max),
            lambda_init=lambda_init, damping=args.damping,
            adversary_step=args.adversary_step, weight_l1=weight_l1,
            log_every=args.log_every, ks_kwargs=ks_kwargs)
    except KSUndefinedError as exc:
        # A bargaining set that does not dominate the disagreement point is a
        # property of the stage, not a solver failure: record it and stop, rather
        # than clamping a nonpositive surplus into a solution that never existed.
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.out_dir / "solution_blocked.json", {
            "status": "kalai_smorodinsky_undefined", "reason": str(exc),
            "diagnostics": exc.diagnostics, "representation": args.representation,
            "aggregation": args.aggregation, "tensor_dir": str(args.tensor_dir),
            "input_hashes": hashes})
        raise SystemExit(f"Game-KS is undefined on this pool; wrote "
                         f"{args.out_dir / 'solution_blocked.json'}. {exc}")

    extra = {}
    if matched_note:
        extra["matched_weight_l1"] = matched_note
    if args.ks_unregularized_diagnostics:
        extra["ks_unregularized_diagnostics"] = ks_unregularized_diagnostics(
            rep, res.pi_t, args.eta, args.R, **ks_kwargs)
    solution = write_generic_solution_artifact(
        args.out_dir, res, meta, hashes, args.tensor_dir, args.stage,
        lambda_warm_started=lambda_init is not None, extra=extra)
    _print_summary(solution)


if __name__ == "__main__":
    main()
