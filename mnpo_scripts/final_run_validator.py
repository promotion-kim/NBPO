"""Refuse a final ICLR-2027 run before it loads a model.

An expensive run that silently inherits the legacy solver is worse than one that
never starts: it produces numbers that look like results. The controlled-v2 audit
measured what the legacy path does once raw Nash multipliers grow -- fixed-point
residual 1.000, min surplus -3.37 times rho*, 0 of 25 cells converged -- so a
final configuration that reaches it is not a configuration, it is a defect.

This validator therefore runs at the very top of the stage, before rubrics,
before pools, before any model is touched, and it raises rather than warns.

Terminology, used exactly. ``inner_solver: exact`` selects a **direct
finite-pool concave inner solve**: the Eq. (18) subproblem is concave and
separates across prompts, so it is solved directly to a declared numerical
tolerance instead of being iterated. That is a statement about one subproblem.
The neural NBPO algorithm as a whole is **not** exact -- it still projects a
finite-pool target into a neural policy by regression -- and nothing here should
be read as claiming otherwise.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

# The frozen final tolerances. A config may be stricter; it may not be looser.
FROZEN = {
    "inner_solver": "exact",
    "dual_solver": "root",
    "dual_tol": 1e-6,             # projected-KKT stopping tolerance, at most this
    "max_inner_residual": 1e-4,   # declared inner stationarity requirement
}
REJECTED_INNER_SOLVERS = {"fixed_point", "legacy", "r_step", "rstep", "alternating"}
MARKER = "final_experiment"
EXPECTED_MARKER = "iclr2027"


class FinalConfigError(ValueError):
    """A final-experiment configuration that must not be allowed to run."""


def is_final_config(cfg: Mapping[str, Any]) -> bool:
    return str(cfg.get(MARKER, "")).strip().lower() == EXPECTED_MARKER


def validate_final_config(cfg: Mapping[str, Any], *, source: str = "<config>",
                          solver_artifact: Optional[Mapping[str, Any]] = None) -> dict:
    """Raise ``FinalConfigError`` unless this config may produce a final number.

    ``solver_artifact`` is an already-written ``solution.json``; when a stage
    warm-starts from, or otherwise consumes, an earlier solve, that artifact is
    checked too -- a final run may not build on a target produced by the legacy
    R-step path.
    """
    if not is_final_config(cfg):
        raise FinalConfigError(
            f"{source}: not marked as a final experiment. A final ICLR-2027 run "
            f"must set `{MARKER}: {EXPECTED_MARKER}` so that this validator "
            "applies to it; an unmarked config cannot silently inherit final "
            "status.")

    scfg = cfg.get("solver")
    if not isinstance(scfg, Mapping):
        raise FinalConfigError(f"{source}: no `solver` block")

    inner = scfg.get("inner_solver")
    if inner is None:
        raise FinalConfigError(
            f"{source}: `solver.inner_solver` is absent. Every final config must "
            "state it explicitly -- the library default exists for backward "
            "compatibility and must never be inherited silently by a final run.")
    inner = str(inner).strip().lower()
    if inner in REJECTED_INNER_SOLVERS:
        raise FinalConfigError(
            f"{source}: `solver.inner_solver: {inner}` is the legacy alternating "
            "map. It is non-convergent once raw Nash multipliers grow (0 of 25 "
            "cells converged in controlled-v2) and may appear only as a failure "
            "ablation, never as a final experiment.")
    if inner != FROZEN["inner_solver"]:
        raise FinalConfigError(
            f"{source}: `solver.inner_solver: {inner}` is not recognised; the "
            f"final path is `{FROZEN['inner_solver']}` (direct finite-pool "
            "concave inner solve).")

    dual = str(scfg.get("dual_solver", "")).strip().lower()
    if dual != FROZEN["dual_solver"]:
        raise FinalConfigError(
            f"{source}: `solver.dual_solver: {dual or '<absent>'}`; the final path "
            f"is `{FROZEN['dual_solver']}`. The projected subgradient stalls near "
            "a residual of 1e-2 and cannot meet the 1e-6 stopping tolerance.")

    tol = scfg.get("dual_tol")
    if tol is None:
        raise FinalConfigError(f"{source}: `solver.dual_tol` is absent")
    if float(tol) > FROZEN["dual_tol"]:
        raise FinalConfigError(
            f"{source}: `solver.dual_tol` = {float(tol):.1e} is looser than the "
            f"frozen final tolerance {FROZEN['dual_tol']:.1e}. A final run may be "
            "stricter, never looser.")

    inner_tol = scfg.get("max_inner_residual", FROZEN["max_inner_residual"])
    if float(inner_tol) > FROZEN["max_inner_residual"]:
        raise FinalConfigError(
            f"{source}: `solver.max_inner_residual` = {float(inner_tol):.1e} is "
            f"looser than the frozen {FROZEN['max_inner_residual']:.1e}.")

    prov = scfg.get("provenance")
    if not isinstance(prov, Mapping) or not prov.get("record_solver_mode"):
        raise FinalConfigError(
            f"{source}: `solver.provenance.record_solver_mode` must be set, so "
            "that every artifact this run writes says which solver produced it. "
            "A result whose solver mode cannot be recovered cannot be defended.")

    if solver_artifact is not None:
        art_mode = (((solver_artifact.get("config") or {}).get("inner_solver"))
                    or solver_artifact.get("inner_solver"))
        if art_mode is None:
            raise FinalConfigError(
                f"{source}: the solver artifact it consumes records no "
                "`inner_solver`, so it cannot be shown to be free of the legacy "
                "path. Re-solve it under the final configuration.")
        if str(art_mode).strip().lower() != FROZEN["inner_solver"]:
            raise FinalConfigError(
                f"{source}: the solver artifact it consumes was produced by "
                f"`inner_solver: {art_mode}`. A final run may not build on a "
                "target from the legacy R-step path.")

    return {
        "validated": True,
        "final_experiment": EXPECTED_MARKER,
        "inner_solver": inner,
        "inner_solver_meaning": "direct finite-pool concave inner solve",
        "algorithm_is_not_exact": (
            "only the finite-pool inner subproblem is solved directly, to a "
            "declared tolerance; the neural projection remains approximate"),
        "dual_solver": dual,
        "dual_tol": float(tol),
        "max_inner_residual": float(inner_tol),
        "solver_artifact_checked": solver_artifact is not None,
    }
