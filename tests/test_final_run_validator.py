"""A final ICLR-2027 run must not be able to start on the legacy solver.

The controlled-v2 audit measured what the legacy alternating path does once raw
Nash multipliers grow: fixed-point residual 1.000, min surplus -3.37 times rho*,
0 of 25 cells converged. A configuration that can reach it is not a
configuration, so these tests pin that every final config passes the validator
and every non-final or legacy config fails it -- before a model is loaded.
"""
from pathlib import Path

import pytest
import yaml

from mnpo_scripts.final_run_validator import (
    FROZEN, FinalConfigError, is_final_config, validate_final_config,
)

FINAL_DIR = Path("training_configs/nbpo/final_iclr2027")
LEGACY_CONFIGS = sorted(Path("training_configs/nbpo").glob("*.yaml"))


def _final(**over):
    cfg = {
        "final_experiment": "iclr2027",
        "solver": {
            "eta": 1.0, "aggregation": "nash",
            "inner_solver": "exact", "dual_solver": "root",
            "dual_tol": 1.0e-6, "max_inner_residual": 1.0e-4,
            "provenance": {"record_solver_mode": True},
        },
    }
    cfg["solver"].update(over.pop("solver", {}))
    cfg.update(over)
    return cfg


def test_every_final_config_on_disk_passes():
    files = sorted(FINAL_DIR.glob("*.yaml"))
    assert files, f"no final configs found in {FINAL_DIR}"
    for f in files:
        cfg = yaml.safe_load(f.read_text())
        report = validate_final_config(cfg, source=str(f))
        assert report["inner_solver"] == "exact"
        assert report["dual_solver"] == "root"
        assert report["dual_tol"] <= FROZEN["dual_tol"]


def test_every_legacy_config_on_disk_fails():
    assert LEGACY_CONFIGS, "no legacy configs found"
    for f in LEGACY_CONFIGS:
        cfg = yaml.safe_load(f.read_text())
        assert not is_final_config(cfg), f"{f} is unexpectedly marked final"
        with pytest.raises(FinalConfigError):
            validate_final_config(cfg, source=str(f))


def test_missing_inner_solver_is_refused():
    cfg = _final()
    cfg["solver"].pop("inner_solver")
    with pytest.raises(FinalConfigError, match="inner_solver` is absent"):
        validate_final_config(cfg)


@pytest.mark.parametrize("bad", sorted({"fixed_point", "legacy", "r_step", "rstep",
                                        "alternating"}))
def test_legacy_inner_solvers_are_refused_by_name(bad):
    with pytest.raises(FinalConfigError, match="legacy alternating map"):
        validate_final_config(_final(solver={"inner_solver": bad}))


def test_a_looser_dual_tolerance_is_refused():
    with pytest.raises(FinalConfigError, match="looser than the frozen"):
        validate_final_config(_final(solver={"dual_tol": 1.0e-4}))
    # stricter is fine
    validate_final_config(_final(solver={"dual_tol": 1.0e-9}))


def test_a_looser_inner_residual_is_refused():
    with pytest.raises(FinalConfigError, match="max_inner_residual"):
        validate_final_config(_final(solver={"max_inner_residual": 1.0e-2}))


def test_the_subgradient_dual_is_refused():
    with pytest.raises(FinalConfigError, match="projected subgradient stalls"):
        validate_final_config(_final(solver={"dual_solver": "subgradient"}))


def test_absent_provenance_is_refused():
    cfg = _final()
    cfg["solver"].pop("provenance")
    with pytest.raises(FinalConfigError, match="record_solver_mode"):
        validate_final_config(cfg)


def test_a_legacy_solver_artifact_is_refused():
    with pytest.raises(FinalConfigError, match="legacy R-step path"):
        validate_final_config(_final(), solver_artifact={
            "config": {"inner_solver": "fixed_point"}})


def test_an_artifact_with_no_recorded_solver_mode_is_refused():
    with pytest.raises(FinalConfigError, match="records no"):
        validate_final_config(_final(), solver_artifact={"config": {}})


def test_a_matching_artifact_is_accepted():
    report = validate_final_config(_final(), solver_artifact={
        "config": {"inner_solver": "exact"}})
    assert report["solver_artifact_checked"] is True


def test_the_validator_never_calls_the_whole_algorithm_exact():
    """Terminology is part of the contract, not decoration."""
    report = validate_final_config(_final())
    assert report["inner_solver_meaning"] == "direct finite-pool concave inner solve"
    assert "approximate" in report["algorithm_is_not_exact"]


def test_the_pair_builder_admits_the_direct_solver_source_and_no_other():
    """The Eq. (26) provenance guard must know the direct solver's source kind.

    `exact_proximal_maximizer` is where nu comes from under the direct inner
    solve: the maximizer of the Eq. (18) subproblem, written to
    update_source_pi.npz and hash-verified like any other source. It was not in
    the whitelist, so a final-path solve produced an artifact the pair builder
    refused. Admitting it must not turn the guard off -- an unrecognised source
    still has to fail.
    """
    import re
    src = Path("scripts/nbpo/build_nbpo_pairs.py").read_text()
    m = re.search(r"valid_sources = \{([^}]*)\}", src, re.S)
    assert m, "the source whitelist has moved"
    allowed = {t.strip().strip('"\'') for t in m.group(1).split(",") if t.strip()}
    assert "exact_proximal_maximizer" in allowed
    assert "proximal_centre" in allowed and "fixed_point_iterate" in allowed
    # the guard is still a whitelist, not a pass-through
    assert "anything" not in allowed and len(allowed) <= 6
