#!/usr/bin/env python3
"""Bind the controlled-v2 artifacts to the code and configuration that made them.

A results directory without this is a set of numbers nobody can re-derive. The
manifest records the source commit, the hash of every generator and solver file
in the path, the full construction configuration, the measured rho* calibration,
and the SHA-256 of every result file, so a later reader can tell in one step
whether a file has moved since it was frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

ARTIFACTS = [
    "controlled_v2_raw.json", "controlled_v2_per_seed.csv",
    "controlled_v2_summary.csv", "controlled_v2_runtime.csv",
    "controlled_v2_gaps.json", "controlled_v2_generated.tex",
    "controlled_v2_gap_figure.pdf", "controlled_v2_gap_figure.png",
    "controlled_v2_applicability.json", "controlled_v2_verification.json",
    "v2_margins.json",
]
GENERATORS = [
    "scripts/experiments/iclr2027_table1_v2/controlled_v2_final.py",
    "scripts/experiments/iclr2027_table1_v2/render_controlled_v2.py",
    "scripts/experiments/iclr2027_table1_v2/verify_controlled_v2.py",
    "scripts/experiments/iclr2027_table1_v2/nontransitivity_v2.py",
    "scripts/experiments/iclr2027_table1_v2/nontransitivity_feasibility.py",
]
SOLVERS = [
    "mnpo_scripts/nbpo_generic.py",
    "mnpo_scripts/nbpo_representations.py",
    "mnpo_scripts/nbpo_core.py",
]


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path,
                    default=Path("results/iclr2027_table1_v2/controlled_v2"))
    a = ap.parse_args()
    raw = json.loads((a.dir / "controlled_v2_raw.json").read_text())
    cfg = raw["config"]
    cells = {}
    for r in raw["rows"]:
        cells.setdefault((r["alpha"], r["method"]), []).append(r)
    alphas = sorted({x for x, _ in cells})

    rho = {}
    for alpha in alphas:
        rs = [r["rho_star"] for r in cells[(alpha, "nbpo_direct")]]
        rho[f"{alpha:.2f}"] = {
            "mean": statistics.fmean(rs), "stdev": statistics.stdev(rs),
            "min": min(rs), "max": max(rs), "per_seed": rs,
            "hit_amplitude_cap": [abs(x - 0.030) > 1e-4 for x in rs]}

    commit = subprocess.run(["git", "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"] + GENERATORS + SOLVERS,
                                capture_output=True, text=True).stdout.strip())

    manifest = {
        "artifact": "controlled_nontransitivity_v2_final",
        "frozen": True,
        "source_commit": commit,
        "source_tree_dirty_for_these_files": dirty,
        "generator_hashes": {p: sha(Path(p)) for p in GENERATORS},
        "solver_hashes": {p: sha(Path(p)) for p in SOLVERS},
        "seeds": [int(s) for s in eval(cfg["seeds"])] if isinstance(cfg["seeds"], str)
                 else cfg["seeds"],
        "alphas": alphas,
        "beta": float(cfg["beta"]),
        "eta": float(cfg["eta"]),
        "objectives_K": int(cfg["objectives"]),
        "prompts_X": int(cfg["prompts"]),
        "responses_I": int(cfg["responses"]),
        "construction_family": {
            "name": "shared-transitive-direction plus alpha-scaled circulant tournament",
            "formula": "A_k(alpha) = T + base_k + alpha * C_k",
            "clipping": "none; amplitudes chosen so |A| <= 1/2 holds outright",
            "disagreement_point": ("measured, and provably alpha-invariant: a "
                                   "circulant tournament on an odd number of "
                                   "responses has exactly zero row sums"),
            "skew_symmetry": "exact, asserted per instance",
            "zero_diagonal": "exact, asserted per instance",
            "base_amp": float(cfg["base_amp"]), "cycle_amp": float(cfg["cycle_amp"]),
        },
        "rho_star_calibration": {
            "target": float(cfg["target_rho_star"]),
            "achieved": rho,
            "statement": ("rho* is 0.030000 (sd <= 7.3e-08) for alpha <= 0.75 and "
                          "0.028334 (sd 2.4e-03, min 0.024746) at alpha = 1.00, where "
                          "the no-clipping amplitude cap binds for 2 of 5 seeds. It is "
                          "NOT exactly 0.03 everywhere. The reported metric is min "
                          "surplus / rho*, which absorbs the remaining difference."),
        },
        "solver_settings": {
            "inner_solver": "exact (direct finite-pool concave inner solve)",
            "dual_solver": "root", "dual_tol": float(cfg["dual_tol"]),
            "max_dual_calls": int(cfg["max_dual_calls"]),
            "legacy_arm": {"inner_solver": "fixed_point",
                           "dual_solver": "subgradient",
                           "M": int(cfg["rstep_dual_iterations"]),
                           "R": int(cfg["rstep_fixed_point_iterations"]),
                           "role": "failure ablation only, never a performance arm"},
        },
        "result_file_hashes": {n: sha(a.dir / n) for n in ARTIFACTS
                               if (a.dir / n).exists()},
        "missing_artifacts": [n for n in ARTIFACTS if not (a.dir / n).exists()],
    }
    (a.dir / "controlled_v2_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in ("result_file_hashes", "generator_hashes",
                                   "solver_hashes")}, indent=2))
    print("\nfile hashes:")
    for n, h in manifest["result_file_hashes"].items():
        print(f"  {h[:16]}  {n}")
    if manifest["missing_artifacts"]:
        raise SystemExit(f"missing artifacts: {manifest['missing_artifacts']}")


if __name__ == "__main__":
    main()
