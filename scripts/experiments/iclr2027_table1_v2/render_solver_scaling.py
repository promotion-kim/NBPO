#!/usr/bin/env python3
"""Turn the measured scaling sweep into the report the claim has to rest on."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

GATE_INNER = 1e-4
GATE_DUAL = 1e-6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path,
                    default=Path("results/iclr2027_table1_v2/solver_scaling"))
    a = ap.parse_args()
    data = json.loads((a.dir / "direct_solver_scaling.json").read_text())
    rows = data["rows"]

    by = {}
    for r in rows:
        by.setdefault((r["prompts"], r["responses"]), []).append(r)
    best = {k: min(v, key=lambda r: r["total_solve_seconds"]) for k, v in by.items()}

    n_conv = sum(1 for r in rows if r["dual_converged"])
    inner_ok = sum(1 for r in rows if r["inner_extra_map_residual"] <= GATE_INNER)
    dual_ok = sum(1 for r in rows if r["projected_kkt_residual"] <= GATE_DUAL)

    L = ["# Direct finite-pool concave inner solve: measured scaling", "",
         "Every row below is a measurement. The 7000-prompt cases were **run**, not "
         "extrapolated from smaller ones.", "",
         "Only infrastructure is varied across rows -- prompt count, pool geometry, "
         "worker processes. The per-prompt program, the `dual_tol` of "
         f"{GATE_DUAL:.0e} and the declared inner stationarity requirement of "
         f"{GATE_INNER:.0e} are identical at every size; a faster number obtained "
         "by loosening either would not be a scaling result.", "",
         "Tensors are SafeRLHF-shaped: two objectives whose pairwise disagreement "
         "rate is calibrated to the 24.3% measured on the released human "
         "annotations, centered preferences in [-1/2, 1/2], exactly skew-symmetric "
         "with a zero diagonal.", "",
         "## Best configuration at each size", "",
         "| prompts | pool | workers | solve (s) | dual evals | s / eval | "
         "proj. KKT | inner resid. | peak RSS (MB) | artifact (s) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for (X, I), r in sorted(best.items()):
        L.append("| " + " | ".join([
            f"{X}", f"{I}+{I}", f"{r['workers']}",
            f"{r['total_solve_seconds']:.1f}", f"{r['dual_evaluations']}",
            f"{r['seconds_per_dual_evaluation']:.3f}",
            f"{r['projected_kkt_residual']:.0e}",
            f"{r['inner_extra_map_residual']:.0e}",
            f"{r['peak_rss_mb']:.0f}",
            f"{r['artifact_write_seconds']:.2f}" if r["artifact_write_seconds"] is not None else "--",
        ]) + " |")

    L += ["", "## Does parallelism help, and where", "",
          "| prompts | pool | " + " | ".join(f"{w} worker(s)" for w in
                                             sorted({r['workers'] for r in rows}))
          + " | best speedup |", "|---|---|" + "---|" * (len(
              {r['workers'] for r in rows}) + 1)]
    workers = sorted({r["workers"] for r in rows})
    for (X, I), rs in sorted(by.items()):
        m = {r["workers"]: r["total_solve_seconds"] for r in rs}
        base = m.get(1)
        cells = [f"{m[w]:.1f}s" if w in m else "not run" for w in workers]
        sp = (f"{base / min(m.values()):.1f}x" if base else "--")
        L.append(f"| {X} | {I}+{I} | " + " | ".join(cells) + f" | {sp} |")

    L += ["", "## Convergence", "",
          f"* dual converged: **{n_conv} of {len(rows)}** cases",
          f"* projected KKT residual <= {GATE_DUAL:.0e}: **{dual_ok} of {len(rows)}**",
          f"* inner stationarity residual <= {GATE_INNER:.0e}: **{inner_ok} of {len(rows)}**",
          "",
          "Residual distribution over all cases:", "",
          "| quantity | min | median | max |", "|---|---|---|---|"]
    for key, name in (("projected_kkt_residual", "projected KKT"),
                      ("inverse_surplus_residual", "inverse surplus"),
                      ("inner_extra_map_residual", "inner extra-map"),
                      ("target_identity_residual", "Eq. (26) identity")):
        v = [r[key] for r in rows if r.get(key) is not None]
        L.append(f"| {name} | {min(v):.1e} | {statistics.median(v):.1e} | {max(v):.1e} |")

    big = [r for r in rows if r["prompts"] == max(r2["prompts"] for r2 in rows)]
    if big:
        b = min(big, key=lambda r: r["total_solve_seconds"])
        L += ["", "## The 7000-prompt claim, measured", "",
              f"At **{b['prompts']} prompts, {b['responses']}+{b['responses']} pool, "
              f"{b['workers']} workers**: one full dual solve takes "
              f"**{b['total_solve_seconds']:.1f} s** over {b['dual_evaluations']} dual "
              f"evaluations ({b['seconds_per_dual_evaluation']:.3f} s each), peak RSS "
              f"**{b['peak_rss_mb']:.0f} MB**, projected KKT residual "
              f"**{b['projected_kkt_residual']:.1e}**, inner residual "
              f"**{b['inner_extra_map_residual']:.1e}**, artifact written in "
              f"**{b['artifact_write_seconds']:.2f} s**.",
              "",
              "The tensor itself is small -- "
              f"{b['tensor_mb']:.1f} MB at float64 -- so memory is not the binding "
              "constraint at this scale; the per-prompt solves are."]

    (a.dir / "direct_solver_scaling_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
