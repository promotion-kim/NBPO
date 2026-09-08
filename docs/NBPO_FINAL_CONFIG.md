# Configuring a final ICLR-2027 NBPO run

A final run is admitted by `mnpo_scripts/final_run_validator.py`, which executes
at the top of `scripts/nbpo/run_nbpo_stage.py` — **before rubrics, before pools,
before any model is loaded** — and raises rather than warns. The reason it is a
hard gate and not a lint: the legacy alternating solver converged on **0 of 25**
controlled-v2 cells and reached a normalized min surplus of **-3.37**, so a run
that silently reaches it produces numbers that look like results.

## The marker

```yaml
final_experiment: iclr2027
```

Without it the validator does not apply, and the config cannot acquire final
status by accident. With it, every rule below is enforced.

## The solver block

```yaml
solver:
  eta: 1.0
  opponent_betas: [0.25, 0.25]
  aggregation: nash              # nash | utilitarian | kalai_smorodinsky
  representation: adaptive_game  # adaptive_game | fixed_reference | bt_reward

  inner_solver: exact            # REQUIRED. direct finite-pool concave inner solve
  dual_solver: root              # REQUIRED. dual stationarity solved as a root problem
  dual_tol: 1.0e-6               # REQUIRED. projected-KKT stopping tolerance
  max_inner_residual: 1.0e-4     # declared inner stationarity requirement
  max_dual_calls: 800            # a CAP, not a schedule: stopping is by residual
  inner_workers: 96              # infrastructure only; output is bit-identical
  lambda_box: [1.0e-3, 1.0e3]

  provenance:
    record_solver_mode: true     # REQUIRED
    reject_legacy_artifacts: true
```

### What each key is for

`inner_solver: exact` — the Eq. (18) subproblem is concave and **separates across
prompts**, so it is solved directly instead of iterated. This names *one*
subproblem. It does not make the algorithm exact; the neural projection remains
approximate, and the validator's own report says so.

`dual_solver: root` — the Nash dual stationarity condition is `s_k(pi(lambda)) =
1/lambda_k`, a `K`-dimensional root problem once the inner subproblem is solved
rather than iterated. The projected subgradient it replaces stalls near a
residual of `1e-2` after 3000 outer iterations and cannot meet `dual_tol`.

`dual_tol` / `max_inner_residual` — a final config may be **stricter** than the
frozen values and may not be looser. Stopping is by residual; `max_dual_calls` is
only a cap, and hitting it makes the run fail rather than report.

`inner_workers` — pure infrastructure. The per-prompt programs are independent,
the output is bit-identical to serial, and the measured speedup is up to 37.4x
(`results/iclr2027_table1_v2/solver_scaling/`). Note that forked workers inherit
a restricted CPU affinity mask; the pool initializer restores it, and a test
fails if it ever stops doing so, because the symptom of that bug is silence.

`provenance.record_solver_mode` — every artifact records which solver produced
it, so a final run cannot build on a legacy target. The validator checks a
consumed `solution.json` for `inner_solver` and refuses both a legacy value and
an absent one.

## What is refused, and why

| refused | reason |
|---|---|
| `inner_solver` absent | the library default is `fixed_point` for backward compatibility and must never be inherited silently |
| `fixed_point`, `legacy`, `r_step`, `rstep`, `alternating` | the non-convergent map; ablation only |
| `dual_solver` not `root` | the subgradient cannot reach `dual_tol` |
| `dual_tol` > `1e-6`, `max_inner_residual` > `1e-4` | a final run may be stricter, never looser |
| `provenance.record_solver_mode` unset | a result whose solver mode cannot be recovered cannot be defended |
| a consumed artifact from the legacy path | a final run may not build on a legacy target |

## Measured cost

From `results/iclr2027_table1_v2/solver_scaling/direct_solver_scaling_report.md`
(all 24 cases converged; the 7000-prompt cases were run, not extrapolated):

| prompts | pool | workers | one full dual solve | peak RSS |
|---|---|---|---|---|
| 200 | 4+4 | 32 | 1.5 s | 467 MB |
| 1000 | 8+8 | 96 | 3.2 s | 504 MB |
| 7000 | 4+4 | 96 | **10.6 s** | 534 MB |
| 7000 | 16+16 | 96 | **15.3 s** | 633 MB |

The solver is not the budget constraint on a 7000-prompt stage; response
generation and the neural fit are.

## The reference config

`training_configs/nbpo/final_iclr2027/saferlhf_core.yaml`. Its three differences
from `training_configs/nbpo/saferlhf_llama.yaml` are the marker, the solver block
above, and a frozen SafeRLHF preference-model ensemble in place of the retired
prompted-Qwen judge.
