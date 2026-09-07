# External baselines: PROSPER / MaxEntBW, MOPO, RACO

Audit of what exists in this repository against what Table-1 rows 8, 9 and 10
require. No baseline was regenerated here — the judgment bank they must be built
from is itself blocked on the v3 protocol gate — so this records what each one
needs when that clears, and which of them cannot be run at all.

## Summary

| row | status | what is missing |
|---|---|---|
| 8. PROSPER / MaxEntBW | **partial** — synthetic finite-game only | a policy-training path on real data |
| 9. MOPO | **implemented** — trainer loss exists | the `rho` precompute on the v3 bank, and the alpha-threshold selection |
| 10. RACO | **BLOCKED** | the algorithm itself is absent, and the wrapper caps at K=3 |

## PROSPER / MaxEntBW — partial

The only implementation is
`analysis/game_nbpo_iclr_acceptance_20260826/run_prosper_mopo_synthetic.py`,
which solves the maximum-entropy Blackwell-winner problem *exactly* on
**synthetic** games (K=3, 5 actions, 200 sampled games). It is a correctness
and separation check, not a training path: nothing in it decodes, judges, or
fits a policy, and it imports `scipy.optimize`, which is not installed in any
environment on this machine.

So PROSPER on Table 1 has to be a **matched-data finite-pool realization**:
solve the MaxEntBW program on the same frozen pool and judgment bank every other
row uses, and fit the resulting finite-pool policy through the identical
realization path. The brief permits exactly this and requires it be labelled as
such — it is *not* a native full-budget reproduction of the paper's method, and
the results table must say so.

`scipy` will need to be installed, or the program re-expressed in the torch
solver already used for the other aggregations.

## MOPO — implemented, needs regeneration

`mnpo_scripts/mnpo_trainer.py` implements `loss_type: mopo` against the paper's
equations: the policy step is importance-weighted behaviour cloning on the
lagged reference (Agnihotri et al. Eq. (7)), with

    rho(y) = exp( tau^-1 [ p(y > y') + lambda^T q(y > y') ] - 1 )      (Eq. (3))

computed offline from the judged pool and carried per row. The trainer refuses
to run without that column rather than defaulting it, which is the right
behaviour.

What is missing for Table 1:

1. the `rho` precompute against the **v3** bank (the existing one is v2-labelled
   and must not be reused);
2. the threshold selection the brief specifies — every objective treated as
   primary in turn, relative attainable-surplus thresholds
   `alpha in {0.25, 0.50, 0.75}`, one primary/threshold combination chosen on
   **validation**, frozen before any final seed or test evaluation.

## RACO — blocked, and this is not a judgement call

The brief permits RACO only with an authoritative implementation, and forbids
inventing one. Three findings, each independently disqualifying:

1. **The algorithm is not present.** `third_party/raco_patches/train_raco.py` is
   a thin wrapper that constructs TRL's `DPOTrainer` and passes `raco=True`,
   `raco_use_cagrad=True` and `raco_weights`. The RACO/CGrad logic is expected
   to live inside a patched TRL — and there is none: `grep -rn raco
   third_party/trl` returns **0** matches, and the installed TRL in `mnpo_train`
   returns 0 as well. The flag reaches a trainer that does not implement it.
2. **The wrapper cannot express K=4.** `--raco_num_objectives` is declared
   `choices=[2, 3]` and `--raco_weights` is validated as two or three floats.
   The protocol requires K=4 explicitly. Raising that cap is not a
   configuration change; it is an extension of somebody else's method that would
   need its own verification.
3. **The recorded source is a reference, not a vendored copy.**
   `results/game_nbpo_iclr_campaign_20260824/RACO_PREFLIGHT.md` pins the official
   repository at commit `84a943c34f38520c7e0c9dd3066517c111b3c8fa`, which is
   exactly the right provenance to have — but the code at that commit is not in
   this tree.

`references/chen_raco_icml_2026.pdf` is present, so a specification-driven
implementation is conceivable. It is not attempted here: reconstructing CGrad
gradient surgery from a PDF and then extending it from K=3 to K=4 would produce
an unverified approximation competing against carefully matched rows, which is
the specific failure the brief rules out.

**To unblock**: vendor the official repository at the pinned commit, confirm the
CGrad implementation is reachable from the wrapper, and establish from the paper
whether the method is defined for K=4 — separately reviewed before it is run.
Until then row 10 stays empty and is reported as blocked rather than estimated.

## Note on the earlier RACO deferral

`RACO_PREFLIGHT.md` records that four RACO training jobs were "stopped during
initialization, before an observed optimizer step". Given finding (1) above,
that is consistent with the jobs never having had a working algorithm to run —
worth knowing before anyone treats the earlier deferral as purely a
prioritisation decision.
