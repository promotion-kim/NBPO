# Instrument report

## Pre-training gates

- Beaver reward human-better agreement on PKU conflict rows: 0.616379 (429/696); preregistered threshold 0.60.
- Negative Beaver cost human-safer agreement on PKU conflict rows: 0.692529 (482/696); preregistered threshold 0.65.
- Shared decoded-pool trade-off gate: median within-prompt Spearman 0.333333, mean 0.185323, reward/cost argmax mismatch 0.595200. The preregistered median-correlation ceiling was +0.5; this gate passed.

## Adversary and target audit

- Top-mass target was invariant across the built kappas: differing fraction 0.000000.
- Target endpoint assertions passed: {'full_exp_corr_topmass_nonincreasing': True, 'full_exp_corr_uniform_nondecreasing': True, 'interior_target_families_nonidentical': True, 'topmass_identical_all_kappas': True}. The full-expectation correlations move from hard-argmax toward the uniform control as kappa increases; see `target_nondegeneracy_audit.json`.
- The static target builder computes sigma once from the precomputed pool. This is not the adaptive OMD adversary in the toy analysis; no claim equates the two mechanisms.

## Target-scale audit

The measured effective target magnitudes and initial gradient norms are retained in `target_scale_audit.json` and the W1 job-status JSONs. They are reported because equal rows and step counts do not imply equal numerical target scales.

## Post-training single-oracle sanity

- HT-MNPO(help.) helpfulness 0.409224 versus Base 0.174756: PASS.
- HT-MNPO(harmless) harmlessness 0.451731 versus Base 0.435932: PASS.
- Overall single-oracle sanity: PASS.

All W1 decoded models passed the unchanged reward-blind stability gate; exact per-model checks are in `validation/gates/*.json`.
