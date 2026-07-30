# P9b: P8 Stage-4 top-mass optimization-seed replication

Locked before P9b training starts: train exactly one auxiliary arm,
`ronpo_topmass_stage4_s43`, from the P7 Stage-3 top-mass seed-42 parent.  It
reuses the byte-identical P8 Stage-4 precomputed pool, 900 optimization steps,
effective batch 16, frozen optimizer, schedule, anchors, and final-step
checkpoint rule.  The only changed training random seed is 43.

This arm was added because host2 GPU 0 was measured idle while P8's only
remaining W1 arm and the separately preregistered P9 OS-versus-IPO replication
were still training.  The decision was made before any P8 reward evaluation
output existed.  It is a diagnostic estimator reproducibility arm, not a
selection candidate and not part of P8's primary comparison.  If it completes,
it will be evaluated separately on the locked P8 fresh 1,000-prompt panel only
after the P8 primary evaluation is complete.  It is an optimization-seed
replication, not an independent end-to-end seed because its response pool and
Stage-3 parent remain seed 42.

One smoke and one full attempt are allowed.  A genuine stability failure is
reported fail-closed.  No reward result may alter the choice of P8 arms,
hyperparameters, or the P8 primary analysis.
