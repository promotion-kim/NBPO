# P5 Stage-1 method note

## Diagnosis before the new runs

The prior SafeRLHF table does not establish that a fixed-target RONPO adversary
improves worst-objective reward. Its OS weights are computed once from a base
response pool, so the adversary cannot react when a trained policy moves toward
one reward head. The previous pool also had only moderate within-prompt
reward/cost disagreement, while squared-loss target magnitudes differed by
orders of magnitude across methods. Those facts make a ranking hard to
attribute to robust adaptation rather than static-target scale or evaluation
noise.

Stage-2 therefore refreshes each arm's response pool from its own Stage-1
parent, preserves the base policy as the reference anchor, and uses that parent
as `history0`. A Stage-2 arm is only compared to other Stage-2 arms generated
under the same parent-refresh rule. It is not compared to a Stage-1 baseline.

## New Stage-1 robust surrogate

Alongside an exact top-mass replication, P5 trains a fixed soft-min
lower-bound target. For the vector of objective-wise expected pair advantages
`g in R^K`, the scalar target is

`r_tau(g) = -tau log(sum_k exp(-g_k/tau))`, with `tau = 0.1` and `K = 2`.

For every finite `g`, the log-sum-exp inequality gives

`min_k g_k - tau log(K) <= r_tau(g) <= min_k g_k`.

Thus it is a smooth lower approximation to the worst objective with a known
additive gap. The target builder records its empirical maximum gap and target
magnitude before training. This guarantee is algebraic and target-level. It
does **not** claim that finite-step optimization of a non-convex neural policy
globally optimizes the original max-min objective. That distinction is kept in
the final report.

## Fixed comparison rule

Both Stage-1 arms use the same frozen P4 data rows, 900 optimization steps,
effective batch 16, seed 42, learning rate, anchor, SFT term, decoding, and
reward scorers. They are evaluated together with the unchanged prior Stage-1
baseline generations on the same 49-prompt validation panel. This panel is
descriptive only and is not a fresh confirmation or a checkpoint-selection
split. Every new model must pass the corrected reward-blind stability gate
before it may be scored or uploaded.

`spent_sealed_split_touched=false`
