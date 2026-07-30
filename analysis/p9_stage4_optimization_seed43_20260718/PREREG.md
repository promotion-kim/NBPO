# P9: P8 Stage-4 optimization-seed replication

Locked before P9 training starts: train exactly two arms, `ronpo_os_stage4_s43`
and `ipo_stage4_s43`, from the corresponding P7 Stage-3 seed-42 parent checkpoints.
Both reuse the byte-identical P8 Stage-4 precomputed pool, 900 optimization steps,
effective batch 16, frozen optimizer/schedule/anchors, and final-step checkpoint rule.
The only changed training random seed is 43. The arms will be evaluated together on
the same locked P8 fresh 1,000-prompt panel only after P8's primary evaluation has
finished; P9 does not change P8 selection or its primary analysis.

This is an optimization-seed replication, not an independent end-to-end seed because
the base response pool and Stage-3 parents remain seed 42. Every result will carry
that limitation. The motivation is review-critical variance evidence for the direct
RONPO-OS versus IPO comparison, selected from the P7 pre-registered comparison rather
than from P8 rewards. Each arm receives one smoke and one full attempt; genuine
stability failures are fail-closed.
