# SPPO Stage-3 stability repair extension

Locked at 2026-07-19 11:05 KST before extension training and without consulting any reward score.

The first anchored retry reduced the full-panel maximum repetition run from 69 to 35 but did not meet the unchanged threshold of 20. All other checks passed. This extension changes only trust-region stabilization, using the same Stage-2 parent, frozen Stage-3 rows, seed, optimizer, and 900-step budget.

| Candidate | LR | reference anchor | preference SFT |
|---|---:|---:|---:|
| `sppo_strong_a` | 1.0e-7 | 0.50 | 0.05 |
| `sppo_strong_b` | 5.0e-8 | 0.75 | 0.075 |

Selection is reward-blind: select A if it passes the unchanged 1,000-prompt stability gate; use B only if A fails. If neither passes, SPPO remains failed. No gate threshold may change.
