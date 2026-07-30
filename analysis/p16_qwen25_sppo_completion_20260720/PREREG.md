# Qwen2.5-7B SPPO Stage-4 completion

Locked on 2026-07-20 before any Stage-4 repair reward was scored. The repair uses only the unchanged 1,000-prompt, reward-blind stability gate.

- INPO-avg already has valid Stage-4 gates for seeds 42, 43, and 44 and is not retrained.
- SPPO Stage-3 candidate `sppo_strong_a` is the fixed seed-42 parent because it is the first preregistered candidate that passes the unchanged gate. Its reward was not consulted.
- Stage-4 candidates are tried in the fixed order `sppo_strong_a`, `sppo_strong_b`, then `sppo_strong_c`. The first configuration passing the unchanged gate for all three seeds is reported. No reward score is used for this selection.
- Every candidate uses the existing seed-specific Stage-3 parent and Stage-4 response pool, 900 steps, effective batch 16, cosine scheduling, warmup 0.1, and online W&B logging.
- After one common SPPO configuration passes all seeds, all methods are rescored over a new common eligible pool. Per-prompt min-max normalization is recomputed; no old normalized number is mixed with the new pool.

Gate thresholds: exactly 1,000 responses; zero empty responses; zero non-empty paired think spans; mean-word ratio to base in [0.33, 2.0]; maximum consecutive identical-word run at most 20.

