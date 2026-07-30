# Qwen2.5-7B IPO Stage-4 seed-43 stability repair

Locked before training on 2026-07-20 KST.

The existing IPO Stage-4 policies for seeds 42 and 44 pass the unchanged 1,000-prompt stability gate. Seed 43 passes Stages 1--3 but its original Stage-4 policy collapses: mean-word ratio 0.1360 and maximum repeat run 497. This repair reuses the exact seed-43 Stage-3 parent and Stage-4 precomputed dataset. No pool, pair, objective, evaluator, seed, step budget, or reporting metric changes.

Four stability-only candidates are trained in parallel. Candidates A and B are the repository's prior IPO repair settings. Candidates C and D strengthen only the anchor and preference-SFT regularization while reducing the learning rate. The fixed selection order is A, B, C, D. Selection uses only the unchanged full 1,000-prompt stability gate and never reward.

All candidates use 900 steps, effective batch 16, cosine scheduling, warmup 0.1, seed 43, bf16, identical rows, and W&B online logging. The first candidate in the fixed order that passes all checks is selected. The checks are exactly 1,000 records, zero empty responses, zero non-empty paired think spans, mean-word ratio to base in [0.33, 2.0], and maximum consecutive identical-word run at most 20.

After selection, the existing seed-42 and seed-44 IPO generations are retained, the selected seed-43 generation replaces only the failed seed-43 artifact, and all three seeds are re-scored over one common eligible pool. Only then may Table 3 be updated.

