# Codex task — Repair the collapsed averaged-oracle baselines and run a fair comparison (deadline 09:00 KST tomorrow, 4x B200)

You are in the `MNPO` repository on a machine with **exactly 4 authorized NVIDIA B200 GPUs**. The RONPO paper's central empirical claim is "averaging hides the failure": averaged-oracle preference optimization is supposed to be brittle under conflicting objectives. But in the paper's main 1.5B study the two averaged-oracle baselines, SPPO-avg and INPO-avg, did not merely underperform, they collapsed far below the untrained base model. That makes the comparison indefensible. Your job overnight is to retrain those two baselines properly (recovered, not collapsed) on the exact same data, pools, and budget as RONPO, then re-score everything fairly, so we learn whether RONPO still wins the worst objective against a healthy averaging baseline. Leave a clear, reviewable answer by **09:00 KST tomorrow**.

## Non-negotiable constraints

1. **4 B200 GPUs only.** Take three read-only `nvidia-smi` samples first; confirm the 4 target GPUs are idle and owned by no other user. Never touch, pause, or kill another user's process. Do not use more than 4 GPUs.
2. **No fabrication.** Every number comes from a measured run. If a config fails to train or fails the stability gate, record it as failed. Do not impute or hand-edit any score.
3. **Fail-closed stability gate.** A model that does not pass the generation stability gate is reported as failed, never silently kept.
4. **Fair comparison.** The retrained baselines must use the same base policy, training prompts, response pools, objective set, evaluation prompts, reward models, normalization, and bootstrap as the RONPO run they are compared against. The only things you may change relative to the collapsed runs are the stabilizing hyperparameters listed below.
5. **Not the sealed test.** This task touches no sealed split and makes no irreversible decision. Do not open any sealed evaluation.
6. **Deadline.** Launch nothing that cannot finish and be verified by 09:00 KST. Prefer a smaller sweep that completes over a large one that does not.

## Background (measured, do not redo)

- Main study backbone: `Qwen/Qwen2.5-1.5B-Instruct`, 19,856 UltraFeedback training prompts, 647 held-out prompts, three objectives (Skywork, Athene, ArmoRM), averaged oracle = prompt-wise mean of the per-prompt min-max-normalized objectives. Common config: bf16 AdamW, lr `5e-7`, cosine, warmup 0.1, one epoch, seed 42, effective batch 16, max length 2048, gradient checkpointing, DeepSpeed ZeRO-3.
- Collapsed baselines to repair (paper Table `tab:stage1-local-rm`): SPPO-avg normalized Avg `0.2049` / Worst `0.0838`; INPO-avg Avg `0.0673` / Worst `0.0274`; base Avg `0.7811` / Worst `0.7285`. Scores far below base indicate training collapse (length drift / repetition), the same failure mode seen in the Qwen3-8B first pass.
- The fix that worked at 8B: add light trust-region anchoring (reference-anchor and SFT-anchor terms) and reduce the step, which stopped the collapse. The unified trainer supports it: `mnpo_scripts/mnpo_config.py` exposes `reference_anchor_weight`, `preference_sft_weight`, `loss_type in {sppo, inpo, ...}`, `eta`; `mnpo_scripts/mnpo_trainer.py` implements the losses. SPPO/INPO need the `chosen_probs`/history columns from the annotated pool.
- RONPO's stage-1 result to compare against (same 647 prompts, same 3 RMs): Avg `0.8885` / Worst `0.8609`.

## Phase A — Locate and diagnose (first ~1 hour)

1. Find the exact data, precomputed pool, and training entry point that produced the current SPPO-avg / INPO-avg artifacts (search `mnpo_scripts/`, `scripts/`, `experiments/ronpo_aaai_*`, and any 1.5B run manifests). Identify the averaged-oracle precomputed dataset RONPO's stage-1 used, so the retrained baselines consume the identical pool.
2. Confirm the collapse hypothesis: decode a small sample from the current SPPO-avg / INPO-avg checkpoints (if available) and measure mean words, max repeat run, and think-tag leakage against base. Record the evidence.
3. If you cannot establish a runnable 1.5B SPPO-avg / INPO-avg recipe on the same pool within ~1 hour, STOP and write exactly what is missing (which script, which precomputed file). Do not substitute an unrelated experiment.

## Phase B — Retrain with a small anchored sweep (4-GPU parallel)

Retrain both SPPO-avg and INPO-avg on the identical pool/budget, changing only the stabilizers. Suggested grid (adjust to fit the deadline; keep it small):

- SPPO-avg: `eta in {0.005, 0.01}` x `reference_anchor_weight in {0.02, 0.05}`, `preference_sft_weight = 0.1 * reference_anchor_weight`, `learning_rate in {5e-7, 2.5e-7}` (pick 3-4 total configs, not the full cross-product).
- INPO-avg: same anchoring/lr grid, `eta in {0.005, 0.01}`.

Keep everything else at the documented common config (cosine, warmup 0.1, one epoch, effective batch 16, gradient checkpointing, ZeRO-3, SDPA, seed 42). Run 4 configs at a time across the 4 B200s. Log progress hourly with step counts and a running error scan (traceback / OOM / NaN).

**Stability gate (fail-closed, per candidate), same thresholds as the flagship protocol:** decode the 128-prompt stability set (or the 647 held-out if time allows), require 0 empty responses, 0 think-tag leakage, mean-word ratio vs base in `[0.33, 2.0]`, and max consecutive identical word run `<= 20`. Candidates that fail are excluded.

## Phase C — Fair re-evaluation

For every stability-passing candidate plus the base model and the paper's RONPO stage-1 checkpoint, generate one response per prompt on the same 647 held-out prompts (vLLM, seed 42, temperature 0.7, top-p 0.9, max 2048 new tokens, bf16), score all responses in one pass with Skywork, Athene, and ArmoRM, then compute per-prompt min-max-normalized scores over this exact model set, the per-prompt average (Avg) and worst objective (Worst), the win rate vs the base response, and 2000-resample paired bootstrap intervals. Select, per method, the single candidate with the best validation Worst that also passed the stability gate.

## Phase D — Report and regenerate

Write `results/baseline_repair_1p5b_<date>/`:
- `repair_summary.json`: for base, RONPO, and the best repaired SPPO-avg and INPO-avg, the Avg / Worst / WR with CIs, plus each candidate's stability-gate result and hyperparameters.
- `REPAIR_REPORT.md`: a human-readable table and a one-line honest verdict answering: did a recovered (above-base) averaged-oracle baseline still lose to RONPO on the worst objective? Report the true answer either way, including if RONPO's advantage shrinks or disappears once the baseline is healthy.
- W&B run ids and exact commands for each candidate.

## Acceptance criteria (put these in your closing message)

- At least one SPPO-avg and one INPO-avg candidate that pass the stability gate and score **above the base model** on Avg (i.e., recovered, not collapsed). If none recover, say so and report the closest attempt with its gate failure.
- The fair Avg/Worst comparison of base, RONPO, and the two repaired baselines on the 647 prompts, with CIs.
- The honest verdict sentence.
- Confirmation that only 4 B200 GPUs were used and no other user's process was touched.

If you cannot finish by 09:00 KST, stop cleanly, report which candidates completed and which did not, and leave all partial artifacts and logs in place.
