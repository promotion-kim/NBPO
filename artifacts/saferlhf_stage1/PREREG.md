# SafeRLHF helpful-versus-harmless Table-4 experiment: preregistration

## Scope

This is a fresh two-objective trade-off experiment. The policy is
`meta-llama/Llama-3.1-8B-Instruct` at revision
`0e9e39f249a16976918f6564b8830bc894c89659`. Helpfulness is the direct scalar
output of `PKU-Alignment/beaver-7b-v1.0-reward` at revision
`375cd6a9f0d7e339d2199b05ba129a4a8906596d`. Harmlessness is negative cost from
`PKU-Alignment/beaver-7b-v1.0-cost` at revision
`c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28`. These are distinct released
checkpoints.

Training uses the pinned `PKU-Alignment/PKU-SafeRLHF` revision
`9421ffafec3fa40a1f1a7d567b4d525079477ecb`, restricted to train rows with
`better_response_id != safer_response_id`. The deterministic pool size is
`min(2500, available deduplicated conflict rows)`. Validation uses every
available conflict row from the pinned test split after normalized-prompt
deduplication against the selected train set and the P1/P2 held-out manifests.
The resulting count is reported; it is never silently padded or reduced.

The P1 604-prompt sealed split is never read, decoded, or scored. There is no
Hugging Face upload, paid judge, or paper edit in this run.

## Shared data and decode protocol

Every trainable arm receives byte-identical prompt rows, four base-policy
responses per prompt (seeds 42, 43, 44, 45), response scores, pairs, reference
log-probabilities, and history log-probabilities. Decode is non-thinking,
bf16, temperature 0.7, top-p 0.9, and `max_new_tokens=512`. Three SHA-256
ordered pairs per prompt are oriented by the mean of the two per-prompt
min-max-normalized objective scores.

All arms use seed 42, 900 steps, effective batch 16 (microbatch 1 and
accumulation 16), bf16, cosine schedule, warmup 0.1, maximum sequence length
2048, maximum prompt length 1024, gradient checkpointing, reference anchor
0.05, preference-SFT anchor 0.005, learning rate `5e-7`, `ronpo_alpha=1.0`,
`ronpo_tau=0.05`, and `eta=0.0075`. The fixed checkpoint rule is the final
step. No per-arm retry or outcome-selected checkpoint is allowed.

## Kappa and arms

Before any model training, real-precomputed-pool normalized sigma entropy
selects three distinct kappas nearest targets 0.15, 0.55, and 0.85 from the
candidate list `[0.001, 0.002, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1,
0.15, 0.2, 0.3, 0.5]`, breaking ties to lower kappa. The OS arm at target
entropy 0.55 is the sole confirmatory RONPO arm. The other OS rows are
diagnostic only. Top-mass is included once because its target is exactly
kappa-invariant. `target_uniform` is the exact no-adversary endpoint.

W1, in fixed order, is: RONPO OS at entropy 0.55, INPO-avg, SPPO-avg, SimPO,
IPO, DPO, HT-MNPO harmlessness, and HT-MNPO helpfulness. Base is decoded but
not trained. W2, only if the clock plan permits, is: RONPO top-mass, RONPO
uniform, diagnostic OS at entropy 0.15, diagnostic OS at entropy 0.85, MNPO,
RONPO OS at learning rate `1e-6`, INPO at learning rate `1e-6`, and the best
non-RONPO W1 arm at `1e-6`. The last selection is outcome-dependent solely to
attack a W1 RONPO result; it is reported separately as a fairness check.
W3 is lower priority and not needed for the primary table.

The actual container exposes four authorized B200 GPUs, not eight. Work is
therefore scheduled in four-GPU waves and this constraint is reported rather
than misrepresented.

## Fail-closed gates and metrics

Before training, Beaver cost must select the human safer response on conflict
rows at least 65% of the time; Beaver reward must select the human better
response at least 60% of the time; the heads must be distinct; and the decoded
pool must have median prompt-level cross-response Spearman at most 0.5 and a
nonzero reward-argmax versus cost-argmax mismatch rate. Kappa target
non-degeneracy and target scale are audited before training.

After training, HT-MNPO helpfulness must exceed Base on helpfulness and
HT-MNPO harmlessness must exceed Base on harmlessness. All completed models
must pass the full validation reward-blind stability gate. Failure makes the
comparison pipeline failed rather than a reason to alter a threshold.

The primary metric is `mean_prompt_worst_norm_score`, the mean across
validation prompts of the minimum of the two per-prompt min-max-normalized
objective scores. Secondary metrics are normalized Avg, each objective,
win-rate versus Base, and minimum objective win-rate versus Base. The primary
gate is a paired 2,000-resample bootstrap (seed 42) for confirmatory OS minus
the best eligible trained non-RONPO arm: its 95% lower bound must be above
zero. The average floor requires the lower 95% bound of OS minus the best
baseline Avg to exceed -0.02. Worst-pass/Avg-fail is reported as a trade-off.

## Deadline and cuts

The hard stop is 21:00 KST on 2026-07-17. After a 20-step smoke, the measured
step rate and four-GPU wave plan are written to `CUT.md` before W1 starts. If
time is short, arms are cut only from the end of W2, then all W3 work. No W1
arm is cut or selectively retuned.
