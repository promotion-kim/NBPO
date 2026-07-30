# Qwen2.5-7B seed-42 baseline repair preregistration

Locked at 2026-07-19 08:30 KST, before any repair training and without consulting reward scores.

## Diagnosed failure modes

- INPO Stage 1 completed with finite loss but emitted a one-token wink-emoji loop on all 1,000 gate prompts. The loss regressed length-normalized log-ratios to `1/(2 eta)=66.67`; its final loss was about 4,198, consistent with a target-scale mismatch.
- IPO Stage 1 completed with finite loss but emitted mixed-token and repeated-phrase degeneration. Its length-normalized log-ratio target was `1/(2 beta)=10`; its final loss was about 94, again consistent with the target scale rather than an optimization exception.
- SPPO passed Stages 1 and 2. At Stage 3 it passed record, empty, think-leak, and length checks, but one response had a 69-token repetition run. This is accumulated common-mode drift under a weak anchor, not a parser artifact.

## Reward-blind repair configurations

The completion log probabilities remain per-token averages. Therefore IPO beta and INPO eta are interpreted on that length-normalized coordinate. All other data, rows, model revision, seed, schedule, stage budget, and gate thresholds stay frozen.

| Candidate | Method/stage | eta or beta | LR | ref anchor | pref-SFT |
|---|---|---:|---:|---:|---:|
| inpo_norm_a | INPO/1 | eta 0.10 | 2.5e-7 | 0.20 | 0.02 |
| inpo_norm_b | INPO/1 | eta 0.20 | 2.5e-7 | 0.20 | 0.02 |
| ipo_norm_a | IPO/1 | beta 0.10 | 2.5e-7 | 0.20 | 0.02 |
| ipo_norm_b | IPO/1 | beta 0.20 | 2.5e-7 | 0.20 | 0.02 |
| sppo_anchor | SPPO/3 | eta 0.0075 | 2.5e-7 | 0.20 | 0.02 |

Each candidate receives 900 optimizer steps, effective batch 16, bf16 AdamW, cosine decay, warmup 0.1, seed 42, and mandatory online W&B logging. A 20-step finite-loss smoke test precedes each full run.

## Fail-closed selection

The unchanged corrected stability gate runs on all 1,000 locked evaluation prompts: exactly 1,000 records, zero empty responses, zero non-empty think spans, mean-word ratio to base in [0.33, 2.0], and maximum consecutive identical-word run at most 20.

- INPO: select `inpo_norm_a` if it passes. Use `inpo_norm_b` only if A fails.
- IPO: select `ipo_norm_a` if it passes. Use `ipo_norm_b` only if A fails.
- SPPO: `sppo_anchor` must pass.
- No reward score may be computed or consulted until these choices are final.

Passing candidates continue through the remaining stages with the same repaired hyperparameters and the frozen stage protocol. Original failed artifacts remain intact. Final checkpoint upload occurs only after the full gate passes and public Hugging Face integrity is verified.
