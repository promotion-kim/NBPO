# The zero-step mismatch: cause, fix, and the measurement that settles it

At `theta = theta_t` the quantity the trainer regresses,

    h_ab = [log pi(a) - log pi_t(a)] - [log pi(b) - log pi_t(b)]

is identically zero: the same weights on the same text. It was measured at RMS
0.30-0.61. This is where it came from.

## Cause

**The two sides of the difference were computed under different batch groupings,
and in bfloat16 a response's log-probability depends on who it is batched with.**
`log pi(a)` came from the online forward during training; `log pi_t(a)` came from
the `precompute` cache, whose batches were grouped differently. h was therefore a
difference between two computations that were never comparable.

Two distinct terms contribute, and both were measured separately.

### Term 1: the accumulated sum is stored in bfloat16

A sequence log-probability is a SUM over hundreds of response tokens and reaches
250-300 nats. bfloat16 carries an 8-bit mantissa, so its spacing is 1 at
|x| in [128, 256) and 2 at [256, 512). `get_batch_logps` ran `log_softmax`, the
gather and the reduction all in bf16, so the answer was quantized to that
spacing. The probe shows it directly -- long-response log-probabilities came back
as exact integers, and the batch-to-batch differences were exactly one ulp:

| \|logp\| | bf16 ulp | observed difference |
|---|---|---|
| 270 | 2 | 2.0000 |
| 248, 241 | 1 | 1.0000 |
| 11.5 | 0.0625 | 0.0625 |

Token counts were identical across conditions (0 rows differed), so this was
never a masking or truncation difference.

**Fix:** `log_softmax(-1, dtype=torch.float32)`, which computes and returns
float32 without materializing a float32 copy of the (batch, seq, vocab) logits,
so the gather and reduction run in float32 at no memory cost. Applied to all
three copies of `get_batch_logps` (`precompute.py`, `run_mnpo.py`,
`simpo_trainer.py`).

This does **not** remove the bf16 error in the forward that produced the logits.
It removes the quantization of the accumulation, and the measurement says how
much that was worth: the maximum batch-composition error fell from exactly
2.0000 to 1.1758.

### Term 2: the bf16 forward itself depends on batch shape

What remained is not precision of the reduction but of the forward. With the
float32 reduction in place, the same response still scored differently alone
versus batched (RMS 0.47) -- the same order as changing the entire weight
precision from bf16 to fp32 at fixed batching (RMS 0.33). PyTorch does not
guarantee bitwise-identical results across batch shapes, and at these sequence
lengths the difference is nats, not ulps.

This one cannot be fixed by precision at the reduction. It is fixed by never
comparing across batchings.

## A hypothesis that was tested and rejected

The collator left-pads prompts and the forward passes no `position_ids`, which
would shift every real token's RoPE position by the batch's padding amount. It
was a good candidate and it is **wrong**: supplying `position_ids` computed from
the attention mask changed the batch sensitivity by a factor of exactly 1.0000.
Recorded because a rejected hypothesis is part of the evidence.

## The fix contract

The proximal centre is now computed by a **frozen reference model forwarded
through the same collated batch** as the policy:

- a separate copy of `pi_t`, in eval mode with `requires_grad_(False)`;
- never the learner detached, and never re-synced to the learner;
- forwarded under `no_grad` on the identical batch, so both sides take the same
  kernel path with the same shapes.

At `pi = pi_t` the two forwards are then bitwise identical -- the probe shows the
computation is exactly reproducible for a fixed batch (repeat error 0.00000) --
so h is exactly 0, not approximately 0. No tolerance was chosen to accommodate
an observed error.

Enabled by `nbpo_online_reference: true` with `nbpo_reference_model_path`.
Default is off, so existing runs stay reproducible and the two paths remain
comparable; the trainer logs `nbpo/ref_online_minus_cache_rms` so the size of the
old discrepancy is visible in any run that uses the new path.

## The measurement that settles it

Zero-step run, learning rate 0, twenty steps, online reference:

| step | h abs | h RMS | h max | online reference − cache, RMS |
|---|---|---|---|---|
| 2 | 0.000000 | 0.000000 | 0.000000 | 0.302271 |
| 4 | 0.000000 | 0.000000 | 0.000000 | 0.501873 |
| 6 | 0.000000 | 0.000000 | 0.000000 | 0.613593 |
| 8 | 0.000000 | 0.000000 | 0.000000 | 0.110709 |
| 10 | 0.000000 | 0.000000 | 0.000000 | 0.354692 |
| 20 | 0.000000 | 0.000000 | 0.000000 | 0.338014 |

The right-hand column is the diagnostic, and it closes the argument: those values
reproduce the OLD path's h to six decimal places (0.302271, 0.501873, 0.613592,
0.110708, 0.354693, ... measured before any change). The entire zero-step error
was the online-versus-cache discrepancy, and nothing else.

## Reproducing it

```
python3 scripts/experiments/iclr2027_table1_v2/zero_step_probe.py \
    --model <pi_t> --precomputed <a scored precompute dir> --split test \
    --n-rows 24 --out zero_step_probe.json
```

Conditions: `repeat` (0 -- the computation is deterministic), `train_mode` (0 --
no stochastic layers), `alone` / `chunk8` / `chunk_all` (batch composition),
`fp32` (weight precision), `cache` (against what precompute stored).

## What this does and does not license

It establishes the cause of the zero-step failure and removes it. It does not by
itself say the gate will pass: the contaminated term was ~14% of the target RMS,
and the paired before/after training comparison is what measures the consequence.
It also does not certify the evaluation path -- the gate computes both of its
terms through the cache and so was internally consistent, but "internally
consistent" is not "correct", and a training signal that was wrong can still have
left the final policies where they are.
