# Stage-3 launch fix log

## 2026-07-18 14:53 KST

The initial queue registration referenced `wait_for_continuation_pool.sh` and
`wait_for_continuation_train.sh` before those two newly added files had been
synchronized to the recreated B200 containers. `nohup` therefore exited before
any queued pool build or training process began. The already running RONPO-OS
pool build used the separately synchronized preparer and was unaffected.

The fix is limited to synchronizing the two missing queue scripts, recording
this event, and re-registering the same frozen queue. No data, target,
hyperparameter, model parent, GPU authorization, or outcome-dependent choice
was changed.

## 2026-07-18 15:04 KST

The first failed-registration recovery left one older top-mass pool waiter
alive while the corrected top-mass pool builder was already running. The old
waiter had not started a builder because GPU 2 was occupied, but it could have
started a duplicate write after the active builder released the GPU. It was
terminated before that point. The active builder, its parent, fixed pool,
recipe, and train queue were not changed.

## 2026-07-18 16:02 KST

RONPO-OS Stage-3 training completed normally with finite metrics and W&B run
`8d712bfa38b2`. Its first stability-decode attempt raced the already queued IPO
pool on GPU 0 after both waiters observed the same idle transition. The vLLM
log records negative available KV-cache memory and engine initialization
failure before any response or gate JSON was written. This is an
infrastructure failure, not a reward-blind model-gate outcome. The original log
is preserved.

The correction serializes each parent-arm gate behind the downstream arm gate
on the same GPU. OS is retried after IPO, top-mass after HT-MNPO harmless, and
INPO after HT-MNPO helpfulness, but only if the original arm still has no gate
JSON. A genuine failed gate JSON is terminal and is never retried. The same
ordering is applied prospectively to Stage 4. One duplicate, idle
HT-helpfulness pool waiter assigned to GPU 1 was removed before it acquired a
GPU; the intended GPU 3 queue remains. No running training process, model,
dataset, target, hyperparameter, decode setting, or gate threshold was changed.
