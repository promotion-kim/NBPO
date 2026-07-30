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
