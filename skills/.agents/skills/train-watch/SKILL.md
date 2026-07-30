---
name: train-watch
description: Monitor named remote GPU training jobs, classify health with evidence, diagnose failures, and safely resume or relaunch authorized runs from checkpoints until explicit completion criteria are met. Use for "주기적으로 job 상태 체크", "이상 있으면 고쳐 다시 제출", "training 완료까지 반복", or equivalent English requests. Do not act on unrelated or unowned jobs.
---

# Purpose
Keep a defined set of training runs healthy until each reaches a verifiable terminal condition.

# Inputs
Build a run ledger from the request and repository with one row per run:
- Run name, algorithm, and stage.
- SSH host alias and expected resolved hostname/GPU model.
- Authorized GPU IDs.
- Tmux session or scheduler job ID.
- Work directory, command/config, log, output, and checkpoint path.
- Completion evidence, such as exit code 0 plus a final checkpoint, expected metric file, or explicit completion marker.
- Poll interval and repair/restart budget.

If a run's identity or ownership is ambiguous, observe it but do not modify it. Mark missing facts `unknown` rather than guessing.

# Hard rules
- Operate only on the named runs and resources owned by the current user.
- On code-server or shared hosts, never kill, stop, detach, rename, or reuse an existing process, tmux session, port, or job unless it is the exact named run under watch and the user explicitly requested repair that requires termination. If another process conflicts, mark the run `blocked` or relaunch on a non-conflicting resource.
- Verify the actual hostname and GPU model; do not trust a remembered server label.
- User-provided credentials may be used only as ephemeral authentication material for the current task. Never save, echo, log, commit, or copy plaintext passwords, TOTP secrets, API keys, tokens, or private keys into command lines, files, logs, tmux history, tracker configs, or reports. Prefer interactive `read -s`, an existing login session, an agent, or environment-managed auth.
- Never use broad kill/delete commands.
- Prefer resume from the newest validated checkpoint over restarting from scratch.
- Do not retry the same unexplained failure indefinitely. After two failed repairs for the same root cause, stop that run and report the blocker.
- Do not call a run complete merely because a tmux pane or process disappeared.
- Keep repair files and names concise but unambiguous; do not create `fix2`, `new`, `final`, or redundant wrappers.
- Record every repair, command, changed file, restart count, and validation result. Do not hide failed attempts.

# Health classification
Classify every run as one of:
- `queued`: valid scheduler job waiting normally.
- `starting`: launched but not yet past the bounded startup window.
- `healthy`: process exists; logs and steps advance; loss is finite; GPU use is plausible; storage is healthy.
- `stalled`: process exists but steps/logs do not advance over multiple checks, or GPU/CPU/I/O evidence indicates a hang.
- `failed`: nonzero exit, traceback, OOM, NCCL/distributed error, missing data/model, disk failure, NaN/Inf, or vanished process before completion.
- `complete`: expected exit status and required final artifacts/metrics are both present and valid.
- `blocked`: authentication, authorization, ownership, corruption risk, review failure, or repeated unknown failure prevents safe repair.

# Poll cycle
1. Resolve the host and verify it is the expected machine.
2. Inspect scheduler state, exact owned process/PID, tmux pane, exit-code file, and command line.
3. Check log modification time, current step/epoch, recent loss/metrics, and error signatures.
4. Sample GPU memory/utilization and process state multiple times; check CPU/RAM, disk bytes/inodes, and recent checkpoint writes.
5. Update the ledger with timestamps and evidence, then classify health.
6. Take no action for `queued`, `starting`, `healthy`, or `complete` except recording evidence.
7. Diagnose `stalled` or `failed`, apply the smallest safe repair, relaunch through `$train-launch`, and verify the first steps.

# Repair policy
- CUDA OOM: lower per-device microbatch one step; increase gradient accumulation to preserve effective batch when possible; show the calculation; keep a memory margin; resume from a validated checkpoint.
- NCCL/distributed failure: verify requested GPU mapping, world size, ranks, rendezvous port, stale owned processes, and repository-supported network settings. Change only the faulty setting.
- Disk full/inode exhaustion: move new large artifacts to an approved external root and remove only clearly disposable files owned by the user. Never delete checkpoints or datasets without explicit authorization.
- Missing model/data: correct a path or verified symlink to an approved storage root; verify expected files before relaunch.
- NaN/Inf or divergent loss: preserve evidence and checkpoint; do not blindly restart. Check precision, learning rate, data integrity, resume compatibility, and recent code/config changes. Mark blocked if correctness is uncertain.
- Stalled input pipeline: confirm log/step stagnation, then inspect storage latency, worker failures, RAM, file descriptors, and loader settings. Make one measured change at a time.
- Authentication/permission/unknown ownership: do not bypass. Mark blocked with the exact user action needed.

# Precise review gate
When a repair changes task-relevant code, script, config, filename, or launch command:
1. Explicitly spawn the `precise` custom agent for a read-only review.
2. Apply material findings and rerun relevant checks.
3. Relaunch only after a `PASS`; otherwise classify the run `blocked` and preserve the evidence.

# Stop conditions
Stop monitoring a run when it is `complete` or `blocked`. Stop the overall loop when all named runs are terminal.

Produce a final ledger:
`run | state | host | GPUs | step | evidence | repairs | restarts | final artifact | unknown/blocker`.

Separate verified facts from inference. A completion claim requires the defined exit status and final artifact, not just process absence.

# Scheduling note
This skill defines the monitoring and repair method. Use it in a durable Codex goal/thread or a scheduled automation when repeated checks are required.
