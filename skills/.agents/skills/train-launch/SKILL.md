---
name: train-launch
description: Safely prepare and launch an authorized GPU training run on an SSH host using requested GPU IDs and tmux, with concise unambiguous names, external-disk artifacts, transparent changes, and throughput-oriented tuning. Use for RONPO, SPPO, INPO, stage training, "training 돌려", "tmux에서 실행", or equivalent requests. Do not use for monitoring-only requests or unapproved hosts/GPUs.
---

# Purpose
Launch a reproducible training run with high stable throughput and no interference with other users.

# Required facts
Obtain from the request, repository, or host policy:
- Algorithm and stage.
- SSH host alias and authorized GPU IDs.
- Project/work directory.
- Tmux session name.
- Existing training config or entrypoint.
- Expected output/checkpoint location and completion signal.

Infer a missing fact only when an existing repository file, prior run, host policy, or explicit user-provided credential establishes it. Label each inference. Never invent credentials, permissions, training flags, or an algorithm implementation.

# Hard rules
- Use only the exact authorized GPU IDs.
- On code-server or shared hosts, never kill, stop, detach, rename, or reuse an existing process, tmux session, port, or job unless the user explicitly identifies the exact owned run and asks for that specific termination. If a conflict exists, choose a non-conflicting resource or report the blocker.
- User-provided credentials may be used only as ephemeral authentication material for the current task. Never save, echo, log, commit, or copy passwords, TOTP codes, API keys, tokens, or private keys into command lines, files, logs, tmux history, tracker configs, or reports. Prefer interactive `read -s`, an existing login session, an agent, or environment-managed auth over embedding secrets in launch commands.
- Never kill or delete an unowned or ambiguously identified resource.
- Never use broad process matching such as `pkill -f` or `killall`.
- Do not change drivers, CUDA, system packages, or shared system configuration.
- Optimize throughput and stability, not literal 100% VRAM occupancy.
- Preserve algorithm semantics, evaluation protocol, and effective global batch size unless explicitly changed.
- Keep names short only when they remain unique and understandable. Do not create `tmp`, `new`, `final`, `run2`, or duplicate `*_v2` artifacts without an explicit version boundary.
- Never hide unsupported options, fallbacks, failed checks, or remaining uncertainty.

# Workflow
## 1. Inspect before acting
1. Read applicable `AGENTS.md`, repository documentation, existing launch scripts/configs, and recent matching logs.
2. Resolve the SSH alias and verify the actual hostname and GPU model.
3. Verify that every requested GPU is authorized and consistently idle using the same checks as `$gpu-scan`.
4. Check the scheduler, matching processes, tmux sessions, ports, free disk bytes/inodes, and existing checkpoints.
5. Locate the canonical training entrypoint. Prefer an existing launcher/config over creating a duplicate or wrapper around a wrapper.
6. Check Git status and record the commit. Do not overwrite unrelated local changes.

## 2. Reconcile existing work
- Prevent duplicate launches by matching algorithm, stage, config, output path, command, and owner.
- If any existing code-server, tmux, training, notebook, or service process is present but not the exact run requested for termination, leave it running and route around it.
- When the user explicitly requests deletion or cancellation, identify the exact owned scheduler job or PID first, stop it gracefully, verify termination, and delete only the exact requested artifact.
- Never delete a directory merely because its name resembles a run name.
- Reuse a valid latest checkpoint unless the user explicitly requests a clean restart or compatibility checks fail.

## 3. Make the smallest clear change
- Keep code and small configs in the workspace.
- Modify the canonical file when safe; create a new file only when it represents a genuinely separate reusable entrypoint or config.
- Use the shortest unambiguous names. Prefer `ronpo-s2`, `sppo-s1`, and `inpo-s1` when they uniquely identify the run.
- Keep control flow readable. Do not trade validation, logging, error handling, or reproducibility for fewer lines.
- Record an exact diff summary: changed files, purpose, assumptions, and behavior changes.

## 4. Place large artifacts
- Select the first existing writable storage root allowed for that host.
- Put models, datasets, Hugging Face/Torch caches, checkpoints, tracker files, and large temporary files there.
- Set explicit cache/output environment variables or create verified symlinks. Check targets, bytes, and inodes before launch.
- Do not silently fall back to a system disk.

## 5. Tune safely
1. Use the repository's supported distributed launcher (`torchrun`, Accelerate, DeepSpeed, scheduler, or existing wrapper). Do not replace it without evidence.
2. Use supported BF16, TF32, fused kernels, Flash Attention, compilation, or optimizer variants only when the current code and environment support them and a smoke test validates them.
3. Tune microbatch size with bounded smoke tests. Increase it until throughput stops improving or memory becomes unsafe, then leave a reasonable margin for variable sequence lengths and allocator spikes.
4. If microbatch changes, adjust gradient accumulation to preserve the intended effective global batch when feasible. Show the batch calculation explicitly.
5. Tune data-loader workers, prefetching, pinned memory, and storage placement only after checking CPU/RAM/I/O limits.
6. For one multi-GPU run, verify world size and rank-to-GPU mapping. For independent one-GPU runs, use separate tmux sessions, output directories, ports, and tracker run IDs.

## 6. Run the precise review gate
When any task-relevant code, script, config, filename, or launch command changed:
1. Explicitly spawn the `precise` custom agent for a read-only review of the changed files and final command.
2. Require exact findings and a `PASS` or `BLOCKED` verdict.
3. Apply material findings in the main thread, rerun relevant checks, and repeat the review when behavior changed.
4. Do not start the long run while the review remains `BLOCKED`.

Skip this gate only when no file or command changed; state that explicitly.

## 7. Launch in tmux
- Reuse the requested tmux session only when it is absent or clearly belongs to the same run. Never overwrite an unrelated active session.
- Export `CUDA_VISIBLE_DEVICES` exactly as requested.
- Run through a login shell, capture stdout/stderr to a timestamped log, and persist the exit code separately.
- Ensure the command can resume from the selected checkpoint.

## 8. Verify health
Do not claim success immediately after process creation. Verify:
- The expected process/PID or scheduler job exists and is owned by the current user.
- The process sees only the intended GPU IDs.
- Logs advance and the first optimizer steps complete.
- Loss/metrics are finite and step/epoch counters progress.
- GPU memory and utilization are stable; CPU, RAM, disk, and data loading are not obvious bottlenecks.
- The checkpoint/output directory and tracker run are correct.

# Output
Return a concise launch manifest:
`run | host | resolved host | GPUs | tmux | PID/job | entry/config | commit | log | output | resume | tracker | review | health`.

Then list:
- `changes`: exact files and behavior changed.
- `checks`: commands/evidence used to verify correctness and health.
- `tuning`: measured change and reason.
- `unknown`: any remaining uncertainty.

Never describe a run as launched, healthy, resumed, or complete without the corresponding evidence.
