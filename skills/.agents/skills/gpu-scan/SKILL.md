---
name: gpu-scan
description: Read-only scan of SSH-configured GPU servers to find authorized idle GPUs, active jobs, storage, and candidate hosts. Use for requests such as "가용 GPU 확인", "ssh config 서버 확인", "H200/B200 비어 있나", "어디서 training을 돌릴 수 있나", or equivalent English requests. Never launch, stop, delete, or modify jobs.
---

# Purpose
Find safe candidate GPUs without changing remote state.

# Inputs
Extract these from the request and local configuration:
- Candidate host aliases, or all relevant aliases from `~/.ssh/config`.
- Required GPU count and, when known, minimum VRAM or GPU class.
- Host-specific allowed GPU IDs.
- Optional workspace and storage requirements.

Read `~/.config/codex-cluster/hosts.yaml` when it exists. Treat it as policy, not as a secret store.

# Hard rules
- Remain read-only. Do not launch, stop, cancel, delete, install, or edit anything.
- On code-server or shared hosts, existing processes are protected evidence. Never kill, stop, detach, rename, or clean them up while scanning.
- User-provided credentials may be used only as ephemeral authentication material for the current task. Never save, echo, log, commit, or copy passwords, TOTP codes, tokens, API keys, or private keys into prompts, files, command lines, tmux history, tracker configs, or reports.
- Prefer SSH aliases and existing key/agent or multiplexed authentication.
- Treat authorization as separate from availability. An idle but unauthorized GPU is not a candidate.
- If allowed GPU IDs are unknown, mark authorization `unknown`; do not recommend that host for launch.
- Do not declare a GPU idle from one `nvidia-smi` snapshot.
- State unknown values as `unknown`; do not fill gaps with plausible guesses.
- Keep labels concise but unambiguous, and identify aliases by their resolved hostname.

# Workflow
1. Enumerate relevant SSH aliases from the explicit request, `~/.ssh/config`, and the optional host policy file. Ignore wildcard-only aliases unless they resolve to a concrete requested host.
2. Test connectivity with a short timeout and no state changes. Record the resolved hostname so aliases cannot be confused.
3. On each reachable host, collect:
   - GPU index, model, total/used memory, utilization, temperature, and compute mode.
   - Compute processes, PIDs, owners, commands, and memory use when visible.
   - Scheduler allocations and the current user's jobs when Slurm or another scheduler is present.
   - Current user's relevant tmux sessions and matching training processes.
   - Free bytes and inodes for the workspace and approved storage roots.
4. Sample GPU memory/utilization and process state at least three times over a short interval. A GPU is a candidate only when all samples are consistent, no scheduler allocation conflicts, and no compute process is present.
5. Rank candidates by:
   - Authorization certainty.
   - Enough contiguous or requested GPU IDs.
   - GPU model/VRAM fit.
   - Low contention and stable idleness.
   - Writable external storage and sufficient free space.
6. Do not infer that two aliases are different machines until their resolved hostnames and GPU inventories differ.

# Output
Return a compact evidence table:
`host | resolved host | GPU | allowed | idle | owner/job | storage | auth/connectivity | status`.

Then provide:
- `verified`: facts confirmed by commands and repeated samples.
- `unknown`: missing authorization, connectivity, ownership, or scheduler facts.
- `recommendation`: one placement and the exact evidence supporting it.
- `blockers`: concise reasons a host cannot be used.

Do not launch anything or describe unverified availability as fact.
