# GPU training skills

This bundle installs three GPU workflow skills and one read-only custom review agent:

- `gpu-scan`: find authorized, consistently idle GPUs without changing state.
- `train-launch`: prepare, tune, launch, and verify a named run.
- `train-watch`: monitor and repair named runs until a verified terminal state.
- `precise`: review changed code, configs, names, commands, and claims for concision, transparency, and correctness.

The durable style rules live in `AGENTS.md.example`; the `precise` agent is an additional review gate for task-relevant changes.

## Personal install

```bash
mkdir -p ~/.agents/skills ~/.codex/agents ~/.config/codex-cluster
cp -R .agents/skills/* ~/.agents/skills/
cp .codex/agents/precise.toml ~/.codex/agents/
cp -n config/hosts.yaml.example ~/.config/codex-cluster/hosts.yaml
# Copy only when absent; otherwise merge AGENTS.md.example manually.
test -e ~/.codex/AGENTS.md || cp AGENTS.md.example ~/.codex/AGENTS.md
chmod 600 ~/.config/codex-cluster/hosts.yaml
```

Edit `~/.config/codex-cluster/hosts.yaml` so every SSH alias, allowed GPU list, workspace, and storage root is accurate. Do not store passwords, OTPs, private keys, API keys, or tokens there; if the user explicitly provides a credential for one task, use it only ephemerally.

## Repository install

```bash
cp -R .agents <repo>/
mkdir -p <repo>/.codex/agents
cp .codex/agents/precise.toml <repo>/.codex/agents/
# Merge AGENTS.md.example into <repo>/AGENTS.md.
```

Restart Codex if the new skill or agent does not appear.

## Use

Read-only capacity scan:

```text
Use $gpu-scan to inspect SSH-configured hosts and rank authorized GPUs for SPPO stage 1 and INPO stage 1. Read only; do not launch anything.
```

Launch with a `precise` review gate:

```text
Use $train-launch to run RONPO stage 2 on odin2 GPUs 0,1 in tmux mnpo.
Reuse the canonical repo config, put large artifacts on approved external storage,
and have the precise agent review any changed file and the final launch command before the long run.
```

Monitor to completion:

```text
Use $train-watch to keep RONPO-s2, SPPO-s1, and INPO-s1 healthy until each has exit code 0 and its required final artifact. Resume from valid checkpoints, apply only evidence-based repairs, and have the precise agent review any code or config repair before relaunch.
```

Standalone review:

```text
Spawn the precise agent to review the changed launch scripts, configs, filenames, and commands. Require exact findings and a PASS or BLOCKED verdict.
```

Remote side-effect skills remain explicit-only. Select them with `$train-launch` or `$train-watch`.
