---
name: hf-prune-models
description: Upload only paper-critical training checkpoints and evaluation-required models from the user's own external-storage namespace, such as /ext_hdd/sjkim, to the user's Hugging Face public model repos, verify remote integrity and minimal reloadability, then delete only redundant local model copies to prevent ext_hdd capacity issues. Use when asked to clean /ext_hdd model storage, preserve checkpoints needed for RONPO/HT-MNPO/SPPO/INPO evaluation or paper tables, upload models to Hugging Face before pruning, or decide which checkpoints can be removed after evaluation. Never read, scan, modify, or delete external-drive folders outside the user's sjkim namespace.
---

# Purpose

Keep external storage from filling while preserving every model needed to reproduce paper evaluations.

# Inputs

Build a retention ledger with one row per candidate model/checkpoint:
- Method, stage, reward oracle, base model, and training dataset.
- Local absolute path, size, modified time, and checkpoint step.
- Whether training is active, complete, failed, superseded, or only an intermediate resume point.
- Evidence that the model is used in an existing evaluation table, report, script, config, or paper comparison.
- Target Hugging Face repo ID, visibility, branch/revision, and remote path.
- Upload status, verification status, and local deletion decision.

Mark missing facts `unknown`. Do not infer that a model is disposable from its name alone.

# Hard Rules

- Never delete a model directory until its required artifacts are uploaded to the intended public Hugging Face repo and verified from the remote revision.
- Treat external drives as no-access except explicit user-owned namespace roots. For this project, `/ext_hdd/sjkim` is allowed; sibling folders such as `/ext_hdd/<other-user>` are forbidden to read, scan, modify, or delete.
- Never enumerate external-drive parent directories or sibling namespaces. Do not run commands such as `ls /ext_hdd`, `find /ext_hdd`, `du -sh /ext_hdd/*`, or globs that expand outside `/ext_hdd/sjkim`.
- Resolve every local candidate with `realpath` before upload or deletion. If the resolved path is not under an approved `sjkim` namespace root, stop and report it as blocked.
- Never delete active training outputs, the newest valid resume checkpoint for an incomplete run, current eval inputs, datasets, logs, W&B metadata, scripts, configs, or token/cache directories unless explicitly requested and separately verified.
- User-provided Hugging Face tokens may be used only as ephemeral authentication material for the current task. Never save, echo, log, commit, or copy tokens into commands, files, logs, model cards, tmux history, tracker configs, or final reports. Prefer an existing `huggingface-cli login` session, interactive token input, or environment-managed auth.
- Never use broad deletion commands. Use exact absolute paths from the ledger and delete only after checking ownership, path prefix, size, and that no process has the path open.
- Do not rely on one listing. Verify candidates with `du`, `find`, `lsof` or process command lines when available, and references from eval reports/configs.
- Keep public repos clean: upload final or paper-critical checkpoints, not every intermediate checkpoint.
- Preserve reproducibility metadata before deletion: training config, eval config, commit hash, source local path, metric summary, date, and base model.
- If a checkpoint is ambiguous, keep it and report why.

# Workflow

## 1. Discover Evidence

1. Read current evaluation reports, comparison tables, launch scripts, training configs, and recent logs.
2. Locate model directories only under approved user namespace roots such as `/ext_hdd/sjkim`; never inspect external-drive siblings or parent listings.
3. Identify active processes and tmux jobs before considering deletion.
4. Group checkpoints by method and run, then identify the final or best paper-relevant checkpoint.

## 2. Decide What Must Be Kept

Keep a local or remote copy of:
- Base models used as explicit baselines only when they are locally fine-tuned or modified; otherwise record the upstream HF ID.
- HT-MNPO, RONPO, SPPO, and INPO stage models that appear in evaluation results or planned paper tables.
- Reward-model variants or checkpoints required to reproduce evaluation rows.
- The newest valid checkpoint for any active or incomplete training run.
- Any model that has better metrics than the final checkpoint, if the paper table uses best checkpoint selection.

Delete candidates only when they are duplicate, superseded, failed, smoke-test-only, or old intermediate checkpoints and a verified remote copy or better local successor exists.

## 3. Upload to Hugging Face

1. Use concise public repo names that still encode method, base, and stage.
2. Add or update a model card with:
   - Method/stage/reward oracle.
   - Base model and important training config.
   - Evaluation config and metric table reference.
   - Source checkpoint path and Git commit.
   - Known limitations and intended research use.
3. Upload required files: model weights, config, tokenizer, generation config, adapter files if applicable, and small reproducibility metadata.
4. Use one revision per uploaded model state when possible; record the commit hash.

## 4. Verify Before Pruning

A remote model is verified only after all applicable checks pass:
- The target repo is public or has the user-requested visibility.
- Remote file listing contains required model, config, tokenizer, and metadata files.
- Remote revision/commit is recorded.
- A fresh download to a temporary cache succeeds for a small or full reload check.
- `AutoConfig.from_pretrained` and tokenizer load works; run `AutoModelForCausalLM.from_pretrained` when time/VRAM/storage allow.
- Local and remote key file sizes or hashes match for weight index/config/tokenizer files where practical.

If full model reload is too expensive, say so and keep the local copy unless the remote upload can be verified by another strong check.

## 5. Prune Local Copies

Before deletion:
1. Confirm the resolved path is under an approved `sjkim` namespace root such as `/ext_hdd/sjkim` and owned by the user.
2. Confirm no active process, tmux command, eval script, symlink, or config references it.
3. Confirm it is not the newest valid resume checkpoint for an active or incomplete run.
4. Print the exact bytes to be freed and the exact path list.
5. Delete exact paths only; then re-run `du` and `df` to verify freed space.

Prefer deleting old `checkpoint-*` directories inside a run after preserving the selected final/best checkpoint. Do not delete the parent run directory if it contains logs, configs, metrics, or artifacts that have not been archived.

# Output

Return a concise ledger:
`method | stage | local path | size | paper/eval evidence | HF repo@revision | verified | action | freed | reason`.

Also report:
- `kept`: models retained locally and why.
- `uploaded`: repos, revisions, and verification evidence.
- `deleted`: exact paths removed and space freed.
- `blocked`: candidates not removed because verification, ownership, active-run, or evidence was insufficient.

Separate verified facts from inference. A remote URL alone is not enough evidence to delete a local model.
