# Qwen2.5-7B seed-42 baseline repair status

Last measured: 2026-07-19 08:38 KST.

Five previously idle, authorized B200 GPUs are occupied by the preregistered repair jobs. No other user's process was touched.

| Host/GPU | Candidate | Status | W&B |
|---|---|---|---|
| aiprlab-ronpo/1 | INPO eta 0.10 | 20-step smoke passed; 900-step full running | `6555327f53a9` |
| aiprlab-ronpo/2 | INPO eta 0.20 | 20-step smoke passed; 900-step full running | `48855506ef03` |
| aiprlab-ronpo/3 | IPO beta 0.10 | 20-step smoke passed; 900-step full running | `6e0d0d632baa` |
| aiprlab-ronpo2/0 | IPO beta 0.20 | 20-step smoke passed; 900-step full running | `a13944975fdb` |
| aiprlab-ronpo2/1 | SPPO Stage-3 anchored retry | 20-step smoke passed; 900-step full running | `5a70b7f317a9` |

Measured training rate is about 3.9 to 4.3 seconds per optimizer step after warmup. Expected full-training completion is 09:32 to 09:40 KST. Each GPU has a detached supervisor that immediately decodes all 1,000 locked prompts and runs the unchanged fail-closed stability gate, with expected gate completion at 09:50 to 10:10 KST.

INPO and IPO selection is stability-only and fixed in `baseline_repair_lock.json`; rewards are not consulted. Passing branches will continue through the remaining stages with the selected repair configuration. Expected Stage-4 completion is 14:30 to 15:30 KST, subject to continuation-pool scoring throughput.

## Gate outcome, measured 2026-07-19 10:54 KST

- `inpo_norm_a`: PASS, mean-word ratio 1.0110, maximum repeat run 2. Selected by the preregistered stability-only rule.
- `inpo_norm_b`: PASS, mean-word ratio 1.0215, maximum repeat run 2. Not selected because candidate A passed.
- `ipo_norm_a`: PASS, mean-word ratio 1.0145, maximum repeat run 2. Selected by the preregistered stability-only rule.
- `ipo_norm_b`: PASS, mean-word ratio 1.0201, maximum repeat run 1. Not selected because candidate A passed.
- `sppo_anchor`: FAIL, mean-word ratio 1.3761 and all non-repetition checks passed, but prompt index 529 contained a 35-token consecutive `🦁🌍` run. The unchanged threshold is 20, so SPPO remains excluded fail-closed.

The first automatic gate attempt pointed to an environment without vLLM and produced no generations. It was rerun without changing data or decode settings using the existing vLLM 0.25.1 environment. INPO and IPO are repaired only through Stage 1 at this point; they have not reached Stage 4 and must not be included in a Stage-4 table.

## Continuation status, measured 2026-07-19 11:22 KST

- SPPO Stage-3 candidates `sppo_strong_a` and `sppo_strong_b` passed finite 20-step smoke tests and are in their W&B-logged 900-step runs on `aiprlab-ronpo` GPUs 0 and 1. Candidate A is selected if it passes the unchanged 1,000-prompt gate; candidate B is considered only if A fails.
- The SPPO full-run W&B IDs are `974ed3f3f28e` (A) and `de4f66890136` (B).
- INPO Stage-2 pool construction completed. The selected `inpo_norm_a` continuation started on `aiprlab-ronpo` GPU 2 with W&B logging.
- The INPO Stage-2 full-run W&B ID is `cf1acb112493`.
- IPO Stage-2 pool construction completed, and selected `ipo_norm_a` continuation training started on `aiprlab-ronpo` GPU 3. The smoke run passed with W&B ID `5e83dd40e706`; the 900-step run uses W&B ID `52677ec1f1d0`.
- Continuation precompute initially invoked the inference environment, which lacks `trl`. Decode, reward scores, and pairs were preserved; only log-probability precompute was resumed with the training environment. No metric, data, or hyperparameter changed.
- Expected completion is 14:30--15:30 KST for SPPO Stage 4 and 17:30--19:00 KST for INPO/IPO Stage 4, including full-panel gates, assuming the measured training rate and no new gate failure.
- Independent gate/pool and Stage-3/4 training supervisors are active for INPO and IPO. They advance only after a finite training status and an unchanged full-panel gate pass.
- SPPO candidate gates, reward-blind A-then-B selection, Stage-4 pool/training, final gate, and the verified public HF upload are also queued under detached supervisors. Local Stage-4 weights are pruned only after public visibility, remote LFS hashes, config loading, and tokenizer loading are verified.

## Current paper evaluation, measured 2026-07-19 11:31 KST

The current six-model eligible pool includes Base, RONPO-OS, SimPO, DPO, and both HT-MNPO single-oracle arms. The 1,000-prompt aggregation is finite. RONPO-OS has helpfulness 0.8074, harmlessness 0.8437, Avg 0.8255, and Worst 0.6998 (95% CI [0.6806, 0.7190]). The missing INPO, IPO, and SPPO Stage-4 rows remain unmeasured and are shown as dashes in the paper until their repaired continuations pass the gate and the full pool is renormalized.

## Container-restart hold, measured 2026-07-19 11:52 KST

At the user's request, all queued continuation, gate, pool, Stage-3/4, and upload supervisors were terminated without touching the four active training jobs. The only remaining GPU jobs are SPPO candidates A/B and the INPO/IPO Stage-2 full runs. Their measured remaining times at 11:51 KST were 30:27, 29:38, 31:11, and 36:59, respectively. Allowing final checkpoint save and W&B synchronization, all four are expected to be safely complete by 12:35 KST. No gate, upload, or later-stage process will start automatically afterward.
