# Dev-eval cadence for the ten remaining DPO/MOPO training arms (2026-09-13)

## What was measured

`uf4_train_dpo_uniform_mse_s44` ran 1,250 optimizer updates in 2:40:20 of wall
clock while its stepping rate was 4.2 s/it, i.e. 87 minutes of stepping inside
a 160-minute run. The gap is the trainer's held-out development evaluation:

```
eval_strategy: steps      eval_steps: 250        max_steps: 1250
per_device_eval_batch_size: 1                    eval_runtime: 849.9 s
eval_samples_per_second: 29.375  ->  ~24,966 pair rows per evaluation
```

Five evaluations at 14.2 minutes each are 71 minutes, 45% of the run. The same
arithmetic reproduces `uf4_train_dpo_uniform_mse_s42` exactly (171 minutes
recorded, 87 + 71 + load/save).

There is no stall and no contention: `nvidia-smi --query-compute-apps` showed
only this job's four ranks on the four cards, and the step-rate profile is flat
at 4.2 s/it on either side of a single 16-minute gap that starts at step 250.

## What changed

`eval_steps: 250 -> 1250` in ten configs, so each keeps exactly one dev
evaluation, at the end of training:

```
dpo_uniform_mse_s43  dpo_if_only_mse_s42   dpo_truth_only_mse_s42
dpo_honesty_only_mse_s42  dpo_help_only_mse_s42  dpo_help_heavy_mse_s42
dpo_truth_heavy_mse_s42   mopo_mse_s42      mopo_mse_s43   mopo_mse_s44
```

Backups sit next to each file as `*.yaml.bak_eval250`; a `diff` against them
shows the single changed line and nothing else. The `config_sha256` recorded in
each job spec was updated to the new file, verified to match for every train
spec in the queue. Nothing DONE or RUNNING was touched: `dpo_uniform_mse_s44`
had already finished under the old cadence, and `dpo_uniform_mse_s43` launched
at 02:45Z, after the edit, so it is the first arm to use it.

`prosper_mse_s42/43/44` keep `eval_steps: 250` deliberately. PROSPER runs under
`loss_type: nbpo`, and `MNPOTrainer.evaluation_loop` computes the regression
transfer diagnostic (nMSE, sign accuracy, Pearson, Spearman on the sampled
support) only for `nbpo` and `nbpo_wbc`. That is the instrument reported as
MSE-250 / WBC-250 for the projection arms, so dropping four of its five points
would leave PROSPER the one family without the trajectory.

## Why this cannot move a result

- The trained model is unaffected. `load_best_model_at_end: false`, so no
  checkpoint is selected by a dev metric, and `save_steps: 1250` with
  `save_only_model: true` means the single saved checkpoint is the final one
  either way.
- Evaluation mutates no training state. The only `self.*` writes on the loss
  path are `_nbpo_token_counts`, `_nbpo_ref_vs_cache`, `_reference_init_report`
  (guarded by `global_step == 0`), `_loss_runtime_dtype` and
  `_nbpo_eval_points`; the last is gated on `loss_type in ("nbpo","nbpo_wbc")`,
  which excludes all ten arms. There is no ratchet or dual variable in the
  trainer -- MOPO's ratchet is solved offline in `mopo_ratchet.py`.
- Nothing consumes the discarded metrics. No script on the host
  (`nbpo_iclr/progress/*.py`) or the pod reads `trainer_state.json`,
  `eval_loss`, or `eval_drift/*` for this campaign, and
  `tab:uf4-projection-details`, the one table that quotes a dev trajectory,
  covers projection arms only and cannot contain a `dpo` or `mopo` row.
- The paper makes no global cadence claim. "first -> last of five evaluations"
  appears only inside the NBPO-MSE and utilitarian row groups of
  `tab:uf4-projection-details`.

## What it buys

Ten arms x four skipped evaluations x 14.2 minutes = 568 minutes, about
9.5 hours of 4xH200, which moves the trade-off figure's completion from
2026-09-15 to 2026-09-14.
