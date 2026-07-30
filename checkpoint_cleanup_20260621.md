# Checkpoint cleanup and upload status - 2026-06-21

## Disk cleanup

`/ext_hdd` was full before cleanup (`934M` available). After removing duplicate or obsolete intermediate checkpoints, uploading selected paper/evaluation models, and deleting the verified local upload sources, `/ext_hdd` has `835G` available.

Removed:

- Intermediate HT-MNPO stage-1 checkpoints under `/ext_hdd/sjkim/mnpo/ht_stage1_out/*stage_1/checkpoint-*`; root final model directories were initially kept for stage-2 policy initialization.
- Intermediate HT-MNPO stage-2 checkpoint `checkpoint-1220` for `athene` and `armo`; kept each player-specified best checkpoint and latest resume checkpoint.
- Intermediate/eval-irrelevant checkpoints under `/ext_hdd/sjkim/mnpo/outputs/qwen2.5-1.5b-instruct_htmnpo_skywork_online_multiobj_stage_1/checkpoint-*`; the root final model was later uploaded to HF and deleted locally.
- Old `athene` stage-1 `checkpoint-200`; `checkpoint-300`, which is used in the local RM evaluation table, was later uploaded to HF and deleted locally.
- Old non-fair RONPO stage-1 output directory under `/ext_hdd/sjkim/mnpo/outputs/qwen2.5-1.5b-instruct_ronpo_vs_htmnpo_online_multiobj_stage_1`.
- Intermediate RONPO fair stage-1 checkpoints `checkpoint-800`, `checkpoint-900`, and `checkpoint-1000`; `checkpoint-1100` used by the table and `checkpoint-1184` final were later uploaded to HF and deleted locally.
- RONPO H200 stage-2 checkpoints `checkpoint-100` through `checkpoint-400`, plus incomplete `checkpoint-600`; kept `checkpoint-500` as the latest complete resume point.
- Old INPO stage-1 intermediate checkpoints under `/ext_hdd/sjkim/mnpo/qwen2.5-1.5b/inpo_stage_1/checkpoint-*`; kept the root final model directory.
- Uploaded and verified local model sources:
  - `/ext_hdd/sjkim/mnpo/outputs/qwen2.5-1.5b-instruct_htmnpo_skywork_online_multiobj_stage_1`
  - `/ext_hdd/sjkim/mnpo/outputs/qwen2.5-1.5b-instruct_htmnpo_athene_online_multiobj_stage_1/checkpoint-300`
  - `/ext_hdd/sjkim/mnpo/ht_stage1_out/qwen2.5-1.5b-instruct_htmnpo_armo_online_multiobj_stage_1`
  - `/ext_hdd/sjkim/mnpo/outputs_ronpo_fair/qwen2.5-1.5b-instruct_ronpo_vs_htmnpo_online_multiobj_stage_1/checkpoint-1100`
  - `/ext_hdd/sjkim/mnpo/outputs_ronpo_fair/qwen2.5-1.5b-instruct_ronpo_vs_htmnpo_online_multiobj_stage_1/checkpoint-1184`

Remaining checkpoint directories:

- `/ext_hdd/sjkim/mnpo/ht_stage1_out/qwen2.5-1.5b-instruct_htmnpo_armo_online_multiobj_stage_2/checkpoint-980`
- `/ext_hdd/sjkim/mnpo/ht_stage1_out/qwen2.5-1.5b-instruct_htmnpo_armo_online_multiobj_stage_2/checkpoint-1234`
- `/ext_hdd/sjkim/mnpo/ht_stage1_out/qwen2.5-1.5b-instruct_htmnpo_athene_online_multiobj_stage_2/checkpoint-1140`
- `/ext_hdd/sjkim/mnpo/ht_stage1_out/qwen2.5-1.5b-instruct_htmnpo_athene_online_multiobj_stage_2/checkpoint-1236`
- `/ext_hdd/sjkim/mnpo/outputs_ronpo_h200/qwen2.5-1.5b-instruct_ronpo_vs_htmnpo_online_multiobj_stage_2/checkpoint-500`

## Hugging Face upload

Upload completed under Hugging Face user `promotion`. Public repos were created/updated with inference-ready artifacts only; DeepSpeed optimizer, RNG, and ZeRO resume state were intentionally excluded.

Uploaded and verified repos:

- `promotion/htmnpo-skywork-qwen25-1p5b-stage1`
- `promotion/htmnpo-athene-qwen25-1p5b-stage1-ckpt300`
- `promotion/htmnpo-armorm-qwen25-1p5b-stage1`
- `promotion/ronpo-qwen25-1p5b-stage1-ckpt1100`
- `promotion/ronpo-qwen25-1p5b-stage1-final`

Required files verified in every repo:

- `README.md`
- `model.safetensors`
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `merges.txt`
- `vocab.json`

## HT-MNPO stage-2 resume status

Ran `scripts/run_htmnpo_stage2_resume_mnpo.sh` in the existing `mnpo` tmux session with `CUDA_VISIBLE_DEVICES=0,1`.

- `athene` resumed from `checkpoint-1236`; it was already at epoch 1.0 / global step 1236, so Trainer skipped all batches and re-saved the root model.
- `armo` resumed from `checkpoint-1234`; it was already at epoch 1.0 / global step 1234, so Trainer skipped all batches and re-saved the root model.
- W&B runs:
  - `athene`: https://wandb.ai/promotion-kim/mnpo/runs/lr10qi7w
  - `armo`: https://wandb.ai/promotion-kim/mnpo/runs/khayef44
