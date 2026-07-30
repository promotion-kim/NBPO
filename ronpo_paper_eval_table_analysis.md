# RONPO Paper Evaluation Table and Analysis

작성일: 2026-06-22
Stage-2 HT-MNPO update: 2026-06-22
Stage-2 RONPO update: 2026-06-23
Stage-2 controlled local RM evaluation update: 2026-06-23
Stage-2 resumed RONPO sanity update: 2026-06-25
Stage-2 IFEval update: 2026-06-26
Final atom-ablation update: 2026-07-01

이 문서는 RONPO 논문에 넣을 수 있는 공정한 비교 표와 결과 분석을 정리한다. Stage-1 표는 기존 evaluation artifact에서 확인한 값이고, stage-2 controlled local RM 표는 2026-06-23 및 2026-06-25에 동일 prompt generation + 3 reward-model rescoring protocol로 생성한 결과다. 모든 수치는 아래 source artifact에서 확인 가능한 값만 사용했다.

## Scope

주 비교표는 `Qwen/Qwen2.5-1.5B-Instruct` 기반 stage-1 모델들의 local reward-model evaluation이다. 평가 대상은 base model, HT-MNPO 단일-oracle baselines, RONPO, SPPO-avg, INPO-avg이며, 647개 held-out prompts에 대해 동일한 generation/evaluation pipeline으로 비교했다. Stage-2 결과는 별도 섹션에 분리해 기록한다. Stage-2 controlled evaluation은 Base, HT-MNPO Skywork/Athene/ArmoRM stage 2, RONPO stage 2만 포함하므로 stage-1 seven-model normalized scores와 직접 섞어 해석하면 안 된다.

가장 안전한 논문용 claim은 다음이다.

> On a controlled local reward-model benchmark aligned with the training objectives, RONPO improves both average and worst-objective performance over homogeneous-oracle baselines, including HT-MNPO, SPPO-avg, and INPO-avg.

이 표는 human evaluation이나 reward-model-independent general capability를 직접 증명하지 않는다.

## Provenance

| Item | Value |
| --- | --- |
| Evaluation split | `data/gemma2_ufb_part1_test.jsonl` |
| Number of prompts | 647 |
| Generation seed | 42 |
| Decoding | vLLM, `temperature=0.7`, `top_p=0.9`, `max_tokens=2048`, `XFORMERS` |
| Reward objectives | `skywork`, `athene`, `armo` |
| Reward models | `Skywork/Skywork-Reward-V2-Llama-3.1-8B`, `Nexusflow/Athene-RM-8B`, `RLHFlow/ArmoRM-Llama3-8B-v0.1` |
| Scoring config | `RM_BATCH_SIZE=1`, `RM_SAMPLE_BATCH_SIZE=8`, `RM_MAX_SEQ_LENGTH=4096` |
| Evaluation command | `CUDA_VISIBLE_DEVICES=2 INCLUDE_SPPO=1 INCLUDE_INPO=1 FORCE_SCORE=1 RM_BATCH_SIZE=1 RM_SAMPLE_BATCH_SIZE=8 RM_MAX_SEQ_LENGTH=4096 DECODE_GPUS=1 CACHE_DIR=/ext_hdd/sjkim/huggingface/hub bash evalscope/run_qwen_htmnpo_stage1_table_update.sh` |
| Evaluation log | `/ext_hdd/sjkim/mnpo/logs/eval_stage1_sppo_inpo_20260622_101215.log` |
| Merged generations | `/ext_hdd/sjkim/mnpo/eval/htmnpo_stage1_base_compare/merged_model_generations_extended.json` |
| Scored files | `/ext_hdd/sjkim/mnpo/eval/htmnpo_stage1_base_compare/scored_extended/eval_{skywork,athene,armo}.jsonl` |
| Result tables | `/ext_hdd/sjkim/mnpo/eval/htmnpo_stage1_base_compare/results_extended/{model_summary,per_objective_scores,pairwise_win_rates}.csv` |

## Compared Models

| Method | Checkpoint / model | Training signal |
| --- | --- | --- |
| Base | `Qwen/Qwen2.5-1.5B-Instruct` | No stage-1 preference training |
| HT-MNPO Skywork | `/ext_hdd/sjkim/mnpo/outputs/qwen2.5-1.5b-instruct_htmnpo_skywork_online_multiobj_stage_1` | Homogeneous Skywork oracle |
| HT-MNPO Athene | `/ext_hdd/sjkim/mnpo/outputs/qwen2.5-1.5b-instruct_htmnpo_athene_online_multiobj_stage_1/checkpoint-300` | Homogeneous Athene oracle |
| HT-MNPO ArmoRM | `/ext_hdd/sjkim/mnpo/outputs/qwen2.5-1.5b-instruct_htmnpo_armo_online_multiobj_stage_1` | Homogeneous ArmoRM oracle |
| RONPO | `/ext_hdd/sjkim/mnpo/outputs_ronpo_fair/qwen2.5-1.5b-instruct_ronpo_vs_htmnpo_online_multiobj_stage_1/checkpoint-1100` | Robust multi-objective sigma adversary |
| SPPO-avg | `/home/sjkim/mnpo_runs/loki3/out/sppo_s1` | Homogeneous prompt-wise average of min-max normalized objectives |
| INPO-avg | `/home/sjkim/mnpo_runs/loki3/out/inpo_s1` | Homogeneous prompt-wise average of min-max normalized objectives |

Note: `checkpoint-1184` also exists for RONPO stage 1, but this evaluation used `checkpoint-1100`. If the paper claims final-checkpoint performance, re-evaluate the final checkpoint with the same protocol.

## Metric Definitions

For prompt `i`, reward objective `o`, and model `m`, let `s(i,o,m)` be the raw reward-model score.

Raw reward is the mean score under one reward model:

```text
raw(o,m) = mean_i s(i,o,m)
```

Raw scales are only meaningful within the same objective. Skywork, Athene, and ArmoRM raw values should not be compared across columns.

Prompt-normalized reward is min-max normalized across the compared model set for each prompt and objective:

```text
n(i,o,m) = (s(i,o,m) - min_m' s(i,o,m')) / (max_m' s(i,o,m') - min_m' s(i,o,m'))
norm(o,m) = mean_i n(i,o,m)
```

If all compared models tie for a prompt/objective, the normalized score is `0.5`. These normalized values are relative to the exact seven-model table.

Win rate vs Base is pairwise preference against the base response under the same reward model:

```text
WR(o,m) = mean_i 1[s(i,o,m) > s(i,o,Base)] * 100
```

Exact ties are counted as `0.5`; the current script uses `tie_threshold=0.0`.

Aggregate robustness metrics:

```text
Avg norm   = mean_o norm(o,m)
Worst norm = min_o norm(o,m)
Std norm   = std_o norm(o,m)
Avg WR     = mean_o WR(o,m)
Worst WR   = min_o WR(o,m)
```

Worst-objective metrics are the key robustness indicators.

## Main Paper Table

Higher is better for every metric. Bold indicates the best value among directly comparable methods.

| Method | n | Skywork norm | Skywork WR | Athene norm | Athene WR | ArmoRM norm | ArmoRM WR | Avg norm | Worst norm | Std norm | Avg WR | Worst WR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 647 | 0.7285 | - | 0.7501 | - | 0.8645 | - | 0.7811 | 0.7285 | 0.0597 | - | - |
| HT-MNPO Skywork | 647 | 0.7637 | 55.95 | 0.7889 | 58.19 | 0.8833 | 58.73 | 0.8119 | 0.7637 | 0.0515 | 57.62 | 55.95 |
| HT-MNPO Athene | 647 | 0.7640 | 56.03 | 0.7557 | 49.69 | 0.8609 | 49.61 | 0.7935 | 0.7557 | 0.0477 | 51.78 | 49.61 |
| HT-MNPO ArmoRM | 647 | 0.7552 | 56.34 | 0.7531 | 55.72 | 0.8494 | 50.70 | 0.7859 | 0.7531 | 0.0449 | 54.25 | 50.70 |
| RONPO | 647 | **0.8609** | **67.47** | **0.8950** | **68.55** | **0.9095** | **59.81** | **0.8885** | **0.8609** | **0.0204** | **65.28** | **59.81** |
| SPPO-avg | 647 | 0.3361 | 13.60 | 0.0838 | 2.78 | 0.1948 | 1.08 | 0.2049 | 0.0838 | 0.1032 | 5.82 | 1.08 |
| INPO-avg | 647 | 0.0457 | 3.86 | 0.1288 | 6.57 | 0.0274 | 0.77 | 0.0673 | 0.0274 | 0.0441 | 3.74 | 0.77 |

Table note: `WR` is win rate against the base model in percent. SPPO-avg and INPO-avg are homogeneous-oracle baselines trained with the prompt-wise average of min-max normalized Skywork, Athene, and ArmoRM scores; they are not single-reward-model baselines.

## Pairwise RONPO Win Rates

This table reports RONPO's direct win rate against each trained baseline. Values are percentages under each reward objective.

| Comparison | Skywork WR | Athene WR | ArmoRM WR | Avg WR | Worst WR |
| --- | ---: | ---: | ---: | ---: | ---: |
| RONPO vs HT-MNPO Skywork | 62.29 | 63.37 | 57.50 | 61.05 | 57.50 |
| RONPO vs HT-MNPO Athene | 63.21 | 68.32 | 63.83 | 65.12 | 63.21 |
| RONPO vs HT-MNPO ArmoRM | 63.68 | 67.31 | 62.21 | 64.40 | 62.21 |
| RONPO vs SPPO-avg | 92.97 | 98.84 | 98.61 | 96.81 | 92.97 |
| RONPO vs INPO-avg | 97.53 | 98.38 | 99.85 | 98.58 | 97.53 |

These pairwise results are stronger evidence than only reporting each method's win rate against the base model: RONPO is preferred over every trained baseline by all three reward objectives.

## Result Analysis

RONPO is the best method on every normalized objective, every win-rate objective, and every aggregate robustness metric in this seven-model local RM benchmark. Its Avg norm is `0.8885`, and its Worst norm is `0.8609`; the strongest non-RONPO trained baseline is HT-MNPO Skywork with `0.8119` Avg norm and `0.7637` Worst norm. This is a gain of `+0.0765` Avg norm and `+0.0972` Worst norm.

The win-rate view gives the same ordering. RONPO reaches `65.28%` Avg WR vs Base and `59.81%` Worst WR vs Base. HT-MNPO Skywork, the strongest non-RONPO baseline by Avg WR, reaches `57.62%` Avg WR and `55.95%` Worst WR. RONPO therefore improves by `+7.65` percentage points in Avg WR and `+3.86` percentage points in Worst WR over the strongest HT-MNPO baseline.

The robustness pattern is important. HT-MNPO Skywork has reasonable performance on all three objectives but a larger cross-objective spread (`Std norm = 0.0515`). RONPO has the lowest spread (`Std norm = 0.0204`) while also improving the mean and worst objective. This supports the paper's core robustness claim: RONPO is not merely optimizing one favorable reward model; it raises the weakest objective floor.

SPPO-avg and INPO-avg perform poorly in this run, with Avg WR vs Base of `5.82%` and `3.74%`, respectively. This should be reported carefully. The fair statement is not that SPPO or INPO are universally weak algorithms, but that under this implementation and average-local-oracle instantiation, they do not provide competitive robustness on this heterogeneous local RM benchmark. Because SPPO/INPO require a homogeneous preference oracle, the average-oracle construction is a reasonable non-cherry-picked baseline, but it is still a different training objective from RONPO's adversarial robust objective.

The most defensible reviewer-facing interpretation is:

> RONPO's advantage is largest on aggregate robustness metrics and remains positive in direct pairwise comparisons against each trained baseline under every reward objective.

## Compatibility Audit

Directly comparable:

- All rows in the main table use the same 647 prompts.
- All model responses are generated with the same decode seed and decoding configuration.
- All rows are scored by the same three local reward models in the same forced rescore pass.
- Normalized values are computed over the same seven-model set.

Partially comparable:

- HT-MNPO, SPPO-avg, and INPO-avg are homogeneous-oracle methods; RONPO is a heterogeneous robust objective. The evaluation is fair because responses are judged by the same multi-objective local RM suite, but the training objectives differ by design.
- SPPO-avg and INPO-avg use an average normalized oracle rather than a single reward model. This avoids selecting a favorable single oracle, but it should be named explicitly in the paper.

Not established by this table:

- Human preference superiority.
- Reward-model-independent general benchmark performance.
- Statistical significance across multiple generation seeds or independent training seeds.
- Final RONPO `checkpoint-1184` performance.

## Stage-2 Training-Internal Results

This section is not part of the directly comparable stage-1 local RM table above. The values below are trainer evaluation metrics on each method's own stage-2 precomputed pair dataset. They verify that the HT-MNPO stage-2 players and RONPO stage 2 completed and record checkpoint provenance, but they should not be mixed with the 647-prompt generation-and-rescoring table unless the same response-generation evaluation is run for all stage-2 models.

### Stage-2 Provenance

| Item | Value |
| --- | --- |
| Methods | HT-MNPO stage 2 with one homogeneous oracle player per run; RONPO stage 2 with robust multi-objective sigma adversary |
| Base policy for stage 2 | HT-MNPO: corresponding HT-MNPO stage-1 player policy plus historical opponent policies. RONPO: RONPO stage-1 policy plus historical opponent policy. |
| HT-MNPO data root | `/ext_hdd/sjkim/mnpo/data/qwen2.5-1.5b-instruct_online_htmnpo_ronpo/htmnpo/{skywork,athene,armo}/iter2/precomputed` |
| RONPO data root | `/data/mnpo/work/qwen2.5-1.5b-instruct_online_htmnpo_ronpo/ronpo/iter2/precomputed_sigma_best_vs_adversary_pairs1_samples0` |
| HT-MNPO output root | `/ext_hdd/sjkim/mnpo/ht_stage1_out/qwen2.5-1.5b-instruct_htmnpo_{player}_online_multiobj_stage_2` |
| RONPO output root | `/data/mnpo/outputs_ronpo_h200/qwen2.5-1.5b-instruct_ronpo_vs_htmnpo_online_multiobj_stage_2` |
| Launcher scripts | HT-MNPO: `scripts/run_htmnpo_stage2_resume_mnpo.sh`, `scripts/run_htmnpo_stage2_skywork_mnpo.sh`; RONPO: `mlxp/ronpo-stage2-h200-1gpu-r6-job.yaml` |
| Accelerator | HT-MNPO: `accelerate_configs/deepspeed_zero3.yaml`, 2 processes, bf16, DeepSpeed ZeRO-3. RONPO: `accelerate_configs/single_gpu.yaml`, 1 H200 GPU, bf16. |
| Effective batch | HT-MNPO: `per_device_train_batch_size=8`, `gradient_accumulation_steps=1`, 2 GPUs, total train batch size 16. RONPO: `per_device_train_batch_size=8`, `gradient_accumulation_steps=2`, 1 GPU, total train batch size 16. |
| Evaluation during training | `do_eval=true`, `eval_steps=100`, `per_device_eval_batch_size=8`, `generate_during_eval=false` |
| Model length | `max_length=2048`, `max_prompt_length=1800` |
| Optimizer schedule | AdamW, learning rate `5e-7`, cosine schedule, warmup ratio `0.1`, weight decay `0.0` |
| Selection metric | `eval_loss` in `trainer_state.json` |
| Source state files | Stage-2 `trainer_state.json` and `all_results.json` under each output root; RONPO H200 DDN files read through `mmpae-pvc-shell` |
| Source logs | HT-MNPO: `/ext_hdd/sjkim/mnpo/logs/htmnpo_stage2_resume_mnpo_20260619_222944.log`, `/ext_hdd/sjkim/mnpo/logs/htmnpo_stage2_resume_mnpo_20260621_192351.log`, `/ext_hdd/sjkim/mnpo/logs/htmnpo_stage2_skywork_mnpo_20260622_122629.log`; RONPO: `/data/mnpo/logs/ronpo_s2_h200_*.log`, W&B run `tz9htnms` |

### Stage-2 Status Table

Lower is better for `eval_loss`; higher is better for `eval_rewards/accuracies` and `eval_rewards/margins`.

| Method / player | Train pairs | Eval pairs | Final step | Final checkpoint | Best saved checkpoint | Best saved eval loss | Best logged eval loss | Last eval loss | Last eval accuracy | Last eval margin | W&B run |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Skywork | 18,997 | 624 | 1,188 | `checkpoint-1188` and root model | `checkpoint-1100` | 0.3420 | 0.3345 at step 940 | 0.3506 | 0.7372 | 0.7636 | `hazohdwq` |
| Athene | 19,761 | 644 | 1,236 | `checkpoint-1236` and root model | `checkpoint-1140` | 0.0866 | 0.0866 at step 1140 | 0.0898 | 0.7500 | 0.0720 | `i1bm8c3g`; resume re-save `lr10qi7w` |
| ArmoRM | 19,741 | 644 | 1,234 | `checkpoint-1234` and root model | `checkpoint-980` | 0.0693 | 0.0693 at step 980 | 0.0733 | 0.7012 | 0.4987 | `k88fzj9v`; resume re-save `khayef44` |
| RONPO | 19,835 | 646 | 1,240 | `checkpoint-1240` and root model | `checkpoint-1100` | 0.7349 | 0.7349 at step 1100 | 0.7355 at step 1200 | 0.7608 | 2.9143 | `tz9htnms` |

Stage-2 compatibility note: these rows are comparable as completion and checkpoint provenance, but not as a paper performance ranking. Each HT-MNPO row evaluates on a player-specific homogeneous-oracle pair dataset; the RONPO row evaluates on a robust sigma-adversary pair dataset. The `eval_loss` values therefore have different targets and should not be compared across methods. The paper-ready comparison requires generating responses from HT-MNPO stage-2 and RONPO stage-2 checkpoints on the same held-out prompts, then rescoring all responses with the same reward-model suite.

### Stage-2 Controlled Local RM Evaluation

This is the controlled stage-2 generation-and-rescoring table requested on 2026-06-23. It should be treated as a diagnostic result, not as paper-ready evidence for RONPO superiority: under this run, RONPO stage 2 performs substantially worse than Base and HT-MNPO stage 2 on the local RM suite.

#### Stage-2 Controlled Provenance

| Item | Value |
| --- | --- |
| Prompt split | `data/gemma2_ufb_part2_test.jsonl` |
| Number of prompts | 647 |
| Models | Base, HT-MNPO Skywork stage 2, HT-MNPO Athene stage 2, HT-MNPO ArmoRM stage 2, RONPO stage 2 |
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| HT-MNPO stage-2 checkpoints | `/ext_hdd/sjkim/mnpo/ht_stage1_out/qwen2.5-1.5b-instruct_htmnpo_{skywork,athene,armo}_online_multiobj_stage_2` |
| RONPO stage-2 checkpoint | `/ext_hdd/sjkim/mnpo/outputs_ronpo_h200/qwen2.5-1.5b-instruct_ronpo_vs_htmnpo_online_multiobj_stage_2` |
| Generation script | `scripts/run_stage2_controlled_eval_parallel.sh` |
| Generation config | vLLM, seed `42`, `temperature=0.7`, `top_p=0.9`, `max_tokens=4096`, `attention_backend=XFORMERS`, `dtype=bfloat16` |
| Reward objectives | `skywork`, `athene`, `armo` |
| Reward models | `Skywork/Skywork-Reward-V2-Llama-3.1-8B`, `Nexusflow/Athene-RM-8B`, `RLHFlow/ArmoRM-Llama3-8B-v0.1` |
| Final scoring config | Skywork retry: `batch_size=4`, `sample_batch_size=8`, `attn_implementation=sdpa`; Athene: `batch_size=8`, `sample_batch_size=32`; ArmoRM: `batch_size=16`, `sample_batch_size=32` |
| Failure handled | Initial Skywork scoring with eager attention and batch 16 OOMed; final table uses completed `score_skywork_retry.log` and 647-line `eval_skywork.jsonl` |
| Work directory | `/ext_hdd/sjkim/mnpo/eval/htmnpo_ronpo_stage2_base_compare` |
| Generation artifacts | `generations/{baseline,htmnpo_skywork,htmnpo_athene,htmnpo_armo,ronpo}/output_42.json` |
| Scored files | `scored/eval_{skywork,athene,armo}.jsonl`, each 647 rows |
| Result tables | `results/{model_summary,per_objective_scores,pairwise_win_rates}.csv` |

#### Stage-2 Controlled Main Table

Higher is better for every metric. Normalized values are computed over this five-model stage-2 comparison set only.

| Method | n | Skywork norm | Skywork WR | Athene norm | Athene WR | ArmoRM norm | ArmoRM WR | Avg norm | Worst norm | Std norm | Avg WR | Worst WR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 647 | 0.5759 | - | 0.5697 | - | **0.7595** | - | 0.6350 | 0.5697 | 0.0880 | - | - |
| HT-MNPO Skywork S2 | 647 | **0.6764** | **58.73** | **0.5754** | **50.39** | 0.7293 | **47.53** | **0.6603** | **0.5754** | **0.0638** | **52.22** | **47.53** |
| HT-MNPO Athene S2 | 647 | 0.6076 | 52.32 | 0.4509 | 40.88 | 0.6374 | 38.18 | 0.5653 | 0.4509 | 0.0818 | 43.79 | 38.18 |
| HT-MNPO ArmoRM S2 | 647 | 0.6307 | 55.18 | 0.4816 | 42.74 | 0.6556 | 37.94 | 0.5893 | 0.4816 | 0.0768 | 45.29 | 37.94 |
| RONPO S2 | 647 | 0.2604 | 26.12 | 0.4785 | 46.21 | 0.2275 | 19.78 | 0.3221 | 0.2275 | 0.1114 | 30.71 | 19.78 |

Table note: `WR` is win rate against the base model in percent. Base has the best ArmoRM normalized score in this five-model stage-2 table, so the best ArmoRM norm is not a trained-method value.

#### Stage-2 Direct Pairwise RONPO Win Rates

Values are RONPO's direct win rates against each comparison model under each reward objective.

| Comparison | Skywork WR | Athene WR | ArmoRM WR | Avg WR | Worst WR |
| --- | ---: | ---: | ---: | ---: | ---: |
| RONPO S2 vs Base | 26.12 | 46.21 | 19.78 | 30.71 | 19.78 |
| RONPO S2 vs HT-MNPO Skywork S2 | 23.65 | 45.60 | 21.87 | 30.37 | 21.87 |
| RONPO S2 vs HT-MNPO Athene S2 | 28.05 | 51.55 | 25.66 | 35.09 | 25.66 |
| RONPO S2 vs HT-MNPO ArmoRM S2 | 26.51 | 49.30 | 23.96 | 33.26 | 23.96 |

#### Stage-2 Controlled Analysis

This controlled stage-2 run does not support a stage-2 RONPO advantage. HT-MNPO Skywork S2 is the strongest trained method by Avg norm (`0.6603`), Worst norm (`0.5754`), Avg WR (`52.22%`), and Worst WR (`47.53%`). RONPO S2 has Avg norm `0.3221`, Worst norm `0.2275`, Avg WR `30.71%`, and Worst WR `19.78%`. It also loses direct pairwise comparisons to all three HT-MNPO S2 players on average.

The main sanity signal is generation length. On the merged stage-2 generations, median response length in characters is:

| Model | Mean chars | Median chars | Min chars | Max chars |
| --- | ---: | ---: | ---: | ---: |
| Base | 1,624 | 829 | 1 | 19,081 |
| HT-MNPO Skywork S2 | 2,163 | 1,490 | 52 | 20,753 |
| HT-MNPO Athene S2 | 3,552 | 1,744 | 68 | 24,299 |
| HT-MNPO ArmoRM S2 | 2,618 | 1,609 | 68 | 24,197 |
| RONPO S2 | 14,649 | 16,559 | 163 | 38,375 |

RONPO S2 is producing much longer outputs than every comparison model under the same `max_tokens=4096` decoding cap. Manual spot checks show verbose and sometimes repetitive phrasing. This likely explains a substantial part of the reward-model penalty and should be investigated before using stage-2 RONPO in the paper.

Reviewer-facing interpretation: do not cite this table as evidence for RONPO superiority. The defensible use is diagnostic: stage-2 RONPO completed training, but the current decoded policy appears length/pathology-prone under the held-out generation protocol. Before final submission, rerun controlled stage-2 evaluation after checking the selected checkpoint, EOS behavior, chat template, and/or length-controlled decoding.

#### Stage-2 Resumed RONPO Sanity Evaluation

This 2026-06-25 rerun evaluates the resumed RONPO stage-2 run, using the best trainer checkpoint (`checkpoint-1400`) and the final checkpoint (`checkpoint-2457`) against the same Base and HT-MNPO stage-2 checkpoints. This supersedes the earlier 2026-06-23 RONPO S2 diagnostic for claims about the resumed RONPO run, but the earlier result should remain in the appendix or internal notes as evidence that checkpoint/training-state selection matters.

| Item | Value |
| --- | --- |
| Prompt split | `data/gemma2_ufb_part2_test.jsonl` |
| Number of prompts | 647 |
| RONPO resumed run | `/ext_hdd/sjkim/mnpo/outputs_ronpo_h200/qwen2.5-1.5b-instruct_ronpo_stage2_relative_lr2e8_od2g2` |
| RONPO checkpoints | `checkpoint-1400` and `checkpoint-2457` |
| Comparison models | Base, HT-MNPO Skywork S2, HT-MNPO Athene S2, HT-MNPO ArmoRM S2 |
| Generation config | vLLM, seed `42`, `temperature=0.7`, `top_p=0.9`, `max_tokens=4096`, `attention_backend=XFORMERS`, `dtype=bfloat16` |
| Reward objectives | `skywork`, `athene`, `armo` |
| Work directory | `/ext_hdd/sjkim/mnpo/eval/ronpo_stage2_resume_sanity_20260625` |
| Result tables | `results/{model_summary,per_objective_scores,pairwise_win_rates,generation_quality_summary}.csv` |

Higher is better for every reward metric. Normalized values are computed over this six-model resumed stage-2 comparison set only.

| Method | n | Avg norm | Worst norm | Std norm | Avg WR vs Base | Worst WR vs Base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 647 | 0.4852 | 0.4099 | 0.0621 | - | - |
| HT-MNPO Skywork S2 | 647 | 0.5020 | 0.4735 | 0.0201 | 52.29 | 47.53 |
| HT-MNPO Athene S2 | 647 | 0.3881 | 0.3372 | 0.0443 | 43.84 | 38.02 |
| HT-MNPO ArmoRM S2 | 647 | 0.4148 | 0.3702 | 0.0433 | 45.36 | 37.87 |
| RONPO S2 `checkpoint-1400` | 647 | 0.6827 | 0.6569 | 0.0215 | 64.66 | 60.36 |
| RONPO S2 `checkpoint-2457` | 647 | **0.7025** | **0.6701** | 0.0239 | **66.05** | 60.05 |

Per-objective scores for the final RONPO checkpoint are also uniformly strong: Skywork norm `0.6701` with `68.86%` WR vs Base, Athene norm `0.7269` with `69.24%` WR vs Base, and ArmoRM norm `0.7106` with `60.05%` WR vs Base. This is the first stage-2 table that supports a RONPO stage-2 advantage under the controlled local RM protocol.

Direct pairwise win rates for the final RONPO checkpoint are:

| Comparison | Skywork WR | Athene WR | ArmoRM WR | Avg WR | Worst WR |
| --- | ---: | ---: | ---: | ---: | ---: |
| RONPO S2 final vs Base | 68.86 | 69.24 | 60.05 | 66.05 | 60.05 |
| RONPO S2 final vs HT-MNPO Skywork S2 | 63.76 | 71.79 | 68.32 | 67.96 | 63.76 |
| RONPO S2 final vs HT-MNPO Athene S2 | 64.76 | 79.91 | 73.96 | 72.88 | 64.76 |
| RONPO S2 final vs HT-MNPO ArmoRM S2 | 64.30 | 77.51 | 73.88 | 71.90 | 64.30 |

The output-length pathology from the earlier stage-2 run is not present in this resumed checkpoint comparison.

| Method | Mean chars | Median chars | P90 chars | Long-rate |
| --- | ---: | ---: | ---: | ---: |
| Base | 1,624 | 829 | 3,452 | 2.01 |
| HT-MNPO Skywork S2 | 2,163 | 1,490 | 3,752 | 2.32 |
| HT-MNPO Athene S2 | 3,552 | 1,744 | 7,495 | 9.43 |
| HT-MNPO ArmoRM S2 | 2,618 | 1,609 | 4,237 | 4.79 |
| RONPO S2 `checkpoint-1400` | 2,174 | 1,761 | 4,190 | **1.08** |
| RONPO S2 `checkpoint-2457` | 2,206 | 1,715 | 4,366 | 1.39 |

Interpretation: the resumed RONPO stage-2 final checkpoint is now the best local-RM candidate for the paper table. It improves both average and worst-objective normalized reward without relying on longer generations. The next reviewer-critical step is reward-model-independent validation, because this table is still local-RM aligned with the training objectives.

#### Stage-2 IFEval / EvalScope Evaluation

IFEval is a reward-model-independent rule-based instruction-following benchmark. It does not judge response helpfulness or preference quality, but it is useful for checking whether preference optimization damaged verifiable instruction compliance. This section evaluates the same stage-2 candidates with EvalScope after repairing the local EvalScope/vLLM environment with workspace-local overlays.

| Item | Value |
| --- | --- |
| Benchmark | IFEval through EvalScope `1.8.1` |
| Dataset | `opencompass/ifeval`, EvalScope `eval_split=train` |
| Number of prompts | 541 |
| Generation backend | vLLM OpenAI-compatible server, one model per GPU |
| Generation config | `temperature=0.0`, `eval_batch_size=20`, `max_model_len=4096`, `dtype=bfloat16` |
| Evaluation script | `scripts/run_stage2_ifeval_suite.sh` plus clean baseline rerun through `evalscope/run_vllm_eval.sh` |
| Environment repair | Workspace-local `evalscope==1.8.1` under `.evalscope_pkgs`; vLLM 0.5.1 compatibility overlays under `.vllm051_pkgs` and `vllm_compat/` |
| Work logs | `/ext_hdd/sjkim/mnpo/eval/ronpo_stage2_ifeval_20260626_r4/logs`, `/ext_hdd/sjkim/mnpo/eval/ronpo_stage2_ifeval_20260626_baseline_clean/logs` |
| Result JSON files | `outputs/20260626_113913/reports/baseline_clean/ifeval.json`, `outputs/20260626_112020/reports/ronpo_s2_final/ifeval.json`, `outputs/20260626_112446/reports/{ronpo_s2_ckpt1400,htmnpo_skywork_s2}/ifeval.json`, `outputs/20260626_113052/reports/{htmnpo_athene_s2,htmnpo_armo_s2}/ifeval.json` |
| Invalidated run | The first baseline run in `outputs/20260626_112020/reports/baseline/ifeval.json` processed only 539 prompts because of an NLTK cache race; it is superseded by `baseline_clean` with 541 prompts. |

Higher is better. `prompt strict` is the primary IFEval metric because every instruction attached to a prompt must be satisfied. `inst strict` is per-instruction strict accuracy. Loose variants apply IFEval's more tolerant matching rules.

| Method | n | Prompt strict | Inst strict | Prompt loose | Inst loose | Avg output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 541 | **0.4140** | **0.5357** | **0.4510** | **0.5687** | 610.2 |
| RONPO S2 `checkpoint-1400` | 541 | 0.4030 | 0.5213 | 0.4473 | 0.5653 | 762.6 |
| RONPO S2 `checkpoint-2457` | 541 | 0.3993 | 0.5182 | 0.4492 | 0.5653 | 725.4 |
| HT-MNPO Skywork S2 | 541 | 0.3530 | 0.4812 | 0.4067 | 0.5323 | 1166.3 |
| HT-MNPO Athene S2 | 541 | 0.3290 | 0.4547 | 0.3956 | 0.5219 | 1969.5 |
| HT-MNPO ArmoRM S2 | 541 | 0.3216 | 0.4529 | 0.3882 | 0.5139 | 1570.7 |

Compatibility note: all valid rows above use the same prompt set, EvalScope version, deterministic decoding temperature, model server wrapper, and metric definitions. The only excluded row is the initial baseline attempt with `n=539`, which is not directly comparable.

Analysis: the base model is best overall on IFEval, so this benchmark does not support a claim that RONPO improves rule-based instruction following over the base model. The useful result is preservation under stage-2 preference training: both RONPO stage-2 checkpoints are much closer to Base than all three HT-MNPO stage-2 players. On the primary prompt-strict metric, `checkpoint-1400` is the best trained model (`0.4030`), and the final checkpoint remains close (`0.3993`). This complements the resumed local-RM result above: RONPO S2 improves local reward-model robustness while largely preserving IFEval compliance, whereas the HT-MNPO S2 players show larger IFEval degradation and much longer average generations.

Reviewer-facing claim: "On IFEval, RONPO stage-2 preserves instruction-following performance substantially better than HT-MNPO stage-2, although the unaligned base model remains strongest on the rule-based metric."

#### Stage-2 Bootstrap CIs and High-Disagreement Stress Test

This 2026-06-26 analysis recomputes prompt-level bootstrap confidence intervals and a reward-source disagreement stress subset from the existing resumed stage-2 local-RM artifacts. It uses no new reward scoring and no new generation.

| Item | Value |
| --- | --- |
| Source artifact | `/ext_hdd/sjkim/mnpo/eval/ronpo_stage2_resume_sanity_20260625` |
| Scored inputs | `scored/eval_{skywork,athene,armo}.jsonl` |
| Compared models | Base, HT-MNPO Skywork/Athene/ArmoRM S2, RONPO S2 `checkpoint-1400`, RONPO S2 `checkpoint-2457` |
| Full prompt count | 647 |
| High-disagreement subset | Top 25% prompts by reward-source pairwise disagreement, `n=162` |
| Disagreement score | Fraction of non-tied objective-pair/model-pair comparisons where two reward objectives prefer opposite candidate responses |
| Bootstrap | 2,000 prompt-level paired bootstrap resamples; 95% percentile confidence intervals |
| Generated report | `analysis/stage2_robustness_20260626/report.md` |
| Generated CSVs | `analysis/stage2_robustness_20260626/{bootstrap_model_metrics,bootstrap_pairwise_ronpo_final,model_summary_disagreement_top25,pairwise_ronpo_final_disagreement_top25}.csv` |

Full held-out set. Higher is better for every metric. Brackets show 95% prompt-bootstrap confidence intervals.

| Method | n | Avg norm | Worst norm | Avg WR vs Base | Worst WR vs Base |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 647 | 0.4852 [0.4607, 0.5114] | 0.4099 [0.3810, 0.4403] | - | - |
| HT-MNPO Skywork S2 | 647 | 0.5020 [0.4775, 0.5252] | 0.4735 [0.4456, 0.4998] | 0.5229 [0.4902, 0.5538] | 0.4753 [0.4359, 0.5108] |
| HT-MNPO Athene S2 | 647 | 0.3881 [0.3647, 0.4125] | 0.3372 [0.3113, 0.3645] | 0.4384 [0.4085, 0.4704] | 0.3802 [0.3454, 0.4158] |
| HT-MNPO ArmoRM S2 | 647 | 0.4148 [0.3937, 0.4384] | 0.3702 [0.3451, 0.3961] | 0.4536 [0.4243, 0.4858] | 0.3787 [0.3423, 0.4189] |
| RONPO S2 `checkpoint-1400` | 647 | 0.6827 [0.6606, 0.7041] | 0.6569 [0.6309, 0.6819] | 0.6466 [0.6156, 0.6759] | 0.6036 [0.5657, 0.6407] |
| RONPO S2 `checkpoint-2457` | 647 | **0.7025 [0.6814, 0.7243]** | **0.6701 [0.6439, 0.6961]** | **0.6605 [0.6314, 0.6896]** | 0.6005 [0.5618, 0.6399] |

High-disagreement top-25% subset. This is the most targeted stress test for RONPO's heterogeneous-objective claim.

| Method | n | Avg norm | Worst norm | Avg WR vs Base | Worst WR vs Base |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 162 | 0.5246 [0.4824, 0.5677] | 0.3788 [0.3172, 0.4371] | - | - |
| HT-MNPO Skywork S2 | 162 | 0.4956 [0.4520, 0.5366] | 0.4550 [0.3932, 0.4985] | 0.4794 [0.4259, 0.5298] | 0.3457 [0.2716, 0.4198] |
| HT-MNPO Athene S2 | 162 | 0.4329 [0.3921, 0.4770] | 0.3392 [0.2836, 0.3962] | 0.4280 [0.3817, 0.4753] | 0.3302 [0.2562, 0.3859] |
| HT-MNPO ArmoRM S2 | 162 | 0.4814 [0.4417, 0.5212] | 0.4113 [0.3577, 0.4567] | 0.4681 [0.4197, 0.5175] | 0.3642 [0.2931, 0.4198] |
| RONPO S2 `checkpoint-1400` | 162 | 0.5694 [0.5300, 0.6087] | 0.5487 [0.4873, 0.5862] | 0.5340 [0.4846, 0.5834] | 0.4599 [0.3858, 0.5278] |
| RONPO S2 `checkpoint-2457` | 162 | **0.6018 [0.5626, 0.6403]** | **0.5733 [0.5176, 0.6166]** | **0.5761 [0.5267, 0.6265]** | **0.4815 [0.4105, 0.5586]** |

Direct pairwise win rates for RONPO S2 final on the high-disagreement subset:

| Comparison | Skywork WR | Athene WR | ArmoRM WR | Avg WR | Worst WR |
| --- | ---: | ---: | ---: | ---: | ---: |
| RONPO S2 final vs Base | 0.6543 | 0.5926 | 0.4815 | 0.5761 [0.5267, 0.6265] | 0.4815 [0.4105, 0.5586] |
| RONPO S2 final vs HT-MNPO Skywork S2 | 0.4907 | 0.6481 | 0.6173 | 0.5854 [0.5391, 0.6348] | 0.4907 [0.4167, 0.5679] |
| RONPO S2 final vs HT-MNPO Athene S2 | 0.4722 | 0.7469 | 0.5926 | 0.6039 [0.5556, 0.6502] | 0.4722 [0.3951, 0.5463] |
| RONPO S2 final vs HT-MNPO ArmoRM S2 | 0.4938 | 0.6883 | 0.6481 | 0.6101 [0.5668, 0.6543] | 0.4938 [0.4228, 0.5710] |

Interpretation: the high-disagreement subset isolates prompts where the reward sources conflict over candidate responses. RONPO S2 final remains the strongest method on average normalized reward, worst-objective normalized reward, average win rate vs Base, and worst win rate vs Base. This directly supports the paper's core robustness claim: objective-adversarial training improves the weakest reward-source floor under heterogeneous judge disagreement, rather than merely improving a favorable average. The pairwise rows are directionally positive on average, but the worst-objective CIs on the stress subset still touch or cross 0.5 for some HT-MNPO comparisons; therefore, avoid claiming statistically decisive pairwise dominance on every individual objective without additional prompts or seeds.

## Stage-2 GPT-5.5 Pairwise Judge Evaluation

This section reports a reward-model-independent GPT-5.5 judge evaluation for the same resumed stage-2 candidates. It should be interpreted separately from the local reward-model tables above: the prompt set and candidate generations are the same, but the evaluator is an external LLM judge rather than Skywork/Athene/ArmoRM. This is therefore evidence about transfer to a held-out preference judge, not another local-RM robustness metric.

| Item | Value |
| --- | --- |
| Source generation artifact | `/ext_hdd/sjkim/mnpo/eval/ronpo_stage2_resume_sanity_20260625/merged_model_generations.json` |
| Judge | `gpt-5.5-2026-04-23` through OpenAI Batch API |
| Compared models | Base, HT-MNPO Skywork/Athene/ArmoRM S2, RONPO S2 `checkpoint-1400`, RONPO S2 `checkpoint-2457` |
| Prompt set | 647 held-out UltraFeedback prompts |
| Pairwise comparisons | All 15 model pairs per prompt, `647 x 15 = 9,705` judgments |
| Response order | Deterministically randomized per prompt-pair |
| Tie handling | Tie counts as 0.5 win for both sides |
| Bootstrap | 2,000 prompt-level paired bootstrap resamples; 95% percentile confidence intervals |
| Coverage | 9,705 / 9,705 parsed judgments after retrying output-limit failures |
| Final report | `analysis/openai_judge_20260626/full_647_final_paper_summary/report.md` |
| CSV artifacts | `analysis/openai_judge_20260626/full_647_final_paper_summary/{full_model_scoreboard_ci,full_pairwise_ci,stress_model_scoreboard_ci,stress_pairwise_ci}.csv` |
| API usage cost estimate | Full original batch: about `$61.53`; corrected retry: about `$17.89`; pilot diagnostic: about `$16.90` |

Full held-out set. Higher mean pairwise win rate is better. Brackets show 95% prompt-bootstrap confidence intervals.

| Rank | Method | Mean pairwise WR | 95% CI | Matchups |
| ---: | --- | ---: | ---: | ---: |
| 1 | RONPO S2 `checkpoint-2457` | **0.5927** | [0.5709, 0.6136] | 5 |
| 2 | Base | 0.5581 | [0.5329, 0.5841] | 5 |
| 3 | RONPO S2 `checkpoint-1400` | 0.5532 | [0.5306, 0.5750] | 5 |
| 4 | HT-MNPO Skywork S2 | 0.5054 | [0.4841, 0.5272] | 5 |
| 5 | HT-MNPO ArmoRM S2 | 0.4263 | [0.4025, 0.4493] | 5 |
| 6 | HT-MNPO Athene S2 | 0.3643 | [0.3425, 0.3855] | 5 |

Key direct pairwise results. `Left WR` is the win rate of the first method named in the comparison.

| Pair | n | Left WR | 95% CI | Tie rate |
| --- | ---: | ---: | ---: | ---: |
| Base vs RONPO S2 `checkpoint-2457` | 647 | 0.4822 | [0.4467, 0.5186] | 0.1360 |
| Base vs RONPO S2 `checkpoint-1400` | 647 | 0.5108 | [0.4753, 0.5456] | 0.1314 |
| HT-MNPO Skywork S2 vs RONPO S2 `checkpoint-2457` | 647 | 0.4134 | [0.3802, 0.4467] | 0.1808 |
| HT-MNPO Athene S2 vs RONPO S2 `checkpoint-2457` | 647 | 0.3145 | [0.2805, 0.3486] | 0.1376 |
| HT-MNPO ArmoRM S2 vs RONPO S2 `checkpoint-2457` | 647 | 0.3617 | [0.3269, 0.3964] | 0.1051 |
| RONPO S2 `checkpoint-1400` vs RONPO S2 `checkpoint-2457` | 647 | 0.4645 | [0.4320, 0.4961] | 0.3385 |

High-disagreement top-25% subset under the local reward-source disagreement ranking. This is a stress subset, not the primary GPT-5.5 judge table.

| Rank | Method | Mean pairwise WR | 95% CI | Matchups |
| ---: | --- | ---: | ---: | ---: |
| 1 | Base | **0.6302** | [0.5802, 0.6772] | 5 |
| 2 | RONPO S2 `checkpoint-2457` | 0.5241 | [0.4790, 0.5679] | 5 |
| 3 | RONPO S2 `checkpoint-1400` | 0.4852 | [0.4414, 0.5272] | 5 |
| 4 | HT-MNPO ArmoRM S2 | 0.4809 | [0.4352, 0.5290] | 5 |
| 5 | HT-MNPO Skywork S2 | 0.4796 | [0.4377, 0.5185] | 5 |
| 6 | HT-MNPO Athene S2 | 0.4000 | [0.3574, 0.4389] | 5 |

Interpretation: the full held-out GPT-5.5 judge table supports the main stage-2 transfer claim: RONPO final is the strongest overall method under an external judge, and it directly beats all three HT-MNPO players by clear margins. The direct Base comparison is weaker: RONPO final beats Base by 51.78% to 48.22%, but the confidence interval for Base's left-win rate includes 0.5, so this should be described as a small directional gain rather than statistically decisive pairwise dominance over Base. On the high-disagreement stress subset, Base is strongest under GPT-5.5 even though RONPO is strongest under local-RM robustness metrics. This means the paper should claim that RONPO improves heterogeneous reward-model robustness and shows favorable full-set external-judge transfer, but should not claim universal dominance on every hard disagreement subset.

## Safety-Conflict Final-Checkpoint Atom Ablation

This section records the 2026-07-01 final-checkpoint ablation for the safety-conflict RONPO experiment. It tests the paper's structural claim that the full atom adversary over `(objective, response)` can outperform an objective-only `k` adversary with the response atom fixed. This result is important, but it is currently diagnostic rather than supportive: under this single-seed final-checkpoint protocol, the `k`-only ablation is stronger than the full atom adversary.

### Ablation Provenance

| Item | Value |
| --- | --- |
| Experiment root | `/home/sjkim/MNPO/experiments/ronpo_safety_conflict_qwen25_1p5b_20260629` |
| Evaluation work directory | `/home/sjkim/MNPO/experiments/ronpo_safety_conflict_qwen25_1p5b_20260629/reward_eval_final_ablation_20260701_092724` |
| Result report | `reward_eval_final_ablation_20260701_092724/results/report.md` |
| Result CSVs | `results/{model_summary,per_objective_scores,pairwise_win_rates,collapse_diagnostics}.csv` |
| Evaluation split | `pairs/full_atom/test_merged_scores.jsonl` |
| Number of prompts | 620 |
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Full atom checkpoint | `/ext_hdd/sjkim/mnpo/experiments/ronpo_safety_conflict_qwen25_1p5b_20260629/outputs/ronpo-safe-full-s1_seed42/checkpoint-3152` |
| k-only checkpoint | `/ext_hdd/sjkim/mnpo/experiments/ronpo_safety_conflict_qwen25_1p5b_20260629/outputs/ronpo-safe-konly-s1_seed42/checkpoint-2227` |
| Generation config | vLLM, seed `42`, `temperature=0.7`, `top_p=0.9`, `max_tokens=512`, `dtype=bfloat16`, `attention_backend=XFORMERS`, `gpu_memory_utilization=0.85` |
| Evaluation objectives | Helpfulness, safety, brevity |
| Evaluators | Helpfulness: `Skywork/Skywork-Reward-V2-Llama-3.1-8B`; safety: `Qwen/Qwen3Guard-Gen-0.6B`; brevity: deterministic length-based normalized reward |
| Evaluation launcher | `scripts/run_ronpo_safety_final_ablation_eval_gpu0.sh` |

### Final-Checkpoint Ablation Table

Higher is better for reward metrics; lower is better for `Repeat>=20`. Normalized scores are computed within this three-model ablation set only and should not be mixed with the Skywork/Athene/ArmoRM stage-1 or stage-2 tables.

| Method | Checkpoint | n | Prompt-worst | Prompt-avg | Min objective | Win vs Base | Mean words | Repeat>=20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-only adversary | `checkpoint-2227` | 620 | **0.347** | **0.688** | **0.509** | **0.630** | 201.8 | 0.003 |
| Base | `Qwen2.5-1.5B-Instruct` | 620 | 0.202 | 0.499 | 0.471 | - | 176.1 | **0.000** |
| Full atom adversary `(k,a)` | `checkpoint-3152` | 620 | 0.075 | 0.374 | 0.209 | 0.414 | 348.3 | 0.005 |

Per-objective normalized scores:

| Method | Helpfulness | Safety | Brevity |
| --- | ---: | ---: | ---: |
| k-only adversary | **0.652** | **0.509** | **0.904** |
| Base | 0.471 | 0.504 | 0.523 |
| Full atom adversary `(k,a)` | 0.416 | 0.498 | 0.209 |

Direct pairwise win rates from the same scoring artifacts:

| Comparison | Helpfulness WR | Safety WR | Brevity WR |
| --- | ---: | ---: | ---: |
| Base vs Full atom `(k,a)` | 0.515 | 0.503 | 0.740 |
| k-only vs Base | 0.591 | 0.502 | 0.796 |
| Full atom `(k,a)` vs k-only | 0.360 | 0.494 | 0.144 |

### Ablation Interpretation

The final-checkpoint ablation does not support the current full atom adversary implementation as a stronger alternative to the objective-only `k` adversary. The `k`-only checkpoint is best on prompt-worst normalized reward (`0.347`), prompt-average normalized reward (`0.688`), minimum objective mean (`0.509`), and mean win rate against Base (`0.630`). The full `(k,a)` checkpoint is worse than Base on the aggregate metrics in this evaluation.

The main failure mode is length drift. Full atom `(k,a)` produces much longer responses than both Base and `k`-only (`348.3` mean words vs `176.1` and `201.8`) and has a very low brevity score (`0.209`). Helpfulness and safety also do not compensate for this: full atom scores `0.416` on helpfulness and `0.498` on safety, both below `k`-only. Pairwise results show the same pattern: full atom beats `k`-only on only `35.97%` of helpfulness comparisons, `49.44%` of safety comparisons, and `14.35%` of brevity comparisons.

Reviewer-facing implication: this result should be reported internally as a falsifiable ablation and should not be used to claim that response-atom adversarial selection improves the model-scale safety-conflict experiment. The defensible statement is that the toy separation supports the structural possibility of atom-level adversaries, but the current model-scale final-checkpoint run has not yet realized that advantage. Before including this as a main-paper positive result, rerun with multiple seeds and inspect checkpoint selection, objective scaling, adversary update dynamics, and length behavior. If the result remains unchanged, the paper should either frame the atom adversary as a theoretical/diagnostic extension or explicitly report the negative ablation in the appendix.

## Current GPU Availability Snapshot

This snapshot was collected read-only on 2026-06-22 around 22:29 KST.

| Host | GPU / node | Allowed | Idle | Evidence | Storage | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `odin2` | A100 GPU 0 | Do not use for new RONPO work | No | 3 samples showed 35-37 GiB used and 74-100% utilization; active Python processes on `cuda:0` | `/ext_hdd/sjkim` has about 766 GiB free | Occupied |
| `odin2` | A100 GPU 1 | Yes, based on current project usage context | Yes | 3 samples showed 4 MiB used, 0% utilization, no compute process | `/ext_hdd/sjkim` has about 766 GiB free | Candidate for evaluation or one 1.5B training run |
| `odin2` | A100 GPU 2 | Yes, based on current project usage context | Yes | 3 samples showed 4 MiB used, 0% utilization, no compute process | `/ext_hdd/sjkim` has about 766 GiB free | Candidate for evaluation or one 1.5B training run |
| MLXP `p-aipr` | `h200-03-w-aa21` | Project namespace accessible | Partially unknown | Running GPU requests visible in namespace: `appgen` 4 GPU, `mmpae` 2 GPU, `mnpo-ronpo-s2-h200-1g-r6` 1 GPU. Node capacity query is forbidden, so exact free GPU count cannot be proven from available permissions. | DDN `/data`: 80T total, 46T free | Keep current RONPO stage-2 job running; launch new H200 jobs only after scheduler availability is verified |
| MLXP `p-aipr` | `h200-03-w-abf8` | Project namespace accessible | No for new work by namespace evidence | `surgical-pod-01` requests 8 GPU on this node; node capacity query is forbidden | DDN `/data`: 80T total, 46T free | Treat as occupied |

## Recommended Next Experiments

Reference used for experiment design: `references/mnpo.pdf` (ICLR 2026, Multiplayer Nash Preference Optimization). The MNPO paper motivates three reviewer-relevant axes: multiplayer competition instead of a single opponent, heterogeneous preference sources, and iterative online training. Based on that, the highest-value RONPO experiments are the ones that isolate robustness under reward disagreement and stage-wise improvement, not another generic ablation.

### 1. Reward-Model-Independent Stage-2 Evaluation

The resumed RONPO stage-2 sanity table now supports a local-RM advantage. The next priority is to verify that the advantage transfers to judge-style and rule-based instruction-following benchmarks that are not the training reward models.

| Check | Why it matters | Suggested action |
| --- | --- | --- |
| GPT-5.5 pairwise judge on held-out UltraFeedback prompts | Tests preference quality with an external LLM judge rather than the training RMs | Completed for Base, HT-MNPO S2 players, and RONPO S2 checkpoints |
| Arena-Hard / MT-Bench | Standard public preference/chat judge benchmarks | Still useful as an additional benchmark if compute/API budget allows; do not conflate with the completed held-out-prompt GPT-5.5 judge table |
| IFEval | Checks rule-based instruction following without an external judge | Completed on 2026-06-26; report RONPO's preservation relative to HT-MNPO, not superiority over Base |
| Stage-2 pairwise matrix | Shows whether RONPO beats each HT-MNPO S2 player directly, not only vs Base | Completed for local RM; add full matrix and high-disagreement matrix to the appendix |
| Bootstrap intervals | Makes the 647-prompt local RM table harder to attack statistically | Completed for stage-2 local RM full set and high-disagreement subset |

This should be addressed before making broad reward-model-independent claims in the paper. The local-RM stage-2 claim is now supported by the 2026-06-25 resumed checkpoint sanity evaluation.

### 2. High-Disagreement Subset Evaluation

Completed on 2026-06-26 for the resumed stage-2 local-RM evaluation. The stress subset ranks prompts by cross-objective pairwise preference disagreement and reports the same normalized reward and win-rate metrics on the top 25% disagreement subset.

This is the cleanest experiment to show RONPO's advantage over HT-MNPO. The claim is narrower and stronger: when preference sources conflict, a robust multi-objective policy maintains a higher worst-objective floor than single-oracle policies.

### 3. Full Pairwise Tournament Matrix

For stage 1 and stage 2 separately, build a pairwise win-rate matrix among Base, HT-MNPO players, SPPO-avg, INPO-avg, and RONPO. Report:

| Metric | Reason |
| --- | --- |
| Average pairwise WR across opponents | Shows global dominance, not only improvement over Base |
| Worst-opponent WR | Captures exploitability-like robustness |
| Worst-objective pairwise WR | Captures heterogeneous reward robustness |

This is aligned with the game-theoretic framing in the MNPO paper and is harder to attack than only reporting WR vs Base.

### 4. Stage-Wise Improvement Curve

Evaluate RONPO and HT-MNPO checkpoints at stage 1 and stage 2 under the same table format:

```text
Base -> Stage 1 -> Stage 2
Avg norm, Worst norm, Std norm, Avg WR, Worst WR
```

The expected useful claim is not simply "stage 2 is higher"; it is that RONPO improves or preserves `Worst norm` and reduces cross-objective spread, while HT-MNPO can improve its own oracle but remains more brittle across other objectives.

### 5. Reward-Model-Independent Evaluation

The current table is objective-aligned local RM evaluation. To survive top-tier review, add at least one reward-model-independent axis. The MNPO paper reports instruction-following and preference-alignment benchmarks such as AlpacaEval 2, Arena-Hard, and MT-Bench, plus general reasoning/coding suites. For RONPO, the minimum defensible add-on is:

| Evaluation | Why it matters | Suggested placement |
| --- | --- | --- |
| AlpacaEval 2 or Arena-Hard | External judge-style preference alignment | Run after final stage-2 checkpoints; use H200 if available, otherwise odin2 GPU 1 or 2 |
| MT-Bench | Conversational instruction following | Same final checkpoints only |
| MMLU/GSM8K/HumanEval subset if available through EvalScope | Checks that robust preference optimization does not destroy base capability | Lower priority than preference benchmarks |

### 6. Fairness Guard for SPPO/INPO

SPPO-avg and INPO-avg are weak in the current stage-1 run. A reviewer may suspect implementation or undertraining rather than a conceptual limitation. If compute remains after debugging the stage-2 RONPO issue, run one of:

| Option | Cost | Reviewer value |
| --- | --- | --- |
| SPPO/INPO stage-2 using the same average normalized oracle | Medium | Strongest fairness check against "one iteration only" criticism |
| Re-evaluate SPPO/INPO with multiple generation seeds and bootstrap confidence intervals | Low to medium | Shows current weakness is not only sampling noise |
| Single-oracle SPPO/INPO for Skywork only | Medium | Clarifies whether average-oracle construction caused degradation |

This should not displace the stage-2 RONPO diagnostic work, but it is useful if SPPO/INPO remain in the main paper table.

## Suggested Paper Caption

Local reward-model evaluation on the 647-prompt held-out UltraFeedback stage-1 split. Each model response is generated with the same vLLM sampling configuration and scored by Skywork, Athene, and ArmoRM reward models. We report per-prompt min-max normalized reward and pairwise win rate against the base model. RONPO improves both average and worst-objective performance over homogeneous-oracle baselines, including single-oracle HT-MNPO and average-oracle SPPO/INPO.

## Missing Evidence Before Final Submission

- Re-evaluate RONPO `checkpoint-1184` if that is the checkpoint used in the final paper.
- Use the 2026-06-25 resumed RONPO S2 `checkpoint-2457` table, not the older 2026-06-23 pathological RONPO S2 table, for any stage-2 local-RM claim.
- GPT-5.5 pairwise judge evaluation is completed for stage-2 held-out prompts; optionally add standard Arena-Hard or MT-Bench if the final submission needs a public benchmark in addition to this controlled held-out-prompt judge table.
- IFEval is now completed for stage-2; use the clean `Num=541` baseline rerun and exclude the invalid initial `Num=539` baseline artifact.
- Stage-2 local-RM bootstrap intervals are complete; add corresponding confidence intervals for the stage-1 table if that table remains central.
- Repeat generation over multiple decoding seeds if compute allows.
- Add at least one external or held-out evaluation axis, such as a held-out reward model, human preference, or general instruction-following benchmark.
- If SPPO/INPO remain in the main paper table, add a short implementation note explaining the average-oracle construction and why it is the fairest homogeneous-oracle instantiation in this multi-objective setting.
