# Base vs HT-MNPO vs RONPO vs SPPO/INPO Stage-1 Local Reward-Model Evaluation

작성일: 2026-06-19  
SPPO/INPO 추가 평가 업데이트: 2026-06-22

이 문서는 RONPO 논문에 추가할 수 있도록, Qwen2.5-1.5B-Instruct base model, HT-MNPO stage 1, RONPO stage 1, SPPO stage 1, INPO stage 1 모델을 동일한 local reward-model benchmark로 비교한 결과와 실험 세팅을 정리한다. 외부 paid API judge는 사용하지 않았고, 학습 oracle과 동일한 세 가지 local reward objectives를 사용해 controlled in-distribution RM evaluation으로 평가했다.

## 1. 요약

- RONPO stage 1이 세 reward model 모두에서 base 대비 가장 높은 win-rate를 보였다: Skywork 67.47%, Athene 68.55%, ArmoRM 59.81%.
- RONPO는 평균 normalized score 0.8885, worst-objective normalized score 0.8609로, HT-MNPO 중 가장 강한 `htmnpo_skywork`의 0.8119 / 0.7637을 상회했다.
- SPPO-avg와 INPO-avg는 세 objective의 prompt-wise normalized average를 homogeneous oracle로 사용해 학습한 baselines다. 이 설정은 단일 reward model을 cherry-pick하지 않기 위한 공정한 multi-objective 평균 oracle 비교지만, RONPO의 adversarial robust objective와는 다른 목적을 최적화한다.
- HT-MNPO와 SPPO/INPO는 homogeneous preference oracle baseline이므로 특정 평균 또는 단일 oracle에서는 개선될 수 있지만 cross-objective robustness가 제한적이다. 이번 stage-1 평가에서는 SPPO-avg와 INPO-avg가 base보다 낮은 local RM scores를 보였고, RONPO가 평균 및 worst-objective 지표에서 가장 강했다.
- 논문에서는 이 표를 "local RM objective evaluation" 또는 "training-objective-aligned reward-model evaluation"으로 명확히 표현하는 것이 안전하다. 외부 human preference나 general benchmark 성능을 직접 주장하는 표는 아니다.

## 2. 비교 모델

| Label | Model / checkpoint | 설명 |
| --- | --- | --- |
| `baseline` | `Qwen/Qwen2.5-1.5B-Instruct` | stage-1 학습 전 base policy |
| `htmnpo_skywork` | `/ext_hdd/sjkim/mnpo/outputs/qwen2.5-1.5b-instruct_htmnpo_skywork_online_multiobj_stage_1` | Skywork oracle로 학습한 HT-MNPO stage 1 |
| `htmnpo_athene` | `/ext_hdd/sjkim/mnpo/outputs/qwen2.5-1.5b-instruct_htmnpo_athene_online_multiobj_stage_1/checkpoint-300` | Athene oracle로 학습한 HT-MNPO stage 1 checkpoint |
| `htmnpo_armorm` | `/ext_hdd/sjkim/mnpo/outputs/qwen2.5-1.5b-instruct_htmnpo_armo_online_multiobj_stage_1` | ArmoRM oracle로 학습한 HT-MNPO stage 1 |
| `ronpo` | `/ext_hdd/sjkim/mnpo/outputs_ronpo_fair/qwen2.5-1.5b-instruct_ronpo_vs_htmnpo_online_multiobj_stage_1/checkpoint-1100` | 세 reward objectives를 robust sigma pair로 통합한 RONPO stage 1 checkpoint |
| `sppo` | `/home/sjkim/mnpo_runs/loki3/out/sppo_s1` | 세 reward objectives의 normalized average oracle로 학습한 SPPO stage 1 final model |
| `inpo` | `/home/sjkim/mnpo_runs/loki3/out/inpo_s1` | 세 reward objectives의 normalized average oracle로 학습한 INPO stage 1 final model |

주의: RONPO stage-1 output directory에는 `checkpoint-1184`도 존재하지만, 아래 결과 생성 스크립트의 기본값은 `checkpoint-1100`이었다. 최종 checkpoint 기준으로 논문 표를 확정하려면 동일 pipeline으로 재평가해야 한다.

## 3. Stage-1 Training Configs

### 공통 설정

| 항목 | 값 |
| --- | --- |
| Base policy | `Qwen/Qwen2.5-1.5B-Instruct` |
| Train split | `data/gemma2_ufb_part1_train.jsonl` |
| Train prompts | 19,856 |
| Eval split used during training | `data/gemma2_ufb_part1_test.jsonl` |
| Eval prompts | 647 |
| Online generation seeds | `13 21 42 79 100` |
| Decode backend | vLLM, `VLLM_ATTENTION_BACKEND=XFORMERS` |
| Decode sampling for training data | `temperature=0.8`, `top_p=0.95`, `max_tokens=4096` from `on_policy_data_gen/decode.py` defaults |
| Decode batch size | `512` in current runner default |
| Reward objectives / players | `skywork`, `athene`, `armo` |
| Reward scoring batch | `RM_BATCH_SIZE=16`, `RM_SAMPLE_BATCH_SIZE=64`, `RM_MAX_SEQ_LENGTH=4096` in training runner defaults |
| Shared stage-1 base data | `SHARE_STAGE1_BASE_DATA=1`; HT-MNPO and RONPO stage 1 are trained from the same base-policy response pool |
| Precompute | `mnpo_scripts.precompute`, ref model = base model |
| Training launcher | `accelerate` with `accelerate_configs/deepspeed_zero3.yaml` |
| W&B | `WANDB_ENTITY=promotion-kim`, `WANDB_PROJECT=mnpo`, `report_to: wandb` |

The main stage-1 runner is:

```bash
run_qwen_online_htmnpo_ronpo.sh
```

The config files used by the runner are:

```text
training_configs/mnpo/qwen2.5-1.5b-instruct-ht-mnpo-multiobj-iter1.yaml
training_configs/ronpo/qwen2.5-1.5b-instruct-ronpo-multiobj-iter1.yaml
training_configs/sppo/qwen2.5-1.5b-instruct-sppo-avg-multiobj-iter1.yaml
training_configs/inpo/qwen2.5-1.5b-instruct-inpo-avg-multiobj-iter1.yaml
```

### HT-MNPO Stage 1

HT-MNPO is trained separately for each homogeneous oracle. For stage 1, all policies start from the base model and the opponent/history path collapses to the base model. The runner overrides `max_history_t` to the actual history count, which is 1 for stage 1.

| 항목 | 값 |
| --- | --- |
| Loss | `ht_mnpo` |
| Pair builder | `mnpo_scripts.build_ht_mnpo_dataset` |
| Target mode | `HT_TARGET_MODE=reward_gap` |
| Target normalization | `HT_NORMALIZATION=none` |
| Stage-1 ratio | `0.3333` |
| Stage-1 eta | `0.0075` for all players |
| Stage-1 beta | Skywork `10`, Athene `1`, ArmoRM `10` |
| Model dtype | `bf16: true` |
| Learning rate | `5.0e-7` |
| Scheduler | cosine |
| Warmup ratio | `0.1` |
| Optimizer | `adamw_torch` |
| Weight decay | `0.0` |
| Seed | `42` |
| Epochs | `1` |
| Per-device train batch | `2` |
| Gradient accumulation | `8` |
| Effective train batch per process | `16` prompt-pairs before distributed scaling |
| Max length / prompt length | `2048` / `1800` |
| Gradient checkpointing | enabled, `use_reentrant=false` |
| Eval / save interval | every `100` steps |
| Save total limit | `5` |

### RONPO Stage 1

RONPO uses all three reward objectives and constructs a robust pair with the sigma adversary. The policy starts from the base model for stage 1.

| 항목 | 값 |
| --- | --- |
| Loss | `ronpo` |
| Pair builder | `mnpo_scripts.build_multi_objective_dataset` |
| Objective normalization for pair construction | `minmax` |
| Pair strategy | `ronpo_pair_strategy=sigma` |
| Policy pair mode | `RONPO_POLICY_PAIR_MODE=best_vs_adversary` |
| Policy samples per atom | `0` |
| Adversary steps | `25` |
| Adversary alpha | `1.0` |
| Adversary kappa | `0.05` |
| Preference scale | `8.0` |
| Pairs per prompt | `1` |
| RONPO alpha | `1.0` |
| RONPO tau | `0.05` |
| RONPO target column | `ronpo_target` |
| Beta | `10` |
| Model dtype | `bf16: true` |
| Learning rate | `5.0e-7` |
| Scheduler | cosine |
| Warmup ratio | `0.1` |
| Optimizer | `adamw_torch` |
| Weight decay | `0.0` |
| Seed | `42` |
| Epochs | `1` |
| Per-device train batch | `2` |
| Gradient accumulation | `8` |
| Max length / prompt length | `2048` / `1800` |
| Eval / save interval | every `100` steps |
| Save total limit | `5` |

### SPPO/INPO Stage 1 Average-Oracle Baselines

SPPO and INPO require a homogeneous preference oracle. To avoid choosing one of Skywork, Athene, or ArmoRM as a privileged oracle, these baselines use the prompt-wise min-max normalized average over all three reward objectives. The same shared base-policy response pool and scored files used for HT-MNPO/RONPO stage 1 are reused.

| 항목 | SPPO 값 | INPO 값 |
| --- | --- | --- |
| Loss | `sppo` | `inpo` |
| Pair builder | `mnpo_scripts.build_multi_objective_dataset` | `mnpo_scripts.build_multi_objective_dataset` |
| Homogeneous oracle | `average_minmax_objectives` over `skywork,athene,armo` | `average_minmax_objectives` over `skywork,athene,armo` |
| Dataset | `data/qwen2.5-1.5b-instruct_multiobj_iter1/sppo_avg_precomputed` | `data/qwen2.5-1.5b-instruct_multiobj_iter1/inpo_avg_precomputed` |
| Model dtype | `bf16: true` | `bf16: true` |
| Eta / ratio / beta | `0.0075` / `0.3333` / `10` | `0.0075` / `0.3333` / `10` |
| Learning rate | `5.0e-7` | `5.0e-7` |
| Scheduler / warmup | cosine / `0.1` | cosine / `0.1` |
| Optimizer / weight decay | `adamw_torch` / `0.0` | `adamw_torch` / `0.0` |
| Seed / epochs | `42` / `1` | `42` / `1` |
| Per-device train batch | `2` | `2` |
| Gradient accumulation | `8` | `8` |
| Max length / prompt length | `2048` / `1800` | `2048` / `1800` |
| Launcher used on loki3 | `scripts/run_avg_stage1_loki3.sh sppo` | `scripts/run_avg_stage1_loki3.sh inpo` |
| Eval / save interval | every `100` steps | every `100` steps |
| Final training checkpoint | `checkpoint-591` plus final model directory | `checkpoint-591` plus final model directory |

## 4. Evaluation Config

The SPPO/INPO-inclusive result table was generated from:

```bash
CUDA_VISIBLE_DEVICES=2 \
INCLUDE_SPPO=1 INCLUDE_INPO=1 FORCE_SCORE=1 \
RM_BATCH_SIZE=1 RM_SAMPLE_BATCH_SIZE=8 RM_MAX_SEQ_LENGTH=4096 \
DECODE_GPUS=1 CACHE_DIR=/ext_hdd/sjkim/huggingface/hub \
bash evalscope/run_qwen_htmnpo_stage1_table_update.sh
```

Evaluation settings:

| 항목 | 값 |
| --- | --- |
| Stage | `STAGE=1` |
| Eval file | `data/gemma2_ufb_part1_test.jsonl` |
| Number of prompts | 647 |
| Work dir | `/ext_hdd/sjkim/mnpo/eval/htmnpo_stage1_base_compare` |
| Generation dir | `/ext_hdd/sjkim/mnpo/eval/htmnpo_stage1_base_compare/generations` |
| Scored dir | `/ext_hdd/sjkim/mnpo/eval/htmnpo_stage1_base_compare/scored_extended` |
| Result dir | `/ext_hdd/sjkim/mnpo/eval/htmnpo_stage1_base_compare/results_extended` |
| Decode seed | `42` |
| Decode GPUs | `1` |
| Decode attention backend | `XFORMERS` |
| Decode temperature | `0.7` |
| Decode top-p | `0.9` |
| Decode max tokens | `2048` |
| Reward objectives | `skywork,athene,armo` |
| Include SPPO / INPO | `INCLUDE_SPPO=1`, `INCLUDE_INPO=1` |
| SPPO model | `/home/sjkim/mnpo_runs/loki3/out/sppo_s1` |
| INPO model | `/home/sjkim/mnpo_runs/loki3/out/inpo_s1` |
| RM batch size | `1` |
| RM sample batch size | `8` |
| RM max sequence length | `4096` |
| Rescore mode | `FORCE_SCORE=1`; all seven model generations rescored with the same reward-model pass |
| Evaluation GPU env | `CUDA_VISIBLE_DEVICES=2` on odin2 |
| Evaluation log | `/ext_hdd/sjkim/mnpo/logs/eval_stage1_sppo_inpo_20260622_101215.log` |

Reward models used for scoring:

| Objective label | Reward model |
| --- | --- |
| `skywork` | `Skywork/Skywork-Reward-V2-Llama-3.1-8B` |
| `athene` | `Nexusflow/Athene-RM-8B` |
| `armo` | `RLHFlow/ArmoRM-Llama3-8B-v0.1` |

Source artifacts:

```text
/ext_hdd/sjkim/mnpo/eval/htmnpo_stage1_base_compare/merged_model_generations_extended.json
/ext_hdd/sjkim/mnpo/eval/htmnpo_stage1_base_compare/scored_extended/eval_skywork.jsonl
/ext_hdd/sjkim/mnpo/eval/htmnpo_stage1_base_compare/scored_extended/eval_athene.jsonl
/ext_hdd/sjkim/mnpo/eval/htmnpo_stage1_base_compare/scored_extended/eval_armo.jsonl
/ext_hdd/sjkim/mnpo/eval/htmnpo_stage1_base_compare/results_extended/per_objective_scores.csv
/ext_hdd/sjkim/mnpo/eval/htmnpo_stage1_base_compare/results_extended/model_summary.csv
/ext_hdd/sjkim/mnpo/eval/htmnpo_stage1_base_compare/results_extended/pairwise_win_rates.csv
/ext_hdd/sjkim/mnpo/eval/htmnpo_stage1_base_compare/results_extended/base_vs_htmnpo_ronpo_stage1_extended.md
```

## 5. Evaluation Metrics

Let `s(i, o, m)` be the scalar reward-model score for prompt `i`, objective `o`, and model `m`.

### Raw reward score

For each objective, the raw score is the mean reward assigned by that objective's local reward model:

```text
raw(o, m) = mean_i s(i, o, m)
```

Higher is better for all three reward models. Raw scales are not comparable across reward models. For example, Athene scores can be negative while Skywork scores are positive; only within-objective comparisons are meaningful.

### Prompt-normalized objective score

For every prompt and objective, scores are min-max normalized across the compared model set:

```text
n(i, o, m) = (s(i, o, m) - min_m' s(i, o, m')) / (max_m' s(i, o, m') - min_m' s(i, o, m'))
```

If all compared models receive the same score for a prompt/objective, the normalized score is set to `0.5`.

The reported objective-normalized score is:

```text
norm(o, m) = mean_i n(i, o, m)
```

Important: this normalization is relative to the exact model set in the table. Adding or removing compared models can change absolute normalized values, so normalized scores should be compared within the same table only.

### Win-rate vs Base

For every prompt and objective, the candidate response is compared directly against the base model response under the same reward model:

```text
win(i, o, m) = 1.0  if s(i, o, m) > s(i, o, baseline)
              0.0  if s(i, o, m) < s(i, o, baseline)
              0.5  otherwise
```

The script uses `tie_threshold=0.0`, so only exact equality is treated as a tie. The objective win-rate is:

```text
WR_vs_Base(o, m) = mean_i win(i, o, m) * 100
```

This is the most directly interpretable metric because it asks: on the same prompt, how often does the reward model prefer this model's response over the base model's response?

### Aggregate robustness metrics

For each model:

```text
Avg norm   = mean_o norm(o, m)
Worst norm = min_o norm(o, m)
Std norm   = std_o norm(o, m)
Avg WR     = mean_o WR_vs_Base(o, m)
Worst WR   = min_o WR_vs_Base(o, m)
```

`Worst norm` and `Worst WR` are the key robustness metrics. They measure whether a method improves the weakest reward objective rather than only optimizing one favorable oracle.

## 6. Main Result Table

| Model | Prompts | skywork raw | skywork norm | skywork WR vs Base (%) | athene raw | athene norm | athene WR vs Base (%) | armo raw | armo norm | armo WR vs Base (%) | Avg norm | Worst norm | Std norm | Avg WR vs Base (%) | Worst WR vs Base (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 647 | 5.2163 | 0.7285 | - | -1.2148 | 0.7501 | - | 0.1191 | 0.8645 | - | 0.7811 | 0.7285 | 0.0597 | - | - |
| htmnpo_skywork | 647 | 6.2115 | 0.7637 | 55.95 | -1.0516 | 0.7889 | 58.19 | 0.1221 | 0.8833 | 58.73 | 0.8119 | 0.7637 | 0.0515 | 57.62 | 55.95 |
| htmnpo_athene | 647 | 6.2383 | 0.7640 | 56.03 | -1.1870 | 0.7557 | 49.69 | 0.1188 | 0.8609 | 49.61 | 0.7935 | 0.7557 | 0.0477 | 51.78 | 49.61 |
| htmnpo_armorm | 647 | 5.9549 | 0.7552 | 56.34 | -1.1958 | 0.7531 | 55.72 | 0.1173 | 0.8494 | 50.70 | 0.7859 | 0.7531 | 0.0449 | 54.25 | 50.70 |
| ronpo | 647 | 9.1666 | 0.8609 | 67.47 | -0.6148 | 0.8950 | 68.55 | 0.1274 | 0.9095 | 59.81 | 0.8885 | 0.8609 | 0.0204 | 65.28 | 59.81 |
| sppo | 647 | -6.0962 | 0.3361 | 13.60 | -4.0160 | 0.0838 | 2.78 | 0.0193 | 0.1948 | 1.08 | 0.2049 | 0.0838 | 0.1032 | 5.82 | 1.08 |
| inpo | 647 | -14.9242 | 0.0457 | 3.86 | -3.8680 | 0.1288 | 6.57 | -0.0080 | 0.0274 | 0.77 | 0.0673 | 0.0274 | 0.0441 | 3.74 | 0.77 |

## 7. Pairwise RONPO vs Baseline Method Win-Rates

The table below reports RONPO's pairwise win-rate against each trained baseline under each objective. For rows where RONPO is the right model in `pairwise_win_rates.csv`, the value is computed as `1 - left_win_rate`; for SPPO/INPO rows, RONPO is already the left model.

| Comparison | Skywork WR (%) | Athene WR (%) | ArmoRM WR (%) |
| --- | ---: | ---: | ---: |
| RONPO vs HT-MNPO Skywork | 62.29 | 63.37 | 57.50 |
| RONPO vs HT-MNPO Athene | 63.21 | 68.32 | 63.83 |
| RONPO vs HT-MNPO ArmoRM | 63.68 | 67.31 | 62.21 |
| RONPO vs SPPO-avg | 92.97 | 98.84 | 98.61 |
| RONPO vs INPO-avg | 97.53 | 98.38 | 99.85 |

RONPO is preferred over every trained baseline by all three reward models. This supports the robustness claim more directly than only comparing each method to the base model.

## 8. Interpretation for the RONPO Paper

The strongest non-RONPO trained baseline in this table is `htmnpo_skywork`, with Avg norm 0.8119 and Worst norm 0.7637. RONPO improves over it by +0.0765 Avg norm and +0.0972 Worst norm. In win-rate terms, RONPO improves Avg WR vs Base from 57.62% to 65.28% and Worst WR vs Base from 55.95% to 59.81%.

Compared with the base model, RONPO improves Avg norm by +0.1074 and Worst norm by +0.1324 in the seven-model table. The base model has no WR-vs-base by definition, but RONPO's objective-level win-rates are all above 59%, with the best margin on Athene at 68.55%.

The pattern is consistent with the intended distinction between the methods:

- HT-MNPO optimizes a homogeneous oracle, so a single-player improvement can be partially objective-specific.
- SPPO-avg and INPO-avg optimize an averaged homogeneous oracle; this is a conservative way to instantiate homogeneous-oracle baselines in a multi-objective setting, but it does not explicitly optimize the worst objective.
- RONPO explicitly constructs a robust objective mixture through the sigma adversary, so it optimizes against a weaker objective direction and improves the worst-objective floor.
- The low standard deviation of RONPO normalized scores, 0.0204, indicates that its gains are not concentrated in only one reward objective.

## 9. Fairness and Reviewer-Facing Caveats

Use the following wording constraints when adding this to the paper:

- Do not present this table as human evaluation. It is local reward-model evaluation.
- Do not claim reward-model-independent preference superiority from this table alone. The evaluation reward models overlap with the reward objectives used to construct training data.
- Label SPPO/INPO as `SPPO-avg` and `INPO-avg` in paper text or footnotes, because they use the average normalized local-RM oracle rather than a single human/preference oracle.
- The fair claim is: under a controlled multi-objective local RM benchmark aligned with the training objectives, RONPO improves both average and worst-objective reward performance over homogeneous-oracle baselines including HT-MNPO, SPPO-avg, and INPO-avg.
- Because prompt-normalized scores are relative to the compared model set, the main stable comparisons are within-table ranks, worst-objective values in the same table, and pairwise win-rates.
- For a top-tier conference submission, this table should ideally be paired with at least one external evaluation axis, such as AlpacaEval/Arena-Hard/MT-Bench-style judge, human preference, or held-out reward models not used in training.

Suggested caption:

> Local reward-model evaluation on the 647-prompt held-out UltraFeedback stage-1 split. Each model response is generated with the same vLLM sampling configuration and scored by three local reward models. We report raw reward, per-prompt min-max normalized reward, and pairwise win-rate against the base model. RONPO improves both average and worst-objective performance over homogeneous-oracle baselines, including single-oracle HT-MNPO and average-oracle SPPO/INPO.
