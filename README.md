# MNPO Training Pipeline

🔔 News

- (2026-01) The paper has been accepted by ICLR 2026!
- (2025-11) The codebase has been updated to include full support for HT-MNPO 

---

This repository packages the full iterative **Multiplayer Nash Preference Optimization (MNPO)** workflow that we used to fine-tune instruction-following language models with on-policy preference data. It bundles scripts for dataset preparation, preference data generation, annotation, and multi-GPU MNPO training so you can reproduce or adapt our alignment pipeline end-to-end.

## Repository Layout

| Path                  | Description                                                                                                                                             |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mnpo_scripts/`       | MNPO orchestration code: configuration dataclasses, precomputation loop, MNPO trainer, and CLI entrypoints such as `run_mnpo.py` and `split_dataset.py`. |
| `on_policy_data_gen/` | Tools for generating and annotating on-policy preference pairs (decoding, post-processing, reward model annotation).                                    |
| `alignment/`          | Shared alignment helpers for data loading, model utilities, and release tooling.                                                                        |
| `training_configs/`   | MNPO hyperparameter YAMLs for each training stage (e.g., `gemma-2-9b-it-mnpo-iter*.yaml`).                                                              |
| `accelerate_configs/` | Launch configurations for Accelerate, DeepSpeed ZeRO, and FSDP setups.                                                                                  |
| `scripts/`            | Auxiliary utilities and launch helpers.                                                                                                                 |
| `run.sh`              | Example shell pipeline that ties together dataset splitting, on-policy data refresh, precomputation, and training loops.                                |
| `evalscope/`          | Example code for evaluation.                                                                                                                            |

## Environment Setup
We separate environments for model training and large-scale decoding. On this server, use **Python 3.10** with CUDA 12.1 PyTorch wheels. The server reports a CUDA 12.0-era driver stack, and PyTorch does not publish CUDA 12.0 wheels; CUDA 12.1 wheels are the closest CUDA 12.x build and are usually the most practical choice here.

Before installing, confirm that the NVIDIA driver is visible from your login shell:

```bash
nvidia-smi
```

If this command fails, fix the driver/session visibility first. The Python environments can still be created, but GPU training or vLLM decoding will not run until `nvidia-smi` works.

<details>
<summary><code>mnpo_train</code> </summary>

```bash
conda create -n mnpo_train python=3.10 -y
conda activate mnpo_train

python -m pip install --upgrade pip setuptools wheel

pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

pip install \
  numpy==1.26.4 \
  accelerate==0.29.2 \
  deepspeed==0.15.4 \
  transformers==4.44.2 \
  trl==0.9.6 \
  datasets==2.18.0 \
  huggingface-hub==0.23.2 \
  peft==0.7.1 \
  wandb

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The MNPO training configs use `attn_implementation: eager`, so FlashAttention is not required for the default training path. If you change a config to `flash_attention_2` or use a reward model script that hard-codes FlashAttention, install it after PyTorch with:

```bash
pip install ninja packaging
pip install flash-attn --no-build-isolation
```

This optional FlashAttention build requires a working CUDA toolkit with `nvcc`.
</details>

<details>
<summary><code>mnpo_infer</code> </summary>

```bash
conda create -n mnpo_infer python=3.10 -y
conda activate mnpo_infer

python -m pip install --upgrade pip setuptools wheel

pip install torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121

pip install \
  vllm==0.5.1 \
  "transformers<4.54.0" \
  datasets==2.18.0 \
  numpy==1.26.4 \
  more_itertools

pip install \
  https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.0.post2/flashinfer_python-0.2.0.post2%2Bcu121torch2.3-cp310-cp310-linux_x86_64.whl

python -c "import torch, vllm; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), vllm.__version__)"
```

`on_policy_data_gen/decode.py` sets `VLLM_ATTENTION_BACKEND=FLASHINFER` for Gemma 2 decoding, so the `flashinfer_python` wheel is installed explicitly in `mnpo_infer`. If vLLM fails to load FlashInfer on your GPU, remove or change that environment-variable line and let vLLM choose its default backend.
</details>

Set `PYTHONPATH` to the repository root before running any module entrypoints:

```bash
export PYTHONPATH=$(pwd)
```

## End-to-End Workflow
The `run.sh` script demonstrates a three-iteration MNPO curriculum and can be adapted to your infrastructure.

1. **Initial dataset split** – `mnpo_scripts.split_dataset` shards the base preference dataset into per-iteration train/test JSONL files to avoid leakage between stages.
2. **On-policy generation (for iteration &gt; 1)** – `on_policy_data_gen.decode` samples multiple responses per prompt, `post_process` filters identical answers, and `reward_model_annotate` scores them with a reward model to produce MNPO-ready pairs. For reproduction and experiments, we included both reward model and preference model annotation.
3. **Precomputation** – `mnpo_scripts.precompute` computes log-probabilities, normalizers, and history buffers used by MNPO training. Previous stage checkpoints can be chained via the `--history_paths` argument.
4. **Training** – `mnpo_scripts.run_mnpo` launches the actual MNPO updates using Accelerate/DeepSpeed and the YAML config for the current iteration. Outputs are written under `outputs/` and fed into the next iteration.

Adjust the variables at the top of run.sh, customize the training and accelerate configurations to match your setup, then execute:
```bash
bash run_iter1/2/3.sh
```

`run_iter1/2/3.sh` demonstrates a pipeline using a single reward model. Other reward models can be substituted by modifying the corresponding reward-model-annotation section.

## Qwen2.5 INPO Quickstart

For faster INPO iteration, use `Qwen/Qwen2.5-1.5B-Instruct` instead of Gemma 2 9B. Do not reuse Gemma precomputed logps: `reference_*` and `history*` columns must be regenerated with the Qwen tokenizer/model.

```bash
conda activate mnpo_train
```

Run precompute and training end-to-end:

```bash
WANDB_ENTITY=promotion-kim \
WANDB_PROJECT=mnpo \
bash run_qwen_inpo_iter1.sh
```

To save checkpoints on the external drive, run training with an `output_dir` override:

```bash
NCCL_P2P_DISABLE=1 \
NCCL_CUMEM_ENABLE=0 \
NCCL_IB_DISABLE=1 \
NCCL_SOCKET_IFNAME=enp3s0f0 \
TRITON_CACHE_DIR=/ext_hdd/sjkim/mnpo/triton_cache \
WANDB_ENTITY=promotion-kim \
WANDB_PROJECT=mnpo \
accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml \
-m mnpo_scripts.run_mnpo \
training_configs/inpo/qwen2.5-1.5b-instruct-inpo-iter1.yaml \
--output_dir=/ext_hdd/sjkim/mnpo/qwen2.5-1.5b/inpo_stage_1
```

Useful paths:

- Qwen training config: `training_configs/inpo/qwen2.5-1.5b-instruct-inpo-iter1.yaml`
- Qwen precomputed dataset: `data/qwen2.5-1.5b-instruct_iter1_precomputed`
- Default output dir: `outputs/qwen2.5-1.5b-instruct_inpo_stage_1`
- W&B project: `https://wandb.ai/promotion-kim/mnpo`

If the dataloader fails with `TypeError: 'NoneType' object cannot be interpreted as an integer`, remove stale tokenization cache files under `data/qwen2.5-1.5b-instruct_iter1_precomputed/*/cache-*.arrow` and rerun. This usually means an old cache was created before Qwen's missing `bos_token_id` handling was fixed.

## Qwen2.5 Online HT-MNPO vs RONPO Training

The recommended small-model experiment now uses `Qwen/Qwen2.5-1.5B-Instruct` and compares **HT-MNPO** against **RONPO** in the same heterogeneous online setting. HT-MNPO follows the upstream MNPO implementation: each player (`skywork`, `athene`, `armo`) builds its own winning/losing pairs from its own reward model, then trains with the standard MNPO loss while other player checkpoints are passed through `history_paths` as opponents. RONPO uses the same generated response pool and reward scores, but updates a prompt-level adversary distribution over `(objective, adversarial response)` atoms and trains with `ronpo_weight`.

Start with stage 1 before launching all stages. This verifies generation, reward scoring, pair construction, precompute, and training without committing to the full multi-stage run.

```bash
cd /home/sjkim/MNPO
conda activate mnpo_train
export PYTHONPATH=$(pwd)

export CUDA_VISIBLE_DEVICES=0,1,2
export RM_GPUS=0,1,2
unset PYTHON_RM
export PYTHON_RM_SKYWORK=/home/sjkim/anaconda3/envs/mnpo_infer/bin/python
export PYTHON_RM_ATHENE=/home/sjkim/anaconda3/envs/mnpo_infer/bin/python
export PYTHON_RM_ARMO=/home/sjkim/anaconda3/envs/mnpo_train/bin/python
export EXT_CACHE_ROOT=/ext_hdd/sjkim
export OUTPUT_ROOT=/ext_hdd/sjkim/mnpo/outputs
export STAGES="1"
export PLAYERS="skywork,athene,armo"
export RUN_HTMNPO=1
export RUN_RONPO=1
export NUM_PROCESSES=1
export DECODE_GPUS=1
export RM_BATCH_SIZE=16
export RM_SAMPLE_BATCH_SIZE=64

bash ./run_qwen_online_htmnpo_ronpo.sh
```

Use `bash ./run_qwen_online_htmnpo_ronpo.sh` unless the script has execute permission. If you prefer `./run_qwen_online_htmnpo_ronpo.sh`, run `chmod +x run_qwen_online_htmnpo_ronpo.sh` once.

Important environment notes:

- Do not use one global `PYTHON_RM` for all reward models on this server. `skywork` and `athene` should use `mnpo_infer`, while `armo` should use `mnpo_train`.
- `PYTHON_RM_SKYWORK=/home/sjkim/anaconda3/envs/mnpo_infer/bin/python` avoids the Skywork tokenizer `ModelWrapper` parse error seen under `mnpo_train`.
- `PYTHON_RM_ARMO=/home/sjkim/anaconda3/envs/mnpo_train/bin/python` is intentional. `RLHFlow/ArmoRM-Llama3-8B-v0.1` fails in `mnpo_infer` because that environment's newer `transformers` no longer exposes `LLAMA_INPUTS_DOCSTRING`, while `mnpo_train` is compatible.
- `RM_GPUS=0,1,2` shards reward scoring across three GPU-local scorer processes. Reduce it to a single GPU if memory is tight.
- `RM_BATCH_SIZE=16` is the conservative default for 8B reward models. Use `RM_BATCH_SIZE=8` after OOM, or try `32` if memory is clearly underused.
- `EXT_CACHE_ROOT` controls Hugging Face, datasets, torch, triton, pip, XDG, and W&B cache locations through `scripts/setup_ext_cache.sh`.
- `OUTPUT_ROOT` controls checkpoint output. The examples below write checkpoints to `/ext_hdd/sjkim/mnpo/outputs`.
- If `/ext_hdd` is unavailable on a server, set both `EXT_CACHE_ROOT` and `OUTPUT_ROOT` to a writable filesystem before running.

The stage-1 script writes intermediate data under:

```text
data/qwen2.5-1.5b-instruct_online_htmnpo_ronpo/
```

and checkpoints under:

```text
/ext_hdd/sjkim/mnpo/outputs/qwen2.5-1.5b-instruct_htmnpo_skywork_online_multiobj_stage_1
/ext_hdd/sjkim/mnpo/outputs/qwen2.5-1.5b-instruct_htmnpo_athene_online_multiobj_stage_1
/ext_hdd/sjkim/mnpo/outputs/qwen2.5-1.5b-instruct_htmnpo_armo_online_multiobj_stage_1
/ext_hdd/sjkim/mnpo/outputs/qwen2.5-1.5b-instruct_ronpo_vs_htmnpo_online_multiobj_stage_1
```

To run stages 1-3 after stage 1 is stable:

```bash
cd /home/sjkim/MNPO
conda activate mnpo_train
export PYTHONPATH=$(pwd)

export CUDA_VISIBLE_DEVICES=0,1,2
export RM_GPUS=0,1,2
unset PYTHON_RM
export PYTHON_RM_SKYWORK=/home/sjkim/anaconda3/envs/mnpo_infer/bin/python
export PYTHON_RM_ATHENE=/home/sjkim/anaconda3/envs/mnpo_infer/bin/python
export PYTHON_RM_ARMO=/home/sjkim/anaconda3/envs/mnpo_train/bin/python
export EXT_CACHE_ROOT=/ext_hdd/sjkim
export OUTPUT_ROOT=/ext_hdd/sjkim/mnpo/outputs
export STAGES="1 2 3"
export PLAYERS="skywork,athene,armo"
export RUN_HTMNPO=1
export RUN_RONPO=1

bash ./run_qwen_online_htmnpo_ronpo.sh
```

Stage `t` requires `data/gemma2_ufb_part<t>_train.jsonl` and `data/gemma2_ufb_part<t>_test.jsonl`. Stage 1 can share the same base Qwen generations and scores between HT-MNPO and RONPO via `SHARE_STAGE1_BASE_DATA=1` (default).

### Pipeline Details

1. Decode the current policy on train/test prompts with seeds `13 21 42 79 100` by default.
2. Post-process generated responses into one response pool per prompt.
3. Score the same response pool with `skywork`, `athene`, and `armo`.
4. Build HT-MNPO pairs per player with `mnpo_scripts.build_ht_mnpo_dataset`. The reward-model scores are used to choose each player's winning/losing response pair; any `ht_target` column is metadata and is not consumed by the upstream MNPO loss.
5. Build RONPO pairs with `mnpo_scripts.build_multi_objective_dataset`.
6. Precompute Qwen reference/history logprobs with `mnpo_scripts.precompute`. This applies the model chat template by default so the stored logprobs and training batches use the same Qwen instruction format as decoding.
7. Train each HT-MNPO player policy and the RONPO policy with `mnpo_scripts.run_mnpo`. For HT-MNPO the active loss is `loss_type=mnpo`; heterogeneity comes from player-specific pairs and opponent `history_paths`.

Key files:

- Runner: `run_qwen_online_htmnpo_ronpo.sh`
- HT-MNPO pair builder: `mnpo_scripts/build_ht_mnpo_dataset.py`
- RONPO pair builder: `mnpo_scripts/build_multi_objective_dataset.py`
- Multi-GPU reward-scoring shard helper: `on_policy_data_gen/shard_outputs.py`
- HT-MNPO config: `training_configs/mnpo/qwen2.5-1.5b-instruct-ht-mnpo-multiobj-iter1.yaml`
- RONPO config: `training_configs/ronpo/qwen2.5-1.5b-instruct-ronpo-multiobj-iter1.yaml`
- External cache setup: `scripts/setup_ext_cache.sh`

RONPO adversary details:

- `c(k,a)` is approximated as `E_{y~pi_t} sigmoid(scale * (score_k(y) - score_k(a)))` over the generated response pool.
- `sigma(k,a)` is updated by exponentiated gradient:
  `sigma <- exp(-alpha*c) * sigma0^(alpha*kappa) * sigma^(1-alpha*kappa)`.
- The RONPO JSONL rows include `ronpo_adversary_mass`, `ronpo_weight`, `ronpo_objective_name`, and `ronpo_adversary_response_index`.
- The trainer reads `ronpo_weight` and upweights high-sigma adversarial atoms in the RONPO loss.

### Resume and Debug

The runner skips completed outputs by default. If generation or some reward scores already exist, rerunning the command continues from the first missing stage artifact.

Common controls:

```bash
# These assume you already exported the common variables from the stage-1
# command above in the same shell.

# Rebuild only generated responses.
FORCE_DECODE=1 bash ./run_qwen_online_htmnpo_ronpo.sh

# Rebuild reward scores. Do not use this if skywork/athene are already complete
# and only armo failed; just rerun without FORCE_SCORE.
FORCE_SCORE=1 bash ./run_qwen_online_htmnpo_ronpo.sh

# Rebuild only selected reward scores. This is useful after changing one scorer.
FORCE_SCORE_OBJECTIVES=athene FORCE_SCORE_SPLITS=train bash ./run_qwen_online_htmnpo_ronpo.sh

# Rebuild pair JSONL files from existing scores.
FORCE_BUILD_PAIRS=1 bash ./run_qwen_online_htmnpo_ronpo.sh

# Re-run precompute from existing pair files.
FORCE_PRECOMPUTE=1 bash ./run_qwen_online_htmnpo_ronpo.sh

# Fast debugging only. Do not use for final results.
RM_MAX_SAMPLES=2000 STAGES="1" bash ./run_qwen_online_htmnpo_ronpo.sh
```

For CUDA OOM, first resume with smaller reward/precompute/training batches. Do not set `FORCE_SCORE=1`; completed score files will still be reused.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RM_BATCH_SIZE=4
export RM_SAMPLE_BATCH_SIZE=16
export PRECOMPUTE_BATCH_SIZE=1
export TRAIN_PER_DEVICE_BATCH_SIZE=1
export TRAIN_PER_DEVICE_EVAL_BATCH_SIZE=1
export TRAIN_GRADIENT_ACCUMULATION_STEPS=16
export TRAIN_GENERATE_DURING_EVAL=false

bash ./run_qwen_online_htmnpo_ronpo.sh
```

If reward scoring still OOMs, set `RM_BATCH_SIZE=2` and `RM_SAMPLE_BATCH_SIZE=8`. If training still OOMs, also lower `TRAIN_MAX_LENGTH=1536`, `TRAIN_MAX_PROMPT_LENGTH=1400`, `PRECOMPUTE_MAX_LENGTH=1536`, and `PRECOMPUTE_MAX_PROMPT_LENGTH=1400`; this changes the token budget and should be recorded with the run.

If `skywork` fails with `data did not match any variant of untagged enum ModelWrapper`, it is running under the wrong reward-model environment. Re-run with `PYTHON_RM_SKYWORK=/home/sjkim/anaconda3/envs/mnpo_infer/bin/python`.

If `armo` fails with `cannot import name 'LLAMA_INPUTS_DOCSTRING'`, it is running under the wrong reward-model environment. Re-run with `PYTHON_RM_ARMO=/home/sjkim/anaconda3/envs/mnpo_train/bin/python`.

If a run is interrupted after `skywork` and `athene` scoring but before `armo`, do not set `FORCE_SCORE=1`. The existing score files will be reused and only missing outputs will be created.

### Robust Evaluation

After stage 1 training finishes, run the HT-MNPO vs RONPO robust evaluation:

```bash
cd /home/sjkim/MNPO
conda activate mnpo_train
export PYTHONPATH=$(pwd)

export CUDA_VISIBLE_DEVICES=0,1,2
export RM_GPUS=0,1,2
unset PYTHON_RM
export PYTHON_RM_SKYWORK=/home/sjkim/anaconda3/envs/mnpo_infer/bin/python
export PYTHON_RM_ATHENE=/home/sjkim/anaconda3/envs/mnpo_infer/bin/python
export PYTHON_RM_ARMO=/home/sjkim/anaconda3/envs/mnpo_train/bin/python
export EXT_CACHE_ROOT=/ext_hdd/sjkim
export OUTPUT_ROOT=/ext_hdd/sjkim/mnpo/outputs
export STAGE=1

bash ./evalscope/run_qwen_htmnpo_ronpo_robust_eval.sh
```

This evaluation generates one response per model for held-out prompts, scores baseline/HT-MNPO/RONPO responses with the same objectives, and writes:

- `model_summary.csv`: mean and minimum objective-normalized scores, plus mean/min win rate vs baseline.
- `per_objective_scores.csv`: objective-wise score and win-rate breakdown.
- `pairwise_win_rates.csv`: model-vs-model win rates per objective.

## RONPO Toy Experiments

Run the priority toy suite:

```bash
python toy/toy_v2.py --experiment all_priority --outdir toy/toy_outputs_v2
```

The suite includes the original decoy game, decoy severity sweep, kappa sweep, stochastic two-query policy update, random heterogeneous tournaments, and a fixed-adversary ablation. The fixed-adversary ablation checks whether gains come from adversarial adaptation rather than just using multiple objectives.

## Evaluation

We adopt [EvalScope](https://github.com/modelscope/evalscope/tree/main) for a unified evaluation pipeline to save time and ensure reproducibility.

Under the evalscope directory, we provide three example setups that cover most common evaluation scenarios. You can adapt them to your needs:

1. evaluating online APIs; 
2. evaluating LLM-as-a-judge tasks;
3. evaluating rule-based tasks.



### Installation

Follow the official GitHub instructions to set up EvalScope:

```bash
conda create -n evalscope python=3.10 -y
conda activate evalscope

# Clone the EvalScope package outside this repository. The local
# MNPO/evalscope directory contains only example scripts, not the package.
cd /tmp
git clone https://github.com/modelscope/evalscope.git evalscope-src
cd evalscope-src/
pip install -e .
```

For rule-based EvalScope tasks, use the generic vLLM wrapper with the target checkpoint:

```bash
conda activate evalscope
export PYTHONPATH=$(pwd)

MODEL_NAME=/ext_hdd/sjkim/mnpo/outputs/qwen2.5-1.5b-instruct_ronpo_vs_htmnpo_online_multiobj_stage_1 \
MODEL_BASENAME=qwen2.5-1.5b-instruct_ronpo_vs_htmnpo_online_multiobj_stage_1 \
GPU_IDS=0 \
DATASETS=ifeval \
EVAL_BATCH_SIZE=20 \
bash evalscope/run_vllm_eval.sh
```

For Arena-Hard or other judge-based tasks, use the same vLLM wrapper with a different task script:

```bash
MODEL_NAME=outputs/qwen2.5-1.5b-instruct_ronpo_stage_1 \
MODEL_BASENAME=qwen2.5-1.5b-instruct_ronpo_stage_1 \
GPU_IDS=0 \
TASK_SCRIPT=$(pwd)/evalscope/run_arena_hard_task.py \
TASK_ARGS="--eval-batch-size 12" \
bash evalscope/run_vllm_eval.sh
```

### Serving the Model with vLLM

Before evaluation, first serve your model via `vllm`:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-2-9b-it \
    --served-model-name google/gemma-2-9b-it \
    --trust_remote_code \
    --port 8801 \
    --tensor-parallel-size 8 # num of gpu
```

### Evaluating Rule-Based Datasets

Both rule-based tasks and LLM-as-judge tasks can be implemented using the task_cfg. For example: 
```python
task_cfg = TaskConfig(
    model='gemma-2-9b-it',
    api_url="http://127.0.0.1:8801/v1",
    api_key="EMPTY",
    eval_type=EvalType.SERVICE,
    datasets=['ifeval'], # add more datasets here
    eval_batch_size=20,
)
```

### Evaluating Datasets with LLM Judges

Some benchmarks require an LLM judge for evaluation. Here is an example script:

```python
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType, JudgeStrategy

task_cfg = TaskConfig(
    model='gemma-2-9b-it',
    api_url='http://127.0.0.1:8801/v1',
    api_key='EMPTY',
    eval_type=EvalType.SERVICE,
    datasets=[
        'alpaca_eval',
    ],
    eval_batch_size=12,
    judge_worker_num=12,
    limit=5,  # optional for debugging
    judge_strategy=JudgeStrategy.AUTO,
    judge_model_args={
        'model_id': 'gpt-5-mini',
        'generation_config': {"reasoning_effort": "minimal"},
        'api_url': 'xx',
        'api_key': 'xx',
    },
)

run_task(task_cfg=task_cfg)
```

> 📌 **Notes**
>
> * The version of EvalScope used in the paper is **1.0.2**.
> * The LLM judge is `gpt-5-mini` (Aug 7, 2025), with `reasoning_effort="minimal"`.
> * Please configure the appropriate model and related parameters according to the specific task type to ensure the evaluation runs correctly.

You can find the list of datasets supported by EvalScope at [the official documentation](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/llm.html).

## Support & Citation
If you build on this codebase in academic work, please cite the MNPO methodology and link back to this repository so others can reproduce your setup.

```
@article{wu2025multiplayer,
  title={Multiplayer Nash Preference Optimization},
  author={Wu, Fang and Huang, Xu and Xuan, Weihao and Zhang, Zhiwei and Xiao, Yijia and Wan, Guancheng and Li, Xiaomin and Hu, Bing and Xia, Peng and Leskovec, Jure and others},
  journal={arXiv preprint arXiv:2509.23102},
  year={2025}
}
```
