# Robust Nash Preference Optimization (RONPO)

Code and paper artifacts for **Robust Nash Preference Optimization**, an objective-adversarial method for preference alignment under conflicting objectives.

RONPO keeps heterogeneous alignment in a two-player game. A single adversary selects an objective and an opponent response, while the policy optimizes a KL-regularized worst-case preference floor. The repository contains the trainer, target builders, evaluation code, toy experiments, paper source, and compact machine-readable result artifacts. Model checkpoints, raw caches, and W&B logs are intentionally excluded.

## Main model-scale result

The current paper evaluates Llama-3.1-8B-Instruct on PKU-SafeRLHF with Beaver helpfulness reward and negated Beaver harmlessness cost. Values below are mean ± sample standard deviation over training seeds 42 and 43.

| Method | Helpful. | Harmless. | Avg | Worst |
|---|---:|---:|---:|---:|
| RONPO (OS) | **0.6624 ± 0.0111** | 0.5856 ± 0.0016 | **0.6240 ± 0.0063** | **0.4412 ± 0.0085** |
| IPO | 0.5420 ± 0.0067 | 0.6539 ± 0.0003 | 0.5979 ± 0.0035 | 0.4046 ± 0.0027 |
| INPO-avg | 0.5023 ± 0.0390 | 0.6448 ± 0.0104 | 0.5736 ± 0.0143 | 0.3806 ± 0.0202 |
| SimPO | 0.4606 ± 0.0034 | **0.7410 ± 0.0116** | 0.6008 ± 0.0041 | 0.3730 ± 0.0002 |

The complete table and provenance are in [`artifacts/saferlhf_stage4_two_seed/`](artifacts/saferlhf_stage4_two_seed/). Two seeds provide only a descriptive estimate of between-seed variation.

## Install

Python 3.10 and a CUDA-capable PyTorch installation are required for model training. The tested training stack is listed in `requirements.txt`. vLLM is kept in a separate inference environment because its CUDA and PyTorch requirements are platform-specific.

```bash
git clone https://github.com/promotion-kim/RONPO.git
cd RONPO

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD"
```

For deterministic logging and downloads, provide credentials only through the environment:

```bash
export WANDB_API_KEY=...
export WANDB_ENTITY=...
export WANDB_PROJECT=mnpo
export HF_TOKEN=...
```

No credential is required to rebuild the included tables or paper.

## Quick verification

```bash
make test
make table4
make paper
```

`make table4` regenerates the two-seed SafeRLHF table from the included seed-level JSON. `make paper` requires `pdflatex` and `bibtex`.

## Training workflow

The model-scale pipeline has four steps:

1. Decode several responses per prompt with `on_policy_data_gen/decode.py`.
2. Score the shared pool with the selected reward models and build preference pairs.
3. Precompute reference and history log probabilities with `mnpo_scripts/precompute.py`, then add RONPO target columns with `mnpo_scripts/build_os_ronpo_targets.py`.
4. Train every method through the unified `mnpo_scripts.run_mnpo` entry point and evaluate with `mnpo_scripts/evaluate_multi_objective_models.py`.

Example target construction and training:

```bash
python mnpo_scripts/build_os_ronpo_targets.py \
  --input_dir /path/to/precomputed_dataset \
  --output_dir /path/to/precomputed_with_ronpo_targets \
  --kappas 0.05,0.1,0.2

accelerate launch --config_file accelerate_configs/single_gpu.yaml \
  -m mnpo_scripts.run_mnpo \
  training_configs/ronpo/qwen2.5-1.5b-instruct-ronpo-multiobj-iter1.yaml \
  --dataset_mixer=/path/to/precomputed_with_ronpo_targets:1.0 \
  --output_dir /path/to/output \
  --run_name ronpo-reproduction
```

The SafeRLHF Stage 1 to 4 orchestration used for the paper is under `analysis/p5_8b_robust_stage1_stage2_20260717/`, `analysis/p7_stage3_fresh_default_test_20260717/`, `analysis/p8_stage4_fresh_default_test_20260718/`, and `analysis/p10_saferlhf_training_seed43_20260718/`. Paths to storage and GPU IDs are command-line arguments or environment settings and should be adapted to the local machine.

## Repository layout

| Path | Contents |
|---|---|
| `mnpo_scripts/` | Unified RONPO, MNPO, INPO, SPPO, DPO, IPO, SimPO, and HT-MNPO trainer plus target builders |
| `on_policy_data_gen/` | Decoding and open reward-model scorers |
| `analysis/` | Paper evaluation, bootstrap, table, and stage-continuation scripts |
| `toy/` | Synthetic games and theory checks |
| `artifacts/` | Compact JSON, CSV, and Markdown results with provenance |
| `ronpo_aaai/` | AAAI paper source, figures, bibliography, and compiled PDF |

This codebase extends the MNPO training pipeline. The RONPO-specific implementation is centered in `mnpo_scripts/mnpo_trainer.py`, `mnpo_scripts/mnpo_config.py`, and `mnpo_scripts/build_os_ronpo_targets.py`.

## Paper

```bash
cd ronpo_aaai
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Citation metadata will be added after the anonymous review period.
