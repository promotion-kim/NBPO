# Nash Bargaining Preference Optimization (NBPO)

Code and manuscript for **NBPO**, a multi-objective alignment method submitted to ICLR 2027.

NBPO separates two decisions that are usually entangled. Within each objective, a preference game
summarises potentially cyclic pairwise feedback into a single game value, without fitting a scalar
reward. Across objectives, Nash bargaining selects one compromise by balancing improvements over a
reference fallback, so the trade-off follows proportional gains rather than prespecified weights.

Under joint improvement the selected outcome is strictly above the reference on every objective,
Pareto optimal, unique in game-value space, proportionally fair, and unchanged by independent
positive affine rescaling of the objective utilities.

The manuscript and its source are in [`game_nbpo_iclr/`](game_nbpo_iclr/).

## Relationship to MNPO

This repository is built on the
[**MNPO** (Multiplayer Nash Preference Optimization)](https://github.com/smiles724/MNPO) codebase,
and the training and decoding pipeline is taken from it largely unchanged: `mnpo_scripts/`,
`alignment/`, `on_policy_data_gen/`, and `accelerate_configs/` are MNPO components.

NBPO is implemented inside that pipeline rather than as a separate stack. The constrained proximal
update reduces to MNPO's existing pairwise regression trainer with the bargaining weights supplied
as per-pair targets, so the additions are a dual solver that produces those weights from the
objective-wise game values and a pair builder that applies them. MNPO's own baselines remain
runnable and are used as controls.

## Two NBPO pipelines: paper-exact vs legacy fixed-reference

The repository contains two distinct implementations. Do not conflate them.

**`scripts/nbpo/` — finite-temperature NBPO (Algorithm 1, paper-exact).** The
manuscript's construction: adaptive KL-regularized opponents
ν\*<sub>k,π</sub> ∝ μ·exp(−r/β<sub>k</sub>) (Eq. 7), soft-min game values
V<sub>k,β</sub> (Eq. 8), a measured disagreement point
d<sub>k</sub> = V<sub>k,β</sub>(μ) (Eq. 10, never assumed zero), projected dual
descent on the **raw** multipliers λ ← Π<sub>Λ</sub>[λ − γ(ŝ − 1/λ)] (Eq. 27),
pairwise regression to (h<sub>t</sub> − η Σ<sub>k</sub> λ<sub>k</sub>Z<sub>k</sub>)²
with sequence-sum log-probabilities (Eq. 26, `loss_type: nbpo`), and the held-out
stage-acceptance gate of Algorithm 1. **The dual solver lives in
`mnpo_scripts/nbpo_solver.py`** (CLI: `scripts/nbpo/solve_nbpo_dual.py`); its
M = 4e3–3e5 dual iterations run on the frozen finite response pool — cheap
tensor updates — and the neural policy is fit once afterwards, not per dual step.
The released configuration uses **R = 1 fixed-point iteration per dual update, a
disclosed practical approximation** whose residual is logged. λ is raw
throughout: any normalized weights that appear in logs are display diagnostics
only, never what the update uses.

**`scripts/bpo/` — fixed-reference Anchored-BPO (β = ∞ baseline, legacy).**
Fixed-anchor surpluses s<sub>k</sub> = P<sub>k</sub>(π≻μ) − ½ against the frozen
reference, static normalized/clipped weight rules, token-mean log-probabilities,
auxiliary anchor/pref-SFT losses in some drivers, and an evaluator
(`eval_bpo_surplus.py`) that clamps nonpositive surpluses. Kept runnable, header-labeled,
and used as the β→∞ baseline; not Algorithm 1.

The evaluators differ the same way: `scripts/nbpo/eval_game_value.py` computes
V, d, and surpluses at finite β and reports Nash welfare as `null` whenever any
surplus is nonpositive (no clamping); `scripts/bpo/eval_bpo_surplus.py` is the
fixed-reference diagnostic.

### Paper-exact stage, command by command

```bash
export PYTHONPATH=$(pwd)

# 1. Judge the full pairwise matrix (policy-vs-ref AND ref-vs-ref for d_k),
#    both presentation orders, retries, loud completeness failure.
python -m scripts.nbpo.judge_pairwise_matrix \
  --policy-files s0=pol0.json s1=pol1.json s2=pol2.json s3=pol3.json \
  --reference-files r0=ref0.json r1=ref1.json r2=ref2.json r3=ref3.json \
  --objectives-config training_configs/nbpo/objectives/ultrafeedback.yaml \
  --objectives instruction_following,truthfulness,honesty,helpfulness \
  --judge-model-path Qwen/Qwen3-32B --backend vllm --output verdicts.jsonl

# 2. Swap-average into centered float64 tensors A_policy and A_ref.
python -m scripts.nbpo.build_preference_tensor \
  --verdicts verdicts.jsonl --policy-files s0=pol0.json ... \
  --reference-files r0=ref0.json ... \
  --objectives instruction_following,truthfulness,honesty,helpfulness \
  --objectives-config training_configs/nbpo/objectives/ultrafeedback.yaml \
  --out-dir tensor/

# 3. Projected dual descent on the frozen pool (Eq. 27). Matched controls via
#    --aggregation utilitarian|absolute_maxmin|surplus_maxmin.
python -m scripts.nbpo.solve_nbpo_dual --tensor-dir tensor/ --out-dir solver/ \
  --beta 0.25 --eta 1.0 --gamma 0.5 -M 40000 -R 1 --aggregation nash

# 4. Pair targets: six unordered pairs, per-objective z_k ~ nu*, sampled
#    Bernoulli Z_k (Eq. 24), UNSCALED sum_k lambda_k Z_k (eta applied in the trainer).
python -m scripts.nbpo.build_nbpo_pairs --tensor-dir tensor/ --solver-dir solver/ \
  --policy-files s0=pol0.json ... --out-dir pairs/ --target-mode sampled

# 5. Precompute logps with SEQUENCE-SUM reduction (writes precompute_meta.json,
#    validated at training time), then train the loss_type=nbpo branch.
python -m mnpo_scripts.precompute --logp_reduction sum --ronpo_target_mode none \
  --model_name_or_path $PI_T --ref_model $BASE --history_paths $PI_T \
  --train_dir pairs/pairs_train.jsonl --output_dir precomputed/ ...
python -m accelerate.commands.launch --num_processes 1 -m mnpo_scripts.run_mnpo run_config.yaml

# 6. Held-out game-value evaluation + Algorithm-1 gate (or run the whole stage):
python -m scripts.nbpo.eval_game_value --tensor-dir tensor_holdout/ --beta 0.25
python -m scripts.nbpo.run_nbpo_stage --config training_configs/nbpo/ultrafeedback_llama.yaml

# CPU-only dry run (mock judge, no LLM anywhere):
python -m scripts.nbpo.run_nbpo_stage --config tests/fixtures/nbpo_toy/config.yaml --dry-run
```

Stage configs with the manuscript hyperparameters are in
`training_configs/nbpo/` (per-dataset M, γ; `opponent_betas` is the explicit
temperature field — the trainer's `beta: 10.0` is a metrics-only display
multiplier). The exact objective rubrics are versioned in
`training_configs/nbpo/objectives/`; the TL;DR rubrics were never committed to
this repository, so the TL;DR config fails loudly at the rubric loader instead
of inventing prompt text. Tests: `tests/test_nbpo_*.py`.

## Repository layout

| Path | Description |
|---|---|
| `game_nbpo_iclr/` | Manuscript source and figures (`main_v2.tex`). |
| `scripts/nbpo/` | **Paper-exact finite-temperature NBPO** (Algorithm 1): judging, tensors, dual solver CLI, pair targets, game-value evaluator, stage runner. |
| `mnpo_scripts/nbpo_core.py`, `nbpo_solver.py` | Finite-pool math and the fixed-point + projected dual solver. |
| `scripts/bpo/` | Legacy **fixed-reference (β = ∞) Anchored-BPO** baselines. |
| `mnpo_scripts/` | Training pipeline: config dataclasses, precomputation, trainer, `run_mnpo.py`. |
| `training_configs/nbpo/` | Stage configs (manuscript hyperparameters) and versioned objective rubrics. |
| `on_policy_data_gen/` | On-policy generation, filtering, and preference annotation. |
| `alignment/` | Shared data-loading and model utilities. |
| `accelerate_configs/` | Accelerate, DeepSpeed ZeRO, and FSDP launch configs. |
| `evalscope/` | Evaluation helpers. |
| `run_iter{1,2,3}.sh` | Example three-iteration pipeline. |

## Environment

Two environments, Python 3.10 with CUDA 12.1 wheels. Confirm `nvidia-smi` works first.

```bash
# training
conda create -n mnpo_train python=3.10 -y && conda activate mnpo_train
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4 accelerate==0.29.2 deepspeed==0.15.4 transformers==4.44.2 \
    trl==0.9.6 datasets==2.18.0 huggingface-hub==0.23.2 peft==0.7.1 wandb

# decoding
conda create -n mnpo_infer python=3.10 -y && conda activate mnpo_infer
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.5.1 "transformers<4.54.0" datasets==2.18.0 numpy==1.26.4 more_itertools
```

Set `PYTHONPATH` to the repository root before running any module entrypoint:

```bash
export PYTHONPATH=$(pwd)
```

The default training configs use `attn_implementation: eager`, so FlashAttention is optional.
`on_policy_data_gen/decode.py` sets `VLLM_ATTENTION_BACKEND=FLASHINFER`; remove that line if
FlashInfer does not load on your GPU.

## Workflow

1. `mnpo_scripts.split_dataset` — shard the preference dataset per iteration.
2. `on_policy_data_gen.decode` → `post_process` → `reward_model_annotate` — build on-policy pairs.
3. `mnpo_scripts.precompute` — log-probabilities, normalizers, and history buffers.
4. `mnpo_scripts.run_mnpo` — the update itself, via Accelerate.

For paper-exact NBPO, step 4 runs `loss_type: nbpo` on the `nbpo_weighted_z` targets produced by
the dual solve (see the command-by-command section above); the matched aggregation controls
(utilitarian, absolute max-min, surplus max-min) share the same tensors, β, d, response pool, and
optimizer budget and differ only in how the weight vector is chosen — utilitarian fixes λ = 1, the
max-min rules run an adversarial two-player solve with a logged duality gap (not a static one-hot
on the pre-training worst objective).

```bash
bash run_iter1.sh   # then run_iter2.sh, run_iter3.sh
```

## Results

Held-out surpluses are measured on prompts audited to be disjoint from each arm's training pairs
and reference pool; the manuscript's prompt-overlap appendix reports that audit and which panels
were rescored after it. The evidence supports Nash aggregation over the utilitarian and max-min
alternatives. It does not currently support the adaptive game-value representation over a simpler
fixed-reference margin, and the manuscript reports that as an open negative rather than omitting it.

## Citation

Please also cite MNPO, whose pipeline this work builds on:

```bibtex
@inproceedings{mnpo2026,
  title     = {Multiplayer Nash Preference Optimization},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
