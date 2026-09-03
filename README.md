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

## Repository layout

| Path | Description |
|---|---|
| `game_nbpo_iclr/` | Manuscript source and figures (`main_v2.tex`). |
| `mnpo_scripts/` | Training pipeline: config dataclasses, precomputation, trainer, `run_mnpo.py`. |
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

For NBPO, step 4 consumes per-pair bargaining targets produced by the dual solve; the aggregation
baselines (utilitarian, absolute max-min, surplus max-min) differ only in the weight vector, which
is what makes the comparison in the paper a matched one.

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
