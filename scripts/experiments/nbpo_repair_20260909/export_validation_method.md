# Fixed1750 export validation

`validate_export.py` is a standalone, read-only post-training validator. It does
not alter frozen training sources, checkpoints, tokenizers, RoPE buffers, or
controller state. Fresh diagnostic outputs are written only below the repair
task root. Metadata-only success does not claim GPU/full-loading checks passed.

## Inputs and controller compatibility

Accept either `controllers/primary_chain_v1/complete.json` with an `arms` list or
the single-arm `wbc_complete.json`. `--arm wbc` resolves exactly
`arms/wbc_primary_v1`, and `--arm mse` resolves exactly `arms/mse_primary_v1`.
Require controller step1750, matching trainer-state SHA256 and actual global
step1750, matching successful job-exit SHA256, and the exact complete shard-name
to weight-SHA256 map. Reject adapters, missing/extra serialized keys, incorrect
tensor shapes, sharded-index disagreement, non-BF16 serialized policy tensors,
and a config/parameter count other than complete Llama3.1-8B.

The base is fixed to revision `0e9e39f249a16976918f6564b8830bc894c89659` and config
SHA256 `29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e`.
`--inventory` supplies the already captured campaign inventory; actual base
weight/config/tokenizer files must match it. Architectural and RoPE settings
must match that pinned base. Tokenizer comparison includes vocab, special tokens,
full backend algorithm/merges/normalizer, native chat template, and exact native
tokens for two fixed harmless prompts. The saved training tokenizer's known
pad=eos fallback is recorded explicitly; no sampled text or token event is changed.
Export and base generation terminal-ID lists must match exactly.

## Full HF checks and their limits

Full mode requires **exactly one visible CUDA GPU**, explicitly allocated by the
root controller/operator; it never schedules or acquires one. Use the existing
`deps_train` Transformers4.45.2 environment, BF16 parameters, SDPA attention, no
DeepSpeed and no adapters. This is a reload/inference probe of the same HF stack,
not an assertion that HF and vLLM kernels produce bitwise-equal logits. Runtime
versions, CUDA kernel switches, memory peak, source and arguments are retained.

All HF missing/unexpected/mismatched/error lists must be empty. Independently
inspect serialized/state keys and reject any unmaterialized meta tensor, because
Transformers may suppress architecture-specific missing-key warnings (a tiny
CPU regression demonstrated this for `lm_head.weight`). No permissive key repair
or random initialization is accepted.

For first/middle/last-layer query projections and the final norm, every loaded
tensor value must exactly match its saved BF16 tensor. Compare257 fixed evenly
spaced flattened locations per selected tensor to the pinned base and report
values, changed counts, L2 norm and max absolute difference. These samples are
preselected, not searched until a change is found. Small updates can disappear
in BF16 export; a zero observed difference is reported rather than used as a
quality/outcome gate. The sample is not a full-model update norm.

Construct an independent Llama rotary module from the pinned base config and
require exact FP32 `inv_freq`, `original_inv_freq`, and attention-scaling equality
before and after forward/generation. This catches both a BF16 buffer and a
rounded BF16-to-FP32 recast. The validator **does not restore** a bad buffer.

Two fixed harmless requests (paper-kite colors; a sentence about a mug beside a
window) are not selected from an evaluation subset. Save their user-only native
input IDs, full generated raw IDs/text/special-token decode and hashes. Repeat
each no-cache forward and greedy generation twice; require finite, bitwise-equal
repeated logits and identical greedy IDs within this one HF execution. Greedy
smoke uses a fresh explicit generation config: no sampling, one beam,
max_new_tokens64, base terminal IDs and pad ID, repetition_penalty1, no forced
tokens or text stops. A terminal may appear only at the final generated position;
otherwise exactly64 generated tokens indicate the explicit length limit. A
length-limited or empty decoded answer is diagnostic, not proof of poor quality
or early EOS. These probes make no broad competence/safety claim.

## Commands (only after export and permission)

CPU metadata mode, with a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=/work/nbpo_repair_20260909/deps_train:/work/nbpo_repair_20260909/code \
python3 -m scripts.experiments.nbpo_repair_20260909.validate_export \
  --root /work/nbpo_repair_20260909 --arm wbc \
  --controller-manifest /work/nbpo_repair_20260909/controllers/primary_chain_v1/complete.json \
  --inventory /work/nbpo_repair_20260909/provenance/inventory.json \
  --output /work/nbpo_repair_20260909/probes/export_wbc_metadata_v1 --stage metadata
```

The inventory path above is an example; supply the actual captured inventory
path. For full validation, set `CUDA_VISIBLE_DEVICES` to the single GPU index
explicitly allocated by root, use `--stage full`, and choose a fresh output such
as `probes/export_wbc_full_v1`. Repeat for `--arm mse`; never overwrite earlier
diagnostic artifacts. The validator saves `source.json`, `validation.json`, and
`manifest.json`; failed checks retain an error report and do not alter the model.

Estimated cost per arm:16–32GB of sequential hashing I/O plus roughly1–3 minutes
for loading/small forward/greedy tests and about20GB GPU memory. These are planning
estimates, not measured 8B throughput. Tiny CPU tests validate mechanics; actual
8B export and GPU integration remain pending until fixed1750 completion and
explicit GPU allocation. No inference is launched automatically.
