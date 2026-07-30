# CODEX PROMPT — pick a base model + helpful/harmless objectives with genuine conflict and headroom (inference-only screen)

## 0. Why and what changed

The Qwen3-8B helpful-harmless screen returned 0/20 for two real reasons, not a bug:
1. **No harmlessness headroom / broken resolution probe.** Qwen3-8B is so safety-saturated that a
   "be blunt / answer-anything" prompt did not make it less safe, so the harmlessness reward models could
   not separate base from the degraded probe (base scored as safe or safer). The scorers are correctly
   signed (`rm_beaver_cost.py` outputs -cost, `rm_guard_logodds.py` outputs logP(safe)-logP(unsafe)).
2. **General helpfulness reward models reward good refusals.** Skywork, Athene, and the ArmoRM helpfulness
   head score a well-written refusal as helpful, so the one harmlessness model that resolved (Llama-Guard-3)
   correlated positively (+0.44) with helpfulness. The conflict-passing model (ShieldGemma) did not resolve,
   so its negative correlation is noise.

Two fixes, plus a base-model change:
- **Change the base** to a standard, current ~8B instruct model that is NOT safety-saturated, so a genuine
  helpful-harmless tension and headroom exist. A large-scale refusal-compliance audit finds Llama-3-class
  models are conservative (over-refusal about 11%, harmful compliance about 0.3%), giving real helpfulness
  headroom, while Qwen-2.5-7B is already balanced on both (over-refusal 0.8%). Different ecosystems sit at
  different points, so screen a few and pick on headroom and conflict.
- **Use a compliance-based helpfulness signal**, not a quality reward model: refusing an answerable request
  must count as unhelpful. This is the operationalization the tradeoff literature uses (split prompts into
  should-answer and should-refuse; helpfulness is answering the should-answer ones).
- **Use a real unsafe resolution probe**, not a prompt instruction: a permissive or non-safety-tuned
  reference model's responses (or a jailbreak), so the harmlessness models can separate safe from unsafe.

Goal: find a (base model, helpfulness signal, harmlessness reward model, dataset) combination with genuine
conflict, two-sided resolution, and base headroom, staying standard and defensible. No training here; this
screen locks the setting for a later RONPO run. Download any model or dataset.

## 1. Hard rules

- Inference only. No preference training, no HuggingFace upload, no paper edits.
- Never touch the spent 604-prompt sealed split (`results/p1_sealed_reward_seed42_20260714/`).
- **Pre-register and hash** the candidate base models, the helpfulness and harmlessness signals, the dataset
  mix, the resolution probes, and the pass thresholds **before** computing any correlation or ranking. Select
  on conflict, resolution, and headroom, which are RONPO-agnostic, never on which setting would make RONPO
  win.
- Both objective signals must come from **independent model lineages** and the base must be a standard,
  widely used model (no bespoke or hand-built reward model, no obscure base).
- Regenerate every number from JSON. Report all candidates, including failures. Use available GPUs (4 B200,
  H200 `air3` if free after a read-only check, free odin2 GPUs; never another user's GPU). Minimal storage.

## 2. Candidate base models (standard, current, screen for headroom)

Download and screen these ~7-9B instruct models. The primary, most defensible choice is Llama-3.1-8B-Instruct;
the others span the safety-calibration spectrum so the screen can find genuine two-sided tension:
- `meta-llama/Llama-3.1-8B-Instruct` (primary; conservative, real over-refusal headroom, the standard 8B
  alignment base).
- `Qwen/Qwen2.5-7B-Instruct` (balanced contrast; likely low headroom, include to show the difference).
- `mistralai/Ministral-8B-Instruct-2410` and/or `mistralai/Mistral-7B-Instruct-v0.3` (permissive; harmful-
  compliance headroom on the safety side).
- Optional: `allenai/Llama-3.1-Tulu-3-8B`, `allenai/OLMo-2-1124-7B-Instruct`, `google/gemma-2-9b-it`.

## 3. Objective signals

**Helpfulness = compliance (refusal penalized).** Do not use a quality reward model as the helpfulness
objective. Use a refusal classifier so that refusing an answerable request scores low: for example
`helpfulness = 1 - P(refusal)` from a refusal detector (WildGuard's refusal head, or a Llama-Guard-style
response-refusal classification), optionally gated by a quality reward model only among non-refusals. Keep a
general quality reward model (Skywork-Reward-V2) as a secondary signal for comparison, but the primary
helpfulness objective is compliance. Write the scorer mirroring the existing `on_policy_data_gen/rm_*.py`.

**Harmlessness = a resolving safety signal (continuous).** Use `meta-llama/Llama-Guard-3-8B` (it resolved on
Qwen3-8B) via `rm_guard_logodds.py`, and also try `PKU-Alignment/beaver-7b-v2.0-cost` and a modern guard
(Aegis2.0, WildGuard, ShieldGemma). Pick the one that resolves on the chosen base.

## 4. Dataset (safety-relevant mix, unchanged)

`PKU-Alignment/PKU-SafeRLHF` (core) + OR-Bench hard (`huggingface.co/bench-llms`) + XSTest
(`allenai/xstest-response`) + higher-severity `PKU-Alignment/BeaverTails`, mixed so both over-refusal
(helpful-low on should-answer prompts) and under-refusal (harmless-low on should-refuse prompts) appear.
Hold out a validation subset and a disjoint fresh subset. Record versions and hashes.

## 5. Screen (per base model)

Decode on the mixed pool with the base's own chat template, seed 42:
- `base` (the candidate);
- helpfulness resolution probe `over_refusing` / `terse` (a known-less-helpful policy);
- **harmlessness resolution probe = a permissive/non-safety-tuned reference model's responses** (for example
  a base pretrained or a light-safety model), or a jailbroken decode of the candidate, so the harm models
  can separate safe from unsafe;
- a small weak model for general resolution.

Score every response set with the compliance helpfulness signal and each harmlessness candidate. Compute:
- **Conflict:** Spearman between helpfulness (compliance) and each harmlessness signal on the pooled
  per-prompt scores; require the winning pair to reach Spearman <= -0.2.
- **Resolution (per objective):** paired prompt bootstrap of base minus the matched known-worse probe, 95% CI
  must exclude 0. The harmlessness probe must be a genuinely-less-safe policy, not a prompt trick.
- **Headroom:** on the chosen pair, is the base beatable on the weaker objective (some policy exceeds base,
  CI excluding 0)? For a conservative base this is expected on the helpfulness/compliance side.

## 6. Selection

Select the (base model, helpfulness signal, harmlessness model, dataset mix) that has the most negative
Spearman among combinations that pass two-sided resolution and have base headroom on the weaker objective,
using independent lineages. Tie-break toward the most standard base (Llama-3.1-8B-Instruct) and the most
widely used reward models. Document why the choice is standard and defensible (a conservative but canonical
8B base with a documented over-refusal tendency is a representative, not cherry-picked, setting).

If no base and objective combination reaches conflict, resolution, and headroom together, report the honest
negative: standard ~8B instruct models plus standard safety reward models do not expose a helpful-harmless
conflict that the policy can move, and the model-scale claim stays scoped to 1.5B and the synthetic games.

## 7. Deliverables (into `results/p1_8b_base_objective_screen_20260716/`)

- `PREREG.md` + `selection_lock.json` (+SHA); dataset manifest (+SHA); scorer scripts for any new signal.
- `CONFLICT.md`, `RESOLUTION.md`, `HEADROOM.md` (full matrices, per base model).
- `SELECTION.md`: the locked base model, helpfulness signal, harmlessness model, and dataset mix, with the
  conflict/resolution/headroom evidence, a defensibility justification, and the recommended RONPO run config.
- The committed script that regenerates every number from the score JSONL.
- `COMPLETION_AUDIT.md`, ending `spent_sealed_split_touched=false`.

## 8. One-paragraph brief (paste at top of the codex run)

> Find a base model and a helpful-harmless objective pair with genuine conflict, two-sided resolution, and
> base headroom, to replace the saturated Qwen3-8B setting where the previous screen returned 0/20. That
> screen failed because Qwen3-8B is too safety-saturated for a prompt-based unsafe probe to work, and because
> general helpfulness reward models reward well-written refusals and so align with rather than conflict with
> safety. Fix all three: (1) screen standard current ~8B instruct base models that are not safety-saturated,
> primarily meta-llama/Llama-3.1-8B-Instruct (conservative, documented over-refusal, real helpfulness
> headroom), plus Qwen2.5-7B-Instruct, Ministral-8B-Instruct-2410 or Mistral-7B-Instruct-v0.3, and optionally
> Tulu-3-8B, OLMo-2-7B-Instruct, or gemma-2-9b-it; (2) use a compliance-based helpfulness objective (a
> refusal classifier so refusing an answerable request is unhelpful), not a quality reward model; (3) use a
> real unsafe resolution probe (a permissive or non-safety-tuned reference model's responses, or a jailbreak)
> so the harmlessness models can separate safe from unsafe. Harmlessness uses Llama-Guard-3-8B and a beaver
> cost or modern guard, continuous. Dataset is the PKU-SafeRLHF, OR-Bench hard, XSTest, higher-severity
> BeaverTails mix, holding both over-refusal and under-refusal cases. Download any model or dataset.
> Pre-register and hash the candidates and thresholds first, then measure conflict (Spearman at most -0.2),
> two-sided resolution (base minus a genuinely-worse policy, 95% CI excluding zero), and headroom, and select
> the standard, independent-lineage combination with the strongest genuine conflict that also resolves and
> has headroom, tie-breaking to Llama-3.1-8B-Instruct. If none qualifies, report the honest negative and keep
> the claim scoped to 1.5B. Do no training, no upload, never touch the spent sealed split, and regenerate
> every number from JSON.
