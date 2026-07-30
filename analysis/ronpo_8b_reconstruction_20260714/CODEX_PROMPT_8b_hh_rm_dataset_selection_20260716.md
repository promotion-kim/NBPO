# CODEX PROMPT — select the 2-objective (helpfulness, harmlessness) reward models and dataset for a RONPO-favorable but defensible Qwen3-8B experiment

## 0. Goal and the defensibility line

Choose the concrete helpfulness reward model, harmlessness reward model, and prompt dataset for a Qwen3-8B
RONPO experiment. We want a setting that favors RONPO, but only in the legitimate sense: RONPO is designed
for genuinely conflicting objectives, so a high-conflict setting plays to its designed strength. The
favorability must come from real objective conflict measured on standard, widely used reward models and
datasets, not from cherry-picking artifacts.

Three rules keep this AAAI-defensible:
1. Every candidate reward model and dataset is a standard, widely cited artifact in multi-objective or
   safety alignment. No bespoke or hand-built reward model.
2. The selection is made on conflict, resolution, and base headroom, which are RONPO-agnostic properties,
   and is **pre-registered and hashed before any RONPO training or any RONPO ranking is computed.** Do not
   pick the pair because it makes RONPO win.
3. The helpfulness and harmlessness signals must come from **independent model lineages** (not two heads of
   one reward model), so their disagreement is a real cross-objective conflict, not shared-model noise.

You may download any model or dataset at any time. This run does no preference training; it decodes, scores,
measures, and locks a choice.

## 1. Candidate reward models (download as needed)

**Helpfulness (pick one).** Standard, strong, general-helpfulness reward models:
- `Skywork/Skywork-Reward-V2-Llama-3.1-8B` (state of the art on RewardBench v1/v2, PPE, RMB, RM-Bench;
  already local at `/ext_hdd/sjkim/huggingface/hub`).
- `Skywork/Skywork-Reward-V2-Qwen3-8B` (same family, matches the policy backbone; worth including).
- `Nexusflow/Athene-RM-8B` (local); ArmoRM helpfulness head (`RLHFlow/ArmoRM-Llama3-8B-v0.1`, local) as a
  secondary comparison only.

**Harmlessness (pick one, continuous score required).** A dedicated safety or cost signal from a different
lineage than the helpfulness model. Use a continuous score: a cost-model reward, or the negated unsafe
probability / unsafe logit of a guard model.
- `PKU-Alignment/beaver-7b-v1.0-cost` (and v2.0) — the canonical Safe-RLHF harmlessness cost model, the
  reference for the helpful-harmless tradeoff (continuous cost; negate for a harmlessness reward).
- `nvidia/Aegis-AI-Content-Safety-*` / Aegis2.0 (Llama-3.1-8B guard, surpasses Llama-Guard-3).
- `allenai/wildguard` (7B), `google/shieldgemma-9b`, `meta-llama/Llama-Guard-3-8B`,
  `Qwen/Qwen3Guard-Gen-8B` (a 0.6B variant is local; the repo already has `on_policy_data_gen/rm_qwen3guard.py`).

Write a scorer for each new model by mirroring the existing `on_policy_data_gen/rm_*.py`. For guard models,
define the harmlessness score once (for example the model's log-probability or probability of the unsafe
label) and use it consistently.

## 2. Candidate datasets (download as needed)

The current UltraFeedback prompts do not induce a helpful-harmless conflict (benign; a good answer is both
helpful and safe). Use a safety-relevant mix where both failure modes appear:
- **Core:** `PKU-Alignment/PKU-SafeRLHF` prompts (dual-preference helpfulness and harmlessness, 19 harm
  categories, borderline by design).
- **Over-refusal (surfaces helpful-low, the strong base over-refuses):** OR-Bench hard subset
  (`huggingface.co/bench-llms`) and XSTest (`allenai/xstest-response`).
- **Harmful slice (surfaces harmless-low, a helpfulness-chaser under-refuses):** a portion of higher-severity
  `PKU-Alignment/BeaverTails` or PKU-SafeRLHF prompts.
- Optional breadth: `Anthropic/hh-rlhf` harmless-base, `allenai/wildguardmix`.

Assemble one mixed pool with both over-refusal and genuinely-harmful cases (not only clearly-harmful).
Hold out a validation subset and a disjoint fresh-confirmation subset. Record dataset versions and hashes.
Never use the spent 604-prompt sealed split.

## 3. Pre-register (write PREREG.md + selection_lock.json (+SHA) FIRST)

Before scoring, lock: the candidate helpfulness models, candidate harmlessness models, the dataset mix and
its manifest, the known-worse probe definitions, and the numeric thresholds below. Hash it.

## 4. Screen (inference only): conflict, resolution, headroom

Decode on the mixed pool with `scripts/revision/flagship/decode_vllm_non_thinking.py` (seed 42, non-thinking,
temp 0.7, top-p 0.9): `base` Qwen3-8B; per-objective known-worse probes (`over_refusing` and `terse` for
helpful-low, `answer_anything` for harmless-low); a small weak model; and the existing stable trained 8B
checkpoints. Score every set of responses with each helpfulness candidate and each harmlessness candidate.

For every helpfulness-by-harmlessness pair compute:
- **Conflict:** Spearman correlation between the two scores over the pooled per-prompt values, plus the top-1
  objective mismatch rate. Report the full matrix.
- **Resolution (per objective):** paired prompt bootstrap of base minus the matched known-worse policy; the
  95% CI must exclude 0. Report the minimum detectable effect. A saturated objective is discarded.
- **Headroom (per objective):** does any policy beat base with a CI excluding 0? The weaker objective must be
  beatable.

## 5. Selection rule

Select the single (helpfulness model, harmlessness model, dataset mix) that:
- has the **most negative Spearman** among pairs whose conflict is substantial (require Spearman <= -0.2, so
  it is a real tradeoff, not marginal noise), and
- passes resolution for **both** objectives, and
- has base headroom on at least the weaker objective, and
- uses **independent lineages** for the two objectives.
Tie-break toward the most widely used models and, all else equal, the backbone-matched
`Skywork-Reward-V2-Qwen3-8B` for helpfulness. Document why the winner is standard and defensible.

If no pair reaches Spearman <= -0.2 with resolution and headroom, report the honest negative: the available
standard reward models do not expose a helpful-harmless conflict that a Qwen3-8B policy can move, and the
model-scale claim stays scoped to 1.5B and the synthetic games.

## 6. Deliverables (into `results/p1_8b_hh_selection_20260716/`)

- `PREREG.md` + `selection_lock.json` (+SHA); dataset manifest (+SHA); scorer scripts for any new model.
- `CONFLICT.md` (full Spearman matrix + top-1 mismatch), `RESOLUTION.md` (base-vs-probe gaps + CIs + minimum
  detectable effect), `HEADROOM.md`.
- `SELECTION.md`: the locked helpfulness model, harmlessness model, and dataset mix, with the conflict,
  resolution, and headroom evidence and a short defensibility justification; plus the recommended RONPO run
  configuration to hand to the next phase.
- The committed script that regenerates every number from the score JSONL.
- `COMPLETION_AUDIT.md`, ending `spent_sealed_split_touched=false`.

## 7. One-paragraph brief (paste at top of the codex run)

> Select the concrete helpfulness reward model, harmlessness reward model, and dataset for a Qwen3-8B RONPO
> experiment on the helpful-vs-harmless conflict. Favor RONPO only legitimately, by finding genuine objective
> conflict on standard, widely used reward models and datasets, and pre-register the selection on conflict,
> resolution, and base headroom before any RONPO ranking, so it is not cherry-picked. Candidate helpfulness
> models: Skywork-Reward-V2 (Llama-3.1-8B, already local, and the backbone-matched Qwen3-8B), Athene, ArmoRM
> helpfulness head. Candidate harmlessness models (continuous score, different lineage): the Safe-RLHF
> beaver cost model, Aegis2.0, WildGuard, ShieldGemma-9B, Llama-Guard-3-8B, or Qwen3Guard, using a cost
> reward or a negated unsafe probability. Dataset: a safety-relevant mix of PKU-SafeRLHF (core), OR-Bench
> hard and XSTest (over-refusal, helpful-low), and higher-severity BeaverTails or PKU-SafeRLHF (harmful,
> harmless-low), because the current UltraFeedback prompts are benign and induce no conflict. Download any
> model or dataset as needed. Decode base, per-objective known-worse probes, a small weak model, and the
> stable trained 8B checkpoints on the mix; score with every candidate pair; and compute the Spearman
> conflict matrix, per-objective resolution (base minus a known-worse policy, 95% CI excluding zero, else
> saturated and discarded), and headroom (is base beatable). Select the independent-lineage pair with the
> most negative Spearman (require at most -0.2) that passes resolution for both objectives and has headroom
> on the weaker one, tie-breaking toward the most standard and backbone-matched models. If none qualifies,
> report the honest negative and keep the claim scoped to 1.5B. Do no training, no upload, never touch the
> spent sealed split, and regenerate every number from JSON.
