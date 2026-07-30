# RONPO — Phase 0/1 "run this now" spec (critical path for the AAAI decision)

**Goal of this week (by ~July 8, go/no-go):** decide whether the atom `(k,a)` adversary
actually beats the weight-only `k`-only adversary once the *selection* bug is fixed.
This single result determines which paper you submit (Branch A vs Branch B).

Everything below is code/config you drop into your existing repo
(`/home/sjkim/MNPO/experiments/ronpo_safety_conflict_qwen25_1p5b_...`). The per-atom
mass `w̄_{k,a}` is already computed in your trainer; the fix is to **stop truncating it**.

---

## 0. The one change that matters: `argmax` → **full expectation**

Your adversary is a categorical `s_t` over atoms `(k,a)`, updated by multiplicative
weights (= entropic OMD = Bayesian/exp-family posterior with a KL-to-`s0` prior).

- **Keep the update rule.** MW is load-bearing for the last-iterate proof: the atom
  cost is adversarial / non-stationary, so you need a no-regret (EXP3/MW) update, **not**
  a stationary Thompson/Bayesian resampler. Do **not** switch this to "Bayesian update
  then sample."
- **Change the selection.** Right now you form the policy loss on the **top-mass
  atom(s)** (`argmax_s`). That (i) breaks the Prop-2 unbiasedness, (ii) contradicts the
  trained soft-min/`κ` objective (argmax = hard `κ→0` regardless of the `κ` you trained),
  (iii) is discontinuous/unstable, and (iv) is the prime suspect for the length collapse
  you saw (`full atom words 348 vs base 176`, brevity 0.209).

### Selection variants to compare (fixed everything else)
```
SELECT ∈ { argmax,        # current (baseline, expected to be worst)
           sample,        # (k,a) ~ s_t
           expectation }  # NEW default: sum over ALL atoms weighted by s_t
```

### Pseudocode of the policy-loss assembly (per prompt x, per step t)
```python
# scores[k, a] = plug-in preference score used to build the target (already computed)
# s_t[k, a]    = current adversary mass over atoms (already computed; sums to 1)
# y, y' ~ p_hat_t : two distinct sampled responses for the regression pair

def policy_loss_for_prompt(s_t, atoms, p_hat, model, select):
    if select == "argmax":
        idx = topk_mass(s_t, k=cfg.top_atoms)                 # CURRENT behavior
        pairs = [make_pair(atoms[k,a]) for (k,a) in idx]
        weights = [1.0/len(pairs)] * len(pairs)
    elif select == "sample":
        (k,a) = categorical_sample(s_t)                       # one atom ~ s_t
        pairs = [make_pair(atoms[k,a])]
        weights = [1.0]
    elif select == "expectation":                             # NEW default
        # NO truncation: every atom contributes, weighted by its mass s_t[k,a].
        pairs, weights = [], []
        for (k,a) in atoms.all():
            w = s_t[k, a]
            if w < cfg.mass_eps:      # numerical prune only (e.g. 1e-6), NOT top-k
                continue
            pairs.append(make_pair(atoms[k,a]))
            weights.append(w)
        # weights already sum to ~1 over kept atoms; renormalize after eps-prune
        weights = normalize(weights)

    # signed INPO-style log-ratio regression target, unchanged:
    loss = 0.0
    for (pair, w) in zip(pairs, weights):
        z   = signed_target(pair)                 # eta_t * Delta_t for that atom
        h   = logratio(model, pair.y, pair.yprime) \
              - eta*tau*logratio(mu,   pair.y, pair.yprime) \
              - (1-eta*tau)*logratio(p_t, pair.y, pair.yprime)
        loss += w * (h - z)**2
    return loss
```
`expectation` is zero-variance in the atom choice, unbiased, and consistent with the
`κ`-softmin you actually trained. It reuses the atom scores you already precompute, so
the extra cost is a `K×A` weighted sum (small).

**Do not** control peakedness with `argmax`. If you want a sharper adversary, lower `κ`
— that is the knob the theory certifies.

### Logging you MUST add (this is how you'll read the result)
- `H(s_t)` = entropy of the adversary each step (collapse detector).
- marginal objective mass `Σ_a s_t[k,a]` per objective `k` (is it finding the safety axis?).
- mean generation length vs step, per config (length-drift detector).
- worst-objective normalized reward on a fixed held-out set vs step.

---

## Phase 0 — prove the conflict exists BEFORE training (1–2 days, no training)

The safety axis was inert last time (all three models ~0.5 safety) → the run collapsed
into a helpfulness↔brevity length fight. Fix the setup, then verify conflict.

1. **Upgrade the safety judge**: `Qwen3Guard-Gen-0.6B` → **`Qwen/Qwen3Guard-Gen-4B`**
   (tri-class safe/controversial/unsafe; map to a scalar). 0.6B was too weak to move.
2. **Use borderline prompts** so safety actually bites: mix
   **XSTest**, **BeaverTails**, **WildJailbreak/WildGuardMix** into the prompt pool
   (not just UltraFeedback — those don't exercise harmlessness).
3. **Generate once per prompt from the base model**, score every response with all three
   axes (helpfulness = `Skywork-Reward-V2-Llama-3.1-8B`, safety = `Qwen3Guard-Gen-4B`,
   brevity = deterministic target-length band — see below), and compute the **3×3
   correlation matrix** of the axis scores across responses.

**Gate:** you need at least one **negative** off-diagonal correlation (e.g. brevity vs
helpfulness, or safety vs helpfulness on the jailbreak subset). If everything is
positively correlated, there is no conflict to protect and no experiment will show the
atom advantage — fix the pool/judges first. Save the matrix as a figure; it is Phase-0
evidence for the paper.

**Brevity as a BAND, not a monotone penalty.** Reward peaks in a target window
(e.g. 80–200 tokens) and *decays on both sides*. A monotone "shorter = better" reward is
exactly what got hacked (and what dragged your metrics). Center the band on the base
model's median length so brevity is orthogonal to raw length.

---

## Phase 1 — the core 6-cell grid (2–4 days on 2×A100 + H200)

`SELECT {argmax, sample, expectation} × ADVERSARY {k-only, full-(k,a)}` = **6 runs**,
seed 1 first.

**Non-negotiable controls (these caused the false-negative last time):**
- **Matched checkpoints.** Last time full-atom was ckpt-3152 vs k-only ckpt-2227 — a
  confound. Select every run's reported checkpoint by the **same rule**: best
  **held-out worst-objective floor**, *not* `eval_loss`, at the *same* step budget.
- **Same seed, same data order, same pool `A(x)` construction** across the 6 cells.
- **Length control on** (target-length band + de-bias the helpfulness RM's length bias;
  see Phase 2) so the comparison isn't a length artifact.

**Read the result (pre-registered):**
- `expectation, full-(k,a)` **≥** `expectation, k-only` on worst-objective floor
  ⟹ **Branch A**: atom adversary validated. Scale to ≥3 seeds + baselines, aim AAAI.
- `full ≈ k-only`  ⟹ **Branch B**: atom becomes a theoretical/diagnostic extension;
  headline the robust-formulation result (k-only already beats scalarization/averaging).
- `full < k-only` even with expectation+length-control ⟹ **Branch B (pivot)**: the paper
  is "robust formulation beats scalarization/MaxMin," which your Stage-1/2 already show.

All three branches are submittable. You only need Phase 1 to know which one you're writing.

---

## After the go/no-go (only if Branch A, weeks 2–3)

- **Length control (Phase 2):** de-bias helpfulness RM, target-length band, 4-cell check.
- **Money table (Phase 3):** base / fixed-avg / tuned-avg / single-oracle×3 /
  **MaxMin-RLHF** / **Group-Robust-PO** / RONPO-k-only / RONPO-full, **≥3 seeds**,
  paired-bootstrap 95% CIs on high judge-disagreement subsets (+ CVaR/lower-tail).
- **Held-out eval (Phase 4):** different-family RM + external judge (GPT-5.5) on
  AlpacaEval-2 / Arena-Hard / MT-Bench / XSTest / HarmBench.

The paper's `% TODO` block (in Limitations) lists exactly these; fill Tables as results
land. **Do not fabricate any number** — leave a cell blank rather than estimate it.
