# The frozen SafeRLHF preference oracle

Three seeds (41, 42, 43) of each representation, matched on backbone, head
parameter budget, optimizer, steps, data, sequence length and selection budget.
The downstream oracle is the **calibrated three-seed ensemble mean**.

Trained on the pod's three H200s after another user took the local A100s; the
splits were **rebuilt on the pod from the deterministic builder rather than
copied**, and all six hashes — three file SHA-256 and three prompt-set SHA-256 —
match the local manifest exactly.

## Calibration: validation only

One temperature per (model, seed, objective), fitted by golden-section search on
validation NLL. A deterministic scalar search rather than a gradient fit, so the
calibration is a property of the model and not of an optimizer seed. **The test
split is never seen by anything that chooses a parameter.**

| model | seed | helpfulness | harmlessness |
|---|---|---|---|
| GPM | 41 / 42 / 43 | 1.290 / 1.288 / 1.275 | 1.266 / 1.294 / 1.305 |
| BT | 41 / 42 / 43 | 1.275 / 1.320 / 1.253 | 1.230 / 1.292 / 1.289 |

Every temperature is above 1: both architectures are **consistently
overconfident**, by a similar amount, and the amount is stable across seeds.

## Held-out test, calibrated three-seed ensemble

| model | objective | NLL | balanced acc | ROC-AUC | Brier | ECE | ensemble disagreement (sd) |
|---|---|---|---|---|---|---|---|
| **GPM** | helpfulness | **0.5503** | **0.7238** | **0.8033** | **0.1859** | 0.1228 | 0.0297 |
| **GPM** | harmlessness | **0.5143** | **0.7570** | **0.8503** | **0.1733** | 0.1779 | 0.0351 |
| BT | helpfulness | 0.5531 | 0.7219 | 0.8004 | 0.1869 | 0.1226 | 0.0296 |
| BT | harmlessness | 0.5150 | 0.7567 | 0.8495 | 0.1736 | 0.1780 | 0.0349 |

Mean pairwise seed correlation 0.960–0.968 for both architectures; ensemble
disagreement (per-example sd across seeds) 0.030–0.035.

Calibration helps NLL modestly and consistently: raw → calibrated ensemble NLL
0.5544 → 0.5503 and 0.5172 → 0.5143 for the GPM, 0.5569 → 0.5531 and 0.5166 →
0.5150 for BT.

**One honest caveat about calibration.** A single temperature reduces NLL but
does **not** fix ECE, and on harmlessness it slightly *worsens* it (0.176 →
0.179 per seed) while improving NLL (0.5246 → 0.5195). Temperature scaling
minimizes NLL, not calibration error, and a residual ECE near 0.18 on
harmlessness means the ensemble's confidence is still miscalibrated in a way one
scalar cannot absorb. This is reported rather than tuned away, because fixing it
with a richer calibrator would need a second validation budget the protocol has
not allocated.

## The architectural guarantees, per seed

| property | seed 41 | seed 42 | seed 43 |
|---|---|---|---|
| `P(y>z) + P(z>y) - 1` | 1.2e-07 | 1.2e-07 | 1.2e-07 |
| `P(y>y) - 0.5` | **0.0** | **0.0** | **0.0** |
| max cyclic logit residual (not scalar-decomposable) | 0.079 | 0.099 | 0.084 |

Antisymmetry and self-tie survive calibration and averaging exactly: temperature
scaling is monotone in the logit and averaging is linear in probability, so
probabilities that sum to one still do.

Head parameters: GPM 1 837 058, BT 1 772 546 — within 3.6 %, so "more capacity"
explains nothing in either direction.

## GPM versus BT — reported, not gated

| objective | ΔNLL | Δbalanced acc | ΔROC-AUC |
|---|---|---|---|
| helpfulness | −0.00275 | +0.00198 | +0.00297 |
| harmlessness | −0.00069 | +0.00028 | +0.00076 |

The GPM is very slightly ahead on all six comparisons after calibration and
ensembling — a more consistent sign than the single-seed runs gave, where the
direction flipped between runs — but the magnitudes are a fraction of a percent
and this is **not** a gate.

It cannot be one: SafeRLHF's comparison graph contains **zero triangles**
(`SAFERLHF_SUPERVISION.md`), so the dataset cannot identify the advantage a joint
anti-symmetric model has over a scalar one, which is the representation of
intransitive preference. The near-null belongs in the appendix as an honest
result about what this data can and cannot show.

## Predicted versus observed cycles, again kept apart

* **Observed** three-cycles in the human annotations: **0** in train, validation
  and test, for both objectives. Structural.
* **Predicted**, on 3 712 held-out triples per objective per seed: **0** for the
  GPM and **0** for BT.

The GPM's cyclic logit residual of 0.079–0.099 on real inputs establishes it is
free to be intransitive on this data. It predicts no cycles anyway. That is a
statement about this model on this dataset — **not** evidence that human
preferences are transitive, which these annotations cannot address either way.

## A confound that persists across seeds

The helpfulness head's logit correlates with the response-length difference at
**r = 0.42–0.44** in every seed, for both architectures. Whatever either model
learned about helpfulness on SafeRLHF is substantially a length preference.
Harmlessness is much cleaner at r ≈ 0.09.

## Role in what follows

* the **GPM ensemble** is the general pairwise interface the adaptive-game
  methods consume;
* the **BT ensemble** is the matched scalar-reward representation control;
* the natural SafeRLHF panel tests **cross-objective bargaining**, not cyclicity.

Artifacts: `saferlhf_ensemble.json`, `pred_{model}_seed{S}_{split}.npz`,
`gpm_vs_bt_seed{S}.json`.
