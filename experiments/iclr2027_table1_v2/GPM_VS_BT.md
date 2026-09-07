# Anti-symmetric GPM vs scalar Bradley-Terry, on SafeRLHF human labels

Trained 2026-09-08 on the prompt-disjoint splits described in
`SAFERLHF_SUPERVISION.md`. Results: `results/iclr2027_table1_v2/saferlhf_gpm/`.

## Matched budget, one difference

Both models share the backbone (`roberta-base`, fine-tuned end to end), the
optimizer, the schedule, the data order, the seed and the loss. The only
difference is the head:

* **BT** `logit_k = r_k(h_y) - r_k(h_z)` -- a scalar reward difference,
  transitive by construction;
* **GPM** `logit_k = ½[a_k(h_y,h_z) - a_k(h_z,h_y)]` -- a joint pair score,
  antisymmetrized exactly.

Labels are the released binary ids. **No tie is invented** -- the schema cannot
express one -- and no label is softened.

| | GPM | BT |
|---|---|---|
| head parameters | 1 837 058 | 1 772 546 |
| total parameters | 126 482 690 | 126 418 178 |
| training seconds | 419 | 420 |

The BT head is widened (512 → 768) so the two are within 3.6 % on head
parameters; "more capacity" is not available as an explanation either way.

## The architectural guarantees hold exactly

| property | measured |
|---|---|
| `P(y>z) + P(z>y) - 1` | max **1.2e-07** (float32 sigmoid precision) |
| `P(y>y) - 0.5` | max **0.0**, exactly |
| not decomposable into `r(y) - r(z)` | max cyclic logit residual **0.081** on random inputs |

A scalar model has cyclic residual identically 0 for every triple, so a materially
nonzero residual is proof the trained GPM is not a Bradley-Terry model in
disguise.

## Held-out test (3 754 prompts, 6 955 rows, prompt-disjoint from training)

| objective | model | NLL | balanced acc | ROC-AUC | Brier | ECE | logit vs length-diff |
|---|---|---|---|---|---|---|---|
| helpfulness | **GPM** | **0.5573** | 0.7228 | **0.8007** | **0.1873** | 0.1229 | 0.416 |
| helpfulness | BT | 0.5612 | 0.7230 | 0.7984 | 0.1879 | 0.1248 | 0.417 |
| helpfulness | constant | 0.6555 | 0.500 | 0.500 | — | — | — |
| harmlessness | GPM | 0.5248 | 0.7536 | 0.8458 | 0.1766 | 0.1748 | 0.092 |
| harmlessness | **BT** | **0.5234** | 0.7542 | 0.8457 | **0.1758** | 0.1758 | 0.089 |
| harmlessness | constant | 0.6042 | 0.500 | 0.500 | — | — | — |

(Run 2, same seed and configuration, different GPU: GPM 0.5587 / 0.7219 / 0.7996
and 0.5232 / 0.7536 / 0.8469; BT 0.5612 / 0.7214 / 0.7982 and 0.5244 / 0.7510 /
0.8440. Both runs are in `gpm_vs_bt_run1.json` and `gpm_vs_bt.json`.)

**Both models clear the constant-predictor floor decisively** — NLL 0.56/0.52
against 0.66/0.60, balanced accuracy 0.72/0.75 against 0.50 — so the route is not
stopped on that clause.

## GPM minus BT, cluster-bootstrapped over prompts (2 000 draws), twice

The configuration was trained **twice** at the same seed, differing only in the
GPU and in nondeterministic kernel scheduling. Both runs are reported, because
the difference between them is the finding.

| objective | run | ΔNLL (GPM − BT) | 95 % CI | Δaccuracy | 95 % CI |
|---|---|---|---|---|---|
| helpfulness | 1 | −0.00389 | [−0.00732, **−0.00050**] | −0.00030 | [−0.00616, +0.00574] |
| helpfulness | 2 | −0.00241 | [−0.00586, **+0.00119**] | −0.00032 | [−0.00603, +0.00549] |
| harmlessness | 1 | +0.00140 | [−0.00236, +0.00529] | −0.00187 | [−0.00808, +0.00469] |
| harmlessness | 2 | −0.00118 | [−0.00520, +0.00283] | +0.00264 | [−0.00405, +0.00908] |

Rows sharing a prompt are not independent, so prompts are the resampling unit.

Run 1's single significant result — a 0.0039-nat NLL gain on helpfulness, 0.6 %
of the loss — **did not replicate**. At the same seed and configuration, run 2
gives −0.0024 with an interval that includes zero, and harmlessness flips sign
between runs. The effect is smaller than run-to-run variance. No accuracy
interval excludes zero in either run.

**On this data the joint anti-symmetric head buys nothing reliable over a scalar
reward difference.**

## Why that is the expected answer here, and what it does and does not mean

The extra expressiveness a GPM has over BT is exactly the ability to represent
intransitive preference. SafeRLHF's annotation graph is a **near-perfect
matching with zero triangles** (`SAFERLHF_SUPERVISION.md`), so there is no
intransitivity in the supervision for the GPM to fit and no observable
nontransitive subset on which to evaluate the difference. A null result is what
the data structure predicts; it is **not** evidence that human preferences are
transitive, and it must not be reported as such.

The pre-registered stop rule reads: *stop if the GPM does not beat a constant
predictor, or has no defensible benefit over BT on any observable nontransitive
subset.* The first clause is passed. The second **cannot be evaluated** — the
subset is empty by construction of the annotation protocol. That is reported as
unavailable, not as a pass and not as a failure.

## Observed versus predicted cycles

These are kept strictly apart, and together they make the sharpest point in this
section.

**Observed** three-cycles in the human annotations: **0** in train, 0 in
validation, 0 in test, for both objectives. Structural, not statistical — the
annotation graph has no triangles to count.

**Predicted** cycles, on 400 held-out prompts with at least three distinct
responses, 3 392 triples per objective — triples annotators never compared, so
these are model predictions and nothing else:

| model | max cyclic logit residual on **real** encodings | predicted 3-cycles, helpfulness | harmlessness |
|---|---|---|---|
| **GPM** | **2.414** | **0 / 3 392** | **0 / 3 392** |
| BT | **0.000** (exactly, as it must be) | 0 / 3 392 | 0 / 3 392 |

The GPM's cyclic residual of 2.41 on *real* response encodings — not on random
vectors — establishes that the trained model is genuinely free to be
intransitive on this data. It has the capacity, on these very inputs. And it
predicts **zero cycles in 3 392 triples**.

So a model that can be intransitive, trained on human preference data, learns a
transitive preference. That is a much stronger statement than the null NLL
comparison, and it is the honest form of the claim: it is evidence about what
this model learned from this dataset, **not** evidence that human preferences are
transitive — the annotations cannot speak to that either way.

## A confound worth naming

The helpfulness head's logit correlates with the response-length difference at
**r = 0.42** — for both architectures, essentially identically (0.416 vs 0.417).
Whatever either model learned about helpfulness on SafeRLHF is substantially a
length preference. Harmlessness is much cleaner at r ≈ 0.09.
