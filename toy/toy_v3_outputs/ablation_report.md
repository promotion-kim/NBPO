# RONPO Atom vs. k-only Adversary Ablation (toy_v3)

Config: n_iter=12000, alpha_pi=0.7, alpha_sigma=0.7, tau=0.015, kappa=0.015. All methods share the reference policy, hyper-parameters, and iteration budget; results are last-iterate. Every method is judged by the SAME honest metric floor_full = min_{k,a} E_{y~pi} P_k(y>a).

## Environment: separation

| Objective | major_spec | minor_spec | balanced | reference | weak |
|---|---:|---:|---:|---:|---:|
| major | 1.00 | 0.15 | 0.60 | 0.35 | 0.05 |
| minor | 0.15 | 1.00 | 0.60 | 0.35 | 0.05 |

Per-objective worst response argmax_a s_k(a): ['major_spec', 'minor_spec'] (indices [0, 1]) -> DIVERGE.

| Method | Reference | True floor (min_{k,a} C) | Self-perceived floor |
|---|---|---:|---:|
| Robust optimum (LP/MW) | - | 0.2568 | - |
| **Full atom adversary** | - | **0.2570** | 0.2570 |
| k-only | reference | 0.1192 | 0.7773 |
| k-only (best fixed ref) | balanced | 0.1846 | - |

**Irreducible atom advantage (full - best k-only): +0.0724**

## Environment: control

| Objective | dominant | hi_major | hi_minor | reference | weak |
|---|---:|---:|---:|---:|---:|
| major | 1.00 | 0.70 | 0.30 | 0.50 | 0.10 |
| minor | 1.00 | 0.20 | 0.80 | 0.40 | 0.10 |

Per-objective worst response argmax_a s_k(a): ['dominant', 'dominant'] (indices [0, 0]) -> SHARED.

| Method | Reference | True floor (min_{k,a} C) | Self-perceived floor |
|---|---|---:|---:|
| Robust optimum (LP/MW) | - | 0.4996 | - |
| **Full atom adversary** | - | **0.5000** | 0.5000 |
| k-only | reference | 0.5000 | 0.9241 |
| k-only (best fixed ref) | dominant | 0.5000 | - |

**Irreducible atom advantage (full - best k-only): -0.0000**

## Study C: worst-response divergence sweep

Delta(d) = full-atom floor - best-k-only floor, as objective worst-responses move from shared (d=0) to divergent (d=1).

adv(natural) = full - k-only(fixed reference response); the realistic model-scale setting. adv(best) = full - best k-only over ALL references; the adversarially generous lower bound on the atom advantage.

| d | worst responses | robust opt | full floor | k-only(ref) | best k-only | adv(natural) | adv(best) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.0 | 0=0 (shared) | 0.4996 | 0.5000 | 0.5000 | 0.5000 | -0.0000 | -0.0000 |
| 0.2 | 0=0 (shared) | 0.4996 | 0.5000 | 0.5000 | 0.5000 | -0.0000 | -0.0000 |
| 0.4 | 0=0 (shared) | 0.4996 | 0.5000 | 0.5000 | 0.5000 | -0.0000 | -0.0000 |
| 0.6 | 1=2 (diverge) | 0.3540 | 0.3538 | 0.3543 | 0.3543 | -0.0005 | -0.0005 |
| 0.8 | 1=2 (diverge) | 0.2596 | 0.2599 | 0.1419 | 0.1893 | +0.1180 | +0.0706 |
| 1.0 | 1=2 (diverge) | 0.2552 | 0.2555 | 0.2425 | 0.2543 | +0.0130 | +0.0012 |

## Verdict

- **separation** (worst responses diverge): atom advantage +0.0724 over the *best possible* k-only. Even the most generous single fixed reference cannot match the full adversary, because major and minor have different worst responses.
- **control** (worst responses shared): atom advantage -0.0000; the best k-only (ref = shared worst response) recovers the full adversary. Response selection buys nothing.
- The k-only(reference) run in the separation env is actively *fooled*: it perceives a floor of 0.7773 while its true worst-atom floor is 0.1192.
- **Sweep is non-monotone by design of the geometry, not noise.** full-RONPO tracks the robust optimum everywhere (gap <= 4e-4). The advantage peaks at intermediate divergence (d=0.8) where a cheap escape from the fixed reference exists, and shrinks at d=1.0 where extreme specialists force the policy to mix regardless. Divergent worst-responses are necessary but not sufficient.

Conclusion: full-RONPO reliably reaches the robust worst-objective optimum; the k-only ablation matches it only when a single fixed reference happens to expose the binding atoms of every objective. The atom advantage is exactly the value of *closing cheap escapes* that a fixed reference leaves open. A model-scale experiment must therefore engineer AND verify a conflict with such an escape (divergent, reference-evading worst responses); otherwise the full (k,a) adversary cannot beat -- and need not beat -- the k-only ablation. This is consistent with the observed model-scale null result.
