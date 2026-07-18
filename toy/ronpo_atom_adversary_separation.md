# RONPO Atom-Adversary Separation Toy

This construction makes the failure depend on a specific decoy response `a*`, not only on the objective weight.
The current policy puts mass on `policy_main` and `policy_alt`; `fixed_ref` is the response used by a weight-only ablation.

| Objective | policy_main | policy_alt | fixed_ref | decoy_a_star |
|---|---:|---:|---:|---:|
| major_quality | 1.00 | 0.90 | 0.00 | 0.10 |
| minor_decoy_sensitive | 0.10 | 0.20 | 0.00 | 1.00 |

| Adversary | Objective | Response atom | E_pi P_k(y beats a) |
|---|---|---|---:|
| k-only fixed-a | major_quality | fixed_ref | 0.9996 |
| k-only fixed-a | minor_decoy_sensitive | fixed_ref | 0.7184 |
| full (k,a) worst | minor_decoy_sensitive | decoy_a_star | 0.0009 |

Weight-only worst case with fixed reference: `minor_decoy_sensitive`, value `0.7184`.
Full atom adversary worst case: `(minor_decoy_sensitive, decoy_a_star)`, value `0.0009`.

The fixed-reference adversary concludes that every objective is protected because the policy beats `fixed_ref` under both objectives.
The full atom adversary instead selects `(minor_decoy_sensitive, decoy_a_star)`, exposing a low floor that objective-only weighting cannot see.
