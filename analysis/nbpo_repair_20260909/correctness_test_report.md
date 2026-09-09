# Correctness test report — NBPO repair, 2026-09-09

Two kinds of evidence are kept separate below, because they answer different
questions: unit tests run on CPU against the production entrypoints, and
integration probes run the actual trainer against the real 8B model on GPU.

## 1. Unit tests, CPU, production entrypoints

Run against the repaired source at commit `288075e` on branch
`exp/iclr27-table1-v2`: **523 passed, 2 skipped, 0 failed** in 301 s, of which
70 are the repair's own tests (`test_nbpo_neural_repair.py`,
`test_nbpo_export_validation.py`, `test_collect_primary_results.py`,
`test_xstest_refusal.py`). The 2 skips are pre-existing and unrelated.

The nine checks the audit asked for, and where each lives:

| # | Requirement | Test |
|---|---|---|
| 1 | BT toy solved target → canonical builder target | `test_nbpo_neural_repair.py::test_dataset_contract_checks_pairs_masses_and_canonical_target` |
| 2 | Train and dev direct targets agree on the same input | one solver backend; verified on artifacts, see §3 |
| 3 | Independent certificate recomputed from the returned policy; bad results rejected | `test_nontransitivity_feasibility.py::test_exact_inner_solve_preserves_the_identity_where_no_bound_is_active` |
| 4 | Representation dispatch smoke test | `test_nbpo_neural_repair.py::test_actual_tiny_trainer_forward_and_reference_initialization` (both loss types) |
| 5 | Long bf16 log-probability change survives an FP32 reduction | `test_nbpo_neural_repair.py::test_long_bf16_response_logp_change_survives`, `::test_chunked_backward_matches_dense_full_vocabulary` |
| 6 | One candidate's tokens/context/mask identical across every partner | `test_nbpo_neural_repair.py::test_immutable_candidate_is_identical_across_partners_and_no_eos_is_invented`, `::test_precompute_collator_consumes_same_immutable_tokens` |
| 7 | Train/dev/test sentinels reach only the correct loader | `test_nbpo_neural_repair.py::test_explicit_split_sentinels_never_select_test` |
| 8 | No eta applied twice at η = 0.5 or 2 | `test_nbpo_neural_repair.py::test_production_canonical_mse_does_not_apply_eta_twice` |
| 9 | WBC all-pair loss and gradient equal the direct weighted teacher | `test_nbpo_neural_repair.py::test_wbc_all_pair_value_and_gradient_matches_direct_empirical_teacher` |

Requirement 9 has a companion that is easy to get wrong in the other direction:
`::test_neutral_finite_wbc_has_empirical_not_false_zero_gradient` pins that a
neutral teacher `p* = p_t` does **not** give a zero WBC gradient on a finite
sample, because 8 IID draws from a (0.7, 0.3) reference land at (0.875, 0.125)
and finite WBC correctly moves toward the empirical teacher. The MSE loss's
zero-target initialization identity is a different statement and is tested
separately.

Two failures the full-suite run surfaced were the tests being right about the
old code, and both are recorded in commit `288075e`: the identity test pinned
the exponential-map behaviour that the audit's finding C called a defect, and
the final-run validator caught the repository's own `final_iclr2027` config
still naming the retired Qwen3-32B judge.

## 2. Integration, real 8B model on GPU

These are not unit tests and are not simulated.

| Probe | Result |
|---|---|
| `probes/reference_init_v1`, `probes/reference_init_long_v1` | Actual trainer collator, frozen reference, longest pooled responses (1024 response tokens): `sequence_max_abs` 0.0, `pair_h_rms` 0.0, `reference_repeat_max_abs` 0.0 |
| `probes/deepspeed_cast_ablation_v1` | Same under the DeepSpeed cast path; rotary buffers observed as `torch.float32` |
| MSE arm, optimizer step 1 | `nbpo/h = 0.0`, `nbpo/h_rms = 0.0` against a nonzero target (`nbpo/target_abs = 2.672`) |
| Every optimizer step of both arms | logits, log-softmax, selected and sequence log-probabilities, loss, ZeRO-2 masters and Adam moments all `torch.float32`; forward parameters `torch.bfloat16`; `rotary_buffer_restorations = 33` |

## 3. Solver certificate, on the actual teacher

From `teachers/nash_repair_v2/inputs.json`, recomputed from the optimizer's
returned policy rather than from the map used to build the target:

| Quantity | Value | Gate |
|---|---|---|
| canonical serialization error | 0.0 | < 1e−9 |
| independent stationarity, ∞-norm | 6.713e−11 | < 1e−4 |
| extra-map residual | 1.272e−11 | < 1e−4 |
| Q recomputed at the returned policy | 0.0 | — |

`train`, `dev` and `test` were solved by one backend with the identical global
λ = (6.32539583922393, 6.185673374281126) fitted on train, so the held-out
splits solve only the fixed-weight inner problem. λ is interior to its box
[1e−3, 1e3]² by at least two decades in log₁₀, which is the condition under
which the projected residual certifies the original problem.
