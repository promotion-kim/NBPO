# P5 completion audit

P5 Stage-1 comparison plus Stage-2 training/gating. Stage-2 reward evaluation was not run because no RONPO Stage-2 arm was eligible.

## Stage-1

The descriptive fixed-panel comparison contains 49 prompts. Its source summary is `/NHNHOME/AIPR/sjkim/MNPO_rev_20260710/results/p5_8b_robust_stage1_stage2_20260717/stage1/fixed_p4_validation/results/ranked_validation_summary.json`.

## Stage-2 gate decision

| RONPO arm | Gate | Length ratio |
|---|---|---:|
| ronpo_os_stage2 | failed | 2.2636 |
| ronpo_topmass_stage2 | failed | 2.0964 |

No RONPO Stage-2 arm passed the unchanged stability gate; a RONPO-versus-baseline Stage-2 reward comparison would be invalid.

## Diagnostic-only score inspection

A 49-prompt diagnostic score table reused existing generations and retained the failed RONPO rows with explicit labels. It is not a paper ranking or selection result: `/NHNHOME/AIPR/sjkim/MNPO_rev_20260710/results/p5_8b_robust_stage1_stage2_20260717/stage2/fixed_p4_validation_diagnostic_including_stability_failed/results/diagnostic_summary.json`.

## Storage and publication

No optimizer state or intermediate checkpoint was pruned. The uploaded Stage-1 top-mass checkpoint remains locally because it is the Stage-2 parent and all Stage-2 evidence is retained pending review.

No spent sealed split was read or modified.

## Hash-backed inputs

| Artifact | SHA-256 | Bytes |
|---|---|---:|
| `/NHNHOME/AIPR/sjkim/MNPO_rev_20260710/results/p5_8b_robust_stage1_stage2_20260717/stage1/fixed_p4_validation/results/ranked_validation_summary.json` | `f11bff97bc20bc2965eec6ef99056756393ff1fd27c70cfcab3dc9549aa94441` | 8660 |
| `/NHNHOME/AIPR/sjkim/MNPO_rev_20260710/results/p5_8b_robust_stage1_stage2_20260717/stage2/stage2_training_gate_summary.json` | `15bfe192322a1bcf13eeec2bbe4867309f9518ab5c5b26ce85f4bd6ba641f65c` | 8733 |
| `/NHNHOME/AIPR/sjkim/MNPO_rev_20260710/results/p5_8b_robust_stage1_stage2_20260717/run_lock.json` | `e2418a86bed49186996ae496d6b2eda70cfcbba16df347777d2820edb8cbf87b` | 1727 |
| `/NHNHOME/AIPR/sjkim/MNPO_rev_20260710/results/p5_8b_robust_stage1_stage2_20260717/hf_uploads.jsonl` | `38d2cf8f8f7a3763d9031a3fe53239d9afede5418677eb36a64355304024f186` | 718 |
| `/NHNHOME/AIPR/sjkim/MNPO_rev_20260710/results/p5_8b_robust_stage1_stage2_20260717/stage2/ronpo_os_stage2/stability_validation/gate.json` | `1856e4cdc8e18e212176dbc9d6399d4384a3fcacf99b1047ba476c9198fcb8af` | 1800 |
| `/NHNHOME/AIPR/sjkim/MNPO_rev_20260710/results/p5_8b_robust_stage1_stage2_20260717/stage2/ronpo_topmass_stage2/stability_validation/gate.json` | `605ce1e17abbf3780e077861ca6d128f31fb3ce68e6af29f5129bc74f1504ed3` | 1802 |
| `/NHNHOME/AIPR/sjkim/MNPO_rev_20260710/results/p5_8b_robust_stage1_stage2_20260717/stage2/inpo_avg_stage2/stability_validation/gate.json` | `9fda6b1538c9ced4f30a57ce7e4c2fbada4e5f8fb43eb3cf3694a682d787bbf8` | 1799 |
| `/NHNHOME/AIPR/sjkim/MNPO_rev_20260710/results/p5_8b_robust_stage1_stage2_20260717/stage2/simpo_stage2/stability_validation/gate.json` | `c18ea715590bc22c7d38b2b4a72b38bfc3f78a5d74a7a1da229433785c4d7762` | 1801 |
| `/NHNHOME/AIPR/sjkim/MNPO_rev_20260710/results/p5_8b_robust_stage1_stage2_20260717/stage2/ipo_stage2/stability_validation/gate.json` | `ae72acf6d4092b96c0580009ce7c5dc3e7cf7c52b6ffde4b5dcce4c5e08559c1` | 1800 |
| `/NHNHOME/AIPR/sjkim/MNPO_rev_20260710/results/p5_8b_robust_stage1_stage2_20260717/stage2/dpo_stage2/stability_validation/gate.json` | `0d53abb53788ee613cf09cb22646bf39371b46323e4d635d5ae8880ce1b17a5d` | 1799 |
| `/NHNHOME/AIPR/sjkim/MNPO_rev_20260710/results/p5_8b_robust_stage1_stage2_20260717/stage2/sppo_avg_stage2/stability_validation/gate.json` | `a72f25526628dec69bce15beb3fa59457621f48a0c1318c1645c88fd037f3807` | 1799 |
| `/NHNHOME/AIPR/sjkim/MNPO_rev_20260710/results/p5_8b_robust_stage1_stage2_20260717/stage2/ht_mnpo_harmless_stage2/stability_validation/gate.json` | `0152773e0d9e3cbfaa9442b4adebf7f44dabccfea4bab3ea78043b2d48d20c04` | 1799 |
| `/NHNHOME/AIPR/sjkim/MNPO_rev_20260710/results/p5_8b_robust_stage1_stage2_20260717/stage2/ht_mnpo_helpfulness_stage2/stability_validation/gate.json` | `e3452b5af6d1e87a2cbbdef53f1a8b16a9b5daefce3c1e254de3307d0a999241` | 1800 |
| `/NHNHOME/AIPR/sjkim/MNPO_rev_20260710/results/p5_8b_robust_stage1_stage2_20260717/stage2/fixed_p4_validation_diagnostic_including_stability_failed/results/diagnostic_summary.json` | `c16634763f5b16dc55f8bac8ae69dbb4d8ed23ac5a8132d1a55e66b8ac6fc265` | 7804 |
