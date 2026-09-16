# Prompt-wise NBPO: code snapshot, 2026-09-16

Byte-identical copy of the code that produced the per-prompt (prompt-wise) NBPO
results, taken from `/work/sub_20260914/code` on the campaign pod. Every file
here hashes the same as its source; the sha256 prefix is recorded per file.

The prompt-wise change is in the solver, not the trainer: NBPO's dual
multipliers are fitted separately at each prompt instead of being shared across
prompts. `solve_pros4_targets.py` is the shared-weight solver and the three
`solve_pros4_pw_*` / `solve_pros4_prosper*` files are generated from it by
`build_pw_solvers.py`, so the arms cannot drift apart in anything except the
rule they implement.

Arm to rule:

| Arm | Representation | Aggregation | Weight scope |
|---|---|---|---|
| NBPO-PW | adaptive game | Nash | per prompt |
| Fixed-reference Nash-PW | fixed reference | Nash | per prompt |
| PROSPER | adaptive game | absolute max-min | per prompt |
| NBPO (shared weights) | adaptive game | Nash | one vector for all prompts |

## Per-prompt solvers (the prompt-wise NBPO line)

- `build_pw_solvers.py` (340e8a981e8c7ac3) - single generator that builds all three per-prompt solvers from the pristine global solver, with count==1 assertions on every anchor
- `solve_pros4_targets.py` (f77d773e36e004b6) - the global solver the three are generated from: one weight vector shared across prompts
- `solve_pros4_targets_v2.py` (f77d773e36e004b6) - same solver, second campaign round
- `solve_pros4_pw_nbpo.py` (6fb577b54dfda813) - NBPO-PW: adaptive game, Nash aggregation, weights fitted per prompt
- `solve_pros4_pw_nbpo_v2.py` (6c56c611edb572df) - NBPO-PW on the scaled 522-prompt round
- `solve_pros4_pw_fixedref.py` (e8eda970156e1657) - fixed-reference Nash, per prompt
- `solve_pros4_pw_fixedref_v2.py` (9e7f7d973549fa2f) - fixed-reference Nash, per prompt, scaled round
- `solve_pros4_prosper.py` (4acef1091e95e5d1) - PROSPER max-min, per prompt (the comparator)
- `solve_pros4_prosper_v2.py` (e5788f0897387777) - PROSPER max-min, per prompt, scaled round
- `solve_pros.py` (6437b4763d8bae50) - earlier PROSPER-setting solver kept for provenance

## Feasibility and representability filters

- `probe_feasible.py` (1723afa0a22fd8e0) - per-prompt certification probe for the three rules, writes the surviving split
- `restrict_common_certified.py` (193c4e4e01aa7bf1) - intersects the arms so they train on one prompt set
- `restrict_v2.py` (b876b7b9118e4c94) - re-probes max-min at the matched multiplier norm and intersects again
- `filter_prompt_level.py` (d8fe2a6c6fd99db5) - prompt-granularity representability filter
- `filter_and_materialize.py` (348f23b16c22117a) - filter plus dataset materialization
- `clear_partial_targets.py` (ab27d66a0ef3c8af) - removes target directories that lack complete.json

## Pipelines and job generation

- `exp2_pipeline.py` (1718fc8fc28f8abb) - scaled round stage 1: judging, scoring, splits
- `exp2_stage2.py` (a4dc999d3f45c9fd) - scaled round stage 2: solve, filter, materialize, queue two-epoch training
- `make_pros_train_jobs.py` (bcabadf496632acb) - per-arm training config and queue spec generation
- `record_and_train.py` (f428f47382634dc4) - records the solve and queues the arm

## Analyses

- `sweep_l1.py` (bcbb8c5d65bd21cf) - multiplier-norm sweep behind the max-min feasibility claim
- `abl_analysis.py` (a1ae9271f1a860e1) - 2-draw versus 10-draw judgment-count ablation
- `analyze_screen.py` (841409da20c363de) - cycle, Condorcet and disagreement statistics, generalized to K objectives
- `analyze_wild.py` (f1226378f4072081) - the same statistics for checklist-native panels
- `wild_selftest.py` (50984f97c019e949) - hand-computed synthetic check for analyze_wild
- `paired_diff.py` (8c02d7c6107a1add) - paired prompt bootstrap for the difference columns

## PROSPER-setting data pipeline

- `generate_pros_pool.py` (7792cdc6dee904c3) - candidate generation at the protocol decoding
- `judge_pool_prosper.py` (0cb23a8b90050bd0) - the five-point single-check judge template, transcribed from the paper
- `score_pros_pool.py` (13f9dcb7dfe96f23) - Bradley-Terry fits and the frozen teacher record
- `merge_ext.py` (f4a16ea8e9dd158e) - merges judgment shards across extensions
- `unify_scoring.py` (1788095d71d88995) - re-scores every prompt in one pass so one teacher record covers them

## Evaluation

- `build_eval_panels.py` (8183ed5efbec9580) - freezes the Arena-Hard and AlpacaEval panels and their static baselines
- `gen_candidates.py` (70e0ac18c3662c32) - greedy generation for evaluation
- `judge_eval_pairwise.py` (bf74b5821f6527ff) - pairwise judging against the static baselines, both orders, ties 0.5

## Not included

The pod's deployed trainer (`/work/nbpo_repair_20260909/code/mnpo_scripts`) also
differs from this repository in `run_mnpo.py` and `nbpo_neural.py`, but those
changes add MOPO importance-weight dataset validation and belong to a different
line of work, so they are left out of this snapshot rather than mixed into it.
`prepare_nbpo_dataset.py` is already identical in both places.
