# NBPO code evidence

Source: uploaded `NBPO-main.zip`, archive comment `92f75310188d11be8141be4da7d207c33b356162`.

Line numbers are original repository file lines, not Markdown lines. These excerpts support the audit; comments are distinguished from executable statements in the report.

## E01 — README describes the old public entry path

Path: `README.md`

SHA256: `5e81387436777df46997add1c86de196c45f200d2835aacbe239c3e1de739965`

Original lines 29–71:

```text
   29 ## Two NBPO pipelines: finite-pool NBPO realization vs legacy fixed-reference
   30 
   31 The repository contains two distinct implementations. Do not conflate them.
   32 
   33 **`scripts/nbpo/` — finite-pool NBPO realization of Algorithm 1.** The
   34 manuscript's construction: adaptive KL-regularized opponents
   35 ν\*<sub>k,π</sub> ∝ μ·exp(−r/β<sub>k</sub>) (Eq. 7), soft-min game values
   36 V<sub>k,β</sub> (Eq. 8), a measured disagreement point
   37 d<sub>k</sub> = V<sub>k,β</sub>(μ) (Eq. 10, never assumed zero; the
   38 reference-vs-reference tensor is exactly skew-symmetric with a zero diagonal by
   39 construction), projected dual descent on the **raw** multipliers
   40 λ ← Π<sub>Λ</sub>[λ − γ(ŝ − 1/λ)] (Eq. 27), pairwise regression to
   41 (h<sub>t</sub> − η Σ<sub>k</sub> λ<sub>k</sub>Z<sub>k</sub>)² with sequence-sum
   42 log-probabilities (Eq. 26, `loss_type: nbpo`), and the held-out
   43 stage-acceptance gate of Algorithm 1. **The dual solver lives in
   44 `mnpo_scripts/nbpo_solver.py`** (CLI: `scripts/nbpo/solve_nbpo_dual.py`).
   45 
   46 Two disclosed approximations, stated plainly:
   47 
   48 - **R = 1.** The released configuration performs one opponent reweighting per
   49   dual update instead of iterating the policy–opponent fixed point to
   50   convergence. The solver exposes `R`, reports the fixed-point residual and a
   51   one-extra-map residual, and writes the *update* opponent (`nu_update.npz`,
   52   what Eq. 26 samples z<sub>k</sub> from) separately from the opponent
   53   recomputed at the final policy (`nu_final_policy.npz`, diagnostics).
   54 - **One-shot neural realization after the frozen-pool dual.** The M = 4e3–3e5
   55   dual iterations run on the frozen finite response pool — cheap tensor updates
   56   — and the neural policy is fit **once** afterwards from the resulting pair
   57   targets. No 8B model is retrained per dual step.
   58 
   59 λ is raw throughout: any normalized weights in logs are display diagnostics
   60 only. The solver reports both the inverse-surplus residual ‖ŝ − 1/λ‖<sub>∞</sub>
   61 and the projected (box-aware) KKT residual with the active-bound coordinates;
   62 λ = 1/s is an empirical equality only where no box bound is active.
   63 
   64 **Provenance binding in real mode.** `run_nbpo_stage.py` materializes the
   65 run_mnpo YAML and parses it with run_mnpo's own argument dataclasses before any
   66 model loads; the precompute sidecar must bind history0 to the parent
   67 checkpoint's *content* fingerprint (every weight shard hashed — tokenizer
   68 equality is not weight equality); the candidate is decoded synchronously by
   69 `scripts/nbpo/decode_candidate.py`, whose manifest binds the responses to the
   70 candidate fingerprint, the exact monitoring prompt set, every seed and every
   71 file hash; every candidate and reference seed file must carry exactly the
```

Original lines 105–159:

```text
  105 and manifest-binding path is exercised end to end by
  106 `tests/test_nbpo_realmode.py` with stub executables and no LLM (this is the
  107 test that had to pass before that phrase is used here). A full real-mode run
  108 with 8B models through this exact path has not yet been executed; the
  109 finite-pool math and the trainer branch match the manuscript equations, the orchestration is
  110 verified at stub level.
  111 
  112 **`scripts/bpo/` — fixed-reference Anchored-BPO (β = ∞ baseline, legacy).**
  113 Fixed-anchor surpluses s<sub>k</sub> = P<sub>k</sub>(π≻μ) − ½ against the frozen
  114 reference, static normalized/clipped weight rules, token-mean log-probabilities,
  115 auxiliary anchor/pref-SFT losses in some drivers, and an evaluator
  116 (`eval_bpo_surplus.py`) that clamps nonpositive surpluses. Kept runnable, header-labeled,
  117 and used as the β→∞ baseline; not Algorithm 1.
  118 
  119 The evaluators differ the same way: `scripts/nbpo/eval_game_value.py` computes
  120 V, d, and surpluses at finite β and reports Nash welfare as `null` whenever any
  121 surplus is nonpositive (no clamping); `scripts/bpo/eval_bpo_surplus.py` is the
  122 fixed-reference diagnostic.
  123 
  124 ### Finite-pool NBPO stage, command by command
  125 
  126 ```bash
  127 export PYTHONPATH=$(pwd)
  128 
  129 # 1. Judge the full pairwise matrix (policy-vs-ref AND ref-vs-ref for d_k),
  130 #    both presentation orders, retries, loud completeness failure.
  131 python -m scripts.nbpo.judge_pairwise_matrix \
  132   --policy-files s0=pol0.json s1=pol1.json s2=pol2.json s3=pol3.json \
  133   --reference-files r0=ref0.json r1=ref1.json r2=ref2.json r3=ref3.json \
  134   --objectives-config training_configs/nbpo/objectives/ultrafeedback.yaml \
  135   --objectives instruction_following,truthfulness,honesty,helpfulness \
  136   --judge-model-path Qwen/Qwen3-32B --backend vllm --output verdicts.jsonl
  137 
  138 # 2. Swap-average into centered float64 tensors A_policy and A_ref.
  139 python -m scripts.nbpo.build_preference_tensor \
  140   --verdicts verdicts.jsonl --policy-files s0=pol0.json ... \
  141   --reference-files r0=ref0.json ... \
  142   --objectives instruction_following,truthfulness,honesty,helpfulness \
  143   --objectives-config training_configs/nbpo/objectives/ultrafeedback.yaml \
  144   --out-dir tensor/
  145 
  146 # 3. Projected dual descent on the frozen pool (Eq. 27). Matched controls via
  147 #    --aggregation utilitarian|absolute_maxmin|surplus_maxmin.
  148 python -m scripts.nbpo.solve_nbpo_dual --tensor-dir tensor/ --out-dir solver/ \
  149   --beta 0.25 --eta 1.0 --gamma 0.5 -M 40000 -R 1 --aggregation nash
  150 
  151 # 4. Pair targets: six unordered pairs, one z_k ~ nu_update per pair AND objective (shared by
  152 #    y and y' of that row), sampled Bernoulli Z_k (Eq. 24), UNSCALED sum_k lambda_k Z_k
  153 #    (eta applied in the trainer). The builder verifies the solver's nu_update.npz hash.
  154 python -m scripts.nbpo.build_nbpo_pairs --tensor-dir tensor/ --solver-dir solver/ \
  155   --policy-files s0=pol0.json ... --out-dir pairs/ --target-mode sampled
  156 
  157 # 5. Precompute logps with SEQUENCE-SUM reduction (writes precompute_meta.json,
  158 #    validated at training time), then train the loss_type=nbpo branch.
  159 python -m mnpo_scripts.precompute --logp_reduction sum --ronpo_target_mode none \
```

## E02 — Current campaign snapshot scope and excluded run artifacts

Path: `analysis/sub_20260914/code_snapshot_20260917_union/README.md`

SHA256: `d9eeb6ea5e68bc8b0db822914c098afe34887c7467b393c437abab63b87f088c`

Original lines 1–46:

```text
    1 # Campaign code snapshot, 2026-09-17 (current)
    2 
    3 Byte-identical copy of `/work/sub_20260914/code` on the campaign pod, plus the
    4 dependency-queue controller from `/work/uf4_20260910/code/queue_controller.py`.
    5 Every file here hashes the same as its source at the time of the copy; the
    6 sha256 prefix is recorded per file in the table at the end.
    7 
    8 This directory is the **current** state of the pipeline. The sibling
    9 `code_snapshot_20260916/` is left frozen on purpose: it records the exact code
   10 that produced the per-prompt (prompt-wise) NBPO numbers reported at that date,
   11 and several of its files have since been edited here, so the two disagree by
   12 design. Where a file appears in both, the 20260916 hash is the provenance of
   13 the published number and the hash here is the code as it now stands.
   14 
   15 ## The prompt-wise NBPO line
   16 
   17 The prompt-wise change is in the solver, not the trainer: NBPO's dual
   18 multipliers are fitted separately at each prompt instead of being shared across
   19 prompts. `solve_pros4_targets.py` is the shared-weight solver, and every
   20 per-prompt and per-panel variant is *generated* from it by a single generator,
   21 so the arms cannot drift apart in anything except the rule they implement.
   22 
   23 | Arm | Representation | Aggregation | Weight scope |
   24 |---|---|---|---|
   25 | NBPO-PW | adaptive game | Nash | per prompt |
   26 | Fixed-reference Nash-PW | fixed reference | Nash | per prompt |
   27 | PROSPER | adaptive game | absolute max-min | per prompt |
   28 | NBPO (shared weights) | adaptive game | Nash | one vector for all prompts |
   29 | MOPO | adaptive game | importance-weighted cloning | per prompt |
   30 
   31 - `build_pw_solvers.py` — generates the three per-prompt solvers from the
   32   pristine global solver, with `count == 1` assertions on every anchor.
   33 - `build_panel_solvers.py` — the same generator for the US/UT/UW panels. It
   34   also rewrites the hard-coded `K = 4` to `K = len(base.OBJECTIVES)` and then
   35   imports every module it generated to assert the objective count agrees. A
   36   generated probe that kept `K = 4` on a two-objective panel refused every
   37   prompt with "beta must be a positive vector of length K", which reads as
   38   infeasibility rather than as a bug; that is why the assertion exists.
   39 - `solve_pros4_targets.py`, `solve_pros4_targets_v2.py` — the shared-weight
   40   solvers the rest are generated from.
   41 - `solve_pros4_pw_nbpo*.py`, `solve_pros4_pw_fixedref*.py`,
   42   `solve_pros4_prosper*.py` — per-prompt Nash, fixed-reference Nash and
   43   absolute max-min, per round and per panel.
   44 - `solve_mopo_targets_us1.py`, `solve_mopo_targets_ut1.py` — MOPO importance
   45   weights, which are per-candidate rather than per-pair.
   46 
```

Original lines 145–158:

```text
  145 
  146 ## Not included
  147 
  148 The deployed trainer lives in `mnpo_scripts/` at the repository root, not here.
  149 Pool candidates, judgments, solver artifacts, datasets and checkpoints stay on
  150 the pod and in the Hugging Face repositories; only code is snapshotted.
  151 
  152 ## File hashes
  153 
  154 | File | sha256 (16) |
  155 |---|---|
  156 | `abl_analysis.py` | `a1ae9271f1a860e1` |
  157 | `ahv2_breakdown.py` | `d4e97430d4f537bb` |
  158 | `ahv2_paired.py` | `85038b4b314c5943` |
```

## E03 — Current scorer wires LL to A_policy and LR to A_ref

Path: `analysis/sub_20260914/code_snapshot_20260917_union/union_score_panel.py`

SHA256: `18908d465c2af93fa65d18787d9b13d3daa616649d5e2f88d050f6fa641d6e98`

Original lines 7–28:

```text
    7 """Build UW game tensors from the union judging graph.
    8 
    9 The recorded PROSPER-setting scorer is a SELF-PLAY construction, and says so in
   10 its own docstring: one bank of eight responses, A_policy antisymmetric among
   11 them, and A_ref a copy of A_policy. The union contract asks for something
   12 different -- a learner bank and a separate reference bank -- so this builds:
   13 
   14   A_policy[k,i,j] = P_k(learner_i > learner_j) - 1/2      from the learner triangle
   15   A_ref[k,i,j]    = P_k(learner_i > reference_j) - 1/2    from the 8x8 cross block
   16 
   17 A_policy is antisymmetric by construction and is checked. A_ref is NOT
   18 antisymmetric: its two indices name different banks, and its diagonal need not
   19 be zero, which the contract states explicitly. The reference triangle is not
   20 part of either matrix; it is kept as the reference-disagreement diagnostic the
   21 contract asks for.
   22 
   23 Both orders must be valid for a pair to count, exactly as declared: with
   24 value_for_i already reversed for the swapped order, the order-balanced estimate
   25 is their mean, which equals 1/2[v_ij/4 + 1 - v_ji/4]. A pair of identical text
   26 is 0.5 by convention and logged as such. A prompt enters only when every
   27 learner pair and every cross pair is resolved for all four items; partial
   28 prompts are dropped with their reason, never filled in.
```

Original lines 112–167:

```text
  112     learner_pairs = list(itertools.combinations(range(POOL), 2))
  113     cross_pairs = [(i, j) for i in range(POOL) for j in range(POOL)]
  114 
  115     def resolve(pid, rubric, role_i, i, role_j, j):
  116         """Order-balanced probability, or None when an order is missing."""
  117         got = obs.get((pid, rubric, role_i, i, role_j, j))
  118         if got is None or 0 not in got or 1 not in got:
  119             return None
  120         return 0.5 * (got[0] + got[1])
  121 
  122     tensors, identity_ties, refdis = {}, Counter(), {}
  123     dropped = Counter()
  124     prompts = sorted({k[0] for k in obs})
  125     for pid in prompts:
  126         if pid not in pool:
  127             dropped["prompt_absent_from_pool"] += 1
  128             continue
  129         A = np.zeros((K, POOL, POOL))
  130         R = np.zeros((K, POOL, POOL))
  131         ok = True
  132         for k, rubric in enumerate(ITEMS):
  133             for i, j in learner_pairs:
  134                 p = resolve(pid, rubric, "learner", i, "learner", j)
  135                 if p is None:
  136                     ok = False; dropped["learner_pair_unresolved"] += 1; break
  137                 A[k, i, j] = p - 0.5
  138                 A[k, j, i] = 0.5 - p
  139             if not ok:
  140                 break
  141             for i, j in cross_pairs:
  142                 p = resolve(pid, rubric, "learner", i, "comparator", j)
  143                 if p is None:
  144                     ok = False; dropped["cross_pair_unresolved"] += 1; break
  145                 if (pool[pid]["learner"].get(i) ==
  146                         pool[pid]["comparator"].get(j) is not None):
  147                     identity_ties[pid] += 1
  148                 R[k, i, j] = p - 0.5
  149             if not ok:
  150                 break
  151         if not ok:
  152             continue
  153         idx = np.arange(POOL)
  154         A[:, idx, idx] = 0.0
  155         if np.abs(A + np.swapaxes(A, -1, -2)).max() > 1e-12:
  156             raise SystemExit("learner payoff not antisymmetric for %s" % pid)
  157         if max(np.abs(A).max(), np.abs(R).max()) > 0.5 + 1e-12:
  158             raise SystemExit("payoff outside [-0.5, 0.5] for %s" % pid)
  159         # reference disagreement: the reference triangle, kept as a diagnostic
  160         vals = []
  161         for k, rubric in enumerate(ITEMS):
  162             for i, j in learner_pairs:
  163                 p = resolve(pid, rubric, "comparator", i, "comparator", j)
  164                 if p is not None:
  165                     vals.append(abs(p - 0.5))
  166         refdis[pid] = float(np.mean(vals)) if vals else None
  167         tensors[pid] = (A, R)
```

Original lines 187–223:

```text
  187     for s in range(args.shards):
  188         chunk = pids[s * per:(s + 1) * per]
  189         if not chunk:
  190             continue
  191         A = np.stack([tensors[p][0] for p in chunk], axis=1)
  192         R = np.stack([tensors[p][1] for p in chunk], axis=1)
  193         d = out / ("shard%d" % s)
  194         d.mkdir(parents=True, exist_ok=True)
  195         path = d / "chunk0000.npz"
  196         np.savez(path, prompt_ids=np.array(chunk), A_policy=A, A_ref=R)
  197         (d / "chunk0000.manifest.json").write_text(json.dumps(
  198             {"prompts": len(chunk), "sha256": file_hash(path),
  199              "shapes": {"A_policy": list(A.shape), "A_ref": list(R.shape)}},
  200             indent=1) + "\n")
  201         (d / ("complete_shard%d.json" % s)).write_text(json.dumps(
  202             {"shard": s, "prompts": len(chunk), "gpm_teacher": teacher,
  203              "bt_teacher": ("absent: the union contract uses direct order-balanced PSC "
  204                             "probabilities for these rows and forbids a scalar BT "
  205                             "projection, so none was fitted and none is written"),
  206              "reference_construction": ("independent reference bank: eight reference "
  207                                         "occurrences per prompt, so A_ref is a cross "
  208                                         "block and not a copy of A_policy")},
  209             indent=1) + "\n")
  210         written.append({"shard": s, "prompts": len(chunk),
  211                         "sha256": file_hash(path),
  212                         "shapes": {"A_policy": list(A.shape), "A_ref": list(R.shape)}})
  213 
  214     verdicts = sum(status.values())
  215     report = {
  216         "out": args.out, "judgment_tag": args.tag, "pool": args.pool,
  217         "construction": {
  218             "A_policy": "learner triangle, antisymmetric, diagonal zero",
  219             "A_ref": ("8x8 cross block learner-by-reference; NOT antisymmetric and its "
  220                       "diagonal need not be zero, per the contract"),
  221             "reference_triangle": "kept as the reference-disagreement diagnostic only",
  222             "not_self_play": ("the recorded PROSPER-setting scorer copies A_policy into "
  223                               "A_ref because that panel has one bank; this panel has two"),
```

## E04 — Original UW scorer has the same wiring

Path: `analysis/sub_20260914/code_snapshot_20260917_union/union_score_uw.py`

SHA256: `aefb140d0af0c027bffe4cb4ce805b195af77cd72b90496a6b43453a1b8bac82`

Original lines 1–22:

```text
    1 """Build UW game tensors from the union judging graph.
    2 
    3 The recorded PROSPER-setting scorer is a SELF-PLAY construction, and says so in
    4 its own docstring: one bank of eight responses, A_policy antisymmetric among
    5 them, and A_ref a copy of A_policy. The union contract asks for something
    6 different -- a learner bank and a separate reference bank -- so this builds:
    7 
    8   A_policy[k,i,j] = P_k(learner_i > learner_j) - 1/2      from the learner triangle
    9   A_ref[k,i,j]    = P_k(learner_i > reference_j) - 1/2    from the 8x8 cross block
   10 
   11 A_policy is antisymmetric by construction and is checked. A_ref is NOT
   12 antisymmetric: its two indices name different banks, and its diagonal need not
   13 be zero, which the contract states explicitly. The reference triangle is not
   14 part of either matrix; it is kept as the reference-disagreement diagnostic the
   15 contract asks for.
   16 
   17 Both orders must be valid for a pair to count, exactly as declared: with
   18 value_for_i already reversed for the swapped order, the order-balanced estimate
   19 is their mean, which equals 1/2[v_ij/4 + 1 - v_ji/4]. A pair of identical text
   20 is 0.5 by convention and logged as such. A prompt enters only when every
   21 learner pair and every cross pair is resolved for all four items; partial
   22 prompts are dropped with their reason, never filled in.
```

Original lines 114–154:

```text
  114             dropped["prompt_absent_from_pool"] += 1
  115             continue
  116         A = np.zeros((K, POOL, POOL))
  117         R = np.zeros((K, POOL, POOL))
  118         ok = True
  119         for k, rubric in enumerate(ITEMS):
  120             for i, j in learner_pairs:
  121                 p = resolve(pid, rubric, "learner", i, "learner", j)
  122                 if p is None:
  123                     ok = False; dropped["learner_pair_unresolved"] += 1; break
  124                 A[k, i, j] = p - 0.5
  125                 A[k, j, i] = 0.5 - p
  126             if not ok:
  127                 break
  128             for i, j in cross_pairs:
  129                 p = resolve(pid, rubric, "learner", i, "comparator", j)
  130                 if p is None:
  131                     ok = False; dropped["cross_pair_unresolved"] += 1; break
  132                 if (pool[pid]["learner"].get(i) ==
  133                         pool[pid]["comparator"].get(j) is not None):
  134                     identity_ties[pid] += 1
  135                 R[k, i, j] = p - 0.5
  136             if not ok:
  137                 break
  138         if not ok:
  139             continue
  140         idx = np.arange(POOL)
  141         A[:, idx, idx] = 0.0
  142         if np.abs(A + np.swapaxes(A, -1, -2)).max() > 1e-12:
  143             raise SystemExit("learner payoff not antisymmetric for %s" % pid)
  144         if max(np.abs(A).max(), np.abs(R).max()) > 0.5 + 1e-12:
  145             raise SystemExit("payoff outside [-0.5, 0.5] for %s" % pid)
  146         # reference disagreement: the reference triangle, kept as a diagnostic
  147         vals = []
  148         for k, rubric in enumerate(ITEMS):
  149             for i, j in learner_pairs:
  150                 p = resolve(pid, rubric, "comparator", i, "comparator", j)
  151                 if p is not None:
  152                     vals.append(abs(p - 0.5))
  153         refdis[pid] = float(np.mean(vals)) if vals else None
  154         tensors[pid] = (A, R)
```

## E05 — Current target loader imports missing public functions and preserves matrix fields

Path: `analysis/sub_20260914/code_snapshot_20260917_union/solve_pros4_targets_uw1.py`

SHA256: `0d4df3ec3ad1de0451ca40cb375a71f46fc231a2af32cee7e0bf7b92809694fb`

Original lines 43–54:

```text
   43 from mnpo_scripts.nbpo_core import uniform_policy
   44 from mnpo_scripts.nbpo_generic import solve_finite_pool, validate_finite_pool_solution
   45 from mnpo_scripts.nbpo_representations import (
   46     AdaptiveGameRepresentation, BTRewardRepresentation, FixedReferenceRepresentation,
   47 )
   48 from scripts.nbpo.build_nbpo_pairs import build_rows, load_canonical_artifact
   49 from scripts.nbpo.solve_nbpo_dual import write_generic_solution_artifact
   50 
   51 ROOT = Path("/work/uf4_20260910")
   52 OBJECTIVES = ("item0", "item1", "item2", "item3")
   53 REFERENCE_CONSTRUCTION = "independent_samples"
   54 POOL = 8
```

Original lines 83–109:

```text
   83 CANONICAL_DECIMALS = 10
   84 
   85 
   86 def quantize_canonical_row(row):
   87     """Round the solver masses to the loader's precision and rebuild the target.
   88 
   89     prepare_nbpo_dataset reads these rows through a JSON parser that keeps ten
   90     decimal places, then checks target == log(w_a/c_a) - log(w_b/c_b) to 1e-9.
   91     Rounding the masses first and deriving the target from the rounded values
   92     makes the row consistent under that parse instead of only before it. A
   93     per-prompt Nash solve concentrates mass, so without this the identity
   94     survives in memory and fails after the parse.
   95     """
   96     import math
   97 
   98     for key in ("nbpo_weight_a", "nbpo_weight_b"):
   99         if key in row:
  100             row[key] = round(float(row[key]), CANONICAL_DECIMALS)
  101     if row.get("target_mode") == "canonical_logratio" and "nbpo_weight_a" in row:
  102         wa, wb = float(row["nbpo_weight_a"]), float(row["nbpo_weight_b"])
  103         ca = float(row.get("nbpo_center_a", 1.0 / POOL))
  104         cb = float(row.get("nbpo_center_b", 1.0 / POOL))
  105         if min(wa, wb) > 0:
  106             row["nbpo_logratio_target"] = round(
  107                 math.log(wa / ca) - math.log(wb / cb), CANONICAL_DECIMALS)
  108             row["canonical_target_quantized_decimals"] = CANONICAL_DECIMALS
  109     return row
```

Original lines 112–150:

```text
  112 def load_scores(score_root, shards):
  113     """prompt_id -> (A_policy[K,8,8], A_ref[K,8,8], r_bt[K,2,8]), hash-verified.
  114 
  115     r_bt is the frozen BT head's scalar reward for the eight learner and eight
  116     comparator occurrences. It is loaded for every representation but only the
  117     bt_reward representation reads it, so nothing else changes.
  118     """
  119     scores, manifests = {}, []
  120     for shard in range(shards):
  121         directory = Path(score_root) / f"shard{shard}"
  122         complete = json.loads((directory / f"complete_shard{shard}.json").read_text())
  123         manifests.append({"shard": shard, "gpm_teacher": complete["gpm_teacher"],
  124                           "bt_teacher": complete["bt_teacher"],
  125                           "reference_construction": complete["reference_construction"]})
  126         for path in sorted(directory.glob("chunk*.npz")):
  127             meta = json.loads(path.with_suffix("").with_suffix(".manifest.json").read_text()) \
  128                 if path.with_suffix("").with_suffix(".manifest.json").exists() else \
  129                 json.loads((directory / (path.stem + ".manifest.json")).read_text())
  130             if file_hash(path) != meta["sha256"]:
  131                 raise ValueError(f"Score chunk hash mismatch: {path}")
  132             arrays = np.load(path, allow_pickle=True)
  133             pids = [str(p) for p in arrays["prompt_ids"]]
  134             A, Aref = arrays["A_policy"], arrays["A_ref"]
  135             # r_bt is the scalar BT projection. The union contract uses direct
  136             # order-balanced probabilities for these rows and forbids that
  137             # projection, so the array may be absent; it is left as None rather
  138             # than filled with zeros, and only the bt_reward representation
  139             # reads it, where None raises instead of training on a placeholder.
  140             Rbt = arrays["r_bt"] if "r_bt" in arrays.files else None
  141             if A.shape[0] != len(OBJECTIVES) or A.shape[2:] != (POOL, POOL):
  142                 raise ValueError(f"Unexpected score tensor shape in {path}: {A.shape}")
  143             if Rbt is not None and (Rbt.shape[0] != len(OBJECTIVES)
  144                                     or Rbt.shape[2:] != (2, POOL)):
  145                 raise ValueError(f"Unexpected BT reward shape in {path}: {Rbt.shape}")
  146             for index, pid in enumerate(pids):
  147                 if pid in scores:
  148                     raise ValueError(f"Duplicate scored prompt {pid}")
  149                 scores[pid] = (A[:, index], Aref[:, index],
  150                                None if Rbt is None else Rbt[:, index])
```

## E06 — Promptwise solver worker and canonical export

Path: `analysis/sub_20260914/code_snapshot_20260917_union/solve_pros4_pw_nbpo_uw1.py`

SHA256: `ba54ce2bf405c8578a83b4f9820d8b515230c123debdbdef192bd34e058b5721`

Original lines 44–87:

```text
   44 sys.path.insert(0, "/work/uf4_20260910/code")
   45 import solve_pros4_targets_uw1 as base                      # loaders, hashing, writers
   46 from mnpo_scripts.nbpo_core import uniform_policy
   47 from mnpo_scripts.nbpo_generic import solve_finite_pool, validate_finite_pool_solution
   48 from mnpo_scripts.nbpo_representations import AdaptiveGameRepresentation
   49 from scripts.nbpo.build_nbpo_pairs import build_rows
   50 
   51 ROOT = Path("/work/uf4_20260910")
   52 OBJECTIVES = base.OBJECTIVES
   53 POOL = base.POOL
   54 _SHARED = {}
   55 
   56 
   57 def _init(A, Aref, beta, eta, weight_l1, max_dual_calls, floor):
   58     _SHARED.update(A=A, Aref=Aref, beta=beta, eta=eta, weight_l1=weight_l1,
   59                    M=max_dual_calls, floor=floor)
   60     torch.set_num_threads(1)
   61 
   62 
   63 def _solve_one(x):
   64     """Solve prompt x alone: its own adversarial weights, its own certificate."""
   65     A = _SHARED["A"][:, x:x + 1]
   66     Aref = _SHARED["Aref"][:, x:x + 1]
   67     rep = AdaptiveGameRepresentation(
   68         torch.from_numpy(np.ascontiguousarray(A)), torch.from_numpy(np.ascontiguousarray(Aref)),
   69         uniform_policy(1, POOL),
   70         torch.full((len(OBJECTIVES),), _SHARED["beta"], dtype=torch.float64),
   71         reference_construction="independent_samples")
   72     result = solve_finite_pool(
   73         rep, "nash", eta=_SHARED["eta"], inner_solver="exact",
   74         dual_solver="root", dual_tol=1e-10, M=_SHARED["M"], inner_workers=1,
   75         probability_floor=_SHARED["floor"], log_every=0)
   76     certificate = validate_finite_pool_solution(result)
   77     return (x,
   78             result.pi.numpy().astype(np.float64),
   79             result.nu_update.numpy().astype(np.float64),
   80             result.weights.numpy().astype(np.float64),
   81             result.target_log_ratio.numpy().astype(np.float64),
   82             float(result.target_log_ratio_check()),
   83             bool(certificate.get("certified", False)),
   84             # the certificate carries the per-objective surplus vector, not its
   85             # minimum; taking .get("min_surplus") silently produced NaN, which
   86             # then made "negative worst surplus" count zero prompts by accident
   87             float(np.min(np.asarray(certificate["surplus"], dtype=np.float64))))
```

Original lines 96–116:

```text
   96     ap.add_argument("--weight-l1", type=float, default=None,
   97                     help="matched multiplier norm as a literal. Only absolute_maxmin "
   98                          "uses it; nash derives its own dual.")
   99     ap.add_argument("--weight-l1-from", type=Path,
  100                     default=None)
  101     ap.add_argument("--beta", type=float, default=0.25)
  102     ap.add_argument("--eta", type=float, default=1.0)
  103     ap.add_argument("--workers", type=int, default=32)
  104     ap.add_argument("--max-dual-calls", type=int, default=200)
  105     ap.add_argument("--probability-floor", type=float, default=1e-12)
  106     ap.add_argument("--limit", type=int, default=0,
  107                     help="pilot on the first N prompts of each split; 0 means all")
  108     ap.add_argument("--skip-dataset", action="store_true")
  109     args = ap.parse_args()
  110 
  111     # nash derives its own dual: no multiplier norm is read, and none is used.
  112     # The global NBPO arm this was matched to is uncertified (independent
  113     # stationarity 13.224), and nothing here is matched to an uncertified value.
  114     weight_l1 = None
  115     print(json.dumps({"weight_l1": None,
  116                       "note": "unused: nash derives its own dual"}), flush=True)
```

Original lines 172–215:

```text
  172                                            args.max_dual_calls, args.probability_floor)) as pool_exec:
  173             for res in pool_exec.map(_solve_one, range(len(pids)), chunksize=8):
  174                 x, pi_x, nu_x, w_x, g_x, ident, cert, ms = res
  175                 pi[x] = pi_x[0]
  176                 nu[:, x] = nu_x[:, 0]
  177                 weights[:, x] = w_x
  178                 g[x] = g_x[0]
  179                 identity[x], certified[x], min_surplus[x] = ident, cert, ms
  180                 done += 1
  181                 if done % 1000 == 0:
  182                     print(json.dumps({"split": split, "solved": done, "of": len(pids),
  183                                       "seconds": round(time.monotonic() - split_start, 1)}), flush=True)
  184         if not certified.all():
  185             raise ValueError(f"{split}: {int((~certified).sum())} prompts have no certificate")
  186         if float(np.abs(identity).max()) > 1e-9:
  187             raise ValueError(f"{split}: target log-ratio identity residual "
  188                              f"{float(np.abs(identity).max())}")
  189 
  190         per_prompt_path = out / f"{split}_per_prompt.npz"
  191         np.savez_compressed(per_prompt_path, pi=pi, weights=weights, g=g,
  192                             min_surplus=min_surplus, identity_residual=identity,
  193                             prompt_ids=np.array(pids, dtype=object))
  194 
  195         # A hash-bound solution artifact at the path the shared job generator
  196         # expects. The global one is not written because per-prompt weights do
  197         # not fit its (K,) weight field, but the generator only needs SOME file
  198         # whose hash pins the solution, and patching the generator would touch
  199         # code the published arms depend on. This file pins the real thing: the
  200         # per-prompt npz that every target in this set was built from.
  201         solver_dir = out / split / "solver"
  202         base.write_json(solver_dir / "solution.json", {
  203             "target_mode": "canonical_logratio", "target_column": "nbpo_logratio_target",
  204             "target_units": "final_logratio_change", "eta_already_included": True,
  205             "representation": "adaptive_game", "aggregation": "prompt_wise_nash",
  206             "weights_scope": "per prompt; there is no shared dual",
  207             "per_prompt_artifact": str(per_prompt_path),
  208             "per_prompt_artifact_sha256": base.file_hash(per_prompt_path),
  209             "n_prompts": len(pids), "split": split,
  210             "all_certified": bool(certified.all()),
  211             "max_identity_residual": float(np.abs(identity).max()),
  212             "min_surplus_mean": float(min_surplus.mean()),
  213             "min_surplus_negative_prompts": int((min_surplus < 0).sum()),
  214             "weight_l1_matched": weight_l1, "beta": args.beta, "eta": args.eta,
  215             "solver_source_sha256": base.file_hash(__file__), **shared_meta})
```

Original lines 217–244:

```text
  217         provenance = {"solver_artifact_sha256": solver_hash,
  218                       "solver_hash": solver_hash,
  219                       "target_artifact_hash": base.file_hash(per_prompt_path),
  220                       "representation": "adaptive_game", "aggregation": "prompt_wise_nash",
  221                       "split": split, "panel": "UF-4", **shared_meta}
  222         betas = np.full(len(OBJECTIVES), args.beta)
  223 
  224         def pair_rows():
  225             for x, pid in enumerate(pids):
  226                 learners = {str(i): {pid: {**pool[pid]["learner"][i],
  227                                            "generated_text": pool[pid]["learner"][i]["response"]}}
  228                             for i in range(POOL)}
  229                 rows = build_rows(
  230                     [pid], list(OBJECTIVES), A[:, x:x + 1], nu[:, x:x + 1], weights[:, x],
  231                     betas, learners, None, np.random.default_rng(42), "canonical_logratio",
  232                     meta, provenance,
  233                     canonical_data={"g": g[x:x + 1], "p_star": pi[x:x + 1],
  234                                     "p_t": np.full((1, POOL), 1.0 / POOL)})
  235                 for row in rows:
  236                     # round masses to the loader's ten decimals and rebuild the
  237                     # target from the rounded values; a per-prompt Nash solve
  238                     # concentrates mass and the identity otherwise fails after the parse
  239                     row = base.quantize_canonical_row(row)
  240                     a, b = row["chosen_candidate_index"], row["rejected_candidate_index"]
  241                     row["chosen_response_id"] = pool[pid]["learner"][a]["candidate_id"]
  242                     row["rejected_response_id"] = pool[pid]["learner"][b]["candidate_id"]
  243                     row["prosper_prompt_weights"] = [float(v) for v in weights[:, x]]
  244                     yield row
```

Original lines 264–271:

```text
  264     prov = {**shared_meta, "objectives": list(OBJECTIVES), "panel": "UF-4",
  265             "aggregation": "prompt_wise_nash", "beta": args.beta, "eta": args.eta,
  266             "weight_l1": weight_l1, "splits": outputs,
  267             "train_pool_sha256": base.object_hash(sorted(outputs)),
  268             "dev_pool_sha256": base.object_hash(sorted(outputs)),
  269             "dev_note": ("no shared dual: each prompt fits its own adversarial weights on "
  270                          "whichever split it is in, so dev measures neural generalisation "
  271                          "but not generalisation of a dual fitted on train")}
```

## E07 — Released builder lacks canonical target API

Path: `scripts/nbpo/build_nbpo_pairs.py`

SHA256: `2609b091e2c7fd9c7911c9119a3fdf49d757779a7fd5d2017f8f10b1073134a4`

Original lines 37–46:

```text
   37 from scripts.nbpo.nbpo_common import (
   38     implementation_contract,
   39     load_response_files,
   40     response_pool_hash,
   41     sha256_file,
   42     sha256_text,
   43     write_json,
   44 )
   45 
   46 TARGET_MODES = ("sampled", "rao_blackwell")
```

Original lines 57–114:

```text
   57 def flip_pair_row(row: dict) -> dict:
   58     """Swap the pair orientation, flipping every Z_k and the aggregate target."""
   59     flipped = dict(row)
   60     flipped["chosen"], flipped["rejected"] = row["rejected"], row["chosen"]
   61     flipped["chosen_response_id"], flipped["rejected_response_id"] = (
   62         row["rejected_response_id"], row["chosen_response_id"])
   63     flipped["nbpo_z"] = {k: -v for k, v in row["nbpo_z"].items()}
   64     flipped["nbpo_weighted_z"] = -row["nbpo_weighted_z"]
   65     return flipped
   66 
   67 
   68 def build_rows(prompt_ids, objectives, A_policy, nu, lam, betas, policy, ref_seed_of,
   69                rng, target_mode, meta_ids, provenance):
   70     """One row per (prompt, unordered learner pair); deterministic given the RNG."""
   71     K = len(objectives)
   72     I = A_policy.shape[2]
   73     policy_ids = meta_ids["policy_learner_ids"]
   74     comparator_ids = meta_ids["comparator_ids"]
   75     rows = []
   76     for x, pid in enumerate(prompt_ids):
   77         seed0 = policy_ids[0].split(":", 1)[1]
   78         prompt_text = str(policy[seed0][pid]["prompt"])
   79         for i1, i2 in itertools.combinations(range(I), 2):
   80             z, opp = {}, {}
   81             for k, obj in enumerate(objectives):
   82                 # Eq. (26): draw (y, y') first, THEN one z_k ~ nu*_k for this pair and
   83                 # objective. The same z_k serves both y and y' of the row; other rows
   84                 # of the same prompt and other objectives draw independently.
   85                 j = int(rng.choice(len(comparator_ids), p=nu[k, x]))
   86                 p1 = float(A_policy[k, x, i1, j]) + 0.5
   87                 p2 = float(A_policy[k, x, i2, j]) + 0.5
   88                 if target_mode == "sampled":
   89                     b1 = float(rng.random() < p1)
   90                     b2 = float(rng.random() < p2)
   91                     z[obj] = b1 - b2
   92                 else:  # rao_blackwell
   93                     z[obj] = p1 - p2
   94                 opp[obj] = comparator_ids[j]
   95             id1, id2 = policy_ids[i1], policy_ids[i2]
   96             rows.append({
   97                 "prompt_id": pid,
   98                 "prompt": prompt_text,
   99                 "chosen": str(policy[id1.split(":", 1)[1]][pid]["generated_text"]),
  100                 "rejected": str(policy[id2.split(":", 1)[1]][pid]["generated_text"]),
  101                 "chosen_response_id": id1,
  102                 "rejected_response_id": id2,
  103                 # Response IDS are not response TEXT: these pin the exact strings
  104                 # this row was built from, so a later pool swap is detectable.
  105                 "chosen_text_sha256": sha256_text(
  106                     str(policy[id1.split(":", 1)[1]][pid]["generated_text"])),
  107                 "rejected_text_sha256": sha256_text(
  108                     str(policy[id2.split(":", 1)[1]][pid]["generated_text"])),
  109                 "nbpo_z": z,
  110                 "nbpo_weighted_z": float(sum(lam[k] * z[obj] for k, obj in enumerate(objectives))),
  111                 "lambda_raw": {obj: float(lam[k]) for k, obj in enumerate(objectives)},
  112                 "opponent_response_id": opp,
  113                 "opponent_beta": {obj: float(betas[k]) for k, obj in enumerate(objectives)},
  114                 "opponent_sampling_scope": "pair_objective",
```

## E08 — Released solver writer and imports

Path: `scripts/nbpo/solve_nbpo_dual.py`

SHA256: `b70ca16d26140fd244b01ad64e77e927d9fd97235c24ff74e2d61847bc3b850e`

Original lines 1–76:

```text
    1 #!/usr/bin/env python3
    2 """Deterministic finite-pool NBPO dual solve (Algorithm 1's inner machinery).
    3 
    4 Reads a versioned preference-tensor artifact (``build_preference_tensor.py``)
    5 and runs projected dual gradient descent on the raw multipliers
    6 (Eq. (27) ``eq:dual-update``) with the fixed-point weighted-policy solve of
    7 Section 5.2 (Eq. (21), centered at the proximal center), or one of the matched
    8 finite-game controls (utilitarian / absolute max-min / surplus max-min) on the
    9 same tensors and budget.
   10 
   11 Scope: every one of the ``M`` dual iterations (4e3--3e5 in the manuscript) is a
   12 cheap tensor computation on the FROZEN finite response pool. The neural policy
   13 is fit afterwards, once, from the targets built by ``build_nbpo_pairs.py`` --
   14 no 8B model is retrained inside this loop.
   15 
   16 Outputs (all raw, none normalized or clamped):
   17 ``solution.json`` -- raw lambda, V, d, surplus, the inverse-surplus residual
   18 ``||s - 1/lambda||_inf`` AND the projected (box-aware) KKT residual with the
   19 active-bound coordinates, fixed-point and one-extra-map residuals, opponent
   20 entropy/ESS of the final policy, the full config (beta, eta, gamma schedule,
   21 M, R, lambda box), input artifact hashes, hashes of the opponent files, and the
   22 iteration history; ``nu_update.npz`` -- the opponent that generated the final
   23 policy (what Eq. (26) pair construction samples from); ``nu_final_policy.npz``
   24 -- ``nu*`` recomputed at the final policy (diagnostics); ``pi_star.npz`` -- the
   25 finite-pool policy.
   26 """
   27 from __future__ import annotations
   28 
   29 import argparse
   30 import json
   31 from pathlib import Path
   32 
   33 import numpy as np
   34 import torch
   35 
   36 from mnpo_scripts.nbpo_core import (
   37     uniform_policy,
   38     validate_centered_preference_tensor,
   39     validate_reference_tensor,
   40 )
   41 from mnpo_scripts.nbpo_solver import AGGREGATIONS, solve_nbpo_dual
   42 from scripts.nbpo.nbpo_common import implementation_contract, sha256_file, write_json
   43 
   44 
   45 def load_tensor_artifact(tensor_dir: Path):
   46     meta = json.loads((tensor_dir / "meta.json").read_text())
   47     A_policy = torch.from_numpy(np.load(tensor_dir / "tensor_policy.npz")["A"])
   48     A_ref = torch.from_numpy(np.load(tensor_dir / "tensor_ref.npz")["A"])
   49     hashes = {name: sha256_file(tensor_dir / name)
   50               for name in ("tensor_policy.npz", "tensor_ref.npz", "meta.json")}
   51     return meta, A_policy, A_ref, hashes
   52 
   53 
   54 def parse_gamma(text: str, M: int):
   55     parts = [float(p) for p in text.split(",") if p.strip()]
   56     return parts[0] if len(parts) == 1 else parts
   57 
   58 
   59 def _array_hash(t) -> str:
   60     """sha256 of an array's exact bytes -- identifies which policy an opponent came from."""
   61     import hashlib
   62 
   63     arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
   64     return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()
   65 
   66 
   67 def write_solution_artifact(out_dir: Path, res, tensor_meta: dict, hashes: dict,
   68                             tensor_dir: Path, stage: int, lambda_warm_started: bool) -> dict:
   69     """Persist a DualSolveResult as the versioned solver artifact (shared with run_nbpo_stage)."""
   70     out_dir.mkdir(parents=True, exist_ok=True)
   71     # Two opponents, written separately and hashed separately (never confuse them):
   72     #   nu_update.npz       -- generated the final policy; Eq. (26) targets sample z_k here
   73     #   nu_final_policy.npz -- nu* recomputed AT the final policy; diagnostics only
   74     np.savez_compressed(out_dir / "nu_update.npz", nu=res.nu_update.numpy())
   75     np.savez_compressed(out_dir / "nu_final_policy.npz", nu=res.nu_final_policy.numpy())
   76     np.savez_compressed(out_dir / "pi_star.npz", pi=res.pi.numpy())
```

## E09 — Core disagreement contract

Path: `mnpo_scripts/nbpo_core.py`

SHA256: `5d028303aa533f03805dbee2e895f8c9d1e459ed4f87c93ee4a4f9b850d2a944`

Original lines 344–383:

```text
  344 def compute_disagreement_point(
  345     A_ref: torch.Tensor, mu: torch.Tensor, beta: TensorLike,
  346     construction: str = "shared_pool",
  347     mu_learner: Optional[torch.Tensor] = None,
  348 ) -> torch.Tensor:
  349     """``d_k = V_{k,beta_k}(mu)`` from the reference-as-learner tensor (Eq. (10)).
  350 
  351     ``A_ref[k, x, i, j]`` holds centered payoffs of reference responses (as
  352     learner, index ``i``) against reference comparators (index ``j``). ``d`` is
  353     generally negative for skew-symmetric payoffs at beta < infinity and is
  354     never replaced by ``g_k(mu, mu) = 0``.
  355 
  356     Both sides of this game are the SAME reference policy ``mu``. The learner
  357     side used to be hard-coded uniform while the comparator side took whatever
  358     ``mu`` was passed, so a nonuniform empirical ``mu`` silently produced
  359     ``V`` of a mismatched pair of distributions -- a ``d`` that is not
  360     ``V_{k,beta}(mu)`` for any single policy. Now either the caller supplies
  361     ``mu_learner`` explicitly, or the comparator ``mu`` is asserted uniform and
  362     that uniform distribution is used on both sides.
  363     """
  364     A_ref = validate_reference_tensor(A_ref, "A_ref", construction)
  365     if mu_learner is None:
  366         mu_c = as_float64(mu)
  367         expected = 1.0 / mu_c.shape[-1]
  368         if not bool((mu_c - expected).abs().max() < UNIFORM_TOL):
  369             raise ValueError(
  370                 "compute_disagreement_point was given a NONUNIFORM comparator mu but no "
  371                 "mu_learner: d_k = V_{k,beta}(mu) requires the same mu on both sides of "
  372                 f"the reference game (max deviation from uniform "
  373                 f"{float((mu_c - expected).abs().max()):.3e}). Pass mu_learner explicitly."
  374             )
  375         pi_ref = uniform_policy(A_ref.shape[1], A_ref.shape[2])
  376     else:
  377         pi_ref = validate_distribution(mu_learner, "mu_learner")
  378         if pi_ref.shape != (A_ref.shape[1], A_ref.shape[2]):
  379             raise ValueError(
  380                 f"mu_learner must have shape {(A_ref.shape[1], A_ref.shape[2])}, "
  381                 f"got {tuple(pi_ref.shape)}")
  382     r_ref = compute_margins(A_ref, pi_ref)
  383     return compute_regularized_game_value(r_ref, mu, beta, form="softmin")
```

## E10 — Representation binds A_ref to disagreement

Path: `mnpo_scripts/nbpo_representations.py`

SHA256: `908ee6e9163f9e60a390c2971bc6c2f16ed42a12e2ab7513f87b02877b04c9f0`

Original lines 120–171:

```text
  120     def info(self) -> RepresentationInfo:
  121         raise NotImplementedError
  122 
  123 
  124 class AdaptiveGameRepresentation(ObjectiveRepresentation):
  125     """The manuscript's finite-temperature game: the comparator best-responds.
  126 
  127     ``nu*_k`` is Eq. (7) at the current policy, ``q_k`` is Eq. (9) against it,
  128     and ``V_{k,beta_k}`` is the soft-min value of Eq. (8).  ``d_k`` comes from
  129     the separate reference-as-learner tensor through the identical value map.
  130     """
  131 
  132     type = "adaptive_game"
  133     policy_adaptive = True
  134     has_opponent = True
  135 
  136     def __init__(self, A_policy, A_ref, mu, beta, reference_construction: str = "shared_pool"):
  137         self.A = validate_game_utility_tensor(A_policy, "A_policy")
  138         self.A_ref = validate_reference_tensor(A_ref, "A_ref", reference_construction)
  139         self.mu = validate_distribution(mu, "mu", require_full_support=True)
  140         self.K, self.X, self.I, self.J = self.A.shape
  141         beta = as_float64(beta)
  142         if beta.dim() == 0:
  143             beta = beta.expand(self.K).clone()
  144         if beta.shape != (self.K,) or not bool((beta > 0).all()):
  145             raise ValueError("beta must be a positive vector of length K")
  146         self.beta = beta
  147         self.reference_construction = reference_construction
  148         self._d = compute_disagreement_point(self.A_ref, self.mu, self.beta, reference_construction)
  149 
  150     def opponent_and_gradient(self, pi):
  151         pi = _check_learner_pi(pi, self.X, self.I)
  152         nu = compute_regularized_opponent(compute_margins(self.A, pi), self.mu, self.beta)
  153         return nu, compute_objective_gradient(self.A, nu)
  154 
  155     def game_values(self, pi):
  156         pi = _check_learner_pi(pi, self.X, self.I)
  157         return compute_regularized_game_value(
  158             compute_margins(self.A, pi), self.mu, self.beta, form="softmin")
  159 
  160     @property
  161     def disagreement(self):
  162         return self._d
  163 
  164     def info(self):
  165         return RepresentationInfo(
  166             type=self.type, policy_adaptive=True, has_opponent=True,
  167             beta=[float(b) for b in self.beta],
  168             reference_construction=self.reference_construction,
  169             detail={"opponent": "kl_regularized_best_response_eq7",
  170                     "value": "soft_min_game_value_eq8"})
  171 
```

## E11 — Direct inner optimization and final-policy opponent

Path: `mnpo_scripts/nbpo_generic.py`

SHA256: `2b3502dbd78e6d3228df32b0830182f28bfc3c997dcf2e751a21a086226c2af5`

Original lines 234–342:

```text
  234 def _solve_prompt_chunk(task):
  235     """One contiguous block of prompts, solved with the worker's resident tensors.
  236 
  237     The scipy import is at module scope, not in this function: a per-call import
  238     cost of about 0.16 s does not amortize over a chunk of a few prompts and was
  239     measured to swallow the whole parallel gain.
  240     """
  241     import numpy as np
  242     from scipy.optimize import minimize
  243     lo, hi, w_np, start_block = task
  244     g = _WORKER_STATE
  245     A, mu, log_pi_t = g["A"], g["mu"], g["log_pi_t"]
  246     beta, eta, X, I = g["beta"], g["eta"], g["X"], g["I"]
  247     floor, maxiter, ftol = g["floor"], g["maxiter"], g["ftol"]
  248     ones = np.ones(I)
  249     bounds = [(floor, 1.0)] * I
  250     out = np.empty((hi - lo, I))
  251     diagnostics = []
  252 
  253     for n, x in enumerate(range(lo, hi)):
  254         Ax, lmu, lpt = A[:, x], np.log(mu[x]), log_pi_t[x]
  255 
  256         def negx(v, Ax=Ax, lmu=lmu, lpt=lpt):
  257             p = np.clip(v, floor, None)
  258             p = p / p.sum()
  259             r = np.einsum("i,kij->kj", p, Ax)
  260             lw = lmu[None, :] - r / beta[:, None]
  261             m = lw.max(axis=-1, keepdims=True)
  262             v_k = -beta * (m[:, 0] + np.log(np.exp(lw - m).sum(axis=-1)))
  263             kl = float((p * (np.log(p) - lpt)).sum())
  264             return -(float(w_np @ v_k) - kl / eta)
  265 
  266         def negx_jac(v, Ax=Ax, lmu=lmu, lpt=lpt):
  267             p = np.clip(v, floor, None)
  268             p = p / p.sum()
  269             r = np.einsum("i,kij->kj", p, Ax)
  270             lw = lmu[None, :] - r / beta[:, None]
  271             lw -= lw.max(axis=-1, keepdims=True)
  272             nu = np.exp(lw)
  273             nu /= nu.sum(axis=-1, keepdims=True)
  274             q = np.einsum("kj,kij->ki", nu, Ax)
  275             gg = w_np @ q
  276             gg -= (np.log(p) - lpt + 1.0) / eta
  277             return -gg
  278 
  279         res = minimize(negx, start_block[n], jac=negx_jac, method="SLSQP",
  280                        bounds=bounds,
  281                        constraints=[{"type": "eq",
  282                                      "fun": lambda v: np.array([v.sum() - 1.0]),
  283                                      "jac": lambda v: ones[None, :]}],
  284                        options={"maxiter": int(maxiter), "ftol": float(ftol)})
  285         v = np.clip(res.x, floor, None)
  286         p = v / v.sum()
  287         # Refine the *optimization problem*, not its exponential-map identity.
  288         # The exact positive-definite Hessian gives a simplex Newton step.
  289         # Removing the irrelevant 1/X factor above keeps tolerance independent
  290         # of the number of prompts in the dataset.
  291         refinement_steps = 0
  292         for refinement_steps in range(60):
  293             grad = negx_jac(p)
  294             residual = eta * np.max(np.abs(grad - p @ grad))
  295             if residual < 1e-10:
  296                 break
  297             r = np.einsum("i,kij->kj", p, Ax)
  298             lw = lmu[None, :] - r / beta[:, None]
  299             lw -= lw.max(axis=-1, keepdims=True)
  300             nu = np.exp(lw)
  301             nu /= nu.sum(axis=-1, keepdims=True)
  302             q = np.einsum("kj,kij->ki", nu, Ax)
  303             covariance = (np.einsum("kj,kij,klj->kil", nu, Ax, Ax)
  304                           - np.einsum("ki,kl->kil", q, q))
  305             hessian = np.einsum("k,kil->il", w_np / beta, covariance)
  306             hessian += np.diag(1.0 / (eta * p))
  307             system = np.block([[hessian, ones[:, None]],
  308                                [ones[None, :], np.zeros((1, 1))]])
  309             delta = np.linalg.solve(system, np.r_[-grad, 0.0])[:I]
  310             alpha = 1.0
  311             falling = delta < 0
  312             if falling.any():
  313                 alpha = min(alpha, 0.99 * np.min((p[falling] - floor) / -delta[falling]))
  314             if alpha <= 0:
  315                 break
  316             previous = negx(p)
  317             slope = float(grad @ delta)
  318             for _ in range(45):
  319                 proposal = p + alpha * delta
  320                 proposal /= proposal.sum()
  321                 # The small allowance only handles objective roundoff; the
  322                 # independently measured stationarity decides convergence.
  323                 if negx(proposal) <= previous + 1e-4 * alpha * slope + 1e-14 * max(1.0, abs(previous)):
  324                     p = proposal
  325                     break
  326                 alpha *= 0.5
  327             else:
  328                 break
  329         out[n] = p
  330         grad = negx_jac(p)
  331         residual = float(eta * np.max(np.abs(grad - p @ grad)))
  332         diagnostics.append({
  333             "prompt_index": x, "slsqp_success": bool(res.success),
  334             "slsqp_status": int(res.status), "slsqp_message": str(res.message),
  335             "slsqp_iterations": int(res.nit), "newton_iterations": refinement_steps,
  336             "backend": "slsqp_with_simplex_newton_refinement",
  337             "refinement_certified": bool(residual < 1e-4),
  338             "independent_stationarity_inf": residual,
  339             "negative_objective": float(negx(p)),
  340             "probability_floor_active": bool(np.any(p <= floor * (1.0 + 1e-6))),
  341         })
  342     return lo, out, diagnostics
```

Original lines 388–448:

```text
  388 def solve_proximal_exact(rep: ObjectiveRepresentation, pi_t: torch.Tensor,
  389                          w: torch.Tensor, eta: float, *,
  390                          pi_init: Optional[torch.Tensor] = None,
  391                          maxiter: int = 400, ftol: float = 1e-14,
  392                          workers: int = 1, probability_floor: float = 1e-12) -> ProximalSolve:
  393     """Direct concave solve, returning the optimizer policy and its own Q/nu.
  394 
  395     Each prompt is an independent simplex program. SLSQP is followed by
  396     analytic-Hessian simplex Newton refinement when needed. Both optimizer
  397     status and independent final-policy stationarity are retained; a failed
  398     SLSQP status is never silently relabeled as success. Canonical targets use
  399     log(pi/pi_t) directly, without an extra exponential-map policy update.
  400     """
  401     import numpy as np
  402 
  403     pi_t = validate_distribution(pi_t, "pi_t", require_full_support=True)
  404     w = as_float64(w)
  405     if not (eta > 0) or w.shape != (rep.K,) or not torch.isfinite(w).all() or bool((w < 0).any()):
  406         raise ValueError("eta must be positive and weights finite/nonnegative with shape (K,)")
  407     if not rep.policy_adaptive:
  408         sol = solve_proximal(rep, pi_t, w, eta, R=1, pi_init=pi_init)
  409         sol.update_source_pi = sol.pi.clone()
  410         sol.update_source_kind = "exact_proximal_maximizer"
  411         sol.optimizer_diagnostics = {"backend": "analytic_exponential_family",
  412                                      "actual_workers": 1, "probability_floor": 0.0,
  413                                      "prompts": [], "all_certified": True}
  414         return sol
  415 
  416     X, I = rep.X, rep.I
  417     floor = float(probability_floor)
  418     if not 0 < floor < 1.0 / I:
  419         raise ValueError("probability_floor must lie in (0, 1/I)")
  420     A, beta, mu = rep.A.numpy(), rep.beta.numpy(), rep.mu.numpy()
  421     log_pi_t = np.log(pi_t.numpy())
  422     start = (pi_t.numpy() if pi_init is None
  423              else validate_distribution(pi_init, "pi_init", require_full_support=True).numpy())
  424     actual_workers = min(int(workers or 1), X)
  425     if actual_workers > 1 and X >= 2 * actual_workers:
  426         out, details = _solve_prompts_parallel(
  427             A, mu, log_pi_t, start, beta, w.numpy(), eta, X, I,
  428             floor, maxiter, ftol, actual_workers)
  429     else:
  430         actual_workers = 1
  431         _WORKER_STATE.update(A=A, mu=mu, log_pi_t=log_pi_t, beta=beta,
  432                              eta=eta, X=X, I=I, floor=floor, maxiter=maxiter, ftol=ftol)
  433         _, out, details = _solve_prompt_chunk((0, X, w.numpy(), start))
  434     pi_star = torch.from_numpy(out)
  435     nu, q = rep.opponent_and_gradient(pi_star)
  436     extra = float((exp_update(pi_t, q, w, eta) - pi_star).abs().max())
  437     return ProximalSolve(
  438         pi=pi_star, nu_update=nu, q_update=q,
  439         nu_final_policy=nu, q_final_policy=q,
  440         fixed_point_residual=extra, extra_map_residual=extra,
  441         iterations=sum(v["slsqp_iterations"] + v["newton_iterations"] for v in details),
  442         update_source_pi=pi_star.clone(),
  443         update_source_kind="exact_proximal_maximizer", update_source_iteration=0,
  444         optimizer_diagnostics={
  445             "backend": "slsqp_with_simplex_newton_refinement",
  446             "actual_workers": actual_workers, "probability_floor": floor,
  447             "prompts": details, "all_certified": all(v["refinement_certified"] for v in details),
  448             "slsqp_failures": sum(not v["slsqp_success"] for v in details)})
```

## E12 — Dual root and certification

Path: `mnpo_scripts/nbpo_generic.py`

SHA256: `2b3502dbd78e6d3228df32b0830182f28bfc3c997dcf2e751a21a086226c2af5`

Original lines 869–972:

```text
  869     def target_log_ratio_check(self) -> float:
  870         """max |(log pi* - log pi_t) - eta sum_k w_k q_k + const| over the pool.
  871 
  872         Zero (to numerical tolerance) iff the stored weights and ``q`` reproduce
  873         the solved policy, which is the identity the pair builder relies on.
  874         """
  875         score = self.eta * torch.einsum("k,kxi->xi", self.weights, self.q_update)
  876         diff = self.target_log_ratio - score
  877         return float((diff - diff.mean(dim=-1, keepdim=True)).abs().max())
  878 
  879 
  880 def validate_finite_pool_solution(res: FinitePoolSolution, *, require_optimality=None,
  881                                  stationarity_tol=1e-4, extra_map_tol=1e-4,
  882                                  dual_tol=1e-6) -> dict:
  883     """One independent certificate for the CLI, stage runner and artifact writer.
  884 
  885     Recompute Q at the policy being serialized, separately from serialization
  886     consistency. Fixed training weights on held-out prompts carry no fitted
  887     held-out Nash-dual claim. Historical R-step artifacts remain identifiable
  888     approximations; they cannot be used as certified canonical teachers.
  889     """
  890     import math
  891     if require_optimality is None:
  892         require_optimality = res.config.get("inner_solver") == "exact"
  893     pi = validate_distribution(res.pi, "pi_star", require_full_support=True)
  894     pt = validate_distribution(res.pi_t, "pi_t", require_full_support=True)
  895     if pi.shape != pt.shape or res.target_log_ratio.shape != pi.shape:
  896         raise ValueError("canonical policy/center/target shapes disagree")
  897     if res.representation_object is None:
  898         raise ValueError("independent validation requires the original representation")
  899     nu, q = res.representation_object.opponent_and_gradient(pi)
  900     log_ratio = torch.log(pi) - torch.log(pt)
  901     serialization = float((res.target_log_ratio - log_ratio).abs().max())
  902     b = log_ratio - res.eta * torch.einsum("k,kxi->xi", res.weights, q)
  903     stationarity = float((b - (pi * b).sum(-1, keepdim=True)).abs().max())
  904     extra_map = float((exp_update(pt, q, res.weights, res.eta) - pi).abs().max())
  905     q_error = float((res.q_update - q).abs().max())
  906     v = res.representation_object.game_values(pi)
  907     surplus = v - res.representation_object.disagreement
  908     diagnostics = res.optimizer_diagnostics
  909     floor = float(diagnostics.get("probability_floor", 0.0))
  910     floor_active = bool(floor and bool((pi <= floor * (1.0 + 1e-6)).any()))
  911     certificate = {
  912         "canonical_serialization_error": serialization,
  913         "independent_stationarity_inf": stationarity,
  914         "extra_map_residual": extra_map, "q_at_returned_policy_error": q_error,
  915         "normalization_error": float((pi.sum(-1) - 1.0).abs().max()),
  916         "minimum_probability": float(pi.min()), "probability_floor": floor,
  917         "probability_floor_active": floor_active,
  918         "objective": float((res.weights * surplus).sum()) - proximal_divergence(pi, pt) / res.eta,
  919         "surplus": surplus.tolist(), "optimizer": diagnostics,
  920         "stationarity_units": "dimensionless_logratio_change",
  921         "tolerances": {"canonical_serialization_error": 1e-9,
  922                        "independent_stationarity_inf": stationarity_tol,
  923                        "extra_map_residual": extra_map_tol, "dual": dual_tol},
  924         "nash_dual_scope": ("fixed_training_weights" if res.config.get("fixed_weights")
  925                             else "training" if res.aggregation == "nash" else "not_applicable"),
  926     }
  927     problems = []
  928     if not math.isfinite(serialization) or serialization >= 1e-9:
  929         problems.append("canonical serialization error >= 1e-9")
  930     if not math.isfinite(stationarity) or stationarity >= stationarity_tol:
  931         problems.append("independent stationarity exceeds tolerance")
  932     if not math.isfinite(extra_map) or extra_map >= extra_map_tol:
  933         problems.append("extra-map residual exceeds tolerance")
  934     if floor_active:
  935         problems.append("probability floor is active")
  936     if diagnostics and not diagnostics.get("all_certified", True):
  937         problems.append("inner optimizer lacks a certified solution/refinement")
  938     if require_optimality and (not math.isfinite(q_error) or q_error >= 1e-9):
  939         problems.append("serialized Q does not match the returned policy")
  940     if require_optimality and (not torch.isfinite(res.nu_update).all() or
  941                               float((res.nu_update - nu).abs().max()) >= 1e-9):
  942         problems.append("serialized opponent does not match the returned policy")
  943     if not all(torch.isfinite(value).all() for value in (res.V, res.d, res.surplus)) or max(float((res.V - v).abs().max()),
  944            float((res.d - res.representation_object.disagreement).abs().max()),
  945            float((res.surplus - surplus).abs().max())) >= 1e-9:
  946         problems.append("stored V/d/s does not match the returned policy")
  947     if res.aggregation == "nash" and not res.config.get("fixed_weights"):
  948         lo, hi = res.config["lambda_box"]
  949         lower, upper = box_active_coordinates(res.weights, lo, hi)
  950         gradient = surplus - 1.0 / res.weights
  951         complementarity = res.weights * surplus - 1.0
  952         projected = projected_kkt_residual(res.weights, surplus, res.gamma_ref or 1.0, lo, hi)
  953         certificate.update({"dual_projected_residual": projected,
  954                             "dual_unprojected_gradient": gradient.tolist(),
  955                             "dual_unprojected_residual": float(gradient.abs().max()),
  956                             "lambda_times_surplus_minus_one": complementarity.tolist(),
  957                             "nash_complementarity_inf": float(complementarity.abs().max()),
  958                             "lambda_at_lower_bound": lower, "lambda_at_upper_bound": upper})
  959         if bool((surplus <= 0).any()):
  960             problems.append("Nash training surplus is not strictly positive")
  961         if projected >= dual_tol or float(gradient.abs().max()) >= dual_tol:
  962             problems.append("original Nash dual stationarity exceeds tolerance")
  963         if float(complementarity.abs().max()) >= dual_tol:
  964             problems.append("lambda * surplus - 1 exceeds tolerance")
  965     if res.aggregation == "kalai_smorodinsky" and res.ks:
  966         if res.ks.get("individual_rationality_violation", 0.0) > 1e-8:
  967             problems.append("KS individual rationality failed")
  968     certificate["certified"] = not problems
  969     certificate["failures"] = problems
  970     if require_optimality and problems:
  971         raise ValueError("finite-pool certificate failed: " + "; ".join(problems))
  972     return certificate
```

Original lines 975–1043:

```text
  975 def _solve_nash_dual_by_root(rep, pi_t, d, lam0, eta, R, inner, damping, lo, hi,
  976                              *, tol, max_calls, history, log_every):
  977     """Solve the Nash dual as a ROOT problem instead of a subgradient descent.
  978 
  979     The dual stationarity condition of the Nash aggregation is exactly
  980 
  981         s_k(pi(lambda)) = 1 / lambda_k      for every k strictly inside the box,
  982 
  983     and ``pi(lambda)`` is a well-defined function once the inner subproblem is
  984     solved rather than iterated. So this is a K-dimensional root problem -- four
  985     dimensions here -- and a projected subgradient with a constant step is a poor
  986     way to attack it: on the controlled benchmark it stalls near a residual of
  987     1e-2 after 3000 outer iterations, nowhere near the 1e-6 gate.
  988 
  989     Solved in log-lambda coordinates so positivity is structural rather than
  990     enforced by clipping, warm-started across evaluations so the inner solves stay
  991     cheap, and reported with the actual number of inner solves used.
  992     """
  993     import numpy as np
  994     from scipy.optimize import root
  995 
  996     state = {"pi": None, "calls": 0}
  997 
  998     def residual(u):
  999         lam = np.exp(np.clip(u, np.log(lo), np.log(hi)))
 1000         sol = inner(rep, pi_t, torch.from_numpy(lam), eta, R,
 1001                     pi_init=state["pi"], damping_=damping)
 1002         state["pi"] = sol.pi
 1003         state["calls"] += 1
 1004         s = (rep.game_values(sol.pi) - d).numpy()
 1005         if log_every and state["calls"] % max(1, log_every) == 0:
 1006             history.append({"iteration": state["calls"], "lambda_raw": lam.tolist(),
 1007                             "surplus": s.tolist(), "min_surplus": float(s.min()),
 1008                             "kkt_residual": float(np.abs(s - 1.0 / lam).max())})
 1009         return s - 1.0 / lam
 1010 
 1011     u0 = np.log(np.clip(as_float64(lam0).numpy(), lo, hi))
 1012     res = root(residual, u0, method="hybr",
 1013                options={"xtol": 1e-14, "maxfev": int(max_calls)})
 1014     history.append({"backend": "scipy_root_hybr", "success": bool(res.success),
 1015                     "status": int(res.status), "message": str(res.message),
 1016                     "function_evaluations": int(res.nfev)})
 1017     lam = torch.from_numpy(np.exp(np.clip(res.x, np.log(lo), np.log(hi))))
 1018     # Convergence is decided by a FRESHLY MEASURED residual at the returned
 1019     # multipliers, not by the optimizer's own success flag. `hybr` reports
 1020     # failure whenever it stops making progress, which it does as soon as the
 1021     # residual reaches the inner solve's own noise floor -- exactly where the
 1022     # answer is correct. Trusting `res.success` would mark good solves as failed
 1023     # and, worse, could mark a bad one as good if the flag ever disagreed.
 1024     final_res = float(np.abs(residual(np.log(lam.numpy()))).max())
 1025 
 1026     # One deterministic restart when the first solve stops short. `lambda = 1/s`
 1027     # at the current point is the natural better start -- it is the stationarity
 1028     # condition itself -- and re-solving from there costs a few dozen inner
 1029     # solves against the tens of thousands a subgradient would need. If it still
 1030     # misses, the miss is reported; the tolerance is not moved to accommodate it.
 1031     if final_res >= tol:
 1032         s_here = (rep.game_values(state["pi"]) - d).numpy() if state["pi"] is not None else None
 1033         if s_here is not None and (s_here > 0).all():
 1034             res2 = root(residual, np.log(np.clip(1.0 / s_here, lo, hi)),
 1035                         method="hybr", options={"xtol": 1e-14, "maxfev": int(max_calls)})
 1036             history.append({"backend": "scipy_root_hybr_restart", "success": bool(res2.success),
 1037                             "status": int(res2.status), "message": str(res2.message),
 1038                             "function_evaluations": int(res2.nfev)})
 1039             lam2 = torch.from_numpy(np.exp(np.clip(res2.x, np.log(lo), np.log(hi))))
 1040             r2 = float(np.abs(residual(np.log(lam2.numpy()))).max())
 1041             if r2 < final_res:
 1042                 lam, final_res = lam2, r2
 1043     return lam, state["pi"], state["calls"], bool(final_res < tol), final_res
```

Original lines 1048–1105:

```text
 1048 def solve_finite_pool(
 1049     rep: ObjectiveRepresentation,
 1050     aggregation: str,
 1051     *,
 1052     eta: float,
 1053     pi_t: Optional[torch.Tensor] = None,
 1054     R: int = 1,
 1055     M: int = 1,
 1056     gamma: Union[float, Sequence[float]] = 0.5,
 1057     lambda_box: Sequence[float] = (1e-3, 1e3),
 1058     lambda_init: Optional[torch.Tensor] = None,
 1059     warm_start_policy: bool = True,
 1060     damping: float = 0.0,
 1061     adversary_step: float = 1.0,
 1062     weight_l1: Optional[float] = None,
 1063     log_every: int = 0,
 1064     ks_kwargs: Optional[dict] = None,
 1065     inner_solver: str = "fixed_point",
 1066     dual_tol: Optional[float] = None,
 1067     dual_solver: str = "subgradient",
 1068     inner_workers: int = 1,
 1069     inner_maxiter: int = 400,
 1070     inner_ftol: float = 1e-14,
 1071     probability_floor: float = 1e-12,
 1072     fixed_weights: Optional[torch.Tensor] = None,
 1073     weights: Optional[torch.Tensor] = None,
 1074     max_bound_expansions: int = 3,
 1075 ) -> FinitePoolSolution:
 1076     """Solve one outer stage for any (representation, aggregation) pair.
 1077 
 1078     ``inner_solver`` selects how the Eq. (18) subproblem is solved at fixed
 1079     weights. ``"fixed_point"`` -- the default, and what every existing artifact
 1080     was produced with -- applies the Eq. (21) map ``R`` times. ``"exact"``
 1081     maximizes the concave subproblem directly, which the controlled-
 1082     nontransitivity audit showed is necessary once raw multipliers grow: at
 1083     ``alpha = 1`` the ``R``-step map has a fixed-point residual of 1.000 and
 1084     lands at min surplus -0.185, while the exact inner solve reaches +0.011
 1085     against an attainable ``rho* = +0.015``. The default is unchanged so no
 1086     existing result moves silently.
 1087     """
 1088     if inner_solver not in ("fixed_point", "exact"):
 1089         raise ValueError("inner_solver must be 'fixed_point' or 'exact'")
 1090     if dual_tol is not None and not (dual_tol > 0):
 1091         raise ValueError("dual_tol must be strictly positive when given")
 1092     if dual_solver not in ("subgradient", "root"):
 1093         raise ValueError("dual_solver must be 'subgradient' or 'root'")
 1094     if aggregation == "kalai_smorodinsky" and (
 1095             inner_workers != 1 or inner_maxiter != 400 or inner_ftol != 1e-14 or probability_floor != 1e-12):
 1096         raise ValueError("KS subsolves currently support only default inner workers/maxiter/ftol/floor")
 1097 
 1098     def _inner(rep_, pi_t_, w_, eta_, R_, *, pi_init=None, damping_=0.0):
 1099         if inner_solver == "exact":
 1100             return solve_proximal_exact(rep_, pi_t_, w_, eta_, pi_init=pi_init,
 1101                                         workers=inner_workers, maxiter=inner_maxiter,
 1102                                         ftol=inner_ftol, probability_floor=probability_floor)
 1103         return solve_proximal(rep_, pi_t_, w_, eta_, R_, pi_init=pi_init,
 1104                               damping=damping_)
 1105 
```

## E13 — Canonical regression branch and mean reduction

Path: `mnpo_scripts/mnpo_trainer.py`

SHA256: `7eeff9c591e530d7c9de2a27406bc8e6ff77469454c9b7e9ba2b366981670159`

Original lines 432–461:

```text
  432             losses = (logits - alpha * target) ** 2
  433 
  434         elif lt == "nbpo":
  435             # Paper-exact finite-temperature NBPO regression, manuscript Eq. (26)
  436             # `eq:regression-loss`:
  437             #   h_t = [log pi(y) - log pi(y')] - [log pi_t(y) - log pi_t(y')]  (Eq. (22))
  438             #   loss = (h_t - eta * nbpo_weighted_z)^2
  439             # history0 is the proximal center pi_t of Eq. (15). The target column
  440             # already carries the RAW dual weights (sum_k lambda_k Z_k, Eq. (24));
  441             # eta is applied exactly once, here. Log-probabilities are sequence
  442             # sums over non-masked response tokens (prompt tokens stay masked by
  443             # the tokenization path); validate_nbpo_args enforces
  444             # logp_reduction="sum" and forbids reference interpolation, alpha/tau,
  445             # multi-history mixing, and every auxiliary loss.
  446             if not history_logps_list:
  447                 raise ValueError("NBPO requires history0 (the proximal center pi_t) via --history_paths.")
  448             if nbpo_target is None:
  449                 raise ValueError(
  450                     f"NBPO requires a `{self.nbpo_target_column}` column containing the "
  451                     "unscaled weighted binary target sum_k lambda_k Z_k."
  452                 )
  453             prev_chosen, prev_rejected = history_logps_list[0]
  454             prev_chosen = torch.as_tensor(prev_chosen, device=device, dtype=torch.float32)
  455             prev_rejected = torch.as_tensor(prev_rejected, device=device, dtype=torch.float32)
  456             target = nbpo_regression_target(
  457                 nbpo_target.to(device=device), self.eta,
  458                 getattr(self, "nbpo_target_mode", "sampled"))
  459             h = pi_logratios - (prev_chosen - prev_rejected)
  460             losses = (h - target) ** 2
  461 
```

Original lines 615–631:

```text
  615             history_logps_list = []
  616         elif (self.nbpo_online_reference or eval_reference) and self.nbpo_reference_model is not None:
  617             counts["reference_forward_tokens"] += input_tokens
  618             counts["reference_response_tokens"] += response_tokens
  619             with torch.no_grad():
  620                 ref_c, ref_r, _, _, _ = self.concatenated_forward(
  621                     self.nbpo_reference_model, batch)
  622             ref_c = ref_c.detach().to(self.accelerator.device, dtype=torch.float32)
  623             ref_r = ref_r.detach().to(self.accelerator.device, dtype=torch.float32)
  624             cached = self.pack_history_logps_from_dataset(batch)
  625             if cached:
  626                 cc, cr = cached[0]
  627                 self._nbpo_ref_vs_cache = (
  628                     (ref_c - cc.to(ref_c.device)) - (ref_r - cr.to(ref_r.device))
  629                 ).detach()
  630             history_logps_list = [(ref_c, ref_r)]
  631             reference_chosen_logps, reference_rejected_logps = ref_c, ref_r
```

Original lines 702–724:

```text
  702         # 3. Compute loss
  703         losses, chosen_rewards, rejected_rewards = self.mnpo_loss(
  704             policy_chosen_logps,
  705             policy_rejected_logps,
  706             reference_chosen_logps,
  707             reference_rejected_logps,
  708             history_logps_list,
  709             chosen_probs,
  710             rejected_probs,
  711             ronpo_target,
  712             ht_target,
  713             nbpo_target,
  714             batch.get("nbpo_weight_a"),
  715             batch.get("nbpo_weight_b"),
  716             batch.get("nbpo_num_candidates"),
  717         )
  718 
  719         if self.loss_type == "ronpo" and ronpo_weight is not None:
  720             weight = ronpo_weight / ronpo_weight.mean().clamp_min(1e-8)
  721             losses = losses * weight
  722 
  723         core_loss = losses.mean()
  724         loss = core_loss
```

## E14 — Canonical target units are not rescaled

Path: `mnpo_scripts/nbpo_neural.py`

SHA256: `b6df54c0771bc5d9b55092a117655fd08da609a19a25cdbeebbb2605a8b67ddc`

Original lines 1–140:

```text
    1 """Canonical neural realization and pair-dataset contracts, without Trainer imports."""
    2 from __future__ import annotations
    3 
    4 import itertools
    5 import math
    6 import torch
    7 
    8 from mnpo_scripts.pair_tokenization import immutable_pair_tokens
    9 
   10 
   11 def nbpo_regression_target(target, eta, mode="sampled"):
   12     if mode == "canonical_logratio":
   13         return target.float()
   14     if mode not in ("sampled", "rb", "rao_blackwell", "canonical"):
   15         raise ValueError(f"Unknown NBPO target mode {mode!r}")
   16     return float(eta) * target.float()
   17 
   18 
   19 def nbpo_wbc_pair_loss(logp_a, logp_b, weight_a, weight_b, num_candidates=8):
   20     """The all-unordered-pair average equals prompt weighted sequence NLL."""
   21     a, b = logp_a.float(), logp_b.float()
   22     wa = torch.as_tensor(weight_a, device=a.device, dtype=torch.float32).detach()
   23     wb = torch.as_tensor(weight_b, device=a.device, dtype=torch.float32).detach()
   24     n = torch.as_tensor(num_candidates, device=a.device, dtype=torch.float32)
   25     if torch.any(n != 8):
   26         raise ValueError("Primary WBC supports fixed N=8 only")
   27     if torch.any(~torch.isfinite(wa)) or torch.any(~torch.isfinite(wb)) or torch.any(wa < 0) or torch.any(wb < 0):
   28         raise ValueError("WBC requires finite nonnegative detached solver probability masses")
   29     if wa.shape != a.shape or wb.shape != b.shape:
   30         raise ValueError("Candidate weights must match their actual pair logps")
   31     return (n / 2) * (-wa * a - wb * b)
   32 
   33 
   34 def select_training_splits(datasets, train_split="train", eval_split="dev"):
   35     """Names are exact; final test is never selected by a substring heuristic."""
   36     if train_split not in datasets:
   37         raise ValueError(f"Missing train_split={train_split!r}; available={list(datasets)}")
   38     if eval_split == "test":
   39         raise ValueError("Final test must use a separate evaluation entrypoint; choose dev explicitly")
   40     if eval_split is not None and eval_split not in datasets:
   41         raise ValueError(f"Missing eval_split={eval_split!r}; available={list(datasets)}")
   42     if train_split == eval_split:
   43         raise ValueError("Train and evaluation splits must be distinct")
   44     return datasets[train_split], datasets[eval_split] if eval_split else None
   45 
   46 
   47 def validate_mopo_rho_dataset(dataset, max_length=2048, max_prompt_length=1024,
   48                               expected_solver_hash=None):
   49     """Check a MOPO importance-weight dataset: one row per (prompt, candidate).
   50 
   51     MOPO's policy step is importance-weighted behaviour cloning, -rho(y) log pi(y)
   52     on the chosen side only, so this format is one row per candidate rather than
   53     per unordered pair. That means it cannot satisfy
   54     validate_canonical_pair_dataset, and it also means the shared 28-pair
   55     enumeration must not be reused: with chosen = i < j, candidate 0 is chosen
   56     seven times and candidate 7 never, which would weight the estimate by
   57     candidate index instead of by rho.
   58 
   59     The checks are the per-candidate analogues of the canonical ones. rho must be
   60     a genuine importance ratio under the reference policy, which on a uniform
   61     eight-candidate pool means the eight rho values of a prompt average to one;
   62     that is the property the whole objective rests on, so it is checked rather
   63     than assumed. The rejected side carries tokens because the shared collator
   64     needs a pair, and it contributes to no term of the loss.
   65     """
   66     groups = {}
   67     solvers = set()
   68     for row in dataset:
   69         if row.get("target_mode") != "mopo_rho" or row.get("target_units") != "importance_ratio":
   70             raise ValueError("MOPO row must declare target_mode mopo_rho and importance_ratio units")
   71         rho = float(row["ronpo_target"])
   72         if not math.isfinite(rho) or rho < 0:
   73             raise ValueError("MOPO importance weight must be finite and nonnegative")
   74         if int(row["nbpo_num_candidates"]) != 8:
   75             raise ValueError("MOPO dataset for this panel requires N=8")
   76         tokens = immutable_pair_tokens(row, max_length, max_prompt_length)
   77         if tokens is None:
   78             raise ValueError("MOPO dataset requires immutable sampled tokens")
   79         key = str(row["prompt_id"])
   80         group = groups.setdefault(key, {"candidates": {}, "prompt": tokens["prompt_input_ids"]})
   81         if group["prompt"] != tokens["prompt_input_ids"]:
   82             raise ValueError("One prompt group has different conditioning token contexts")
   83         candidate_id = row.get("chosen_response_id", row.get("chosen_candidate_id"))
   84         if candidate_id is None:
   85             raise ValueError("MOPO row must identify the cloned candidate occurrence")
   86         value = (tokens["chosen_token_sha256"], rho)
   87         old = group["candidates"].setdefault(str(candidate_id), value)
   88         if old != value:
   89             raise ValueError("A candidate appears twice with different tokens or weight")
   90         solver_hash = row.get("solver_artifact_sha256")
   91         if not solver_hash:
   92             raise ValueError("MOPO row lacks solver_artifact_sha256")
   93         if expected_solver_hash and solver_hash != expected_solver_hash:
   94             raise ValueError("Unexpected solver artifact hash")
   95         solvers.add(solver_hash)
   96     if not groups:
   97         raise ValueError("Empty MOPO dataset")
   98     for key, group in groups.items():
   99         weights = [value[1] for value in group["candidates"].values()]
  100         if len(weights) != 8:
  101             raise ValueError("Every prompt requires one row per each of its 8 candidates")
  102         if not math.isclose(sum(weights) / 8.0, 1.0, abs_tol=1e-6):
  103             raise ValueError("Prompt importance weights do not average to one under the "
  104                              "uniform reference; rho is then not an importance ratio")
  105     return {"prompts": len(groups), "rows": len(dataset), "candidates_per_prompt": 8,
  106             "format": "one row per (prompt, candidate)",
  107             "solver_artifact_sha256": sorted(solvers)}
  108 
  109 
  110 def validate_canonical_pair_dataset(dataset, max_length=2048, max_prompt_length=1024,
  111                                     expected_solver_hash=None):
  112     """Check pair completeness, candidate mass/context identity and eta units.
  113 
  114     Duplicate sampled strings remain separate occurrences. Each of the 28
  115     unordered pairs must occur exactly once for every prompt, with fixed N=8.
  116     """
  117     groups = {}
  118     solvers = set()
  119     for row in dataset:
  120         if row.get("target_mode") != "canonical_logratio" or row.get("target_units") != "final_logratio_change" or row.get("eta_already_included") is not True:
  121             raise ValueError("Canonical row must declare final logratio target units and eta included")
  122         if not math.isfinite(float(row["nbpo_logratio_target"])):
  123             raise ValueError("Nonfinite canonical target")
  124         if int(row["nbpo_num_candidates"]) != 8:
  125             raise ValueError("Primary all-pair dataset requires N=8")
  126         wa, wb = float(row["nbpo_weight_a"]), float(row["nbpo_weight_b"])
  127         if min(wa, wb) <= 0 or not math.isfinite(wa + wb):
  128             raise ValueError("Canonical targets require strictly positive solver masses")
  129         center_a = float(row.get("nbpo_center_a", 1.0 / 8))
  130         center_b = float(row.get("nbpo_center_b", 1.0 / 8))
  131         if center_a != 1.0 / 8 or center_b != 1.0 / 8:
  132             raise ValueError("Primary IID occurrence center must be exactly 1/8")
  133         expected_target = math.log(wa / center_a) - math.log(wb / center_b)
  134         if abs(float(row["nbpo_logratio_target"]) - expected_target) > 1e-9:
  135             raise ValueError("Canonical target disagrees with log(p_star/p_t) candidate masses")
  136         tokens = immutable_pair_tokens(row, max_length, max_prompt_length)
  137         if tokens is None:
  138             raise ValueError("Canonical dataset requires immutable sampled tokens")
  139         key = str(row["prompt_id"])
  140         group = groups.setdefault(key, {"pairs": set(), "candidates": {}, "prompt": tokens["prompt_input_ids"]})
```

## E15 — Response log probabilities and backward

Path: `mnpo_scripts/response_logps.py`

SHA256: `ad2a8205003df1a2949bf2a7d33b94aede3eb978086e426919448a7dc88793a1`

Original lines 1–63:

```text
    1 """Shared FP32 response scoring, with bounded full-vocabulary temporaries.
    2 
    3 The custom backward recomputes one chunk's softmax, avoiding retention of an
    4 entire FP32 [batch, sequence, vocabulary] activation until backward. It is the
    5 ordinary full-vocabulary log-softmax derivative, not a candidate softmax.
    6 """
    7 from __future__ import annotations
    8 
    9 import torch
   10 import torch.nn.functional as F
   11 
   12 LOGP_IMPLEMENTATION = "fp32_chunked_selected_logsoftmax_v1"
   13 
   14 
   15 class _SelectedLogps(torch.autograd.Function):
   16     @staticmethod
   17     def forward(ctx, logits, labels, chunk_size):
   18         ctx.save_for_backward(logits, labels)
   19         ctx.chunk_size = chunk_size
   20         selected = torch.empty(labels.shape, device=logits.device, dtype=torch.float32)
   21         for start in range(0, logits.shape[1], chunk_size):
   22             end = start + chunk_size
   23             normalized = F.log_softmax(logits[:, start:end], dim=-1, dtype=torch.float32)
   24             selected[:, start:end] = normalized.gather(-1, labels[:, start:end, None]).squeeze(-1)
   25         return selected
   26 
   27     @staticmethod
   28     def backward(ctx, grad_selected):
   29         logits, labels = ctx.saved_tensors
   30         grad_logits = torch.empty_like(logits)
   31         for start in range(0, logits.shape[1], ctx.chunk_size):
   32             end = start + ctx.chunk_size
   33             derivative = -F.softmax(logits[:, start:end], dim=-1, dtype=torch.float32)
   34             index = labels[:, start:end, None]
   35             derivative.scatter_add_(-1, index, torch.ones_like(index, dtype=torch.float32))
   36             derivative.mul_(grad_selected[:, start:end, None].float())
   37             grad_logits[:, start:end] = derivative.to(logits.dtype)
   38         return grad_logits, None, None
   39 
   40 
   41 def response_logps(logits, labels, average_log_prob=False, label_pad_token_id=-100,
   42                    is_encoder_decoder=False, chunk_size=64):
   43     """FP32 log-softmax, selected token values and response sum/legacy mean.
   44 
   45 Only labels define the response event. No EOS is added and no prompt or padded
   46 token is scored. Decoder-only logits predict the next label.
   47 """
   48     if logits.ndim != 3 or logits.shape[:-1] != labels.shape:
   49         raise ValueError("Logits and labels must have the same batch/sequence shape")
   50     if int(chunk_size) <= 0:
   51         raise ValueError("logp chunk_size must be positive")
   52     if not is_encoder_decoder:
   53         labels, logits = labels[:, 1:], logits[:, :-1]
   54     mask = labels.ne(label_pad_token_id)
   55     safe_labels = labels.masked_fill(~mask, 0)
   56     selected = _SelectedLogps.apply(logits, safe_labels, int(chunk_size))
   57     result = selected.masked_fill(~mask, 0).sum(-1, dtype=torch.float32)
   58     if average_log_prob:
   59         lengths = mask.sum(-1)
   60         if torch.any(lengths == 0):
   61             raise ValueError("Cannot average a response with no scored tokens")
   62         result = result / lengths.float()
   63     return result
```

## E16 — Frozen reference and unconditional final save

Path: `mnpo_scripts/run_mnpo.py`

SHA256: `def0546b64f7d2d5dd6360a7a3ff6995c126a06ff0e63326c965443cdd05d36a`

Original lines 312–331:

```text
  312     # The frozen proximal centre, loaded AFTER the trainer so it shares the
  313     # accelerator's device placement. It is a separate copy of pi_t held in eval
  314     # mode with gradients off: it is never the learner detached, and it is never
  315     # re-synced to the learner during training. Its only job is to be forwarded
  316     # through the same collated batch as the policy, so that h is a difference of
  317     # two log-probabilities computed on the identical kernel path.
  318     if (getattr(training_args, "nbpo_online_reference", False)
  319             or getattr(training_args, "nbpo_eval_online_reference", False)):
  320         ref_path = getattr(training_args, "nbpo_reference_model_path", "") or \
  321             model_args.model_name_or_path
  322         logger.info(f"*** Loading frozen NBPO reference (pi_t) from {ref_path} ***")
  323         ref = AutoModelForCausalLM.from_pretrained(
  324             ref_path, torch_dtype=next(trainer.model.parameters()).dtype, use_cache=False,
  325             revision=model_args.model_revision, attn_implementation=model_args.attn_implementation,
  326             trust_remote_code=model_args.trust_remote_code)
  327         ref.eval()
  328         for prm in ref.parameters():
  329             prm.requires_grad_(False)
  330         trainer.nbpo_reference_model = ref.to(trainer.accelerator.device)
  331         logger.info("*** NBPO online reference active: pi_t forwarded per batch ***")
```

Original lines 347–387:

```text
  347     ###############
  348     # Training loop
  349     ###############
  350     checkpoint = None
  351     if training_args.resume_from_checkpoint is not None:
  352         checkpoint = training_args.resume_from_checkpoint
  353     elif last_checkpoint is not None:
  354         checkpoint = last_checkpoint
  355     train_result = trainer.train(resume_from_checkpoint=checkpoint)
  356     metrics = train_result.metrics
  357     metrics["train_samples"] = len(train_dataset)
  358     trainer.log_metrics("train", metrics)
  359     trainer.save_metrics("train", metrics)
  360     trainer.save_state()
  361 
  362     logger.info("*** Training complete ***")
  363     if training_args.nbpo_profile_updates > 0:
  364         logger.info("Disposable runtime profile complete at %d updates; scheduler horizon was %d. No model export.",
  365                     trainer.state.global_step, training_args.max_steps)
  366         return
  367 
  368     skip_final_save = os.environ.get("MNPO_SKIP_FINAL_SAVE", "").lower() in {"1", "true", "yes"}
  369     if skip_final_save:
  370         logger.info("*** Skip final model save because MNPO_SKIP_FINAL_SAVE is set ***")
  371         if trainer.accelerator.is_main_process:
  372             trainer.tokenizer.save_pretrained(training_args.output_dir)
  373             trainer.model.config.use_cache = True
  374             trainer.model.config.save_pretrained(training_args.output_dir)
  375         logger.info("*** Training complete! ***")
  376         return
  377 
  378     ##################################
  379     # Save model and create model card
  380     ##################################
  381     logger.info("*** Save model ***")
  382     trainer.save_model(training_args.output_dir)
  383     logger.info(f"Model saved to {training_args.output_dir}")
  384 
  385     # Add this step to explicitly save the tokenizer.
  386     if trainer.accelerator.is_main_process:
  387         trainer.tokenizer.save_pretrained(training_args.output_dir)
```

## E17 — Current campaign invokes bare trainer, with pod-only config

Path: `analysis/sub_20260914/code_snapshot_20260917_union/make_pros_train_jobs.py`

SHA256: `bcabadf496632acbcefd6334b84aae5e95f5a85f129885cd4b526f0ee3feba62`

Original lines 20–31:

```text
   20 import hashlib
   21 import json
   22 from pathlib import Path
   23 
   24 ROOT = Path("/work/uf4_20260910")
   25 BASE_CONFIG = Path("/work/nbpo_repair_20260909/configs/mse_short_primary_v1.yaml")
   26 
   27 
   28 def file_hash(path):
   29     h = hashlib.sha256()
   30     with open(path, "rb") as stream:
   31         for chunk in iter(lambda: stream.read(1 << 20), b""):
```

Original lines 75–107:

```text
   75     complete = json.loads((target_dir / "complete.json").read_text())
   76     dataset = complete["dataset_path"]
   77     manifest = complete["dataset_manifest_sha256"]
   78     solver = file_hash(target_dir / "train/solver/solution.json")
   79     if not dataset or not manifest:
   80         raise ValueError(f"{args.targets} has no materialized dataset")
   81 
   82     base = BASE_CONFIG.read_text()
   83     target_mode = args.target_mode or ("mopo_rho" if args.loss_type == "mopo"
   84                                        else "canonical_logratio")
   85     replacements = {
   86         "/work/nbpo_repair_20260909/datasets/nash_repair_v2: 1.0": f"{dataset}: 1.0",
   87         "output_dir: /work/nbpo_repair_20260909/arms/mse_short_primary_v1":
   88             f"output_dir: {ROOT}/arms/{args.arm}",
   89         "run_name: mse_short_primary_v1": f"run_name: {args.arm}",
   90         "nbpo_expected_dataset_manifest_sha256: 856ba968818f4672eace40d2794a1889a99ebd7e866da0531e64d57e01564fac":
   91             f"nbpo_expected_dataset_manifest_sha256: {manifest}",
   92         "nbpo_expected_solver_artifact_sha256: 4fd522a7891fa7ba4ab3bcfcd82f0d72a5b50de23a8b9239a8e480b74588afc8":
   93             f"nbpo_expected_solver_artifact_sha256: {solver}",
   94         "max_steps: 250": f"max_steps: {args.max_steps}",
   95         # Newline-anchored: "seed: 42" is a substring of "data_seed: 42", so an
   96         # unanchored replacement rewrites both and then loses its own anchor.
   97         # At seed 42 that was invisible because the rewrite was a no-op.
   98         "\nseed: 42": f"\nseed: {args.seed}",
   99         "\ndata_seed: 42": f"\ndata_seed: {args.seed}",
  100         "save_steps: 250": f"save_steps: {args.max_steps}",
  101         # Diagnostic eval only: load_best_model_at_end is false and save_steps
  102         # equals max_steps, so nothing selects on it. Measured cost is 15 min 45 s
  103         # per pass on the 28,000-pair dev set at eval batch 1, so eval_steps=50 on
  104         # a 1250-update run spends 82% of wall time and 26 GPU-hours on logging.
  105         # 250 restores the five evaluations the validated 250-update recipe ran.
  106         "eval_steps: 50": f"eval_steps: {args.eval_steps}",
  107     }
```

Original lines 143–168:

```text
  143     config_path = ROOT / "configs" / f"{args.arm}.yaml"
  144     config_path.parent.mkdir(parents=True, exist_ok=True)
  145     if config_path.exists() and config_path.read_text() != config:
  146         raise ValueError(f"Refusing to overwrite {config_path} with different content")
  147     config_path.write_text(config)
  148 
  149     spec = {
  150         "job_id": f"{args.job_prefix}_train_{args.arm}",
  151         "priority": args.priority,
  152         "gpus": 4,
  153         "cwd": "/work/nbpo_repair_20260909/code",
  154         "env": {"PYTHONPATH": "/work/nbpo_repair_20260909/deps_train:/work/nbpo_repair_20260909/code",
  155                 "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "4",
  156                 "MNPO_DISABLE_APEX": "1", "HF_HUB_OFFLINE": "1",
  157                 "TOKENIZERS_PARALLELISM": "false", "WANDB_MODE": "disabled",
  158                 "VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
  159         "command": ["python3", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
  160                     "--nproc_per_node=4", "-m", "mnpo_scripts.run_mnpo", str(config_path)],
  161         "timeout_s": 86400,
  162         "artifacts": [f"{ROOT}/arms/{args.arm}/config.json"],
  163         # The spec names the config by path, so without this a changed recipe
  164         # leaves the spec byte-identical and a queue keyed on spec bytes would
  165         # never notice that this is a different run.
  166         "config_sha256": file_hash(config_path),
  167         "dataset_manifest_sha256": manifest,
  168         "solver_artifact_sha256": solver,
```

## E18 — Panel pipeline invokes scorer and trainer directly

Path: `analysis/sub_20260914/code_snapshot_20260917_union/panel_stage2.py`

SHA256: `e0528503bf22fa970cc2d9a963b35a183bd3a970719240c2d3e6b87c6667734b`

Original lines 94–111:

```text
   94     p, K = args.panel, args.objectives
   95     certified = "%sp" % args.split_tag        # us_v1 -> us_v1p
   96     representable = "%sr" % args.split_tag    # us_v1 -> us_v1r
   97     log = SUB / ("%s_stage2.log" % p)
   98     record_path = SUB / ("%s_stage2.json" % p)
   99     record = json.loads(record_path.read_text()) if record_path.exists() else {"panel": p}
  100 
  101     def step(n):
  102         return args.from_step <= n <= args.to_step
  103 
  104     if step(1):
  105         run(["python3", str(CODE / "union_score_panel.py"), "--objectives", str(K),
  106              "--tag", "%s_psc" % p, "--judge-shards", str(args.shards),
  107              "--pool", p, "--pool-shards", str(args.shards),
  108              "--out", p, "--shards", str(args.shards)],
  109             SOLVE_ENV, "1_score", log)
  110         record["score_dir"] = str(UF / "scores" / p)
  111 
```

Original lines 264–349:

```text
  264 ARM_SPECS = [
  265     # arm id, dataset suffix, loss, extra config. The three baseline rows keep
  266     # the hyperparameters their UW counterparts used, so a panel-to-panel
  267     # difference is the panel and not a retuning.
  268     ("nbpo", "nbpo", "nbpo", {}),
  269     ("pw_nbpo", "pw_nbpo", "nbpo", {}),
  270     ("dpo_soft", "soft_pavg", "dpo_soft", {"dpo_beta": 0.1}),
  271     ("inpo_soft", "soft_pavg", "inpo_soft", {"eta": 0.005, "ratio": 1.0 / 3.0,
  272                                              "inpo_prev_equals_reference": True}),
  273     ("prosper", "prosper", "ronpo", {"ronpo_alpha": 1.0, "ronpo_tau": 0.0,
  274                                      "ronpo_target_column": "ronpo_target",
  275                                      "eta": 1.0,
  276                                      "inpo_prev_equals_reference": True}),
  277 ]
  278 
  279 
  280 def queue_training(panel, epochs, record, max_length, max_prompt_length,
  281                    config_dir, queue_dir):
  282     """One 4-GPU training spec per arm, at a step count matched across arms."""
  283     import yaml
  284     base = yaml.safe_load(Path(UF, "configs", "uw1_pw_nbpo.yaml").read_text())
  285     tokens_per_step = (base["per_device_train_batch_size"]
  286                        * base["gradient_accumulation_steps"] * 4)
  287     queued = {}
  288     for arm, suffix, loss, extra in ARM_SPECS:
  289         ds = UF / "datasets" / ("%s_%s" % (panel, suffix))
  290         if not ds.exists():
  291             print(json.dumps({"skip_arm": arm, "why": "no dataset at %s" % ds}), flush=True)
  292             continue
  293         rows = dataset_rows(str(ds), "train")
  294         steps = int(math.ceil(epochs * rows / tokens_per_step))
  295         cfg = dict(base)
  296         cfg["max_length"] = max_length
  297         cfg["max_prompt_length"] = max_prompt_length
  298         cfg.update(extra)
  299         cfg["loss_type"] = loss
  300         cfg["max_steps"] = steps
  301         cfg["eval_steps"] = steps
  302         cfg["save_steps"] = steps
  303         cfg["dataset_mixer"] = {str(ds): 1.0}
  304         cfg["output_dir"] = str(UF / "arms" / ("%s_%s" % (panel, arm)))
  305         cfg["run_name"] = "%s_%s" % (panel, arm)
  306         cfg["model_name_or_path"] = BASE_MODEL
  307         cfg["model_revision"] = BASE_REV
  308         cfg["nbpo_reference_model_path"] = BASE_MODEL
  309         # The reference-handling and immutable-token plumbing stays on for every
  310         # arm, exactly as the verified UW baseline configs have it: the soft-label
  311         # and PROSPER losses also read reference log-ratios and must score the
  312         # pool's own tokens. Only the target-column fields are NBPO-specific.
  313         for key in ("nbpo_expected_dataset_manifest_sha256",
  314                     "nbpo_expected_solver_artifact_sha256"):
  315             cfg.pop(key, None)
  316         if loss != "nbpo":
  317             for key in ("nbpo_target_mode", "nbpo_target_column", "nbpo_target_units",
  318                         "nbpo_eta_already_included"):
  319                 cfg.pop(key, None)
  320         if loss == "dpo_soft":
  321             cfg.pop("eta", None)              # dpo_soft has no proximal step
  322         if loss == "nbpo":
  323             # pin the dataset manifest and the solver solution this arm was built
  324             # from, so a config cannot be pointed at a different solve later
  325             done = json.loads(Path(UF, "targets", "%s_%s" % (panel, arm),
  326                                    "complete.json").read_text())
  327             man = done.get("dataset_manifest_sha256")
  328             sol = (done["splits"]["train"].get("solver_solution_sha256")
  329                    if "splits" in done else None)
  330             if not man or not sol:
  331                 raise SystemExit("%s: solve record has no manifest/solution hash" % arm)
  332             cfg["nbpo_expected_dataset_manifest_sha256"] = man
  333             cfg["nbpo_expected_solver_artifact_sha256"] = sol
  334         config_dir.mkdir(parents=True, exist_ok=True)
  335         path = config_dir / ("%s_%s.yaml" % (panel, arm))
  336         path.write_text(yaml.safe_dump(cfg, sort_keys=True))
  337         spec = {
  338             "job_id": "%s_train_%s" % (panel, arm), "priority": 120, "gpus": 4,
  339             "depends_on": [], "cwd": "%s/code" % DEPS,
  340             "env": {"PYTHONPATH": "%s/deps_train:%s/code" % (DEPS, DEPS),
  341                     "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
  342                     "MKL_NUM_THREADS": "4", "MNPO_DISABLE_APEX": "1",
  343                     "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
  344                     "WANDB_MODE": "disabled",
  345                     "VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
  346             "timeout_s": 86400,
  347             "command": ["python3", "-m", "torch.distributed.run", "--standalone",
  348                         "--nnodes=1", "--nproc_per_node=4", "-m",
  349                         "mnpo_scripts.run_mnpo", str(path)],
```

## E19 — Legacy gate and legacy global solver entry

Path: `scripts/nbpo/run_nbpo_stage.py`

SHA256: `d1823f4f204edef279d79fa0b5ec80cf295efb99c4e6ccbdae1e310328c2f475`

Original lines 679–689:

```text
  679 def apply_gate(min_surplus: float, parent_dir: Path, candidate_dir: Path,
  680                promote_to: Path, stage: int = 0, fingerprint: str = None) -> dict:
  681     """Algorithm 1 lines 11-15: reject on any nonpositive held-out surplus, else promote."""
  682     if min_surplus <= 0:
  683         return {"accepted": False, "promoted_path": str(parent_dir),
  684                 "reason": f"min held-out surplus {min_surplus:.6f} <= 0; "
  685                           "stage flagged empirically infeasible, pi_t retained"}
  686     fp = fingerprint or checkpoint_fingerprint(str(candidate_dir))
  687     rec = promote_candidate(candidate_dir, promote_to, stage, fp)
  688     return {"accepted": True, "promoted_path": rec["versioned_dir"], "symlink": str(promote_to),
  689             "reason": f"min held-out surplus {min_surplus:.6f} > 0", "promotion": rec}
```

Original lines 845–888:

```text
  845     scfg = cfg["solver"]
  846     tensor_meta_construction = json.loads(
  847         (tensor_dir / "meta.json").read_text()).get("reference_construction", "shared_pool")
  848     A_policy = torch.from_numpy(np.load(tensor_dir / "tensor_policy.npz")["A"])
  849     A_ref = torch.from_numpy(np.load(tensor_dir / "tensor_ref.npz")["A"])
  850     mu = uniform_policy(A_policy.shape[1], A_policy.shape[3])
  851     beta = torch.tensor(scfg["opponent_betas"], dtype=torch.float64)
  852     lambda_init = None
  853     warm = scfg.get("warm_start_lambda")
  854     if warm:
  855         prev = json.loads(_resolve(base, warm).read_text())
  856         lambda_init = torch.tensor(prev["lambda_raw"], dtype=torch.float64)
  857     res = solve_nbpo_dual(
  858         A_policy, A_ref, mu, beta,
  859         eta=float(scfg["eta"]), gamma=scfg["gamma"], M=int(scfg["M"]), R=int(scfg["R"]),
  860         lambda_box=tuple(scfg.get("lambda_box", (1e-3, 1e3))),
  861         lambda_init=lambda_init, aggregation=scfg.get("aggregation", "nash"),
  862         reference_construction=tensor_meta_construction,
  863         damping=float(scfg.get("damping", 0.0)),
  864     )
  865     from scripts.nbpo.solve_nbpo_dual import write_solution_artifact
  866     tensor_meta = json.loads((tensor_dir / "meta.json").read_text())
  867     hashes = {name: sha256_file(tensor_dir / name)
  868               for name in ("tensor_policy.npz", "tensor_ref.npz", "meta.json")}
  869     solver_dir = workdir / "solver"
  870     solution = write_solution_artifact(solver_dir, res, tensor_meta, hashes, tensor_dir,
  871                                        stage, lambda_warm_started=lambda_init is not None)
  872     _step("solve_dual", f"lambda={solution['lambda_raw']} "
  873                         f"inv_surplus_res={solution['inverse_surplus_residual']} "
  874                         f"projected_kkt={solution['projected_kkt_residual']} R={scfg['R']}"
  875                         + (" (R=1: disclosed approximation)" if int(scfg["R"]) == 1 else ""))
  876 
  877     # 5. Pair targets from nu_update (hash-verified inside the builder).
  878     tcfg = cfg.get("targets", {})
  879     pairs_dir = workdir / "pairs"
  880     summary = build_pairs_from_artifacts(
  881         tensor_dir, solver_dir, policy_specs, pairs_dir,
  882         target_mode=tcfg.get("mode", "sampled"), seed=int(tcfg.get("seed", 42)), stage=stage,
  883         reproduction_mode=reproduction_mode,
  884         learner_manifest_sha256=(pool_bindings.get("learner_pool") or {}).get("manifest_sha256"),
  885         # Prompt-level hold-out: whole prompts move to the test split, so no
  886         # prompt's pairs straddle it.
  887         test_prompts=int(tcfg.get("test_prompts", 0) or 0),
  888         split_salt=str(tcfg.get("split_salt", "nbpo-v1")))
```

## E20 — Legacy monitoring evaluates prompt-averaged game surplus

Path: `scripts/nbpo/eval_game_value.py`

SHA256: `0f4c17b91c3ab2ff6524ae1eefcf1847f75ce77acfcbf33aa51becbed6364f03`

Original lines 41–64:

```text
   41 def evaluate_game_value(A_policy: torch.Tensor, A_ref: torch.Tensor, beta: torch.Tensor,
   42                         reference_construction: str = "shared_pool") -> dict:
   43     """Pure evaluation given the two centered tensors; policy uniform over its pool."""
   44     K, X, I, J = A_policy.shape
   45     mu = uniform_policy(X, J)
   46     pi = uniform_policy(X, I)
   47     r = compute_margins(A_policy, pi)
   48     V = compute_regularized_game_value(r, mu, beta, form="softmin")
   49     d = compute_disagreement_point(A_ref, mu, beta, reference_construction)
   50     s = V - d
   51     nu = compute_regularized_opponent(r, mu, beta)
   52     all_positive = bool((s > 0).all())
   53     return {
   54         "V": [float(v) for v in V],
   55         "d": [float(v) for v in d],
   56         "surplus": [float(v) for v in s],
   57         "min_surplus": float(s.min()),
   58         "avg_surplus": float(s.mean()),
   59         "nash_welfare_defined": all_positive,
   60         # Nash welfare only exists on the individually-rational set (Eq. (11));
   61         # a nonpositive surplus makes it undefined, not "very negative".
   62         "nash_welfare": float(torch.log(s).sum()) if all_positive else None,
   63         "opponent_entropy": [float(v) for v in opponent_entropy(nu)],
   64         "opponent_ess": [float(v) for v in opponent_ess(nu)],
```

## E21 — Learner-pair baseline depends on LL tensor

Path: `analysis/sub_20260914/code_snapshot_20260917_union/build_panel_softlabels.py`

SHA256: `e534b35d1813072fac894427f2ea8daa2a829080e9b19d0b8c1bc2785c878438`

Original lines 1–17:

```text
    1 """Attach order-balanced soft preference labels to a panel's pair dataset.
    2 
    3 The scalarized-DPO and mean-preference-INPO rows of Table 3 need, for each
    4 training pair, the probability that the chosen response is preferred to the
    5 rejected one under the SAME judgments the NBPO teacher used. That probability is
    6 the panel's objective-averaged direct PSC probability
    7 
    8     p = (1/K) sum_k [ A_policy[k, x, i, j] + 1/2 ],
    9 
   10 where A_policy is the antisymmetric, zero-diagonal learner block written by the
   11 panel scorer, i and j are the row's own candidate indices, and the +1/2 turns
   12 the antisymmetric deviation back into a probability. No Bradley-Terry fit and no
   13 surrogate reward model enters: these are the measured order-averaged verdicts.
   14 
   15 The tokenization, prompts and pair set are copied unchanged from the NBPO-PW
   16 dataset of the same panel, so the two baselines see exactly the pairs the NBPO
   17 arms see. Only the two probability columns are added.
```

Original lines 32–42:

```text
   32 def load_policy_block(scores_dir, shards):
   33     """prompt_id -> A_policy, the antisymmetric learner block."""
   34     out = {}
   35     for shard in range(shards):
   36         d = Path(scores_dir) / ("shard%d" % shard)
   37         for path in sorted(d.glob("chunk*.npz")):
   38             z = np.load(path, allow_pickle=True)
   39             pids = [str(v) for v in z["prompt_ids"]]
   40             A = z["a_policy"] if "a_policy" in z.files else z["A_policy"]
   41             for x, pid in enumerate(pids):
   42                 out[pid] = np.asarray(A[:, x], dtype=np.float64)
```

Original lines 66–88:

```text
   66     def attach(batch):
   67         ps = []
   68         for pid, i, j in zip(batch["prompt_id"], batch["chosen_candidate_index"],
   69                              batch["rejected_candidate_index"]):
   70             A = blocks[pid]
   71             vals = A[:, i, j] + 0.5
   72             ps.append(float(vals.mean()))
   73         return {"chosen_probs": ps, "rejected_probs": [1.0 - p for p in ps]}
   74 
   75     out = {}
   76     for split in data:
   77         d = data[split]
   78         missing = sorted({p for p in d["prompt_id"] if p not in blocks})
   79         if missing:
   80             raise SystemExit("%s: %d prompts have no score block" % (split, len(missing)))
   81         any_pid = d["prompt_id"][0]
   82         K = blocks[any_pid].shape[0]
   83         # the block must be antisymmetric with a zero diagonal, or +1/2 is wrong
   84         A = blocks[any_pid]
   85         if float(np.abs(A + np.swapaxes(A, 1, 2)).max()) > 1e-9:
   86             raise SystemExit("learner block is not antisymmetric")
   87         if float(np.abs(np.einsum("kii->ki", A)).max()) > 1e-12:
   88             raise SystemExit("learner block has a nonzero diagonal")
```

## E22 — Existing root tensor builder correctly distinguishes learner/reference roles

Path: `scripts/nbpo/build_preference_tensor.py`

SHA256: `905e4916b7d42ee77b3dacf3957cf41e1c14286c84d2d3b96d82538f191ea593`

Original lines 1–18:

```text
    1 #!/usr/bin/env python3
    2 """Build the centered preference tensors for the NBPO finite-pool solver.
    3 
    4 Aggregates the judged rows of ``judge_pairwise_matrix.py`` into two float64
    5 tensors (saved as a versioned artifact directory):
    6 
    7 - ``tensor_policy.npz``: ``A_policy[k, x, i, j] = P_k(y_i > z_j | x) - 1/2``
    8   (Eq. (2) ``eq:centered``) for current-policy learners ``i`` vs reference
    9   comparators ``j``;
   10 - ``tensor_ref.npz``: the reference-as-learner tensor used for the
   11   disagreement point ``d_k = V_{k,beta_k}(mu)`` (Eq. (10)) -- ``d`` is never
   12   replaced by ``g_k(mu, mu) = 0``.
   13 
   14 Each semantic comparison must have BOTH presentation orders judged valid; the
   15 two are converted to P(learner > comparator) and swap-averaged. A missing or
   16 single-order cell is a hard error naming the cell (never imputed as 0.5),
   17 unless ``--allow-single-order-ablation`` is passed for an explicitly labeled
   18 ablation (recorded in the metadata). Reference self-comparisons are the
```

Original lines 92–151:

```text
   92     """Dense (K, X, I, J) tensor for learner responses vs reference comparators.
   93 
   94     Learner and comparator supports are DIFFERENT response sets, so no
   95     symmetry is assumed or imposed here.
   96     """
   97     K, X, I, J = len(objectives), len(prompt_ids), len(learner_ids), len(comparator_ids)
   98     A = np.full((K, X, I, J), np.nan, dtype=np.float64)
   99     for k, obj in enumerate(objectives):
  100         for x, pid in enumerate(prompt_ids):
  101             for i, lid in enumerate(learner_ids):
  102                 for j, cid in enumerate(comparator_ids):
  103                     val = payoff.get((pid, obj, "policy", lid, cid))
  104                     if val is not None:
  105                         A[k, x, i, j] = val
  106     _report_missing(A, objectives, prompt_ids, learner_ids, comparator_ids, "policy")
  107     return A
  108 
  109 
  110 def fill_reference_tensor(payoff, objectives, prompt_ids, ref_ids):
  111     """Square (K, X, J, J) reference-as-learner tensor with EXACT skew symmetry.
  112 
  113     One response set sits on both sides, so the paper's assumptions
  114     ``A(i, j) = -A(j, i)`` and ``A(i, i) = 0`` (Eqs. (1)-(2)) hold by
  115     construction: each unordered pair ``i < j`` contributes one swap-averaged
  116     ``a_ij`` written as ``A[i, j] = a_ij`` and ``A[j, i] = -a_ij``.
  117 
  118     Legacy verdict files judged both ordered directions independently. For
  119     those, both directions are read, the pre-projection skew residual
  120     ``max |a_ij + a_ji|`` is reported, and the tensor is projected onto the
  121     skew-symmetric subspace ``0.5 (A - A^T)``. Returns ``(A, stats)``.
  122     """
  123     K, X, J = len(objectives), len(prompt_ids), len(ref_ids)
  124     A = np.zeros((K, X, J, J), dtype=np.float64)
  125     found = np.zeros((K, X, J, J), dtype=bool)
  126     residuals = []
  127     projected_cells = 0
  128     for k, obj in enumerate(objectives):
  129         for x, pid in enumerate(prompt_ids):
  130             for i in range(J):
  131                 found[k, x, i, i] = True                      # Eq. (1) identity, A = 0
  132                 for j in range(i + 1, J):
  133                     fwd = payoff.get((pid, obj, "reference", ref_ids[i], ref_ids[j]))
  134                     bwd = payoff.get((pid, obj, "reference", ref_ids[j], ref_ids[i]))
  135                     if fwd is None and bwd is None:
  136                         continue
  137                     if fwd is not None and bwd is not None:   # legacy: both directions judged
  138                         residuals.append(abs(fwd + bwd))
  139                         projected_cells += 1
  140                         a = 0.5 * (fwd - bwd)
  141                     elif fwd is not None:
  142                         a = fwd
  143                     else:
  144                         a = -bwd
  145                     A[k, x, i, j] = a
  146                     A[k, x, j, i] = -a
  147                     found[k, x, i, j] = found[k, x, j, i] = True
  148     if not found.all():
  149         A_nan = A.copy(); A_nan[~found] = np.nan
  150         _report_missing(A_nan, objectives, prompt_ids, ref_ids, ref_ids, "reference")
  151     stats = {
```

Original lines 236–247:

```text
  236         policy, reference, allow_partial=args.allow_partial_prompt_intersection)
  237     if args.max_prompts:
  238         prompt_ids = prompt_ids[:args.max_prompts]
  239     policy_ids = [f"policy:{s}" for s in sorted(policy)]
  240     ref_ids = [f"ref:{s}" for s in sorted(reference)]
  241 
  242     rows = read_jsonl(args.verdicts)
  243     payoff = aggregate_cells(rows, allow_single_order=args.allow_single_order_ablation)
  244     A_policy = fill_policy_tensor(payoff, objectives, prompt_ids, policy_ids, ref_ids)
  245     A_ref, skew_stats = fill_reference_tensor(payoff, objectives, prompt_ids, ref_ids)
  246     validate_centered_preference_tensor(torch.from_numpy(A_policy), "A_policy")
  247     validate_reference_tensor(torch.from_numpy(A_ref), "A_ref", "shared_pool")
```

## E23 — Custom length adjustment is not official LC

Path: `analysis/sub_20260914/code_snapshot_20260917_union/lc_winrate.py`

SHA256: `e37049402ba47c3d3f81b38f581c2fbd232017c234d9de293b12c9aa3433b57a`

Original lines 1–22:

```text
    1 """Length-adjusted win rate: the win rate an arm would have at equal length.
    2 
    3 Table 8 showed the trained-minus-base gap is monotone in response length, which
    4 leaves the win rates themselves uninterpretable: an arm that answers at twice
    5 the length of the released baseline is partly being rewarded for that. The
    6 published length-controlled AlpacaEval metric handles this by regressing the
    7 verdict on the length difference and reading the fit at zero difference. This is
    8 the same idea in its simplest form, stated rather than hidden:
    9 
   10     y_p = a + b * tanh((len_arm_p - len_baseline_p) / sigma) + e_p
   11 
   12 fitted by least squares over prompts, with sigma the standard deviation of the
   13 raw length difference. y_p is the arm's score on prompt p, already averaged over
   14 both presentation orders, so ties enter as 0.5. The intercept a is the win rate
   15 at equal length and b is the length slope in win-rate units; both carry a
   16 whole-prompt bootstrap. Baseline lengths are tokenized here with the policy
   17 tokenizer, since the judged text is what carries the bias.
   18 
   19 This is a linear adjustment on averaged scores, not the official GLM, and it is
   20 still conditioning on a variable the training changed. It answers one question:
   21 how much of the reported win rate survives setting the length difference to zero.
   22 """
```

Original lines 103–116:

```text
  103 def fit(y, x, rng, reps):
  104     """Least squares y = a + b * x, plus a whole-prompt bootstrap."""
  105     def solve(idx):
  106         X = np.column_stack([np.ones(len(idx)), x[idx]])
  107         coef, *_ = np.linalg.lstsq(X, y[idx], rcond=None)
  108         return coef
  109     a, b = solve(np.arange(len(y)))
  110     draws = np.empty((reps, 2))
  111     for k in range(reps):
  112         draws[k] = solve(rng.integers(0, len(y), len(y)))
  113     lo_a, hi_a = np.percentile(draws[:, 0], [2.5, 97.5])
  114     lo_b, hi_b = np.percentile(draws[:, 1], [2.5, 97.5])
  115     return ({"estimate": float(a), "ci95": [float(lo_a), float(hi_a)]},
  116             {"estimate": float(b), "ci95": [float(lo_b), float(hi_b)]})
```

Original lines 138–157:

```text
  138     y = np.array([y_map[p] for p in common])
  139     delta = np.array([a_tok[p] - b_tok[p] for p in common], dtype=float)
  140     sigma = float(delta.std(ddof=1))
  141     x = np.tanh(delta / sigma)
  142     rng = np.random.default_rng(args.seed)
  143     intercept, slope = fit(y, x, rng, args.bootstrap)
  144     report = {
  145         "tag": args.tag, "kind": args.kind, "n_prompts": len(common),
  146         "raw_win_rate": float(y.mean()),
  147         "length_adjusted_win_rate": intercept,
  148         "length_slope_in_win_rate_units": slope,
  149         "length_difference_tokens": {"mean": float(delta.mean()),
  150                                      "median": float(np.median(delta)),
  151                                      "sd": sigma},
  152         "median_tokens": {"arm": float(np.median([a_tok[p] for p in common])),
  153                           "baseline": float(np.median([b_tok[p] for p in common]))},
  154         "model": "y = a + b * tanh((arm tokens - baseline tokens) / sd)",
  155         "bootstrap": {"replicates": args.bootstrap, "unit": "whole prompt"},
  156         "caveat": ("a linear adjustment on order-averaged scores, not the official "
  157                    "length-controlled GLM, and length is a post-treatment variable"),
```

## E24 — Later Arena-Hard evaluator default

Path: `analysis/sub_20260914/code_snapshot_20260917_union/judge_eval_batch_ahv2.py`

SHA256: `3131fe76a233fc017a4a0c08b55352feff600d3e933b75ebd334a74b430cf2a0`

Original lines 249–274:

```text
  249 def main():
  250     import copy
  251     ap = argparse.ArgumentParser(description=__doc__)
  252     ap.add_argument("--arms", nargs="+", required=True)
  253     ap.add_argument("--benches", nargs="+", default=["arenahard", "alpacaeval"])
  254     ap.add_argument("--responses-prefix", default="eval_uw_",
  255                     help="responses tag is prefix + arm + '_' + bench")
  256     ap.add_argument("--base-responses-prefix", default="eval_base_",
  257                     help="the base row reuses existing responses")
  258     ap.add_argument("--out-prefix", default="uw_e72_")
  259     ap.add_argument("--judge", default="/work/models/bases/Qwen2.5-72B-Instruct")
  260     ap.add_argument("--judge-revision",
  261                     default="495f39366efef23836d0cfae4fbe635880d2be31")
  262     ap.add_argument("--tensor-parallel-size", type=int, default=4)
  263     ap.add_argument("--max-tokens", type=int, default=1024)
  264     ap.add_argument("--max-model-len", type=int, default=32768)
  265     ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
  266     ap.add_argument("--max-num-seqs", type=int, default=64)
  267     ap.add_argument("--bootstrap", type=int, default=2000)
  268     ap.add_argument("--seed", type=int, default=20260915)
  269     top = ap.parse_args()
  270 
  271     from transformers import AutoTokenizer
  272     from vllm import LLM
  273 
  274     tok = AutoTokenizer.from_pretrained(top.judge, local_files_only=True)
```

## E25 — Older evaluator default

Path: `analysis/sub_20260914/code_snapshot_20260917_union/judge_eval_pairwise.py`

SHA256: `9cbbfff767d21966bd89939624807d882db5b5a50768252e6a28bad7b065147a`

Original lines 91–116:

```text
   91     out = {norm(d["instruction"]): d["output"] for d in json.load(path.open())}
   92     return out, "instruction", path, "gpt4_1106_preview"
   93 
   94 
   95 def main():
   96     ap = argparse.ArgumentParser(description=__doc__)
   97     ap.add_argument("--kind", required=True, choices=("arenahard", "alpacaeval"))
   98     ap.add_argument("--panel", required=True, help="panel file under panel/")
   99     ap.add_argument("--arm-responses", required=True, help="directory under responses/")
  100     ap.add_argument("--arm-name", required=True)
  101     ap.add_argument("--out-tag", required=True)
  102     ap.add_argument("--judge", default=str(UF / "assets/Qwen3-14B"))
  103     ap.add_argument("--judge-revision", default="40c069824f4251a91eefaf281ebe4c544efd3e18")
  104     ap.add_argument("--tensor-parallel-size", type=int, default=1,
  105                     help="GPUs to shard the judge across. A 70B in bf16 does not fit "
  106                          "on one card; left at 1 the earlier Qwen3-14B runs reproduce.")
  107     ap.add_argument("--max-tokens", type=int, default=256)
  108     ap.add_argument("--max-model-len", type=int, default=16384)
  109     ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
  110     ap.add_argument("--max-num-seqs", type=int, default=64)
  111     ap.add_argument("--bootstrap", type=int, default=2000)
  112     ap.add_argument("--seed", type=int, default=20260915)
  113     args = ap.parse_args()
  114 
  115     from transformers import AutoTokenizer
  116     from vllm import LLM, SamplingParams
```

## E26 — Historical generator temperature/top-p and seed rule

Path: `analysis/sub_20260914/code_snapshot_20260917_union/generate_pros_pool.py`

SHA256: `7792cdc6dee904c34b1d936c06b5c4e5fb14ae52699cfcb08f9e25c40b2484ca`

Original lines 1–18:

```text
    1 # Generated copy of generate_uf4_pool.py -- do not edit by hand.
    2 # source sha256 eadb32374e8bfa1dc4aa55f43b153555e3c452d2da87c686679b3d9610a75c5d
    3 # change: sampling temperature 1.0 -> 0.8 and top_p 1.0 -> 0.9, in BOTH the
    4 #         recorded settings dict and the SamplingParams call, so the record
    5 #         and the draw cannot disagree.
    6 # Reason: PROSPER Appendix C Table 5 samples responses at T=0.8, top_p=0.9,
    7 #         2048 new tokens. The token budget is already a CLI flag.
    8 """Sample the shared UF-4 candidate pool: 8 learners and 8 comparators per prompt.
    9 
   10 Every method and every seed reads this one pool, so it is generated once from the
   11 original base under raw-policy sampling and never regenerated per arm. Each
   12 occurrence keeps its own token event; duplicate text is kept as a separate
   13 occurrence with its multiplicity intact, because the occurrence measure the
   14 solver uses is uniform over draws, not over distinct strings.
   15 
   16 Chunks are hash-verified and resumable, so the 200-prompt profile is literally
   17 the first chunks of the full run: same prompts, same per-candidate seeds, same
   18 settings file. Nothing is generated twice.
```

Original lines 156–180:

```text
  156     plan = [(role, index) for role in ROLES for index in range(POOL)]
  157     seeds = {f"{row['prompt_id']}:{role}:{index}":
  158              int(digest(SEED_NAMESPACE + f"{row['prompt_id']}:{role}:{index}")[:16], 16) % (2**63 - 1)
  159              for row in rows for role, index in plan}
  160     if len(set(seeds.values())) != len(seeds):
  161         raise ValueError("Random stream seed collision")
  162 
  163     out = ROOT / "pools" / args.out_name / f"shard{args.shard}"
  164     out.mkdir(parents=True, exist_ok=True)
  165     settings = {"model": args.model, "model_revision": args.model_revision,
  166                 "splits_dir": str(split_dir),
  167                 "split_files_sha256": {s: file_hash(split_dir / f"{s}.jsonl") for s in args.split_names},
  168                 "roles": list(ROLES), "pool_per_role": POOL,
  169                 "reference_construction": "comparator_self_bank",
  170                 "temperature": 0.8, "top_p": 0.9, "top_k": -1, "min_p": 0.0,
  171                 "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
  172                 "max_tokens": args.max_tokens, "max_prompt_tokens": args.max_prompt_tokens,
  173                 "max_model_len": args.max_model_len, "stop_token_ids": terminal,
  174                 "generation_config": "vllm", "n": 1,
  175                 "occurrence_mass": 1.0 / POOL, "duplicate_occurrences_retained": True,
  176                 "seed_rule": f"SHA256('{SEED_NAMESPACE}' + candidate_id)[:16] mod (2**63-1)",
  177                 "shard": args.shard, "shards": args.shards,
  178                 "prompts_per_chunk": args.prompts_per_chunk,
  179                 "chat_template_sha256": digest(tokenizer.chat_template or ""),
  180                 "source_sha256": file_hash(__file__)}
```

Original lines 216–232:

```text
  216                       generation_config="vllm", seed=20260910, trust_remote_code=False)
  217         requests, params, mapping = [], [], []
  218         for prompt in prompts:
  219             for role, index in plan:
  220                 candidate_id = f"{prompt['prompt_id']}:{role}:{index}"
  221                 seed = seeds[candidate_id]
  222                 requests.append({"prompt_token_ids": prompt["prompt_token_ids"]})
  223                 params.append(SamplingParams(n=1, temperature=0.8, top_p=0.9, top_k=-1, min_p=0.0,
  224                                              repetition_penalty=1.0, presence_penalty=0.0,
  225                                              frequency_penalty=0.0, max_tokens=args.max_tokens,
  226                                              seed=seed, stop_token_ids=terminal, ignore_eos=False))
  227                 mapping.append((prompt, role, index, seed))
  228         before = time.monotonic()
  229         generated = llm.generate(requests, params, use_tqdm=False)
  230         if len(generated) != len(requests) or any(len(r.outputs) != 1 for r in generated):
  231             raise ValueError("Generation lost requests or produced the wrong number of completions")
  232         events = [make_event(*item, result.outputs[0], tokenizer, terminal, args.max_model_len)
```
