# 97e03c0 source evidence
All excerpts are original uploaded ZIP bytes. Numbers below are original file line numbers.

## E01 — README: current prompt-wise path versus Global Nash
Path: `README.md`
SHA256: `3022c115e46e6061a3cbf3d2d0012bb4b9735e526927b5ed321fd546ee73d2fd`
```text
   25 | as per-pair targets, so the additions are a dual solver that produces those weights from the
   26 | objective-wise game values and a pair builder that applies them. MNPO's own baselines remain
   27 | runnable and are used as controls.
   28 | 
   29 | ## Which pipeline the paper's NBPO row comes from
   30 | 
   31 | Start here, because the answer is not the pipeline this section used to lead
   32 | with. The manuscript's `NBPO` is **prompt-wise**: the dual multipliers are fitted
   33 | separately at every prompt. The campaign code that produces it lives in
   34 | `analysis/sub_20260914/code_snapshot_20260917_union/`, whose
   35 | `solve_pros4_pw_nbpo_*.py` are generated from the shared-weight
   36 | `solve_pros4_targets_*.py` by `build_panel_solvers.py`, and whose scorer
   37 | `union_score_panel.py` builds the game tensors. That snapshot is the
   38 | authoritative current path, and its README documents the tensor roles and the
   39 | serialization contract.
   40 | 
   41 | `scripts/nbpo/` below is the **shared-weight (Global Nash) R=1 path**: one
   42 | multiplier vector for all prompts. The manuscript labels that arm a control, not
   43 | NBPO. It remains a faithful finite-pool realization of the population algorithm
   44 | and it is the right entry point for reading the math, but a result produced by it
   45 | is a Global Nash result and must be reported as one.
   46 | 
   47 | A score directory written before 2026-09-18 carries no `tensor_role_schema` and
   48 | its `A_policy` is the learner triangle rather than the learner-by-reference
   49 | block; the current solvers refuse such a directory rather than solving it into a
   50 | different finite game, and `union_score_uw.py`, which wrote them, refuses to run.
   51 | 
   52 | ## Two NBPO pipelines: finite-pool realization vs legacy fixed-reference
   53 | 
   54 | The repository contains two distinct implementations. Do not conflate them.
   55 | 
   56 | **`scripts/nbpo/` — finite-pool realization of Algorithm 1 with SHARED weights
   57 | (Global Nash).** The
   58 | manuscript's construction: adaptive KL-regularized opponents
   59 | ν\*<sub>k,π</sub> ∝ μ·exp(−r/β<sub>k</sub>) (Eq. 7), soft-min game values
   60 | V<sub>k,β</sub> (Eq. 8), a measured disagreement point
   61 | d<sub>k</sub> = V<sub>k,β</sub>(μ) (Eq. 10, never assumed zero; the
   62 | reference-vs-reference tensor is exactly skew-symmetric with a zero diagonal by
   63 | construction), projected dual descent on the **raw** multipliers
   64 | λ ← Π<sub>Λ</sub>[λ − γ(ŝ − 1/λ)] (Eq. 27), pairwise regression to
   65 | (h<sub>t</sub> − η Σ<sub>k</sub> λ<sub>k</sub>Z<sub>k</sub>)² with sequence-sum
   66 | log-probabilities (Eq. 26, `loss_type: nbpo`), and the held-out
   67 | stage-acceptance gate of Algorithm 1. **The dual solver lives in
   68 | `mnpo_scripts/nbpo_solver.py`** (CLI: `scripts/nbpo/solve_nbpo_dual.py`). Its
   69 | multipliers are shared across prompts, which is what makes it the Global Nash
   70 | control rather than the manuscript's prompt-wise NBPO.
   71 | 
   72 | Two disclosed approximations, stated plainly:
   73 | 

```

## E02 — Canonical builder, flipping, and verified artifact loader
Path: `scripts/nbpo/build_nbpo_pairs.py`
SHA256: `e931faed478feae7c9ddee3501beb6d0fdbcf8dffe2b99f01e8e490013bb1549`
```text
   56 | def _array_sha256(arr) -> str:
   57 |     """sha256 of an array's exact float64 bytes -- matches the solver's hash."""
   58 |     import hashlib
   59 | 
   60 |     return hashlib.sha256(
   61 |         np.ascontiguousarray(np.asarray(arr), dtype=np.float64).tobytes()).hexdigest()
   62 | 
   63 | 
   64 | def flip_pair_row(row: dict) -> dict:
   65 |     """Swap the pair orientation, flipping every Z_k and the aggregate target."""
   66 |     flipped = dict(row)
   67 |     flipped["chosen"], flipped["rejected"] = row["rejected"], row["chosen"]
   68 |     flipped["chosen_response_id"], flipped["rejected_response_id"] = (
   69 |         row["rejected_response_id"], row["chosen_response_id"])
   70 |     flipped["nbpo_z"] = {k: -v for k, v in row["nbpo_z"].items()}
   71 |     if "nbpo_weighted_z" in row:
   72 |         flipped["nbpo_weighted_z"] = -row["nbpo_weighted_z"]
   73 |     if "nbpo_logratio_target" in row:
   74 |         flipped["nbpo_logratio_target"] = -row["nbpo_logratio_target"]
   75 |         flipped["nbpo_weight_a"], flipped["nbpo_weight_b"] = row["nbpo_weight_b"], row["nbpo_weight_a"]
   76 |         if "nbpo_center_a" in row and "nbpo_center_b" in row:
   77 |             flipped["nbpo_center_a"], flipped["nbpo_center_b"] = row["nbpo_center_b"], row["nbpo_center_a"]
   78 |     for suffix in ("text_sha256", "token_sha256", "candidate_index", "input_ids", "attention_mask", "labels"):
   79 |         a, b = "chosen_" + suffix, "rejected_" + suffix
   80 |         if a in row and b in row:
   81 |             flipped[a], flipped[b] = row[b], row[a]
   82 |     return flipped
   83 | 
   84 | 
   85 | def resolve_opponent_betas(solution: dict):
   86 |     """The opponent temperatures to stamp on every pair row, or ``None``.
   87 | 

  110 | def build_rows(prompt_ids, objectives, A_policy, nu, lam, betas, policy, ref_seed_of,
  111 |                rng, target_mode, meta_ids, provenance, canonical_data=None):
  112 |     """One row per (prompt, unordered learner pair); deterministic given the RNG."""
  113 |     K = len(objectives)
  114 |     I = A_policy.shape[2]
  115 |     policy_ids = meta_ids["policy_learner_ids"]
  116 |     comparator_ids = meta_ids["comparator_ids"]
  117 |     if target_mode == "canonical_logratio":
  118 |         if canonical_data is None:
  119 |             raise ValueError("canonical_logratio requires the solved p_star/pi_t/g artifact")
  120 |         g = np.asarray(canonical_data["g"], dtype=np.float64)
  121 |         p_star = np.asarray(canonical_data["p_star"], dtype=np.float64)
  122 |         p_t = np.asarray(canonical_data["p_t"], dtype=np.float64) if "p_t" in canonical_data else np.full_like(p_star, 1.0 / I)
  123 |         if g.shape != (len(prompt_ids), I) or p_star.shape != g.shape:
  124 |             raise ValueError("canonical candidate shape does not match the response pool")
  125 |     rows = []
  126 |     for x, pid in enumerate(prompt_ids):
  127 |         seed0 = policy_ids[0].split(":", 1)[1]
  128 |         prompt_text = str(policy[seed0][pid]["prompt"])
  129 |         candidate_events = None
  130 |         if target_mode == "canonical_logratio":
  131 |             candidates = [policy[rid.split(":", 1)[1]][pid] for rid in policy_ids]
  132 |             has_tokens = [any(key in c for key in ("input_ids", "response_token_ids", "token_ids"))
  133 |                           for c in candidates]
  134 |             if any(has_tokens):
  135 |                 from mnpo_scripts.pair_tokenization import candidate_event_tokens, IMMUTABLE_TOKENIZATION_SCHEMA
  136 |                 prompt_tokens = candidates[0].get("prompt_token_ids")
  137 |                 if not all(has_tokens) or prompt_tokens is None:
  138 |                     raise ValueError("canonical pool has incomplete immutable candidate events")
  139 |                 if any(c.get("prompt_token_ids") != prompt_tokens for c in candidates):
  140 |                     raise ValueError("canonical candidates disagree on their conditioning prompt tokens")
  141 |                 candidate_events = [candidate_event_tokens(prompt_tokens, c) for c in candidates]
  142 |         for i1, i2 in itertools.combinations(range(I), 2):
  143 |             z, opp = {}, {}
  144 |             for k, obj in enumerate(objectives):
  145 |                 if target_mode == "canonical_logratio":
  146 |                     continue
  147 |                 # Eq. (26): draw (y, y') first, THEN one z_k ~ nu*_k for this pair and
  148 |                 # objective. The same z_k serves both y and y' of the row; other rows
  149 |                 # of the same prompt and other objectives draw independently.
  150 |                 if target_mode == "canonical":
  151 |                     # no draw at all: the full expectation over the opponent
  152 |                     z[obj] = float(
  153 |                         (nu[k, x] * (A_policy[k, x, i1, :] - A_policy[k, x, i2, :])).sum())
  154 |                     opp[obj] = "expectation:nu_star"
  155 |                     continue
  156 |                 j = int(rng.choice(len(comparator_ids), p=nu[k, x]))
  157 |                 p1 = float(A_policy[k, x, i1, j]) + 0.5
  158 |                 p2 = float(A_policy[k, x, i2, j]) + 0.5
  159 |                 if target_mode == "sampled":
  160 |                     b1 = float(rng.random() < p1)
  161 |                     b2 = float(rng.random() < p2)
  162 |                     z[obj] = b1 - b2
  163 |                 else:  # rao_blackwell
  164 |                     z[obj] = p1 - p2
  165 |                 opp[obj] = comparator_ids[j]
  166 |             id1, id2 = policy_ids[i1], policy_ids[i2]
  167 |             row = {
  168 |                 "prompt_id": pid,
  169 |                 "prompt": prompt_text,
  170 |                 "chosen": str(policy[id1.split(":", 1)[1]][pid]["generated_text"]),
  171 |                 "rejected": str(policy[id2.split(":", 1)[1]][pid]["generated_text"]),
  172 |                 "chosen_response_id": id1,
  173 |                 "rejected_response_id": id2,
  174 |                 # Response IDS are not response TEXT: these pin the exact strings
  175 |                 # this row was built from, so a later pool swap is detectable.
  176 |                 "chosen_text_sha256": sha256_text(
  177 |                     str(policy[id1.split(":", 1)[1]][pid]["generated_text"])),
  178 |                 "rejected_text_sha256": sha256_text(
  179 |                     str(policy[id2.split(":", 1)[1]][pid]["generated_text"])),
  180 |                 "nbpo_z": z,
  181 |                 "nbpo_weighted_z": (None if target_mode == "canonical_logratio" else
  182 |                                     float(sum(lam[k] * z[obj] for k, obj in enumerate(objectives)))),
  183 |                 "lambda_raw": {obj: float(lam[k]) for k, obj in enumerate(objectives)},
  184 |                 "opponent_response_id": opp,
  185 |                 "opponent_beta": (None if betas is None else
  186 |                                   {obj: float(betas[k]) for k, obj in enumerate(objectives)}),
  187 |                 "opponent_sampling_scope": "pair_objective",
  188 |                 "target_mode": target_mode,
  189 |                 **provenance,
  190 |             }
  191 |             if target_mode == "canonical_logratio":
  192 |                 row.update({
  193 |                     "nbpo_logratio_target": float(g[x, i1] - g[x, i2]),
  194 |                     "nbpo_weight_a": float(p_star[x, i1]),
  195 |                     "nbpo_weight_b": float(p_star[x, i2]),
  196 |                     "nbpo_center_a": float(p_t[x, i1]),
  197 |                     "nbpo_center_b": float(p_t[x, i2]),
  198 |                     "nbpo_num_candidates": I,
  199 |                     "chosen_candidate_index": i1, "rejected_candidate_index": i2,
  200 |                     "target_column": "nbpo_logratio_target",
  201 |                     "target_units": "final_logratio_change", "eta_already_included": True,
  202 |                     "opponent_sampling_scope": "none_canonical_artifact",
  203 |                 })
  204 |                 # Kept only as a compatibility diagnostic; the named canonical
  205 |                 # target column is the sole training target for this mode.
  206 |                 row.pop("nbpo_weighted_z")
  207 |                 if candidate_events is not None:
  208 |                     row.update({"prompt_input_ids": prompt_tokens,
  209 |                                 "prompt_attention_mask": [1] * len(prompt_tokens),
  210 |                                 "candidate_token_schema": IMMUTABLE_TOKENIZATION_SCHEMA})
  211 |                     for prefix, index in (("chosen", i1), ("rejected", i2)):
  212 |                         row.update({prefix + "_" + key: value
  213 |                                     for key, value in candidate_events[index].items()})
  214 |             rows.append(row)
  215 |     return rows
  216 | 
  217 | 
  218 | def load_canonical_artifact(solver_dir: Path, solution: dict, *, expected_prompt_ids=None,
  219 |                             expected_representation=None, expected_aggregation=None) -> dict:
  220 |     """Load only a hash-bound, independently certified final-logratio teacher."""
  221 |     required = {"target_mode": "canonical_logratio", "target_column": "nbpo_logratio_target",
  222 |                 "target_units": "final_logratio_change", "eta_already_included": True}
  223 |     for name, expected in required.items():
  224 |         if solution.get(name) != expected:
  225 |             raise ValueError(f"canonical artifact schema mismatch: {name}")
  226 |     for name, expected in (("prompt_ids", expected_prompt_ids),
  227 |                            ("representation", expected_representation),
  228 |                            ("aggregation", expected_aggregation)):
  229 |         if expected is not None and solution.get(name) != expected:
  230 |             raise ValueError(f"canonical artifact binding mismatch: {name}")
  231 |     certificate = solution.get("certificate") or {}
  232 |     if (solution.get("config") or {}).get("inner_solver") != "exact":
  233 |         raise ValueError("canonical artifact requires the direct inner solver")
  234 |     if certificate.get("certified") is not True:
  235 |         raise ValueError("canonical artifact has no independent solver certificate")
  236 |     arrays = {}
  237 |     for fname, key, dest in (("pi_star.npz", "pi", "p_star"), ("pi_t.npz", "pi", "p_t"),
  238 |                               ("target_log_ratio.npz", "h", "g")):
  239 |         path = Path(solver_dir) / fname
  240 |         if sha256_file(path) != (solution.get("artifact_hashes") or {}).get(fname):
  241 |             raise ValueError(f"canonical artifact hash mismatch: {fname}")
  242 |         arrays[dest] = np.asarray(np.load(path)[key], dtype=np.float64)
  243 |     p, pt, g = arrays["p_star"], arrays["p_t"], arrays["g"]
  244 |     if p.shape != pt.shape or p.shape != g.shape or p.ndim != 2:
  245 |         raise ValueError("canonical artifact shapes disagree")
  246 |     if not all(np.isfinite(v).all() for v in (p, pt, g)) or np.any(p <= 0) or np.any(pt <= 0):
  247 |         raise ValueError("canonical artifact requires finite values and positive support")
  248 |     if max(np.abs(p.sum(-1)-1).max(), np.abs(pt.sum(-1)-1).max()) > 1e-10:
  249 |         raise ValueError("canonical artifact probability normalization failed")
  250 |     if np.max(np.abs(g - (np.log(p)-np.log(pt)))) >= 1e-9:
  251 |         raise ValueError("canonical serialization error >= 1e-9")
  252 |     return arrays
  253 | 
  254 | 
  255 | def verify_solver_input_chain(tensor_dir: Path, solver_dir: Path, solution: dict,
  256 |                               reproduction_mode_required: bool = True) -> dict:
  257 |     """Prove the tensors on disk ARE the solver's inputs, and the opponents its outputs.
  258 | 
  259 |     Verifying only the ``nu`` hashes left the biggest hole open: nothing showed
  260 |     that ``tensor_policy.npz`` / ``tensor_ref.npz`` / ``meta.json`` currently in

```

## E03 — Generic solver artifact writer
Path: `scripts/nbpo/solve_nbpo_dual.py`
SHA256: `d9a2b91837dce03f297ad1ce9e6faf86fa66595e9541347a61f479f10cb936d4`
```text
  200 | def write_generic_solution_artifact(out_dir: Path, res, tensor_meta: dict, hashes: dict,
  201 |                                     tensor_dir: Path, stage: int,
  202 |                                     lambda_warm_started: bool, extra: dict) -> dict:
  203 |     """Persist a FinitePoolSolution as the generic finite-pool target artifact.
  204 | 
  205 |     Carries everything a downstream consumer needs to rebuild the training target
  206 |     without re-solving: the proximal centre, the solved policy, the raw weights,
  207 |     the response-level objective scores, the target log-ratios, every residual,
  208 |     and the identity check that ties them together.
  209 |     """
  210 |     from mnpo_scripts.nbpo_generic import validate_finite_pool_solution
  211 |     certificate = validate_finite_pool_solution(res)
  212 |     out_dir.mkdir(parents=True, exist_ok=True)
  213 |     np.savez_compressed(out_dir / "nu_update.npz", nu=res.nu_update.numpy())
  214 |     np.savez_compressed(out_dir / "nu_final_policy.npz", nu=res.nu_final_policy.numpy())
  215 |     np.savez_compressed(out_dir / "pi_star.npz", pi=res.pi.numpy())
  216 |     np.savez_compressed(out_dir / "pi_t.npz", pi=res.pi_t.numpy())
  217 |     np.savez_compressed(out_dir / "q_update.npz", q=res.q_update.numpy())
  218 |     np.savez_compressed(out_dir / "target_log_ratio.npz", h=res.target_log_ratio.numpy())
  219 |     np.savez_compressed(out_dir / "update_source_pi.npz", pi=res.update_source_pi.numpy())
  220 |     identity = res.target_log_ratio_check()
  221 |     solution = {
  222 |         **implementation_contract(dual_iterations=res.config.get("M"),
  223 |                                   fixed_point_steps=res.config.get("R"),
  224 |                                   inner_solver=res.config.get("inner_solver"),
  225 |                                   dual_solver=res.config.get("dual_solver")),
  226 |         "solver_path": "generic_solve_finite_pool",
  227 |         "representation": res.representation,
  228 |         "aggregation": res.aggregation,
  229 |         "stage": int(stage),
  230 |         "objectives": tensor_meta.get("objectives"),
  231 |         "prompt_ids": tensor_meta.get("prompt_ids"),
  232 |         # RAW weights. For nash these are lambda; for the controls they are the
  233 |         # rule's weight vector at the matched L1 norm. Never normalized for training.
  234 |         "lambda_raw": [float(v) for v in res.weights],
  235 |         "aggregation_weights_raw": [float(v) for v in res.weights],
  236 |         "V": [float(v) for v in res.V],
  237 |         "d": [float(v) for v in res.d],
  238 |         "surplus": [float(v) for v in res.surplus],
  239 |         "min_surplus": float(res.surplus.min()),
  240 |         "kkt_residual": res.kkt_residual,
  241 |         "inverse_surplus_residual": res.kkt_residual,
  242 |         "projected_kkt_residual": res.projected_kkt_residual,
  243 |         "gamma_ref": res.gamma_ref,
  244 |         "lambda_at_lower_bound": res.lambda_at_lower_bound,
  245 |         "lambda_at_upper_bound": res.lambda_at_upper_bound,
  246 |         "control_residual": res.control_residual,
  247 |         "fixed_point_residual": res.fixed_point_residual,
  248 |         "extra_map_residual": res.extra_map_residual,
  249 |         "proximal_kl": res.proximal_kl,
  250 |         "opponent_entropy": [float(v) for v in res.opponent_entropy],
  251 |         "opponent_ess": [float(v) for v in res.opponent_ess],
  252 |         "opponent_diagnostics_from": "nu_final_policy",
  253 |         # The identity the pair builder relies on:
  254 |         #   [log pi*(y) - log pi_t(y)] - [log pi*(y') - log pi_t(y')]
  255 |         #     == eta * sum_k w_k (q_k(y) - q_k(y'))
  256 |         # Verified numerically here rather than assumed downstream.
  257 |         "target_log_ratio_identity_residual": identity,
  258 |         "target_log_ratio_identity_holds": bool(identity < 1e-9),
  259 |         "canonical_schema_version": 1,
  260 |         "target_mode": "canonical_logratio",
  261 |         "target_column": "nbpo_logratio_target",
  262 |         "target_units": "final_logratio_change",
  263 |         "eta_already_included": True,
  264 |         "certificate": certificate,
  265 |         "canonical_serialization_error": certificate["canonical_serialization_error"],
  266 |         "independent_stationarity_inf": certificate["independent_stationarity_inf"],
  267 |         "eta": res.eta,
  268 |         "eta_applications_in_solver": 1,
  269 |         "ks": res.ks,
  270 |         "artifact_hashes": {n: sha256_file(out_dir / n) for n in
  271 |                             ("nu_update.npz", "nu_final_policy.npz", "pi_star.npz",
  272 |                              "pi_t.npz", "q_update.npz", "target_log_ratio.npz",
  273 |                              "update_source_pi.npz")},
  274 |         "opponent_artifacts": {
  275 |             "nu_update.npz": {
  276 |                 "artifact_kind": "regularized_opponent",
  277 |                 "source_policy": res.update_source_kind,
  278 |                 "source_policy_hash": _array_hash(res.update_source_pi),
  279 |                 "source_policy_artifact": "update_source_pi.npz",
  280 |                 "source_fixed_point_iteration": int(res.update_source_iteration),
  281 |                 "used_for": "eq26_target",
  282 |             },
  283 |             "nu_final_policy.npz": {
  284 |                 "artifact_kind": "regularized_opponent",
  285 |                 "source_policy": "final_policy",
  286 |                 "source_policy_hash": _array_hash(res.pi),
  287 |                 "source_policy_artifact": "pi_star.npz",
  288 |                 "used_for": "diagnostics",
  289 |             },
  290 |         },
  291 |         "config": res.config,
  292 |         "lambda_warm_started": bool(lambda_warm_started),
  293 |         "input_hashes": hashes,
  294 |         "tensor_dir": str(tensor_dir),
  295 |         "history": res.history,
  296 |         **extra,
  297 |     }
  298 |     # The historical weighted-Q identity is a diagnostic. Independent
  299 |     # final-policy stationarity and exact serialization have distinct units
  300 |     # and tolerances, checked centrally before any artifact is written.
  301 |     write_json(out_dir / "solution.json", solution)
  302 |     return solution
  303 | 
  304 | 
  305 | def _print_summary(solution: dict) -> None:
  306 |     keys = ("solver_path", "representation", "aggregation", "lambda_raw", "surplus",

```

## E04 — Correct tensor schema and scorer output
Path: `analysis/sub_20260914/code_snapshot_20260917_union/union_score_panel.py`
SHA256: `e3bd9c1c2486715f0e1e4a07efe15419fc499507f5d2c633039cde6b2d17da83`
```text
   71 | POOL = 8
   72 | K = None                    # set from --objectives; no default objective count
   73 | ITEMS = None
   74 | # Bumped when the MEANING of a stored tensor changes. Shards without it were
   75 | # written by the pre-A01 scorer, whose A_policy is the learner triangle and whose
   76 | # A_ref is the cross block; those arrays hash and shape exactly like correct ones,
   77 | # so the version is the only thing that can tell them apart and the solver loader
   78 | # refuses a shard that does not carry it.
   79 | TENSOR_ROLE_SCHEMA = "lr_rr_v2"
   80 | 
   81 | 
   82 | def file_hash(p) -> str:
   83 |     h = hashlib.sha256()
   84 |     with open(p, "rb") as s:
   85 |         for c in iter(lambda: s.read(1 << 20), b""):
   86 |             h.update(c)
   87 |     return h.hexdigest()

  151 |         got = obs.get((pid, rubric, role_i, i, role_j, j))
  152 |         if got is None or 0 not in got or 1 not in got:
  153 |             return None
  154 |         return 0.5 * (got[0] + got[1])
  155 | 
  156 |     tensors, identity_ties, refdis = {}, Counter(), {}
  157 |     dropped = Counter()
  158 |     prompts = sorted({k[0] for k in obs})
  159 |     for pid in prompts:
  160 |         if pid not in pool:
  161 |             dropped["prompt_absent_from_pool"] += 1
  162 |             continue
  163 |         A_LL = np.zeros((K, POOL, POOL))
  164 |         A_LR = np.zeros((K, POOL, POOL))
  165 |         A_RR = np.zeros((K, POOL, POOL))
  166 |         ok = True
  167 |         for k, rubric in enumerate(ITEMS):
  168 |             for i, j in learner_pairs:
  169 |                 p = resolve(pid, rubric, "learner", i, "learner", j)
  170 |                 if p is None:
  171 |                     ok = False; dropped["learner_pair_unresolved"] += 1; break
  172 |                 A_LL[k, i, j] = p - 0.5
  173 |                 A_LL[k, j, i] = 0.5 - p
  174 |             if not ok:
  175 |                 break
  176 |             for i, j in cross_pairs:
  177 |                 p = resolve(pid, rubric, "learner", i, "comparator", j)
  178 |                 if p is None:
  179 |                     ok = False; dropped["cross_pair_unresolved"] += 1; break
  180 |                 if (pool[pid]["learner"].get(i) ==
  181 |                         pool[pid]["comparator"].get(j) is not None):
  182 |                     identity_ties[pid] += 1
  183 |                 A_LR[k, i, j] = p - 0.5
  184 |             if not ok:
  185 |                 break
  186 |             # the reference triangle is now required: d = V_beta(mu) is defined
  187 |             # on it, so a prompt without it has no disagreement point and is
  188 |             # dropped rather than carried with a silently wrong d
  189 |             for i, j in learner_pairs:
  190 |                 p = resolve(pid, rubric, "comparator", i, "comparator", j)
  191 |                 if p is None:
  192 |                     ok = False; dropped["reference_pair_unresolved"] += 1; break
  193 |                 A_RR[k, i, j] = p - 0.5
  194 |                 A_RR[k, j, i] = 0.5 - p
  195 |             if not ok:
  196 |                 break
  197 |         if not ok:
  198 |             continue
  199 |         idx = np.arange(POOL)
  200 |         A_LL[:, idx, idx] = 0.0
  201 |         A_RR[:, idx, idx] = 0.0
  202 |         for name, M in (("learner", A_LL), ("reference", A_RR)):
  203 |             if np.abs(M + np.swapaxes(M, -1, -2)).max() > 1e-12:
  204 |                 raise SystemExit("%s payoff not antisymmetric for %s" % (name, pid))
  205 |         if max(np.abs(A_LL).max(), np.abs(A_LR).max(),
  206 |                np.abs(A_RR).max()) > 0.5 + 1e-12:
  207 |             raise SystemExit("payoff outside [-0.5, 0.5] for %s" % pid)
  208 |         # the same scalar the old file reported, so the two rounds stay comparable
  209 |         iu = np.triu_indices(POOL, 1)
  210 |         refdis[pid] = float(np.mean(np.abs(A_RR[:, iu[0], iu[1]])))
  211 |         tensors[pid] = (A_LL, A_LR, A_RR)
  212 | 
  213 |     pids = sorted(tensors)
  214 |     if not pids:
  215 |         raise SystemExit("no prompt survived scoring")
  216 |     out = UF / "scores" / args.out

  236 |         A_LR = np.stack([tensors[p][1] for p in chunk], axis=1)
  237 |         A_RR = np.stack([tensors[p][2] for p in chunk], axis=1)
  238 |         d = out / ("shard%d" % s)
  239 |         d.mkdir(parents=True, exist_ok=True)
  240 |         path = d / "chunk0000.npz"
  241 |         # A_policy and A_ref are the names the solver reads. They are aliases of
  242 |         # A_LR and A_RR, written explicitly so an existing reader gets the
  243 |         # paper's game without being changed, while A_LL travels under its own
  244 |         # name for the pair-label consumers.
  245 |         np.savez(path, prompt_ids=np.array(chunk),
  246 |                  A_policy=A_LR, A_ref=A_RR,
  247 |                  A_LL=A_LL, A_LR=A_LR, A_RR=A_RR)
  248 |         (d / "chunk0000.manifest.json").write_text(json.dumps(
  249 |             {"prompts": len(chunk), "sha256": file_hash(path),
  250 |              "shapes": {"A_policy": list(A_LR.shape), "A_ref": list(A_RR.shape),
  251 |                         "A_LL": list(A_LL.shape), "A_LR": list(A_LR.shape),
  252 |                         "A_RR": list(A_RR.shape)},
  253 |              "tensor_role_schema": TENSOR_ROLE_SCHEMA,
  254 |              "roles": {"A_policy": "A_LR (learner vs reference)",
  255 |                        "A_ref": "A_RR (reference vs reference)"},
  256 |              "bank_ids": {"A_policy": ["learner", "comparator"],
  257 |                           "A_ref": ["comparator", "comparator"],
  258 |                           "A_LL": ["learner", "learner"]}},
  259 |             indent=1) + "\n")
  260 |         (d / ("complete_shard%d.json" % s)).write_text(json.dumps(
  261 |             {"shard": s, "prompts": len(chunk), "gpm_teacher": teacher,
  262 |              "bt_teacher": ("absent: the union contract uses direct order-balanced PSC "
  263 |                             "probabilities for these rows and forbids a scalar BT "
  264 |                             "projection, so none was fitted and none is written"),
  265 |              "tensor_role_schema": TENSOR_ROLE_SCHEMA,
  266 |              "reference_construction": ("independent reference bank: eight reference "
  267 |                                         "occurrences per prompt, so A_policy is the "
  268 |                                         "learner-by-reference cross block and A_ref is "
  269 |                                         "the reference triangle, not a copy of it")},
  270 |             indent=1) + "\n")
  271 |         written.append({"shard": s, "prompts": len(chunk),
  272 |                         "sha256": file_hash(path),
  273 |                         "shapes": {"A_policy": list(A_LR.shape),
  274 |                                    "A_ref": list(A_RR.shape),
  275 |                                    "A_LL": list(A_LL.shape)}})
  276 | 
  277 |     verdicts = sum(status.values())
  278 |     report = {
  279 |         "out": args.out, "judgment_tag": args.tag, "pool": args.pool,
  280 |         "construction": {
  281 |             "A_policy": ("A_LR: the 8x8 learner-by-reference cross block; NOT "
  282 |                          "antisymmetric and its diagonal need not be zero"),
  283 |             "A_ref": ("A_RR: the reference triangle, antisymmetric with zero diagonal; "
  284 |                       "this is the tensor d = V_beta(mu) is defined on"),
  285 |             "A_LL": ("the learner triangle, antisymmetric with zero diagonal; written "
  286 |                      "for single-objective pair labels, not fed to the NBPO game"),
  287 |             "corrected": ("an earlier version of this scorer set A_policy to the learner "
  288 |                           "triangle and A_ref to the cross block, which flips the sign of "

```

## E05 — Float64 serializer, split digest and schema check
Path: `analysis/sub_20260914/code_snapshot_20260917_union/solve_pros4_targets_uw3.py`
SHA256: `ec8226dd9b8202d19e36f58b6ba521f9e3a669650a7566f482d323f61db91a82`
```text
   87 | 
   88 | 
   89 | # Kept for readers of older artifacts: rows written before the A05 fix carry
   90 | # canonical_target_quantized_decimals = 10 and their masses were rounded to that
   91 | # many fixed decimals. New rows are written at exact float64 precision and
   92 | # record canonical_target_serialization instead.
   93 | CANONICAL_DECIMALS_LEGACY = 10
   94 | # The tensor-role schema this solver's game assumes: A_policy is the
   95 | # learner-by-reference block and A_ref the reference triangle.
   96 | REQUIRED_TENSOR_ROLE_SCHEMA = "lr_rr_v2"
   97 | # The panel this module was generated for. build_panel_solvers.py rewrites it, so a
   98 | # US/UT/UW run records its own panel instead of the UF-4 label of the source file.
   99 | PANEL_LABEL = "UW3"
  100 | 
  101 | 
  102 | def split_pool_digest(pool, prompt_ids, roles=("learner", "comparator")):
  103 |     """A content digest of the pool rows this split actually uses.
  104 | 
  105 |     The fields it hashes are the immutable identity of each candidate: the
  106 |     prompt, the role, the occurrence index, the candidate id and the sha256 of
  107 |     the response text. Mutating one byte of one response changes its
  108 |     response_sha256 and therefore this digest, which is the property the audit
  109 |     asks for and the property the previous value did not have -- that one hashed
  110 |     `sorted(outputs)`, i.e. the split NAMES, so train and dev came out equal to
  111 |     each other and equal across panels no matter what the pool contained.
  112 |     """
  113 |     import hashlib
  114 |     h = hashlib.sha256()
  115 |     for pid in sorted(prompt_ids):
  116 |         entry = pool.get(pid)
  117 |         if entry is None:
  118 |             raise ValueError("split prompt %s is absent from the pool" % pid)
  119 |         for role in roles:
  120 |             for index in sorted(entry.get(role, {})):
  121 |                 row = entry[role][index]
  122 |                 identity = (str(pid), role, str(index),
  123 |                             str(row.get("candidate_id", "")),
  124 |                             str(row.get("response_sha256")
  125 |                                 or hashlib.sha256(
  126 |                                     str(row.get("response", "")).encode()).hexdigest()))
  127 |                 h.update(("\x1f".join(identity) + "\x1e").encode())
  128 |     return h.hexdigest()
  129 | 
  130 | 
  131 | def quantize_canonical_row(row):
  132 |     """Serialize the solver masses so they survive the round trip exactly.
  133 | 
  134 |     prepare_nbpo_dataset reads these rows with a standard JSON parser and then
  135 |     checks target == log(w_a/c_a) - log(w_b/c_b) to 1e-9 absolute. An earlier
  136 |     version of this function rounded both masses to ten FIXED DECIMALS and
  137 |     rebuilt the target from the rounded values, which made the identity hold
  138 |     after the parse but changed the certified solution: a per-prompt Nash solve
  139 |     concentrates mass, its smallest masses reach the 1e-12 probability floor,
  140 |     and 1e-12 rounded to ten decimals is exactly 0. The row then carries a
  141 |     non-positive mass, which the canonical validator rejects outright, and any
  142 |     mass between 1e-10 and 1e-12 that did survive was quantized to a value whose
  143 |     log differs from the certified one by far more than the solver's own
  144 |     residual. That is the audit's A05: a target changed after it was certified.
  145 | 
  146 |     Fixed decimals are the wrong instrument for a quantity that spans twelve
  147 |     orders of magnitude. Python's float repr round-trips a float64 exactly and
  148 |     json.dumps emits it, so no rounding is needed at all: the masses are written
  149 |     as they were solved and the target is rebuilt from those same doubles. The
  150 |     identity then holds to the float64 relative error of a logarithm, roughly
  151 |     1e-16, comfortably inside the 1e-9 gate, with the certified solution intact.
  152 | 
  153 |     A mass that is genuinely non-positive is left alone here and refused by the
  154 |     validator, which is the correct outcome: it means the inner solve hit the
  155 |     boundary and the canonical log-ratio does not exist for that pair.
  156 |     """
  157 |     import math
  158 | 
  159 |     for key in ("nbpo_weight_a", "nbpo_weight_b"):
  160 |         if key in row:
  161 |             row[key] = float(row[key])
  162 |     if row.get("target_mode") == "canonical_logratio" and "nbpo_weight_a" in row:
  163 |         wa, wb = float(row["nbpo_weight_a"]), float(row["nbpo_weight_b"])
  164 |         ca = float(row.get("nbpo_center_a", 1.0 / POOL))
  165 |         cb = float(row.get("nbpo_center_b", 1.0 / POOL))
  166 |         if min(wa, wb) > 0:
  167 |             row["nbpo_logratio_target"] = math.log(wa / ca) - math.log(wb / cb)
  168 |             row["canonical_target_quantized_decimals"] = None
  169 |             row["canonical_target_serialization"] = "exact_float64_round_trip"
  170 |     return row
  171 | 
  172 | 
  173 | def load_scores(score_root, shards):
  174 |     """prompt_id -> (A_policy[K,8,8], A_ref[K,8,8], r_bt[K,2,8]), hash-verified.
  175 | 
  176 |     r_bt is the frozen BT head's scalar reward for the eight learner and eight
  177 |     comparator occurrences. It is loaded for every representation but only the
  178 |     bt_reward representation reads it, so nothing else changes.
  179 |     """
  180 |     scores, manifests = {}, []
  181 |     for shard in range(shards):
  182 |         directory = Path(score_root) / f"shard{shard}"
  183 |         complete = json.loads((directory / f"complete_shard{shard}.json").read_text())
  184 |         # A shard written before the A01 tensor-role fix stores the learner
  185 |         # triangle as A_policy and the cross block as A_ref. Those arrays have
  186 |         # the right dtype, the right shape and a matching sha256, so nothing
  187 |         # downstream can notice: the hash certifies that the bytes are the ones
  188 |         # that were written, not that they mean what this solver assumes. The
  189 |         # declared schema is the only discriminator, and a shard without it is
  190 |         # refused rather than solved into a different finite game.
  191 |         schema = complete.get("tensor_role_schema")
  192 |         # read through globals() so the guard still holds when this function is
  193 |         # lifted out of the module into a bare namespace, as audit harnesses do;
  194 |         # a missing constant must not become a missing check
  195 |         required = globals().get("REQUIRED_TENSOR_ROLE_SCHEMA", "lr_rr_v2")
  196 |         if schema != required:
  197 |             raise ValueError(
  198 |                 f"{directory} declares tensor_role_schema {schema!r}, and this solver "
  199 |                 f"requires {required!r}. A shard with no schema predates "
  200 |                 "the A01 fix: its A_policy is the learner triangle and its A_ref is the "
  201 |                 "learner-by-reference cross block, which is not the game Section 5.2 defines. "
  202 |                 "Re-score the panel with union_score_panel.py instead of reusing it.")
  203 |         manifests.append({"shard": shard, "gpm_teacher": complete["gpm_teacher"],
  204 |                           "bt_teacher": complete["bt_teacher"],
  205 |                           "tensor_role_schema": schema,
  206 |                           "reference_construction": complete["reference_construction"]})
  207 |         for path in sorted(directory.glob("chunk*.npz")):
  208 |             meta = json.loads(path.with_suffix("").with_suffix(".manifest.json").read_text()) \
  209 |                 if path.with_suffix("").with_suffix(".manifest.json").exists() else \
  210 |                 json.loads((directory / (path.stem + ".manifest.json")).read_text())
  211 |             if file_hash(path) != meta["sha256"]:
  212 |                 raise ValueError(f"Score chunk hash mismatch: {path}")
  213 |             arrays = np.load(path, allow_pickle=True)
  214 |             pids = [str(p) for p in arrays["prompt_ids"]]
  215 |             A, Aref = arrays["A_policy"], arrays["A_ref"]
  216 |             # r_bt is the scalar BT projection. The union contract uses direct
  217 |             # order-balanced probabilities for these rows and forbids that
  218 |             # projection, so the array may be absent; it is left as None rather
  219 |             # than filled with zeros, and only the bt_reward representation
  220 |             # reads it, where None raises instead of training on a placeholder.
  221 |             Rbt = arrays["r_bt"] if "r_bt" in arrays.files else None
  222 |             if A.shape[0] != len(OBJECTIVES) or A.shape[2:] != (POOL, POOL):
  223 |                 raise ValueError(f"Unexpected score tensor shape in {path}: {A.shape}")
  224 |             if Rbt is not None and (Rbt.shape[0] != len(OBJECTIVES)
  225 |                                     or Rbt.shape[2:] != (2, POOL)):
  226 |                 raise ValueError(f"Unexpected BT reward shape in {path}: {Rbt.shape}")
  227 |             for index, pid in enumerate(pids):
  228 |                 if pid in scores:
  229 |                     raise ValueError(f"Duplicate scored prompt {pid}")
  230 |                 scores[pid] = (A[:, index], Aref[:, index],
  231 |                                None if Rbt is None else Rbt[:, index])
  232 |     identities = {object_hash(m["gpm_teacher"]) for m in manifests}
  233 |     if len(identities) != 1:
  234 |         raise ValueError("Score shards disagree on the frozen GPM teacher")
  235 |     return scores, manifests
  236 | 

```

## E06 — Actual prompt-wise worker and dataset export
Path: `analysis/sub_20260914/code_snapshot_20260917_union/solve_pros4_pw_nbpo_uw3.py`
SHA256: `edbbb366099cb865b3e9b858610d26c148756deb4fb0fb0d7ab6ca1f9cae0920`
```text
   48 | 
   49 | sys.path.insert(0, "/work/uf4_20260910/code")
   50 | import solve_pros4_targets_uw3 as base                      # loaders, hashing, writers
   51 | from mnpo_scripts.nbpo_core import uniform_policy
   52 | from mnpo_scripts.nbpo_generic import solve_finite_pool, validate_finite_pool_solution
   53 | from mnpo_scripts.nbpo_representations import AdaptiveGameRepresentation
   54 | from scripts.nbpo.build_nbpo_pairs import build_rows
   55 | 
   56 | ROOT = Path("/work/uf4_20260910")
   57 | OBJECTIVES = base.OBJECTIVES
   58 | POOL = base.POOL
   59 | _SHARED = {}
   60 | 
   61 | 
   62 | def _init(A, Aref, beta, eta, weight_l1, max_dual_calls, floor):
   63 |     _SHARED.update(A=A, Aref=Aref, beta=beta, eta=eta, weight_l1=weight_l1,
   64 |                    M=max_dual_calls, floor=floor)
   65 |     torch.set_num_threads(1)
   66 | 
   67 | 
   68 | def _solve_one(x):
   69 |     """Solve prompt x alone: its own adversarial weights, its own certificate."""
   70 |     A = _SHARED["A"][:, x:x + 1]
   71 |     Aref = _SHARED["Aref"][:, x:x + 1]
   72 |     rep = AdaptiveGameRepresentation(
   73 |         torch.from_numpy(np.ascontiguousarray(A)), torch.from_numpy(np.ascontiguousarray(Aref)),
   74 |         uniform_policy(1, POOL),
   75 |         torch.full((len(OBJECTIVES),), _SHARED["beta"], dtype=torch.float64),
   76 |         reference_construction="independent_samples")
   77 |     result = solve_finite_pool(
   78 |         rep, "nash", eta=_SHARED["eta"], inner_solver="exact",
   79 |         dual_solver="root", dual_tol=1e-10, M=_SHARED["M"], inner_workers=1,
   80 |         probability_floor=_SHARED["floor"], log_every=0)
   81 |     certificate = validate_finite_pool_solution(result)
   82 |     return (x,
   83 |             result.pi.numpy().astype(np.float64),
   84 |             result.nu_update.numpy().astype(np.float64),
   85 |             result.weights.numpy().astype(np.float64),
   86 |             result.target_log_ratio.numpy().astype(np.float64),
   87 |             float(result.target_log_ratio_check()),
   88 |             bool(certificate.get("certified", False)),
   89 |             # the certificate carries the per-objective surplus vector, not its
   90 |             # minimum; taking .get("min_surplus") silently produced NaN, which
   91 |             # then made "negative worst surplus" count zero prompts by accident
   92 |             float(np.min(np.asarray(certificate["surplus"], dtype=np.float64))))
   93 | 
   94 | 
   95 | def main():
   96 |     ap = argparse.ArgumentParser(description=__doc__)
   97 |     ap.add_argument("--out-name", default="prosper_v1")
   98 |     ap.add_argument("--shards", type=int, default=8,
   99 |                     help="score and pool shard count; 8 once the 400-prompt "
  100 |                          "extension is merged in as shards 4..7")
  101 |     ap.add_argument("--weight-l1", type=float, default=None,
  102 |                     help="matched multiplier norm as a literal. Only absolute_maxmin "
  103 |                          "uses it; nash derives its own dual.")
  104 |     ap.add_argument("--weight-l1-from", type=Path,

  124 |     out.mkdir(parents=True, exist_ok=False)
  125 |     torch.set_num_threads(1)
  126 |     started = time.monotonic()
  127 |     shared_meta, outputs = None, {}
  128 |     split_digests = {}
  129 |     # the base module owns the digest so every arm hashes identically
  130 |     digest_of = getattr(base, "split_pool_digest", None) or split_pool_digest
  131 | 
  132 |     for split, (score_root, pool_root, split_file) in (
  133 |             ("train", (ROOT / "scores/uw3", ROOT / "pools/uw3", "policy_train")),
  134 |             ("dev", (ROOT / "scores/uw3", ROOT / "pools/uw3", "policy_dev"))):
  135 |         split_start = time.monotonic()
  136 |         scores, score_manifests = base.load_scores(str(score_root), args.shards)
  137 |         pool, pool_settings = base.load_pool(str(pool_root), args.shards)
  138 |         with (ROOT / "splits/uw_v3p" / f"{split_file}.jsonl").open() as stream:
  139 |             pids = [json.loads(line)["prompt_id"] for line in stream if line.strip()]
  140 |         if args.limit:
  141 |             pids = pids[:args.limit]
  142 |         missing = [p for p in pids if p not in scores or p not in pool]
  143 |         if missing:
  144 |             raise ValueError(f"{split}: {len(missing)} prompts have no scores or no pool")
  145 | 
  146 |         A = np.stack([scores[p][0] for p in pids], axis=1)
  147 |         Aref = np.stack([scores[p][1] for p in pids], axis=1)
  148 |         if shared_meta is None:
  149 |             from transformers import AutoTokenizer
  150 |             from mnpo_scripts.precompute_provenance import tokenizer_content_hashes
  151 |             tokenizer = AutoTokenizer.from_pretrained(pool_settings["model"], local_files_only=True)
  152 |             pad_fallback = tokenizer.pad_token_id is None
  153 |             if pad_fallback:
  154 |                 tokenizer.pad_token_id = tokenizer.eos_token_id
  155 |             hashes = tokenizer_content_hashes(tokenizer)
  156 |             if hashes["chat_template_hash"] != pool_settings["chat_template_sha256"]:
  157 |                 raise ValueError("Pool chat template does not match the training tokenizer's")
  158 |             shared_meta = {"model_revision": pool_settings["model_revision"], **hashes,
  159 |                            "training_pad_token_fallback_to_eos": pad_fallback,
  160 |                            "pool_settings_sha256": base.object_hash(pool_settings),
  161 |                            "teacher_manifest_sha256": base.object_hash(score_manifests),
  162 |                            "gpm_teacher": score_manifests[0]["gpm_teacher"],
  163 |                            "bt_teacher": score_manifests[0]["bt_teacher"]}
  164 |         meta = {"prompt_ids": pids, "objectives": list(OBJECTIVES),
  165 |                 "reference_construction": "independent_samples",

  180 |                                            args.max_dual_calls, args.probability_floor)) as pool_exec:
  181 |             for res in pool_exec.map(_solve_one, range(len(pids)), chunksize=8):
  182 |                 x, pi_x, nu_x, w_x, g_x, ident, cert, ms = res
  183 |                 pi[x] = pi_x[0]
  184 |                 nu[:, x] = nu_x[:, 0]
  185 |                 weights[:, x] = w_x
  186 |                 g[x] = g_x[0]
  187 |                 identity[x], certified[x], min_surplus[x] = ident, cert, ms
  188 |                 done += 1
  189 |                 if done % 1000 == 0:
  190 |                     print(json.dumps({"split": split, "solved": done, "of": len(pids),
  191 |                                       "seconds": round(time.monotonic() - split_start, 1)}), flush=True)
  192 |         if not certified.all():
  193 |             raise ValueError(f"{split}: {int((~certified).sum())} prompts have no certificate")
  194 |         if float(np.abs(identity).max()) > 1e-9:
  195 |             raise ValueError(f"{split}: target log-ratio identity residual "
  196 |                              f"{float(np.abs(identity).max())}")
  197 | 
  198 |         per_prompt_path = out / f"{split}_per_prompt.npz"
  199 |         np.savez_compressed(per_prompt_path, pi=pi, weights=weights, g=g,
  200 |                             min_surplus=min_surplus, identity_residual=identity,
  201 |                             prompt_ids=np.array(pids, dtype=object))
  202 | 
  203 |         # A hash-bound solution artifact at the path the shared job generator
  204 |         # expects. The global one is not written because per-prompt weights do
  205 |         # not fit its (K,) weight field, but the generator only needs SOME file
  206 |         # whose hash pins the solution, and patching the generator would touch
  207 |         # code the published arms depend on. This file pins the real thing: the
  208 |         # per-prompt npz that every target in this set was built from.
  209 |         solver_dir = out / split / "solver"
  210 |         base.write_json(solver_dir / "solution.json", {
  211 |             "target_mode": "canonical_logratio", "target_column": "nbpo_logratio_target",
  212 |             "target_units": "final_logratio_change", "eta_already_included": True,
  213 |             "representation": "adaptive_game", "aggregation": "prompt_wise_nash",
  214 |             "weights_scope": "per prompt; there is no shared dual",
  215 |             "per_prompt_artifact": str(per_prompt_path),
  216 |             "per_prompt_artifact_sha256": base.file_hash(per_prompt_path),
  217 |             "n_prompts": len(pids), "split": split,
  218 |             "all_certified": bool(certified.all()),
  219 |             "max_identity_residual": float(np.abs(identity).max()),
  220 |             "min_surplus_mean": float(min_surplus.mean()),
  221 |             "min_surplus_negative_prompts": int((min_surplus < 0).sum()),
  222 |             "weight_l1_matched": weight_l1, "beta": args.beta, "eta": args.eta,
  223 |             "solver_source_sha256": base.file_hash(__file__), **shared_meta})
  224 |         solver_hash = base.file_hash(solver_dir / "solution.json")
  225 |         provenance = {"solver_artifact_sha256": solver_hash,
  226 |                       "solver_hash": solver_hash,
  227 |                       "target_artifact_hash": base.file_hash(per_prompt_path),
  228 |                       "representation": "adaptive_game", "aggregation": "prompt_wise_nash",
  229 |                       "split": split, "panel": getattr(base, "PANEL_LABEL", "UF-4"), **shared_meta}
  230 |         betas = np.full(len(OBJECTIVES), args.beta)
  231 | 
  232 |         def pair_rows():
  233 |             for x, pid in enumerate(pids):
  234 |                 learners = {str(i): {pid: {**pool[pid]["learner"][i],
  235 |                                            "generated_text": pool[pid]["learner"][i]["response"]}}
  236 |                             for i in range(POOL)}
  237 |                 rows = build_rows(
  238 |                     [pid], list(OBJECTIVES), A[:, x:x + 1], nu[:, x:x + 1], weights[:, x],
  239 |                     betas, learners, None, np.random.default_rng(42), "canonical_logratio",
  240 |                     meta, provenance,
  241 |                     canonical_data={"g": g[x:x + 1], "p_star": pi[x:x + 1],
  242 |                                     "p_t": np.full((1, POOL), 1.0 / POOL)})
  243 |                 for row in rows:
  244 |                     # round masses to the loader's ten decimals and rebuild the
  245 |                     # target from the rounded values; a per-prompt Nash solve
  246 |                     # concentrates mass and the identity otherwise fails after the parse
  247 |                     row = base.quantize_canonical_row(row)
  248 |                     a, b = row["chosen_candidate_index"], row["rejected_candidate_index"]
  249 |                     row["chosen_response_id"] = pool[pid]["learner"][a]["candidate_id"]
  250 |                     row["rejected_response_id"] = pool[pid]["learner"][b]["candidate_id"]
  251 |                     row["prosper_prompt_weights"] = [float(v) for v in weights[:, x]]
  252 |                     yield row
  253 | 
  254 |         pair_path = out / "pairs" / f"{split}.jsonl"
  255 |         base.write_jsonl(pair_path, pair_rows())
  256 |         outputs[split] = {"n_prompts": len(pids), "n_pairs": len(pids) * 28,
  257 |                           "pairs_path": str(pair_path), "pairs_sha256": base.file_hash(pair_path),
  258 |                           "per_prompt_weight_mean": [float(v) for v in weights.mean(axis=1)],
  259 |                           "per_prompt_weight_sd": [float(v) for v in weights.std(axis=1, ddof=1)],
  260 |                           "effective_objectives_mean": float(np.mean(
  261 |                               (weights.sum(axis=0) ** 2) / (weights ** 2).sum(axis=0))),
  262 |                           "min_surplus_mean": float(min_surplus.mean()),
  263 |                           "min_surplus_negative_prompts": int((min_surplus < 0).sum()),
  264 |                           "max_identity_residual": float(np.abs(identity).max()),
  265 |                           "all_certified": bool(certified.all()),
  266 |                           "solver_solution_sha256": solver_hash,
  267 |                           "seconds": time.monotonic() - split_start}
  268 |         split_digests[split] = digest_of(pool, pids)
  269 |         base.write_json(out / split / "complete.json", outputs[split])
  270 |         print(json.dumps({k: v for k, v in outputs[split].items()
  271 |                           if k not in ("pairs_path", "pairs_sha256")}), flush=True)
  272 | 
  273 |     prov = {**shared_meta, "objectives": list(OBJECTIVES), "panel": getattr(base, "PANEL_LABEL", "UF-4"),
  274 |             "aggregation": "prompt_wise_nash", "beta": args.beta, "eta": args.eta,
  275 |             "weight_l1": weight_l1, "splits": outputs,
  276 |             # content digests of the rows each split actually uses, not of the
  277 |             # split NAMES: the previous value hashed sorted(outputs), so train and
  278 |             # dev were equal to each other and unchanged by the pool's contents
  279 |             "train_pool_sha256": split_digests.get("train"),
  280 |             "dev_pool_sha256": split_digests.get("dev"),
  281 |             "pool_digest_fields": ("prompt_id, role, occurrence index, candidate_id, "
  282 |                                    "response_sha256"),
  283 |             "dev_note": ("no shared dual: each prompt fits its own adversarial weights on "
  284 |                          "whichever split it is in, so dev measures neural generalisation "
  285 |                          "but not generalisation of a dual fitted on train")}
  286 |     base.write_json(out / "dataset_provenance.json", prov)
  287 | 
  288 |     manifest = None
  289 |     if not args.skip_dataset:
  290 |         import subprocess
  291 |         dataset_out = ROOT / "datasets" / args.out_name
  292 |         env = dict(os.environ, HF_DATASETS_CACHE=str(out / "arrow_cache"),
  293 |                    PYTHONDONTWRITEBYTECODE="1")
  294 |         done = subprocess.run(
  295 |             [sys.executable, "-m", "mnpo_scripts.prepare_nbpo_dataset",
  296 |              "--train", str(out / "pairs/train.jsonl"), "--dev", str(out / "pairs/dev.jsonl"),
  297 |              "--output", str(dataset_out), "--provenance", str(out / "dataset_provenance.json")],
  298 |             env=env, capture_output=True, text=True)
  299 |         if done.returncode:
  300 |             base.write_json(out / "dataset_materialization_failure.json",
  301 |                             {"returncode": done.returncode, "stdout": done.stdout[-4000:],
  302 |                              "stderr": done.stderr[-4000:]})
  303 |             raise RuntimeError("Dataset materialization failed; diagnostic preserved")
  304 |         manifest = base.file_hash(dataset_out / "precompute_manifest.json")
  305 | 
  306 |     base.write_json(out / "complete.json", {

```

## E07 — Superseded scorer refuses unacknowledged execution
Path: `analysis/sub_20260914/code_snapshot_20260917_union/union_score_uw.py`
SHA256: `3752827f4cd0fcda933c2d3057dbe76ff7eb9cac0c055aca70080a707f6c732a`
```text
   65 |     return out, settings
   66 | 
   67 | 
   68 | SUPERSEDED = (
   69 |     "union_score_uw.py is superseded by union_score_panel.py and refuses to run.\n"
   70 |     "\n"
   71 |     "It wires A_policy to the learner triangle and A_ref to the learner-by-reference\n"
   72 |     "cross block, keeping the reference triangle only as a scalar diagnostic. That is\n"
   73 |     "not the finite game Section 5.2 defines, and it is not what\n"
   74 |     "compute_disagreement_point documents its argument to be: with every learner tied\n"
   75 |     "to every other learner, every reference tied to every other reference, and every\n"
   76 |     "learner beating every reference at .75, this wiring turns a surplus of +.25 into\n"
   77 |     "-.25. The surplus is what finite-pool feasibility tests, so a score directory\n"
   78 |     "written here cannot be solved into the declared problem.\n"
   79 |     "\n"
   80 |     "Use:\n"
   81 |     "  python3 union_score_panel.py --objectives 4 --tag <tag> --pool <pool> --out <dir>\n"
   82 |     "\n"
   83 |     "It reproduces this file's own UW tensors exactly where they agree, writes A_LL,\n"
   84 |     "A_LR and A_RR each under its own name, and stamps tensor_role_schema so the\n"
   85 |     "solver can tell a corrected shard from a pre-fix one. This file is kept only as\n"
   86 |     "the provenance of scores written before 2026-09-18; run it with\n"
   87 |     "--i-know-this-is-the-superseded-scorer if you are deliberately reproducing those.\n"
   88 | )
   89 | 
   90 | 
   91 | def refuse_unless_acknowledged():
   92 |     import sys
   93 |     flag = "--i-know-this-is-the-superseded-scorer"
   94 |     if flag in sys.argv:
   95 |         sys.argv.remove(flag)
   96 |         sys.stderr.write("WARNING: running the superseded UW scorer on purpose.\n")
   97 |         return
   98 |     raise SystemExit(SUPERSEDED)
   99 | 
  100 | 
  101 | def main():
  102 |     refuse_unless_acknowledged()
  103 |     ap = argparse.ArgumentParser(description=__doc__)
  104 |     ap.add_argument("--tag", default="uw1_psc")
  105 |     ap.add_argument("--judge-shards", type=int, default=4)
  106 |     ap.add_argument("--pool", default="uw1")
  107 |     ap.add_argument("--pool-shards", type=int, default=4)
  108 |     ap.add_argument("--out", default="uw1")
  109 |     ap.add_argument("--shards", type=int, default=4)
  110 |     args = ap.parse_args()
  111 | 
  112 |     pool, pool_settings = load_pool(args.pool, args.pool_shards)
  113 |     status = Counter()

```

## E08 — Shared style counter, fit status and complete-order aggregation
Path: `analysis/sub_20260914/code_snapshot_20260917_union/ahv2_style_control.py`
SHA256: `ee230bf19864ed5a4b3e8ebb3c5f151f052123211df7523402189ad212541bef`
```text
   62 | HEADER = re.compile(r"^\s{0,3}#{1,6}\s", re.M)
   63 | BOLD = re.compile(r"\*\*[^*\n]+\*\*|__[^_\n]+__")
   64 | LIST = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s)", re.M)
   65 | ORDERS = 2
   66 | 
   67 | 
   68 | def make_length(unit: str, tokenizer_path: str | None):
   69 |     """One length counter, applied to both answers.
   70 | 
   71 |     The unit has to be identical on the two sides or the difference is not a
   72 |     style difference. Returning a single closure is how that is enforced: there
   73 |     is no second code path for the side whose file happens to store a token
   74 |     count.
   75 |     """
   76 |     if unit == "tokens":
   77 |         if not tokenizer_path:
   78 |             raise SystemExit("--length-unit tokens needs --tokenizer")
   79 |         from transformers import AutoTokenizer
   80 |         tok = AutoTokenizer.from_pretrained(tokenizer_path)
   81 | 
   82 |         def count(text: str) -> int:
   83 |             return len(tok(text, add_special_tokens=False)["input_ids"])
   84 |         return count, {"unit": "tokenizer_tokens", "tokenizer": tokenizer_path,
   85 |                        "add_special_tokens": False}
   86 |     if unit == "words":
   87 |         return (lambda text: len(text.split())), {"unit": "whitespace_words"}
   88 |     raise SystemExit("unknown --length-unit %r" % unit)
   89 | 
   90 | 
   91 | def style(text: str, length) -> np.ndarray:
   92 |     """The four official style features, length measured by the shared counter."""
   93 |     return np.array([float(length(text)),
   94 |                      float(len(HEADER.findall(text))),
   95 |                      float(len(BOLD.findall(text))),
   96 |                      float(len(LIST.findall(text)))], dtype=np.float64)
   97 | 
   98 | 
   99 | def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  100 |     total = a + b
  101 |     out = np.zeros_like(a)
  102 |     live = total > 0
  103 |     out[live] = (a[live] - b[live]) / total[live]
  104 |     return out
  105 | 
  106 | 
  107 | def fit(y: np.ndarray, X: np.ndarray, iters: int = 200, tol: float = 1e-10):
  108 |     """Newton-Raphson logistic fit on continuous y in [0, 1].
  109 | 
  110 |     The judged outcome is 1, 0 or .5 for a tie, so this is the Bernoulli
  111 |     log-likelihood evaluated at a fractional response -- the same objective the
  112 |     official fit uses when it splits a tie across both sides.
  113 | 
  114 |     Returns (beta, info) and returns None only when the linear system itself
  115 |     fails. An earlier version returned the coefficients after exhausting its
  116 |     iterations with no way for the caller to tell, so a replicate that had not
  117 |     converged was counted as a successful one and narrowed the interval. The
  118 |     caller now reads `info["converged"]`, and the bootstrap discards a replicate
  119 |     that did not converge or produced a non-finite coefficient.
  120 |     """
  121 |     n, d = X.shape
  122 |     beta = np.zeros(d)
  123 |     for it in range(1, iters + 1):
  124 |         eta = np.clip(X @ beta, -30.0, 30.0)
  125 |         p = 1.0 / (1.0 + np.exp(-eta))
  126 |         w = np.maximum(p * (1.0 - p), 1e-10)
  127 |         grad = X.T @ (y - p)
  128 |         hess = X.T @ (X * w[:, None])
  129 |         hess.flat[:: d + 1] += 1e-8          # ridge, so a collinear replicate still solves
  130 |         try:
  131 |             step = np.linalg.solve(hess, grad)
  132 |         except np.linalg.LinAlgError:
  133 |             return None
  134 |         beta = beta + step
  135 |         if not np.all(np.isfinite(beta)):
  136 |             return None
  137 |         if np.max(np.abs(step)) < tol:
  138 |             return beta, {"converged": True, "iterations": it,
  139 |                           "gradient_inf_norm": float(np.abs(grad).max()),
  140 |                           "last_step_inf_norm": float(np.abs(step).max())}
  141 |     eta = np.clip(X @ beta, -30.0, 30.0)
  142 |     p = 1.0 / (1.0 + np.exp(-eta))
  143 |     return beta, {"converged": False, "iterations": iters,
  144 |                   "gradient_inf_norm": float(np.abs(X.T @ (y - p)).max()),

  155 |                          "judging run used, and the baseline is keyed by uid")
  156 |     ap.add_argument("--replicates", type=int, default=2000)
  157 |     ap.add_argument("--seed", type=int, default=20260918)
  158 |     ap.add_argument("--length-unit", default="tokens", choices=("tokens", "words"),
  159 |                     help="how BOTH answers' length is counted; never one each way")
  160 |     ap.add_argument("--tokenizer", default="/work/models/bases/Qwen2.5-7B-Instruct")
  161 |     ap.add_argument("--out", required=True)
  162 |     args = ap.parse_args()
  163 | 
  164 |     length, length_meta = make_length(args.length_unit, args.tokenizer)
  165 |     # the defect this flag exists for: identical text must give a zero difference
  166 |     probe = style("one two three four five six seven", length)
  167 |     if float(normalized_difference(probe, probe)[0]) != 0.0:
  168 |         raise SystemExit("length counter is not self-consistent")
  169 | 
  170 |     judged = Path(args.judged)
  171 |     complete = json.loads((judged / "complete.json").read_text())
  172 |     arm_path = Path(complete["arm_responses"])
  173 | 
  174 |     arm = {}
  175 |     for line in arm_path.open():
  176 |         r = json.loads(line)
  177 |         arm[r["prompt_id"]] = style(r["response"], length)
  178 | 
  179 |     # the same join the judging run used: panel prompt_id -> uid -> baseline answer
  180 |     uid_of = {}
  181 |     for line in Path(args.panel).open():
  182 |         r = json.loads(line)
  183 |         uid_of[r["prompt_id"]] = r["uid"]
  184 | 
  185 |     by_uid = {}
  186 |     for line in Path(args.baseline).open():
  187 |         r = json.loads(line)
  188 |         text = r["messages"][-1]["content"] if "messages" in r else r.get("response", "")
  189 |         if isinstance(text, dict):
  190 |             text = text.get("answer", "")
  191 |         by_uid[r["uid"]] = style(text, length)
  192 |     base = {pid: by_uid[uid] for pid, uid in uid_of.items() if uid in by_uid}
  193 | 
  194 |     # both orders or neither: the raw figure averages the two presentations, so a
  195 |     # prompt that parsed in only one of them would enter the control carrying a
  196 |     # first-position bias the raw figure does not have
  197 |     seen = {}
  198 |     skipped = {"no_style": 0, "not_ok": 0, "single_order_prompts": 0}
  199 |     for line in (judged / "verdicts.jsonl").open():
  200 |         v = json.loads(line)
  201 |         if v.get("status") != "ok":
  202 |             skipped["not_ok"] += 1
  203 |             continue
  204 |         pid = v["prompt_id"]
  205 |         if pid not in arm or pid not in base:
  206 |             skipped["no_style"] += 1
  207 |             continue
  208 |         seen.setdefault(pid, {})[int(v["order"])] = float(v["value_for_arm"])
  209 | 
  210 |     rows = []
  211 |     for pid, byorder in seen.items():
  212 |         if len(byorder) != ORDERS:
  213 |             skipped["single_order_prompts"] += 1
  214 |             continue
  215 |         d = normalized_difference(arm[pid], base[pid])
  216 |         for order in sorted(byorder):
  217 |             rows.append((pid, byorder[order], d))
  218 |     if not rows:
  219 |         raise SystemExit("no usable observation: check the prompt-id join")
  220 | 
  221 |     pids = sorted({r[0] for r in rows})
  222 |     index = {p: i for i, p in enumerate(pids)}
  223 |     by_prompt: list[list[int]] = [[] for _ in pids]
  224 |     for i, (pid, _, _) in enumerate(rows):
  225 |         by_prompt[index[pid]].append(i)
  226 | 
  227 |     y = np.array([r[1] for r in rows])
  228 |     D = np.stack([r[2] for r in rows])
  229 |     # token length enters on a log scale in the official fit, because a 100-vs-200
  230 |     # token gap is not the same stylistic distance as 2000-vs-2100; the normalized
  231 |     # difference is already scale free, so it is used directly here and the raw
  232 |     # token counts are reported so the choice is auditable.
  233 |     X = np.column_stack([np.ones(len(rows)), D])
  234 | 
  235 |     solved = fit(y, X)
  236 |     if solved is None:
  237 |         raise SystemExit("the point fit did not solve")
  238 |     beta, fit_info = solved
  239 |     if not fit_info["converged"]:
  240 |         raise SystemExit("the point fit exhausted its iterations: %s" % fit_info)
  241 |     point = 1.0 / (1.0 + math.exp(-beta[0]))
  242 |     raw = float(y.mean())
  243 | 
  244 |     rng = np.random.default_rng(args.seed)
  245 |     draws, failed = [], 0
  246 |     n = len(pids)
  247 |     for _ in range(args.replicates):
  248 |         pick = rng.integers(0, n, n)
  249 |         idx = np.concatenate([by_prompt[k] for k in pick])
  250 |         got = fit(y[idx], X[idx])
  251 |         if got is None or not got[1]["converged"]:
  252 |             failed += 1
  253 |             continue
  254 |         draws.append(1.0 / (1.0 + math.exp(-got[0][0])))
  255 |     draws = np.sort(np.asarray(draws))
  256 |     lo, hi = (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))) \
  257 |         if draws.size else (float("nan"), float("nan"))
  258 | 
  259 |     out = {
  260 |         "arm": complete["arm"],
  261 |         "kind": complete["kind"],

  267 |             "whole arena with a per-competitor term and the official judge. This is a per-arm "
  268 |             "logistic fit against the frozen baseline under the campaign's own judge "
  269 |             + str(complete["judge"]) + ". Same estimand, different estimator and different "
  270 |             "judge; do not report it as the official SC figure."),
  271 |         "style_features": ["answer_length", "markdown_headers", "bold_spans", "list_items"],
  272 |         "feature_encoding": "(arm - baseline) / (arm + baseline), 0 when both are zero",
  273 |         "length_counter": length_meta,
  274 |         "length_counter_note": ("one counter for both answers; an identical-text "
  275 |                                 "self-difference of zero is asserted before any data is read"),
  276 |         "point_fit": fit_info,
  277 |         "order_rule": "a prompt enters only when both presentation orders parsed",
  278 |         "raw_win_rate": raw,
  279 |         "style_controlled_win_rate": point,
  280 |         "style_controlled_ci95": [lo, hi],
  281 |         "shift_from_raw": point - raw,
  282 |         "coefficients": {"intercept": float(beta[0]),
  283 |                          **{k: float(v) for k, v in zip(
  284 |                              ["answer_length", "markdown_headers", "bold_spans", "list_items"],
  285 |                              beta[1:])}},
  286 |         "mean_feature_difference": {k: float(v) for k, v in zip(
  287 |             ["answer_length", "markdown_headers", "bold_spans", "list_items"], D.mean(axis=0))},
  288 |         "observations": len(rows),
  289 |         "prompts": len(pids),
  290 |         "skipped": skipped,
  291 |         "bootstrap": {"replicates": args.replicates, "unit": "prompt",
  292 |                       "refit_per_replicate": True, "failed_replicates": failed,
  293 |                       "seed": args.seed},
  294 |         "raw_win_rate_recorded_by_the_judging_run": complete["WIN_RATE_VS_BASELINE"],
  295 |         "join": {"panel": args.panel, "prompt_id_to_uid": len(uid_of),
  296 |                  "baseline_answers": len(by_uid)},
  297 |     }
  298 |     Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  299 |     Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
  300 |     print(json.dumps({k: out[k] for k in (
  301 |         "arm", "raw_win_rate", "style_controlled_win_rate", "style_controlled_ci95",
  302 |         "shift_from_raw", "prompts", "observations")}))
  303 |     return 0

```

## E09 — Paired style loader and failure-aware bootstrap
Path: `analysis/sub_20260914/code_snapshot_20260917_union/ahv2_sc_paired.py`
SHA256: `668adab4d7f31fb5e3614f9284b059ed40302ee30ac943d550a66f08c8ac52c0`
```text
   25 | 
   26 | sys.path.insert(0, "/work/sub_20260914/code")
   27 | from ahv2_style_control import ORDERS, fit, make_length, normalized_difference, style
   28 | 
   29 | 
   30 | def load(judged: Path, baseline: Path, panel: Path, length=None, dropped=None):
   31 |     """Both answers counted by the SAME `length`, and both orders or neither.
   32 | 
   33 |     The two rules are the point of this loader. Measuring the arm in tokenizer
   34 |     tokens and the baseline in whitespace words made identical answers differ by
   35 |     .44 on the length feature, which biases the intercept the control exists to
   36 |     isolate; and a prompt present in one presentation order only would enter with
   37 |     a first-position bias the raw figure does not carry.
   38 |     """
   39 |     if length is None:
   40 |         # a caller that does not care about the unit still gets ONE unit on both
   41 |         # sides; the defect this guards against was two different units, not the
   42 |         # choice between them
   43 |         length, _ = make_length("words", None)
   44 |     complete = json.loads((judged / "complete.json").read_text())
   45 |     arm = {}
   46 |     for line in Path(complete["arm_responses"]).open():
   47 |         r = json.loads(line)
   48 |         arm[r["prompt_id"]] = style(r["response"], length)
   49 |     uid_of = {}
   50 |     for line in panel.open():
   51 |         r = json.loads(line)
   52 |         uid_of[r["prompt_id"]] = r["uid"]
   53 |     by_uid = {}
   54 |     for line in baseline.open():
   55 |         r = json.loads(line)
   56 |         text = r["messages"][-1]["content"] if "messages" in r else r.get("response", "")
   57 |         if isinstance(text, dict):
   58 |             text = text.get("answer", "")
   59 |         by_uid[r["uid"]] = style(text, length)
   60 |     base = {pid: by_uid[uid] for pid, uid in uid_of.items() if uid in by_uid}
   61 |     if dropped is None:
   62 |         dropped = {}
   63 |     seen, single_order = {}, 0
   64 |     for line in (judged / "verdicts.jsonl").open():
   65 |         v = json.loads(line)
   66 |         if v.get("status") != "ok":
   67 |             continue
   68 |         pid = v["prompt_id"]
   69 |         if pid in arm and pid in base:
   70 |             seen.setdefault(pid, {})[int(v["order"])] = float(v["value_for_arm"])
   71 |     rows = []
   72 |     for pid, byorder in seen.items():
   73 |         if len(byorder) != ORDERS:
   74 |             single_order += 1
   75 |             continue
   76 |         d = normalized_difference(arm[pid], base[pid])
   77 |         for order in sorted(byorder):
   78 |             rows.append((pid, byorder[order], d))
   79 |     # reported through the caller's dict rather than a third return value, so an
   80 |     # existing two-value caller keeps working and still gets the both-order rule
   81 |     dropped["single_order_prompts"] = single_order
   82 |     return complete["arm"], rows
   83 | 
   84 | 
   85 | def assemble(rows, pids):
   86 |     index = {p: i for i, p in enumerate(pids)}
   87 |     buckets = [[] for _ in pids]
   88 |     y = np.empty(len(rows))
   89 |     D = np.empty((len(rows), 4))
   90 |     for i, (pid, val, feat) in enumerate(rows):
   91 |         y[i] = val
   92 |         D[i] = feat
   93 |         buckets[index[pid]].append(i)
   94 |     X = np.column_stack([np.ones(len(rows)), D])
   95 |     return y, X, buckets
   96 | 
   97 | 
   98 | def sc(y, X, idx):
   99 |     """The style-free win rate, or None if the fit did not converge."""
  100 |     got = fit(y[idx], X[idx])
  101 |     if got is None or not got[1]["converged"]:
  102 |         return None
  103 |     return 1.0 / (1.0 + math.exp(-got[0][0]))
  104 | 
  105 | 
  106 | def main() -> int:

  115 |     ap.add_argument("--tokenizer", default="/work/models/bases/Qwen2.5-7B-Instruct")
  116 |     ap.add_argument("--out", required=True)
  117 |     args = ap.parse_args()
  118 | 
  119 |     length, length_meta = make_length(args.length_unit, args.tokenizer)
  120 |     probe = style("one two three four five six seven", length)
  121 |     if float(normalized_difference(probe, probe)[0]) != 0.0:
  122 |         raise SystemExit("length counter is not self-consistent")
  123 | 
  124 |     baseline, panel = Path(args.baseline), Path(args.panel)
  125 |     drop_a, drop_b = {}, {}
  126 |     name_a, rows_a = load(Path(args.a), baseline, panel, length, drop_a)
  127 |     name_b, rows_b = load(Path(args.b), baseline, panel, length, drop_b)
  128 |     # the paired unit is a prompt scored in BOTH runs
  129 |     pids = sorted({r[0] for r in rows_a} & {r[0] for r in rows_b})
  130 |     rows_a = [r for r in rows_a if r[0] in set(pids)]
  131 |     rows_b = [r for r in rows_b if r[0] in set(pids)]
  132 |     ya, Xa, ba = assemble(rows_a, pids)
  133 |     yb, Xb, bb = assemble(rows_b, pids)
  134 | 
  135 |     pa, pb = sc(ya, Xa, np.arange(len(ya))), sc(yb, Xb, np.arange(len(yb)))
  136 |     if pa is None or pb is None:
  137 |         raise SystemExit("a point fit did not solve")
  138 | 
  139 |     rng = np.random.default_rng(args.seed)
  140 |     n = len(pids)
  141 |     diffs, failed = [], 0
  142 |     for _ in range(args.replicates):
  143 |         pick = rng.integers(0, n, n)
  144 |         ia = np.concatenate([ba[k] for k in pick])
  145 |         ib = np.concatenate([bb[k] for k in pick])
  146 |         sa, sb = sc(ya, Xa, ia), sc(yb, Xb, ib)
  147 |         if sa is None or sb is None:
  148 |             failed += 1
  149 |             continue
  150 |         diffs.append(sa - sb)
  151 |     d = np.asarray(diffs)
  152 |     lo, hi = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
  153 |     # two-sided bootstrap p: how often the sign disagrees with the point estimate
  154 |     point = pa - pb
  155 |     p_two = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
  156 |     out = {
  157 |         "arm": name_a, "comparator": name_b,
  158 |         "metric": "style_controlled_win_rate_local_implementation",
  159 |         "style_controlled": {name_a: pa, name_b: pb},
  160 |         "difference": point,
  161 |         "ci95": [lo, hi],
  162 |         "p_two_sided_bootstrap": min(p_two, 1.0),
  163 |         "contains_zero": bool(lo <= 0.0 <= hi),
  164 |         "reading": ("an interval containing zero means this comparison does not separate the "
  165 |                     "two arms; it is not evidence that they are equal"),
  166 |         "paired_prompts": n,
  167 |         "length_counter": length_meta,
  168 |         "single_order_prompts_dropped": {
  169 |             name_a: drop_a.get("single_order_prompts"),
  170 |             name_b: drop_b.get("single_order_prompts")},
  171 |         "bootstrap": {"replicates": args.replicates, "unit": "prompt",
  172 |                       "refit_both_arms_per_replicate": True,
  173 |                       "failed_replicates": failed, "seed": args.seed},
  174 |     }
  175 |     Path(args.out).parent.mkdir(parents=True, exist_ok=True)

```

## E10 — Legacy gate rejects nonfinite but consumes a scalar
Path: `scripts/nbpo/run_nbpo_stage.py`
SHA256: `303e15682a26ebd4158c4a54b551185eb439dbafbd2caa02040772b120878db6`
```text
  708 |         "decode_params": man.get("decode_params"),
  709 |         "chat_template_kwargs": man.get("chat_template_kwargs"),
  710 |     }
  711 | 
  712 | 
  713 | # --------------------------------------------------------------------------- #
  714 | # gate + promotion
  715 | # --------------------------------------------------------------------------- #
  716 | def promote_candidate(candidate_dir: Path, promote_to: Path, stage: int, fingerprint: str) -> dict:
  717 |     """Versioned, atomic promotion: move the candidate into
  718 |     ``<promote_to>.versions/stage<t>_<fp12>`` and atomically repoint the
  719 |     ``promote_to`` symlink. A pre-existing real directory at ``promote_to`` is
  720 |     renamed aside first; nothing is ever partially overwritten.
  721 |     """
  722 |     promote_to = Path(promote_to)
  723 |     versions = promote_to.parent / (promote_to.name + ".versions")
  724 |     versions.mkdir(parents=True, exist_ok=True)
  725 |     target = versions / f"stage{stage}_{fingerprint[:12]}"
  726 |     if target.exists():
  727 |         raise FileExistsError(f"versioned checkpoint already exists: {target}")
  728 |     os.replace(candidate_dir, target)                      # same filesystem, atomic
  729 |     if promote_to.exists() or promote_to.is_symlink():
  730 |         if promote_to.is_symlink():
  731 |             promote_to.unlink()
  732 |         else:
  733 |             aside = promote_to.parent / f"{promote_to.name}.prev_{int(time.time())}"
  734 |             os.replace(promote_to, aside)
  735 |     tmp_link = promote_to.parent / f".{promote_to.name}.link.tmp"
  736 |     if tmp_link.is_symlink() or tmp_link.exists():
  737 |         tmp_link.unlink()
  738 |     os.symlink(os.path.relpath(target, promote_to.parent), tmp_link)
  739 |     os.replace(tmp_link, promote_to)                        # atomic symlink swap
  740 |     record = {"stage": stage, "fingerprint": fingerprint, "versioned_dir": str(target),
  741 |               "symlink": str(promote_to), "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
  742 |     write_json(versions / "PROMOTION.json", record)
  743 |     return record
  744 | 
  745 | 
  746 | def apply_gate(min_surplus: float, parent_dir: Path, candidate_dir: Path,
  747 |                promote_to: Path, stage: int = 0, fingerprint: str = None) -> dict:
  748 |     """Algorithm 1 lines 11-15: reject on any nonpositive held-out surplus, else promote.
  749 | 
  750 |     A non-finite surplus is a rejection, not a promotion. `nan <= 0` is False in
  751 |     IEEE 754, so a NaN used to fall through to the promote branch and a stage
  752 |     whose surplus could not be computed was accepted as though it had passed.
  753 |     """
  754 |     if not math.isfinite(min_surplus):
  755 |         return {"accepted": False, "promoted_path": str(parent_dir),
  756 |                 "reason": f"min held-out surplus {min_surplus!r} is not finite; the gate "
  757 |                           "cannot be evaluated, so pi_t is retained"}
  758 |     if min_surplus <= 0:
  759 |         return {"accepted": False, "promoted_path": str(parent_dir),
  760 |                 "reason": f"min held-out surplus {min_surplus:.6f} <= 0; "
  761 |                           "stage flagged empirically infeasible, pi_t retained"}
  762 |     fp = fingerprint or checkpoint_fingerprint(str(candidate_dir))
  763 |     rec = promote_candidate(candidate_dir, promote_to, stage, fp)
  764 |     return {"accepted": True, "promoted_path": rec["versioned_dir"], "symlink": str(promote_to),
  765 |             "reason": f"min held-out surplus {min_surplus:.6f} > 0", "promotion": rec}

```

## E11 — Legacy evaluator still averages prompt surpluses
Path: `scripts/nbpo/eval_game_value.py`
SHA256: `0f4c17b91c3ab2ff6524ae1eefcf1847f75ce77acfcbf33aa51becbed6364f03`
```text
   41 | def evaluate_game_value(A_policy: torch.Tensor, A_ref: torch.Tensor, beta: torch.Tensor,
   42 |                         reference_construction: str = "shared_pool") -> dict:
   43 |     """Pure evaluation given the two centered tensors; policy uniform over its pool."""
   44 |     K, X, I, J = A_policy.shape
   45 |     mu = uniform_policy(X, J)
   46 |     pi = uniform_policy(X, I)
   47 |     r = compute_margins(A_policy, pi)
   48 |     V = compute_regularized_game_value(r, mu, beta, form="softmin")
   49 |     d = compute_disagreement_point(A_ref, mu, beta, reference_construction)
   50 |     s = V - d
   51 |     nu = compute_regularized_opponent(r, mu, beta)
   52 |     all_positive = bool((s > 0).all())
   53 |     return {
   54 |         "V": [float(v) for v in V],
   55 |         "d": [float(v) for v in d],
   56 |         "surplus": [float(v) for v in s],
   57 |         "min_surplus": float(s.min()),
   58 |         "avg_surplus": float(s.mean()),
   59 |         "nash_welfare_defined": all_positive,
   60 |         # Nash welfare only exists on the individually-rational set (Eq. (11));
   61 |         # a nonpositive surplus makes it undefined, not "very negative".
   62 |         "nash_welfare": float(torch.log(s).sum()) if all_positive else None,
   63 |         "opponent_entropy": [float(v) for v in opponent_entropy(nu)],
   64 |         "opponent_ess": [float(v) for v in opponent_ess(nu)],
   65 |     }
   66 | 
   67 | 
   68 | def main() -> None:

```

## E12 — Stage runner dispatch and downstream gate
Path: `scripts/nbpo/run_nbpo_stage.py`
SHA256: `303e15682a26ebd4158c4a54b551185eb439dbafbd2caa02040772b120878db6`
```text
  279 | def solve_stage_finite_pool(scfg, A_policy, A_ref, mu, beta, construction, *,
  280 |                            base=Path("."), lambda_init=None, fixed_weights=None):
  281 |     """Shared train/dev dispatch; held-out calls only solve the fixed-weight inner problem."""
  282 |     from types import SimpleNamespace
  283 |     from mnpo_scripts.nbpo_generic import solve_finite_pool, validate_finite_pool_solution
  284 |     from scripts.nbpo.solve_nbpo_dual import build_objective_representation
  285 |     reward_table = scfg.get("reward_table")
  286 |     rep = build_objective_representation(
  287 |         SimpleNamespace(representation=scfg.get("representation", "adaptive_game"),
  288 |                         reward_table=(_resolve(base, reward_table) if reward_table else None)),
  289 |         A_policy, A_ref, mu, beta, construction)
  290 |     result = solve_finite_pool(
  291 |         rep, scfg.get("aggregation", "nash"), eta=float(scfg["eta"]),
  292 |         R=int(scfg.get("R", 1)), M=int(scfg.get("max_dual_calls", scfg.get("M", 800))),
  293 |         gamma=scfg.get("gamma", 0.5), lambda_box=tuple(scfg.get("lambda_box", (1e-3, 1e3))),
  294 |         lambda_init=lambda_init, inner_solver=str(scfg.get("inner_solver", "exact")),
  295 |         dual_solver=str(scfg.get("dual_solver", "subgradient")),
  296 |         dual_tol=(float(scfg["dual_tol"]) if scfg.get("dual_tol") else None),
  297 |         inner_workers=int(scfg.get("inner_workers", 1)),
  298 |         inner_maxiter=int(scfg.get("inner_maxiter", 400)),
  299 |         inner_ftol=float(scfg.get("inner_ftol", 1e-14)),
  300 |         probability_floor=float(scfg.get("probability_floor", 1e-12)),
  301 |         fixed_weights=fixed_weights,
  302 |         weights=(torch.tensor(scfg["weights"], dtype=torch.float64) if "weights" in scfg else None),
  303 |         weight_l1=scfg.get("weight_l1"), warm_start_policy=bool(scfg.get("warm_start_policy", True)),
  304 |         damping=float(scfg.get("damping", 0.0)), adversary_step=float(scfg.get("adversary_step", 1.0)),
  305 |         log_every=int(scfg.get("log_every", 0)), ks_kwargs=scfg.get("ks_kwargs"),
  306 |         max_bound_expansions=int(scfg.get("max_bound_expansions", 3)))
  307 |     validate_finite_pool_solution(
  308 |         result, stationarity_tol=min(1e-4, float(scfg.get("max_stationarity_residual", 1e-4))),
  309 |         extra_map_tol=min(1e-4, float(scfg.get("max_inner_residual", 1e-4))))
  310 |     return result
  311 | 
  312 | 

  935 |     A_ref = torch.from_numpy(np.load(tensor_dir / "tensor_ref.npz")["A"])
  936 |     mu = uniform_policy(A_policy.shape[1], A_policy.shape[3])
  937 |     beta = torch.tensor(scfg["opponent_betas"], dtype=torch.float64)
  938 |     lambda_init = None
  939 |     warm = scfg.get("warm_start_lambda")
  940 |     if warm:
  941 |         prev = json.loads(_resolve(base, warm).read_text())
  942 |         lambda_init = torch.tensor(prev["lambda_raw"], dtype=torch.float64)
  943 |     tensor_meta = json.loads((tensor_dir / "meta.json").read_text())
  944 |     hashes = {name: sha256_file(tensor_dir / name)
  945 |               for name in ("tensor_policy.npz", "tensor_ref.npz", "meta.json")}
  946 |     solver_dir = workdir / "solver"
  947 |     inner_solver = scfg.get("inner_solver")
  948 | 
  949 |     if inner_solver is None:
  950 |         # The legacy alternating path, unchanged, for every existing config.
  951 |         res = solve_nbpo_dual(
  952 |             A_policy, A_ref, mu, beta,
  953 |             eta=float(scfg["eta"]), gamma=scfg["gamma"], M=int(scfg["M"]), R=int(scfg["R"]),
  954 |             lambda_box=tuple(scfg.get("lambda_box", (1e-3, 1e3))),
  955 |             lambda_init=lambda_init, aggregation=scfg.get("aggregation", "nash"),
  956 |             reference_construction=tensor_meta_construction,
  957 |             damping=float(scfg.get("damping", 0.0)),
  958 |         )
  959 |         from scripts.nbpo.solve_nbpo_dual import write_solution_artifact
  960 |         solution = write_solution_artifact(solver_dir, res, tensor_meta, hashes, tensor_dir,
  961 |                                            stage, lambda_warm_started=lambda_init is not None)
  962 |     else:
  963 |         # A config that ASKS for a solver must get it or stop. Silently falling
  964 |         # back would produce a result that claims a solver it never used.
  965 |         from scripts.nbpo.solve_nbpo_dual import write_generic_solution_artifact
  966 |         res = solve_stage_finite_pool(
  967 |             scfg, A_policy, A_ref, mu, beta, tensor_meta_construction,
  968 |             base=base, lambda_init=lambda_init)
  969 |         solution = write_generic_solution_artifact(
  970 |             solver_dir, res, tensor_meta, hashes, tensor_dir, stage,
  971 |             lambda_warm_started=lambda_init is not None,
  972 |             extra={"final_config_validation": final_validation,
  973 |                    "solver_mode": {"inner_solver": str(inner_solver),
  974 |                                    "dual_solver": str(scfg.get("dual_solver")),
  975 |                                    "dual_tol": scfg.get("dual_tol"),
  976 |                                    "outer_iterations_used": res.outer_iterations_used,

 1182 |                                     f"reference seed files == monitoring set ({len(monitoring_ids)})")
 1183 | 
 1184 |     # 8. Monitoring judge + game-value evaluation.
 1185 |     mon_tensor_dir, mon_cells = judge_and_build_tensors(
 1186 |         candidate_specs, mon_ref_specs, objectives, objectives_config, judges["monitoring"],
 1187 |         workdir, tag="monitoring", role="monitoring", reproduction_mode=reproduction_mode)
 1188 |     mon_meta = json.loads((mon_tensor_dir / "meta.json").read_text())
 1189 |     mon_eval = evaluate_game_value(
 1190 |         torch.from_numpy(np.load(mon_tensor_dir / "tensor_policy.npz")["A"]),
 1191 |         torch.from_numpy(np.load(mon_tensor_dir / "tensor_ref.npz")["A"]),
 1192 |         beta)
 1193 |     _step("eval_monitoring", f"min_surplus={mon_eval['min_surplus']:.6f} "
 1194 |                              f"nash_defined={mon_eval['nash_welfare_defined']} "
 1195 |                              f"judge={judges['monitoring']['model_path']}")
 1196 | 
 1197 |     # 9. Gate.
 1198 |     cand_fp = checkpoint_fingerprint(str(candidate_dir))
 1199 |     if binding is not None and binding["candidate_fingerprint"] != cand_fp:
 1200 |         raise RuntimeError("candidate changed between decode and gate")
 1201 |     gate = apply_gate(mon_eval["min_surplus"], parent_dir, candidate_dir, promote_to,
 1202 |                       stage=stage, fingerprint=cand_fp)
 1203 |     record = {
 1204 |         "stage": stage,
 1205 |         "accepted": gate["accepted"],
 1206 |         "reason": gate["reason"],
 1207 |         "candidate_fingerprint": cand_fp,
 1208 |         "parent_fingerprint": parent_fp,
 1209 |         "lineage": {"parent": str(parent_dir), "candidate": str(candidate_dir),

```

## E13 — Campaign job launches trainer directly
Path: `analysis/sub_20260914/code_snapshot_20260917_union/make_pros_train_jobs.py`
SHA256: `bcabadf496632acbcefd6334b84aae5e95f5a85f129885cd4b526f0ee3feba62`
```text
  148 | 
  149 |     spec = {
  150 |         "job_id": f"{args.job_prefix}_train_{args.arm}",
  151 |         "priority": args.priority,
  152 |         "gpus": 4,
  153 |         "cwd": "/work/nbpo_repair_20260909/code",
  154 |         "env": {"PYTHONPATH": "/work/nbpo_repair_20260909/deps_train:/work/nbpo_repair_20260909/code",
  155 |                 "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "4",
  156 |                 "MNPO_DISABLE_APEX": "1", "HF_HUB_OFFLINE": "1",
  157 |                 "TOKENIZERS_PARALLELISM": "false", "WANDB_MODE": "disabled",
  158 |                 "VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
  159 |         "command": ["python3", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
  160 |                     "--nproc_per_node=4", "-m", "mnpo_scripts.run_mnpo", str(config_path)],
  161 |         "timeout_s": 86400,
  162 |         "artifacts": [f"{ROOT}/arms/{args.arm}/config.json"],
  163 |         # The spec names the config by path, so without this a changed recipe
  164 |         # leaves the spec byte-identical and a queue keyed on spec bytes would
  165 |         # never notice that this is a different run.
  166 |         "config_sha256": file_hash(config_path),
  167 |         "dataset_manifest_sha256": manifest,
  168 |         "solver_artifact_sha256": solver,
  169 |     }
  170 |     queue_path = ROOT / "jobs" / "queue" / f"{args.priority}_train_{args.arm}.json"
  171 |     queue_path.write_text(json.dumps(spec, indent=2) + "\n")
  172 | 
  173 |     record = {"arm": args.arm, "targets": args.targets, "dataset": dataset,
  174 |               "effective_batch_note": ("per_device x grad_accum x 4 GPUs; PROSPER "
  175 |                                        "Table 7 uses 128"),
  176 |               "loss_type": args.loss_type, "nbpo_target_mode": target_mode,
  177 |               "dataset_manifest_sha256": manifest, "solver_artifact_sha256": solver,
  178 |               "config_path": str(config_path), "config_sha256": file_hash(config_path),
  179 |               "base_config": str(BASE_CONFIG), "base_config_sha256": file_hash(BASE_CONFIG),
  180 |               "recipe_diff_vs_resolved_base": applied,
  181 |               "horizon_note": ("1250 updates at 32 pairs is the planned exposure of about four "
  182 |                                "pairs per prompt over 10,000 prompts. It is the plan's starting "
  183 |                                "horizon, not a horizon shown to be optimal on this data."),
  184 |               "queue_spec": str(queue_path)}
  185 |     out = ROOT / "provenance" / f"train_job_{args.arm}.json"
  186 |     out.write_text(json.dumps(record, indent=2) + "\n")
  187 |     print(json.dumps({"queued": spec["job_id"], "config": str(config_path),
  188 |                       "dataset_manifest": manifest[:16], "solver": solver[:16]}), flush=True)

```

## E14 — Panel training jobs launch trainer directly
Path: `analysis/sub_20260914/code_snapshot_20260917_union/panel_stage2.py`
SHA256: `e0528503bf22fa970cc2d9a963b35a183bd3a970719240c2d3e6b87c6667734b`
```text
  309 |         # The reference-handling and immutable-token plumbing stays on for every
  310 |         # arm, exactly as the verified UW baseline configs have it: the soft-label
  311 |         # and PROSPER losses also read reference log-ratios and must score the
  312 |         # pool's own tokens. Only the target-column fields are NBPO-specific.
  313 |         for key in ("nbpo_expected_dataset_manifest_sha256",
  314 |                     "nbpo_expected_solver_artifact_sha256"):
  315 |             cfg.pop(key, None)
  316 |         if loss != "nbpo":
  317 |             for key in ("nbpo_target_mode", "nbpo_target_column", "nbpo_target_units",
  318 |                         "nbpo_eta_already_included"):
  319 |                 cfg.pop(key, None)
  320 |         if loss == "dpo_soft":
  321 |             cfg.pop("eta", None)              # dpo_soft has no proximal step
  322 |         if loss == "nbpo":
  323 |             # pin the dataset manifest and the solver solution this arm was built
  324 |             # from, so a config cannot be pointed at a different solve later
  325 |             done = json.loads(Path(UF, "targets", "%s_%s" % (panel, arm),
  326 |                                    "complete.json").read_text())
  327 |             man = done.get("dataset_manifest_sha256")
  328 |             sol = (done["splits"]["train"].get("solver_solution_sha256")
  329 |                    if "splits" in done else None)
  330 |             if not man or not sol:
  331 |                 raise SystemExit("%s: solve record has no manifest/solution hash" % arm)
  332 |             cfg["nbpo_expected_dataset_manifest_sha256"] = man
  333 |             cfg["nbpo_expected_solver_artifact_sha256"] = sol
  334 |         config_dir.mkdir(parents=True, exist_ok=True)
  335 |         path = config_dir / ("%s_%s.yaml" % (panel, arm))
  336 |         path.write_text(yaml.safe_dump(cfg, sort_keys=True))
  337 |         spec = {
  338 |             "job_id": "%s_train_%s" % (panel, arm), "priority": 120, "gpus": 4,
  339 |             "depends_on": [], "cwd": "%s/code" % DEPS,
  340 |             "env": {"PYTHONPATH": "%s/deps_train:%s/code" % (DEPS, DEPS),
  341 |                     "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
  342 |                     "MKL_NUM_THREADS": "4", "MNPO_DISABLE_APEX": "1",
  343 |                     "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
  344 |                     "WANDB_MODE": "disabled",
  345 |                     "VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
  346 |             "timeout_s": 86400,
  347 |             "command": ["python3", "-m", "torch.distributed.run", "--standalone",
  348 |                         "--nnodes=1", "--nproc_per_node=4", "-m",
  349 |                         "mnpo_scripts.run_mnpo", str(path)],
  350 |             "artifacts": [str(UF / "arms" / ("%s_%s" % (panel, arm)) / "config.json")],
  351 |             "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
  352 |             "note": ("%s panel, %s arm: %d train rows, %d steps for %d epochs at %d "
  353 |                      "sequences per step" % (panel.upper(), arm, rows, steps, epochs,
  354 |                                              tokens_per_step)),
  355 |         }
  356 |         queue_dir.mkdir(parents=True, exist_ok=True)
  357 |         (queue_dir / ("120_%s_train_%s.json" % (panel, arm))).write_text(
  358 |             json.dumps(spec, indent=1) + "\n")
  359 |         queued[arm] = {"rows": rows, "max_steps": steps, "config": str(path),
  360 |                        "loss_type": loss}
  361 |         print(json.dumps({arm: queued[arm]}), flush=True)
  362 |     return queued
  363 | 
  364 | 
  365 | if __name__ == "__main__":

```

## E15 — Trainer training and save entry
Path: `mnpo_scripts/run_mnpo.py`
SHA256: `def0546b64f7d2d5dd6360a7a3ff6995c126a06ff0e63326c965443cdd05d36a`
```text
  318 |     if (getattr(training_args, "nbpo_online_reference", False)
  319 |             or getattr(training_args, "nbpo_eval_online_reference", False)):
  320 |         ref_path = getattr(training_args, "nbpo_reference_model_path", "") or \
  321 |             model_args.model_name_or_path
  322 |         logger.info(f"*** Loading frozen NBPO reference (pi_t) from {ref_path} ***")
  323 |         ref = AutoModelForCausalLM.from_pretrained(
  324 |             ref_path, torch_dtype=next(trainer.model.parameters()).dtype, use_cache=False,
  325 |             revision=model_args.model_revision, attn_implementation=model_args.attn_implementation,
  326 |             trust_remote_code=model_args.trust_remote_code)
  327 |         ref.eval()
  328 |         for prm in ref.parameters():
  329 |             prm.requires_grad_(False)
  330 |         trainer.nbpo_reference_model = ref.to(trainer.accelerator.device)

  347 |     ###############
  348 |     # Training loop
  349 |     ###############
  350 |     checkpoint = None
  351 |     if training_args.resume_from_checkpoint is not None:
  352 |         checkpoint = training_args.resume_from_checkpoint
  353 |     elif last_checkpoint is not None:
  354 |         checkpoint = last_checkpoint
  355 |     train_result = trainer.train(resume_from_checkpoint=checkpoint)
  356 |     metrics = train_result.metrics
  357 |     metrics["train_samples"] = len(train_dataset)
  358 |     trainer.log_metrics("train", metrics)
  359 |     trainer.save_metrics("train", metrics)
  360 |     trainer.save_state()
  361 | 
  362 |     logger.info("*** Training complete ***")
  363 |     if training_args.nbpo_profile_updates > 0:
  364 |         logger.info("Disposable runtime profile complete at %d updates; scheduler horizon was %d. No model export.",
  365 |                     trainer.state.global_step, training_args.max_steps)
  366 |         return
  367 | 
  368 |     skip_final_save = os.environ.get("MNPO_SKIP_FINAL_SAVE", "").lower() in {"1", "true", "yes"}
  369 |     if skip_final_save:
  370 |         logger.info("*** Skip final model save because MNPO_SKIP_FINAL_SAVE is set ***")
  371 |         if trainer.accelerator.is_main_process:
  372 |             trainer.tokenizer.save_pretrained(training_args.output_dir)
  373 |             trainer.model.config.use_cache = True
  374 |             trainer.model.config.save_pretrained(training_args.output_dir)
  375 |         logger.info("*** Training complete! ***")
  376 |         return
  377 | 
  378 |     ##################################
  379 |     # Save model and create model card
  380 |     ##################################
  381 |     logger.info("*** Save model ***")
  382 |     trainer.save_model(training_args.output_dir)
  383 |     logger.info(f"Model saved to {training_args.output_dir}")
  384 | 
  385 |     # Add this step to explicitly save the tokenizer.
  386 |     if trainer.accelerator.is_main_process:
  387 |         trainer.tokenizer.save_pretrained(training_args.output_dir)
  388 |         logger.info(f"Tokenizer saved to {training_args.output_dir}")
  389 | 
  390 |     kwargs = {

```

## E16 — Residual panel label in source UW1 template
Path: `analysis/sub_20260914/code_snapshot_20260917_union/solve_pros4_targets_uw1.py`
SHA256: `32d8a051c2f36bb19ac04ea6e0318a7571a874da3a061d37e081c8ecac1f7483`
```text
   87 | CANONICAL_DECIMALS_LEGACY = 10
   88 | # The tensor-role schema this solver's game assumes: A_policy is the
   89 | # learner-by-reference block and A_ref the reference triangle.
   90 | REQUIRED_TENSOR_ROLE_SCHEMA = "lr_rr_v2"
   91 | # The panel this module was generated for. build_panel_solvers.py rewrites it, so a
   92 | # US/UT/UW run records its own panel instead of the UF-4 label of the source file.
   93 | PANEL_LABEL = "UF-4"
   94 | 
   95 | 
   96 | def split_pool_digest(pool, prompt_ids, roles=("learner", "comparator")):
   97 |     """A content digest of the pool rows this split actually uses.
   98 | 
   99 |     The fields it hashes are the immutable identity of each candidate: the
  100 |     prompt, the role, the occurrence index, the candidate id and the sha256 of
  101 |     the response text. Mutating one byte of one response changes its
  102 |     response_sha256 and therefore this digest, which is the property the audit
  103 |     asks for and the property the previous value did not have -- that one hashed
  104 |     `sorted(outputs)`, i.e. the split NAMES, so train and dev came out equal to
  105 |     each other and equal across panels no matter what the pool contained.
  106 |     """
  107 |     import hashlib
  108 |     h = hashlib.sha256()
  109 |     for pid in sorted(prompt_ids):
  110 |         entry = pool.get(pid)
  111 |         if entry is None:
  112 |             raise ValueError("split prompt %s is absent from the pool" % pid)
  113 |         for role in roles:
  114 |             for index in sorted(entry.get(role, {})):
  115 |                 row = entry[role][index]
  116 |                 identity = (str(pid), role, str(index),
  117 |                             str(row.get("candidate_id", "")),
  118 |                             str(row.get("response_sha256")
  119 |                                 or hashlib.sha256(
  120 |                                     str(row.get("response", "")).encode()).hexdigest()))
  121 |                 h.update(("\x1f".join(identity) + "\x1e").encode())
  122 |     return h.hexdigest()
  123 | 
  124 | 
  125 | def quantize_canonical_row(row):
  126 |     """Serialize the solver masses so they survive the round trip exactly.
  127 | 
  128 |     prepare_nbpo_dataset reads these rows with a standard JSON parser and then
  129 |     checks target == log(w_a/c_a) - log(w_b/c_b) to 1e-9 absolute. An earlier
  130 |     version of this function rounded both masses to ten FIXED DECIMALS and

  232 | def load_pool(pool_root, shards):
  233 |     """prompt_id -> {role: {index: event}}, from the immutable pool chunks."""
  234 |     pool = {}
  235 |     settings = None
  236 |     split_provenance = []
  237 |     # "shard" is the shard index; splits_dir and split_files_sha256 record WHICH
  238 |     # prompts a shard drew, not HOW they were sampled. This pool is assembled
  239 |     # from two prompt batches under byte-identical sampling settings, so those
  240 |     # three are collected rather than compared. Everything else, including any
  241 |     # field added later, still has to match exactly.
  242 |     PROVENANCE_KEYS = ("shard", "splits_dir", "split_files_sha256")
  243 |     for shard in range(shards):
  244 |         directory = Path(pool_root) / f"shard{shard}"
  245 |         shard_settings = json.loads((directory / "settings.json").read_text())
  246 |         comparable = {k: v for k, v in shard_settings.items()
  247 |                       if k not in PROVENANCE_KEYS}
  248 |         split_provenance.append({k: shard_settings.get(k) for k in PROVENANCE_KEYS})
  249 |         if settings is None:
  250 |             settings = comparable
  251 |         elif settings != comparable:
  252 |             differing = sorted(k for k in set(settings) | set(comparable)
  253 |                                if settings.get(k) != comparable.get(k))
  254 |             raise ValueError("Pool shards were generated under different sampling "
  255 |                              "settings; differing fields: %s" % differing)
  256 |         for path in sorted(directory.glob("chunk*.jsonl")):
  257 |             manifest = json.loads((directory / (path.stem + ".manifest.json")).read_text())
  258 |             if file_hash(path) != manifest["sha256"]:
  259 |                 raise ValueError(f"Pool chunk hash mismatch: {path}")
  260 |             with path.open() as stream:
  261 |                 for line in stream:
  262 |                     event = json.loads(line)
  263 |                     pool.setdefault(event["prompt_id"], {}).setdefault(
  264 |                         event["role"], {})[event["sample_index"]] = event
  265 |     settings = dict(settings or {}, shard_prompt_provenance=split_provenance)
  266 |     return pool, settings
  267 | 
  268 | 
  269 | def certificate_record(result, solution_certificate):
  270 |     """Everything needed to tell 'solved' from 'at a bound' from 'unresolved'."""
  271 |     weights = result.weights.numpy()
  272 |     return {"certified": bool(solution_certificate.get("certified")),
  273 |             "independent_stationarity_inf": solution_certificate.get("independent_stationarity_inf"),
  274 |             "unprojected_kkt_residual": result.kkt_residual,
  275 |             "projected_kkt_residual": result.projected_kkt_residual,

```

## E17 — Panel generator uses a copied base
Path: `analysis/sub_20260914/code_snapshot_20260917_union/build_panel_solvers.py`
SHA256: `e96df0927515180c114ff2880d330dcc1701ea35b1fbfd7a9c325527477d3f87`
```text
   34 | 
   35 | def main():
   36 |     ap = argparse.ArgumentParser(description=__doc__)
   37 |     ap.add_argument("--panel", required=True, help="us1 or ut1")
   38 |     ap.add_argument("--objectives", type=int, required=True)
   39 |     ap.add_argument("--split-dir", required=True,
   40 |                     help="the restricted/certified split directory, e.g. us_v1p")
   41 |     args = ap.parse_args()
   42 | 
   43 |     p, K = args.panel, args.objectives
   44 |     objs = ", ".join('"item%d"' % k for k in range(K))
   45 |     written = {}
   46 | 
   47 |     # 1. the panel base: loaders, hashing, writers, and the objective slots
   48 |     src = CODE / "solve_pros4_targets_uw1.py"
   49 |     t = src.read_text()
   50 |     t = sub(t, [('OBJECTIVES = ("item0", "item1", "item2", "item3")',
   51 |                  "OBJECTIVES = (%s)" % objs),
   52 |                 # the panel label travels with the module, so a US/UT/UW run does
   53 |                 # not record the UF-4 label of the file it was generated from
   54 |                 ('PANEL_LABEL = "UF-4"', 'PANEL_LABEL = "%s"' % p.upper())],
   55 |             src.name)
   56 |     header = ("# Generated from solve_pros4_targets_uw1.py by build_panel_solvers.py\n"
   57 |               "# -- do not edit by hand. source sha256 %s\n"
   58 |               "# change: OBJECTIVES -> item0..item%d, the %s panel's declared objective\n"
   59 |               "#         count. The rubric text behind each slot is that panel's frozen\n"
   60 |               "#         rubric, recorded in panel/%s/freeze.json; slot k of two panels\n"
   61 |               "#         is never pooled as one objective.\n"
   62 |               % (sha(src), K - 1, p.upper()[:2], args.split_dir.replace("_v1p", "_v1")))
   63 |     base_name = "solve_pros4_targets_%s.py" % p
   64 |     (CODE / base_name).write_text(header + t)
   65 |     written[base_name] = None
   66 | 
   67 |     # 2. the two per-prompt arms and the probe: repoint at this panel
   68 |     roots = [('("train", (ROOT / "scores/uw1", ROOT / "pools/uw1", "policy_train")),',
   69 |               '("train", (ROOT / "scores/%s", ROOT / "pools/%s", "policy_train")),' % (p, p)),
   70 |              ('("dev", (ROOT / "scores/uw1", ROOT / "pools/uw1", "policy_dev"))):',
   71 |               '("dev", (ROOT / "scores/%s", ROOT / "pools/%s", "policy_dev"))):' % (p, p)),
   72 |              ('with (ROOT / "splits/uw_v1p" / f"{split_file}.jsonl").open() as stream:',
   73 |               'with (ROOT / "splits/%s" / f"{split_file}.jsonl").open() as stream:'
   74 |               % args.split_dir),
   75 |              ("import solve_pros4_targets_uw1 as base",
   76 |               "import solve_pros4_targets_%s as base" % p),
   77 |              # a stale literal objective count silently turns into a fake
   78 |              # infeasibility on a panel with a different K
   79 |              ("K = 4", "K = len(base.OBJECTIVES)")]
   80 |     for stem in ("solve_pros4_pw_nbpo", "solve_pros4_prosper", "solve_pros4_pw_fixedref"):
   81 |         s = CODE / ("%s_uw1.py" % stem)
   82 |         if not s.exists():
   83 |             continue
   84 |         text = s.read_text()
   85 |         pairs = [r for r in roots if text.count(r[0]) == 1]
   86 |         if not any(r[0].startswith("import solve") for r in pairs):
   87 |             raise SystemExit("%s: base import anchor missing" % s.name)
   88 |         text = sub(text, pairs, s.name)
   89 |         head = ("# Generated from %s by build_panel_solvers.py -- do not edit by hand.\n"
   90 |                 "# source sha256 %s\n"
   91 |                 "# change: score/pool roots -> %s, restricted split -> %s, base module ->\n"
   92 |                 "#         solve_pros4_targets_%s. The rule this arm implements is\n"
   93 |                 "#         untouched.\n" % (s.name, sha(s), p, args.split_dir, p))

  139 |             raise SystemExit("%s --help failed: %s" % (name, out.stderr[-400:]))
  140 |         # The guard is that no generated module still points at the source panel.
  141 |         # Matching the bare substring "uw1" was wrong: it fires on "uw1c", whose
  142 |         # roots ARE correctly repointed, so a legitimate generation reported a
  143 |         # failure after having already written its files. Compare whole tokens
  144 |         # and exempt the panel being generated.
  145 |         stale = {tok for tok in re.findall(r"uw[0-9a-z]*", out.stdout)
  146 |                  if tok != p and not tok.startswith(p)}
  147 |         if stale:
  148 |             raise SystemExit("%s still advertises %s rather than %s: %s"
  149 |                              % (name, sorted(stale), p, out.stdout[-600:]))
  150 |     import json
  151 |     print(json.dumps({"panel": p, "objectives": list(mod.OBJECTIVES),
  152 |                       "generated": written}, indent=1))
  153 |     return 0
  154 | 
  155 | 
  156 | if __name__ == "__main__":
  157 |     raise SystemExit(main())

```

## E18 — Source claims about audit closure and copy regeneration
Path: `analysis/sub_20260914/code_snapshot_20260917_union/README.md`
SHA256: `794f584b4dca4a5ca8546a8743ed3f72207b3cd8f29b7ed9c72e48baa322954d`
```text
   61 | this carry `canonical_target_quantized_decimals = 10`; new rows carry
   62 | `canonical_target_serialization = "exact_float64_round_trip"`.
   63 | 
   64 | ## Follow-up audit, 2026-09-18 (b949919)
   65 | 
   66 | A second audit pass on the first round of fixes found three of them incomplete
   67 | and three new defects in the style-control code added with them. All six are
   68 | closed here.
   69 | 
   70 | **The scorer fix reached only the generalized scorer.** `union_score_uw.py`,
   71 | which wrote the published UW scores, still carried the old wiring, and the
   72 | solver's loader accepted a score directory on hash and shape alone — a sha256
   73 | certifies that the bytes are the ones that were written, not that they mean what
   74 | the reader assumes. Score shards now carry `tensor_role_schema` and per-tensor
   75 | `bank_ids`; the solver refuses a shard that does not declare the schema this
   76 | game needs, and `union_score_uw.py` refuses to run at all unless explicitly
   77 | acknowledged. A directory written before 2026-09-18 has no schema and is
   78 | rejected rather than solved into a different finite game.
   79 | 
   80 | **The serializer fix reached only uw1 and uw1c.** The us1, ut1 and uw3 modules
   81 | had been generated from the pre-fix base, so they still rounded masses to ten
   82 | fixed decimals: a 1e-12 mass went to 0 and the target drifted by .399. Every
   83 | active solver, the two legacy shared-weight ones included, now serializes at
   84 | exact float64 precision, and the acceptance fixture is parameterized across all
   85 | seven.
   86 | 
   87 | **The pool digest hashed names, not content.** `train_pool_sha256` and
   88 | `dev_pool_sha256` were both `object_hash(sorted(outputs))` — the split NAMES —
   89 | so they were equal to each other and unmoved by the pool. They are now a digest
   90 | over each split's own rows (prompt id, role, occurrence index, candidate id,
   91 | response sha256); mutating one byte of one response changes it, and a change
   92 | outside the split does not. The hardcoded `panel: UF-4` is rewritten per panel
   93 | by the generator.
   94 | 
   95 | **The style-control length feature compared tokens with words.** The arm's
   96 | length came from its stored token count and the baseline's from
   97 | `len(text.split())`, so two identical answers differed by .44 on the feature the
   98 | control exists to remove. One frozen tokenizer now measures both sides, an
   99 | identical-text self-difference of zero is asserted before any data is read, and
  100 | the corrected numbers replace the ones computed on 2026-09-17. The fit reports
  101 | whether it converged and the bootstrap discards a replicate that did not; a
  102 | prompt enters only when both presentation orders parsed.
  103 | 
  104 | **The generator's own guard was a false positive.** It rejected any generated
  105 | module whose `--help` contained the substring `uw1`, which fires on `uw1c` —
  106 | correctly repointed — after the files had already been written. It now compares
  107 | whole tokens and exempts the panel being generated.
  108 | 

```

## E19 — Canonical validation of immutable all-pair data
Path: `mnpo_scripts/nbpo_neural.py`
SHA256: `b6df54c0771bc5d9b55092a117655fd08da609a19a25cdbeebbb2605a8b67ddc`
```text
  125 |             raise ValueError("Primary all-pair dataset requires N=8")
  126 |         wa, wb = float(row["nbpo_weight_a"]), float(row["nbpo_weight_b"])
  127 |         if min(wa, wb) <= 0 or not math.isfinite(wa + wb):
  128 |             raise ValueError("Canonical targets require strictly positive solver masses")
  129 |         center_a = float(row.get("nbpo_center_a", 1.0 / 8))
  130 |         center_b = float(row.get("nbpo_center_b", 1.0 / 8))
  131 |         if center_a != 1.0 / 8 or center_b != 1.0 / 8:
  132 |             raise ValueError("Primary IID occurrence center must be exactly 1/8")
  133 |         expected_target = math.log(wa / center_a) - math.log(wb / center_b)
  134 |         if abs(float(row["nbpo_logratio_target"]) - expected_target) > 1e-9:
  135 |             raise ValueError("Canonical target disagrees with log(p_star/p_t) candidate masses")
  136 |         tokens = immutable_pair_tokens(row, max_length, max_prompt_length)
  137 |         if tokens is None:
  138 |             raise ValueError("Canonical dataset requires immutable sampled tokens")
  139 |         key = str(row["prompt_id"])
  140 |         group = groups.setdefault(key, {"pairs": set(), "candidates": {}, "prompt": tokens["prompt_input_ids"]})
  141 |         if group["prompt"] != tokens["prompt_input_ids"]:
  142 |             raise ValueError("One prompt group has different conditioning token contexts")
  143 |         ids = [row.get(f"{side}_response_id", row.get(f"{side}_candidate_id"))
  144 |                for side in ("chosen", "rejected")]
  145 |         if None in ids or ids[0] == ids[1]:
  146 |             raise ValueError("Pair must identify two distinct sampled candidate occurrences")
  147 |         pair = tuple(sorted(map(str, ids)))
  148 |         if pair in group["pairs"]:
  149 |             raise ValueError("Duplicate unordered candidate pair")
  150 |         group["pairs"].add(pair)
  151 |         for side, candidate_id, weight_key in zip(("chosen", "rejected"), ids, ("nbpo_weight_a", "nbpo_weight_b")):
  152 |             mass = float(row[weight_key])
  153 |             if not math.isfinite(mass) or not 0 <= mass <= 1:
  154 |                 raise ValueError("Invalid solver probability mass")
  155 |             value = (tokens[f"{side}_token_sha256"], mass)
  156 |             old = group["candidates"].setdefault(str(candidate_id), value)
  157 |             if old != value:
  158 |                 raise ValueError("Candidate tokens or solver mass depend on pair partner")
  159 |         solver_hash = row.get("solver_artifact_sha256")
  160 |         if not solver_hash:
  161 |             raise ValueError("Canonical row lacks solver_artifact_sha256")
  162 |         if expected_solver_hash and solver_hash != expected_solver_hash:
  163 |             raise ValueError("Unexpected solver artifact hash")
  164 |         solvers.add(solver_hash)
  165 |     if not groups:
  166 |         raise ValueError("Empty canonical dataset")
  167 |     for group in groups.values():
  168 |         candidates = group["candidates"]
  169 |         if len(candidates) != 8 or group["pairs"] != set(itertools.combinations(sorted(candidates), 2)):
  170 |             raise ValueError("Every prompt requires all 28 unordered pairs of 8 candidates")
  171 |         if not math.isclose(sum(value[1] for value in candidates.values()), 1.0, abs_tol=1e-7):
  172 |             raise ValueError("Prompt solver probability masses do not sum to one")
  173 |     return {"prompts": len(groups), "rows": len(dataset), "candidates_per_prompt": 8,
  174 |             "unordered_pairs_per_prompt": 28, "solver_artifact_sha256": sorted(solvers)}

```

## E20 — Independent solution certificate
Path: `mnpo_scripts/nbpo_generic.py`
SHA256: `2b3502dbd78e6d3228df32b0830182f28bfc3c997dcf2e751a21a086226c2af5`
```text
  880 | def validate_finite_pool_solution(res: FinitePoolSolution, *, require_optimality=None,
  881 |                                  stationarity_tol=1e-4, extra_map_tol=1e-4,
  882 |                                  dual_tol=1e-6) -> dict:
  883 |     """One independent certificate for the CLI, stage runner and artifact writer.
  884 | 
  885 |     Recompute Q at the policy being serialized, separately from serialization
  886 |     consistency. Fixed training weights on held-out prompts carry no fitted
  887 |     held-out Nash-dual claim. Historical R-step artifacts remain identifiable
  888 |     approximations; they cannot be used as certified canonical teachers.
  889 |     """
  890 |     import math
  891 |     if require_optimality is None:
  892 |         require_optimality = res.config.get("inner_solver") == "exact"
  893 |     pi = validate_distribution(res.pi, "pi_star", require_full_support=True)
  894 |     pt = validate_distribution(res.pi_t, "pi_t", require_full_support=True)
  895 |     if pi.shape != pt.shape or res.target_log_ratio.shape != pi.shape:
  896 |         raise ValueError("canonical policy/center/target shapes disagree")
  897 |     if res.representation_object is None:
  898 |         raise ValueError("independent validation requires the original representation")
  899 |     nu, q = res.representation_object.opponent_and_gradient(pi)
  900 |     log_ratio = torch.log(pi) - torch.log(pt)
  901 |     serialization = float((res.target_log_ratio - log_ratio).abs().max())
  902 |     b = log_ratio - res.eta * torch.einsum("k,kxi->xi", res.weights, q)
  903 |     stationarity = float((b - (pi * b).sum(-1, keepdim=True)).abs().max())
  904 |     extra_map = float((exp_update(pt, q, res.weights, res.eta) - pi).abs().max())
  905 |     q_error = float((res.q_update - q).abs().max())
  906 |     v = res.representation_object.game_values(pi)
  907 |     surplus = v - res.representation_object.disagreement
  908 |     diagnostics = res.optimizer_diagnostics
  909 |     floor = float(diagnostics.get("probability_floor", 0.0))
  910 |     floor_active = bool(floor and bool((pi <= floor * (1.0 + 1e-6)).any()))
  911 |     certificate = {
  912 |         "canonical_serialization_error": serialization,
  913 |         "independent_stationarity_inf": stationarity,
  914 |         "extra_map_residual": extra_map, "q_at_returned_policy_error": q_error,
  915 |         "normalization_error": float((pi.sum(-1) - 1.0).abs().max()),
  916 |         "minimum_probability": float(pi.min()), "probability_floor": floor,
  917 |         "probability_floor_active": floor_active,
  918 |         "objective": float((res.weights * surplus).sum()) - proximal_divergence(pi, pt) / res.eta,
  919 |         "surplus": surplus.tolist(), "optimizer": diagnostics,
  920 |         "stationarity_units": "dimensionless_logratio_change",
  921 |         "tolerances": {"canonical_serialization_error": 1e-9,
  922 |                        "independent_stationarity_inf": stationarity_tol,
  923 |                        "extra_map_residual": extra_map_tol, "dual": dual_tol},
  924 |         "nash_dual_scope": ("fixed_training_weights" if res.config.get("fixed_weights")
  925 |                             else "training" if res.aggregation == "nash" else "not_applicable"),
  926 |     }
  927 |     problems = []
  928 |     if not math.isfinite(serialization) or serialization >= 1e-9:
  929 |         problems.append("canonical serialization error >= 1e-9")
  930 |     if not math.isfinite(stationarity) or stationarity >= stationarity_tol:
  931 |         problems.append("independent stationarity exceeds tolerance")
  932 |     if not math.isfinite(extra_map) or extra_map >= extra_map_tol:
  933 |         problems.append("extra-map residual exceeds tolerance")
  934 |     if floor_active:
  935 |         problems.append("probability floor is active")
  936 |     if diagnostics and not diagnostics.get("all_certified", True):
  937 |         problems.append("inner optimizer lacks a certified solution/refinement")
  938 |     if require_optimality and (not math.isfinite(q_error) or q_error >= 1e-9):
  939 |         problems.append("serialized Q does not match the returned policy")
  940 |     if require_optimality and (not torch.isfinite(res.nu_update).all() or
  941 |                               float((res.nu_update - nu).abs().max()) >= 1e-9):
  942 |         problems.append("serialized opponent does not match the returned policy")
  943 |     if not all(torch.isfinite(value).all() for value in (res.V, res.d, res.surplus)) or max(float((res.V - v).abs().max()),
  944 |            float((res.d - res.representation_object.disagreement).abs().max()),
  945 |            float((res.surplus - surplus).abs().max())) >= 1e-9:
  946 |         problems.append("stored V/d/s does not match the returned policy")
  947 |     if res.aggregation == "nash" and not res.config.get("fixed_weights"):
  948 |         lo, hi = res.config["lambda_box"]
  949 |         lower, upper = box_active_coordinates(res.weights, lo, hi)
  950 |         gradient = surplus - 1.0 / res.weights
  951 |         complementarity = res.weights * surplus - 1.0
  952 |         projected = projected_kkt_residual(res.weights, surplus, res.gamma_ref or 1.0, lo, hi)
  953 |         certificate.update({"dual_projected_residual": projected,
  954 |                             "dual_unprojected_gradient": gradient.tolist(),
  955 |                             "dual_unprojected_residual": float(gradient.abs().max()),
  956 |                             "lambda_times_surplus_minus_one": complementarity.tolist(),
  957 |                             "nash_complementarity_inf": float(complementarity.abs().max()),
  958 |                             "lambda_at_lower_bound": lower, "lambda_at_upper_bound": upper})
  959 |         if bool((surplus <= 0).any()):
  960 |             problems.append("Nash training surplus is not strictly positive")
  961 |         if projected >= dual_tol or float(gradient.abs().max()) >= dual_tol:
  962 |             problems.append("original Nash dual stationarity exceeds tolerance")
  963 |         if float(complementarity.abs().max()) >= dual_tol:
  964 |             problems.append("lambda * surplus - 1 exceeds tolerance")
  965 |     if res.aggregation == "kalai_smorodinsky" and res.ks:
  966 |         if res.ks.get("individual_rationality_violation", 0.0) > 1e-8:
  967 |             problems.append("KS individual rationality failed")
  968 |     certificate["certified"] = not problems
  969 |     certificate["failures"] = problems
  970 |     if require_optimality and problems:
  971 |         raise ValueError("finite-pool certificate failed: " + "; ".join(problems))
  972 |     return certificate
  973 | 
  974 | 
  975 | def _solve_nash_dual_by_root(rep, pi_t, d, lam0, eta, R, inner, damping, lo, hi,
  976 |                              *, tol, max_calls, history, log_every):
  977 |     """Solve the Nash dual as a ROOT problem instead of a subgradient descent.
  978 | 
  979 |     The dual stationarity condition of the Nash aggregation is exactly
  980 | 
  981 |         s_k(pi(lambda)) = 1 / lambda_k      for every k strictly inside the box,
  982 | 
  983 |     and ``pi(lambda)`` is a well-defined function once the inner subproblem is
  984 |     solved rather than iterated. So this is a K-dimensional root problem -- four
  985 |     dimensions here -- and a projected subgradient with a constant step is a poor
  986 |     way to attack it: on the controlled benchmark it stalls near a residual of
  987 |     1e-2 after 3000 outer iterations, nowhere near the 1e-6 gate.
  988 | 
  989 |     Solved in log-lambda coordinates so positivity is structural rather than
  990 |     enforced by clipping, warm-started across evaluations so the inner solves stay
  991 |     cheap, and reported with the actual number of inner solves used.
  992 |     """
  993 |     import numpy as np
  994 |     from scipy.optimize import root
  995 | 

```