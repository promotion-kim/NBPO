# b949919 code evidence

All line numbers refer to the uploaded new ZIP. No audited source file has been modified.

## E01 — Corrected three-block scoring and RR completeness

`analysis/sub_20260914/code_snapshot_20260917_union/union_score_panel.py:119-262`

SHA256: `c1e4c07bd5078678368037e566a3d11637b49b199660e53fd282ad1b04279710`

```text
  119 |     pool, pool_settings = load_pool(args.pool, args.pool_shards)
  120 |     status = Counter()
  121 |     # (pid, rubric, role_i, i, role_j, j) -> {order: value_for_i}
  122 |     obs = defaultdict(dict)
  123 |     sources = []
  124 |     for shard in range(args.judge_shards):
  125 |         d = SUB / "pool_judgments" / args.tag / ("shard%d" % shard)
  126 |         if not (d / "complete.json").exists():
  127 |             raise SystemExit("judging shard %d is not complete" % shard)
  128 |         for path in sorted(d.glob("chunk*.jsonl")):
  129 |             sources.append({"path": str(path), "sha256": file_hash(path)})
  130 |             with path.open() as stream:
  131 |                 for line in stream:
  132 |                     r = json.loads(line)
  133 |                     status[r["status"]] += 1
  134 |                     if r["status"] != "ok":
  135 |                         continue
  136 |                     key = (r["prompt_id"], r["rubric"], r["role_i"], r["i"],
  137 |                            r["role_j"], r["j"])
  138 |                     obs[key][r["order"]] = float(r["value_for_i"])
  139 | 
  140 |     learner_pairs = list(itertools.combinations(range(POOL), 2))
  141 |     cross_pairs = [(i, j) for i in range(POOL) for j in range(POOL)]
  142 | 
  143 |     def resolve(pid, rubric, role_i, i, role_j, j):
  144 |         """Order-balanced probability, or None when an order is missing."""
  145 |         got = obs.get((pid, rubric, role_i, i, role_j, j))
  146 |         if got is None or 0 not in got or 1 not in got:
  147 |             return None
  148 |         return 0.5 * (got[0] + got[1])
  149 | 
  150 |     tensors, identity_ties, refdis = {}, Counter(), {}
  151 |     dropped = Counter()
  152 |     prompts = sorted({k[0] for k in obs})
  153 |     for pid in prompts:
  154 |         if pid not in pool:
  155 |             dropped["prompt_absent_from_pool"] += 1
  156 |             continue
  157 |         A_LL = np.zeros((K, POOL, POOL))
  158 |         A_LR = np.zeros((K, POOL, POOL))
  159 |         A_RR = np.zeros((K, POOL, POOL))
  160 |         ok = True
  161 |         for k, rubric in enumerate(ITEMS):
  162 |             for i, j in learner_pairs:
  163 |                 p = resolve(pid, rubric, "learner", i, "learner", j)
  164 |                 if p is None:
  165 |                     ok = False; dropped["learner_pair_unresolved"] += 1; break
  166 |                 A_LL[k, i, j] = p - 0.5
  167 |                 A_LL[k, j, i] = 0.5 - p
  168 |             if not ok:
  169 |                 break
  170 |             for i, j in cross_pairs:
  171 |                 p = resolve(pid, rubric, "learner", i, "comparator", j)
  172 |                 if p is None:
  173 |                     ok = False; dropped["cross_pair_unresolved"] += 1; break
  174 |                 if (pool[pid]["learner"].get(i) ==
  175 |                         pool[pid]["comparator"].get(j) is not None):
  176 |                     identity_ties[pid] += 1
  177 |                 A_LR[k, i, j] = p - 0.5
  178 |             if not ok:
  179 |                 break
  180 |             # the reference triangle is now required: d = V_beta(mu) is defined
  181 |             # on it, so a prompt without it has no disagreement point and is
  182 |             # dropped rather than carried with a silently wrong d
  183 |             for i, j in learner_pairs:
  184 |                 p = resolve(pid, rubric, "comparator", i, "comparator", j)
  185 |                 if p is None:
  186 |                     ok = False; dropped["reference_pair_unresolved"] += 1; break
  187 |                 A_RR[k, i, j] = p - 0.5
  188 |                 A_RR[k, j, i] = 0.5 - p
  189 |             if not ok:
  190 |                 break
  191 |         if not ok:
  192 |             continue
  193 |         idx = np.arange(POOL)
  194 |         A_LL[:, idx, idx] = 0.0
  195 |         A_RR[:, idx, idx] = 0.0
  196 |         for name, M in (("learner", A_LL), ("reference", A_RR)):
  197 |             if np.abs(M + np.swapaxes(M, -1, -2)).max() > 1e-12:
  198 |                 raise SystemExit("%s payoff not antisymmetric for %s" % (name, pid))
  199 |         if max(np.abs(A_LL).max(), np.abs(A_LR).max(),
  200 |                np.abs(A_RR).max()) > 0.5 + 1e-12:
  201 |             raise SystemExit("payoff outside [-0.5, 0.5] for %s" % pid)
  202 |         # the same scalar the old file reported, so the two rounds stay comparable
  203 |         iu = np.triu_indices(POOL, 1)
  204 |         refdis[pid] = float(np.mean(np.abs(A_RR[:, iu[0], iu[1]])))
  205 |         tensors[pid] = (A_LL, A_LR, A_RR)
  206 | 
  207 |     pids = sorted(tensors)
  208 |     if not pids:
  209 |         raise SystemExit("no prompt survived scoring")
  210 |     out = UF / "scores" / args.out
  211 |     out.mkdir(parents=True, exist_ok=True)
  212 |     teacher = {
  213 |         "judge": "/work/uf4_20260910/assets/Qwen3-14B",
  214 |         "judge_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
  215 |         "judgment_tag": args.tag,
  216 |         "template": "PROSPER Figure 4 PSC five-point single-check",
  217 |         "scale": "verdict in {0..4} for the FIRST response; p = verdict/4",
  218 |         "draws_per_order": 1,
  219 |         "orders": "both, swapped verdict reversed before averaging",
  220 |         "graph": "cross 8x8 + learner C(8,2) + reference C(8,2) per item",
  221 |         "items_per_prompt": K,
  222 |     }
  223 |     per = (len(pids) + args.shards - 1) // args.shards
  224 |     written = []
  225 |     for s in range(args.shards):
  226 |         chunk = pids[s * per:(s + 1) * per]
  227 |         if not chunk:
  228 |             continue
  229 |         A_LL = np.stack([tensors[p][0] for p in chunk], axis=1)
  230 |         A_LR = np.stack([tensors[p][1] for p in chunk], axis=1)
  231 |         A_RR = np.stack([tensors[p][2] for p in chunk], axis=1)
  232 |         d = out / ("shard%d" % s)
  233 |         d.mkdir(parents=True, exist_ok=True)
  234 |         path = d / "chunk0000.npz"
  235 |         # A_policy and A_ref are the names the solver reads. They are aliases of
  236 |         # A_LR and A_RR, written explicitly so an existing reader gets the
  237 |         # paper's game without being changed, while A_LL travels under its own
  238 |         # name for the pair-label consumers.
  239 |         np.savez(path, prompt_ids=np.array(chunk),
  240 |                  A_policy=A_LR, A_ref=A_RR,
  241 |                  A_LL=A_LL, A_LR=A_LR, A_RR=A_RR)
  242 |         (d / "chunk0000.manifest.json").write_text(json.dumps(
  243 |             {"prompts": len(chunk), "sha256": file_hash(path),
  244 |              "shapes": {"A_policy": list(A_LR.shape), "A_ref": list(A_RR.shape),
  245 |                         "A_LL": list(A_LL.shape), "A_LR": list(A_LR.shape),
  246 |                         "A_RR": list(A_RR.shape)},
  247 |              "roles": {"A_policy": "A_LR (learner vs reference)",
  248 |                        "A_ref": "A_RR (reference vs reference)"}},
  249 |             indent=1) + "\n")
  250 |         (d / ("complete_shard%d.json" % s)).write_text(json.dumps(
  251 |             {"shard": s, "prompts": len(chunk), "gpm_teacher": teacher,
  252 |              "bt_teacher": ("absent: the union contract uses direct order-balanced PSC "
  253 |                             "probabilities for these rows and forbids a scalar BT "
  254 |                             "projection, so none was fitted and none is written"),
  255 |              "reference_construction": ("independent reference bank: eight reference "
  256 |                                         "occurrences per prompt, so A_policy is the "
  257 |                                         "learner-by-reference cross block and A_ref is "
  258 |                                         "the reference triangle, not a copy of it")},
  259 |             indent=1) + "\n")
  260 |         written.append({"shard": s, "prompts": len(chunk),
  261 |                         "sha256": file_hash(path),
  262 |                         "shapes": {"A_policy": list(A_LR.shape),
```

## E02 — Baseline-label loader uses LL and rejects obsolete schema

`analysis/sub_20260914/code_snapshot_20260917_union/build_panel_softlabels.py:1-58`

SHA256: `567ef4618851beb061bc7e5911a88a2f9dcd5a4d5eed9b58e5b519be1aff7d2b`

```text
    1 | """Attach order-balanced soft preference labels to a panel's pair dataset.
    2 | 
    3 | The scalarized-DPO and mean-preference-INPO rows of Table 3 need, for each
    4 | training pair, the probability that the chosen response is preferred to the
    5 | rejected one under the SAME judgments the NBPO teacher used. That probability is
    6 | the panel's objective-averaged direct PSC probability
    7 | 
    8 |     p = (1/K) sum_k [ A_LL[k, x, i, j] + 1/2 ],
    9 | 
   10 | where A_LL is the antisymmetric, zero-diagonal learner block written by the
   11 | panel scorer, i and j are the row's own candidate indices, and the +1/2 turns
   12 | the antisymmetric deviation back into a probability. No Bradley-Terry fit and no
   13 | surrogate reward model enters: these are the measured order-averaged verdicts.
   14 | 
   15 | The tokenization, prompts and pair set are copied unchanged from the NBPO-PW
   16 | dataset of the same panel, so the two baselines see exactly the pairs the NBPO
   17 | arms see. Only the two probability columns are added.
   18 | 
   19 | Written with the trainer's own datasets version (deps_train first on PYTHONPATH)
   20 | because a newer writer emits a feature type the trainer's reader rejects.
   21 | """
   22 | from __future__ import annotations
   23 | 
   24 | import argparse, glob, hashlib, json
   25 | from pathlib import Path
   26 | 
   27 | import numpy as np
   28 | 
   29 | UF = Path("/work/uf4_20260910")
   30 | 
   31 | 
   32 | def load_policy_block(scores_dir, shards):
   33 |     """prompt_id -> A_LL, the antisymmetric learner block."""
   34 |     out = {}
   35 |     for shard in range(shards):
   36 |         d = Path(scores_dir) / ("shard%d" % shard)
   37 |         for path in sorted(d.glob("chunk*.npz")):
   38 |             z = np.load(path, allow_pickle=True)
   39 |             pids = [str(v) for v in z["prompt_ids"]]
   40 |             # A_LL, explicitly: the pair label is a learner-versus-learner
   41 |             # probability, and since the A01 fix A_policy carries the
   42 |             # learner-versus-reference cross block instead. A score directory
   43 |             # written before that fix has no A_LL, and must be re-scored rather
   44 |             # than read here, so its absence raises instead of falling back.
   45 |             if "A_LL" not in z.files:
   46 |                 raise SystemExit(
   47 |                     "%s predates the A01 tensor-role fix: it has no A_LL, and its "
   48 |                     "A_policy is the learner triangle only by coincidence of the old "
   49 |                     "wiring. Re-score this panel with union_score_panel.py before "
   50 |                     "building soft labels." % path)
   51 |             A = z["A_LL"]
   52 |             for x, pid in enumerate(pids):
   53 |                 out[pid] = np.asarray(A[:, x], dtype=np.float64)
   54 |     return out
   55 | 
   56 | 
   57 | def main():
   58 |     ap = argparse.ArgumentParser(description=__doc__)
```

## E03 — Current base still imports absent canonical APIs

`analysis/sub_20260914/code_snapshot_20260917_union/solve_pros4_targets_uw1c.py:34-59`

SHA256: `07d7c7567a56781fff5777b65b7375b9b2828feb633916637a926bf3c0b601b5`

```text
   34 | """
   35 | from __future__ import annotations
   36 | 
   37 | import argparse
   38 | import hashlib
   39 | import json
   40 | import os
   41 | import subprocess
   42 | import sys
   43 | import time
   44 | from pathlib import Path
   45 | 
   46 | import numpy as np
   47 | import torch
   48 | 
   49 | from mnpo_scripts.nbpo_core import uniform_policy
   50 | from mnpo_scripts.nbpo_generic import solve_finite_pool, validate_finite_pool_solution
   51 | from mnpo_scripts.nbpo_representations import (
   52 |     AdaptiveGameRepresentation, BTRewardRepresentation, FixedReferenceRepresentation,
   53 | )
   54 | from scripts.nbpo.build_nbpo_pairs import build_rows, load_canonical_artifact
   55 | from scripts.nbpo.solve_nbpo_dual import write_generic_solution_artifact
   56 | 
   57 | ROOT = Path("/work/uf4_20260910")
   58 | OBJECTIVES = ("item0", "item1", "item2", "item3")
   59 | REFERENCE_CONSTRUCTION = "independent_samples"
```

## E04 — Released pair builder lacks canonical mode and canonical_data

`scripts/nbpo/build_nbpo_pairs.py:34-125`

SHA256: `2609b091e2c7fd9c7911c9119a3fdf49d757779a7fd5d2017f8f10b1073134a4`

```text
   34 | 
   35 | import numpy as np
   36 | 
   37 | from scripts.nbpo.nbpo_common import (
   38 |     implementation_contract,
   39 |     load_response_files,
   40 |     response_pool_hash,
   41 |     sha256_file,
   42 |     sha256_text,
   43 |     write_json,
   44 | )
   45 | 
   46 | TARGET_MODES = ("sampled", "rao_blackwell")
   47 | 
   48 | 
   49 | def _array_sha256(arr) -> str:
   50 |     """sha256 of an array's exact float64 bytes -- matches the solver's hash."""
   51 |     import hashlib
   52 | 
   53 |     return hashlib.sha256(
   54 |         np.ascontiguousarray(np.asarray(arr), dtype=np.float64).tobytes()).hexdigest()
   55 | 
   56 | 
   57 | def flip_pair_row(row: dict) -> dict:
   58 |     """Swap the pair orientation, flipping every Z_k and the aggregate target."""
   59 |     flipped = dict(row)
   60 |     flipped["chosen"], flipped["rejected"] = row["rejected"], row["chosen"]
   61 |     flipped["chosen_response_id"], flipped["rejected_response_id"] = (
   62 |         row["rejected_response_id"], row["chosen_response_id"])
   63 |     flipped["nbpo_z"] = {k: -v for k, v in row["nbpo_z"].items()}
   64 |     flipped["nbpo_weighted_z"] = -row["nbpo_weighted_z"]
   65 |     return flipped
   66 | 
   67 | 
   68 | def build_rows(prompt_ids, objectives, A_policy, nu, lam, betas, policy, ref_seed_of,
   69 |                rng, target_mode, meta_ids, provenance):
   70 |     """One row per (prompt, unordered learner pair); deterministic given the RNG."""
   71 |     K = len(objectives)
   72 |     I = A_policy.shape[2]
   73 |     policy_ids = meta_ids["policy_learner_ids"]
   74 |     comparator_ids = meta_ids["comparator_ids"]
   75 |     rows = []
   76 |     for x, pid in enumerate(prompt_ids):
   77 |         seed0 = policy_ids[0].split(":", 1)[1]
   78 |         prompt_text = str(policy[seed0][pid]["prompt"])
   79 |         for i1, i2 in itertools.combinations(range(I), 2):
   80 |             z, opp = {}, {}
   81 |             for k, obj in enumerate(objectives):
   82 |                 # Eq. (26): draw (y, y') first, THEN one z_k ~ nu*_k for this pair and
   83 |                 # objective. The same z_k serves both y and y' of the row; other rows
   84 |                 # of the same prompt and other objectives draw independently.
   85 |                 j = int(rng.choice(len(comparator_ids), p=nu[k, x]))
   86 |                 p1 = float(A_policy[k, x, i1, j]) + 0.5
   87 |                 p2 = float(A_policy[k, x, i2, j]) + 0.5
   88 |                 if target_mode == "sampled":
   89 |                     b1 = float(rng.random() < p1)
   90 |                     b2 = float(rng.random() < p2)
   91 |                     z[obj] = b1 - b2
   92 |                 else:  # rao_blackwell
   93 |                     z[obj] = p1 - p2
   94 |                 opp[obj] = comparator_ids[j]
   95 |             id1, id2 = policy_ids[i1], policy_ids[i2]
   96 |             rows.append({
   97 |                 "prompt_id": pid,
   98 |                 "prompt": prompt_text,
   99 |                 "chosen": str(policy[id1.split(":", 1)[1]][pid]["generated_text"]),
  100 |                 "rejected": str(policy[id2.split(":", 1)[1]][pid]["generated_text"]),
  101 |                 "chosen_response_id": id1,
  102 |                 "rejected_response_id": id2,
  103 |                 # Response IDS are not response TEXT: these pin the exact strings
  104 |                 # this row was built from, so a later pool swap is detectable.
  105 |                 "chosen_text_sha256": sha256_text(
  106 |                     str(policy[id1.split(":", 1)[1]][pid]["generated_text"])),
  107 |                 "rejected_text_sha256": sha256_text(
  108 |                     str(policy[id2.split(":", 1)[1]][pid]["generated_text"])),
  109 |                 "nbpo_z": z,
  110 |                 "nbpo_weighted_z": float(sum(lam[k] * z[obj] for k, obj in enumerate(objectives))),
  111 |                 "lambda_raw": {obj: float(lam[k]) for k, obj in enumerate(objectives)},
  112 |                 "opponent_response_id": opp,
  113 |                 "opponent_beta": {obj: float(betas[k]) for k, obj in enumerate(objectives)},
  114 |                 "opponent_sampling_scope": "pair_objective",
  115 |                 "target_mode": target_mode,
  116 |                 **provenance,
  117 |             })
  118 |     return rows
  119 | 
  120 | 
  121 | def verify_solver_input_chain(tensor_dir: Path, solver_dir: Path, solution: dict,
  122 |                               reproduction_mode_required: bool = True) -> dict:
  123 |     """Prove the tensors on disk ARE the solver's inputs, and the opponents its outputs.
  124 | 
  125 |     Verifying only the ``nu`` hashes left the biggest hole open: nothing showed
```

## E05 — Corrected float64 serializer

`analysis/sub_20260914/code_snapshot_20260917_union/solve_pros4_targets_uw1c.py:89-135`

SHA256: `07d7c7567a56781fff5777b65b7375b9b2828feb633916637a926bf3c0b601b5`

```text
   89 | # Kept for readers of older artifacts: rows written before the A05 fix carry
   90 | # canonical_target_quantized_decimals = 10 and their masses were rounded to that
   91 | # many fixed decimals. New rows are written at exact float64 precision and
   92 | # record canonical_target_serialization instead.
   93 | CANONICAL_DECIMALS_LEGACY = 10
   94 | 
   95 | 
   96 | def quantize_canonical_row(row):
   97 |     """Serialize the solver masses so they survive the round trip exactly.
   98 | 
   99 |     prepare_nbpo_dataset reads these rows with a standard JSON parser and then
  100 |     checks target == log(w_a/c_a) - log(w_b/c_b) to 1e-9 absolute. An earlier
  101 |     version of this function rounded both masses to ten FIXED DECIMALS and
  102 |     rebuilt the target from the rounded values, which made the identity hold
  103 |     after the parse but changed the certified solution: a per-prompt Nash solve
  104 |     concentrates mass, its smallest masses reach the 1e-12 probability floor,
  105 |     and 1e-12 rounded to ten decimals is exactly 0. The row then carries a
  106 |     non-positive mass, which the canonical validator rejects outright, and any
  107 |     mass between 1e-10 and 1e-12 that did survive was quantized to a value whose
  108 |     log differs from the certified one by far more than the solver's own
  109 |     residual. That is the audit's A05: a target changed after it was certified.
  110 | 
  111 |     Fixed decimals are the wrong instrument for a quantity that spans twelve
  112 |     orders of magnitude. Python's float repr round-trips a float64 exactly and
  113 |     json.dumps emits it, so no rounding is needed at all: the masses are written
  114 |     as they were solved and the target is rebuilt from those same doubles. The
  115 |     identity then holds to the float64 relative error of a logarithm, roughly
  116 |     1e-16, comfortably inside the 1e-9 gate, with the certified solution intact.
  117 | 
  118 |     A mass that is genuinely non-positive is left alone here and refused by the
  119 |     validator, which is the correct outcome: it means the inner solve hit the
  120 |     boundary and the canonical log-ratio does not exist for that pair.
  121 |     """
  122 |     import math
  123 | 
  124 |     for key in ("nbpo_weight_a", "nbpo_weight_b"):
  125 |         if key in row:
  126 |             row[key] = float(row[key])
  127 |     if row.get("target_mode") == "canonical_logratio" and "nbpo_weight_a" in row:
  128 |         wa, wb = float(row["nbpo_weight_a"]), float(row["nbpo_weight_b"])
  129 |         ca = float(row.get("nbpo_center_a", 1.0 / POOL))
  130 |         cb = float(row.get("nbpo_center_b", 1.0 / POOL))
  131 |         if min(wa, wb) > 0:
  132 |             row["nbpo_logratio_target"] = math.log(wa / ca) - math.log(wb / cb)
  133 |             row["canonical_target_quantized_decimals"] = None
  134 |             row["canonical_target_serialization"] = "exact_float64_round_trip"
  135 |     return row
```

## E06 — New uw3 still uses fixed-decimal serializer

`analysis/sub_20260914/code_snapshot_20260917_union/solve_pros4_targets_uw3.py:79-116`

SHA256: `c4f3b3100341275b708d9a6dc760f9c1f9f55eeab414e0f3e6a36cb2b5ed6fe1`

```text
   79 | 
   80 | 
   81 | def write_jsonl(path, rows):
   82 |     path = Path(path)
   83 |     path.parent.mkdir(parents=True, exist_ok=True)
   84 |     with path.open("x") as stream:
   85 |         for row in rows:
   86 |             stream.write(json.dumps(row, ensure_ascii=False) + "\n")
   87 | 
   88 | 
   89 | CANONICAL_DECIMALS = 10
   90 | 
   91 | 
   92 | def quantize_canonical_row(row):
   93 |     """Round the solver masses to the loader's precision and rebuild the target.
   94 | 
   95 |     prepare_nbpo_dataset reads these rows through a JSON parser that keeps ten
   96 |     decimal places, then checks target == log(w_a/c_a) - log(w_b/c_b) to 1e-9.
   97 |     Rounding the masses first and deriving the target from the rounded values
   98 |     makes the row consistent under that parse instead of only before it. A
   99 |     per-prompt Nash solve concentrates mass, so without this the identity
  100 |     survives in memory and fails after the parse.
  101 |     """
  102 |     import math
  103 | 
  104 |     for key in ("nbpo_weight_a", "nbpo_weight_b"):
  105 |         if key in row:
  106 |             row[key] = round(float(row[key]), CANONICAL_DECIMALS)
  107 |     if row.get("target_mode") == "canonical_logratio" and "nbpo_weight_a" in row:
  108 |         wa, wb = float(row["nbpo_weight_a"]), float(row["nbpo_weight_b"])
  109 |         ca = float(row.get("nbpo_center_a", 1.0 / POOL))
  110 |         cb = float(row.get("nbpo_center_b", 1.0 / POOL))
  111 |         if min(wa, wb) > 0:
  112 |             row["nbpo_logratio_target"] = round(
  113 |                 math.log(wa / ca) - math.log(wb / cb), CANONICAL_DECIMALS)
  114 |             row["canonical_target_quantized_decimals"] = CANONICAL_DECIMALS
  115 |     return row
  116 | 
```

## E07 — SafeRLHF serializer still quantizes

`analysis/sub_20260914/code_snapshot_20260917_union/solve_pros4_targets_us1.py:79-116`

SHA256: `841d8e1587384b7cadc8c6f191d73bfc6c11f757bcf850167e976e123f649068`

```text
   79 | 
   80 | 
   81 | def write_jsonl(path, rows):
   82 |     path = Path(path)
   83 |     path.parent.mkdir(parents=True, exist_ok=True)
   84 |     with path.open("x") as stream:
   85 |         for row in rows:
   86 |             stream.write(json.dumps(row, ensure_ascii=False) + "\n")
   87 | 
   88 | 
   89 | CANONICAL_DECIMALS = 10
   90 | 
   91 | 
   92 | def quantize_canonical_row(row):
   93 |     """Round the solver masses to the loader's precision and rebuild the target.
   94 | 
   95 |     prepare_nbpo_dataset reads these rows through a JSON parser that keeps ten
   96 |     decimal places, then checks target == log(w_a/c_a) - log(w_b/c_b) to 1e-9.
   97 |     Rounding the masses first and deriving the target from the rounded values
   98 |     makes the row consistent under that parse instead of only before it. A
   99 |     per-prompt Nash solve concentrates mass, so without this the identity
  100 |     survives in memory and fails after the parse.
  101 |     """
  102 |     import math
  103 | 
  104 |     for key in ("nbpo_weight_a", "nbpo_weight_b"):
  105 |         if key in row:
  106 |             row[key] = round(float(row[key]), CANONICAL_DECIMALS)
  107 |     if row.get("target_mode") == "canonical_logratio" and "nbpo_weight_a" in row:
  108 |         wa, wb = float(row["nbpo_weight_a"]), float(row["nbpo_weight_b"])
  109 |         ca = float(row.get("nbpo_center_a", 1.0 / POOL))
  110 |         cb = float(row.get("nbpo_center_b", 1.0 / POOL))
  111 |         if min(wa, wb) > 0:
  112 |             row["nbpo_logratio_target"] = round(
  113 |                 math.log(wa / ca) - math.log(wb / cb), CANONICAL_DECIMALS)
  114 |             row["canonical_target_quantized_decimals"] = CANONICAL_DECIMALS
  115 |     return row
  116 | 
```

## E08 — TLDR serializer still quantizes

`analysis/sub_20260914/code_snapshot_20260917_union/solve_pros4_targets_ut1.py:79-116`

SHA256: `ea3fc9e20f1599a622ef30fc442e0f554364daa5e5c103a6713b8fc47f331872`

```text
   79 | 
   80 | 
   81 | def write_jsonl(path, rows):
   82 |     path = Path(path)
   83 |     path.parent.mkdir(parents=True, exist_ok=True)
   84 |     with path.open("x") as stream:
   85 |         for row in rows:
   86 |             stream.write(json.dumps(row, ensure_ascii=False) + "\n")
   87 | 
   88 | 
   89 | CANONICAL_DECIMALS = 10
   90 | 
   91 | 
   92 | def quantize_canonical_row(row):
   93 |     """Round the solver masses to the loader's precision and rebuild the target.
   94 | 
   95 |     prepare_nbpo_dataset reads these rows through a JSON parser that keeps ten
   96 |     decimal places, then checks target == log(w_a/c_a) - log(w_b/c_b) to 1e-9.
   97 |     Rounding the masses first and deriving the target from the rounded values
   98 |     makes the row consistent under that parse instead of only before it. A
   99 |     per-prompt Nash solve concentrates mass, so without this the identity
  100 |     survives in memory and fails after the parse.
  101 |     """
  102 |     import math
  103 | 
  104 |     for key in ("nbpo_weight_a", "nbpo_weight_b"):
  105 |         if key in row:
  106 |             row[key] = round(float(row[key]), CANONICAL_DECIMALS)
  107 |     if row.get("target_mode") == "canonical_logratio" and "nbpo_weight_a" in row:
  108 |         wa, wb = float(row["nbpo_weight_a"]), float(row["nbpo_weight_b"])
  109 |         ca = float(row.get("nbpo_center_a", 1.0 / POOL))
  110 |         cb = float(row.get("nbpo_center_b", 1.0 / POOL))
  111 |         if min(wa, wb) > 0:
  112 |             row["nbpo_logratio_target"] = round(
  113 |                 math.log(wa / ca) - math.log(wb / cb), CANONICAL_DECIMALS)
  114 |             row["canonical_target_quantized_decimals"] = CANONICAL_DECIMALS
  115 |     return row
  116 | 
```

## E09 — Corrected scorer not propagated to other callable scorer

`analysis/sub_20260914/code_snapshot_20260917_union/union_score_uw.py:114-179`

SHA256: `aefb140d0af0c027bffe4cb4ce805b195af77cd72b90496a6b43453a1b8bac82`

```text
  114 |             dropped["prompt_absent_from_pool"] += 1
  115 |             continue
  116 |         A = np.zeros((K, POOL, POOL))
  117 |         R = np.zeros((K, POOL, POOL))
  118 |         ok = True
  119 |         for k, rubric in enumerate(ITEMS):
  120 |             for i, j in learner_pairs:
  121 |                 p = resolve(pid, rubric, "learner", i, "learner", j)
  122 |                 if p is None:
  123 |                     ok = False; dropped["learner_pair_unresolved"] += 1; break
  124 |                 A[k, i, j] = p - 0.5
  125 |                 A[k, j, i] = 0.5 - p
  126 |             if not ok:
  127 |                 break
  128 |             for i, j in cross_pairs:
  129 |                 p = resolve(pid, rubric, "learner", i, "comparator", j)
  130 |                 if p is None:
  131 |                     ok = False; dropped["cross_pair_unresolved"] += 1; break
  132 |                 if (pool[pid]["learner"].get(i) ==
  133 |                         pool[pid]["comparator"].get(j) is not None):
  134 |                     identity_ties[pid] += 1
  135 |                 R[k, i, j] = p - 0.5
  136 |             if not ok:
  137 |                 break
  138 |         if not ok:
  139 |             continue
  140 |         idx = np.arange(POOL)
  141 |         A[:, idx, idx] = 0.0
  142 |         if np.abs(A + np.swapaxes(A, -1, -2)).max() > 1e-12:
  143 |             raise SystemExit("learner payoff not antisymmetric for %s" % pid)
  144 |         if max(np.abs(A).max(), np.abs(R).max()) > 0.5 + 1e-12:
  145 |             raise SystemExit("payoff outside [-0.5, 0.5] for %s" % pid)
  146 |         # reference disagreement: the reference triangle, kept as a diagnostic
  147 |         vals = []
  148 |         for k, rubric in enumerate(ITEMS):
  149 |             for i, j in learner_pairs:
  150 |                 p = resolve(pid, rubric, "comparator", i, "comparator", j)
  151 |                 if p is not None:
  152 |                     vals.append(abs(p - 0.5))
  153 |         refdis[pid] = float(np.mean(vals)) if vals else None
  154 |         tensors[pid] = (A, R)
  155 | 
  156 |     pids = sorted(tensors)
  157 |     if not pids:
  158 |         raise SystemExit("no prompt survived scoring")
  159 |     out = UF / "scores" / args.out
  160 |     out.mkdir(parents=True, exist_ok=True)
  161 |     teacher = {
  162 |         "judge": "/work/uf4_20260910/assets/Qwen3-14B",
  163 |         "judge_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
  164 |         "judgment_tag": args.tag,
  165 |         "template": "PROSPER Figure 4 PSC five-point single-check",
  166 |         "scale": "verdict in {0..4} for the FIRST response; p = verdict/4",
  167 |         "draws_per_order": 1,
  168 |         "orders": "both, swapped verdict reversed before averaging",
  169 |         "graph": "cross 8x8 + learner C(8,2) + reference C(8,2) per item",
  170 |         "items_per_prompt": K,
  171 |     }
  172 |     per = (len(pids) + args.shards - 1) // args.shards
  173 |     written = []
  174 |     for s in range(args.shards):
  175 |         chunk = pids[s * per:(s + 1) * per]
  176 |         if not chunk:
  177 |             continue
  178 |         A = np.stack([tensors[p][0] for p in chunk], axis=1)
  179 |         R = np.stack([tensors[p][1] for p in chunk], axis=1)
```

## E10 — New solver loader validates hash but not tensor-role schema

`analysis/sub_20260914/code_snapshot_20260917_union/solve_pros4_targets_uw1c.py:138-180`

SHA256: `07d7c7567a56781fff5777b65b7375b9b2828feb633916637a926bf3c0b601b5`

```text
  138 | def load_scores(score_root, shards):
  139 |     """prompt_id -> (A_policy[K,8,8], A_ref[K,8,8], r_bt[K,2,8]), hash-verified.
  140 | 
  141 |     r_bt is the frozen BT head's scalar reward for the eight learner and eight
  142 |     comparator occurrences. It is loaded for every representation but only the
  143 |     bt_reward representation reads it, so nothing else changes.
  144 |     """
  145 |     scores, manifests = {}, []
  146 |     for shard in range(shards):
  147 |         directory = Path(score_root) / f"shard{shard}"
  148 |         complete = json.loads((directory / f"complete_shard{shard}.json").read_text())
  149 |         manifests.append({"shard": shard, "gpm_teacher": complete["gpm_teacher"],
  150 |                           "bt_teacher": complete["bt_teacher"],
  151 |                           "reference_construction": complete["reference_construction"]})
  152 |         for path in sorted(directory.glob("chunk*.npz")):
  153 |             meta = json.loads(path.with_suffix("").with_suffix(".manifest.json").read_text()) \
  154 |                 if path.with_suffix("").with_suffix(".manifest.json").exists() else \
  155 |                 json.loads((directory / (path.stem + ".manifest.json")).read_text())
  156 |             if file_hash(path) != meta["sha256"]:
  157 |                 raise ValueError(f"Score chunk hash mismatch: {path}")
  158 |             arrays = np.load(path, allow_pickle=True)
  159 |             pids = [str(p) for p in arrays["prompt_ids"]]
  160 |             A, Aref = arrays["A_policy"], arrays["A_ref"]
  161 |             # r_bt is the scalar BT projection. The union contract uses direct
  162 |             # order-balanced probabilities for these rows and forbids that
  163 |             # projection, so the array may be absent; it is left as None rather
  164 |             # than filled with zeros, and only the bt_reward representation
  165 |             # reads it, where None raises instead of training on a placeholder.
  166 |             Rbt = arrays["r_bt"] if "r_bt" in arrays.files else None
  167 |             if A.shape[0] != len(OBJECTIVES) or A.shape[2:] != (POOL, POOL):
  168 |                 raise ValueError(f"Unexpected score tensor shape in {path}: {A.shape}")
  169 |             if Rbt is not None and (Rbt.shape[0] != len(OBJECTIVES)
  170 |                                     or Rbt.shape[2:] != (2, POOL)):
  171 |                 raise ValueError(f"Unexpected BT reward shape in {path}: {Rbt.shape}")
  172 |             for index, pid in enumerate(pids):
  173 |                 if pid in scores:
  174 |                     raise ValueError(f"Duplicate scored prompt {pid}")
  175 |                 scores[pid] = (A[:, index], Aref[:, index],
  176 |                                None if Rbt is None else Rbt[:, index])
  177 |     identities = {object_hash(m["gpm_teacher"]) for m in manifests}
  178 |     if len(identities) != 1:
  179 |         raise ValueError("Score shards disagree on the frozen GPM teacher")
  180 |     return scores, manifests
```

## E11 — New local wrapper, import selection and exact/root settings

`analysis/sub_20260914/code_snapshot_20260917_union/solve_pros4_pw_nbpo_uw1c.py:39-121`

SHA256: `edfc8c47f092ab73509cdefa7896066fd95198238ba4bc718215ebf0688b0d72`

```text
   39 | """
   40 | from __future__ import annotations
   41 | 
   42 | import argparse, json, os, sys, time
   43 | from concurrent.futures import ProcessPoolExecutor
   44 | from pathlib import Path
   45 | 
   46 | import numpy as np
   47 | import torch
   48 | 
   49 | sys.path.insert(0, "/work/uf4_20260910/code")
   50 | import solve_pros4_targets_uw1c as base                      # loaders, hashing, writers
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
  105 |                     default=None)
  106 |     ap.add_argument("--beta", type=float, default=0.25)
  107 |     ap.add_argument("--eta", type=float, default=1.0)
  108 |     ap.add_argument("--workers", type=int, default=32)
  109 |     ap.add_argument("--max-dual-calls", type=int, default=200)
  110 |     ap.add_argument("--probability-floor", type=float, default=1e-12)
  111 |     ap.add_argument("--limit", type=int, default=0,
  112 |                     help="pilot on the first N prompts of each split; 0 means all")
  113 |     ap.add_argument("--skip-dataset", action="store_true")
  114 |     args = ap.parse_args()
  115 | 
  116 |     # nash derives its own dual: no multiplier norm is read, and none is used.
  117 |     # The global NBPO arm this was matched to is uncertified (independent
  118 |     # stationarity 13.224), and nothing here is matched to an uncertified value.
  119 |     weight_l1 = None
  120 |     print(json.dumps({"weight_l1": None,
  121 |                       "note": "unused: nash derives its own dual"}), flush=True)
```

## E12 — Canonical builder call and unchanged split-name pool hashes

`analysis/sub_20260914/code_snapshot_20260917_union/solve_pros4_pw_nbpo_uw1c.py:200-295`

SHA256: `edfc8c47f092ab73509cdefa7896066fd95198238ba4bc718215ebf0688b0d72`

```text
  200 |         # A hash-bound solution artifact at the path the shared job generator
  201 |         # expects. The global one is not written because per-prompt weights do
  202 |         # not fit its (K,) weight field, but the generator only needs SOME file
  203 |         # whose hash pins the solution, and patching the generator would touch
  204 |         # code the published arms depend on. This file pins the real thing: the
  205 |         # per-prompt npz that every target in this set was built from.
  206 |         solver_dir = out / split / "solver"
  207 |         base.write_json(solver_dir / "solution.json", {
  208 |             "target_mode": "canonical_logratio", "target_column": "nbpo_logratio_target",
  209 |             "target_units": "final_logratio_change", "eta_already_included": True,
  210 |             "representation": "adaptive_game", "aggregation": "prompt_wise_nash",
  211 |             "weights_scope": "per prompt; there is no shared dual",
  212 |             "per_prompt_artifact": str(per_prompt_path),
  213 |             "per_prompt_artifact_sha256": base.file_hash(per_prompt_path),
  214 |             "n_prompts": len(pids), "split": split,
  215 |             "all_certified": bool(certified.all()),
  216 |             "max_identity_residual": float(np.abs(identity).max()),
  217 |             "min_surplus_mean": float(min_surplus.mean()),
  218 |             "min_surplus_negative_prompts": int((min_surplus < 0).sum()),
  219 |             "weight_l1_matched": weight_l1, "beta": args.beta, "eta": args.eta,
  220 |             "solver_source_sha256": base.file_hash(__file__), **shared_meta})
  221 |         solver_hash = base.file_hash(solver_dir / "solution.json")
  222 |         provenance = {"solver_artifact_sha256": solver_hash,
  223 |                       "solver_hash": solver_hash,
  224 |                       "target_artifact_hash": base.file_hash(per_prompt_path),
  225 |                       "representation": "adaptive_game", "aggregation": "prompt_wise_nash",
  226 |                       "split": split, "panel": "UF-4", **shared_meta}
  227 |         betas = np.full(len(OBJECTIVES), args.beta)
  228 | 
  229 |         def pair_rows():
  230 |             for x, pid in enumerate(pids):
  231 |                 learners = {str(i): {pid: {**pool[pid]["learner"][i],
  232 |                                            "generated_text": pool[pid]["learner"][i]["response"]}}
  233 |                             for i in range(POOL)}
  234 |                 rows = build_rows(
  235 |                     [pid], list(OBJECTIVES), A[:, x:x + 1], nu[:, x:x + 1], weights[:, x],
  236 |                     betas, learners, None, np.random.default_rng(42), "canonical_logratio",
  237 |                     meta, provenance,
  238 |                     canonical_data={"g": g[x:x + 1], "p_star": pi[x:x + 1],
  239 |                                     "p_t": np.full((1, POOL), 1.0 / POOL)})
  240 |                 for row in rows:
  241 |                     # round masses to the loader's ten decimals and rebuild the
  242 |                     # target from the rounded values; a per-prompt Nash solve
  243 |                     # concentrates mass and the identity otherwise fails after the parse
  244 |                     row = base.quantize_canonical_row(row)
  245 |                     a, b = row["chosen_candidate_index"], row["rejected_candidate_index"]
  246 |                     row["chosen_response_id"] = pool[pid]["learner"][a]["candidate_id"]
  247 |                     row["rejected_response_id"] = pool[pid]["learner"][b]["candidate_id"]
  248 |                     row["prosper_prompt_weights"] = [float(v) for v in weights[:, x]]
  249 |                     yield row
  250 | 
  251 |         pair_path = out / "pairs" / f"{split}.jsonl"
  252 |         base.write_jsonl(pair_path, pair_rows())
  253 |         outputs[split] = {"n_prompts": len(pids), "n_pairs": len(pids) * 28,
  254 |                           "pairs_path": str(pair_path), "pairs_sha256": base.file_hash(pair_path),
  255 |                           "per_prompt_weight_mean": [float(v) for v in weights.mean(axis=1)],
  256 |                           "per_prompt_weight_sd": [float(v) for v in weights.std(axis=1, ddof=1)],
  257 |                           "effective_objectives_mean": float(np.mean(
  258 |                               (weights.sum(axis=0) ** 2) / (weights ** 2).sum(axis=0))),
  259 |                           "min_surplus_mean": float(min_surplus.mean()),
  260 |                           "min_surplus_negative_prompts": int((min_surplus < 0).sum()),
  261 |                           "max_identity_residual": float(np.abs(identity).max()),
  262 |                           "all_certified": bool(certified.all()),
  263 |                           "solver_solution_sha256": solver_hash,
  264 |                           "seconds": time.monotonic() - split_start}
  265 |         base.write_json(out / split / "complete.json", outputs[split])
  266 |         print(json.dumps({k: v for k, v in outputs[split].items()
  267 |                           if k not in ("pairs_path", "pairs_sha256")}), flush=True)
  268 | 
  269 |     prov = {**shared_meta, "objectives": list(OBJECTIVES), "panel": "UF-4",
  270 |             "aggregation": "prompt_wise_nash", "beta": args.beta, "eta": args.eta,
  271 |             "weight_l1": weight_l1, "splits": outputs,
  272 |             "train_pool_sha256": base.object_hash(sorted(outputs)),
  273 |             "dev_pool_sha256": base.object_hash(sorted(outputs)),
  274 |             "dev_note": ("no shared dual: each prompt fits its own adversarial weights on "
  275 |                          "whichever split it is in, so dev measures neural generalisation "
  276 |                          "but not generalisation of a dual fitted on train")}
  277 |     base.write_json(out / "dataset_provenance.json", prov)
  278 | 
  279 |     manifest = None
  280 |     if not args.skip_dataset:
  281 |         import subprocess
  282 |         dataset_out = ROOT / "datasets" / args.out_name
  283 |         env = dict(os.environ, HF_DATASETS_CACHE=str(out / "arrow_cache"),
  284 |                    PYTHONDONTWRITEBYTECODE="1")
  285 |         done = subprocess.run(
  286 |             [sys.executable, "-m", "mnpo_scripts.prepare_nbpo_dataset",
  287 |              "--train", str(out / "pairs/train.jsonl"), "--dev", str(out / "pairs/dev.jsonl"),
  288 |              "--output", str(dataset_out), "--provenance", str(out / "dataset_provenance.json")],
  289 |             env=env, capture_output=True, text=True)
  290 |         if done.returncode:
  291 |             base.write_json(out / "dataset_materialization_failure.json",
  292 |                             {"returncode": done.returncode, "stdout": done.stdout[-4000:],
  293 |                              "stderr": done.stderr[-4000:]})
  294 |             raise RuntimeError("Dataset materialization failed; diagnostic preserved")
  295 |         manifest = base.file_hash(dataset_out / "precompute_manifest.json")
```

## E13 — New uw3 wrapper selects its own stale serializer module

`analysis/sub_20260914/code_snapshot_20260917_union/solve_pros4_pw_nbpo_uw3.py:39-55`

SHA256: `5b23f08c0bc55d403880d9f4043ad7f2f0aafa9c1d250c6e36de633b8d0e4e33`

```text
   39 | """
   40 | from __future__ import annotations
   41 | 
   42 | import argparse, json, os, sys, time
   43 | from concurrent.futures import ProcessPoolExecutor
   44 | from pathlib import Path
   45 | 
   46 | import numpy as np
   47 | import torch
   48 | 
   49 | sys.path.insert(0, "/work/uf4_20260910/code")
   50 | import solve_pros4_targets_uw3 as base                      # loaders, hashing, writers
   51 | from mnpo_scripts.nbpo_core import uniform_policy
   52 | from mnpo_scripts.nbpo_generic import solve_finite_pool, validate_finite_pool_solution
   53 | from mnpo_scripts.nbpo_representations import AdaptiveGameRepresentation
   54 | from scripts.nbpo.build_nbpo_pairs import build_rows
   55 | 
```

## E14 — Training job directly launches run_mnpo

`analysis/sub_20260914/code_snapshot_20260917_union/make_pros_train_jobs.py:136-175`

SHA256: `bcabadf496632acbcefd6334b84aae5e95f5a85f129885cd4b526f0ee3feba62`

```text
  136 |     config = base
  137 |     applied = []
  138 |     for old, new in replacements.items():
  139 |         if old not in config:
  140 |             raise ValueError(f"Recipe anchor missing from the resolved base config: {old!r}")
  141 |         config = config.replace(old, new)
  142 |         applied.append({"from": old, "to": new})
  143 |     config_path = ROOT / "configs" / f"{args.arm}.yaml"
  144 |     config_path.parent.mkdir(parents=True, exist_ok=True)
  145 |     if config_path.exists() and config_path.read_text() != config:
  146 |         raise ValueError(f"Refusing to overwrite {config_path} with different content")
  147 |     config_path.write_text(config)
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
```

## E15 — Panel driver training command

`analysis/sub_20260914/code_snapshot_20260917_union/panel_stage2.py:317-363`

SHA256: `e0528503bf22fa970cc2d9a963b35a183bd3a970719240c2d3e6b87c6667734b`

```text
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
```

## E16 — Trainer exports candidate; local acceptance not present in this entry

`mnpo_scripts/run_mnpo.py:346-405`

SHA256: `def0546b64f7d2d5dd6360a7a3ff6995c126a06ff0e63326c965443cdd05d36a`

```text
  346 | 
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
  391 |         "finetuned_from": model_args.model_name_or_path,
  392 |         "dataset": list(data_args.dataset_mixer.keys()),
  393 |         "dataset_tags": list(data_args.dataset_mixer.keys()),
  394 |         "tags": ["alignment-handbook", "mnpo"],  # MODIFIED
  395 |     }
  396 |     if trainer.accelerator.is_main_process:
  397 |         trainer.create_model_card(**kwargs)
  398 |         trainer.model.config.use_cache = True
  399 |         trainer.model.config.save_pretrained(training_args.output_dir)
  400 | 
  401 |     ##########
  402 |     # Evaluate
  403 |     ##########
  404 |     # if training_args.do_eval:
  405 |     #     logger.info("*** Evaluate ***")
```

## E17 — Legacy gate tests global scalar and accepts NaN in isolated test

`scripts/nbpo/run_nbpo_stage.py:679-716`

SHA256: `d1823f4f204edef279d79fa0b5ec80cf295efb99c4e6ccbdae1e310328c2f475`

```text
  679 | def apply_gate(min_surplus: float, parent_dir: Path, candidate_dir: Path,
  680 |                promote_to: Path, stage: int = 0, fingerprint: str = None) -> dict:
  681 |     """Algorithm 1 lines 11-15: reject on any nonpositive held-out surplus, else promote."""
  682 |     if min_surplus <= 0:
  683 |         return {"accepted": False, "promoted_path": str(parent_dir),
  684 |                 "reason": f"min held-out surplus {min_surplus:.6f} <= 0; "
  685 |                           "stage flagged empirically infeasible, pi_t retained"}
  686 |     fp = fingerprint or checkpoint_fingerprint(str(candidate_dir))
  687 |     rec = promote_candidate(candidate_dir, promote_to, stage, fp)
  688 |     return {"accepted": True, "promoted_path": rec["versioned_dir"], "symlink": str(promote_to),
  689 |             "reason": f"min held-out surplus {min_surplus:.6f} > 0", "promotion": rec}
  690 | 
  691 | 
  692 | # --------------------------------------------------------------------------- #
  693 | # stage
  694 | # --------------------------------------------------------------------------- #
  695 | def _run(cmd: str, name: str) -> None:
  696 |     print(f"[nbpo-stage] running commands.{name}: {cmd}", flush=True)
  697 |     env = dict(os.environ)
  698 |     env.setdefault("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
  699 |     env.setdefault("MKL_THREADING_LAYER", "GNU")   # torch+numpy in a fresh interpreter
  700 |     subprocess.run(cmd, shell=True, check=True, env=env)
  701 | 
  702 | 
  703 | def run_stage(config_path: Path, workdir: Path, dry_run: bool) -> dict:
  704 |     base = config_path.parent
  705 |     cfg = yaml.safe_load(config_path.read_text())
  706 |     workdir.mkdir(parents=True, exist_ok=True)
  707 |     stage = int(cfg.get("stage", 0))
  708 |     objectives = list(cfg["objectives"]["names"])
  709 |     objectives_config = _resolve(base, cfg["objectives"]["config"])
  710 |     reproduction_mode = bool(cfg.get("reproduction_mode", True))
  711 |     allow_flat_judge = bool(cfg.get("allow_legacy_flat_judge_config", False))
  712 |     allow_partial = bool(cfg.get("allow_partial_prompt_intersection", False))
  713 |     if reproduction_mode and allow_partial:
  714 |         raise ValueError("allow_partial_prompt_intersection is diagnostic-only and cannot be "
  715 |                          "combined with reproduction_mode")
  716 |     # Every invariant the manuscript numbers depend on, asserted before any work.
```

## E18 — Legacy evaluator averages prompts before taking minimum

`scripts/nbpo/eval_game_value.py:33-90`

SHA256: `0f4c17b91c3ab2ff6524ae1eefcf1847f75ce77acfcbf33aa51becbed6364f03`

```text
   33 |     compute_regularized_opponent,
   34 |     opponent_entropy,
   35 |     opponent_ess,
   36 |     uniform_policy,
   37 | )
   38 | from scripts.nbpo.nbpo_common import sha256_file, write_json
   39 | 
   40 | 
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
   69 |     ap = argparse.ArgumentParser(description=__doc__,
   70 |                                  formatter_class=argparse.RawDescriptionHelpFormatter)
   71 |     ap.add_argument("--tensor-dir", type=Path, required=True,
   72 |                     help="held-out preference-tensor artifact (policy = the evaluated model)")
   73 |     ap.add_argument("--beta", required=True,
   74 |                     help="opponent temperatures: one value or comma list per objective")
   75 |     ap.add_argument("--label", default="policy")
   76 |     ap.add_argument("--out", type=Path)
   77 |     args = ap.parse_args()
   78 | 
   79 |     meta = json.loads((args.tensor_dir / "meta.json").read_text())
   80 |     A_policy = torch.from_numpy(np.load(args.tensor_dir / "tensor_policy.npz")["A"])
   81 |     A_ref = torch.from_numpy(np.load(args.tensor_dir / "tensor_ref.npz")["A"])
   82 |     K = A_policy.shape[0]
   83 |     beta_vals = [float(b) for b in args.beta.split(",") if b.strip()]
   84 |     beta = torch.tensor(beta_vals * K if len(beta_vals) == 1 else beta_vals, dtype=torch.float64)
   85 |     if beta.shape != (K,):
   86 |         raise ValueError(f"--beta must give 1 or {K} values, got {len(beta_vals)}")
   87 | 
   88 |     construction = meta.get("reference_construction")
   89 |     if construction is None:
   90 |         raise ValueError(
```

## E19 — Root README still labels R=1 shared path as paper realization

`README.md:28-72`

SHA256: `5e81387436777df46997add1c86de196c45f200d2835aacbe239c3e1de739965`

```text
   28 | 
   29 | ## Two NBPO pipelines: finite-pool NBPO realization vs legacy fixed-reference
   30 | 
   31 | The repository contains two distinct implementations. Do not conflate them.
   32 | 
   33 | **`scripts/nbpo/` — finite-pool NBPO realization of Algorithm 1.** The
   34 | manuscript's construction: adaptive KL-regularized opponents
   35 | ν\*<sub>k,π</sub> ∝ μ·exp(−r/β<sub>k</sub>) (Eq. 7), soft-min game values
   36 | V<sub>k,β</sub> (Eq. 8), a measured disagreement point
   37 | d<sub>k</sub> = V<sub>k,β</sub>(μ) (Eq. 10, never assumed zero; the
   38 | reference-vs-reference tensor is exactly skew-symmetric with a zero diagonal by
   39 | construction), projected dual descent on the **raw** multipliers
   40 | λ ← Π<sub>Λ</sub>[λ − γ(ŝ − 1/λ)] (Eq. 27), pairwise regression to
   41 | (h<sub>t</sub> − η Σ<sub>k</sub> λ<sub>k</sub>Z<sub>k</sub>)² with sequence-sum
   42 | log-probabilities (Eq. 26, `loss_type: nbpo`), and the held-out
   43 | stage-acceptance gate of Algorithm 1. **The dual solver lives in
   44 | `mnpo_scripts/nbpo_solver.py`** (CLI: `scripts/nbpo/solve_nbpo_dual.py`).
   45 | 
   46 | Two disclosed approximations, stated plainly:
   47 | 
   48 | - **R = 1.** The released configuration performs one opponent reweighting per
   49 |   dual update instead of iterating the policy–opponent fixed point to
   50 |   convergence. The solver exposes `R`, reports the fixed-point residual and a
   51 |   one-extra-map residual, and writes the *update* opponent (`nu_update.npz`,
   52 |   what Eq. 26 samples z<sub>k</sub> from) separately from the opponent
   53 |   recomputed at the final policy (`nu_final_policy.npz`, diagnostics).
   54 | - **One-shot neural realization after the frozen-pool dual.** The M = 4e3–3e5
   55 |   dual iterations run on the frozen finite response pool — cheap tensor updates
   56 |   — and the neural policy is fit **once** afterwards from the resulting pair
   57 |   targets. No 8B model is retrained per dual step.
   58 | 
   59 | λ is raw throughout: any normalized weights in logs are display diagnostics
   60 | only. The solver reports both the inverse-surplus residual ‖ŝ − 1/λ‖<sub>∞</sub>
   61 | and the projected (box-aware) KKT residual with the active-bound coordinates;
   62 | λ = 1/s is an empirical equality only where no box bound is active.
   63 | 
   64 | **Provenance binding in real mode.** `run_nbpo_stage.py` materializes the
   65 | run_mnpo YAML and parses it with run_mnpo's own argument dataclasses before any
   66 | model loads; the precompute sidecar must bind history0 to the parent
   67 | checkpoint's *content* fingerprint (every weight shard hashed — tokenizer
   68 | equality is not weight equality); the candidate is decoded synchronously by
   69 | `scripts/nbpo/decode_candidate.py`, whose manifest binds the responses to the
   70 | candidate fingerprint, the exact monitoring prompt set, every seed and every
   71 | file hash; every candidate and reference seed file must carry exactly the
   72 | monitoring prompt set; promotion is versioned and atomic
```

## E20 — New SC feature construction and solver exhaustion

`analysis/sub_20260914/code_snapshot_20260917_union/ahv2_style_control.py:50-142`

SHA256: `52ab5d10515a644081bbc561e6e22dc45b1121d724da48bf4a64396f2fa347ea`

```text
   50 | def style(text: str, n_tokens: int | None = None) -> np.ndarray:
   51 |     """The four official style features. Token length is the pipeline's own count."""
   52 |     if n_tokens is None:
   53 |         n_tokens = len(text.split())
   54 |     return np.array([float(n_tokens),
   55 |                      float(len(HEADER.findall(text))),
   56 |                      float(len(BOLD.findall(text))),
   57 |                      float(len(LIST.findall(text)))], dtype=np.float64)
   58 | 
   59 | 
   60 | def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
   61 |     total = a + b
   62 |     out = np.zeros_like(a)
   63 |     live = total > 0
   64 |     out[live] = (a[live] - b[live]) / total[live]
   65 |     return out
   66 | 
   67 | 
   68 | def fit(y: np.ndarray, X: np.ndarray, iters: int = 200, tol: float = 1e-10):
   69 |     """Newton-Raphson logistic fit on continuous y in [0, 1]; returns the coefficients.
   70 | 
   71 |     The judged outcome is 1, 0 or .5 for a tie, so this is the Bernoulli
   72 |     log-likelihood evaluated at a fractional response -- the same objective the
   73 |     official fit uses when it splits a tie across both sides.
   74 |     """
   75 |     n, d = X.shape
   76 |     beta = np.zeros(d)
   77 |     for _ in range(iters):
   78 |         eta = np.clip(X @ beta, -30.0, 30.0)
   79 |         p = 1.0 / (1.0 + np.exp(-eta))
   80 |         w = np.maximum(p * (1.0 - p), 1e-10)
   81 |         grad = X.T @ (y - p)
   82 |         hess = X.T @ (X * w[:, None])
   83 |         hess.flat[:: d + 1] += 1e-8          # ridge, so a collinear replicate still solves
   84 |         try:
   85 |             step = np.linalg.solve(hess, grad)
   86 |         except np.linalg.LinAlgError:
   87 |             return None
   88 |         beta = beta + step
   89 |         if np.max(np.abs(step)) < tol:
   90 |             return beta
   91 |     return beta
   92 | 
   93 | 
   94 | def main() -> int:
   95 |     ap = argparse.ArgumentParser(description=__doc__)
   96 |     ap.add_argument("--judged", required=True,
   97 |                     help="eval_pairwise directory holding verdicts.jsonl and complete.json")
   98 |     ap.add_argument("--baseline", required=True, help="the frozen baseline jsonl")
   99 |     ap.add_argument("--panel", required=True,
  100 |                     help="the frozen panel jsonl; it carries the prompt_id -> uid join the "
  101 |                          "judging run used, and the baseline is keyed by uid")
  102 |     ap.add_argument("--replicates", type=int, default=2000)
  103 |     ap.add_argument("--seed", type=int, default=20260918)
  104 |     ap.add_argument("--out", required=True)
  105 |     args = ap.parse_args()
  106 | 
  107 |     judged = Path(args.judged)
  108 |     complete = json.loads((judged / "complete.json").read_text())
  109 |     arm_path = Path(complete["arm_responses"])
  110 | 
  111 |     arm = {}
  112 |     for line in arm_path.open():
  113 |         r = json.loads(line)
  114 |         arm[r["prompt_id"]] = style(r["response"], r.get("n_tokens"))
  115 | 
  116 |     # the same join the judging run used: panel prompt_id -> uid -> baseline answer
  117 |     uid_of = {}
  118 |     for line in Path(args.panel).open():
  119 |         r = json.loads(line)
  120 |         uid_of[r["prompt_id"]] = r["uid"]
  121 | 
  122 |     by_uid = {}
  123 |     for line in Path(args.baseline).open():
  124 |         r = json.loads(line)
  125 |         text = r["messages"][-1]["content"] if "messages" in r else r.get("response", "")
  126 |         if isinstance(text, dict):
  127 |             text = text.get("answer", "")
  128 |         by_uid[r["uid"]] = style(text)
  129 |     base = {pid: by_uid[uid] for pid, uid in uid_of.items() if uid in by_uid}
  130 | 
  131 |     rows, skipped = [], {"no_style": 0, "not_ok": 0}
  132 |     for line in (judged / "verdicts.jsonl").open():
  133 |         v = json.loads(line)
  134 |         if v.get("status") != "ok":
  135 |             skipped["not_ok"] += 1
  136 |             continue
  137 |         pid = v["prompt_id"]
  138 |         if pid not in arm or pid not in base:
  139 |             skipped["no_style"] += 1
  140 |             continue
  141 |         rows.append((pid, float(v["value_for_arm"]),
  142 |                      normalized_difference(arm[pid], base[pid])))
```

## E21 — Same asymmetric feature construction in paired SC

`analysis/sub_20260914/code_snapshot_20260917_union/ahv2_sc_paired.py:28-89`

SHA256: `cc0e229136f1aee2508c31fb16d3a13b06129ba0d24e8301b1ea54bcbe5639c3`

```text
   28 | 
   29 | 
   30 | def load(judged: Path, baseline: Path, panel: Path):
   31 |     complete = json.loads((judged / "complete.json").read_text())
   32 |     arm = {}
   33 |     for line in Path(complete["arm_responses"]).open():
   34 |         r = json.loads(line)
   35 |         arm[r["prompt_id"]] = style(r["response"], r.get("n_tokens"))
   36 |     uid_of = {}
   37 |     for line in panel.open():
   38 |         r = json.loads(line)
   39 |         uid_of[r["prompt_id"]] = r["uid"]
   40 |     by_uid = {}
   41 |     for line in baseline.open():
   42 |         r = json.loads(line)
   43 |         text = r["messages"][-1]["content"] if "messages" in r else r.get("response", "")
   44 |         if isinstance(text, dict):
   45 |             text = text.get("answer", "")
   46 |         by_uid[r["uid"]] = style(text)
   47 |     base = {pid: by_uid[uid] for pid, uid in uid_of.items() if uid in by_uid}
   48 |     rows = []
   49 |     for line in (judged / "verdicts.jsonl").open():
   50 |         v = json.loads(line)
   51 |         if v.get("status") != "ok":
   52 |             continue
   53 |         pid = v["prompt_id"]
   54 |         if pid in arm and pid in base:
   55 |             rows.append((pid, float(v["value_for_arm"]),
   56 |                          normalized_difference(arm[pid], base[pid])))
   57 |     return complete["arm"], rows
   58 | 
   59 | 
   60 | def assemble(rows, pids):
   61 |     index = {p: i for i, p in enumerate(pids)}
   62 |     buckets = [[] for _ in pids]
   63 |     y = np.empty(len(rows))
   64 |     D = np.empty((len(rows), 4))
   65 |     for i, (pid, val, feat) in enumerate(rows):
   66 |         y[i] = val
   67 |         D[i] = feat
   68 |         buckets[index[pid]].append(i)
   69 |     X = np.column_stack([np.ones(len(rows)), D])
   70 |     return y, X, buckets
   71 | 
   72 | 
   73 | def sc(y, X, idx):
   74 |     b = fit(y[idx], X[idx])
   75 |     if b is None:
   76 |         return None
   77 |     return 1.0 / (1.0 + math.exp(-b[0]))
   78 | 
   79 | 
   80 | def main() -> int:
   81 |     ap = argparse.ArgumentParser(description=__doc__)
   82 |     ap.add_argument("--a", required=True, help="judged directory of the arm under test")
   83 |     ap.add_argument("--b", required=True, help="judged directory of the comparator")
   84 |     ap.add_argument("--baseline", required=True)
   85 |     ap.add_argument("--panel", required=True)
   86 |     ap.add_argument("--replicates", type=int, default=2000)
   87 |     ap.add_argument("--seed", type=int, default=20260918)
   88 |     ap.add_argument("--out", required=True)
   89 |     args = ap.parse_args()
```

## E22 — Generation n_tokens field counts model tokens

`analysis/sub_20260914/code_snapshot_20260917_union/gen_candidates.py:118-135`

SHA256: `38d4ac85a746065987fae5620aef63ee8e6b375829795f67748b3f342af060ee`

```text
  118 | 
  119 |     started = time.monotonic()
  120 |     generated = llm.generate(requests, params, use_tqdm=False)
  121 |     elapsed = time.monotonic() - started
  122 | 
  123 |     events = []
  124 |     for (row, index, seed), result in zip(mapping, generated):
  125 |         o = result.outputs[0]
  126 |         text = tok.decode(list(o.token_ids), skip_special_tokens=True,
  127 |                           clean_up_tokenization_spaces=False)
  128 |         events.append({"prompt_id": row["prompt_id"], "source": row["source"],
  129 |                        "instruction": row["instruction"],
  130 |                        "response_id": "%s:%d" % (row["prompt_id"], index),
  131 |                        "response_index": index, "seed": seed, "response": text,
  132 |                        "n_tokens": len(o.token_ids), "finish_reason": o.finish_reason,
  133 |                        "capped": o.finish_reason == "length",
  134 |                        "response_sha256": digest(text)})
  135 |     with dest.open("x") as stream:
```

## E23 — Actual canonical validator used in the 28-pair regression

`mnpo_scripts/nbpo_neural.py:110-174`

SHA256: `b6df54c0771bc5d9b55092a117655fd08da609a19a25cdbeebbb2605a8b67ddc`

```text
  110 | def validate_canonical_pair_dataset(dataset, max_length=2048, max_prompt_length=1024,
  111 |                                     expected_solver_hash=None):
  112 |     """Check pair completeness, candidate mass/context identity and eta units.
  113 | 
  114 |     Duplicate sampled strings remain separate occurrences. Each of the 28
  115 |     unordered pairs must occur exactly once for every prompt, with fixed N=8.
  116 |     """
  117 |     groups = {}
  118 |     solvers = set()
  119 |     for row in dataset:
  120 |         if row.get("target_mode") != "canonical_logratio" or row.get("target_units") != "final_logratio_change" or row.get("eta_already_included") is not True:
  121 |             raise ValueError("Canonical row must declare final logratio target units and eta included")
  122 |         if not math.isfinite(float(row["nbpo_logratio_target"])):
  123 |             raise ValueError("Nonfinite canonical target")
  124 |         if int(row["nbpo_num_candidates"]) != 8:
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

## E24 — Snapshot documentation declares corrected scope

`analysis/sub_20260914/code_snapshot_20260917_union/README.md:13-75`

SHA256: `c78576e048ce28cf991e287da42ee752c8037ea1c564c426d34604ee3bcf0a63`

```text
   13 | the published number and the hash here is the code as it now stands.
   14 | 
   15 | ## Corrected tensor roles (2026-09-18)
   16 | 
   17 | An external implementation audit found that this scorer wired the game tensors
   18 | against the paper's own definition, and the fix is the most consequential change
   19 | in this directory:
   20 | 
   21 |     was:  A_policy <- learner-learner      A_ref <- learner-reference
   22 |     now:  A_policy <- learner-reference    A_ref <- reference-reference
   23 | 
   24 | Section 5.2 defines the policy game on the learner bank against the comparator
   25 | bank, and `compute_disagreement_point` documents its own argument as "centered
   26 | payoffs of reference responses (as learner) against reference comparators". The
   27 | old wiring satisfied neither. It is not a naming slip: with every learner tied
   28 | to every other learner, every reference tied to every other reference, and every
   29 | learner beating every reference at .75, the paper's wiring gives policy value
   30 | +.25, disagreement 0 and surplus +.25, while the old wiring gives 0, +.25 and
   31 | surplus **-.25**. The sign of the surplus flips, and the surplus is what
   32 | finite-pool feasibility tests, so the solver was solving a different finite
   33 | problem exactly rather than the declared one approximately.
   34 | 
   35 | `A_LL` also cannot carry the policy game: it is antisymmetric with a zero
   36 | diagonal, so a symmetric strategy scores identically zero against itself and the
   37 | surplus carries almost no signal about the learner bank — consistent with the
   38 | near-zero target correlations the UW and US rounds measured.
   39 | 
   40 | `union_score_panel.py` now writes `A_LL`, `A_LR` and `A_RR` each under its own
   41 | name and aliases `A_policy`/`A_ref` to `A_LR`/`A_RR`, so an existing reader gets
   42 | the paper's game unchanged while a pair-label consumer can ask for the learner
   43 | block explicitly. `build_panel_softlabels.py` was reading `A_policy` and
   44 | expecting the learner block, so it now reads `A_LL` and refuses a score
   45 | directory written before this fix instead of silently taking the cross block.
   46 | The reference triangle was previously kept only as a scalar diagnostic and is
   47 | now required for a prompt to be retained, because `d` is not defined without it.
   48 | 
   49 | `prosper_targets_panel.py` separates the roles itself from the raw judgments and
   50 | is unaffected. Re-scoring needs no new judging: all three role blocks were
   51 | already fully judged on every panel.
   52 | 
   53 | Serialization changed with it. `quantize_canonical_row` used to round both solver
   54 | masses to ten FIXED decimals and rebuild the target from the rounded values; a
   55 | per-prompt Nash solve concentrates mass down to the 1e-12 probability floor, and
   56 | 1e-12 at ten decimals is exactly 0, which the canonical validator rejects. Fixed
   57 | decimals are the wrong instrument for a quantity spanning twelve orders of
   58 | magnitude, so masses are now written at exact float64 round-trip precision and
   59 | the target is rebuilt from those same doubles. The identity holds to ~1e-16,
   60 | inside the 1e-9 gate, with the certified solution intact. Rows written before
   61 | this carry `canonical_target_quantized_decimals = 10`; new rows carry
   62 | `canonical_target_serialization = "exact_float64_round_trip"`.
   63 | 
   64 | ## The prompt-wise NBPO line
   65 | 
   66 | The prompt-wise change is in the solver, not the trainer: NBPO's dual
   67 | multipliers are fitted separately at each prompt instead of being shared across
   68 | prompts. `solve_pros4_targets.py` is the shared-weight solver, and every
   69 | per-prompt and per-panel variant is *generated* from it by a single generator,
   70 | so the arms cannot drift apart in anything except the rule they implement.
   71 | 
   72 | | Arm | Representation | Aggregation | Weight scope |
   73 | |---|---|---|---|
   74 | | NBPO-PW | adaptive game | Nash | per prompt |
   75 | | Fixed-reference Nash-PW | fixed reference | Nash | per prompt |
```

## E25 — Global-script writer export and imports

`scripts/nbpo/solve_nbpo_dual.py:1-115`

SHA256: `b70ca16d26140fd244b01ad64e77e927d9fd97235c24ff74e2d61847bc3b850e`

```text
    1 | #!/usr/bin/env python3
    2 | """Deterministic finite-pool NBPO dual solve (Algorithm 1's inner machinery).
    3 | 
    4 | Reads a versioned preference-tensor artifact (``build_preference_tensor.py``)
    5 | and runs projected dual gradient descent on the raw multipliers
    6 | (Eq. (27) ``eq:dual-update``) with the fixed-point weighted-policy solve of
    7 | Section 5.2 (Eq. (21), centered at the proximal center), or one of the matched
    8 | finite-game controls (utilitarian / absolute max-min / surplus max-min) on the
    9 | same tensors and budget.
   10 | 
   11 | Scope: every one of the ``M`` dual iterations (4e3--3e5 in the manuscript) is a
   12 | cheap tensor computation on the FROZEN finite response pool. The neural policy
   13 | is fit afterwards, once, from the targets built by ``build_nbpo_pairs.py`` --
   14 | no 8B model is retrained inside this loop.
   15 | 
   16 | Outputs (all raw, none normalized or clamped):
   17 | ``solution.json`` -- raw lambda, V, d, surplus, the inverse-surplus residual
   18 | ``||s - 1/lambda||_inf`` AND the projected (box-aware) KKT residual with the
   19 | active-bound coordinates, fixed-point and one-extra-map residuals, opponent
   20 | entropy/ESS of the final policy, the full config (beta, eta, gamma schedule,
   21 | M, R, lambda box), input artifact hashes, hashes of the opponent files, and the
   22 | iteration history; ``nu_update.npz`` -- the opponent that generated the final
   23 | policy (what Eq. (26) pair construction samples from); ``nu_final_policy.npz``
   24 | -- ``nu*`` recomputed at the final policy (diagnostics); ``pi_star.npz`` -- the
   25 | finite-pool policy.
   26 | """
   27 | from __future__ import annotations
   28 | 
   29 | import argparse
   30 | import json
   31 | from pathlib import Path
   32 | 
   33 | import numpy as np
   34 | import torch
   35 | 
   36 | from mnpo_scripts.nbpo_core import (
   37 |     uniform_policy,
   38 |     validate_centered_preference_tensor,
   39 |     validate_reference_tensor,
   40 | )
   41 | from mnpo_scripts.nbpo_solver import AGGREGATIONS, solve_nbpo_dual
   42 | from scripts.nbpo.nbpo_common import implementation_contract, sha256_file, write_json
   43 | 
   44 | 
   45 | def load_tensor_artifact(tensor_dir: Path):
   46 |     meta = json.loads((tensor_dir / "meta.json").read_text())
   47 |     A_policy = torch.from_numpy(np.load(tensor_dir / "tensor_policy.npz")["A"])
   48 |     A_ref = torch.from_numpy(np.load(tensor_dir / "tensor_ref.npz")["A"])
   49 |     hashes = {name: sha256_file(tensor_dir / name)
   50 |               for name in ("tensor_policy.npz", "tensor_ref.npz", "meta.json")}
   51 |     return meta, A_policy, A_ref, hashes
   52 | 
   53 | 
   54 | def parse_gamma(text: str, M: int):
   55 |     parts = [float(p) for p in text.split(",") if p.strip()]
   56 |     return parts[0] if len(parts) == 1 else parts
   57 | 
   58 | 
   59 | def _array_hash(t) -> str:
   60 |     """sha256 of an array's exact bytes -- identifies which policy an opponent came from."""
   61 |     import hashlib
   62 | 
   63 |     arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
   64 |     return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()
   65 | 
   66 | 
   67 | def write_solution_artifact(out_dir: Path, res, tensor_meta: dict, hashes: dict,
   68 |                             tensor_dir: Path, stage: int, lambda_warm_started: bool) -> dict:
   69 |     """Persist a DualSolveResult as the versioned solver artifact (shared with run_nbpo_stage)."""
   70 |     out_dir.mkdir(parents=True, exist_ok=True)
   71 |     # Two opponents, written separately and hashed separately (never confuse them):
   72 |     #   nu_update.npz       -- generated the final policy; Eq. (26) targets sample z_k here
   73 |     #   nu_final_policy.npz -- nu* recomputed AT the final policy; diagnostics only
   74 |     np.savez_compressed(out_dir / "nu_update.npz", nu=res.nu_update.numpy())
   75 |     np.savez_compressed(out_dir / "nu_final_policy.npz", nu=res.nu_final_policy.numpy())
   76 |     np.savez_compressed(out_dir / "pi_star.npz", pi=res.pi.numpy())
   77 |     # The policy nu_update was actually built from, saved so the claim can be
   78 |     # checked rather than trusted. solve_nbpo_dual warm-starts the policy iterate
   79 |     # across dual iterations, so at R = 1 this is the warm-start iterate, not the
   80 |     # proximal centre -- which is what the artifact used to assert.
   81 |     if res.update_source_pi is None:
   82 |         raise ValueError("solver result carries no update_source_pi; the policy that "
   83 |                          "generated nu_update cannot be identified")
   84 |     np.savez_compressed(out_dir / "update_source_pi.npz", pi=res.update_source_pi.numpy())
   85 |     solution = {
   86 |         # What actually ran (audit section 0): the dual below is a frozen finite-pool
   87 |         # optimization; the neural policy is realized once, afterwards.
   88 |         **implementation_contract(dual_iterations=res.config.get("M"),
   89 |                                   fixed_point_steps=res.config.get("R")),
   90 |         "aggregation": res.aggregation,
   91 |         "stage": int(stage),
   92 |         "objectives": tensor_meta.get("objectives"),
   93 |         "lambda_raw": [float(v) for v in res.lam],
   94 |         "V": [float(v) for v in res.V],
   95 |         "d": [float(v) for v in res.d],
   96 |         "surplus": [float(v) for v in res.surplus],
   97 |         "min_surplus": float(res.surplus.min()),
   98 |         "kkt_residual": res.kkt_residual,
   99 |         "inverse_surplus_residual": res.kkt_residual,
  100 |         "projected_kkt_residual": res.projected_kkt_residual,
  101 |         "gamma_ref": res.gamma_ref,
  102 |         "lambda_at_lower_bound": res.lambda_at_lower_bound,
  103 |         "lambda_at_upper_bound": res.lambda_at_upper_bound,
  104 |         "kkt_note": ("lambda_k = 1/s_k is an empirical equality only for coordinates "
  105 |                      "strictly inside the box; when a bound is active read "
  106 |                      "projected_kkt_residual, not inverse_surplus_residual"),
  107 |         "control_residual": res.control_residual,
  108 |         "fixed_point_residual": res.fixed_point_residual,
  109 |         "extra_map_residual": res.extra_map_residual,
  110 |         "opponent_entropy": [float(v) for v in res.opponent_entropy],
  111 |         "opponent_ess": [float(v) for v in res.opponent_ess],
  112 |         "opponent_diagnostics_from": "nu_final_policy",
  113 |         "artifact_hashes": {
  114 |             "nu_update.npz": sha256_file(out_dir / "nu_update.npz"),
  115 |             "nu_final_policy.npz": sha256_file(out_dir / "nu_final_policy.npz"),
```
