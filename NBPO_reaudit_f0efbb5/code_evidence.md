# f0efbb5 original-source evidence


## E01 — New gate: contract, imports, CLI defaults

Path: `analysis/sub_20260914/code_snapshot_20260917_union/nbpo_local_gate.py`
SHA256: `a5acfd5dbe44b48381efc3d0d3d52d50bd6bf2d2afe88c623ac0161538992485`

```text
    1  """Algorithm 1's acceptance step for the PROMPT-WISE path, evaluated per prompt.
    2  
    3  The manuscript's Algorithm 1 does not stop at a certified finite target. It
    4  trains a neural candidate against that target, checks the candidate on held-out
    5  development data, and only then makes it the next policy; a candidate that fails
    6  the check is discarded and the parent is retained. An implementation audit found
    7  that the campaign's prompt-wise training path ran
    8  
    9      job generator -> torch.distributed.run -> run_mnpo -> train() -> save_model()
   10  
   11  with nothing between or after it that evaluated, rejected or promoted anything.
   12  The certified solver residual is not a substitute: it certifies the finite target,
   13  not the policy fitted to it.
   14  
   15  The existing `scripts/nbpo/eval_game_value.py` cannot be reused here. It reduces
   16  to a scalar minimum over PROMPT-AVERAGED surpluses, which is the right object for
   17  the shared-weight Global Nash control and the wrong one for a rule whose
   18  multipliers are fitted per prompt: two prompts at +.2 and -.1 average to +.05 and
   19  pass, while one of them is a prompt the compromise failed on. This file evaluates
   20  the array and applies the declared policy to it.
   21  
   22  What it measures, for each development prompt x and objective k:
   23  
   24    the candidate's distribution over that prompt's eight learner occurrences,
   25    by importance reweighting on the sampled support,
   26        p(i) proportional to p_t(i) * exp(log pi_cand(y_i) - log pi_parent(y_i))
   27    its regularized game value V_{k,beta}(p) against the reference bank, and
   28    the surplus s_k(x) = V_{k,beta}(p) - d_k(x)
   29    against the disagreement point d the solver recorded for that prompt.
   30  
   31  The declared checks, all three of which must hold:
   32  
   33    finite    every surplus is finite; a NaN or an infinity is a failure, never a
   34              pass, because `nan <= 0` is False and would otherwise promote
   35    coverage  the fraction of development prompts whose surplus is positive on
   36              EVERY objective is at least --coverage-min
   37    fit       the held-out nMSE of the realized log-ratio against the canonical
   38              target is at most --nmse-max
   39  
   40  On a pass the candidate is promoted to a versioned accepted directory with a
   41  symlink, and the promotion record names the checkpoint, its fingerprint and the
   42  exact artifacts the decision was read from. On a failure nothing is promoted and
   43  the parent path is returned, which is what the caller must then evaluate.
   44  
   45  This gate exists from 2026-09-18. It does not apply retroactively: every result
   46  already reported in the manuscript came from an ungated projection, and that is
   47  a fact about those runs, not something a later gate can change. Anything reported
   48  as gated must name a promotion record written by this file.
   49  """
   50  from __future__ import annotations
   51  
   52  import argparse
   53  import hashlib
   54  import json
   55  import os
   56  import sys
   57  import time
   58  from pathlib import Path
   59  
   60  import numpy as np
   61  import torch
   62  
   63  sys.path.insert(0, "/work/uf4_20260910/code")
   64  from diag_neural_pools import load_pool_rows, sequence_logprobs   # noqa: E402
   65  
   66  ROOT = Path("/work/uf4_20260910")
   67  POOL = 8
   68  
   69  
   70  def fingerprint(path: Path) -> str:
   71      """A content digest of the checkpoint's weight and config files."""
   72      h = hashlib.sha256()
   73      for f in sorted(path.rglob("*")):
   74          if f.is_file() and f.suffix in (".safetensors", ".bin", ".json", ".model"):
   75              h.update(f.name.encode())
   76              with f.open("rb") as s:
   77                  for c in iter(lambda: s.read(1 << 22), b""):
   78                      h.update(c)
   79      return h.hexdigest()
   80  
   81  
   82  def main() -> int:
   83      ap = argparse.ArgumentParser(description=__doc__,
   84                                   formatter_class=argparse.RawDescriptionHelpFormatter)
   85      ap.add_argument("--candidate", required=True, help="the trained checkpoint under test")
   86      ap.add_argument("--parent", required=True,
   87                      help="the policy retained if the candidate fails; also the "
   88                           "reference the importance weights are taken against")
   89      ap.add_argument("--targets", required=True,
   90                      help="targets/<name>: the per-prompt solve this candidate was fitted to")
   91      ap.add_argument("--pool", required=True)
   92      ap.add_argument("--scores", required=True,
   93                      help="scores/<dir> the solve read; the gate evaluates the SAME game, "
   94                           "so it goes through the solver's own loader and inherits its "
   95                           "tensor-role schema check")
   96      ap.add_argument("--solver-module", required=True,
   97                      help="the panel's solve_pros4_targets_<panel>.py, whose load_scores "
   98                           "and OBJECTIVES define the game being gated")
   99      ap.add_argument("--split", default="dev")
  100      ap.add_argument("--beta", type=float, default=0.25)
  101      ap.add_argument("--coverage-min", type=float, default=0.5,
  102                      help="minimum fraction of development prompts positive on every "
  103                           "objective; declare it before running, not after reading")
  104      ap.add_argument("--nmse-max", type=float, default=1.0,
  105                      help="maximum held-out nMSE; 1.0 is the value of predicting zero")
  106      ap.add_argument("--nmse-from", default=None,
  107                      help="trainer_state.json of the candidate, for the recorded "
  108                           "held-out nMSE; omit to skip the fit check and say so")
  109      ap.add_argument("--promote-to", default=None,
  110                      help="symlink to point at the accepted checkpoint; without it "
  111                           "the gate reports and promotes nothing")
  112      ap.add_argument("--batch", type=int, default=4)
  113      ap.add_argument("--logprobs", default=None,
  114                      help="JSON {candidate: {id: logp}, parent: {id: logp}} of sequence "
  115                           "log-probabilities already computed. Lets the same decision be "
  116                           "re-derived, and tested, without a forward pass; the models are "
  117                           "not loaded at all when it is given.")
  118      ap.add_argument("--out", required=True)
  119      args = ap.parse_args()
  120  
  121      tdir = Path(args.targets)
  122      npz = tdir / ("%s_per_prompt.npz" % args.split)
  123      if not npz.exists():
  124          raise SystemExit(
  125              "%s has no %s_per_prompt.npz. This gate is defined on a per-prompt solve; a "
  126              "shared-weight solve has no per-prompt disagreement point and must be gated "
  127              "with the global evaluator instead." % (tdir, args.split))
  128      z = np.load(npz, allow_pickle=True)
  129      pids = [str(p) for p in z["prompt_ids"]]
  130      recorded_min = np.asarray(z["min_surplus"], dtype=np.float64)
  131  
  132      candidates = {pid: ["%s:learner:%d" % (pid, i) for i in range(POOL)] for pid in pids}
  133      wanted = {c for v in candidates.values() for c in v}
  134      order = [c for pid in pids for c in candidates[pid]]
  135      if args.logprobs:
  136          rows = None            # no tokenization is needed when the log-probs are given
  137      else:
```


## E02 — Occurrence reweighting and one-representation-per-prompt evaluation

Path: `analysis/sub_20260914/code_snapshot_20260917_union/nbpo_local_gate.py`
SHA256: `a5acfd5dbe44b48381efc3d0d3d52d50bd6bf2d2afe88c623ac0161538992485`

```text
  138          rows = load_pool_rows(args.pool, wanted)
  139          missing = sorted(wanted - set(rows))
  140          if missing:
  141              raise SystemExit("missing cached tokenization for %d occurrences" % len(missing))
  142  
  143      if args.logprobs:
  144          cached = json.loads(Path(args.logprobs).read_text())
  145          parent_logp = {k: float(v) for k, v in cached["parent"].items()}
  146          cand_logp = {k: float(v) for k, v in cached["candidate"].items()}
  147          for name, d in (("parent", parent_logp), ("candidate", cand_logp)):
  148              gaps = [c for c in order if c not in d]
  149              if gaps:
  150                  raise SystemExit("%s log-probabilities are missing %d occurrences"
  151                                   % (name, len(gaps)))
  152          logprob_source = args.logprobs
  153      else:
  154          device = "cuda" if torch.cuda.is_available() else "cpu"
  155  
  156          def score(path: str):
  157              from transformers import AutoModelForCausalLM
  158              model = AutoModelForCausalLM.from_pretrained(
  159                  path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
  160                  local_files_only=True).to(device).eval()
  161              out = sequence_logprobs(model, rows, order, device, args.batch)
  162              del model
  163              if device == "cuda":
  164                  torch.cuda.empty_cache()
  165              return dict(zip(order, out))
  166  
  167          parent_logp = score(args.parent)
  168          cand_logp = score(args.candidate)
  169          logprob_source = "forward pass on %s" % device
  170  
  171      # the candidate's distribution over each prompt's own eight occurrences
  172      pt = np.full(POOL, 1.0 / POOL)
  173      p_cand = np.empty((len(pids), POOL), dtype=np.float64)
  174      unusable = set()
  175      for x, pid in enumerate(pids):
  176          cs = candidates[pid]
  177          delta = np.array([cand_logp[c] - parent_logp[c] for c in cs], dtype=np.float64)
  178          if not np.all(np.isfinite(delta)):
  179              # A prompt whose log-probability is NaN or infinite has no usable
  180              # distribution. It is recorded as a failure of the finiteness check,
  181              # not raised: a gate that crashes on a bad value leaves the caller
  182              # with no decision, and no decision is indistinguishable downstream
  183              # from a pass.
  184              unusable.add(pid)
  185              p_cand[x] = pt
  186              continue
  187          logits = np.log(pt) + delta
  188          logits -= logits.max()
  189          w = np.exp(logits)
  190          p_cand[x] = w / w.sum()
  191  
  192      # the surplus is evaluated by the paper's own object rather than a second
  193      # implementation of Eq. 8 here: same A_policy, same A_ref, same beta, same
  194      # reference measure as the solve, so a disagreement between the two would be
  195      # a disagreement about the policy and not about the algebra
  196      import importlib.util
  197      spec = importlib.util.spec_from_file_location("gated_solver", args.solver_module)
  198      solver = importlib.util.module_from_spec(spec)
  199      sys.modules["gated_solver"] = solver
  200      spec.loader.exec_module(solver)
  201      from mnpo_scripts.nbpo_core import uniform_policy
  202      from mnpo_scripts.nbpo_representations import AdaptiveGameRepresentation
  203  
  204      scores, _ = solver.load_scores(args.scores, 4)
  205      absent = [pid for pid in pids if pid not in scores]
  206      if absent:
  207          raise SystemExit("%d development prompts have no score tensor" % len(absent))
  208      A = np.stack([scores[pid][0] for pid in pids], axis=1)
  209      Aref = np.stack([scores[pid][1] for pid in pids], axis=1)
  210      K = A.shape[0]
  211      construction = getattr(solver, "REFERENCE_CONSTRUCTION", "shared_pool")
  212      # ONE REPRESENTATION PER PROMPT. Built over all prompts at once,
  213      # `game_values` divides by X -- it returns the prompt-averaged value, which is
  214      # the object the shared-weight control is gated on and the wrong one here: two
  215      # prompts at +.2 and -.1 average to +.05 and would pass while the compromise
  216      # failed on one of them. Constructing X=1 at a time keeps the audited
  217      # implementation of Eq. (8) and Eq. (10) and evaluates it where the rule is
  218      # defined.
  219      surplus = np.empty((K, len(pids)), dtype=np.float64)
  220      beta_vec = torch.full((K,), args.beta, dtype=torch.float64)
  221      for x in range(len(pids)):
  222          if pids[x] in unusable:
  223              surplus[:, x] = np.nan
  224              continue
  225          rep = AdaptiveGameRepresentation(
  226              torch.from_numpy(A[:, x:x + 1]), torch.from_numpy(Aref[:, x:x + 1]),
  227              uniform_policy(1, POOL), beta_vec,
  228              reference_construction=construction)
  229          s = rep.surplus(torch.from_numpy(p_cand[x:x + 1])).detach().cpu().numpy()
  230          surplus[:, x] = np.asarray(s, dtype=np.float64).reshape(K)
  231  
  232      per_prompt, nonfinite = [], 0
  233      for x, pid in enumerate(pids):
  234          s = np.asarray(surplus[:, x], dtype=np.float64)
  235          ok = pid not in unusable and np.all(np.isfinite(s))
  236          if not ok:
  237              nonfinite += 1
  238          per_prompt.append({
  239              "prompt_id": pid,
  240              "surplus": None if pid in unusable else s.tolist(),
  241              "min_surplus": float(np.min(s)) if ok else None,
  242              "recorded_target_min_surplus": float(recorded_min[x]),
  243              "reason": ("candidate log-probability is not finite" if pid in unusable
  244                         else ("surplus is not finite" if not ok else None))})
  245  
  246      finite_ok = nonfinite == 0
  247      mins = np.array([r["min_surplus"] if r["min_surplus"] is not None else np.nan
  248                       for r in per_prompt], dtype=np.float64)
  249      positive = np.isfinite(mins) & (mins > 0)
  250      coverage = float(positive.mean())
```


## E03 — Fit predicate, decision, fingerprint/promotion and policy result

Path: `analysis/sub_20260914/code_snapshot_20260917_union/nbpo_local_gate.py`
SHA256: `a5acfd5dbe44b48381efc3d0d3d52d50bd6bf2d2afe88c623ac0161538992485`

```text
  252      nmse, fit_ok, fit_note = None, None, "not checked: --nmse-from was not given"
  253      if args.nmse_from:
  254          state = json.loads(Path(args.nmse_from).read_text())
  255          seen = [e for e in state.get("log_history", []) if "eval_nbpo/nmse" in e]
  256          if seen:
  257              nmse = float(seen[-1]["eval_nbpo/nmse"])
  258              fit_ok = nmse <= args.nmse_max
  259              fit_note = "last recorded held-out nMSE"
  260          else:
  261              fit_note = "no eval_nbpo/nmse in the trainer state"
  262  
  263      checks = {"finite": finite_ok,
  264                "coverage": coverage >= args.coverage_min,
  265                "fit": fit_ok}
  266      accepted = bool(finite_ok and checks["coverage"] and (fit_ok is not False))
  267  
  268      record = {
  269          "gate": "prompt_wise_local_acceptance",
  270          "algorithm_1_step": "held-out development check before the candidate becomes pi_{t+1}",
  271          "decided_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  272          "candidate": args.candidate,
  273          "parent": args.parent,
  274          "targets": str(tdir),
  275          "pool": args.pool,
  276          "scores": args.scores,
  277          "solver_module": args.solver_module,
  278          "split": args.split,
  279          "prompts": len(pids),
  280          "logprob_source": logprob_source,
  281          "thresholds": {"coverage_min": args.coverage_min, "nmse_max": args.nmse_max,
  282                         "beta": args.beta},
  283          "measured": {"coverage": coverage, "nonfinite_prompts": nonfinite,
  284                       "prompts_with_unusable_logprobs": sorted(unusable),
  285                       "held_out_nmse": nmse, "nmse_note": fit_note,
  286                       "min_surplus_quantiles": {
  287                           q: (float(np.nanquantile(mins, v)) if np.any(np.isfinite(mins)) else None)
  288                           for q, v in (("p05", .05), ("p50", .5), ("p95", .95))}},
  289          "checks": checks,
  290          "accepted": accepted,
  291          "estimator": ("candidate distribution by importance reweighting on the sampled "
  292                        "support, p ∝ p_t · exp(log pi_cand − log pi_parent); this is not the "
  293                        "policy's distribution over all responses"),
  294          "not_retroactive": ("this gate exists from 2026-09-18; results reported before it "
  295                              "came from an ungated projection and must not be described as "
  296                              "gated"),
  297          "per_prompt": per_prompt,
  298      }
  299  
  300      if accepted and args.promote_to:
  301          stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
  302          fp = fingerprint(Path(args.candidate))
  303          versioned = Path(args.promote_to).parent / ("%s.accepted.%s" % (
  304              Path(args.promote_to).name, stamp))
  305          versioned.parent.mkdir(parents=True, exist_ok=True)
  306          (versioned).mkdir(exist_ok=False)
  307          (versioned / "ACCEPTED.json").write_text(json.dumps(
  308              {"checkpoint": args.candidate, "fingerprint": fp,
  309               "gate_record": args.out, "decided_utc": record["decided_utc"]},
  310              indent=1) + "\n")
  311          link = Path(args.promote_to)
  312          if link.is_symlink() or link.exists():
  313              link.unlink()
  314          link.symlink_to(versioned)
  315          record["promotion"] = {"versioned_dir": str(versioned), "symlink": str(link),
  316                                 "candidate_fingerprint": fp}
  317          record["policy_after_this_stage"] = args.candidate
  318      else:
  319          record["promotion"] = None
  320          record["policy_after_this_stage"] = args.parent
  321          if not accepted:
  322              record["consequence"] = ("the candidate is rejected and the parent is retained; "
  323                                       "evaluation must be run on the parent, and this stage "
  324                                       "reports no improvement rather than a smaller one")
  325  
  326      Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  327      Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
  328      print(json.dumps({k: record[k] for k in (
  329          "accepted", "checks", "measured", "policy_after_this_stage")}, indent=1))
  330      return 0 if accepted else 3
  331  
  332  
  333  if __name__ == "__main__":
  334      raise SystemExit(main())
```


## E04 — Score schema, alias equality and over-restrictive A_LL equality rejection

Path: `analysis/sub_20260914/code_snapshot_20260917_union/solve_pros4_targets_uw1c.py`
SHA256: `f392ff5452639e5a681c0177019fecc37e574b8d9036a20c896a104904320480`

```text
  212  def load_scores(score_root, shards):
  213      """prompt_id -> (A_policy[K,8,8], A_ref[K,8,8], r_bt[K,2,8]), hash-verified.
  214  
  215      r_bt is the frozen BT head's scalar reward for the eight learner and eight
  216      comparator occurrences. It is loaded for every representation but only the
  217      bt_reward representation reads it, so nothing else changes.
  218      """
  219      scores, manifests = {}, []
  220      for shard in range(shards):
  221          directory = Path(score_root) / f"shard{shard}"
  222          complete = json.loads((directory / f"complete_shard{shard}.json").read_text())
  223          # A shard written before the A01 tensor-role fix stores the learner
  224          # triangle as A_policy and the cross block as A_ref. Those arrays have
  225          # the right dtype, the right shape and a matching sha256, so nothing
  226          # downstream can notice: the hash certifies that the bytes are the ones
  227          # that were written, not that they mean what this solver assumes. The
  228          # declared schema is the only discriminator, and a shard without it is
  229          # refused rather than solved into a different finite game.
  230          schema = complete.get("tensor_role_schema")
  231          # read through globals() so the guard still holds when this function is
  232          # lifted out of the module into a bare namespace, as audit harnesses do;
  233          # a missing constant must not become a missing check
  234          required = globals().get("REQUIRED_TENSOR_ROLE_SCHEMA", "lr_rr_v2")
  235          if schema != required:
  236              raise ValueError(
  237                  f"{directory} declares tensor_role_schema {schema!r}, and this solver "
  238                  f"requires {required!r}. A shard with no schema predates "
  239                  "the A01 fix: its A_policy is the learner triangle and its A_ref is the "
  240                  "learner-by-reference cross block, which is not the game Section 5.2 defines. "
  241                  "Re-score the panel with union_score_panel.py instead of reusing it.")
  242          manifests.append({"shard": shard, "gpm_teacher": complete["gpm_teacher"],
  243                            "bt_teacher": complete["bt_teacher"],
  244                            "tensor_role_schema": schema,
  245                            "reference_construction": complete["reference_construction"]})
  246          for path in sorted(directory.glob("chunk*.npz")):
  247              meta = json.loads(path.with_suffix("").with_suffix(".manifest.json").read_text()) \
  248                  if path.with_suffix("").with_suffix(".manifest.json").exists() else \
  249                  json.loads((directory / (path.stem + ".manifest.json")).read_text())
  250              if file_hash(path) != meta["sha256"]:
  251                  raise ValueError(f"Score chunk hash mismatch: {path}")
  252              arrays = np.load(path, allow_pickle=True)
  253              # The schema string is a declaration; this checks it. A shard that
  254              # says lr_rr_v2 but whose A_policy is not its own A_LR, or whose
  255              # A_ref is not its own A_RR, is mislabeled rather than merely old,
  256              # and mislabeled is the case a version tag cannot catch on its own.
  257              aliases = (("A_policy", "A_LR"), ("A_ref", "A_RR"))
  258              if all(name in arrays.files for pair in aliases for name in pair):
  259                  for alias, semantic in aliases:
  260                      if not np.array_equal(arrays[alias], arrays[semantic]):
  261                          raise ValueError(
  262                              f"{path} declares {required} but its {alias} is not its "
  263                              f"{semantic}; the tensor roles do not match the schema it "
  264                              "claims, so this shard cannot be solved as the declared game")
  265                  # A_LL must not be the policy game: if it were, the surplus would
  266                  # carry the sign defect the schema exists to rule out
  267                  if "A_LL" in arrays.files and np.array_equal(arrays["A_policy"],
  268                                                               arrays["A_LL"]):
  269                      raise ValueError(
  270                          f"{path} has A_policy equal to A_LL, which is the pre-fix "
  271                          "wiring under a post-fix label")
  272              expected_banks = {"A_policy": ["learner", "comparator"],
  273                                "A_ref": ["comparator", "comparator"],
  274                                "A_LL": ["learner", "learner"]}
  275              banks = meta.get("bank_ids")
  276              if banks is not None:
  277                  for name, want in expected_banks.items():
  278                      got = banks.get(name)
  279                      if got is not None and list(got) != want:
  280                          raise ValueError(
  281                              f"{path} records {name} drawn from {got}, not {want}")
  282              pids = [str(p) for p in arrays["prompt_ids"]]
  283              A, Aref = arrays["A_policy"], arrays["A_ref"]
  284              # r_bt is the scalar BT projection. The union contract uses direct
  285              # order-balanced probabilities for these rows and forbids that
  286              # projection, so the array may be absent; it is left as None rather
  287              # than filled with zeros, and only the bt_reward representation
  288              # reads it, where None raises instead of training on a placeholder.
  289              Rbt = arrays["r_bt"] if "r_bt" in arrays.files else None
  290              if A.shape[0] != len(OBJECTIVES) or A.shape[2:] != (POOL, POOL):
  291                  raise ValueError(f"Unexpected score tensor shape in {path}: {A.shape}")
  292              if Rbt is not None and (Rbt.shape[0] != len(OBJECTIVES)
  293                                      or Rbt.shape[2:] != (2, POOL)):
  294                  raise ValueError(f"Unexpected BT reward shape in {path}: {Rbt.shape}")
  295              for index, pid in enumerate(pids):
  296                  if pid in scores:
  297                      raise ValueError(f"Duplicate scored prompt {pid}")
  298                  scores[pid] = (A[:, index], Aref[:, index],
  299                                 None if Rbt is None else Rbt[:, index])
  300      identities = {object_hash(m["gpm_teacher"]) for m in manifests}
  301      if len(identities) != 1:
  302          raise ValueError("Score shards disagree on the frozen GPM teacher")
  303      return scores, manifests
```


## E05 — Panel label, content digest, row digest verification, serializer

Path: `analysis/sub_20260914/code_snapshot_20260917_union/solve_pros4_targets_uw1c.py`
SHA256: `f392ff5452639e5a681c0177019fecc37e574b8d9036a20c896a104904320480`

```text
   93  CANONICAL_DECIMALS_LEGACY = 10
   94  # The tensor-role schema this solver's game assumes: A_policy is the
   95  # learner-by-reference block and A_ref the reference triangle.
   96  REQUIRED_TENSOR_ROLE_SCHEMA = "lr_rr_v2"
   97  # The panel this module was generated for. build_panel_solvers.py rewrites it, so a
   98  # US/UT/UW run records its own panel instead of the UF-4 label of the source file.
   99  PANEL_LABEL = "UW1C"
  100  
  101  
  102  def split_pool_digest(pool, prompt_ids, roles=("learner", "comparator")):
  103      """A content digest of the pool rows this split actually uses.
  104  
  105      The fields it hashes are the immutable identity of each candidate: the
  106      prompt, the role, the occurrence index, the candidate id and the sha256 of
  107      the response text. Mutating one byte of one response changes its
  108      response_sha256 and therefore this digest, which is the property the audit
  109      asks for and the property the previous value did not have -- that one hashed
  110      `sorted(outputs)`, i.e. the split NAMES, so train and dev came out equal to
  111      each other and equal across panels no matter what the pool contained.
  112      """
  113      import hashlib
  114      h = hashlib.sha256()
  115      for pid in sorted(prompt_ids):
  116          entry = pool.get(pid)
  117          if entry is None:
  118              raise ValueError("split prompt %s is absent from the pool" % pid)
  119          for role in roles:
  120              for index in sorted(entry.get(role, {})):
  121                  row = entry[role][index]
  122                  # Recompute from the text rather than trusting the stored
  123                  # field. A row whose response was edited while its cached
  124                  # response_sha256 was left alone used to produce an unchanged
  125                  # digest, which is exactly the binding this digest is for. This
  126                  # function stays a pure function of content and never raises;
  127                  # the integrity complaint belongs to verify_pool_row_digests,
  128                  # which the solve calls, so a diagnostic caller can still get a
  129                  # digest of what the rows actually say.
  130                  identity = (str(pid), role, str(index),
  131                              str(row.get("candidate_id", "")),
  132                              hashlib.sha256(
  133                                  str(row.get("response", "")).encode("utf-8")).hexdigest())
  134                  h.update(("\x1f".join(identity) + "\x1e").encode())
  135      return h.hexdigest()
  136  
  137  
  138  def verify_pool_row_digests(pool, prompt_ids, roles=("learner", "comparator")):
  139      """Refuse a pool row whose cached response_sha256 disagrees with its text.
  140  
  141      Separated from split_pool_digest so that the digest is a pure function of
  142      content while the pipeline still stops on an internally inconsistent pool.
  143      load_pool checks each chunk's outer file hash, which catches a corrupted
  144      file but not a row whose text was edited and whose cached digest was left
  145      behind; that is the case this closes. Returns the number of rows checked.
  146      """
  147      import hashlib
  148      checked = 0
  149      for pid in sorted(prompt_ids):
  150          entry = pool.get(pid)
  151          if entry is None:
  152              raise ValueError("split prompt %s is absent from the pool" % pid)
  153          for role in roles:
  154              for index in sorted(entry.get(role, {})):
  155                  row = entry[role][index]
  156                  claimed = row.get("response_sha256")
  157                  if not claimed:
  158                      continue
  159                  actual = hashlib.sha256(
  160                      str(row.get("response", "")).encode("utf-8")).hexdigest()
  161                  if str(claimed) != actual:
  162                      raise ValueError(
  163                          "pool row %s/%s/%s carries response_sha256 %s but its text hashes "
  164                          "to %s; a stale cached digest is refused rather than hashed around"
  165                          % (pid, role, index, claimed, actual))
  166                  checked += 1
  167      return checked
  168  
  169  
  170  def quantize_canonical_row(row):
  171      """Serialize the solver masses so they survive the round trip exactly.
  172  
  173      prepare_nbpo_dataset reads these rows with a standard JSON parser and then
  174      checks target == log(w_a/c_a) - log(w_b/c_b) to 1e-9 absolute. An earlier
  175      version of this function rounded both masses to ten FIXED DECIMALS and
  176      rebuilt the target from the rounded values, which made the identity hold
  177      after the parse but changed the certified solution: a per-prompt Nash solve
  178      concentrates mass, its smallest masses reach the 1e-12 probability floor,
  179      and 1e-12 rounded to ten decimals is exactly 0. The row then carries a
  180      non-positive mass, which the canonical validator rejects outright, and any
  181      mass between 1e-10 and 1e-12 that did survive was quantized to a value whose
  182      log differs from the certified one by far more than the solver's own
  183      residual. That is the audit's A05: a target changed after it was certified.
  184  
  185      Fixed decimals are the wrong instrument for a quantity that spans twelve
  186      orders of magnitude. Python's float repr round-trips a float64 exactly and
  187      json.dumps emits it, so no rounding is needed at all: the masses are written
  188      as they were solved and the target is rebuilt from those same doubles. The
  189      identity then holds to the float64 relative error of a logarithm, roughly
  190      1e-16, comfortably inside the 1e-9 gate, with the certified solution intact.
  191  
  192      A mass that is genuinely non-positive is left alone here and refused by the
  193      validator, which is the correct outcome: it means the inner solve hit the
  194      boundary and the canonical log-ratio does not exist for that pair.
  195      """
  196      import math
  197  
  198      for key in ("nbpo_weight_a", "nbpo_weight_b"):
  199          if key in row:
  200              row[key] = float(row[key])
  201      if row.get("target_mode") == "canonical_logratio" and "nbpo_weight_a" in row:
  202          wa, wb = float(row["nbpo_weight_a"]), float(row["nbpo_weight_b"])
  203          ca = float(row.get("nbpo_center_a", 1.0 / POOL))
  204          cb = float(row.get("nbpo_center_b", 1.0 / POOL))
  205          if min(wa, wb) > 0:
  206              row["nbpo_logratio_target"] = math.log(wa / ca) - math.log(wb / cb)
  207              row["canonical_target_quantized_decimals"] = None
  208              row["canonical_target_serialization"] = "exact_float64_round_trip"
  209      return row
```


## E06 — Actual target-worker and manifests; pool digest check placement

Path: `analysis/sub_20260914/code_snapshot_20260917_union/solve_pros4_pw_nbpo_uw1c.py`
SHA256: `5d7f022f99f3f30048440a64ab12e137c80e2f34bb63f35e7abba45f4cfeb5b0`

```text
   63      _SHARED.update(A=A, Aref=Aref, beta=beta, eta=eta, weight_l1=weight_l1,
   64                     M=max_dual_calls, floor=floor)
   65      torch.set_num_threads(1)
   66  
   67  
   68  def _solve_one(x):
   69      """Solve prompt x alone: its own adversarial weights, its own certificate."""
   70      A = _SHARED["A"][:, x:x + 1]
   71      Aref = _SHARED["Aref"][:, x:x + 1]
   72      rep = AdaptiveGameRepresentation(
   73          torch.from_numpy(np.ascontiguousarray(A)), torch.from_numpy(np.ascontiguousarray(Aref)),
   74          uniform_policy(1, POOL),
   75          torch.full((len(OBJECTIVES),), _SHARED["beta"], dtype=torch.float64),
   76          reference_construction="independent_samples")
   77      result = solve_finite_pool(
   78          rep, "nash", eta=_SHARED["eta"], inner_solver="exact",
   79          dual_solver="root", dual_tol=1e-10, M=_SHARED["M"], inner_workers=1,
   80          probability_floor=_SHARED["floor"], log_every=0)
   81      certificate = validate_finite_pool_solution(result)
   82      return (x,
   83              result.pi.numpy().astype(np.float64),
   84              result.nu_update.numpy().astype(np.float64),
   85              result.weights.numpy().astype(np.float64),
   86              result.target_log_ratio.numpy().astype(np.float64),
   87              float(result.target_log_ratio_check()),
   88              bool(certificate.get("certified", False)),
```

```text
  132      for split, (score_root, pool_root, split_file) in (
  133              ("train", (ROOT / "scores/uw1c", ROOT / "pools/uw1c", "policy_train")),
  134              ("dev", (ROOT / "scores/uw1c", ROOT / "pools/uw1c", "policy_dev"))):
  135          split_start = time.monotonic()
  136          scores, score_manifests = base.load_scores(str(score_root), args.shards)
  137          pool, pool_settings = base.load_pool(str(pool_root), args.shards)
  138          with (ROOT / "splits/uw_v1cp" / f"{split_file}.jsonl").open() as stream:
  139              pids = [json.loads(line)["prompt_id"] for line in stream if line.strip()]
  140          if args.limit:
  141              pids = pids[:args.limit]
  142          missing = [p for p in pids if p not in scores or p not in pool]
  143          if missing:
  144              raise ValueError(f"{split}: {len(missing)} prompts have no scores or no pool")
  145  
  146          A = np.stack([scores[p][0] for p in pids], axis=1)
  147          Aref = np.stack([scores[p][1] for p in pids], axis=1)
```

```text
  198          per_prompt_path = out / f"{split}_per_prompt.npz"
  199          np.savez_compressed(per_prompt_path, pi=pi, weights=weights, g=g,
  200                              min_surplus=min_surplus, identity_residual=identity,
  201                              prompt_ids=np.array(pids, dtype=object))
  202  
  203          # A hash-bound solution artifact at the path the shared job generator
  204          # expects. The global one is not written because per-prompt weights do
  205          # not fit its (K,) weight field, but the generator only needs SOME file
  206          # whose hash pins the solution, and patching the generator would touch
  207          # code the published arms depend on. This file pins the real thing: the
  208          # per-prompt npz that every target in this set was built from.
  209          solver_dir = out / split / "solver"
  210          base.write_json(solver_dir / "solution.json", {
  211              "target_mode": "canonical_logratio", "target_column": "nbpo_logratio_target",
  212              "target_units": "final_logratio_change", "eta_already_included": True,
  213              "representation": "adaptive_game", "aggregation": "prompt_wise_nash",
  214              "weights_scope": "per prompt; there is no shared dual",
  215              "per_prompt_artifact": str(per_prompt_path),
  216              "per_prompt_artifact_sha256": base.file_hash(per_prompt_path),
  217              "n_prompts": len(pids), "split": split,
  218              "all_certified": bool(certified.all()),
  219              "max_identity_residual": float(np.abs(identity).max()),
  220              "min_surplus_mean": float(min_surplus.mean()),
  221              "min_surplus_negative_prompts": int((min_surplus < 0).sum()),
  222              "weight_l1_matched": weight_l1, "beta": args.beta, "eta": args.eta,
  223              "solver_source_sha256": base.file_hash(__file__), **shared_meta})
  224          solver_hash = base.file_hash(solver_dir / "solution.json")
  225          provenance = {"solver_artifact_sha256": solver_hash,
  226                        "solver_hash": solver_hash,
  227                        "target_artifact_hash": base.file_hash(per_prompt_path),
  228                        "representation": "adaptive_game", "aggregation": "prompt_wise_nash",
  229                        "split": split, "panel": getattr(base, "PANEL_LABEL", "UF-4"), **shared_meta}
  230          betas = np.full(len(OBJECTIVES), args.beta)
  231  
  232          def pair_rows():
  233              for x, pid in enumerate(pids):
  234                  learners = {str(i): {pid: {**pool[pid]["learner"][i],
  235                                             "generated_text": pool[pid]["learner"][i]["response"]}}
  236                              for i in range(POOL)}
  237                  rows = build_rows(
  238                      [pid], list(OBJECTIVES), A[:, x:x + 1], nu[:, x:x + 1], weights[:, x],
  239                      betas, learners, None, np.random.default_rng(42), "canonical_logratio",
  240                      meta, provenance,
  241                      canonical_data={"g": g[x:x + 1], "p_star": pi[x:x + 1],
  242                                      "p_t": np.full((1, POOL), 1.0 / POOL)})
  243                  for row in rows:
```

```text
  254          pair_path = out / "pairs" / f"{split}.jsonl"
  255          base.write_jsonl(pair_path, pair_rows())
  256          outputs[split] = {"n_prompts": len(pids), "n_pairs": len(pids) * 28,
  257                            "pairs_path": str(pair_path), "pairs_sha256": base.file_hash(pair_path),
  258                            "per_prompt_weight_mean": [float(v) for v in weights.mean(axis=1)],
  259                            "per_prompt_weight_sd": [float(v) for v in weights.std(axis=1, ddof=1)],
  260                            "effective_objectives_mean": float(np.mean(
  261                                (weights.sum(axis=0) ** 2) / (weights ** 2).sum(axis=0))),
  262                            "min_surplus_mean": float(min_surplus.mean()),
  263                            "min_surplus_negative_prompts": int((min_surplus < 0).sum()),
  264                            "max_identity_residual": float(np.abs(identity).max()),
  265                            "all_certified": bool(certified.all()),
  266                            "solver_solution_sha256": solver_hash,
  267                            "seconds": time.monotonic() - split_start}
  268          (getattr(base, "verify_pool_row_digests", None) or (lambda *a: 0))(pool, pids)
  269          split_digests[split] = digest_of(pool, pids)
  270          base.write_json(out / split / "complete.json", outputs[split])
  271          print(json.dumps({k: v for k, v in outputs[split].items()
  272                            if k not in ("pairs_path", "pairs_sha256")}), flush=True)
  273  
  274      prov = {**shared_meta, "objectives": list(OBJECTIVES), "panel": getattr(base, "PANEL_LABEL", "UF-4"),
  275              "aggregation": "prompt_wise_nash", "beta": args.beta, "eta": args.eta,
  276              "weight_l1": weight_l1, "splits": outputs,
  277              # content digests of the rows each split actually uses, not of the
  278              # split NAMES: the previous value hashed sorted(outputs), so train and
  279              # dev were equal to each other and unchanged by the pool's contents
  280              "train_pool_sha256": split_digests.get("train"),
  281              "dev_pool_sha256": split_digests.get("dev"),
  282              "pool_digest_fields": ("prompt_id, role, occurrence index, candidate_id, "
  283                                     "response_sha256"),
  284              "dev_note": ("no shared dual: each prompt fits its own adversarial weights on "
  285                           "whichever split it is in, so dev measures neural generalisation "
  286                           "but not generalisation of a dual fitted on train")}
```


## E07 — New README assertions and explicit historical ungated status

Path: `analysis/sub_20260914/code_snapshot_20260917_union/README.md`
SHA256: `69b12858ec7719f4e15f8a3c01515352f1c3c374982b2f683857dccab9c1f3e9`

```text
  109  ## Third audit pass, 2026-09-18 (97e03c0)
  110  
  111  Six of the previous round's items were confirmed resolved. Four remained, and
  112  all four are closed here.
  113  
  114  **A04, the one that mattered: Algorithm 1's acceptance step was never wired into
  115  the prompt-wise training path.** The traced path was job generator —
  116  `torch.distributed.run` — `run_mnpo` — `train()` — `save_model()`, with
  117  nothing between or after it that evaluated, rejected or promoted a candidate. A
  118  certified solver residual does not substitute: it certifies the finite target,
  119  not the policy fitted to it. `nbpo_local_gate.py` now performs that step. It
  120  builds ONE REPRESENTATION PER PROMPT — the batched `game_values` divides by
  121  the prompt count, so a batch containing +.2 and -.1 averages to +.05 and passes
  122  while the compromise failed on one of those prompts — and applies three
  123  declared checks: every surplus finite, the fraction of development prompts
  124  positive on every objective at least `--coverage-min`, and the held-out nMSE at
  125  most `--nmse-max`. On a pass it writes a versioned accepted directory carrying
  126  the checkpoint's fingerprint; on a failure it promotes nothing and returns the
  127  parent. `scripts/nbpo/eval_game_value.py` stays global and now says so in its
  128  own docstring, because global is correct for the Global Nash control.
  129  
  130  The gate exists from 2026-09-18 and is not retroactive. Every number already
  131  reported came from an ungated projection, and a later gate cannot change that;
  132  anything described as gated must name a promotion record this file wrote.
  133  
  134  **C01: the schema tag was a declaration with nothing behind it.** A shard could
  135  say `lr_rr_v2` while its `A_policy` was not its own `A_LR`. The loader now checks
  136  the aliases against the semantic arrays, refuses a shard whose `A_policy` equals
  137  its `A_LL` (the pre-fix wiring under a post-fix label), and checks the recorded
  138  bank ids.
  139  
  140  **C02: the pool digest trusted a cached hash.** A row whose response text was
  141  edited while its `response_sha256` was left behind produced an unchanged digest.
  142  `split_pool_digest` now recomputes from the text, and the refusal lives in a
  143  separate `verify_pool_row_digests` that the solve calls first — so the digest
  144  stays a pure function of content while an internally inconsistent pool still
  145  stops the run.
  146  
  147  **C03: the UW1 base still labelled itself UF-4.** It is `UW1` now, and the
  148  generator's substitution anchor follows.
  149  
  150  ## The prompt-wise NBPO line
  151  
```


## E08 — Campaign training job construction

Path: `analysis/sub_20260914/code_snapshot_20260917_union/make_pros_train_jobs.py`
SHA256: `bcabadf496632acbcefd6334b84aae5e95f5a85f129885cd4b526f0ee3feba62`

```text
  126          replacements["model_name_or_path: /work/models/bases/Llama-3.1-8B-Instruct"] = \
  127              "model_name_or_path: %s" % args.model
  128          # NBPO reads the reference policy's log-probabilities. Left at the base
  129          # recipe's Llama path, Qwen token ids overflow a 128256-row embedding and
  130          # the CUDA lookup asserts. The reference for these arms is the same
  131          # untrained base the pool was sampled from.
  132          replacements["nbpo_reference_model_path: /work/models/bases/Llama-3.1-8B-Instruct"] = \
  133              "nbpo_reference_model_path: %s" % args.model
  134          replacements["model_revision: 0e9e39f249a16976918f6564b8830bc894c89659"] = \
  135              "model_revision: %s" % args.model_revision
  136      config = base
  137      applied = []
  138      for old, new in replacements.items():
  139          if old not in config:
  140              raise ValueError(f"Recipe anchor missing from the resolved base config: {old!r}")
  141          config = config.replace(old, new)
  142          applied.append({"from": old, "to": new})
  143      config_path = ROOT / "configs" / f"{args.arm}.yaml"
  144      config_path.parent.mkdir(parents=True, exist_ok=True)
  145      if config_path.exists() and config_path.read_text() != config:
  146          raise ValueError(f"Refusing to overwrite {config_path} with different content")
  147      config_path.write_text(config)
  148  
  149      spec = {
  150          "job_id": f"{args.job_prefix}_train_{args.arm}",
  151          "priority": args.priority,
  152          "gpus": 4,
  153          "cwd": "/work/nbpo_repair_20260909/code",
  154          "env": {"PYTHONPATH": "/work/nbpo_repair_20260909/deps_train:/work/nbpo_repair_20260909/code",
  155                  "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "4",
  156                  "MNPO_DISABLE_APEX": "1", "HF_HUB_OFFLINE": "1",
  157                  "TOKENIZERS_PARALLELISM": "false", "WANDB_MODE": "disabled",
  158                  "VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
  159          "command": ["python3", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
  160                      "--nproc_per_node=4", "-m", "mnpo_scripts.run_mnpo", str(config_path)],
  161          "timeout_s": 86400,
  162          "artifacts": [f"{ROOT}/arms/{args.arm}/config.json"],
  163          # The spec names the config by path, so without this a changed recipe
  164          # leaves the spec byte-identical and a queue keyed on spec bytes would
  165          # never notice that this is a different run.
  166          "config_sha256": file_hash(config_path),
  167          "dataset_manifest_sha256": manifest,
  168          "solver_artifact_sha256": solver,
  169      }
  170      queue_path = ROOT / "jobs" / "queue" / f"{args.priority}_train_{args.arm}.json"
  171      queue_path.write_text(json.dumps(spec, indent=2) + "\n")
  172  
  173      record = {"arm": args.arm, "targets": args.targets, "dataset": dataset,
  174                "effective_batch_note": ("per_device x grad_accum x 4 GPUs; PROSPER "
  175                                         "Table 7 uses 128"),
  176                "loss_type": args.loss_type, "nbpo_target_mode": target_mode,
  177                "dataset_manifest_sha256": manifest, "solver_artifact_sha256": solver,
  178                "config_path": str(config_path), "config_sha256": file_hash(config_path),
  179                "base_config": str(BASE_CONFIG), "base_config_sha256": file_hash(BASE_CONFIG),
  180                "recipe_diff_vs_resolved_base": applied,
  181                "horizon_note": ("1250 updates at 32 pairs is the planned exposure of about four "
  182                                 "pairs per prompt over 10,000 prompts. It is the plan's starting "
  183                                 "horizon, not a horizon shown to be optimal on this data."),
  184                "queue_spec": str(queue_path)}
  185      out = ROOT / "provenance" / f"train_job_{args.arm}.json"
  186      out.write_text(json.dumps(record, indent=2) + "\n")
  187      print(json.dumps({"queued": spec["job_id"], "config": str(config_path),
  188                        "dataset_manifest": manifest[:16], "solver": solver[:16]}), flush=True)
  189  
  190  
```


## E09 — Panel training stage construction

Path: `analysis/sub_20260914/code_snapshot_20260917_union/panel_stage2.py`
SHA256: `e0528503bf22fa970cc2d9a963b35a183bd3a970719240c2d3e6b87c6667734b`

```text
  309          # The reference-handling and immutable-token plumbing stays on for every
  310          # arm, exactly as the verified UW baseline configs have it: the soft-label
  311          # and PROSPER losses also read reference log-ratios and must score the
  312          # pool's own tokens. Only the target-column fields are NBPO-specific.
  313          for key in ("nbpo_expected_dataset_manifest_sha256",
  314                      "nbpo_expected_solver_artifact_sha256"):
  315              cfg.pop(key, None)
  316          if loss != "nbpo":
  317              for key in ("nbpo_target_mode", "nbpo_target_column", "nbpo_target_units",
  318                          "nbpo_eta_already_included"):
  319                  cfg.pop(key, None)
  320          if loss == "dpo_soft":
  321              cfg.pop("eta", None)              # dpo_soft has no proximal step
  322          if loss == "nbpo":
  323              # pin the dataset manifest and the solver solution this arm was built
  324              # from, so a config cannot be pointed at a different solve later
  325              done = json.loads(Path(UF, "targets", "%s_%s" % (panel, arm),
  326                                     "complete.json").read_text())
  327              man = done.get("dataset_manifest_sha256")
  328              sol = (done["splits"]["train"].get("solver_solution_sha256")
  329                     if "splits" in done else None)
  330              if not man or not sol:
  331                  raise SystemExit("%s: solve record has no manifest/solution hash" % arm)
  332              cfg["nbpo_expected_dataset_manifest_sha256"] = man
  333              cfg["nbpo_expected_solver_artifact_sha256"] = sol
  334          config_dir.mkdir(parents=True, exist_ok=True)
  335          path = config_dir / ("%s_%s.yaml" % (panel, arm))
  336          path.write_text(yaml.safe_dump(cfg, sort_keys=True))
  337          spec = {
  338              "job_id": "%s_train_%s" % (panel, arm), "priority": 120, "gpus": 4,
  339              "depends_on": [], "cwd": "%s/code" % DEPS,
  340              "env": {"PYTHONPATH": "%s/deps_train:%s/code" % (DEPS, DEPS),
  341                      "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
  342                      "MKL_NUM_THREADS": "4", "MNPO_DISABLE_APEX": "1",
  343                      "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
  344                      "WANDB_MODE": "disabled",
  345                      "VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
  346              "timeout_s": 86400,
  347              "command": ["python3", "-m", "torch.distributed.run", "--standalone",
  348                          "--nnodes=1", "--nproc_per_node=4", "-m",
  349                          "mnpo_scripts.run_mnpo", str(path)],
  350              "artifacts": [str(UF / "arms" / ("%s_%s" % (panel, arm)) / "config.json")],
  351              "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
  352              "note": ("%s panel, %s arm: %d train rows, %d steps for %d epochs at %d "
  353                       "sequences per step" % (panel.upper(), arm, rows, steps, epochs,
  354                                               tokens_per_step)),
  355          }
  356          queue_dir.mkdir(parents=True, exist_ok=True)
  357          (queue_dir / ("120_%s_train_%s.json" % (panel, arm))).write_text(
  358              json.dumps(spec, indent=1) + "\n")
  359          queued[arm] = {"rows": rows, "max_steps": steps, "config": str(path),
  360                         "loss_type": loss}
  361          print(json.dumps({arm: queued[arm]}), flush=True)
  362      return queued
  363  
  364  
  365  if __name__ == "__main__":
  366      raise SystemExit(main())
```


## E10 — Training and checkpoint saving

Path: `mnpo_scripts/run_mnpo.py`
SHA256: `def0546b64f7d2d5dd6360a7a3ff6995c126a06ff0e63326c965443cdd05d36a`

```text
  300      # =====================================================================================
  301      # Instantiate MNPOTrainer
  302      # =====================================================================================
  303      trainer = MNPOTrainer(
  304          model=model,
  305          args=training_args,
  306          train_dataset=train_dataset,
  307          eval_dataset=eval_dataset,
  308          tokenizer=tokenizer,
  309          peft_config=get_peft_config(model_args),
  310      )
  311  
  312      # The frozen proximal centre, loaded AFTER the trainer so it shares the
  313      # accelerator's device placement. It is a separate copy of pi_t held in eval
  314      # mode with gradients off: it is never the learner detached, and it is never
  315      # re-synced to the learner during training. Its only job is to be forwarded
  316      # through the same collated batch as the policy, so that h is a difference of
  317      # two log-probabilities computed on the identical kernel path.
  318      if (getattr(training_args, "nbpo_online_reference", False)
  319              or getattr(training_args, "nbpo_eval_online_reference", False)):
  320          ref_path = getattr(training_args, "nbpo_reference_model_path", "") or \
  321              model_args.model_name_or_path
  322          logger.info(f"*** Loading frozen NBPO reference (pi_t) from {ref_path} ***")
  323          ref = AutoModelForCausalLM.from_pretrained(
  324              ref_path, torch_dtype=next(trainer.model.parameters()).dtype, use_cache=False,
  325              revision=model_args.model_revision, attn_implementation=model_args.attn_implementation,
  326              trust_remote_code=model_args.trust_remote_code)
  327          ref.eval()
  328          for prm in ref.parameters():
  329              prm.requires_grad_(False)
  330          trainer.nbpo_reference_model = ref.to(trainer.accelerator.device)
  331          logger.info("*** NBPO online reference active: pi_t forwarded per batch ***")
  332  
  333      if training_args.nbpo_require_fp32_optimizer:
  334          from mnpo_scripts.nbpo_runtime import NBPOPrecisionCallback
  335          trainer.add_callback(NBPOPrecisionCallback(trainer))
  336  
  337      # =====================================================================================
  338  
  339      if os.environ.get("MNPO_EVAL_ONLY", "").lower() in {"1", "true", "yes"}:
  340          if eval_dataset is None:
  341              raise ValueError("MNPO_EVAL_ONLY requires the explicitly selected dev split")
  342          metrics = trainer.evaluate()
  343          trainer.log_metrics("eval", metrics)
  344          trainer.save_metrics("eval", metrics)
  345          return
  346  
  347      ###############
  348      # Training loop
  349      ###############
  350      checkpoint = None
  351      if training_args.resume_from_checkpoint is not None:
  352          checkpoint = training_args.resume_from_checkpoint
  353      elif last_checkpoint is not None:
  354          checkpoint = last_checkpoint
  355      train_result = trainer.train(resume_from_checkpoint=checkpoint)
  356      metrics = train_result.metrics
  357      metrics["train_samples"] = len(train_dataset)
  358      trainer.log_metrics("train", metrics)
  359      trainer.save_metrics("train", metrics)
  360      trainer.save_state()
  361  
  362      logger.info("*** Training complete ***")
  363      if training_args.nbpo_profile_updates > 0:
  364          logger.info("Disposable runtime profile complete at %d updates; scheduler horizon was %d. No model export.",
  365                      trainer.state.global_step, training_args.max_steps)
  366          return
  367  
  368      skip_final_save = os.environ.get("MNPO_SKIP_FINAL_SAVE", "").lower() in {"1", "true", "yes"}
  369      if skip_final_save:
  370          logger.info("*** Skip final model save because MNPO_SKIP_FINAL_SAVE is set ***")
  371          if trainer.accelerator.is_main_process:
  372              trainer.tokenizer.save_pretrained(training_args.output_dir)
  373              trainer.model.config.use_cache = True
  374              trainer.model.config.save_pretrained(training_args.output_dir)
  375          logger.info("*** Training complete! ***")
  376          return
  377  
  378      ##################################
  379      # Save model and create model card
  380      ##################################
  381      logger.info("*** Save model ***")
  382      trainer.save_model(training_args.output_dir)
  383      logger.info(f"Model saved to {training_args.output_dir}")
  384  
  385      # Add this step to explicitly save the tokenizer.
  386      if trainer.accelerator.is_main_process:
  387          trainer.tokenizer.save_pretrained(training_args.output_dir)
  388          logger.info(f"Tokenizer saved to {training_args.output_dir}")
  389  
  390      kwargs = {
  391          "finetuned_from": model_args.model_name_or_path,
  392          "dataset": list(data_args.dataset_mixer.keys()),
  393          "dataset_tags": list(data_args.dataset_mixer.keys()),
  394          "tags": ["alignment-handbook", "mnpo"],  # MODIFIED
  395      }
  396      if trainer.accelerator.is_main_process:
  397          trainer.create_model_card(**kwargs)
  398          trainer.model.config.use_cache = True
  399          trainer.model.config.save_pretrained(training_args.output_dir)
  400  
  401      ##########
  402      # Evaluate
  403      ##########
  404      # if training_args.do_eval:
  405      #     logger.info("*** Evaluate ***")
  406      #     metrics = trainer.evaluate()
  407      #     if "test" in raw_datasets:
  408      #         metrics["eval_samples"] = len(raw_datasets["test"])
  409      #     trainer.log_metrics("eval", metrics)
  410      #     trainer.save_metrics("eval", metrics)
```


## E11 — The actual helper dependency, pool tokens and log-probabilities

Path: `analysis/uf4_20260910/code_snapshot_20260917/diag_neural_pools.py`
SHA256: `2e5b06529cd8f4c30c85553e39b7f3bbadc52f2fd00ceb887cdff5dd60c7db29`

```text
    1  """Importance-reweight the learner pool by a fitted checkpoint, on the frozen panel.
    2  
    3  For every panel prompt and each of its eight learner occurrences this computes
    4  the sequence log-likelihood under the fitted policy and under the proximal
    5  centre pi_t, using the pool's own cached training tokenization -- the same
    6  input_ids, attention mask and label mask the trainer consumed, summed over
    7  non-masked response tokens -- and then
    8  
    9      p_neural(i) proportional to p_t(i) exp{log pi_theta(y_i) - log pi_t(y_i)},
   10  
   11  normalized by log-sum-exp. This is an importance diagnostic on the sampled
   12  support, not the policy's distribution over responses. Raw-likelihood
   13  normalization and length-normalized scores are deliberately not used.
   14  
   15  Also reports KL(p* || p_neural), KL(p* || p_t), total variation and the
   16  log-ratio scale, which are the projection columns of the pilot table.
   17  """
   18  from __future__ import annotations
   19  
   20  import argparse
   21  import glob
   22  import json
   23  from pathlib import Path
   24  
   25  import numpy as np
   26  import torch
   27  
   28  ROOT = Path("/work/uf4_20260910")
   29  DIAG = ROOT / "analysis/diag_20260914"
   30  
   31  
   32  def load_pool_rows(pool, wanted):
   33      """candidate_id -> the cached training tokenization of that occurrence."""
   34      out = {}
   35      for path in sorted(glob.glob(str(ROOT / "pools" / pool / "shard*/chunk*.jsonl"))):
   36          with open(path) as stream:
   37              for line in stream:
   38                  r = json.loads(line)
   39                  if r["candidate_id"] in wanted:
   40                      out[r["candidate_id"]] = {"input_ids": r["input_ids"],
   41                                                "labels": r["labels"],
   42                                                "attention_mask": r["attention_mask"],
   43                                                "n_tokens": r["n_tokens"]}
   44      return out
   45  
   46  
   47  @torch.no_grad()
   48  def sequence_logprobs(model, rows, order, device, batch=4):
   49      """Sum of log p(token) over non-masked label positions, one value per row."""
   50      values = []
   51      for start in range(0, len(order), batch):
   52          chunk = order[start:start + batch]
   53          width = max(len(rows[c]["input_ids"]) for c in chunk)
   54          ids = torch.zeros(len(chunk), width, dtype=torch.long)
   55          att = torch.zeros(len(chunk), width, dtype=torch.long)
   56          lab = torch.full((len(chunk), width), -100, dtype=torch.long)
   57          for k, cid in enumerate(chunk):
   58              r = rows[cid]
   59              n = len(r["input_ids"])
   60              ids[k, :n] = torch.tensor(r["input_ids"], dtype=torch.long)
   61              att[k, :n] = torch.tensor(r["attention_mask"], dtype=torch.long)
   62              lab[k, :n] = torch.tensor(r["labels"], dtype=torch.long)
   63          ids, att, lab = ids.to(device), att.to(device), lab.to(device)
   64          logits = model(input_ids=ids, attention_mask=att).logits.float()
   65          logp = torch.log_softmax(logits[:, :-1], dim=-1)
   66          target = lab[:, 1:]
   67          mask = target != -100
   68          gathered = torch.gather(logp, 2, target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
   69          values.extend((gathered * mask).sum(dim=1).tolist())
   70      return values
   71  
   72  
   73  def main():
```


## E12 — Hard-coded pod selftest; fit check omitted and textual pass/fail reporting

Path: `analysis/sub_20260914/code_snapshot_20260917_union/nbpo_local_gate_selftest.py`
SHA256: `5d96083801cf0306d24c42c9a18c5266d66adce14b2655ee523b61f1a3c6cd22`

```text
    1  """A04 acceptance: the gate accepts, rejects and never promotes on a bad value.
    2  
    3  Runs the real nbpo_local_gate.py against the real uw1c dev solve and the real
    4  score tensors, with the forward pass replaced by supplied log-probabilities, so
    5  the decision logic and the promotion filesystem behaviour are exercised without
    6  competing for a card. Three cases:
    7  
    8    accept   a candidate whose mass moves toward each prompt's solved p*
    9    reject   a candidate whose mass moves away from it
   10    refuse   a candidate with a non-finite log-probability
   11  
   12  and in every rejecting case the parent must be what the stage returns and
   13  nothing may be promoted.
   14  """
   15  import json, subprocess, sys, tempfile
   16  from pathlib import Path
   17  import numpy as np
   18  
   19  CODE = "/work/sub_20260914/code"
   20  T = "/work/uf4_20260910/targets/uw1c_pw_nbpo"
   21  POOL = 8
   22  z = np.load(T + "/dev_per_prompt.npz", allow_pickle=True)
   23  pids = [str(p) for p in z["prompt_ids"]]
   24  pi = np.asarray(z["pi"], dtype=np.float64)
   25  
   26  def write_logprobs(path, mode):
   27      """parent flat; candidate tilted toward (or away from) the solved p*."""
   28      parent, cand = {}, {}
   29      for x, pid in enumerate(pids):
   30          ps = np.clip(pi[x], 1e-12, None)
   31          for i in range(POOL):
   32              cid = "%s:learner:%d" % (pid, i)
   33              parent[cid] = 0.0
   34              if mode == "toward":
   35                  cand[cid] = float(np.log(ps[i]) - np.log(1.0 / POOL))
   36              elif mode == "away":
   37                  cand[cid] = float(-(np.log(ps[i]) - np.log(1.0 / POOL)))
   38              elif mode == "nonfinite":
   39                  cand[cid] = float("nan") if i == 0 else 0.0
   40      Path(path).write_text(json.dumps({"parent": parent, "candidate": cand}))
   41  
   42  def run(mode, promote_dir, coverage_min="0.5"):
   43      with tempfile.TemporaryDirectory() as d:
   44          lp = str(Path(d) / "lp.json"); write_logprobs(lp, mode)
   45          out = str(Path(d) / "gate.json")
   46          link = str(Path(promote_dir) / ("policy_%s" % mode))
   47          r = subprocess.run(
   48              [sys.executable, CODE + "/nbpo_local_gate.py",
   49               "--candidate", "/work/uf4_20260910/arms/uw1c_pw_nbpo/checkpoint-70",
   50               "--parent", "/work/models/bases/Qwen2.5-7B-Instruct",
   51               "--targets", T, "--pool", "uw1c",
   52               "--scores", "/work/uf4_20260910/scores/uw1c",
   53               "--solver-module", CODE + "/solve_pros4_targets_uw1c.py",
   54               "--split", "dev", "--beta", "0.25",
   55               "--coverage-min", coverage_min, "--nmse-max", "1.0",
   56               "--logprobs", lp, "--promote-to", link, "--out", out],
   57              capture_output=True, text=True)
   58          rec = json.loads(Path(out).read_text()) if Path(out).exists() else None
   59          return r.returncode, rec, Path(link)
   60  
   61  fail = []
   62  with tempfile.TemporaryDirectory() as promote:
   63      for mode, want_accept in (("toward", True), ("away", False), ("nonfinite", False)):
   64          rc, rec, link = run(mode, promote)
   65          if rec is None:
   66              fail.append("%s: no record written (rc=%d)" % (mode, rc)); continue
   67          got = rec["accepted"]
   68          print("  %-10s accepted=%-5s coverage=%.3f nonfinite=%d rc=%d promoted=%s"
   69                % (mode, got, rec["measured"]["coverage"],
   70                   rec["measured"]["nonfinite_prompts"], rc,
   71                   bool(rec.get("promotion"))))
   72          if got != want_accept:
   73              fail.append("%s: accepted=%s, wanted %s" % (mode, got, want_accept))
   74          if not got:
   75              if rec["policy_after_this_stage"] != rec["parent"]:
   76                  fail.append("%s: a rejection did not return the parent" % mode)
   77              if rec.get("promotion") is not None or link.exists():
   78                  fail.append("%s: a rejection promoted something" % mode)
   79              if rc == 0:
   80                  fail.append("%s: a rejection exited 0" % mode)
   81          else:
   82              if rec["policy_after_this_stage"] != rec["candidate"]:
   83                  fail.append("%s: an acceptance did not return the candidate" % mode)
   84              if not link.is_symlink():
   85                  fail.append("%s: an acceptance wrote no symlink" % mode)
   86              else:
   87                  acc = link.resolve() / "ACCEPTED.json"
   88                  if not acc.exists():
   89                      fail.append("%s: the accepted dir has no record" % mode)
   90                  else:
   91                      a = json.loads(acc.read_text())
   92                      if not a.get("fingerprint"):
   93                          fail.append("%s: the accepted record has no fingerprint" % mode)
   94      # a threshold above what the good candidate achieves must reject it
   95      rc, rec, link = run("toward", promote, coverage_min="1.01")
   96      print("  %-10s accepted=%-5s (coverage-min 1.01)" % ("threshold", rec["accepted"]))
   97      if rec["accepted"]:
   98          fail.append("an unreachable coverage threshold still accepted")
   99  
  100  print()
  101  print("A04 VERIFIED" if not fail else "STILL BROKEN: " + "; ".join(fail))
```


## E13 — Correct LL/LR/RR scorer retained

Path: `analysis/sub_20260914/code_snapshot_20260917_union/union_score_panel.py`
SHA256: `e3bd9c1c2486715f0e1e4a07efe15419fc499507f5d2c633039cde6b2d17da83`

```text
  140                      if r["status"] != "ok":
  141                          continue
  142                      key = (r["prompt_id"], r["rubric"], r["role_i"], r["i"],
  143                             r["role_j"], r["j"])
  144                      obs[key][r["order"]] = float(r["value_for_i"])
  145  
  146      learner_pairs = list(itertools.combinations(range(POOL), 2))
  147      cross_pairs = [(i, j) for i in range(POOL) for j in range(POOL)]
  148  
  149      def resolve(pid, rubric, role_i, i, role_j, j):
  150          """Order-balanced probability, or None when an order is missing."""
  151          got = obs.get((pid, rubric, role_i, i, role_j, j))
  152          if got is None or 0 not in got or 1 not in got:
  153              return None
  154          return 0.5 * (got[0] + got[1])
  155  
  156      tensors, identity_ties, refdis = {}, Counter(), {}
  157      dropped = Counter()
  158      prompts = sorted({k[0] for k in obs})
  159      for pid in prompts:
  160          if pid not in pool:
  161              dropped["prompt_absent_from_pool"] += 1
  162              continue
  163          A_LL = np.zeros((K, POOL, POOL))
  164          A_LR = np.zeros((K, POOL, POOL))
  165          A_RR = np.zeros((K, POOL, POOL))
  166          ok = True
  167          for k, rubric in enumerate(ITEMS):
  168              for i, j in learner_pairs:
  169                  p = resolve(pid, rubric, "learner", i, "learner", j)
  170                  if p is None:
  171                      ok = False; dropped["learner_pair_unresolved"] += 1; break
  172                  A_LL[k, i, j] = p - 0.5
  173                  A_LL[k, j, i] = 0.5 - p
  174              if not ok:
  175                  break
  176              for i, j in cross_pairs:
  177                  p = resolve(pid, rubric, "learner", i, "comparator", j)
  178                  if p is None:
  179                      ok = False; dropped["cross_pair_unresolved"] += 1; break
  180                  if (pool[pid]["learner"].get(i) ==
  181                          pool[pid]["comparator"].get(j) is not None):
  182                      identity_ties[pid] += 1
  183                  A_LR[k, i, j] = p - 0.5
  184              if not ok:
  185                  break
  186              # the reference triangle is now required: d = V_beta(mu) is defined
  187              # on it, so a prompt without it has no disagreement point and is
  188              # dropped rather than carried with a silently wrong d
  189              for i, j in learner_pairs:
  190                  p = resolve(pid, rubric, "comparator", i, "comparator", j)
  191                  if p is None:
  192                      ok = False; dropped["reference_pair_unresolved"] += 1; break
  193                  A_RR[k, i, j] = p - 0.5
  194                  A_RR[k, j, i] = 0.5 - p
  195              if not ok:
  196                  break
  197          if not ok:
  198              continue
  199          idx = np.arange(POOL)
  200          A_LL[:, idx, idx] = 0.0
  201          A_RR[:, idx, idx] = 0.0
  202          for name, M in (("learner", A_LL), ("reference", A_RR)):
  203              if np.abs(M + np.swapaxes(M, -1, -2)).max() > 1e-12:
  204                  raise SystemExit("%s payoff not antisymmetric for %s" % (name, pid))
  205          if max(np.abs(A_LL).max(), np.abs(A_LR).max(),
  206                 np.abs(A_RR).max()) > 0.5 + 1e-12:
  207              raise SystemExit("payoff outside [-0.5, 0.5] for %s" % pid)
  208          # the same scalar the old file reported, so the two rounds stay comparable
  209          iu = np.triu_indices(POOL, 1)
  210          refdis[pid] = float(np.mean(np.abs(A_RR[:, iu[0], iu[1]])))
  211          tensors[pid] = (A_LL, A_LR, A_RR)
  212  
  213      pids = sorted(tensors)
  214      if not pids:
  215          raise SystemExit("no prompt survived scoring")
  216      out = UF / "scores" / args.out
  217      out.mkdir(parents=True, exist_ok=True)
  218      teacher = {
  219          "judge": "/work/uf4_20260910/assets/Qwen3-14B",
  220          "judge_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
  221          "judgment_tag": args.tag,
  222          "template": "PROSPER Figure 4 PSC five-point single-check",
  223          "scale": "verdict in {0..4} for the FIRST response; p = verdict/4",
  224          "draws_per_order": 1,
  225          "orders": "both, swapped verdict reversed before averaging",
  226          "graph": "cross 8x8 + learner C(8,2) + reference C(8,2) per item",
  227          "items_per_prompt": K,
  228      }
  229      per = (len(pids) + args.shards - 1) // args.shards
  230      written = []
  231      for s in range(args.shards):
  232          chunk = pids[s * per:(s + 1) * per]
  233          if not chunk:
  234              continue
  235          A_LL = np.stack([tensors[p][0] for p in chunk], axis=1)
  236          A_LR = np.stack([tensors[p][1] for p in chunk], axis=1)
  237          A_RR = np.stack([tensors[p][2] for p in chunk], axis=1)
  238          d = out / ("shard%d" % s)
  239          d.mkdir(parents=True, exist_ok=True)
  240          path = d / "chunk0000.npz"
  241          # A_policy and A_ref are the names the solver reads. They are aliases of
  242          # A_LR and A_RR, written explicitly so an existing reader gets the
  243          # paper's game without being changed, while A_LL travels under its own
  244          # name for the pair-label consumers.
  245          np.savez(path, prompt_ids=np.array(chunk),
  246                   A_policy=A_LR, A_ref=A_RR,
  247                   A_LL=A_LL, A_LR=A_LR, A_RR=A_RR)
  248          (d / "chunk0000.manifest.json").write_text(json.dumps(
  249              {"prompts": len(chunk), "sha256": file_hash(path),
  250               "shapes": {"A_policy": list(A_LR.shape), "A_ref": list(A_RR.shape),
  251                          "A_LL": list(A_LL.shape), "A_LR": list(A_LR.shape),
  252                          "A_RR": list(A_RR.shape)},
  253               "tensor_role_schema": TENSOR_ROLE_SCHEMA,
  254               "roles": {"A_policy": "A_LR (learner vs reference)",
  255                         "A_ref": "A_RR (reference vs reference)"},
  256               "bank_ids": {"A_policy": ["learner", "comparator"],
  257                            "A_ref": ["comparator", "comparator"],
  258                            "A_LL": ["learner", "learner"]}},
  259              indent=1) + "\n")
  260          (d / ("complete_shard%d.json" % s)).write_text(json.dumps(
  261              {"shard": s, "prompts": len(chunk), "gpm_teacher": teacher,
  262               "bt_teacher": ("absent: the union contract uses direct order-balanced PSC "
  263                              "probabilities for these rows and forbids a scalar BT "
  264                              "projection, so none was fitted and none is written"),
  265               "tensor_role_schema": TENSOR_ROLE_SCHEMA,
  266               "reference_construction": ("independent reference bank: eight reference "
  267                                          "occurrences per prompt, so A_policy is the "
  268                                          "learner-by-reference cross block and A_ref is "
  269                                          "the reference triangle, not a copy of it")},
  270              indent=1) + "\n")
  271          written.append({"shard": s, "prompts": len(chunk),
  272                          "sha256": file_hash(path),
  273                          "shapes": {"A_policy": list(A_LR.shape),
  274                                     "A_ref": list(A_RR.shape),
  275                                     "A_LL": list(A_LL.shape)}})
  276  
  277      verdicts = sum(status.values())
  278      report = {
```


## E14 — Generator panel-name propagation

Path: `analysis/sub_20260914/code_snapshot_20260917_union/build_panel_solvers.py`
SHA256: `d958a4ffd5bc3a4b2e81228054c1f11eb67bc1e7f80573a1064e7a362c6b126b`

```text
   30                               % (label, old[:70], text.count(old)))
   31          text = text.replace(old, new)
   32      return text
   33  
   34  
   35  def main():
   36      ap = argparse.ArgumentParser(description=__doc__)
   37      ap.add_argument("--panel", required=True, help="us1 or ut1")
   38      ap.add_argument("--objectives", type=int, required=True)
   39      ap.add_argument("--split-dir", required=True,
   40                      help="the restricted/certified split directory, e.g. us_v1p")
   41      args = ap.parse_args()
   42  
   43      p, K = args.panel, args.objectives
   44      objs = ", ".join('"item%d"' % k for k in range(K))
   45      written = {}
   46  
   47      # 1. the panel base: loaders, hashing, writers, and the objective slots
   48      src = CODE / "solve_pros4_targets_uw1.py"
   49      t = src.read_text()
   50      t = sub(t, [('OBJECTIVES = ("item0", "item1", "item2", "item3")',
   51                   "OBJECTIVES = (%s)" % objs),
   52                  # the panel label travels with the module, so a US/UT/UW run does
   53                  # not record the UF-4 label of the file it was generated from
   54                  ('PANEL_LABEL = "UW1"', 'PANEL_LABEL = "%s"' % p.upper())],
   55              src.name)
   56      header = ("# Generated from solve_pros4_targets_uw1.py by build_panel_solvers.py\n"
   57                "# -- do not edit by hand. source sha256 %s\n"
   58                "# change: OBJECTIVES -> item0..item%d, the %s panel's declared objective\n"
   59                "#         count. The rubric text behind each slot is that panel's frozen\n"
   60                "#         rubric, recorded in panel/%s/freeze.json; slot k of two panels\n"
   61                "#         is never pooled as one objective.\n"
   62                % (sha(src), K - 1, p.upper()[:2], args.split_dir.replace("_v1p", "_v1")))
   63      base_name = "solve_pros4_targets_%s.py" % p
   64      (CODE / base_name).write_text(header + t)
   65      written[base_name] = None
   66  
   67      # 2. the two per-prompt arms and the probe: repoint at this panel
   68      roots = [('("train", (ROOT / "scores/uw1", ROOT / "pools/uw1", "policy_train")),',
   69                '("train", (ROOT / "scores/%s", ROOT / "pools/%s", "policy_train")),' % (p, p)),
   70               ('("dev", (ROOT / "scores/uw1", ROOT / "pools/uw1", "policy_dev"))):',
   71                '("dev", (ROOT / "scores/%s", ROOT / "pools/%s", "policy_dev"))):' % (p, p)),
   72               ('with (ROOT / "splits/uw_v1p" / f"{split_file}.jsonl").open() as stream:',
   73                'with (ROOT / "splits/%s" / f"{split_file}.jsonl").open() as stream:'
   74                % args.split_dir),
   75               ("import solve_pros4_targets_uw1 as base",
   76                "import solve_pros4_targets_%s as base" % p),
   77               # a stale literal objective count silently turns into a fake
   78               # infeasibility on a panel with a different K
   79               ("K = 4", "K = len(base.OBJECTIVES)")]
   80      for stem in ("solve_pros4_pw_nbpo", "solve_pros4_prosper", "solve_pros4_pw_fixedref"):
   81          s = CODE / ("%s_uw1.py" % stem)
   82          if not s.exists():
```


## E15 — Global evaluator intentionally retained for the Global Nash control

Path: `scripts/nbpo/eval_game_value.py`
SHA256: `60c017e5b9926c0c973f211ec6b30ee3c15b4b3f1f0dadf739d09c7f2c3f1a61`

```text
    1  #!/usr/bin/env python3
    2  """Game-value evaluator for NBPO (the finite-temperature counterpart of eval_bpo_surplus).
    3  
    4  On held-out prompts with a separately generated reference comparator pool
    5  (a preference-tensor artifact from ``build_preference_tensor.py``), computes:
    6  
    7  - ``V_{k,beta}(pi)`` (Eq. (8)) for the evaluated policy's response pool,
    8  - ``d_k = V_{k,beta}(mu)`` (Eq. (10)) from the reference-as-learner tensor,
    9  - ``s_k = V_{k,beta}(pi) - d_k``, min and average surplus,
   10  - Nash welfare ``sum_k log s_k`` ONLY when every ``s_k > 0``: when any surplus
   11    is nonpositive the output carries ``nash_welfare: null`` and
   12    ``nash_welfare_defined: false`` -- surpluses are NEVER clamped before the log,
   13  - opponent entropy and effective sample size at the evaluated policy.
   14  
   15  The legacy ``scripts/bpo/eval_bpo_surplus.py`` is a fixed-reference
   16  (beta = infinity) diagnostic that clamps nonpositive surpluses; it is kept,
   17  separately labeled, and is NOT this evaluator.
   18  
   19  SCOPE: this evaluator is GLOBAL, and deliberately so. Its margins are taken
   20  over the whole prompt batch, so the surplus it reports is a prompt-AVERAGED
   21  quantity and its minimum is a minimum over objectives, not over prompts. That is
   22  the right object for the shared-weight Global Nash control, whose multipliers are
   23  one vector for all prompts, and it is the wrong object for gating a prompt-wise
   24  arm: two prompts whose local surpluses are +.2 and -.1 average to +.05 and would
   25  pass, while the compromise failed on one of them. Do not reuse this to accept or
   26  reject a prompt-wise candidate. The prompt-wise gate is
   27  ``analysis/sub_20260914/code_snapshot_20260917_union/nbpo_local_gate.py``, which
   28  builds one representation per prompt and applies its checks to the array.
   29  """
   30  from __future__ import annotations
   31  
   32  import argparse
   33  import json
   34  import math
   35  from pathlib import Path
   36  
   37  import numpy as np
   38  import torch
   39  
   40  from mnpo_scripts.nbpo_core import (
   41      compute_disagreement_point,
   42      compute_margins,
   43      compute_regularized_game_value,
   44      compute_regularized_opponent,
   45      opponent_entropy,
   46      opponent_ess,
   47      uniform_policy,
   48  )
   49  from scripts.nbpo.nbpo_common import sha256_file, write_json
   50  
   51  
   52  def evaluate_game_value(A_policy: torch.Tensor, A_ref: torch.Tensor, beta: torch.Tensor,
   53                          reference_construction: str = "shared_pool") -> dict:
   54      """Pure evaluation given the two centered tensors; policy uniform over its pool."""
   55      K, X, I, J = A_policy.shape
   56      mu = uniform_policy(X, J)
   57      pi = uniform_policy(X, I)
   58      r = compute_margins(A_policy, pi)
   59      V = compute_regularized_game_value(r, mu, beta, form="softmin")
   60      d = compute_disagreement_point(A_ref, mu, beta, reference_construction)
   61      s = V - d
   62      nu = compute_regularized_opponent(r, mu, beta)
   63      all_positive = bool((s > 0).all())
   64      return {
   65          "V": [float(v) for v in V],
   66          "d": [float(v) for v in d],
   67          "surplus": [float(v) for v in s],
   68          "min_surplus": float(s.min()),
   69          "avg_surplus": float(s.mean()),
   70          "nash_welfare_defined": all_positive,
   71          # Nash welfare only exists on the individually-rational set (Eq. (11));
   72          # a nonpositive surplus makes it undefined, not "very negative".
   73          "nash_welfare": float(torch.log(s).sum()) if all_positive else None,
   74          "opponent_entropy": [float(v) for v in opponent_entropy(nu)],
   75          "opponent_ess": [float(v) for v in opponent_ess(nu)],
   76      }
   77  
   78  
   79  def main() -> None:
   80      ap = argparse.ArgumentParser(description=__doc__,
   81                                   formatter_class=argparse.RawDescriptionHelpFormatter)
   82      ap.add_argument("--tensor-dir", type=Path, required=True,
   83                      help="held-out preference-tensor artifact (policy = the evaluated model)")
   84      ap.add_argument("--beta", required=True,
   85                      help="opponent temperatures: one value or comma list per objective")
   86      ap.add_argument("--label", default="policy")
   87      ap.add_argument("--out", type=Path)
   88      args = ap.parse_args()
   89  
   90      meta = json.loads((args.tensor_dir / "meta.json").read_text())
   91      A_policy = torch.from_numpy(np.load(args.tensor_dir / "tensor_policy.npz")["A"])
   92      A_ref = torch.from_numpy(np.load(args.tensor_dir / "tensor_ref.npz")["A"])
   93      K = A_policy.shape[0]
   94      beta_vals = [float(b) for b in args.beta.split(",") if b.strip()]
   95      beta = torch.tensor(beta_vals * K if len(beta_vals) == 1 else beta_vals, dtype=torch.float64)
   96      if beta.shape != (K,):
   97          raise ValueError(f"--beta must give 1 or {K} values, got {len(beta_vals)}")
   98  
   99      construction = meta.get("reference_construction")
  100      if construction is None:
```
