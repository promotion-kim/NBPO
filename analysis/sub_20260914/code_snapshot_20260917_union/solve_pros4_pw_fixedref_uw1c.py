# Generated from solve_pros4_pw_fixedref_uw1.py by build_panel_solvers.py -- do not edit by hand.
# source sha256 7fa7013b8b90779451876afe66e77e45409201066eddf83a85d34acea803b7ed
# change: score/pool roots -> uw1c, restricted split -> uw_v1cp, base module ->
#         solve_pros4_targets_uw1c. The rule this arm implements is
#         untouched.
# Generated from solve_prosper_targets.py by build_pw_solvers.py -- do not edit.
# source sha256 1809a5f5b63019c3d85842e6e3246db3e7a22a2e03c50944f1a8510ff4d2a1bb
# arm: per-prompt Nash, fixed reference
# shared changes: base module -> solve_pros4_targets; score/pool/split roots ->
#   pros_scores_all / pros2_pool_v1 / splits/pros_v1f; hardcoded 4 shards ->
#   --shards (default 8); matched multiplier norm optional.
# Objectives are item0..item3, four of each prompt's OWN checklist items
# (PROSPER PSC); item k of two prompts is never pooled as one objective.
"""Faithful PROSPER adaptation: per-prompt adversarial weights over the same game values.

PROSPER's criterion (app:uf-protocol) minimises over w: X -> Delta_K and over the
KL-regularised opponent nu. A linear function on the simplex is minimised at a
vertex, so the inner weight minimisation is the per-prompt worst objective, and
with no shared dual the problem decomposes across prompts. This therefore solves
absolute_maxmin ONCE PER PROMPT on the same adaptive-game values, at the same
opponent temperature, with the multiplier norm matched to the Nash dual exactly
as the utilitarian and global-maxmin controls are matched.

It is not the global game-maxmin control and must never be labelled as one.
Global maxmin changes the compromise over prompt-averaged values; this also
changes the order of aggregation across prompts, which is the distinction the
manuscript's two-prompt example draws.

Nothing shared with the published arms is modified. The per-prompt solution is
handed to the verified build_rows directly as canonical_data, which is exactly
what load_canonical_artifact would have reconstructed from disk; the difference
is that the weights accompanying each prompt are that prompt's own. Per-prompt
weights and certificates are written to this script's own provenance, because
the shared solution artifact is shaped for one global weight vector.

One consequence is recorded rather than hidden: with no shared dual, policy_dev
is no longer a held-out measurement of a fit made on policy_train, since each
prompt fits its own weights on whichever split it is in.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/work/uf4_20260910/code")
import solve_pros4_targets_uw1c as base                      # loaders, hashing, writers
from mnpo_scripts.nbpo_core import uniform_policy
from mnpo_scripts.nbpo_generic import solve_finite_pool, validate_finite_pool_solution
from mnpo_scripts.nbpo_representations import (AdaptiveGameRepresentation,
                                               FixedReferenceRepresentation)
from scripts.nbpo.build_nbpo_pairs import build_rows

ROOT = Path("/work/uf4_20260910")
OBJECTIVES = base.OBJECTIVES
POOL = base.POOL
_SHARED = {}


def _init(A, Aref, beta, eta, weight_l1, max_dual_calls, floor):
    _SHARED.update(A=A, Aref=Aref, beta=beta, eta=eta, weight_l1=weight_l1,
                   M=max_dual_calls, floor=floor)
    torch.set_num_threads(1)


def _solve_one(x):
    """Solve prompt x alone: its own adversarial weights, its own certificate."""
    A = _SHARED["A"][:, x:x + 1]
    Aref = _SHARED["Aref"][:, x:x + 1]
    rep = FixedReferenceRepresentation(
        torch.from_numpy(np.ascontiguousarray(A)), torch.from_numpy(np.ascontiguousarray(Aref)),
        uniform_policy(1, POOL),
        reference_construction="independent_samples")
    result = solve_finite_pool(
        rep, "nash", eta=_SHARED["eta"], inner_solver="exact",
        dual_solver="root", dual_tol=1e-10, M=_SHARED["M"], inner_workers=1,
        probability_floor=_SHARED["floor"], log_every=0)
    certificate = validate_finite_pool_solution(result)
    return (x,
            result.pi.numpy().astype(np.float64),
            result.nu_update.numpy().astype(np.float64),
            result.weights.numpy().astype(np.float64),
            result.target_log_ratio.numpy().astype(np.float64),
            float(result.target_log_ratio_check()),
            bool(certificate.get("certified", False)),
            # the certificate carries the per-objective surplus vector, not its
            # minimum; taking .get("min_surplus") silently produced NaN, which
            # then made "negative worst surplus" count zero prompts by accident
            float(np.min(np.asarray(certificate["surplus"], dtype=np.float64))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-name", default="prosper_v1")
    ap.add_argument("--shards", type=int, default=8,
                    help="score and pool shard count; 8 once the 400-prompt "
                         "extension is merged in as shards 4..7")
    ap.add_argument("--weight-l1", type=float, default=None,
                    help="matched multiplier norm as a literal. Only absolute_maxmin "
                         "uses it; nash derives its own dual.")
    ap.add_argument("--weight-l1-from", type=Path,
                    default=None)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-dual-calls", type=int, default=200)
    ap.add_argument("--probability-floor", type=float, default=1e-12)
    ap.add_argument("--limit", type=int, default=0,
                    help="pilot on the first N prompts of each split; 0 means all")
    ap.add_argument("--skip-dataset", action="store_true")
    args = ap.parse_args()

    # nash derives its own dual: no multiplier norm is read, and none is used.
    # The global NBPO arm this was matched to is uncertified (independent
    # stationarity 13.224), and nothing here is matched to an uncertified value.
    weight_l1 = None
    print(json.dumps({"weight_l1": None,
                      "note": "unused: nash derives its own dual"}), flush=True)

    out = ROOT / "targets" / args.out_name
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.monotonic()
    shared_meta, outputs = None, {}
    split_digests = {}
    # the base module owns the digest so every arm hashes identically
    digest_of = getattr(base, "split_pool_digest", None) or split_pool_digest

    for split, (score_root, pool_root, split_file) in (
            ("train", (ROOT / "scores/uw1c", ROOT / "pools/uw1c", "policy_train")),
            ("dev", (ROOT / "scores/uw1c", ROOT / "pools/uw1c", "policy_dev"))):
        split_start = time.monotonic()
        scores, score_manifests = base.load_scores(str(score_root), args.shards)
        pool, pool_settings = base.load_pool(str(pool_root), args.shards)
        with (ROOT / "splits/uw_v1cp" / f"{split_file}.jsonl").open() as stream:
            pids = [json.loads(line)["prompt_id"] for line in stream if line.strip()]
        if args.limit:
            pids = pids[:args.limit]
        missing = [p for p in pids if p not in scores or p not in pool]
        if missing:
            raise ValueError(f"{split}: {len(missing)} prompts have no scores or no pool")

        A = np.stack([scores[p][0] for p in pids], axis=1)
        Aref = np.stack([scores[p][1] for p in pids], axis=1)
        if shared_meta is None:
            from transformers import AutoTokenizer
            from mnpo_scripts.precompute_provenance import tokenizer_content_hashes
            tokenizer = AutoTokenizer.from_pretrained(pool_settings["model"], local_files_only=True)
            pad_fallback = tokenizer.pad_token_id is None
            if pad_fallback:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            hashes = tokenizer_content_hashes(tokenizer)
            if hashes["chat_template_hash"] != pool_settings["chat_template_sha256"]:
                raise ValueError("Pool chat template does not match the training tokenizer's")
            shared_meta = {"model_revision": pool_settings["model_revision"], **hashes,
                           "training_pad_token_fallback_to_eos": pad_fallback,
                           "pool_settings_sha256": base.object_hash(pool_settings),
                           "teacher_manifest_sha256": base.object_hash(score_manifests),
                           "gpm_teacher": score_manifests[0]["gpm_teacher"],
                           "bt_teacher": score_manifests[0]["bt_teacher"]}
        meta = {"prompt_ids": pids, "objectives": list(OBJECTIVES),
                "reference_construction": "independent_samples",
                "policy_learner_ids": [f"policy:{i}" for i in range(POOL)],
                "comparator_ids": [f"ref:{j}" for j in range(POOL)],
                "split": split, **shared_meta}

        print(json.dumps({"phase": "solve", "split": split, "prompts": len(pids),
                          "workers": args.workers, "per_prompt_weights": True}), flush=True)
        pi = np.zeros((len(pids), POOL))
        nu = np.zeros((len(OBJECTIVES), len(pids), POOL))
        weights = np.zeros((len(OBJECTIVES), len(pids)))
        g = np.zeros((len(pids), POOL))
        identity, certified, min_surplus = np.zeros(len(pids)), np.zeros(len(pids), bool), np.zeros(len(pids))
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                                 initargs=(A, Aref, args.beta, args.eta, weight_l1,
                                           args.max_dual_calls, args.probability_floor)) as pool_exec:
            for res in pool_exec.map(_solve_one, range(len(pids)), chunksize=8):
                x, pi_x, nu_x, w_x, g_x, ident, cert, ms = res
                pi[x] = pi_x[0]
                nu[:, x] = nu_x[:, 0]
                weights[:, x] = w_x
                g[x] = g_x[0]
                identity[x], certified[x], min_surplus[x] = ident, cert, ms
                done += 1
                if done % 1000 == 0:
                    print(json.dumps({"split": split, "solved": done, "of": len(pids),
                                      "seconds": round(time.monotonic() - split_start, 1)}), flush=True)
        if not certified.all():
            raise ValueError(f"{split}: {int((~certified).sum())} prompts have no certificate")
        if float(np.abs(identity).max()) > 1e-9:
            raise ValueError(f"{split}: target log-ratio identity residual "
                             f"{float(np.abs(identity).max())}")

        per_prompt_path = out / f"{split}_per_prompt.npz"
        np.savez_compressed(per_prompt_path, pi=pi, weights=weights, g=g,
                            min_surplus=min_surplus, identity_residual=identity,
                            prompt_ids=np.array(pids, dtype=object))

        # A hash-bound solution artifact at the path the shared job generator
        # expects. The global one is not written because per-prompt weights do
        # not fit its (K,) weight field, but the generator only needs SOME file
        # whose hash pins the solution, and patching the generator would touch
        # code the published arms depend on. This file pins the real thing: the
        # per-prompt npz that every target in this set was built from.
        solver_dir = out / split / "solver"
        base.write_json(solver_dir / "solution.json", {
            "target_mode": "canonical_logratio", "target_column": "nbpo_logratio_target",
            "target_units": "final_logratio_change", "eta_already_included": True,
            "representation": "fixed_reference", "aggregation": "prompt_wise_nash_fixed_reference",
            "weights_scope": "per prompt; there is no shared dual",
            "per_prompt_artifact": str(per_prompt_path),
            "per_prompt_artifact_sha256": base.file_hash(per_prompt_path),
            "n_prompts": len(pids), "split": split,
            "all_certified": bool(certified.all()),
            "max_identity_residual": float(np.abs(identity).max()),
            "min_surplus_mean": float(min_surplus.mean()),
            "min_surplus_negative_prompts": int((min_surplus < 0).sum()),
            "weight_l1_matched": weight_l1, "beta": args.beta, "eta": args.eta,
            "solver_source_sha256": base.file_hash(__file__), **shared_meta})
        solver_hash = base.file_hash(solver_dir / "solution.json")
        provenance = {"solver_artifact_sha256": solver_hash,
                      "solver_hash": solver_hash,
                      "target_artifact_hash": base.file_hash(per_prompt_path),
                      "representation": "fixed_reference", "aggregation": "prompt_wise_nash_fixed_reference",
                      "split": split, "panel": getattr(base, "PANEL_LABEL", "UF-4"), **shared_meta}
        betas = np.full(len(OBJECTIVES), args.beta)

        def pair_rows():
            for x, pid in enumerate(pids):
                learners = {str(i): {pid: {**pool[pid]["learner"][i],
                                           "generated_text": pool[pid]["learner"][i]["response"]}}
                            for i in range(POOL)}
                rows = build_rows(
                    [pid], list(OBJECTIVES), A[:, x:x + 1], nu[:, x:x + 1], weights[:, x],
                    betas, learners, None, np.random.default_rng(42), "canonical_logratio",
                    meta, provenance,
                    canonical_data={"g": g[x:x + 1], "p_star": pi[x:x + 1],
                                    "p_t": np.full((1, POOL), 1.0 / POOL)})
                for row in rows:
                    # round masses to the loader's ten decimals and rebuild the
                    # target from the rounded values; a per-prompt Nash solve
                    # concentrates mass and the identity otherwise fails after the parse
                    row = base.quantize_canonical_row(row)
                    a, b = row["chosen_candidate_index"], row["rejected_candidate_index"]
                    row["chosen_response_id"] = pool[pid]["learner"][a]["candidate_id"]
                    row["rejected_response_id"] = pool[pid]["learner"][b]["candidate_id"]
                    row["prosper_prompt_weights"] = [float(v) for v in weights[:, x]]
                    yield row

        pair_path = out / "pairs" / f"{split}.jsonl"
        base.write_jsonl(pair_path, pair_rows())
        outputs[split] = {"n_prompts": len(pids), "n_pairs": len(pids) * 28,
                          "pairs_path": str(pair_path), "pairs_sha256": base.file_hash(pair_path),
                          "per_prompt_weight_mean": [float(v) for v in weights.mean(axis=1)],
                          "per_prompt_weight_sd": [float(v) for v in weights.std(axis=1, ddof=1)],
                          "effective_objectives_mean": float(np.mean(
                              (weights.sum(axis=0) ** 2) / (weights ** 2).sum(axis=0))),
                          "min_surplus_mean": float(min_surplus.mean()),
                          "min_surplus_negative_prompts": int((min_surplus < 0).sum()),
                          "max_identity_residual": float(np.abs(identity).max()),
                          "all_certified": bool(certified.all()),
                          "solver_solution_sha256": solver_hash,
                          "seconds": time.monotonic() - split_start}
        split_digests[split] = digest_of(pool, pids)
        base.write_json(out / split / "complete.json", outputs[split])
        print(json.dumps({k: v for k, v in outputs[split].items()
                          if k not in ("pairs_path", "pairs_sha256")}), flush=True)

    prov = {**shared_meta, "objectives": list(OBJECTIVES), "panel": getattr(base, "PANEL_LABEL", "UF-4"),
            "aggregation": "prompt_wise_nash_fixed_reference", "beta": args.beta, "eta": args.eta,
            "weight_l1": weight_l1, "splits": outputs,
            # content digests of the rows each split actually uses, not of the
            # split NAMES: the previous value hashed sorted(outputs), so train and
            # dev were equal to each other and unchanged by the pool's contents
            "train_pool_sha256": split_digests.get("train"),
            "dev_pool_sha256": split_digests.get("dev"),
            "pool_digest_fields": ("prompt_id, role, occurrence index, candidate_id, "
                                   "response_sha256"),
            "dev_note": ("no shared dual: each prompt fits its own adversarial weights on "
                         "whichever split it is in, so dev measures neural generalisation "
                         "but not generalisation of a dual fitted on train")}
    base.write_json(out / "dataset_provenance.json", prov)

    manifest = None
    if not args.skip_dataset:
        import subprocess
        dataset_out = ROOT / "datasets" / args.out_name
        env = dict(os.environ, HF_DATASETS_CACHE=str(out / "arrow_cache"),
                   PYTHONDONTWRITEBYTECODE="1")
        done = subprocess.run(
            [sys.executable, "-m", "mnpo_scripts.prepare_nbpo_dataset",
             "--train", str(out / "pairs/train.jsonl"), "--dev", str(out / "pairs/dev.jsonl"),
             "--output", str(dataset_out), "--provenance", str(out / "dataset_provenance.json")],
            env=env, capture_output=True, text=True)
        if done.returncode:
            base.write_json(out / "dataset_materialization_failure.json",
                            {"returncode": done.returncode, "stdout": done.stdout[-4000:],
                             "stderr": done.stderr[-4000:]})
            raise RuntimeError("Dataset materialization failed; diagnostic preserved")
        manifest = base.file_hash(dataset_out / "precompute_manifest.json")

    base.write_json(out / "complete.json", {
        "splits": outputs, "aggregation": "prompt_wise_nash_fixed_reference",
        "beta": args.beta, "eta": args.eta, "weight_l1": weight_l1,
        "dataset_path": None if args.skip_dataset else str(ROOT / "datasets" / args.out_name),
        "dataset_manifest_sha256": manifest,
        "faithfulness_note": ("Nash aggregation per prompt against a frozen reference "
                              "policy rather than an adaptive one. It is NOT PROSPER's "
                              "max-min rule and NOT the adaptive-reference arm."),
        "seconds": time.monotonic() - started, "source_sha256": base.file_hash(__file__)})
    print(json.dumps({"targets": str(out), "dataset_manifest_sha256": manifest,
                      "seconds": round(time.monotonic() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
