"""Canonical neural realization and pair-dataset contracts, without Trainer imports."""
from __future__ import annotations

import itertools
import math
import torch

from mnpo_scripts.pair_tokenization import immutable_pair_tokens


def nbpo_regression_target(target, eta, mode="sampled"):
    if mode == "canonical_logratio":
        return target.float()
    if mode not in ("sampled", "rb", "rao_blackwell", "canonical"):
        raise ValueError(f"Unknown NBPO target mode {mode!r}")
    return float(eta) * target.float()


def nbpo_wbc_pair_loss(logp_a, logp_b, weight_a, weight_b, num_candidates=8):
    """The all-unordered-pair average equals prompt weighted sequence NLL."""
    a, b = logp_a.float(), logp_b.float()
    wa = torch.as_tensor(weight_a, device=a.device, dtype=torch.float32).detach()
    wb = torch.as_tensor(weight_b, device=a.device, dtype=torch.float32).detach()
    n = torch.as_tensor(num_candidates, device=a.device, dtype=torch.float32)
    if torch.any(n != 8):
        raise ValueError("Primary WBC supports fixed N=8 only")
    if torch.any(~torch.isfinite(wa)) or torch.any(~torch.isfinite(wb)) or torch.any(wa < 0) or torch.any(wb < 0):
        raise ValueError("WBC requires finite nonnegative detached solver probability masses")
    if wa.shape != a.shape or wb.shape != b.shape:
        raise ValueError("Candidate weights must match their actual pair logps")
    return (n / 2) * (-wa * a - wb * b)


def select_training_splits(datasets, train_split="train", eval_split="dev"):
    """Names are exact; final test is never selected by a substring heuristic."""
    if train_split not in datasets:
        raise ValueError(f"Missing train_split={train_split!r}; available={list(datasets)}")
    if eval_split == "test":
        raise ValueError("Final test must use a separate evaluation entrypoint; choose dev explicitly")
    if eval_split is not None and eval_split not in datasets:
        raise ValueError(f"Missing eval_split={eval_split!r}; available={list(datasets)}")
    if train_split == eval_split:
        raise ValueError("Train and evaluation splits must be distinct")
    return datasets[train_split], datasets[eval_split] if eval_split else None


def validate_canonical_pair_dataset(dataset, max_length=2048, max_prompt_length=1024,
                                    expected_solver_hash=None):
    """Check pair completeness, candidate mass/context identity and eta units.

    Duplicate sampled strings remain separate occurrences. Each of the 28
    unordered pairs must occur exactly once for every prompt, with fixed N=8.
    """
    groups = {}
    solvers = set()
    for row in dataset:
        if row.get("target_mode") != "canonical_logratio" or row.get("target_units") != "final_logratio_change" or row.get("eta_already_included") is not True:
            raise ValueError("Canonical row must declare final logratio target units and eta included")
        if not math.isfinite(float(row["nbpo_logratio_target"])):
            raise ValueError("Nonfinite canonical target")
        if int(row["nbpo_num_candidates"]) != 8:
            raise ValueError("Primary all-pair dataset requires N=8")
        wa, wb = float(row["nbpo_weight_a"]), float(row["nbpo_weight_b"])
        if min(wa, wb) <= 0 or not math.isfinite(wa + wb):
            raise ValueError("Canonical targets require strictly positive solver masses")
        center_a = float(row.get("nbpo_center_a", 1.0 / 8))
        center_b = float(row.get("nbpo_center_b", 1.0 / 8))
        if center_a != 1.0 / 8 or center_b != 1.0 / 8:
            raise ValueError("Primary IID occurrence center must be exactly 1/8")
        expected_target = math.log(wa / center_a) - math.log(wb / center_b)
        if abs(float(row["nbpo_logratio_target"]) - expected_target) > 1e-9:
            raise ValueError("Canonical target disagrees with log(p_star/p_t) candidate masses")
        tokens = immutable_pair_tokens(row, max_length, max_prompt_length)
        if tokens is None:
            raise ValueError("Canonical dataset requires immutable sampled tokens")
        key = str(row["prompt_id"])
        group = groups.setdefault(key, {"pairs": set(), "candidates": {}, "prompt": tokens["prompt_input_ids"]})
        if group["prompt"] != tokens["prompt_input_ids"]:
            raise ValueError("One prompt group has different conditioning token contexts")
        ids = [row.get(f"{side}_response_id", row.get(f"{side}_candidate_id"))
               for side in ("chosen", "rejected")]
        if None in ids or ids[0] == ids[1]:
            raise ValueError("Pair must identify two distinct sampled candidate occurrences")
        pair = tuple(sorted(map(str, ids)))
        if pair in group["pairs"]:
            raise ValueError("Duplicate unordered candidate pair")
        group["pairs"].add(pair)
        for side, candidate_id, weight_key in zip(("chosen", "rejected"), ids, ("nbpo_weight_a", "nbpo_weight_b")):
            mass = float(row[weight_key])
            if not math.isfinite(mass) or not 0 <= mass <= 1:
                raise ValueError("Invalid solver probability mass")
            value = (tokens[f"{side}_token_sha256"], mass)
            old = group["candidates"].setdefault(str(candidate_id), value)
            if old != value:
                raise ValueError("Candidate tokens or solver mass depend on pair partner")
        solver_hash = row.get("solver_artifact_sha256")
        if not solver_hash:
            raise ValueError("Canonical row lacks solver_artifact_sha256")
        if expected_solver_hash and solver_hash != expected_solver_hash:
            raise ValueError("Unexpected solver artifact hash")
        solvers.add(solver_hash)
    if not groups:
        raise ValueError("Empty canonical dataset")
    for group in groups.values():
        candidates = group["candidates"]
        if len(candidates) != 8 or group["pairs"] != set(itertools.combinations(sorted(candidates), 2)):
            raise ValueError("Every prompt requires all 28 unordered pairs of 8 candidates")
        if not math.isclose(sum(value[1] for value in candidates.values()), 1.0, abs_tol=1e-7):
            raise ValueError("Prompt solver probability masses do not sum to one")
    return {"prompts": len(groups), "rows": len(dataset), "candidates_per_prompt": 8,
            "unordered_pairs_per_prompt": 28, "solver_artifact_sha256": sorted(solvers)}
