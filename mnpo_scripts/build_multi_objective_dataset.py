import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_records(path: str) -> Iterable[Dict[str, Any]]:
    suffix = Path(path).suffix.lower()
    with open(path, "r", encoding="utf-8") as f:
        if suffix in {".jsonl", ".jsonlines"}:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        elif suffix == ".json":
            data = json.load(f)
            if isinstance(data, list):
                yield from data
            elif isinstance(data, dict):
                for value in data.values():
                    if isinstance(value, list):
                        yield from value
                        return
                yield data
            else:
                raise ValueError(f"Unsupported JSON structure in {path}")
        else:
            raise ValueError(f"Unsupported file suffix for {path}")


def parse_named_paths(values: List[str]) -> List[Tuple[str, str]]:
    out = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected name=path, got {value!r}")
        name, path = value.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Expected non-empty name=path, got {value!r}")
        out.append((name, path))
    return out


def prompt_key(record: Dict[str, Any]) -> str:
    if record.get("prompt_id"):
        return str(record["prompt_id"])
    prompt = str(record.get("prompt", ""))
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def response_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        last = value[-1]
        if isinstance(last, dict):
            return last.get("content")
        if isinstance(last, str):
            return last
    if isinstance(value, dict):
        return value.get("content")
    return None


def make_messages(prompt: str, response: str) -> List[Dict[str, str]]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def minmax(values: List[float]) -> List[float]:
    clean = [float(v) for v in values]
    lo = min(clean)
    hi = max(clean)
    if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) < 1e-12:
        return [0.5 for _ in clean]
    return [(v - lo) / (hi - lo) for v in clean]


def rank_normalize(values: List[float]) -> List[float]:
    clean = [float(v) for v in values]
    if len(clean) == 1:
        return [0.5]
    order = sorted(range(len(clean)), key=lambda i: clean[i])
    ranks = [0.0] * len(clean)
    for rank, idx in enumerate(order):
        ranks[idx] = rank / float(len(clean) - 1)
    return ranks


def normalize_scores(scores: Dict[str, List[float]], mode: str) -> Dict[str, List[float]]:
    if mode == "none":
        return {k: [float(v) for v in vals] for k, vals in scores.items()}
    if mode == "minmax":
        return {k: minmax(vals) for k, vals in scores.items()}
    if mode == "rank":
        return {k: rank_normalize(vals) for k, vals in scores.items()}
    raise ValueError(f"Unsupported normalization mode: {mode}")


def argmax(values: List[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i])


def argmin(values: List[float]) -> int:
    return min(range(len(values)), key=lambda i: values[i])


def average_per_response(norm_scores: Dict[str, List[float]], objective_names: List[str]) -> List[float]:
    n = len(norm_scores[objective_names[0]])
    return [sum(norm_scores[obj][i] for obj in objective_names) / len(objective_names) for i in range(n)]


def min_per_response(norm_scores: Dict[str, List[float]], objective_names: List[str]) -> List[float]:
    n = len(norm_scores[objective_names[0]])
    return [min(norm_scores[obj][i] for obj in objective_names) for i in range(n)]


def softmax(values: List[float], temperature: float) -> List[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [float(v) / temperature for v in values]
    offset = max(scaled)
    exps = [math.exp(v - offset) for v in scaled]
    total = sum(exps)
    return [v / total for v in exps]


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def estimate_preference_prob(score_y: float, score_a: float, scale: float) -> float:
    return sigmoid(scale * (float(score_y) - float(score_a)))


def average_win_probs(avg_scores: List[float], scale: float) -> List[float]:
    """Estimate SPPO P_hat(y > pi_t) from the homogeneous average oracle.

    SPPO uses a single preference oracle, so for multi-objective data we expose
    the average normalized objective score as that homogeneous oracle.  The
    empirical policy support is the candidate response pool generated for the
    same prompt; each response is compared against the other sampled responses.
    """
    n = len(avg_scores)
    if n <= 1:
        return [0.5 for _ in avg_scores]

    probs: List[float] = []
    for i, score_i in enumerate(avg_scores):
        comparisons = [
            estimate_preference_prob(score_i, score_j, scale)
            for j, score_j in enumerate(avg_scores)
            if j != i
        ]
        probs.append(sum(comparisons) / len(comparisons) if comparisons else 0.5)
    return probs


def policy_distribution(avg_scores: List[float], mode: str, temperature: float) -> List[float]:
    if mode == "uniform":
        return [1.0 / len(avg_scores) for _ in avg_scores]
    if mode == "softmax_avg":
        return softmax(avg_scores, temperature)
    raise ValueError(f"Unsupported policy distribution mode: {mode}")


def deterministic_unit_interval(*parts: Any) -> float:
    key = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def weighted_index(weights: List[float], u: float) -> int:
    total = float(sum(max(0.0, w) for w in weights))
    if total <= 0.0:
        return min(int(u * len(weights)), len(weights) - 1)
    threshold = u * total
    cdf = 0.0
    for idx, weight in enumerate(weights):
        cdf += max(0.0, float(weight))
        if threshold <= cdf:
            return idx
    return len(weights) - 1


def sample_distinct_policy_pair(
    pi: List[float],
    prompt_id: Any,
    objective_index: Any,
    adversary_index: Any,
    sample_index: int,
) -> Tuple[int, int]:
    n_resp = len(pi)
    if n_resp < 2:
        raise ValueError("Need at least two responses to build a RONPO pair.")
    first = weighted_index(
        pi,
        deterministic_unit_interval(prompt_id, objective_index, adversary_index, sample_index, "y"),
    )
    second = weighted_index(
        pi,
        deterministic_unit_interval(prompt_id, objective_index, adversary_index, sample_index, "y_prime"),
    )
    if second == first:
        offset = 1 + int(
            deterministic_unit_interval(prompt_id, objective_index, adversary_index, sample_index, "offset")
            * (n_resp - 1)
        )
        second = (first + offset) % n_resp
    return first, second


def sample_common_policy_pair(
    pi: List[float],
    prompt_id: Any,
    sample_index: int,
    common_pair_seed: str,
    fallback_objective_index: Any = 0,
    fallback_adversary_index: Any = 0,
) -> Tuple[int, int]:
    if common_pair_seed:
        return sample_distinct_policy_pair(
            pi,
            prompt_id,
            "common_pair",
            common_pair_seed,
            sample_index,
        )
    return sample_distinct_policy_pair(
        pi,
        prompt_id,
        fallback_objective_index,
        fallback_adversary_index,
        sample_index,
    )


def update_adversary_distribution(
    norm_scores: Dict[str, List[float]],
    objective_names: List[str],
    pi: List[float],
    steps: int,
    alpha: float,
    kappa: float,
    preference_scale: float,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Exponentiated-gradient update for sigma(k, a).

    c(k,a) approximates E_{y~pi_t} P_k(y beats a).  Low c means the current
    policy is weak against objective-response atom (k,a), so the adversary
    update increases that atom's mass.
    """
    n_obj = len(objective_names)
    n_resp = len(pi)
    sigma0 = [[1.0 / (n_obj * n_resp) for _ in range(n_resp)] for _ in range(n_obj)]
    sigma = [row[:] for row in sigma0]

    for _ in range(steps):
        costs = []
        for k, obj in enumerate(objective_names):
            row = []
            for a in range(n_resp):
                c = 0.0
                for y in range(n_resp):
                    c += pi[y] * estimate_preference_prob(
                        norm_scores[obj][y],
                        norm_scores[obj][a],
                        preference_scale,
                    )
                row.append(c)
            costs.append(row)

        logits = []
        for k in range(n_obj):
            row = []
            for a in range(n_resp):
                row.append(
                    (1.0 - alpha * kappa) * math.log(max(sigma[k][a], 1e-12))
                    + alpha * kappa * math.log(max(sigma0[k][a], 1e-12))
                    - alpha * costs[k][a]
                )
            logits.append(row)

        flat = [v for row in logits for v in row]
        offset = max(flat)
        exp_flat = [math.exp(v - offset) for v in flat]
        total = sum(exp_flat)
        idx = 0
        for k in range(n_obj):
            for a in range(n_resp):
                sigma[k][a] = exp_flat[idx] / total
                idx += 1

    return sigma, sigma0


def select_adversary_atoms(
    sigma: List[List[float]],
    count: int,
    mode: str,
    prompt_id: Any = "",
) -> List[Tuple[int, int, float]]:
    atoms = []
    for k, row in enumerate(sigma):
        for a, mass in enumerate(row):
            atoms.append((k, a, float(mass)))

    if mode == "top":
        atoms.sort(key=lambda item: item[2], reverse=True)
        return atoms[: max(1, count)]
    if mode == "all":
        return atoms
    if mode == "sample":
        weights = [mass for _, _, mass in atoms]
        draws = []
        for sample_idx in range(max(1, count)):
            idx = weighted_index(
                weights,
                deterministic_unit_interval(prompt_id, "sigma_atom", sample_idx),
            )
            draws.append(atoms[idx])
        return draws
    raise ValueError(f"Unsupported adversary atom selection mode: {mode}")


def distribution_entropy(values: Iterable[float]) -> float:
    entropy = 0.0
    for value in values:
        p = float(value)
        if p > 0.0:
            entropy -= p * math.log(p)
    return entropy


def update_objective_adversary_distribution(
    norm_scores: Dict[str, List[float]],
    objective_names: List[str],
    pi: List[float],
    fixed_adversary_idx: int,
    steps: int,
    alpha: float,
    kappa: float,
    preference_scale: float,
) -> List[float]:
    """
    Objective-only ablation for the RONPO atom adversary.

    The full RONPO adversary places mass on objective-response atoms sigma(k,a).
    This ablation fixes a single adversarial response atom a_ref for the prompt
    and only lets the adversary choose the objective k. It is intentionally less
    expressive and is meant to test whether response-atom selection is necessary.
    """
    n_obj = len(objective_names)
    sigma0 = [1.0 / n_obj for _ in range(n_obj)]
    sigma = sigma0[:]

    for _ in range(steps):
        costs = []
        for obj in objective_names:
            c = 0.0
            for y in range(len(pi)):
                c += pi[y] * estimate_preference_prob(
                    norm_scores[obj][y],
                    norm_scores[obj][fixed_adversary_idx],
                    preference_scale,
                )
            costs.append(c)

        logits = [
            (1.0 - alpha * kappa) * math.log(max(sigma[k], 1e-12))
            + alpha * kappa * math.log(max(sigma0[k], 1e-12))
            - alpha * costs[k]
            for k in range(n_obj)
        ]
        offset = max(logits)
        exps = [math.exp(v - offset) for v in logits]
        total = sum(exps)
        sigma = [v / total for v in exps]

    return sigma


def update_objective_adversary_distribution_uniform_a(
    norm_scores: Dict[str, List[float]],
    objective_names: List[str],
    pi: List[float],
    steps: int,
    alpha: float,
    kappa: float,
    preference_scale: float,
) -> List[float]:
    """Objective-only MW update with the response atom marginalized uniformly."""
    n_obj = len(objective_names)
    n_resp = len(pi)
    sigma0 = [1.0 / n_obj for _ in range(n_obj)]
    sigma = sigma0[:]

    for _ in range(steps):
        costs = []
        for obj in objective_names:
            c_obj = 0.0
            for a in range(n_resp):
                c_atom = 0.0
                for y in range(n_resp):
                    c_atom += pi[y] * estimate_preference_prob(
                        norm_scores[obj][y],
                        norm_scores[obj][a],
                        preference_scale,
                    )
                c_obj += c_atom / n_resp
            costs.append(c_obj)

        logits = [
            (1.0 - alpha * kappa) * math.log(max(sigma[k], 1e-12))
            + alpha * kappa * math.log(max(sigma0[k], 1e-12))
            - alpha * costs[k]
            for k in range(n_obj)
        ]
        offset = max(logits)
        exps = [math.exp(v - offset) for v in logits]
        total = sum(exps)
        sigma = [v / total for v in exps]

    return sigma


def update_response_adversary_distribution(
    norm_scores: Dict[str, List[float]],
    objective_names: List[str],
    pi: List[float],
    steps: int,
    alpha: float,
    kappa: float,
    preference_scale: float,
) -> List[float]:
    """Response-only MW update with objectives marginalized uniformly."""
    n_obj = len(objective_names)
    n_resp = len(pi)
    sigma0 = [1.0 / n_resp for _ in range(n_resp)]
    sigma = sigma0[:]

    for _ in range(steps):
        costs = []
        for a in range(n_resp):
            c_resp = 0.0
            for obj in objective_names:
                c_obj = 0.0
                for y in range(n_resp):
                    c_obj += pi[y] * estimate_preference_prob(
                        norm_scores[obj][y],
                        norm_scores[obj][a],
                        preference_scale,
                    )
                c_resp += c_obj / n_obj
            costs.append(c_resp)

        logits = [
            (1.0 - alpha * kappa) * math.log(max(sigma[a], 1e-12))
            + alpha * kappa * math.log(max(sigma0[a], 1e-12))
            - alpha * costs[a]
            for a in range(n_resp)
        ]
        offset = max(logits)
        exps = [math.exp(v - offset) for v in logits]
        total = sum(exps)
        sigma = [v / total for v in exps]

    return sigma


def add_pair_fields(
    base: Dict[str, Any],
    chosen_idx: int,
    rejected_idx: int,
    target: float,
    pair_source: str,
    objective_name: Optional[str],
    objective_index: Optional[int],
    objective_gap: Optional[float],
    ronpo_weight: float = 1.0,
    adversary_response_index: Optional[int] = None,
    adversary_mass: Optional[float] = None,
) -> Dict[str, Any]:
    prompt = base["prompt"]
    responses = base["all_generated_responses"]
    out = dict(base)
    out["chosen"] = make_messages(prompt, responses[chosen_idx])
    out["rejected"] = make_messages(prompt, responses[rejected_idx])
    out["chosen_index"] = int(chosen_idx)
    out["rejected_index"] = int(rejected_idx)
    out["pair_source"] = pair_source
    out["ronpo_target"] = float(target)
    out["ronpo_objective_name"] = objective_name
    out["ronpo_objective_index"] = objective_index
    out["ronpo_objective_gap"] = objective_gap
    out["ronpo_weight"] = float(ronpo_weight)
    out["ronpo_adversary_response_index"] = adversary_response_index
    out["ronpo_adversary_mass"] = adversary_mass
    return out


def load_objective_records(named_paths: List[Tuple[str, str]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    by_objective: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name, path in named_paths:
        records = {}
        for record in read_records(path):
            records[prompt_key(record)] = record
        by_objective[name] = records
    return by_objective


def merge_prompt_records(
    by_objective: Dict[str, Dict[str, Dict[str, Any]]],
    objective_names: List[str],
) -> List[Dict[str, Any]]:
    common_keys = set(by_objective[objective_names[0]].keys())
    for name in objective_names[1:]:
        common_keys &= set(by_objective[name].keys())

    merged = []
    for key in sorted(common_keys):
        first = by_objective[objective_names[0]][key]
        prompt = first["prompt"]
        responses = first["all_generated_responses"]
        if not isinstance(responses, list) or len(responses) < 2:
            continue

        objective_scores: Dict[str, List[float]] = {}
        ok = True
        for name in objective_names:
            record = by_objective[name][key]
            if record.get("prompt") != prompt:
                ok = False
                break
            if record.get("all_generated_responses") != responses:
                ok = False
                break
            scores = record.get("all_rm_scores")
            if not isinstance(scores, list) or len(scores) != len(responses):
                ok = False
                break
            objective_scores[name] = [float(s) for s in scores]
        if not ok:
            continue

        merged.append(
            {
                "prompt_id": first.get("prompt_id", key),
                "prompt": prompt,
                "all_generated_responses": responses,
                "objective_names": objective_names,
                "objective_scores": objective_scores,
            }
        )
    return merged


def build_pairs_for_record(
    record: Dict[str, Any],
    normalization: str,
    ronpo_pair_strategy: str,
    tie_threshold: float,
    adversary_steps: int,
    adversary_alpha: float,
    adversary_kappa: float,
    preference_scale: float,
    policy_mode: str,
    policy_temperature: float,
    pairs_per_prompt: int,
    adversary_selection: str,
    ronpo_policy_pair_mode: str,
    ronpo_policy_samples_per_atom: int,
    k_only_fixed_atom: str,
    k_only_response_mode: str,
    common_pair_seed: str,
    expected_support_k: int = 0,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    objective_names = list(record["objective_names"])
    norm_scores = normalize_scores(record["objective_scores"], normalization)
    avg_scores = average_per_response(norm_scores, objective_names)
    min_scores = min_per_response(norm_scores, objective_names)
    avg_win_probs = average_win_probs(avg_scores, preference_scale)

    base = dict(record)
    base["normalized_objective_scores"] = norm_scores
    base["avg_objective_scores"] = avg_scores
    base["min_objective_scores"] = min_scores
    base["avg_oracle_win_probs"] = avg_win_probs
    base["homogeneous_oracle"] = f"average_{normalization}_objectives"
    base["homogeneous_oracle_objectives"] = objective_names
    base["homogeneous_oracle_preference_scale"] = float(preference_scale)

    avg_best = argmax(avg_scores)
    avg_worst = argmin(avg_scores)
    if avg_best == avg_worst:
        mnpo_pair = None
    else:
        mnpo_pair = add_pair_fields(
            base,
            chosen_idx=avg_best,
            rejected_idx=avg_worst,
            target=1.0,
            pair_source="mnpo_average_objective",
            objective_name=None,
            objective_index=None,
            objective_gap=avg_scores[avg_best] - avg_scores[avg_worst],
        )
        mnpo_pair["chosen_probs"] = float(avg_win_probs[avg_best])
        mnpo_pair["rejected_probs"] = float(avg_win_probs[avg_worst])
        mnpo_pair["avg_oracle_chosen_score"] = float(avg_scores[avg_best])
        mnpo_pair["avg_oracle_rejected_score"] = float(avg_scores[avg_worst])
        mnpo_pair["avg_oracle_score_gap"] = float(avg_scores[avg_best] - avg_scores[avg_worst])
        mnpo_pair["avg_oracle_chosen_win_prob"] = float(avg_win_probs[avg_best])
        mnpo_pair["avg_oracle_rejected_win_prob"] = float(avg_win_probs[avg_worst])

    ronpo_pairs: List[Dict[str, Any]] = []
    if ronpo_pair_strategy == "none":
        pass

    elif ronpo_pair_strategy == "sigma":
        pi = policy_distribution(avg_scores, policy_mode, policy_temperature)
        sigma, _ = update_adversary_distribution(
            norm_scores=norm_scores,
            objective_names=objective_names,
            pi=pi,
            steps=adversary_steps,
            alpha=adversary_alpha,
            kappa=adversary_kappa,
            preference_scale=preference_scale,
        )
        atoms = select_adversary_atoms(
            sigma,
            pairs_per_prompt,
            adversary_selection,
            prompt_id=base.get("prompt_id") or base.get("prompt"),
        )
        if atoms:
            mean_mass = sum(mass for _, _, mass in atoms) / len(atoms)
        else:
            mean_mass = 1.0
        sigma_flat = [mass for row in sigma for mass in row]

        base["ronpo_sigma"] = {
            objective_names[k]: row for k, row in enumerate(sigma)
        }
        base["ronpo_sigma_entropy"] = distribution_entropy(sigma_flat)
        base["ronpo_sigma_effective_atoms"] = math.exp(base["ronpo_sigma_entropy"])
        base["ronpo_sigma_max_mass"] = max(sigma_flat) if sigma_flat else 0.0
        base["ronpo_adversary_selection"] = adversary_selection

        sigma_for_expectation = [row[:] for row in sigma]
        total_atoms = sum(len(row) for row in sigma_for_expectation)
        if 0 < expected_support_k < total_atoms:
            ranked = sorted(
                ((sigma_for_expectation[k][a], k, a)
                 for k in range(len(sigma_for_expectation))
                 for a in range(len(sigma_for_expectation[k]))),
                reverse=True,
            )[:expected_support_k]
            kept = {(k, a) for _mass, k, a in ranked}
            kept_mass = sum(mass for mass, _k, _a in ranked)
            sigma_for_expectation = [
                [mass / kept_mass if (k, a) in kept else 0.0 for a, mass in enumerate(row)]
                for k, row in enumerate(sigma_for_expectation)
            ]
        base["ronpo_expected_support_k"] = int(expected_support_k)
        base["ronpo_expected_support_total"] = int(total_atoms)

        def atom_weight(mass: float, divisor: float = 1.0) -> float:
            if adversary_selection == "sample":
                return 1.0 / max(divisor, 1.0)
            return (mass / max(mean_mass, 1e-12)) / max(divisor, 1.0)

        if ronpo_policy_pair_mode == "expected_relative_policy_vs_policy":
            # True per-prompt atom EXPECTATION (Rao-Blackwellized 'expectation'
            # selection). Instead of expanding one row per atom (which injects
            # sign-conflicting, mass-spiky rows that only cancel as minibatch
            # noise), we analytically aggregate over the FULL sigma into a single
            # signed target per sampled response pair:
            #   r_bar(y) = sum_{k,a} sigma(k,a) * P_hat_k(y > a)
            #   target(y, y') = r_bar(y) - r_bar(y')   ,  weight = 1
            # This is the exact expectation the theory regresses toward; every row
            # carries uniform mass so the trainer's mean-normalization is a no-op.
            n_resp = len(avg_scores)
            r_bar = [0.0] * n_resp
            for y in range(n_resp):
                acc = 0.0
                for k, obj in enumerate(objective_names):
                    row = norm_scores[obj]
                    sig_row = sigma_for_expectation[k]
                    sy = row[y]
                    for a in range(n_resp):
                        acc += sig_row[a] * estimate_preference_prob(sy, row[a], preference_scale)
                r_bar[y] = acc
            n_pairs = pairs_per_prompt if pairs_per_prompt > 0 else n_resp
            prompt_id = base.get("prompt_id") or base.get("prompt")
            for sample_idx in range(max(1, n_pairs)):
                chosen_idx, rejected_idx = sample_common_policy_pair(
                    pi,
                    prompt_id,
                    sample_idx,
                    common_pair_seed,
                    fallback_objective_index=0,
                    fallback_adversary_index=0,
                )
                target = r_bar[chosen_idx] - r_bar[rejected_idx]
                if tie_threshold > 0.0 and abs(target) <= tie_threshold:
                    continue
                ronpo_pairs.append(
                    add_pair_fields(
                        base,
                        chosen_idx=chosen_idx,
                        rejected_idx=rejected_idx,
                        target=target,
                        pair_source="ronpo_sigma_expected_relative",
                        objective_name="__expected__",
                        objective_index=-1,
                        objective_gap=target,
                        ronpo_weight=1.0,
                        adversary_response_index=-1,
                        adversary_mass=1.0,
                    )
                )

        for obj_idx, adversary_idx, mass in atoms:
            obj = objective_names[obj_idx]
            if ronpo_policy_pair_mode == "best_vs_adversary":
                chosen_idx = argmax(norm_scores[obj])
                rejected_idx = adversary_idx
                if chosen_idx == rejected_idx:
                    rejected_idx = argmin(norm_scores[obj])
                gap = norm_scores[obj][chosen_idx] - norm_scores[obj][rejected_idx]
                if chosen_idx != rejected_idx and abs(gap) > tie_threshold:
                    if gap < 0:
                        chosen_idx, rejected_idx = rejected_idx, chosen_idx
                        gap = -gap
                    ronpo_pairs.append(
                        add_pair_fields(
                            base,
                            chosen_idx=chosen_idx,
                            rejected_idx=rejected_idx,
                            target=1.0,
                            pair_source="ronpo_sigma_best_vs_adversary",
                            objective_name=obj,
                            objective_index=obj_idx,
                            objective_gap=gap,
                            ronpo_weight=atom_weight(mass),
                            adversary_response_index=adversary_idx,
                            adversary_mass=mass,
                        )
                    )
            elif ronpo_policy_pair_mode == "all_policy_vs_adversary":
                candidate_indices = [i for i in range(len(avg_scores)) if i != adversary_idx]
                if ronpo_policy_samples_per_atom > 0:
                    candidate_indices = candidate_indices[:ronpo_policy_samples_per_atom]
                for policy_idx in candidate_indices:
                    gap = norm_scores[obj][policy_idx] - norm_scores[obj][adversary_idx]
                    if abs(gap) <= tie_threshold:
                        target = 0.0
                    else:
                        target = 1.0 if gap > 0 else -1.0
                    ronpo_pairs.append(
                        add_pair_fields(
                            base,
                            chosen_idx=policy_idx,
                            rejected_idx=adversary_idx,
                            target=target,
                            pair_source="ronpo_sigma_policy_vs_adversary",
                            objective_name=obj,
                            objective_index=obj_idx,
                            objective_gap=gap,
                            ronpo_weight=atom_weight(mass, len(candidate_indices)),
                            adversary_response_index=adversary_idx,
                            adversary_mass=mass,
                        )
                    )
            elif ronpo_policy_pair_mode == "relative_policy_vs_policy":
                samples = ronpo_policy_samples_per_atom if ronpo_policy_samples_per_atom > 0 else len(avg_scores)
                for sample_idx in range(samples):
                    chosen_idx, rejected_idx = sample_distinct_policy_pair(
                        pi,
                        base.get("prompt_id") or base.get("prompt"),
                        obj_idx,
                        adversary_idx,
                        sample_idx,
                    )
                    chosen_prob = estimate_preference_prob(
                        norm_scores[obj][chosen_idx],
                        norm_scores[obj][adversary_idx],
                        preference_scale,
                    )
                    rejected_prob = estimate_preference_prob(
                        norm_scores[obj][rejected_idx],
                        norm_scores[obj][adversary_idx],
                        preference_scale,
                    )
                    target = chosen_prob - rejected_prob
                    if tie_threshold > 0.0 and abs(target) <= tie_threshold:
                        continue
                    ronpo_pairs.append(
                        add_pair_fields(
                            base,
                            chosen_idx=chosen_idx,
                            rejected_idx=rejected_idx,
                            target=target,
                            pair_source="ronpo_sigma_relative_policy_vs_policy",
                            objective_name=obj,
                            objective_index=obj_idx,
                            objective_gap=target,
                            ronpo_weight=atom_weight(mass),
                            adversary_response_index=adversary_idx,
                            adversary_mass=mass,
                        )
                    )
            elif ronpo_policy_pair_mode == "expected_relative_policy_vs_policy":
                # Rows already emitted by the aggregated block above; no per-atom
                # expansion for the expectation estimator.
                pass
            else:
                raise ValueError(f"Unsupported ronpo_policy_pair_mode: {ronpo_policy_pair_mode}")

    elif ronpo_pair_strategy == "sigma_k_only":
        pi = policy_distribution(avg_scores, policy_mode, policy_temperature)
        if k_only_fixed_atom == "avg_worst":
            fixed_idx = avg_worst
        elif k_only_fixed_atom == "avg_best":
            fixed_idx = avg_best
        elif k_only_fixed_atom == "first":
            fixed_idx = 0
        else:
            raise ValueError(f"Unsupported k_only_fixed_atom: {k_only_fixed_atom}")

        if k_only_response_mode == "fixed":
            objective_sigma = update_objective_adversary_distribution(
                norm_scores=norm_scores,
                objective_names=objective_names,
                pi=pi,
                fixed_adversary_idx=fixed_idx,
                steps=adversary_steps,
                alpha=adversary_alpha,
                kappa=adversary_kappa,
                preference_scale=preference_scale,
            )
        elif k_only_response_mode == "uniform":
            objective_sigma = update_objective_adversary_distribution_uniform_a(
                norm_scores=norm_scores,
                objective_names=objective_names,
                pi=pi,
                steps=adversary_steps,
                alpha=adversary_alpha,
                kappa=adversary_kappa,
                preference_scale=preference_scale,
            )
        else:
            raise ValueError(f"Unsupported k_only_response_mode: {k_only_response_mode}")
        objective_atoms = [(k, mass) for k, mass in enumerate(objective_sigma)]
        if adversary_selection == "top":
            ranked_objectives = sorted(objective_atoms, key=lambda item: item[1], reverse=True)[
                : max(1, pairs_per_prompt)
            ]
        elif adversary_selection == "all":
            ranked_objectives = objective_atoms
        elif adversary_selection == "sample":
            weights = [mass for _, mass in objective_atoms]
            ranked_objectives = []
            for sample_idx in range(max(1, pairs_per_prompt)):
                idx = weighted_index(
                    weights,
                    deterministic_unit_interval(
                        base.get("prompt_id") or base.get("prompt"),
                        "objective_sigma",
                        sample_idx,
                    ),
                )
                ranked_objectives.append(objective_atoms[idx])
        else:
            raise ValueError(f"Unsupported adversary atom selection mode: {adversary_selection}")

        mean_mass = sum(mass for _, mass in ranked_objectives) / len(ranked_objectives)

        base["ronpo_objective_sigma"] = {
            objective_names[k]: objective_sigma[k] for k in range(len(objective_names))
        }
        base["ronpo_objective_sigma_entropy"] = distribution_entropy(objective_sigma)
        base["ronpo_objective_sigma_effective_objectives"] = math.exp(
            base["ronpo_objective_sigma_entropy"]
        )
        base["ronpo_objective_sigma_max_mass"] = max(objective_sigma) if objective_sigma else 0.0
        base["ronpo_k_only_fixed_atom"] = k_only_fixed_atom
        base["ronpo_k_only_fixed_response_index"] = fixed_idx
        base["ronpo_k_only_response_mode"] = k_only_response_mode

        objective_sigma_for_expectation = objective_sigma[:]
        if 0 < expected_support_k < len(objective_sigma_for_expectation):
            kept_indices = sorted(
                range(len(objective_sigma_for_expectation)),
                key=lambda index: objective_sigma_for_expectation[index],
                reverse=True,
            )[:expected_support_k]
            kept = set(kept_indices)
            kept_mass = sum(objective_sigma_for_expectation[index] for index in kept_indices)
            objective_sigma_for_expectation = [
                mass / kept_mass if index in kept else 0.0
                for index, mass in enumerate(objective_sigma_for_expectation)
            ]
        base["ronpo_expected_support_k"] = int(expected_support_k)
        base["ronpo_expected_support_total"] = len(objective_sigma)

        if ronpo_policy_pair_mode == "expected_relative_policy_vs_policy":
            # Matched objective-only expectation baseline for the full atom
            # expectation estimator above. With k_only_response_mode=uniform,
            # the response atom is marginalized uniformly:
            #   r_bar(y) = sum_k sigma(k) mean_a P_hat_k(y > a)
            # With k_only_response_mode=fixed, it keeps the legacy fixed atom:
            #   r_bar(y) = sum_k sigma(k) P_hat_k(y > fixed_idx)
            #   target(y, y') = r_bar(y) - r_bar(y')   ,  weight = 1
            n_resp = len(avg_scores)
            r_bar = [0.0] * n_resp
            for y in range(n_resp):
                acc = 0.0
                for k, obj in enumerate(objective_names):
                    row = norm_scores[obj]
                    if k_only_response_mode == "uniform":
                        objective_value = sum(
                            estimate_preference_prob(row[y], row[a], preference_scale)
                            for a in range(n_resp)
                        ) / n_resp
                    else:
                        objective_value = estimate_preference_prob(
                            row[y],
                            row[fixed_idx],
                            preference_scale,
                        )
                    acc += objective_sigma_for_expectation[k] * objective_value
                r_bar[y] = acc
            n_pairs = pairs_per_prompt if pairs_per_prompt > 0 else n_resp
            prompt_id = base.get("prompt_id") or base.get("prompt")
            for sample_idx in range(max(1, n_pairs)):
                chosen_idx, rejected_idx = sample_common_policy_pair(
                    pi,
                    prompt_id,
                    sample_idx,
                    common_pair_seed,
                    fallback_objective_index=0,
                    fallback_adversary_index=fixed_idx,
                )
                target = r_bar[chosen_idx] - r_bar[rejected_idx]
                if tie_threshold > 0.0 and abs(target) <= tie_threshold:
                    continue
                ronpo_pairs.append(
                    add_pair_fields(
                        base,
                        chosen_idx=chosen_idx,
                        rejected_idx=rejected_idx,
                        target=target,
                        pair_source="ronpo_sigma_k_only_expected_relative",
                        objective_name="__k_expected__",
                        objective_index=-1,
                        objective_gap=target,
                        ronpo_weight=1.0,
                        adversary_response_index=fixed_idx,
                        adversary_mass=1.0,
                    )
                )
            return mnpo_pair, ronpo_pairs

        for obj_idx, mass in ranked_objectives:
            obj = objective_names[obj_idx]
            chosen_idx = argmax(norm_scores[obj])
            rejected_idx = fixed_idx
            if chosen_idx == rejected_idx:
                rejected_idx = argmin(norm_scores[obj])
            gap = norm_scores[obj][chosen_idx] - norm_scores[obj][rejected_idx]
            if chosen_idx != rejected_idx and abs(gap) > tie_threshold:
                if gap < 0:
                    chosen_idx, rejected_idx = rejected_idx, chosen_idx
                    gap = -gap
                ronpo_pairs.append(
                    add_pair_fields(
                        base,
                        chosen_idx=chosen_idx,
                        rejected_idx=rejected_idx,
                        target=1.0,
                        pair_source="ronpo_sigma_k_only_fixed_atom",
                        objective_name=obj,
                        objective_index=obj_idx,
                        objective_gap=gap,
                        ronpo_weight=(
                            1.0 if adversary_selection == "sample" else mass / max(mean_mass, 1e-12)
                        ),
                        adversary_response_index=fixed_idx,
                        adversary_mass=mass,
                    )
                )

    elif ronpo_pair_strategy in {"uniform", "sigma_a_only", "maxmin_pointwise"}:
        pi = policy_distribution(avg_scores, policy_mode, policy_temperature)
        n_resp = len(avg_scores)
        if ronpo_pair_strategy == "uniform":
            r_bar = []
            for y in range(n_resp):
                acc = 0.0
                for obj in objective_names:
                    row = norm_scores[obj]
                    acc += sum(
                        estimate_preference_prob(row[y], row[a], preference_scale)
                        for a in range(n_resp)
                    ) / n_resp
                r_bar.append(acc / len(objective_names))
            base["ronpo_uniform_atoms"] = len(objective_names) * n_resp
            pair_source = "ronpo_uniform_expected_relative"
            objective_name = "__uniform_expected__"
            sigma_entropy = math.log(max(len(objective_names) * n_resp, 1))
            sigma_max_mass = 1.0 / max(len(objective_names) * n_resp, 1)
        elif ronpo_pair_strategy == "sigma_a_only":
            response_sigma = update_response_adversary_distribution(
                norm_scores=norm_scores,
                objective_names=objective_names,
                pi=pi,
                steps=adversary_steps,
                alpha=adversary_alpha,
                kappa=adversary_kappa,
                preference_scale=preference_scale,
            )
            r_bar = []
            for y in range(n_resp):
                acc = 0.0
                for a in range(n_resp):
                    atom_value = 0.0
                    for obj in objective_names:
                        row = norm_scores[obj]
                        atom_value += estimate_preference_prob(
                            row[y],
                            row[a],
                            preference_scale,
                        ) / len(objective_names)
                    acc += response_sigma[a] * atom_value
                r_bar.append(acc)
            base["ronpo_response_sigma"] = response_sigma
            base["ronpo_response_sigma_entropy"] = distribution_entropy(response_sigma)
            base["ronpo_response_sigma_effective_responses"] = math.exp(
                base["ronpo_response_sigma_entropy"]
            )
            base["ronpo_response_sigma_max_mass"] = max(response_sigma) if response_sigma else 0.0
            pair_source = "ronpo_sigma_a_only_expected_relative"
            objective_name = "__a_expected__"
            sigma_entropy = base["ronpo_response_sigma_entropy"]
            sigma_max_mass = base["ronpo_response_sigma_max_mass"]
        else:
            r_bar = []
            for y in range(n_resp):
                score_y = min_scores[y]
                r_bar.append(
                    sum(
                        estimate_preference_prob(score_y, min_scores[a], preference_scale)
                        for a in range(n_resp)
                    )
                    / n_resp
                )
            base["ronpo_maxmin_pointwise_scores"] = min_scores
            pair_source = "ronpo_maxmin_pointwise_expected_relative"
            objective_name = "__maxmin_pointwise__"
            sigma_entropy = math.log(max(n_resp, 1))
            sigma_max_mass = 1.0 / max(n_resp, 1)

        n_pairs = pairs_per_prompt if pairs_per_prompt > 0 else n_resp
        prompt_id = base.get("prompt_id") or base.get("prompt")
        for sample_idx in range(max(1, n_pairs)):
            chosen_idx, rejected_idx = sample_common_policy_pair(
                pi,
                prompt_id,
                sample_idx,
                common_pair_seed,
                fallback_objective_index=ronpo_pair_strategy,
                fallback_adversary_index=0,
            )
            target = r_bar[chosen_idx] - r_bar[rejected_idx]
            if tie_threshold > 0.0 and abs(target) <= tie_threshold:
                continue
            pair = add_pair_fields(
                base,
                chosen_idx=chosen_idx,
                rejected_idx=rejected_idx,
                target=target,
                pair_source=pair_source,
                objective_name=objective_name,
                objective_index=-1,
                objective_gap=target,
                ronpo_weight=1.0,
                adversary_response_index=-1,
                adversary_mass=1.0,
            )
            pair["ronpo_sigma_entropy"] = sigma_entropy
            pair["ronpo_sigma_effective_atoms"] = math.exp(sigma_entropy)
            pair["ronpo_sigma_max_mass"] = sigma_max_mass
            ronpo_pairs.append(pair)

    elif ronpo_pair_strategy == "adversarial":
        vulnerability = [avg_scores[i] - min_scores[i] for i in range(len(avg_scores))]
        rejected_idx = argmax(vulnerability)
        if vulnerability[rejected_idx] <= 1e-12:
            rejected_idx = avg_best
        weakest_obj_idx = argmin([norm_scores[obj][rejected_idx] for obj in objective_names])
        weakest_obj = objective_names[weakest_obj_idx]
        chosen_idx = argmax(norm_scores[weakest_obj])
        if chosen_idx == rejected_idx:
            rejected_idx = argmin(norm_scores[weakest_obj])
        gap = norm_scores[weakest_obj][chosen_idx] - norm_scores[weakest_obj][rejected_idx]
        if chosen_idx != rejected_idx and abs(gap) > tie_threshold:
            if gap < 0:
                chosen_idx, rejected_idx = rejected_idx, chosen_idx
                gap = -gap
            ronpo_pairs.append(
                add_pair_fields(
                    base,
                    chosen_idx=chosen_idx,
                    rejected_idx=rejected_idx,
                    target=1.0,
                    pair_source="ronpo_weakest_objective_vs_vulnerable_response",
                    objective_name=weakest_obj,
                    objective_index=weakest_obj_idx,
                    objective_gap=gap,
                )
            )
    elif ronpo_pair_strategy == "all_objectives":
        for obj_idx, obj in enumerate(objective_names):
            chosen_idx = argmax(norm_scores[obj])
            rejected_idx = argmin(norm_scores[obj])
            gap = norm_scores[obj][chosen_idx] - norm_scores[obj][rejected_idx]
            if chosen_idx != rejected_idx and abs(gap) > tie_threshold:
                ronpo_pairs.append(
                    add_pair_fields(
                        base,
                        chosen_idx=chosen_idx,
                        rejected_idx=rejected_idx,
                        target=1.0,
                        pair_source="ronpo_objective_best_vs_worst",
                        objective_name=obj,
                        objective_index=obj_idx,
                        objective_gap=gap,
                    )
                )
    elif ronpo_pair_strategy == "maximin_vs_avg":
        robust_best = argmax(min_scores)
        weakest_obj_idx = argmin([norm_scores[obj][avg_best] for obj in objective_names])
        weakest_obj = objective_names[weakest_obj_idx]
        chosen_idx = robust_best
        rejected_idx = avg_best
        if chosen_idx == rejected_idx:
            rejected_idx = argmin(norm_scores[weakest_obj])
        gap = norm_scores[weakest_obj][chosen_idx] - norm_scores[weakest_obj][rejected_idx]
        if chosen_idx != rejected_idx and abs(gap) > tie_threshold:
            target = 1.0 if gap > 0 else -1.0
            ronpo_pairs.append(
                add_pair_fields(
                    base,
                    chosen_idx=chosen_idx,
                    rejected_idx=rejected_idx,
                    target=target,
                    pair_source="ronpo_maximin_vs_avg_best",
                    objective_name=weakest_obj,
                    objective_index=weakest_obj_idx,
                    objective_gap=gap,
                )
            )
    else:
        raise ValueError(f"Unsupported ronpo_pair_strategy: {ronpo_pair_strategy}")

    return mnpo_pair, ronpo_pairs


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: str, rows: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MNPO-average and RONPO-robust pairs from multiple objective scores.")
    parser.add_argument("--scored_files", nargs="+", required=True, help="Objective scored files, e.g. skywork=... athene=...")
    parser.add_argument("--mnpo_output", required=True)
    parser.add_argument("--ronpo_output", required=True)
    parser.add_argument("--merged_output", default=None)
    parser.add_argument("--summary_output", default=None)
    parser.add_argument("--normalization", default="minmax", choices=["minmax", "rank", "none"])
    parser.add_argument("--ronpo_pair_strategy", default="sigma",
                        choices=[
                            "none",
                            "sigma",
                            "sigma_k_only",
                            "sigma_a_only",
                            "uniform",
                            "maxmin_pointwise",
                            "adversarial",
                            "all_objectives",
                            "maximin_vs_avg",
                        ])
    parser.add_argument("--tie_threshold", type=float, default=0.0)
    parser.add_argument("--adversary_steps", type=int, default=25)
    parser.add_argument("--adversary_alpha", type=float, default=1.0)
    parser.add_argument("--adversary_kappa", type=float, default=0.05)
    parser.add_argument("--preference_scale", type=float, default=8.0)
    parser.add_argument("--policy_mode", default="uniform", choices=["uniform", "softmax_avg"])
    parser.add_argument("--policy_temperature", type=float, default=0.2)
    parser.add_argument("--pairs_per_prompt", type=int, default=3)
    parser.add_argument("--adversary_selection", default="top", choices=["top", "all", "sample"])
    parser.add_argument("--ronpo_policy_pair_mode", default="best_vs_adversary",
                        choices=["best_vs_adversary", "all_policy_vs_adversary", "relative_policy_vs_policy",
                                 "expected_relative_policy_vs_policy"])
    parser.add_argument("--ronpo_policy_samples_per_atom", type=int, default=0,
                        help=(
                            "For all_policy_vs_adversary, 0 means use every non-adversary response in the pool. "
                            "For relative_policy_vs_policy, values >0 set the number of deterministic pi_t "
                            "response-pair samples per adversarial atom; 0 uses len(response_pool)."
                        ))
    parser.add_argument("--k_only_fixed_atom", default="avg_worst", choices=["avg_worst", "avg_best", "first"],
                        help=(
                            "For ronpo_pair_strategy=sigma_k_only, fix the response atom before the adversary "
                            "chooses objectives. avg_worst is the average-objective worst response in the prompt pool."
                        ))
    parser.add_argument("--k_only_response_mode", default="fixed", choices=["fixed", "uniform"],
                        help=(
                            "For ronpo_pair_strategy=sigma_k_only. fixed preserves the legacy fixed-a baseline; "
                            "uniform marginalizes a uniformly to match the weight-only adversary in the AAAI ablation."
                        ))
    parser.add_argument("--common_pair_seed", default="",
                        help=(
                            "If non-empty, expectation-style target builders sample the same deterministic "
                            "(y,y') pair set across modes for common-random-number ablations."
                        ))
    parser.add_argument(
        "--expected_support_k", type=int, default=0,
        help=(
            "For expectation-style RONPO targets, keep and renormalize only the K highest-mass "
            "adversary atoms/objectives. Zero retains the full support and preserves legacy behavior."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    named_paths = parse_named_paths(args.scored_files)
    objective_names = [name for name, _ in named_paths]
    by_objective = load_objective_records(named_paths)
    merged = merge_prompt_records(by_objective, objective_names)

    mnpo_rows: List[Dict[str, Any]] = []
    ronpo_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for record in merged:
        mnpo_pair, pairs = build_pairs_for_record(
            record,
            normalization=args.normalization,
            ronpo_pair_strategy=args.ronpo_pair_strategy,
            tie_threshold=args.tie_threshold,
            adversary_steps=args.adversary_steps,
            adversary_alpha=args.adversary_alpha,
            adversary_kappa=args.adversary_kappa,
            preference_scale=args.preference_scale,
            policy_mode=args.policy_mode,
            policy_temperature=args.policy_temperature,
            pairs_per_prompt=args.pairs_per_prompt,
            adversary_selection=args.adversary_selection,
            ronpo_policy_pair_mode=args.ronpo_policy_pair_mode,
            ronpo_policy_samples_per_atom=args.ronpo_policy_samples_per_atom,
            k_only_fixed_atom=args.k_only_fixed_atom,
            k_only_response_mode=args.k_only_response_mode,
            common_pair_seed=args.common_pair_seed,
            expected_support_k=args.expected_support_k,
        )
        if mnpo_pair is not None:
            mnpo_rows.append(mnpo_pair)
        ronpo_rows.extend(pairs)
        for pair in pairs:
            summary_rows.append(
                {
                    "prompt_id": pair.get("prompt_id"),
                    "pair_source": pair.get("pair_source"),
                    "objective": pair.get("ronpo_objective_name"),
                    "objective_gap": pair.get("ronpo_objective_gap"),
                    "chosen_index": pair.get("chosen_index"),
                    "rejected_index": pair.get("rejected_index"),
                    "adversary_response_index": pair.get("ronpo_adversary_response_index"),
                    "adversary_mass": pair.get("ronpo_adversary_mass"),
                    "ronpo_weight": pair.get("ronpo_weight"),
                    "sigma_entropy": pair.get("ronpo_sigma_entropy"),
                    "sigma_effective_atoms": pair.get("ronpo_sigma_effective_atoms"),
                    "sigma_max_mass": pair.get("ronpo_sigma_max_mass"),
                    "adversary_selection": pair.get("ronpo_adversary_selection"),
                }
            )

    write_jsonl(args.mnpo_output, mnpo_rows)
    write_jsonl(args.ronpo_output, ronpo_rows)
    if args.merged_output:
        write_jsonl(args.merged_output, merged)
    if args.summary_output:
        write_summary(args.summary_output, summary_rows)

    print(f"Loaded common prompts: {len(merged)}")
    print(f"Wrote MNPO-average pairs: {len(mnpo_rows)} -> {args.mnpo_output}")
    print(f"Wrote RONPO-robust pairs: {len(ronpo_rows)} -> {args.ronpo_output}")


if __name__ == "__main__":
    main()
