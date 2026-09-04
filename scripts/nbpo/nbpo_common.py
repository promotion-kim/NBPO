"""Shared helpers for the finite-pool NBPO realization (``scripts/nbpo``).

This path never touches scalar reward-model scores or Bradley--Terry
reconstruction, and never imputes a missing comparison as a tie.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import yaml

SCHEMA_VERSION = 1
COMPARISON_SCHEMA_VERSION = 1

# Canonical judged-row schema (every row written by judge_pairwise_matrix.py).
JUDGE_ROW_FIELDS = (
    "prompt_id",
    "objective",
    "learner_response_id",
    "comparator_response_id",
    "presentation_order",
    "policy_win",
    "valid",
    "judge_model",
    "rubric_version",
    "comparison_content_hash",
)
PRESENTATION_ORDERS = ("learner_first", "comparator_first")
LEARNER_POOLS = ("policy", "reference")

# ---------------------------------------------------------------------------
# What this code actually runs (audit section 0).
#
# The manuscript's Algorithm 1 is written at the POPULATION level: the inner
# maximiser of Eq. (17) is the exact regularized best response, and the dual
# loop of Eq. (27) is stated over that exact inner solution. This code is the
# PRACTICAL realization of that algorithm and is a different abstraction level:
# the dual runs on a FROZEN FINITE RESPONSE POOL where the inner maximiser is a
# closed-form distribution over the pooled responses, and the 8B neural policy
# is fit ONCE, after the dual has converged, from the Eq. (26) targets. No
# neural training happens inside a dual iteration.
#
# Every artifact this pipeline writes carries these fields so a reader never has
# to infer which of the two is on disk. See docs/NBPO_ALGORITHM_MAPPING.md.
# ---------------------------------------------------------------------------
IMPLEMENTATION_TYPE = "finite_pool_one_shot_neural_realization"
DUAL_POLICY_REPRESENTATION = "finite_response_distribution"
NEURAL_FITS_PER_OUTER_STAGE = 1


def implementation_contract(dual_iterations=None, fixed_point_steps=None) -> dict:
    """The mandatory contract block embedded in every NBPO artifact."""
    return {
        "implementation_type": IMPLEMENTATION_TYPE,
        "dual_policy_representation": DUAL_POLICY_REPRESENTATION,
        "neural_fits_per_outer_stage": NEURAL_FITS_PER_OUTER_STAGE,
        "fixed_point_steps": (None if fixed_point_steps is None else int(fixed_point_steps)),
        "dual_iterations": (None if dual_iterations is None else int(dual_iterations)),
        # Deliberately NOT "note": artifacts carry their own notes (the pair
        # summary's says the target is unscaled), and a generic key would clobber
        # them when this block is spread into one.
        "implementation_note": (
            "finite-pool NBPO realization: the dual of Eq. (27) runs on a frozen "
            "finite response pool and the neural policy is realized once from the "
            "Eq. (26) targets afterwards -- this is NOT population Algorithm 1, "
            "which solves the inner problem exactly at every dual step"),
    }


# ---------------------------------------------------------------------------
# Reproduction-mode cardinalities (audit section 6). The manuscript's sampling
# table fixes these exactly; outside reproduction mode any pool size is allowed
# but the observed cardinalities are still recorded.
# ---------------------------------------------------------------------------
REPRODUCTION_CARDINALITY = {
    "learner_responses_per_prompt": 4,
    "reference_comparators_per_prompt": 4,
    "unordered_learner_pairs": 6,
    "presentation_orders_per_semantic_comparison": 2,
}


class RubricUnavailableError(RuntimeError):
    """Raised when a requested objective has no migrated exact rubric."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False))


def stable_hash_int(*parts: str) -> int:
    """Deterministic 64-bit integer from string parts (for seeds and the mock judge)."""
    return int(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16], 16)


def read_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def jsonl_has_records(path) -> bool:
    """True iff ``path`` exists and holds at least one parseable JSON object.

    ``build_nbpo_pairs`` writes ``pairs_test.jsonl`` unconditionally, so the file
    exists even when no prompts were held out. Passing that empty file to
    ``--test_dir`` used to reach ``torch.cat([])`` inside precompute; every
    caller now gates the flag on this predicate instead of on file existence.
    A malformed line is a hard error, not a reason to call the file empty.
    """
    path = Path(path)
    if not path.is_file():
        return False
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} is not valid JSON: {e}") from e
            return True
    return False


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def load_response_files(specs: Sequence[str], reproduction_mode: bool = True
                        ) -> Dict[str, Dict[str, dict]]:
    """Parse ``seed=path.json`` specs into ``{seed: {prompt_id: row}}``.

    Each file is a JSON list of rows with at least ``prompt_id``, ``prompt``,
    and ``generated_text`` (the repo's decode output format). The seed key is
    the generation seed label and is preserved into artifact metadata so every
    judged row is traceable to (response_id, generation seed).

    A duplicate prompt id inside one file, or a duplicate seed key across specs,
    used to be silently overwritten by the dict build -- which quietly dropped a
    response from the pool and changed the pool cardinality without a trace.
    Both now raise, naming the file, the seed, and the duplicated id. Empty
    generated text is also rejected in reproduction mode: an empty response is a
    decode failure, not a valid pool member.
    """
    out: Dict[str, Dict[str, dict]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"response file spec must be seed=path.json, got {spec!r}")
        seed, path = spec.split("=", 1)
        if seed in out:
            raise ValueError(f"duplicate seed key {seed!r} in response file specs: {list(specs)}")
        rows = json.loads(Path(path).read_text())
        by_id: Dict[str, dict] = {}
        for r in rows:
            pid = str(r["prompt_id"])
            if pid in by_id:
                raise ValueError(
                    f"duplicate prompt_id {pid!r} in {path} (seed {seed!r}); refusing to "
                    "silently overwrite -- the pool cardinality would change unrecorded"
                )
            if reproduction_mode and not str(r.get("generated_text", "")).strip():
                raise ValueError(
                    f"empty generated_text for prompt_id {pid!r} in {path} (seed {seed!r}); "
                    "an empty response is a decode failure, not a pool member"
                )
            by_id[pid] = r
        out[seed] = by_id
    if not out:
        raise ValueError("no response files given")
    return out


def check_pool_cardinality(learner_pools, comparator_pools, reproduction_mode: bool) -> dict:
    """Record -- and in reproduction mode enforce -- the manuscript pool sizes.

    Returns the observed cardinalities either way, so a non-reproduction run is
    still self-describing rather than silently unlabelled.
    """
    n_learner = len(learner_pools)
    n_comparator = len(comparator_pools)
    observed = {
        "learner_responses_per_prompt": n_learner,
        "reference_comparators_per_prompt": n_comparator,
        "unordered_learner_pairs": n_learner * (n_learner - 1) // 2,
        "presentation_orders_per_semantic_comparison": len(PRESENTATION_ORDERS),
        "reproduction_mode": bool(reproduction_mode),
    }
    if not reproduction_mode:
        return observed
    problems = [
        f"{key}={observed[key]} (manuscript requires {want})"
        for key, want in REPRODUCTION_CARDINALITY.items()
        if observed[key] != want
    ]
    if problems:
        raise ValueError(
            "reproduction_mode cardinality violation: " + "; ".join(problems)
            + " -- set reproduction_mode: false to run a different pool geometry"
        )
    return observed


def load_objective_rubrics(config_path: Path, objectives: Sequence[str]) -> dict:
    """Load rubrics for ``objectives`` from a versioned YAML file, failing loudly.

    The YAML schema is::

        version: <int>
        dataset: <name>
        objectives:
          <name>:
            available: true|false
            source: <provenance path in this repo>
            system: <exact system prompt>            # required when available
            user_template: <template with {prompt},{a},{b}>

    Any requested objective that is missing, or marked ``available: false``,
    raises :class:`RubricUnavailableError` naming it -- rubrics whose exact
    experimental text is not in this repository are never invented.
    """
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text())
    declared = cfg.get("objectives", {})
    missing, unavailable = [], []
    rubrics = {}
    for obj in objectives:
        entry = declared.get(obj)
        if entry is None:
            missing.append(obj)
            continue
        if not entry.get("available", False):
            unavailable.append((obj, entry.get("reason", "no reason recorded")))
            continue
        if not entry.get("system") or not entry.get("user_template"):
            raise RubricUnavailableError(
                f"objective {obj!r} in {config_path} is marked available but lacks "
                "system/user_template text"
            )
        rubrics[obj] = {
            "system": entry["system"],
            "user_template": entry["user_template"],
            "source": entry.get("source", "unknown"),
        }
    problems = []
    if missing:
        problems.append(f"not defined in {config_path}: {missing}")
    if unavailable:
        details = "; ".join(f"{o} ({reason})" for o, reason in unavailable)
        problems.append(f"marked unavailable in {config_path}: {details}")
    if problems:
        raise RubricUnavailableError(
            "exact rubric(s) unavailable -- refusing to proceed rather than invent "
            "prompt text. " + " | ".join(problems)
        )
    return {
        "version": cfg.get("version"),
        "dataset": cfg.get("dataset"),
        "config_hash": sha256_file(config_path),
        "rubrics": rubrics,
    }


# ---------------------------------------------------------------------------
# Content-addressed judging (audit section 2).
#
# Task identity used to be (prompt_id, objective, pool, learner_id,
# comparator_id, order) -- all IDENTIFIERS, no CONTENT. A newly trained
# candidate decoded into the same seed ids under the same prompt ids therefore
# collided with the previous candidate's cache keys, and its verdicts were
# reused verbatim: the gate would score the OLD candidate's responses. The task
# id is now the sha256 of a canonical payload over the exact texts plus every
# setting that can change a verdict, so any change of text, rubric, judge or
# decoding parameter is a different task and forces re-judging.
# ---------------------------------------------------------------------------
COMPARISON_PAYLOAD_FIELDS = (
    "schema_version", "prompt_id", "prompt_sha256", "objective",
    "learner_pool", "learner_response_id", "learner_text_sha256",
    "comparator_response_id", "comparator_text_sha256", "presentation_order",
    "judge_model", "judge_revision", "tokenizer_revision", "rubric_sha256",
    "judge_temperature", "judge_top_p", "judge_top_k", "judge_max_tokens",
    "judge_max_retries", "judge_retry_temperature",
    "judge_chat_template_kwargs", "renderer_schema_version",
)


def comparison_payload(task: dict, judge_identity: dict) -> dict:
    """Canonical, fully content-addressed description of one judging cell."""
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "prompt_id": str(task["prompt_id"]),
        "prompt_sha256": sha256_text(str(task["prompt"])),
        "objective": str(task["objective"]),
        "learner_pool": str(task.get("learner_pool", "policy")),
        "learner_response_id": str(task["learner_response_id"]),
        "learner_text_sha256": sha256_text(str(task["learner_text"])),
        "comparator_response_id": str(task["comparator_response_id"]),
        "comparator_text_sha256": sha256_text(str(task["comparator_text"])),
        "presentation_order": str(task["presentation_order"]),
        "judge_model": str(judge_identity.get("judge_model")),
        "judge_revision": judge_identity.get("judge_revision"),
        "tokenizer_revision": judge_identity.get("tokenizer_revision"),
        "rubric_sha256": judge_identity.get("rubric_sha256"),
        "judge_temperature": judge_identity.get("temperature"),
        "judge_top_p": judge_identity.get("top_p"),
        "judge_top_k": judge_identity.get("top_k"),
        "judge_max_tokens": judge_identity.get("max_tokens"),
        "judge_max_retries": judge_identity.get("max_retries"),
        "judge_retry_temperature": judge_identity.get("retry_temperature"),
        # The kwargs that were APPLIED when rendering, plus the renderer version:
        # both change the prompt text the judge saw, so both change the task id.
        "judge_chat_template_kwargs": judge_identity.get("chat_template_kwargs"),
        "renderer_schema_version": judge_identity.get("renderer_schema_version"),
    }


def comparison_content_hash(task: dict, judge_identity: dict) -> str:
    """``sha256(canonical_json(payload))`` -- the cache key and the task id."""
    return sha256_json(comparison_payload(task, judge_identity))


def judge_config_hash(judge_identity: dict) -> str:
    """Identity of the judge configuration alone (model, rubric, decoding, retries).

    Verdict caches are keyed by BOTH the response-pool hash and this, so two
    judge configurations can never share a cache directory.
    """
    # chat_template_kwargs and the renderer version belong here: they change the
    # prompt string the judge actually sees, so two configs that differ only in
    # them must not share a verdict cache.
    keys = ("judge_model", "judge_revision", "tokenizer_revision", "rubric_sha256",
            "temperature", "top_p", "top_k", "max_tokens", "max_retries",
            "retry_temperature", "seed_policy", "backend",
            "chat_template_kwargs", "renderer_schema_version")
    return sha256_json({k: judge_identity.get(k) for k in keys})


def response_pool_hash(*pools) -> str:
    """Identity of the judged response pools: every response id and its exact text."""
    payload = []
    for pool in pools:
        entry = {}
        for seed in sorted(pool):
            entry[seed] = {pid: sha256_text(str(row.get("generated_text", "")))
                           for pid, row in sorted(pool[seed].items())}
        payload.append(entry)
    return sha256_json(payload)


def verdict_cache_path(root, pool_hash: str, judge_cfg_hash: str) -> Path:
    """``verdicts/<pool_hash>/<judge_config_hash>/verdicts.jsonl``."""
    return Path(root) / "verdicts" / pool_hash[:16] / judge_cfg_hash[:16] / "verdicts.jsonl"


DEFAULT_JUDGE_DECODING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "max_tokens": 512,
    "retry_temperature": 0.3,
}


def normalize_judge_config(cfg: dict, role: str, rubric_sha256=None,
                           max_retries=None) -> dict:
    """Flatten a ``judge.<role>`` block into the identity every artifact records.

    Decoding values live under ``decoding:``; anything absent takes the
    documented default rather than a backend's hard-coded constant, because the
    backend is required to APPLY exactly what this returns (see
    ``JudgeBackend.effective_config``).
    """
    dec = dict(DEFAULT_JUDGE_DECODING)
    dec.update(cfg.get("decoding") or {})
    retries = cfg.get("max_retries", 2) if max_retries is None else max_retries
    return {
        "role": role,
        "backend": cfg.get("backend"),
        "judge_model": cfg.get("model_path"),
        "judge_revision": cfg.get("model_revision"),
        "tokenizer_revision": cfg.get("tokenizer_revision"),
        "rubric_sha256": rubric_sha256,
        "temperature": float(dec["temperature"]),
        "top_p": float(dec["top_p"]),
        "top_k": int(dec["top_k"]),
        "max_tokens": int(dec["max_tokens"]),
        "retry_temperature": float(dec["retry_temperature"]),
        "max_retries": int(retries),
        "seed_policy": str(cfg.get("seed_policy", "fixed_42_plus_attempt")),
        "chat_template_kwargs": dict(cfg.get("chat_template_kwargs") or {}),
    }


# ---------------------------------------------------------------------------
# Reproduction-mode invariants (audit section 11).
#
# These are the settings under which the manuscript numbers were produced. Any
# one of them silently flipped changes the objective being optimized, so in
# reproduction mode they are asserted rather than defaulted.
# ---------------------------------------------------------------------------
def check_reproduction_invariants(solver_cfg: dict, trainer_cfg: dict,
                                  target_summary: dict = None) -> dict:
    """Assert every reproduction-mode invariant; returns the checked values."""
    problems = []
    damping = float(solver_cfg.get("damping", 0.0) or 0.0)
    if damping != 0.0:
        problems.append(f"solver.damping must be 0 in reproduction mode, got {damping}")
    if solver_cfg.get("normalize_lambda"):
        problems.append("lambda must stay RAW in reproduction mode "
                        "(normalizing it rescales the Eq. (21) step by sum_k lambda_k)")
    reduction = str(trainer_cfg.get("logp_reduction", "mean")).lower()
    if reduction != "sum":
        problems.append(f"logp_reduction must be 'sum', got {reduction!r}")
    if int(trainer_cfg.get("max_history_t", 0) or 0) != 1:
        problems.append("max_history_t must be 1 (exactly one proximal centre pi_t)")
    hw = [float(w) for w in (trainer_cfg.get("history_weights") or [])]
    if hw and hw != [1.0]:
        problems.append(f"history_weights must be [1.0], got {hw}")
    if float(trainer_cfg.get("reference_anchor_weight", 0.0) or 0.0) != 0.0:
        problems.append("reference_anchor_weight must be 0 (Eq. (26) has no anchor loss)")
    if float(trainer_cfg.get("preference_sft_weight", 0.0) or 0.0) != 0.0:
        problems.append("preference_sft_weight must be 0 (Eq. (26) has no preference-SFT loss)")
    # eta appears in exactly one place: the trainer multiplies the stored target
    # by it. If the pair builder had also scaled the target, eta would enter
    # twice and the realized step would be eta^2.
    if target_summary is not None:
        note = str(target_summary.get("note", ""))
        if "unscaled" not in note:
            problems.append(
                "pair artifact does not declare its target UNSCALED; eta must be applied "
                "exactly once, by the trainer")
        scope = target_summary.get("opponent_sampling_scope")
        if scope != "pair_objective":
            problems.append(
                f"opponent_sampling_scope must be 'pair_objective', got {scope!r} "
                "(one z_k per response pair AND objective, Eq. (26))")
    if problems:
        raise ValueError("reproduction_mode invariant violation: " + "; ".join(problems))
    return {
        "damping": damping,
        "lambda_scale": "raw",
        "logp_reduction": reduction,
        "max_history_t": int(trainer_cfg.get("max_history_t", 1) or 1),
        "history_weights": hw or [1.0],
        "reference_anchor_weight": 0.0,
        "preference_sft_weight": 0.0,
        "eta_applications": 1,
        "opponent_sampling_scope": "pair_objective",
    }
