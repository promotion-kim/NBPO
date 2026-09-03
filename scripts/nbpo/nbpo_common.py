"""Shared helpers for the paper-exact NBPO pipeline (``scripts/nbpo``).

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
)
PRESENTATION_ORDERS = ("learner_first", "comparator_first")
LEARNER_POOLS = ("policy", "reference")


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


def load_response_files(specs: Sequence[str]) -> Dict[str, Dict[str, dict]]:
    """Parse ``seed=path.json`` specs into ``{seed: {prompt_id: row}}``.

    Each file is a JSON list of rows with at least ``prompt_id``, ``prompt``,
    and ``generated_text`` (the repo's decode output format). The seed key is
    the generation seed label and is preserved into artifact metadata so every
    judged row is traceable to (response_id, generation seed).
    """
    out: Dict[str, Dict[str, dict]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"response file spec must be seed=path.json, got {spec!r}")
        seed, path = spec.split("=", 1)
        rows = json.loads(Path(path).read_text())
        out[seed] = {str(r["prompt_id"]): r for r in rows}
    if not out:
        raise ValueError("no response files given")
    return out


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
