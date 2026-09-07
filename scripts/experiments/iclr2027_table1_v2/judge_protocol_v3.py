#!/usr/bin/env python3
"""Judge protocol v3: calibrated, uncertainty-aware pairwise scoring.

The v2 hard-verdict protocol forced a three-way choice and threw away how sure
the judge was. On a pool of same-policy samples that is the wrong instrument:
a pair the judge finds genuinely close and a pair it flips on by position both
come out as `p_hat = 0.5`, and the tensor cannot tell them apart. v3 keeps the
uncertainty instead of discarding it.

Protocols
---------

``P0`` -- the v2 hard verdict, retained only as a baseline diagnostic.

``P1`` -- **restricted-choice soft logits** (preferred). The chat template with
``enable_thinking: false`` ends exactly where the verdict token goes, so one
constrained decoding step gives an exact distribution over three verified
single-token labels. Probabilities are read from the sampler's own restricted
log-softmax, not estimated from a sampled winner.

    A verified-tokenization note that matters: under the Qwen3 tokenizer
    ``TIE`` is TWO tokens (`[51, 5371]`) and ``[[A]]`` is three, so scoring
    those strings as single candidates would be silently wrong. The label set
    is therefore the verified single-token sentinels ``A`` / ``B`` / ``T``,
    mapped back to A / B / TIE in every stored artifact.

``P2`` -- **deliberative structured verdict**: a short objective-specific
decision procedure, at most one rationale sentence, then the marker. Two
semantically equivalent templates, both orders, averaged.

Semantics (identical for every protocol, so they are comparable)
----------------------------------------------------------------

With ``y`` the learner response and ``z`` the comparator::

    forward order  (Response A = y, Response B = z):
        s_fwd(y) = p_A + 0.5 * p_T
    reverse order  (Response A = z, Response B = y):
        s_rev(y) = p_B + 0.5 * p_T

    p_hat(y > z) = 0.5 * (s_fwd(y) + s_rev(y))
    Delta(y, z)  = p_hat - 0.5,     Delta(z, y) = -Delta(y, z)

A tie contributes half to each side, so a confident tie gives ``Delta = 0`` --
the same value a position flip gives, *but now the two are distinguishable*,
because a confident tie has high ``p_T`` and low entropy while a flip has a
large forward/reverse gap. That distinction is the entire point of v3.

Adaptive adjudication (Section D)
---------------------------------
A pair is re-examined under two further independently worded templates when
``|s_fwd - s_rev| > order_gap_threshold`` or the normalized choice entropy
exceeds a calibration-selected threshold. Stable pairs cost 2 evaluations,
unstable ones at most 6. After the budget, an unresolved pair is **not** forced
to a winner: it keeps its soft near-zero margin, is flagged
``unresolved_uncertainty``, stays in the training tensor, and is excluded from
hard-cycle diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

FORWARD, REVERSE = "learner_first", "comparator_first"
CHOICES = ("A", "B", "TIE")


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# semantics
# --------------------------------------------------------------------------

def semantic_score(probs: dict, order: str) -> float:
    """Probability mass this order assigns to the LEARNER, ties split evenly."""
    if order == FORWARD:      # learner is in the A slot
        return probs["A"] + 0.5 * probs["TIE"]
    if order == REVERSE:      # learner is in the B slot
        return probs["B"] + 0.5 * probs["TIE"]
    raise ValueError(f"unknown presentation order {order!r}")


def normalized_entropy(probs: dict) -> float:
    """Shannon entropy over the three choices, scaled to [0, 1]."""
    h = 0.0
    for c in CHOICES:
        p = float(probs[c])
        if p > 0:
            h -= p * math.log(p)
    return h / math.log(len(CHOICES))


def combine(observations: List[dict]) -> dict:
    """Average semantic scores over every (order, template) observation.

    Both orders must be present, or the average carries the position bias that
    swap averaging exists to cancel.
    """
    fwd = [o for o in observations if o["presentation_order"] == FORWARD]
    rev = [o for o in observations if o["presentation_order"] == REVERSE]
    if not fwd or not rev:
        raise ValueError("both presentation orders are required for a semantic score")
    s_fwd = sum(o["semantic_score"] for o in fwd) / len(fwd)
    s_rev = sum(o["semantic_score"] for o in rev) / len(rev)
    p_hat = 0.5 * (s_fwd + s_rev)
    ent = [o["normalized_entropy"] for o in observations]
    tie = [o["probs"]["TIE"] for o in observations]
    return {
        "p_hat": p_hat,
        "delta": p_hat - 0.5,
        "semantic_score_forward": s_fwd,
        "semantic_score_reverse": s_rev,
        "order_gap": abs(s_fwd - s_rev),
        "mean_normalized_entropy": sum(ent) / len(ent),
        "max_normalized_entropy": max(ent),
        "mean_tie_probability": sum(tie) / len(tie),
        "n_observations": len(observations),
        "hard_argmax_verdict": max(CHOICES, key=lambda c: sum(o["probs"][c]
                                                             for o in observations)),
    }


def needs_adjudication(agg: dict, order_gap_threshold: float,
                       entropy_threshold: float) -> bool:
    return (agg["order_gap"] > order_gap_threshold
            or agg["mean_normalized_entropy"] > entropy_threshold)


# --------------------------------------------------------------------------
# protocol spec
# --------------------------------------------------------------------------

@dataclass
class Protocol:
    name: str
    kind: str                       # P0 | P1 | P2
    rubric_path: Path
    rubric_sha256: str
    labels: dict                    # {"A": {"token": "A", "id": 32}, ...}
    common_system: str
    verdict_instruction: str
    templates: List[dict]
    order_gap_threshold: float = 0.25
    entropy_threshold: float = 0.90
    max_templates_when_unstable: int = 3
    max_tokens: int = 1
    decoding: dict = field(default_factory=dict)

    @property
    def label_ids(self) -> List[int]:
        return [self.labels[c]["id"] for c in CHOICES]

    def sha256(self) -> str:
        payload = json.dumps({
            "name": self.name, "kind": self.kind,
            "rubric_sha256": self.rubric_sha256,
            "labels": self.labels, "common_system": self.common_system,
            "verdict_instruction": self.verdict_instruction,
            "templates": self.templates,
            "order_gap_threshold": self.order_gap_threshold,
            "entropy_threshold": self.entropy_threshold,
            "max_tokens": self.max_tokens, "decoding": self.decoding,
        }, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()

    def manifest(self) -> dict:
        return {"protocol_name": self.name, "protocol_kind": self.kind,
                "protocol_sha256": self.sha256(),
                "rubric_path": str(self.rubric_path),
                "rubric_sha256": self.rubric_sha256,
                "labels": self.labels,
                "label_mapping_note": ("stored artifacts always use A / B / TIE; the "
                                       "single-token sentinels are an implementation "
                                       "detail of exact candidate scoring"),
                "order_gap_threshold": self.order_gap_threshold,
                "entropy_threshold": self.entropy_threshold,
                "max_templates_when_unstable": self.max_templates_when_unstable,
                "n_templates": len(self.templates),
                "decoding": self.decoding}


def load_protocol(path: Path) -> Protocol:
    cfg = yaml.safe_load(Path(path).read_text())
    rubric = Path(cfg["rubric_file"])
    if not rubric.is_absolute():
        # Walk up from the protocol file until the repo-relative rubric path
        # resolves, rather than hard-coding a directory depth that breaks the
        # moment a protocol is moved.
        here = Path(path).resolve()
        for base in [Path.cwd(), *here.parents]:
            candidate = base / rubric
            if candidate.exists():
                rubric = candidate
                break
        else:
            raise SystemExit(
                f"protocol {path} references rubric {cfg['rubric_file']!r}, which was not "
                f"found from {Path.cwd()} or any parent of the protocol file")
    rubric_hash = hashlib.sha256(rubric.read_bytes()).hexdigest()
    declared = cfg.get("rubric_sha256")
    if declared and declared != rubric_hash:
        raise SystemExit(
            f"protocol declares rubric_sha256 {declared[:16]}… but "
            f"{rubric} hashes to {rubric_hash[:16]}…; the protocol and the rubric it "
            "was calibrated against have diverged")
    return Protocol(
        name=cfg["name"], kind=cfg["kind"], rubric_path=rubric,
        rubric_sha256=rubric_hash, labels=cfg["labels"],
        common_system=cfg["common_system"],
        verdict_instruction=cfg["verdict_instruction"], templates=cfg["templates"],
        order_gap_threshold=float(cfg.get("order_gap_threshold", 0.25)),
        entropy_threshold=float(cfg.get("entropy_threshold", 0.90)),
        max_templates_when_unstable=int(cfg.get("max_templates_when_unstable", 3)),
        max_tokens=int(cfg.get("max_tokens", 1)),
        decoding=cfg.get("decoding", {}))


def verify_label_tokenization(protocol: Protocol, tokenizer, prefix: str) -> dict:
    """Assert every label is exactly one token IN CONTEXT, and say so in the artifact.

    BPE is context dependent, so ``tokenizer("A")`` proves nothing about how the
    label tokenizes after a chat template. The only sound check is that
    ``tokenize(prefix + label)`` equals ``tokenize(prefix)`` plus one id.
    """
    base = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    report = {}
    for choice in CHOICES:
        token = protocol.labels[choice]["token"]
        full = tokenizer(prefix + token, add_special_tokens=False)["input_ids"]
        if full[:len(base)] != base:
            raise SystemExit(f"label {token!r} for {choice} perturbs the prefix "
                             "tokenization; candidate scoring would not be exact")
        appended = full[len(base):]
        if len(appended) != 1:
            raise SystemExit(
                f"label {token!r} for {choice} is {len(appended)} tokens in context "
                f"({appended}); pick a verified single-token sentinel instead")
        declared = protocol.labels[choice]["id"]
        if appended[0] != declared:
            raise SystemExit(f"label {token!r} for {choice} is id {appended[0]} in "
                             f"context but the protocol declares {declared}")
        report[choice] = {"token": token, "id": appended[0]}
    return {"verified_single_token_in_context": True, "labels": report,
            "prefix_token_count": len(base)}


def objective_criterion(rubric_cfg: dict, objective: str) -> str:
    """The objective's criterion paragraph, verbatim from the v2 rubric file.

    Only the criterion is taken. v2's shared preamble ends by demanding
    ``[[A]] / [[B]] / [[TIE]]``, which contradicts v3's single-character verdict
    format, so the preamble comes from the protocol instead. The *semantic
    definitions* of the four objectives are therefore unchanged between v2 and
    v3 -- only how the verdict is expressed changes, which is what makes the two
    protocols comparable at all.
    """
    entry = (rubric_cfg.get("objectives") or {}).get(objective)
    if entry is None:
        raise SystemExit(f"objective {objective!r} is not defined in the rubric file")
    criterion = entry.get("criterion")
    if not criterion:
        raise SystemExit(
            f"objective {objective!r} has no `criterion`; v3 composes its own preamble "
            "and needs the criterion separately, so a v1-style rubric that inlines the "
            "whole system prompt cannot be used here")
    return criterion.strip()


def render(protocol: Protocol, template: dict, criterion: str,
           prompt: str, response_a: str, response_b: str):
    """(system, user) for one (template, presentation order) rendering."""
    system = "\n\n".join(x for x in (
        protocol.common_system.strip(),
        criterion,
        template.get("extra_system", "").strip(),
        protocol.verdict_instruction.strip()) if x)
    user = template["user_template"].format(prompt=prompt, a=response_a, b=response_b)
    return system, user


def build_observation(probs: dict, order: str, template_id: str,
                      raw: Optional[str] = None) -> dict:
    total = sum(probs[c] for c in CHOICES)
    if not (0.999 <= total <= 1.001):
        raise ValueError(f"choice probabilities must be a restricted softmax over "
                         f"{CHOICES}; they sum to {total:.6f}")
    return {"presentation_order": order, "template_id": template_id,
            "probs": {c: float(probs[c]) for c in CHOICES},
            "semantic_score": semantic_score(probs, order),
            "normalized_entropy": normalized_entropy(probs),
            "hard_argmax": max(CHOICES, key=lambda c: probs[c]),
            "raw_judge_output": raw}


def hard_probs_from_verdict(verdict: str) -> dict:
    """P0 as a degenerate case of the same algebra: a one-hot choice vector.

    Expressing the hard protocol this way is what makes P0 and P1 comparable on
    identical metrics -- P0 is simply v3 with all the probability mass forced
    onto one label.
    """
    v = str(verdict).strip().upper().strip("[]")
    if v not in CHOICES:
        raise ValueError(f"unparseable verdict {verdict!r}")
    return {c: (1.0 if c == v else 0.0) for c in CHOICES}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--print-manifest", action="store_true")
    a = ap.parse_args()
    p = load_protocol(a.protocol)
    print(json.dumps(p.manifest(), indent=2) if a.print_manifest else p.sha256())
