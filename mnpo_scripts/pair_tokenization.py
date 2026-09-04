"""Canonical preference-pair tokenization -- ONE implementation for both paths.

Eq. (22) subtracts the proximal centre's sequence log-probability from the
current policy's::

    h_t = (log pi(y|x) - log pi(y'|x)) - (log pi_t(y|x) - log pi_t(y'|x))

Those two terms come from two different code paths -- ``mnpo_scripts.precompute``
scores pi_t offline, ``scripts.simpo_trainer`` scores pi online -- and the
subtraction is only meaningful if both score EXACTLY the same token ids under
exactly the same attention and label masks. They did not:

* precompute tokenized prompt, chosen and rejected **separately**, so any
  tokenizer that merges a token across the prompt/response boundary produced a
  different id sequence than the joint tokenization the trainer uses;
* precompute zeroed the attention mask at every EOS position **inside the
  prompt**, hiding real context tokens from the model -- prompt exclusion
  belongs in ``labels``, not in ``attention_mask``;
* it computed ``new_attention_mask_c`` for the chosen response and then never
  assigned it, so chosen and rejected were treated asymmetrically;
* it appended EOS unconditionally, doubling it when the response already ended
  with one, while the trainer appends only if absent;
* it never added BOS, while the trainer adds it at most once.

Every one of those makes ``h_t(pi_t) != 0`` at initialization for a model that
IS pi_t -- a nonzero regression target before a single gradient step.

This module is the single implementation both paths now call. It follows the
online trainer's semantics, which were the correct ones:

1. joint-tokenize ``prompt + answer`` and slice the answer off, backing the
   boundary up by one token when the tokenizer merged across it;
2. take the prompt length as the MINIMUM over the chosen and rejected
   tokenizations, so the two rows share a prompt prefix;
3. add BOS at most once, EOS at most once;
4. truncate the prompt first (``keep_start`` / ``keep_end``), then the response;
5. keep every valid prompt token in the attention mask;
6. mask prompt positions in ``labels`` with ``label_pad_token_id``;
7. treat chosen and rejected symmetrically.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional

TOKENIZATION_SCHEMA_VERSION = 1
TRUNCATION_MODES = ("keep_start", "keep_end")
BOS_POLICY = "add_if_absent"
EOS_POLICY = "add_if_absent"


def tokenization_config(max_length: int, max_prompt_length: int, truncation_mode: str,
                        label_pad_token_id: int = -100) -> dict:
    """The canonical tokenization settings, in the form that gets hashed."""
    if truncation_mode not in TRUNCATION_MODES:
        raise ValueError(f"truncation_mode must be one of {TRUNCATION_MODES}, "
                         f"got {truncation_mode!r}")
    return {
        "tokenization_schema_version": TOKENIZATION_SCHEMA_VERSION,
        "max_length": int(max_length),
        "max_prompt_length": int(max_prompt_length),
        "truncation_mode": str(truncation_mode),
        "label_pad_token_id": int(label_pad_token_id),
        "bos_policy": BOS_POLICY,
        "eos_policy": EOS_POLICY,
        "add_special_tokens": False,
        "joint_prompt_answer_tokenization": True,
        "boundary_merge_handling": "backoff_one_token",
        "prompt_masked_in": "labels",     # never in attention_mask
    }


def tokenization_config_hash(cfg: dict) -> str:
    """sha256 of the canonical tokenization config (goes into every artifact)."""
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def build_tokenized_answer(tokenizer, prompt: str, answer: str) -> Dict[str, object]:
    """Joint-tokenize ``prompt + answer`` and locate the prompt/answer boundary.

    ``enc(a + b) != enc(a) + enc(b)`` for Llama-family tokenizers, but
    ``enc(a + b) == enc(a) + enc(a + b)[len(enc(a)):]`` holds. When the last
    prompt token merges with the first answer token the prefix differs, and the
    boundary backs up by exactly one token so the pair still concatenates to the
    joint sequence.

    The FULL sequence and the boundary index are returned, not just the split
    halves: the pair builder has to re-split both responses at a COMMON boundary,
    and it can only do that without losing tokens if it still has the full
    sequence to cut.
    """
    full = tokenizer(prompt + answer, add_special_tokens=False)
    full_ids = list(full["input_ids"])
    full_mask = list(full["attention_mask"])
    prompt_input_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])

    if len(full_ids) < len(prompt_input_ids):
        raise ValueError("joint tokenization is shorter than the prompt alone")

    split_idx = len(prompt_input_ids)
    # A merged boundary token makes the joint prefix differ from enc(prompt).
    if prompt_input_ids != full_ids[:split_idx]:
        split_idx -= 1
    if len(full_ids) != len(full_mask):
        raise ValueError("joint input ids and attention mask lengths differ")
    return {
        "full_input_ids": full_ids,
        "full_attention_mask": full_mask,
        "split_idx": split_idx,
        "prompt_input_ids": full_ids[:split_idx],
        "prompt_attention_mask": full_mask[:split_idx],
        "input_ids": full_ids[split_idx:],
        "attention_mask": full_mask[split_idx:],
    }


def split_at(toks: Dict[str, object], split_idx: int) -> Dict[str, object]:
    """Re-cut a joint sequence at ``split_idx``; no token is created or lost.

    Slicing ``prompt_input_ids`` alone would DELETE the tokens it drops. Here a
    token moved off the prompt side lands on the response side, so
    ``prompt_input_ids + input_ids`` still reconstructs the joint tokenization
    exactly.
    """
    full_ids = list(toks["full_input_ids"])
    full_mask = list(toks["full_attention_mask"])
    if not 0 <= split_idx <= len(full_ids):
        raise ValueError(f"split_idx {split_idx} outside the joint sequence "
                         f"of length {len(full_ids)}")
    out = dict(toks)
    out["split_idx"] = split_idx
    out["prompt_input_ids"] = full_ids[:split_idx]
    out["prompt_attention_mask"] = full_mask[:split_idx]
    out["input_ids"] = full_ids[split_idx:]
    out["attention_mask"] = full_mask[split_idx:]
    return out


def _add_bos_if_absent(tokenizer, toks: Dict[str, List[int]], prompt_key="prompt_input_ids",
                       mask_key="prompt_attention_mask") -> None:
    bos = tokenizer.bos_token_id
    if bos is None:
        return
    if len(toks[prompt_key]) == 0 or toks[prompt_key][0] != bos:
        toks[prompt_key] = [bos] + toks[prompt_key]
        toks[mask_key] = [1] + toks[mask_key]


def _add_eos_if_absent(tokenizer, toks: Dict[str, List[int]]) -> None:
    eos = tokenizer.eos_token_id
    if eos is None:
        return
    if len(toks["input_ids"]) == 0 or toks["input_ids"][-1] != eos:
        toks["input_ids"] = toks["input_ids"] + [eos]
        toks["attention_mask"] = toks["attention_mask"] + [1]


def tokenize_prompt_answer(tokenizer, prompt: str, answer: str, max_length: int,
                           max_prompt_length: int, truncation_mode: str = "keep_start",
                           label_pad_token_id: int = -100,
                           split_idx: Optional[int] = None) -> Dict[str, List[int]]:
    """Canonical single ``(prompt, answer)`` tokenization with labels.

    ``split_idx`` re-cuts the JOINT sequence at a caller-chosen boundary (the
    pair builder passes the common boundary of the two responses). It replaces
    the old ``prompt_len_override``, which sliced ``prompt_input_ids`` and threw
    the dropped tokens away instead of moving them to the response side.
    """
    if truncation_mode not in TRUNCATION_MODES:
        raise ValueError(f"Unknown truncation mode: {truncation_mode}")
    toks = build_tokenized_answer(tokenizer, prompt, answer)
    if split_idx is not None:
        toks = split_at(toks, split_idx)
    _add_bos_if_absent(tokenizer, toks)
    _add_eos_if_absent(tokenizer, toks)
    _truncate(toks, len(toks["input_ids"]), max_length, max_prompt_length, truncation_mode)
    return _assemble(toks, label_pad_token_id)


def _truncate(toks: Dict[str, List[int]], longer_response_length: int, max_length: int,
              max_prompt_length: int, truncation_mode: str) -> None:
    """Prompt first, then response -- identical order and bounds in both paths.

    ``longer_response_length`` is the max over BOTH responses, so the prompt is
    truncated to the same length for the chosen and the rejected row. A
    prompt-only dict (no ``input_ids``) gets the prompt step only, which is what
    the online trainer did.
    """
    if len(toks["prompt_input_ids"]) + longer_response_length > max_length:
        if truncation_mode == "keep_start":
            toks["prompt_input_ids"] = toks["prompt_input_ids"][:max_prompt_length]
            toks["prompt_attention_mask"] = toks["prompt_attention_mask"][:max_prompt_length]
        else:  # keep_end
            toks["prompt_input_ids"] = toks["prompt_input_ids"][-max_prompt_length:]
            toks["prompt_attention_mask"] = toks["prompt_attention_mask"][-max_prompt_length:]
    if "input_ids" not in toks:
        return
    if len(toks["prompt_input_ids"]) + longer_response_length > max_length:
        keep = max_length - max_prompt_length
        toks["input_ids"] = toks["input_ids"][:keep]
        toks["attention_mask"] = toks["attention_mask"][:keep]


def _assemble(toks: Dict[str, List[int]], label_pad_token_id: int) -> Dict[str, List[int]]:
    """Concatenate prompt+answer and mask the prompt in LABELS only."""
    n_prompt = len(toks["prompt_input_ids"])
    input_ids = toks["prompt_input_ids"] + toks["input_ids"]
    attention_mask = toks["prompt_attention_mask"] + toks["attention_mask"]
    labels = list(input_ids)
    labels[:n_prompt] = [label_pad_token_id] * n_prompt
    return {
        "prompt_input_ids": list(toks["prompt_input_ids"]),
        "prompt_attention_mask": list(toks["prompt_attention_mask"]),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def tokenize_preference_pair(tokenizer, prompt: str, chosen: str, rejected: str,
                             max_length: int, max_prompt_length: int,
                             truncation_mode: str = "keep_start",
                             label_pad_token_id: int = -100) -> Dict[str, object]:
    """Canonical preference-pair tokenization -- the ONE implementation.

    Called by both ``PreferenceDataCollatorWithPadding.tokenize_batch_element``
    (precompute) and ``SimPOTrainer.tokenize_row`` (training). Chosen and
    rejected are handled by identical code, so no asymmetry can creep back in.
    """
    for name, value in (("prompt", prompt), ("chosen", chosen), ("rejected", rejected)):
        if not isinstance(value, str):
            raise ValueError(f"{name} should be a str but got {type(value)}")
    if truncation_mode not in TRUNCATION_MODES:
        raise ValueError(f"Unknown truncation mode: {truncation_mode}")

    chosen_tokens = build_tokenized_answer(tokenizer, prompt, chosen)
    rejected_tokens = build_tokenized_answer(tokenizer, prompt, rejected)

    # A merge can fire for only ONE of the two responses -- enc("AB") merges
    # while enc("AC") does not -- and then the two rows were conditioned on
    # different prompt prefixes, so they were not log pi(y|x) and log pi(y'|x)
    # for one common x. Both JOINT sequences are re-cut at the common boundary;
    # a token that leaves the prompt side moves to the response side rather than
    # being deleted, so each row still reconstructs its own joint tokenization.
    common_split = min(chosen_tokens["split_idx"], rejected_tokens["split_idx"])
    chosen_tokens = split_at(chosen_tokens, common_split)
    rejected_tokens = split_at(rejected_tokens, common_split)
    if chosen_tokens["prompt_input_ids"] != rejected_tokens["prompt_input_ids"]:
        raise ValueError(
            "chosen and rejected do not share a prompt prefix after re-splitting at the "
            f"common boundary {common_split}: "
            f"{chosen_tokens['prompt_input_ids']} vs {rejected_tokens['prompt_input_ids']}. "
            "The two responses would be scored under different conditioning contexts."
        )
    prompt_tokens = {"prompt_input_ids": list(chosen_tokens["prompt_input_ids"]),
                     "prompt_attention_mask": list(chosen_tokens["prompt_attention_mask"])}

    for toks in (prompt_tokens, chosen_tokens, rejected_tokens):
        _add_bos_if_absent(tokenizer, toks)
    for toks in (chosen_tokens, rejected_tokens):
        _add_eos_if_absent(tokenizer, toks)

    longer_response_length = max(len(chosen_tokens["input_ids"]),
                                 len(rejected_tokens["input_ids"]))
    for toks in (chosen_tokens, rejected_tokens, prompt_tokens):
        _truncate(toks, longer_response_length, max_length, max_prompt_length, truncation_mode)

    chosen_seq = _assemble(chosen_tokens, label_pad_token_id)
    rejected_seq = _assemble(rejected_tokens, label_pad_token_id)
    # Independent of the merge logic above: after BOS/EOS and truncation the two
    # rows must still be conditioned on one identical prompt prefix, or the pair
    # does not represent log pi(y|x) and log pi(y'|x) for a single x.
    n_c, n_r = len(chosen_tokens["prompt_input_ids"]), len(rejected_tokens["prompt_input_ids"])
    if chosen_seq["input_ids"][:n_c] != rejected_seq["input_ids"][:n_r]:
        raise ValueError(
            "chosen and rejected prompt prefixes differ after assembly: "
            f"{chosen_seq['input_ids'][:n_c]} vs {rejected_seq['input_ids'][:n_r]}")

    batch: Dict[str, object] = {}
    for key, seq in (("chosen", chosen_seq), ("rejected", rejected_seq)):
        batch[f"{key}_input_ids"] = seq["input_ids"]
        batch[f"{key}_attention_mask"] = seq["attention_mask"]
        batch[f"{key}_labels"] = seq["labels"]
    batch["prompt_input_ids"] = prompt_tokens["prompt_input_ids"]
    batch["prompt_attention_mask"] = prompt_tokens["prompt_attention_mask"]
    batch["prompt"] = prompt
    batch["chosen"] = prompt + chosen
    batch["rejected"] = prompt + rejected
    batch["chosen_response_only"] = chosen
    batch["rejected_response_only"] = rejected
    return batch
