"""Canonical-tokenization tests (audit P0).

Eq. (22) subtracts the proximal centre's log-probability from the current
policy's. The two come from different code paths -- offline
``mnpo_scripts.precompute`` and online ``scripts.simpo_trainer`` -- so unless
both score the SAME token ids under the SAME attention and label masks, the
regression target is nonzero before a single gradient step. Every test here
compares the two paths' actual outputs rather than reading the code.

The fake tokenizer deliberately MERGES a token across the prompt/response
boundary, so any implementation that tokenizes prompt and response separately
fails these tests.
"""
import json
from types import SimpleNamespace

import pytest
import torch

from mnpo_scripts.pair_tokenization import (
    TOKENIZATION_SCHEMA_VERSION,
    build_tokenized_answer,
    tokenization_config,
    tokenization_config_hash,
    tokenize_preference_pair,
    tokenize_prompt_answer,
)

BOS, EOS, PAD = 1, 2, 0
MERGE_ID = 999          # id produced only when "A" and "B" are adjacent


class MergingTokenizer:
    """Character-level fake tokenizer that merges "AB" into one token.

    ``enc("...A") + enc("B...") != enc("...AB")`` -- exactly the Llama-family
    behaviour that separate prompt/response tokenization gets wrong.
    """

    bos_token_id, eos_token_id, pad_token_id = BOS, EOS, PAD

    def __init__(self):
        self.vocab = {c: 10 + i for i, c in enumerate(
            "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ.?!")}

    def _ids(self, text):
        ids, i = [], 0
        while i < len(text):
            if text[i:i + 2] == "AB":
                ids.append(MERGE_ID)          # the merge
                i += 2
            else:
                ids.append(self.vocab.get(text[i], 55))
                i += 1
        return ids

    def __call__(self, text, add_special_tokens=False):
        ids = self._ids(text)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class PlainTokenizer(MergingTokenizer):
    """Same vocabulary, no merging -- the easy case."""

    def _ids(self, text):
        return [self.vocab.get(c, 55) for c in text]


TOK_KW = dict(max_length=64, max_prompt_length=32, truncation_mode="keep_end",
              label_pad_token_id=-100)


# --------------------------------------------------------------------------- #
# the two production paths agree, token for token
# --------------------------------------------------------------------------- #
def _precompute_row(tokenizer, prompt, chosen, rejected, **kw):
    """What mnpo_scripts.precompute's collator produces."""
    from mnpo_scripts.precompute_trainer import PreferenceDataCollatorWithPadding

    coll = PreferenceDataCollatorWithPadding(
        tokenizer=tokenizer, max_length=kw["max_length"],
        max_prompt_length=kw["max_prompt_length"], label_pad_token_id=kw["label_pad_token_id"],
        padding_value=0, truncation_mode=kw["truncation_mode"], is_encoder_decoder=False,
        max_target_length=None)
    return coll.tokenize_batch_element(prompt, chosen, rejected)


def _training_row(tokenizer, prompt, chosen, rejected, **kw):
    """What scripts.simpo_trainer's tokenize_row produces."""
    from scripts.simpo_trainer import SimPOTrainer

    trainer = SimPOTrainer.__new__(SimPOTrainer)
    trainer.tokenizer = tokenizer
    trainer.is_encoder_decoder = False
    trainer.max_length = kw["max_length"]
    trainer.max_prompt_length = kw["max_prompt_length"]
    trainer.truncation_mode = kw["truncation_mode"]
    trainer.label_pad_token_id = kw["label_pad_token_id"]
    return trainer.tokenize_row({"prompt": prompt, "chosen": chosen, "rejected": rejected})


CASES = [
    ("plain", PlainTokenizer, "what is a fact?", " a true thing.", " a false thing."),
    # the boundary merge: prompt ends in A, responses start with B
    ("boundary merge", MergingTokenizer, "answer with A", "B is the answer.", "B is wrong."),
    ("prompt with internal eos", MergingTokenizer, "one. two. three?", " reply here.", " other."),
    ("response already ends with eos", MergingTokenizer, "a question?", " ends here.", " x."),
]


@pytest.mark.parametrize("label,tok_cls,prompt,chosen,rejected", CASES)
def test_precompute_and_training_token_ids_identical(label, tok_cls, prompt, chosen, rejected):
    tok = tok_cls()
    pre = _precompute_row(tok, prompt, chosen, rejected, **TOK_KW)
    tra = _training_row(tok, prompt, chosen, rejected, **TOK_KW)
    for key in ("chosen_input_ids", "rejected_input_ids", "prompt_input_ids"):
        assert pre[key] == tra[key], f"{label}: {key} differs\npre={pre[key]}\ntra={tra[key]}"


@pytest.mark.parametrize("label,tok_cls,prompt,chosen,rejected", CASES)
def test_precompute_and_training_attention_masks_identical(label, tok_cls, prompt, chosen,
                                                           rejected):
    tok = tok_cls()
    pre = _precompute_row(tok, prompt, chosen, rejected, **TOK_KW)
    tra = _training_row(tok, prompt, chosen, rejected, **TOK_KW)
    for key in ("chosen_attention_mask", "rejected_attention_mask", "prompt_attention_mask"):
        assert pre[key] == tra[key], f"{label}: {key} differs"
    # and the prompt is NEVER hidden from the model: exclusion lives in labels
    assert all(m == 1 for m in pre["chosen_attention_mask"]), \
        "no prompt/EOS position may be zeroed in the attention mask"
    assert all(m == 1 for m in pre["rejected_attention_mask"])


@pytest.mark.parametrize("label,tok_cls,prompt,chosen,rejected", CASES)
def test_precompute_and_training_labels_identical(label, tok_cls, prompt, chosen, rejected):
    tok = tok_cls()
    pre = _precompute_row(tok, prompt, chosen, rejected, **TOK_KW)
    tra = _training_row(tok, prompt, chosen, rejected, **TOK_KW)
    for key in ("chosen_labels", "rejected_labels"):
        assert pre[key] == tra[key], f"{label}: {key} differs"
    n_prompt = len(pre["prompt_input_ids"])
    assert pre["chosen_labels"][:n_prompt] == [-100] * n_prompt
    assert pre["chosen_labels"][n_prompt:] == pre["chosen_input_ids"][n_prompt:]


# --------------------------------------------------------------------------- #
# specific tokenizer hazards
# --------------------------------------------------------------------------- #
def test_prompt_response_boundary_merge_tokenizer():
    """The merge must actually happen, and be handled by backing the boundary up."""
    tok = MergingTokenizer()
    prompt, answer = "answer with A", "B is the answer."
    assert MERGE_ID in tok(prompt + answer)["input_ids"], "fixture must exercise a merge"
    assert MERGE_ID not in tok(prompt)["input_ids"]
    out = build_tokenized_answer(tok, prompt, answer)
    # prompt + answer reconstructs the joint tokenization exactly
    assert out["prompt_input_ids"] + out["input_ids"] == tok(prompt + answer)["input_ids"]
    # the boundary backed up: the prompt is one token shorter than enc(prompt)
    assert len(out["prompt_input_ids"]) == len(tok(prompt)["input_ids"]) - 1
    assert out["input_ids"][0] == MERGE_ID
    # a naive separate tokenization would have produced a different sequence
    naive = tok(prompt)["input_ids"] + tok(answer)["input_ids"]
    assert naive != tok(prompt + answer)["input_ids"]


def test_prompt_contains_internal_eos():
    """An EOS-valued token inside the prompt stays visible in the attention mask."""
    class EosInPromptTokenizer(PlainTokenizer):
        def _ids(self, text):
            return [EOS if c == "." else self.vocab.get(c, 55) for c in text]

    tok = EosInPromptTokenizer()
    prompt = "a. b. c"
    pre = _precompute_row(tok, prompt, " x", " y", **TOK_KW)
    tra = _training_row(tok, prompt, " x", " y", **TOK_KW)
    n_prompt = len(pre["prompt_input_ids"])
    assert EOS in pre["chosen_input_ids"][:n_prompt], "fixture must put EOS inside the prompt"
    assert all(m == 1 for m in pre["chosen_attention_mask"][:n_prompt]), \
        "internal EOS must NOT be removed from the attention mask"
    assert pre["chosen_attention_mask"] == tra["chosen_attention_mask"]


def test_response_already_ends_with_eos():
    """EOS is appended only when absent -- never doubled."""
    class EosEndTokenizer(PlainTokenizer):
        def _ids(self, text):
            return [EOS if c == "!" else self.vocab.get(c, 55) for c in text]

    tok = EosEndTokenizer()
    with_eos = _precompute_row(tok, "q", " ends!", " also!", **TOK_KW)
    assert with_eos["chosen_input_ids"][-1] == EOS
    assert with_eos["chosen_input_ids"][-2] != EOS, "EOS must not be doubled"
    without = _precompute_row(tok, "q", " ends", " also", **TOK_KW)
    assert without["chosen_input_ids"][-1] == EOS, "EOS must be added when absent"
    assert without["chosen_input_ids"].count(EOS) == 1


def test_chosen_rejected_tokenization_is_symmetric():
    """Swapping the two responses swaps the two outputs exactly."""
    tok = MergingTokenizer()
    a, b = "B first response.", "B second one."
    row = _precompute_row(tok, "prompt with A", a, b, **TOK_KW)
    swapped = _precompute_row(tok, "prompt with A", b, a, **TOK_KW)
    for key in ("input_ids", "attention_mask", "labels"):
        assert row[f"chosen_{key}"] == swapped[f"rejected_{key}"], key
        assert row[f"rejected_{key}"] == swapped[f"chosen_{key}"], key


def test_bos_and_eos_added_at_most_once():
    tok = PlainTokenizer()
    row = _precompute_row(tok, "hello", " world", " other", **TOK_KW)
    for key in ("chosen_input_ids", "rejected_input_ids"):
        assert row[key][0] == BOS
        assert row[key].count(BOS) == 1, f"{key}: BOS added more than once"
        assert row[key][-1] == EOS
        assert row[key].count(EOS) == 1, f"{key}: EOS added more than once"

    class BosTokenizer(PlainTokenizer):
        def _ids(self, text):
            return [BOS] + [self.vocab.get(c, 55) for c in text]

    tok2 = BosTokenizer()
    row2 = _precompute_row(tok2, "hello", " world", " other", **TOK_KW)
    assert row2["chosen_input_ids"].count(BOS) == 1, "existing BOS must not be duplicated"


@pytest.mark.parametrize("truncation_mode", ["keep_start", "keep_end"])
def test_truncation_identical_in_both_paths(truncation_mode):
    tok = MergingTokenizer()
    kw = dict(TOK_KW, max_length=24, max_prompt_length=12, truncation_mode=truncation_mode)
    prompt = "a very long prompt indeed that will need truncating A"
    pre = _precompute_row(tok, prompt, "B long chosen response here", "B long rejected one", **kw)
    tra = _training_row(tok, prompt, "B long chosen response here", "B long rejected one", **kw)
    for key in ("chosen_input_ids", "rejected_input_ids", "chosen_attention_mask",
                "rejected_attention_mask", "chosen_labels", "rejected_labels"):
        assert pre[key] == tra[key], f"{truncation_mode}: {key} differs"
    assert len(pre["chosen_input_ids"]) <= kw["max_length"]


def test_unknown_truncation_mode_raises():
    tok = PlainTokenizer()
    with pytest.raises(ValueError, match="truncation mode"):
        tokenize_preference_pair(tok, "p", "a", "b", max_length=64, max_prompt_length=32,
                                 truncation_mode="keep_middle")


def test_tokenize_prompt_answer_matches_the_pair_builder():
    tok = MergingTokenizer()
    pair = tokenize_preference_pair(tok, "prompt A", "B chosen", "B chosen", **TOK_KW)
    single = tokenize_prompt_answer(tok, "prompt A", "B chosen", **TOK_KW)
    assert single["input_ids"] == pair["chosen_input_ids"]
    assert single["labels"] == pair["chosen_labels"]


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def test_wrong_tokenization_schema_fails():
    from mnpo_scripts.mnpo_trainer import validate_nbpo_args

    cfg = tokenization_config(max_length=2048, max_prompt_length=1024,
                              truncation_mode="keep_end")
    good_meta = {"logp_reduction": "sum", **cfg,
                 "tokenization_config_sha256": tokenization_config_hash(cfg)}
    cols = ["nbpo_weighted_z", "history0_chosen_logps", "history0_rejected_logps"]

    def args(**over):
        base = dict(loss_type="nbpo", reference_anchor_weight=0.0, preference_sft_weight=0.0,
                    logp_reduction="sum", max_history_t=1, history_weights=[1.0],
                    nbpo_target_column="nbpo_weighted_z",
                    nbpo_expected_pair_artifact_sha256=None,
                    nbpo_expected_solver_artifact_sha256=None,
                    nbpo_expected_parent_checkpoint_fingerprint=None,
                    nbpo_expected_precompute_manifest_sha256=None,
                    nbpo_expected_tokenization_config_sha256=None,
                    max_length=2048, max_prompt_length=1024, truncation_mode="keep_end")
        base.update(over)
        return SimpleNamespace(**base)

    validate_nbpo_args(args(), dataset_columns=cols, precompute_meta=good_meta)   # baseline

    with pytest.raises(ValueError, match="records no tokenization_config_sha256"):
        validate_nbpo_args(args(), dataset_columns=cols,
                           precompute_meta={"logp_reduction": "sum"})
    with pytest.raises(ValueError, match="tokenization schema mismatch"):
        validate_nbpo_args(args(nbpo_expected_tokenization_config_sha256="f" * 64),
                           dataset_columns=cols, precompute_meta=good_meta)
    with pytest.raises(ValueError, match="tokenization schema v99"):
        validate_nbpo_args(args(), dataset_columns=cols,
                           precompute_meta={**good_meta, "tokenization_schema_version": 99})
    with pytest.raises(ValueError, match="truncate differently"):
        validate_nbpo_args(args(truncation_mode="keep_start"), dataset_columns=cols,
                           precompute_meta=good_meta)
    with pytest.raises(ValueError, match="truncate differently"):
        validate_nbpo_args(args(max_length=512), dataset_columns=cols,
                           precompute_meta=good_meta)


def test_tokenization_config_hash_is_sensitive_to_every_setting():
    base = tokenization_config(2048, 1024, "keep_end")
    h = tokenization_config_hash(base)
    assert h != tokenization_config_hash(tokenization_config(2048, 1024, "keep_start"))
    assert h != tokenization_config_hash(tokenization_config(1024, 1024, "keep_end"))
    assert h != tokenization_config_hash(tokenization_config(2048, 512, "keep_end"))
    assert base["tokenization_schema_version"] == TOKENIZATION_SCHEMA_VERSION
    assert base["prompt_masked_in"] == "labels"


def test_mask_prompt_is_refused_rather_than_silently_ignored():
    from mnpo_scripts.precompute_trainer import PreferenceDataCollatorWithPadding

    coll = PreferenceDataCollatorWithPadding(
        tokenizer=PlainTokenizer(), max_length=64, max_prompt_length=32,
        label_pad_token_id=-100, padding_value=0, truncation_mode="keep_end",
        is_encoder_decoder=False, max_target_length=None, mask_prompt=True)
    with pytest.raises(ValueError, match="belongs in labels"):
        coll.tokenize_batch_element("p", "a", "b")


# --------------------------------------------------------------------------- #
# the end-to-end invariant: h_t(pi_t) == 0 before any update
# --------------------------------------------------------------------------- #
def _tiny_model_and_tokenizer(tmp_path):
    """A small frozen causal LM whose tokenizer MERGES across the boundary.

    The merging tokenizer is the point: with separate prompt/response
    tokenization the two paths score different token ids, so the logps differ
    and h_t != 0 even though the model is literally pi_t.
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=1000, n_positions=128, n_embd=32, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, MergingTokenizer()


def _sequence_logps(model, input_ids, attention_mask, labels, average: bool):
    """Sequence log-probability with the production reduction semantics."""
    from scripts.simpo_trainer import SimPOTrainer

    ids = torch.tensor([input_ids], dtype=torch.long)
    mask = torch.tensor([attention_mask], dtype=torch.long)
    labs = torch.tensor([labels], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask).logits
    return SimPOTrainer.get_batch_logps(logits, labs, average_log_prob=average,
                                        label_pad_token_id=-100)


def test_h_t_of_pi_t_is_zero_before_any_optimizer_step(tmp_path):
    """The strongest invariant: with pi == pi_t, Eq. (22) must give exactly h_t = 0.

    ``history0`` logps come from the PRECOMPUTE collator and the current-policy
    logps from the TRAINER's tokenization, with one frozen model playing both
    roles. Token ids, attention masks and labels must be exactly equal; the
    log-probabilities then agree to floating-point tolerance and h_t vanishes.
    """
    model, tok = _tiny_model_and_tokenizer(tmp_path)
    prompt, chosen, rejected = "explain topic A", "B chosen answer here.", "B rejected answer."

    pre = _precompute_row(tok, prompt, chosen, rejected, **TOK_KW)
    tra = _training_row(tok, prompt, chosen, rejected, **TOK_KW)

    # exact equality of what gets scored -- not a tolerance
    for key in ("chosen_input_ids", "rejected_input_ids", "chosen_attention_mask",
                "rejected_attention_mask", "chosen_labels", "rejected_labels"):
        assert pre[key] == tra[key], f"{key} differs between the two production paths"
    assert MERGE_ID in pre["chosen_input_ids"], "the fixture must exercise a boundary merge"

    # sequence-sum reduction, as loss_type=nbpo requires
    hist_chosen = _sequence_logps(model, pre["chosen_input_ids"], pre["chosen_attention_mask"],
                                  pre["chosen_labels"], average=False)
    hist_rejected = _sequence_logps(model, pre["rejected_input_ids"],
                                    pre["rejected_attention_mask"], pre["rejected_labels"],
                                    average=False)
    cur_chosen = _sequence_logps(model, tra["chosen_input_ids"], tra["chosen_attention_mask"],
                                 tra["chosen_labels"], average=False)
    cur_rejected = _sequence_logps(model, tra["rejected_input_ids"],
                                   tra["rejected_attention_mask"], tra["rejected_labels"],
                                   average=False)

    assert torch.allclose(cur_chosen, hist_chosen, atol=1e-6), \
        f"current_chosen_logps {cur_chosen} != history0_chosen_logps {hist_chosen}"
    assert torch.allclose(cur_rejected, hist_rejected, atol=1e-6), \
        f"current_rejected_logps {cur_rejected} != history0_rejected_logps {hist_rejected}"

    # Eq. (22): h_t = (log pi(y) - log pi(y')) - (log pi_t(y) - log pi_t(y'))
    h_t = (cur_chosen - cur_rejected) - (hist_chosen - hist_rejected)
    assert torch.allclose(h_t, torch.zeros_like(h_t), atol=1e-6), f"h_t(pi_t) = {h_t}"
    print(f"\n[h_t invariant] h_t(pi_t) = {float(h_t[0]):.3e} "
          f"(chosen {float(cur_chosen[0]):.6f} vs {float(hist_chosen[0]):.6f}, "
          f"rejected {float(cur_rejected[0]):.6f} vs {float(hist_rejected[0]):.6f})")


def test_separate_tokenization_would_break_the_invariant(tmp_path):
    """Guard the guard: the OLD separate-tokenization scheme gives h_t != 0.

    Without this, the invariant test above could pass for a trivial reason (a
    tokenizer that never merges) and stop protecting anything.
    """
    model, tok = _tiny_model_and_tokenizer(tmp_path)
    prompt, chosen, rejected = "explain topic A", "B chosen answer here.", "B rejected answer."

    def legacy_row(answer):
        """What the pre-fix precompute did: tokenize the parts separately."""
        p_ids = tok(prompt)["input_ids"]
        a_ids = tok(answer)["input_ids"] + [EOS]
        ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + a_ids
        return ids, [1] * len(ids), labels

    lc_ids, lc_mask, lc_lab = legacy_row(chosen)
    lr_ids, lr_mask, lr_lab = legacy_row(rejected)
    tra = _training_row(tok, prompt, chosen, rejected, **TOK_KW)
    assert lc_ids != tra["chosen_input_ids"], \
        "the fixture must make separate tokenization differ from joint tokenization"

    legacy_c = _sequence_logps(model, lc_ids, lc_mask, lc_lab, average=False)
    legacy_r = _sequence_logps(model, lr_ids, lr_mask, lr_lab, average=False)
    cur_c = _sequence_logps(model, tra["chosen_input_ids"], tra["chosen_attention_mask"],
                            tra["chosen_labels"], average=False)
    cur_r = _sequence_logps(model, tra["rejected_input_ids"], tra["rejected_attention_mask"],
                            tra["rejected_labels"], average=False)
    h_t_legacy = (cur_c - cur_r) - (legacy_c - legacy_r)
    assert not torch.allclose(h_t_legacy, torch.zeros_like(h_t_legacy), atol=1e-6), \
        "separate tokenization must NOT accidentally satisfy the invariant"
