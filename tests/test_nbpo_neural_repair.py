import copy
import itertools
import math
from types import SimpleNamespace

import pytest
import torch

from mnpo_scripts.nbpo_neural import (
    nbpo_wbc_pair_loss, select_training_splits, validate_canonical_pair_dataset,
)
from mnpo_scripts.pair_tokenization import pair_from_candidate_events, immutable_pair_tokens
from mnpo_scripts.response_logps import response_logps
from mnpo_scripts.mnpo_trainer import MNPOTrainer
from scripts.simpo_trainer import SimPOTrainer


def pair_rows():
    masses = torch.arange(1, 9, dtype=torch.float64).div(36).tolist()
    candidates = [{"response_token_ids": [10 + i, 2]} for i in range(8)]
    rows = []
    for a, b in itertools.combinations(range(8), 2):
        rows.append({
            **pair_from_candidate_events([1, 3, 4], candidates[a], candidates[b], 16, 8),
            "prompt_id": "p", "prompt": "unused", "chosen": "unused", "rejected": "unused",
            "chosen_response_id": str(a), "rejected_response_id": str(b),
            "target_mode": "canonical_logratio", "target_units": "final_logratio_change",
            "eta_already_included": True, "nbpo_num_candidates": 8,
            "nbpo_weight_a": masses[a], "nbpo_weight_b": masses[b],
            "nbpo_logratio_target": math.log(masses[a]) - math.log(masses[b]),
            "solver_artifact_sha256": "a" * 64,
        })
    return rows


def test_long_bf16_response_logp_change_survives():
    logits = torch.zeros(2, 1025, 2, dtype=torch.bfloat16)
    logits[1, 500, 0] = 1
    labels = torch.zeros(2, 1025, dtype=torch.long)
    actual = SimPOTrainer.get_batch_logps(logits, labels, average_log_prob=False)
    expected = torch.nn.functional.log_softmax(logits.float()[:, :-1], dim=-1)[:, :, 0].contiguous().sum(-1)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)
    assert 0.37 < float(actual[1] - actual[0]) < 0.39
    old = logits[:, :-1].log_softmax(-1)[:, :, 0].sum(-1)
    assert old[1] == old[0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunked_backward_matches_dense_full_vocabulary(dtype):
    torch.manual_seed(19)
    a = torch.randn(3, 137, 31, dtype=dtype).requires_grad_()
    b = a.detach().clone().requires_grad_()
    labels = torch.randint(0, 31, (3, 137))
    labels[:, :8] = -100
    labels[1, 70:] = -100
    upstream = torch.tensor([.2, -1.3, .8])
    output = response_logps(a, labels, chunk_size=19)
    safe = labels[:, 1:].clamp_min(0)
    reference = torch.nn.functional.log_softmax(b[:, :-1], dim=-1, dtype=torch.float32)
    reference = reference.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    reference = reference.masked_fill(labels[:, 1:] == -100, 0).sum(-1)
    (output * upstream).sum().backward()
    (reference * upstream).sum().backward()
    torch.testing.assert_close(output, reference, rtol=0, atol=1e-5)
    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=3e-7 if dtype == torch.float32 else .008)


@pytest.mark.parametrize("eta", [.5, 2.0])
def test_production_canonical_mse_does_not_apply_eta_twice(eta):
    fake = SimpleNamespace(accelerator=SimpleNamespace(device=torch.device("cpu")),
                           loss_type="nbpo", eta=eta, beta=1,
                           nbpo_target_mode="canonical_logratio",
                           nbpo_target_column="nbpo_logratio_target")
    p, q, old = torch.tensor([-4.]), torch.tensor([-5.]), torch.tensor([-6.])
    target = torch.tensor([.7])
    loss, _, _ = MNPOTrainer.mnpo_loss(fake, p, q, old, old, [(old, old)], nbpo_target=target)
    torch.testing.assert_close(loss, torch.tensor([.09]), rtol=0, atol=1e-7)


def test_wbc_all_pair_value_and_gradient_matches_direct_empirical_teacher():
    torch.manual_seed(41)
    parameter = torch.randn(8, 13, requires_grad=True)
    direct_parameter = parameter.detach().clone().requires_grad_()
    target = torch.softmax(torch.randn(8), 0).requires_grad_()
    a, b = torch.tensor(list(itertools.combinations(range(8), 2))).T
    logps = parameter.log_softmax(-1)[:, 0]
    direct_logps = direct_parameter.log_softmax(-1)[:, 0]
    loss = nbpo_wbc_pair_loss(logps[a], logps[b], target[a], target[b], 8).mean()
    direct = -(target.detach() * direct_logps).sum()
    loss.backward()
    direct.backward()
    torch.testing.assert_close(loss, direct, rtol=0, atol=3e-7)
    torch.testing.assert_close(parameter.grad, direct_parameter.grad, rtol=0, atol=5e-8)
    assert target.grad is None


def test_neutral_finite_wbc_has_empirical_not_false_zero_gradient():
    # Seven occurrences of action0 and one action1; base mass(.7,.3).
    theta = torch.tensor([math.log(.7), math.log(.3)], requires_grad=True)
    actions = torch.tensor([0] * 7 + [1])
    logps = theta.log_softmax(0)[actions]
    a, b = torch.tensor(list(itertools.combinations(range(8), 2))).T
    weights = torch.full((28,), 1 / 8)
    nbpo_wbc_pair_loss(logps[a], logps[b], weights, weights).mean().backward()
    torch.testing.assert_close(theta.grad, torch.tensor([-.175, .175]), rtol=0, atol=3e-8)


def test_immutable_candidate_is_identical_across_partners_and_no_eos_is_invented():
    prompt = [1, 3, 4]
    a, b, c = ({"response_token_ids": x} for x in ([5, 6], [7], [8, 9, 10, 11, 12]))
    ab = pair_from_candidate_events(prompt, a, b, 10, 4)
    ac = pair_from_candidate_events(prompt, a, c, 10, 4)
    for key in ("chosen_input_ids", "chosen_attention_mask", "chosen_labels", "chosen_token_sha256"):
        assert ab[key] == ac[key]
    assert ab["chosen_input_ids"][-1] == 6
    assert immutable_pair_tokens(ab, 10, 4) == ab
    fake = SimpleNamespace(is_encoder_decoder=False, max_length=10, max_prompt_length=4,
                           label_pad_token_id=-100)
    assert SimPOTrainer.tokenize_row(fake, ab) == ab  # no tokenizer needed


def test_precompute_collator_consumes_same_immutable_tokens():
    from mnpo_scripts.precompute_trainer import PreferenceDataCollatorWithPadding
    tokenizer = SimpleNamespace(pad_token_id=0)
    collator = PreferenceDataCollatorWithPadding(tokenizer, max_length=16, max_prompt_length=8)
    row = pair_rows()[0]
    batch = collator([row])
    for key in ("chosen_input_ids", "chosen_attention_mask", "chosen_labels"):
        assert batch[key][0].tolist() == row[key]


def test_explicit_split_sentinels_never_select_test():
    source = {"test": ["test-sentinel"], "dev": ["dev-sentinel"], "train": ["train-sentinel"]}
    assert select_training_splits(source) == (["train-sentinel"], ["dev-sentinel"])
    with pytest.raises(ValueError, match="separate"):
        select_training_splits(source, eval_split="test")
    with pytest.raises(ValueError, match="eval_split"):
        select_training_splits({"train": [], "test": []})


def test_dataset_contract_checks_pairs_masses_and_canonical_target():
    rows = pair_rows()
    assert validate_canonical_pair_dataset(rows, 16, 8)["rows"] == 28
    with pytest.raises(ValueError, match="28"):
        validate_canonical_pair_dataset(rows[:-1], 16, 8)
    changed = copy.deepcopy(rows)
    changed[0]["nbpo_logratio_target"] *= -1
    with pytest.raises(ValueError, match="disagrees"):
        validate_canonical_pair_dataset(changed, 16, 8)


@pytest.mark.parametrize("loss_type", ["nbpo_wbc", "nbpo"])
def test_actual_tiny_trainer_forward_and_reference_initialization(tmp_path, loss_type):
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from mnpo_scripts.mnpo_config import MNPOConfig

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {str(i): i for i in range(32)}, unk_token="0")), pad_token="0", bos_token="1", eos_token="2")
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=1, num_attention_heads=2,
                                        num_key_value_heads=2, attention_dropout=0))
    args = MNPOConfig(output_dir=str(tmp_path), use_cpu=True, report_to="none",
                      loss_type=loss_type, nbpo_target_mode="canonical_logratio",
                      nbpo_target_column="nbpo_logratio_target", nbpo_target_units="final_logratio_change",
                      nbpo_eta_already_included=True, nbpo_require_immutable_tokens=True,
                      max_length=16, max_prompt_length=8, logp_reduction="sum",
                      max_history_t=1, history_weights=[1.], remove_unused_columns=False,
                      per_device_train_batch_size=2, max_steps=1, save_strategy="no",
                      nbpo_online_reference=loss_type == "nbpo",
                      nbpo_verify_reference_initialization=loss_type == "nbpo")
    trainer = MNPOTrainer(model=model, args=args, train_dataset=Dataset.from_list(pair_rows()), tokenizer=tokenizer)
    if loss_type == "nbpo":
        trainer.nbpo_reference_model = copy.deepcopy(model).eval().requires_grad_(False)
    batch = next(iter(trainer.get_train_dataloader()))
    loss, metrics = trainer.get_batch_loss_metrics(model, batch)
    assert torch.isfinite(loss) and loss.dtype == torch.float32
    if loss_type == "nbpo_wbc":
        assert "drift/pair_logratio_to_reference_abs" not in metrics
        assert trainer.nbpo_reference_model is None
    else:
        assert trainer._reference_init_report["sequence_max_abs"] == 0
        assert trainer._reference_init_report["pair_h_rms"] == 0
    loss.backward()
    assert model.lm_head.weight.grad.abs().sum() > 0
    if loss_type == "nbpo_wbc":
        args.nbpo_eval_online_reference = True
        trainer.nbpo_reference_model = copy.deepcopy(model).eval().requires_grad_(False)
    evaluation = trainer.evaluate(eval_dataset=trainer.train_dataset)
    assert evaluation["eval_nbpo/pair_rows"] == 28
    assert evaluation["eval_nbpo/nmse"] == pytest.approx(1.0, abs=1e-6)
    trainer.train()
    assert trainer.state.global_step == 1


def test_zero_bfloat16_rotary_cast_restores_exact_pretrained_frequencies():
    from transformers import LlamaConfig, LlamaForCausalLM
    from mnpo_scripts.nbpo_runtime import FP32RotaryBufferGuard

    torch.manual_seed(17)
    config = LlamaConfig(vocab_size=31, hidden_size=64, intermediate_size=96,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=2, max_position_embeddings=256,
                         rope_theta=500000.0, attention_dropout=0.0)
    model = LlamaForCausalLM._from_config(config, torch_dtype=torch.bfloat16).eval()
    guard = FP32RotaryBufferGuard(model)
    assert guard.snapshots and all(module.inv_freq.dtype == torch.float32
                                   for _, module, _ in guard.snapshots)
    original_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    ids = torch.arange(129).remainder(31).reshape(1, -1)
    with torch.no_grad():
        baseline = model(ids).logits
        model.bfloat16()  # Exact whole-module cast in DeepSpeed engine setup.
        assert any(not torch.equal(module.inv_freq.float(), original)
                   for _, module, original in guard.snapshots)
        guard.restore_if_cast()
        restored = model(ids).logits
    torch.testing.assert_close(restored, baseline, rtol=0, atol=0)
    for parameter, original in zip(model.parameters(), original_parameters):
        assert parameter.dtype == torch.bfloat16
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)
    assert all(module.inv_freq.dtype == torch.float32 for _, module, _ in guard.snapshots)
    assert any(row["rounding_max_abs"] > 0 for row in guard.restorations)
    count = len(guard.restorations)
    guard.restore_if_cast()
    assert len(guard.restorations) == count
