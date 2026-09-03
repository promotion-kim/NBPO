"""Trainer-branch tests for loss_type=nbpo (mnpo_scripts/mnpo_trainer.py).

Covers spec test 10 (sequence-sum logps and length-invariance of the target
scale), exactness of the nbpo loss branch against Eq. (26), and every
rejection of validate_nbpo_args. Follows the repo's fake-object pattern from
tests/test_revision_losses.py -- no model, no GPU, no dataset.
"""
from types import SimpleNamespace

import pytest
import torch

from mnpo_scripts.mnpo_config import MNPOConfig
from mnpo_scripts.mnpo_trainer import MNPOTrainer, validate_nbpo_args
from scripts.simpo_trainer import SimPOTrainer


class _FakeAccelerator:
    device = torch.device("cpu")

    def gather(self, tensor):
        return tensor


def _fake_trainer(**overrides):
    ns = SimpleNamespace(
        accelerator=_FakeAccelerator(),
        loss_type="nbpo",
        eta=1.0,
        beta=10.0,  # metrics-only display multiplier
        nbpo_target_column="nbpo_weighted_z",
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


# ---------------------------------------------------------------- test 10 --


def test_get_batch_logps_sum_vs_mean():
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 11)
    labels = torch.tensor([
        [-100, -100, 3, 4, 5, 6],       # 4 response tokens (first position is shifted off)
        [-100, 2, 3, -100, -100, -100],  # 1 response token
    ])
    lp_sum = SimPOTrainer.get_batch_logps(logits, labels, average_log_prob=False)
    lp_mean = SimPOTrainer.get_batch_logps(logits, labels, average_log_prob=True)
    shifted = labels[:, 1:].clone()
    mask = shifted != -100
    shifted[~mask] = 0
    per_token = torch.gather(logits[:, :-1].log_softmax(-1), 2,
                             shifted.unsqueeze(2)).squeeze(2)
    expected_sum = (per_token * mask).sum(-1)
    assert torch.allclose(lp_sum, expected_sum, atol=1e-6)
    assert torch.allclose(lp_mean, expected_sum / mask.sum(-1), atol=1e-6)
    # sums are NOT token means: reductions differ whenever lengths exceed 1
    assert not torch.allclose(lp_sum[0], lp_mean[0])


def test_sum_reduction_makes_target_scale_length_invariant():
    # Two pairs with identical SEQUENCE-SUM log-ratio changes but different
    # response lengths must produce the identical nbpo loss: under sum
    # reduction, h_t carries no length-dependent scaling of the target.
    fake = _fake_trainer()
    z = torch.tensor([0.8, 0.8])
    # pair 0: short responses; pair 1: long responses; same summed logps
    pcl = torch.tensor([-5.0, -50.0])
    prl = torch.tensor([-6.0, -51.0])
    hcl = torch.tensor([-5.5, -50.5])
    hrl = torch.tensor([-6.2, -51.2])
    losses, _, _ = MNPOTrainer.mnpo_loss(
        fake, pcl, prl, hcl * 0, hrl * 0,  # reference logps unused by the branch
        [(hcl, hrl)], nbpo_target=z,
    )
    assert torch.allclose(losses[0], losses[1], atol=1e-6)


# ------------------------------------------------- nbpo branch exactness --


def test_nbpo_loss_branch_matches_equation_26():
    fake = _fake_trainer(eta=2.5)
    pcl = torch.tensor([-10.0, -20.0])
    prl = torch.tensor([-12.0, -19.0])
    rcl = torch.tensor([-11.0, -21.0])   # reference logps: must NOT enter the loss
    rrl = torch.tensor([-13.0, -20.0])
    hcl = torch.tensor([-10.5, -20.5])
    hrl = torch.tensor([-12.5, -19.5])
    z = torch.tensor([0.4, -1.2])
    losses, chosen_rewards, rejected_rewards = MNPOTrainer.mnpo_loss(
        fake, pcl, prl, rcl, rrl, [(hcl, hrl)], nbpo_target=z,
    )
    h = (pcl - prl) - (hcl - hrl)
    expected = (h - 2.5 * z) ** 2
    assert torch.allclose(losses, expected, atol=1e-6)
    # reference logps affect only the display-reward metrics, never the loss
    losses2, _, _ = MNPOTrainer.mnpo_loss(
        fake, pcl, prl, rcl - 100.0, rrl + 100.0, [(hcl, hrl)], nbpo_target=z,
    )
    assert torch.allclose(losses, losses2, atol=1e-6)


def test_nbpo_loss_requires_history_and_target():
    fake = _fake_trainer()
    t = torch.zeros(2)
    with pytest.raises(ValueError, match="history0"):
        MNPOTrainer.mnpo_loss(fake, t, t, t, t, [], nbpo_target=t)
    with pytest.raises(ValueError, match="nbpo_weighted_z"):
        MNPOTrainer.mnpo_loss(fake, t, t, t, t, [(t, t)], nbpo_target=None)


# ------------------------------------------------------- validate_nbpo_args --


def _ok_args(**overrides):
    ns = SimpleNamespace(
        loss_type="nbpo", reference_anchor_weight=0.0, preference_sft_weight=0.0,
        logp_reduction="sum", max_history_t=1, history_weights=[1.0], weights=None,
        nbpo_target_column="nbpo_weighted_z",
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


GOOD_COLUMNS = ["prompt", "chosen", "rejected", "nbpo_weighted_z",
                "reference_chosen_logps", "reference_rejected_logps",
                "history0_chosen_logps", "history0_rejected_logps"]
GOOD_META = {"logp_reduction": "sum", "tokenizer_hash": "tok", "chat_template_hash": "chat"}


def test_validate_accepts_the_paper_configuration():
    validate_nbpo_args(_ok_args())
    validate_nbpo_args(_ok_args(), dataset_columns=GOOD_COLUMNS, precompute_meta=GOOD_META,
                       tokenizer_hash="tok", chat_template_hash="chat")
    validate_nbpo_args(SimpleNamespace(loss_type="ronpo"))  # no-op for other losses


@pytest.mark.parametrize("overrides,match", [
    ({"reference_anchor_weight": 0.05}, "reference_anchor_weight"),
    ({"preference_sft_weight": 0.005}, "preference_sft_weight"),
    ({"logp_reduction": "mean"}, "logp_reduction"),
    ({"max_history_t": 2}, "max_history_t"),
    ({"history_weights": [0.5, 0.5]}, "history_weights"),
    ({"weights": [0.7, 0.3]}, "history_weights"),
])
def test_validate_rejects_each_config_violation(overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_nbpo_args(_ok_args(**overrides))


def test_validate_rejects_dataset_and_metadata_violations():
    ok = _ok_args()
    with pytest.raises(ValueError, match="target column"):
        validate_nbpo_args(ok, dataset_columns=[c for c in GOOD_COLUMNS
                                                if c != "nbpo_weighted_z"],
                           precompute_meta=GOOD_META)
    with pytest.raises(ValueError, match="history0"):
        validate_nbpo_args(ok, dataset_columns=["nbpo_weighted_z"], precompute_meta=GOOD_META)
    with pytest.raises(ValueError, match="more than one history"):
        validate_nbpo_args(ok, dataset_columns=GOOD_COLUMNS + ["history1_chosen_logps",
                                                               "history1_rejected_logps"],
                           precompute_meta=GOOD_META)
    with pytest.raises(ValueError, match="precompute_meta.json is missing"):
        validate_nbpo_args(ok, dataset_columns=GOOD_COLUMNS, precompute_meta=None)
    with pytest.raises(ValueError, match="reduction='sum'"):
        validate_nbpo_args(ok, dataset_columns=GOOD_COLUMNS,
                           precompute_meta={**GOOD_META, "logp_reduction": "mean"})
    with pytest.raises(ValueError, match="tokenizer hash"):
        validate_nbpo_args(ok, dataset_columns=GOOD_COLUMNS, precompute_meta=GOOD_META,
                           tokenizer_hash="other", chat_template_hash="chat")
    with pytest.raises(ValueError, match="chat-template hash"):
        validate_nbpo_args(ok, dataset_columns=GOOD_COLUMNS, precompute_meta=GOOD_META,
                           tokenizer_hash="tok", chat_template_hash="other")


def test_config_defaults_preserve_legacy_behavior():
    cfg = MNPOConfig(output_dir="/tmp/nbpo-test-unused")
    assert cfg.logp_reduction == "mean"   # every legacy loss keeps token means
    assert cfg.nbpo_target_column == "nbpo_weighted_z"
    assert cfg.loss_type == "mnpo"
