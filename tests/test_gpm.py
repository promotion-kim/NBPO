"""The anti-symmetric GPM: guarantees that must hold by construction, not by fit.

A scalar reward model can only express a transitive preference. That is the
representational limit the whole NBPO argument is about, so a learned judge for
this project must provably not have it. These tests check the guarantees on
RANDOM weights -- if they only held after training they would be properties of
the fit, not of the architecture.
"""
import pytest
import torch
import torch.nn as nn

from mnpo_scripts.gpm import (
    AntiSymmetricGPM, GPMProvenance, PairwiseHead, antisymmetrize,
    assert_not_scalar_decomposable, preference_probability, swap_augmented_loss,
)


class TinyEncoder(nn.Module):
    """Deterministic stand-in for a language model encoder."""

    def __init__(self, vocab=64, hidden=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)

    def forward(self, input_ids, attention_mask=None):
        out = type("o", (), {})()
        out.last_hidden_state = self.emb(input_ids)
        return out


@pytest.fixture
def model():
    torch.manual_seed(0)
    return AntiSymmetricGPM(TinyEncoder(), hidden=16, n_objectives=3, width=32)


@pytest.fixture
def h():
    torch.manual_seed(1)
    return torch.randn(8, 16, dtype=torch.float32)


# --- the two exact guarantees ----------------------------------------------

def test_antisymmetry_is_exact_on_random_weights(model, h):
    """P(y>z) + P(z>y) == 1, to floating point, for untrained parameters."""
    for k in range(model.n_objectives):
        p_yz = model.probability(h[:4], h[4:], k)
        p_zy = model.probability(h[4:], h[:4], k)
        assert torch.allclose(p_yz + p_zy, torch.ones_like(p_yz), atol=1e-6)


def test_self_comparison_is_exactly_a_tie(model, h):
    """P(y>y) == 0.5 exactly: a(y,y) - a(y,y) = 0 regardless of parameters."""
    for k in range(model.n_objectives):
        p = model.probability(h, h, k)
        assert torch.allclose(p, torch.full_like(p, 0.5), atol=1e-7)
        assert torch.allclose(model.logit(h, h, k), torch.zeros(h.shape[0]), atol=1e-7)


def test_guarantees_survive_training(model, h):
    """They are architectural, so a few optimizer steps cannot break them."""
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    target = torch.rand(4)
    for _ in range(20):
        opt.zero_grad()
        swap_augmented_loss(model, h[:4], h[4:], 0, target).backward()
        opt.step()
    p = model.probability(h[:4], h[4:], 0)
    assert torch.allclose(p + model.probability(h[4:], h[:4], 0),
                          torch.ones_like(p), atol=1e-6)
    assert torch.allclose(model.probability(h, h, 0),
                          torch.full((8,), 0.5), atol=1e-6)


def test_antisymmetrize_helper_is_exact():
    a, b = torch.randn(16), torch.randn(16)
    assert torch.allclose(antisymmetrize(a, b), -antisymmetrize(b, a), atol=1e-7)
    assert torch.allclose(antisymmetrize(a, a), torch.zeros(16), atol=1e-7)


# --- the representational claim --------------------------------------------

def test_model_is_not_a_scalar_reward_model(model, h):
    """A scalar model telescopes: ell(i,j)+ell(j,k)+ell(k,i) == 0 for every
    triple. Finding a triple where it does not is proof this model can represent
    a cycle -- which is the entire reason for not using a reward model."""
    rep = assert_not_scalar_decomposable(model, h, k=0)
    assert rep["scalar_decomposable"] is False
    assert rep["max_cyclic_residual"] > 1e-3


def test_a_scalar_model_is_detected_as_scalar(h):
    """The detector must not simply always say 'not scalar'."""
    class ScalarGPM(AntiSymmetricGPM):
        def joint_score(self, h_y, h_z, k):
            return self.heads[k].net[0].weight[:, :h_y.shape[-1]].sum() * 0 + h_y.sum(-1)
    torch.manual_seed(0)
    m = ScalarGPM(TinyEncoder(), hidden=16, n_objectives=1, width=32)
    rep = assert_not_scalar_decomposable(m, h, k=0)
    assert rep["scalar_decomposable"] is True
    assert rep["max_cyclic_residual"] < 1e-4


def test_pair_head_is_order_sensitive(h):
    """An order-insensitive head would force ell to zero everywhere."""
    torch.manual_seed(2)
    head = PairwiseHead(16, width=32)
    assert not torch.allclose(head(h[:4], h[4:]), head(h[4:], h[:4]), atol=1e-4)


# --- loss ------------------------------------------------------------------

def test_swap_augmented_loss_is_identical_under_swap(model, h):
    """Because the model is exactly antisymmetric, the swapped loss is the same
    quantity -- so explicit swap augmentation would only double the gradient."""
    p = torch.tensor([0.9, 0.25, 0.5, 0.75])
    a = swap_augmented_loss(model, h[:4], h[4:], 0, p)
    b = swap_augmented_loss(model, h[4:], h[:4], 0, 1.0 - p)
    assert torch.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_loss_accepts_soft_and_hard_labels(model, h, p):
    v = swap_augmented_loss(model, h[:4], h[4:], 0, torch.full((4,), p))
    assert torch.isfinite(v)


def test_loss_rejects_out_of_range_labels(model, h):
    with pytest.raises(ValueError, match=r"p_hat must lie"):
        swap_augmented_loss(model, h[:4], h[4:], 0, torch.tensor([1.5, 0., 0., 0.]))


def test_a_tie_label_drives_the_logit_to_zero(model, h):
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    for _ in range(60):
        opt.zero_grad()
        swap_augmented_loss(model, h[:4], h[4:], 0, torch.full((4,), 0.5)).backward()
        opt.step()
    assert model.logit(h[:4], h[4:], 0).abs().max() < 0.35


# --- invariance, batching, provenance --------------------------------------

def test_prediction_does_not_depend_on_which_slot_a_response_is_placed_in(model, h):
    """Position/identifier invariance: the model sees encodings, and the
    antisymmetrization makes the two orders one quantity."""
    p = model.probability(h[:4], h[4:], 0)
    q = 1.0 - model.probability(h[4:], h[:4], 0)
    assert torch.allclose(p, q, atol=1e-6)


def test_batched_and_single_predictions_agree(model, h):
    batch = model.probability(h[:4], h[4:], 0)
    single = torch.stack([model.probability(h[i:i+1], h[4+i:5+i], 0)[0] for i in range(4)])
    assert torch.allclose(batch, single, atol=1e-6)


def test_encode_reads_the_last_non_padding_token(model):
    ids = torch.tensor([[3, 4, 5, 0, 0], [7, 8, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    got = model.encode(ids, mask)
    want = torch.stack([model.encoder.emb(ids[0])[2], model.encoder.emb(ids[1])[1]])
    assert torch.allclose(got, want, atol=1e-7)


def test_provenance_fingerprint_binds_the_configuration():
    a = GPMProvenance("enc", "full_finetune", 4, 16, 32)
    b = GPMProvenance("enc", "peft_lora", 4, 16, 32)
    assert a.fingerprint() != b.fingerprint()
    d = a.to_dict()
    assert d["adaptation_mode"] == "full_finetune"     # declared, never implicit
    assert "not decomposable into r(y) - r(z)" in d["guarantees"]
