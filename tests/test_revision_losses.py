from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.revision.loss_formulas import (
    dpo_sigmoid_loss,
    ipo_loss,
    simpo_sigmoid_loss,
)


class _FakeAccelerator:
    device = torch.device("cpu")

    def gather(self, value):
        return value


def _toy_logps():
    return {
        "pc": torch.tensor([-1.0, -0.4, -1.7], dtype=torch.float32),
        "pr": torch.tensor([-1.6, -0.8, -1.2], dtype=torch.float32),
        "rc": torch.tensor([-1.2, -0.5, -1.5], dtype=torch.float32),
        "rr": torch.tensor([-1.4, -0.9, -1.1], dtype=torch.float32),
    }


def test_dpo_sigmoid_matches_trl_reference():
    from trl.trainer.dpo_trainer import DPOTrainer, FDivergenceType

    x = _toy_logps()
    fake = SimpleNamespace(
        accelerator=_FakeAccelerator(),
        reference_free=False,
        f_divergence_type=FDivergenceType.REVERSE_KL.value,
        f_divergence_params=None,
        loss_type="sigmoid",
        beta=0.05,
        label_smoothing=0.0,
    )
    official, _, _ = DPOTrainer.dpo_loss(fake, x["pc"], x["pr"], x["rc"], x["rr"])
    ours = dpo_sigmoid_loss(x["pc"], x["pr"], x["rc"], x["rr"], beta=0.05)
    assert torch.allclose(ours, official, atol=1e-7)


def test_ipo_matches_trl_reference():
    from trl.trainer.dpo_trainer import DPOTrainer, FDivergenceType

    x = _toy_logps()
    fake = SimpleNamespace(
        accelerator=_FakeAccelerator(),
        reference_free=False,
        f_divergence_type=FDivergenceType.REVERSE_KL.value,
        f_divergence_params=None,
        loss_type="ipo",
        beta=0.1,
        label_smoothing=0.0,
    )
    official, _, _ = DPOTrainer.dpo_loss(fake, x["pc"], x["pr"], x["rc"], x["rr"])
    ours = ipo_loss(x["pc"], x["pr"], x["rc"], x["rr"], beta=0.1)
    assert torch.allclose(ours, official, atol=1e-7)


def test_simpo_matches_local_official_port():
    from scripts.simpo_trainer import SimPOTrainer

    x = _toy_logps()
    fake = SimpleNamespace(
        accelerator=_FakeAccelerator(),
        beta=2.5,
        gamma_beta_ratio=0.4,
        loss_type="sigmoid",
        label_smoothing=0.0,
    )
    official, _, _ = SimPOTrainer.simpo_loss(fake, x["pc"], x["pr"])
    ours = simpo_sigmoid_loss(x["pc"], x["pr"], beta=2.5, gamma_over_beta=0.4)
    assert torch.allclose(ours, official, atol=1e-7)
