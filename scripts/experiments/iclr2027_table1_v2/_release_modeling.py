"""Reference implementation for the NBPO SafeRLHF preference models.

Published as ``modeling_nbpo_preference.py`` alongside the weights. It is
deliberately dependency-light -- ``torch``, ``transformers``, ``safetensors`` --
and builds the encoder from the config stored in the checkpoint directory, so
loading needs no second download and cannot silently pick up a different
``roberta-base`` revision.

Two heads share one backbone and differ only in what they compute:

``gpm``  ell_k(x,y,z) = 1/2 [ a_k(h_y,h_z) - a_k(h_z,h_y) ]
``bt``   ell_k(x,y,z) = r_k(h_y) - r_k(h_z)

Both give P_k(y>z|x) = sigmoid(ell_k). The GPM is exactly antisymmetric and
exactly self-tying for *any* parameters, and -- unlike the BT head -- it can
represent a cyclic preference; ``max_cyclic_residual`` in the model card is the
measurement of that.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn


class PairwiseHead(nn.Module):
    """Joint scorer over an ORDER-SENSITIVE concatenation ``[h_y, h_z]``."""

    def __init__(self, hidden: int, width: int = 512, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden, width), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width // 2), nn.GELU(),
            nn.Linear(width // 2, 1))

    def forward(self, h_y, h_z):
        return self.net(torch.cat([h_y, h_z], dim=-1)).squeeze(-1)


class AntiSymmetricGPM(nn.Module):
    def __init__(self, encoder, hidden, n_objectives, width=512, dropout=0.0):
        super().__init__()
        self.encoder = encoder
        self.n_objectives = n_objectives
        self.heads = nn.ModuleList(
            [PairwiseHead(hidden, width, dropout) for _ in range(n_objectives)])

    def encode(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids,
                         attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(dim=1) - 1
        return h[torch.arange(h.shape[0], device=h.device), last]

    def logit(self, h_y, h_z, k: int):
        return 0.5 * (self.heads[k](h_y, h_z) - self.heads[k](h_z, h_y))


class ScalarBT(nn.Module):
    def __init__(self, encoder, hidden, n_objectives, width=768, dropout=0.0):
        super().__init__()
        self.encoder = encoder
        self.n_objectives = n_objectives
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, width), nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(width, width // 2), nn.GELU(),
                          nn.Linear(width // 2, 1))
            for _ in range(n_objectives)])

    encode = AntiSymmetricGPM.encode

    def reward(self, h, k: int):
        return self.heads[k](h).squeeze(-1)

    def logit(self, h_y, h_z, k: int):
        return self.reward(h_y, k) - self.reward(h_z, k)


def load_checkpoint(directory, device="cpu"):
    """Build the model and tokenizer for one released seed directory."""
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, RobertaConfig, RobertaModel

    directory = Path(directory)
    cfg = json.loads((directory / "config.json").read_text())
    encoder = RobertaModel(RobertaConfig(**cfg["encoder_config"]))
    n_obj = len(cfg["objectives"])
    if cfg["kind"] == "gpm":
        model = AntiSymmetricGPM(encoder, cfg["hidden_size"], n_obj,
                                 width=cfg["gpm_head_width"], dropout=cfg["dropout"])
    else:
        model = ScalarBT(encoder, cfg["hidden_size"], n_obj,
                         width=cfg["bt_head_width"], dropout=cfg["dropout"])
    missing, unexpected = model.load_state_dict(
        load_file(str(directory / "model.safetensors")), strict=False)
    ignorable = {"encoder.embeddings.position_ids"}
    hard_missing = [k for k in missing if k not in ignorable]
    hard_unexpected = [k for k in unexpected if k not in ignorable]
    if hard_missing or hard_unexpected:
        raise RuntimeError(f"state dict mismatch: missing={hard_missing} "
                           f"unexpected={hard_unexpected}")
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(str(directory))
    return model, tok, cfg


@torch.no_grad()
def pair_probability(model, tok, cfg, prompt, y, z, device="cpu"):
    """``P_k(y > z | prompt)`` per objective, CALIBRATED, for one seed."""
    def enc(resp):
        return {k: v.to(device) for k, v in
                tok([prompt], [resp], padding=True, truncation="longest_first",
                    max_length=cfg["max_length"], return_tensors="pt").items()}
    ey, ez = enc(y), enc(z)
    h_y = model.encode(ey["input_ids"], ey["attention_mask"])
    h_z = model.encode(ez["input_ids"], ez["attention_mask"])
    out = {}
    for k, o in enumerate(cfg["objectives"]):
        ell = model.logit(h_y, h_z, k)
        T = cfg["calibration_temperature"][o]
        out[o] = {"logit": float(ell), "calibrated_probability": float(torch.sigmoid(ell / T))}
    return out


class CalibratedEnsemble:
    """The oracle used in the paper: calibrate each seed, THEN average.

    The order is not cosmetic. Averaging logits and calibrating afterwards is a
    different estimator; this class implements the one the numbers come from.
    """

    def __init__(self, root, kind="gpm", seeds=(41, 42, 43), device="cpu"):
        self.members = [load_checkpoint(Path(root) / kind / f"seed{s}", device)
                        for s in seeds]
        self.device = device
        self.objectives = self.members[0][2]["objectives"]

    @torch.no_grad()
    def probability(self, prompt, y, z):
        acc = {o: [] for o in self.objectives}
        for model, tok, cfg in self.members:
            p = pair_probability(model, tok, cfg, prompt, y, z, self.device)
            for o in self.objectives:
                acc[o].append(p[o]["calibrated_probability"])
        return {o: sum(v) / len(v) for o, v in acc.items()}
