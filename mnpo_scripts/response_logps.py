"""Shared FP32 response scoring, with bounded full-vocabulary temporaries.

The custom backward recomputes one chunk's softmax, avoiding retention of an
entire FP32 [batch, sequence, vocabulary] activation until backward. It is the
ordinary full-vocabulary log-softmax derivative, not a candidate softmax.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

LOGP_IMPLEMENTATION = "fp32_chunked_selected_logsoftmax_v1"


class _SelectedLogps(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, chunk_size):
        ctx.save_for_backward(logits, labels)
        ctx.chunk_size = chunk_size
        selected = torch.empty(labels.shape, device=logits.device, dtype=torch.float32)
        for start in range(0, logits.shape[1], chunk_size):
            end = start + chunk_size
            normalized = F.log_softmax(logits[:, start:end], dim=-1, dtype=torch.float32)
            selected[:, start:end] = normalized.gather(-1, labels[:, start:end, None]).squeeze(-1)
        return selected

    @staticmethod
    def backward(ctx, grad_selected):
        logits, labels = ctx.saved_tensors
        grad_logits = torch.empty_like(logits)
        for start in range(0, logits.shape[1], ctx.chunk_size):
            end = start + ctx.chunk_size
            derivative = -F.softmax(logits[:, start:end], dim=-1, dtype=torch.float32)
            index = labels[:, start:end, None]
            derivative.scatter_add_(-1, index, torch.ones_like(index, dtype=torch.float32))
            derivative.mul_(grad_selected[:, start:end, None].float())
            grad_logits[:, start:end] = derivative.to(logits.dtype)
        return grad_logits, None, None


def response_logps(logits, labels, average_log_prob=False, label_pad_token_id=-100,
                   is_encoder_decoder=False, chunk_size=64):
    """FP32 log-softmax, selected token values and response sum/legacy mean.

Only labels define the response event. No EOS is added and no prompt or padded
token is scored. Decoder-only logits predict the next label.
"""
    if logits.ndim != 3 or logits.shape[:-1] != labels.shape:
        raise ValueError("Logits and labels must have the same batch/sequence shape")
    if int(chunk_size) <= 0:
        raise ValueError("logp chunk_size must be positive")
    if not is_encoder_decoder:
        labels, logits = labels[:, 1:], logits[:, :-1]
    mask = labels.ne(label_pad_token_id)
    safe_labels = labels.masked_fill(~mask, 0)
    selected = _SelectedLogps.apply(logits, safe_labels, int(chunk_size))
    result = selected.masked_fill(~mask, 0).sum(-1, dtype=torch.float32)
    if average_log_prob:
        lengths = mask.sum(-1)
        if torch.any(lengths == 0):
            raise ValueError("Cannot average a response with no scored tokens")
        result = result / lengths.float()
    return result
