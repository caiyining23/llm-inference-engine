"""Attention correctness baselines: explicit causal reference and torch SDPA.

Inputs are [batch, heads, query_tokens, head_dim] and
[batch, heads, key_tokens, head_dim]. Query tokens are the *suffix* of
the key/value sequence (incremental decode); masks use bottom-right alignment.

This module uses ordinary PyTorch operators. It does NOT implement CUDA kernels
or paged attention. torch SDPA may select optimized kernels on supported GPUs.
"""
import math


def _validate(query, key, value):
    import torch
    if any(not isinstance(t, torch.Tensor) or t.ndim != 4
           for t in (query, key, value)):
        raise ValueError("query/key/value must be rank-4 PyTorch tensors")
    b, h, q, d = query.shape
    if q < 1 or key.shape[:2] != (b, h) or value.shape[:2] != (b, h):
        raise ValueError("batch and head dimensions must match")
    if key.shape[2] < q or key.shape[3] != d or value.shape != key.shape:
        raise ValueError("key/value shape mismatch or key shorter than query")
    if any(t.device != query.device or t.dtype != query.dtype for t in (key, value)):
        raise ValueError("device and dtype must match")
    if not torch.is_floating_point(query):
        raise ValueError("attention requires floating-point tensors")


def suffix_causal_mask(query_length, key_length, device):
    """True = attend; query i can attend through key_length-query_length+i."""
    import torch
    if query_length < 1 or key_length < query_length:
        raise ValueError("expected key_length >= query_length >= 1")
    qi = torch.arange(query_length, device=device)[:, None]
    ki = torch.arange(key_length, device=device)[None, :]
    return ki <= (key_length - query_length + qi)


def reference_attention(query, key, value):
    """Readable FP32-accumulation reference, returns input dtype."""
    import torch
    _validate(query, key, value)
    mask = suffix_causal_mask(query.shape[-2], key.shape[-2], query.device)
    scores = (query.float() @ key.float().transpose(-2, -1)) / math.sqrt(query.shape[-1])
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return (probs @ value.float()).to(query.dtype)


def sdpa_attention(query, key, value):
    """PyTorch built-in scaled dot-product attention with explicit suffix mask.

    We intentionally pass is_causal=False: is_causal=True has different
    alignments across versions when query_length != key_length.
    """
    import torch.nn.functional as F
    _validate(query, key, value)
    mask = suffix_causal_mask(query.shape[-2], key.shape[-2], query.device)
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False
    )
