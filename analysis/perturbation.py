"""
perturbation.py

Two minimal causal attention perturbation tests:

A) High-attention masking
   - Use attention to identify top X% attended timesteps
   - Zero those *input* timesteps
   - Recompute embedding
   - Measure cosine distance shift

B) Uniform attention ablation
   - Replace each encoder layer self-attention with uniform weights (values averaged)
   - Recompute embeddings
   - Measure cluster reassignment rate (nearest-centroid in embedding space)

Constraints:
- Keep residual connections intact (we only change the attention mix, not the encoder structure).
- GPU compatible.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cosine_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a = F.normalize(a, dim=-1, eps=eps)
    b = F.normalize(b, dim=-1, eps=eps)
    return 1.0 - (a * b).sum(dim=-1)


def _cls_attention_to_timesteps(attn: torch.Tensor) -> torch.Tensor:
    """
    Convert full attention maps into a per-timestep importance score using CLS query.

    Args:
        attn: (B, L, H, S, S) where S=T+1 and index 0 is CLS

    Returns:
        scores: (B, T) averaged over layers+heads, using CLS->token attention
    """
    if attn.dim() != 5:
        raise ValueError(f"Expected attn shape (B,L,H,S,S), got {tuple(attn.shape)}")
    # CLS query index 0 attends over keys 0..S-1. We use keys 1..T (exclude CLS key).
    cls_to_all = attn[:, :, :, 0, 1:]  # (B, L, H, T)
    return cls_to_all.mean(dim=(1, 2))  # (B, T)


def high_attention_masking_test(
    model: nn.Module,
    x: torch.Tensor,
    attn: torch.Tensor,
    top_frac: float = 0.1,
) -> Dict[str, np.ndarray]:
    """
    Test A: mask top-attended timesteps (zero input) and measure embedding shift.

    Args:
        model: eval-mode transformer
        x: (B, T, C) input on device
        attn: (B, L, H, S, S) attention on device (or CPU; will be moved)
        top_frac: fraction of timesteps to mask (0..1]

    Returns:
        dict with:
        - "cosine_distance_shift": (B,) float32
        - "masked_frac": scalar float32
    """
    if not (0.0 < top_frac <= 1.0):
        raise ValueError("top_frac must be in (0, 1].")

    device = x.device
    attn = attn.to(device)

    with torch.no_grad():
        emb0 = model(x)  # (B, D)

        scores = _cls_attention_to_timesteps(attn)  # (B, T)
        B, T = scores.shape
        k = max(1, int(round(top_frac * T)))
        top_idx = scores.topk(k=k, dim=1, largest=True).indices  # (B, k)

        x_masked = x.clone()
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
        x_masked[batch_idx, top_idx, :] = 0.0

        emb1 = model(x_masked)
        shift = _cosine_distance(emb0, emb1)  # (B,)

    return {
        "cosine_distance_shift": shift.detach().float().cpu().numpy().astype(np.float32),
        "masked_frac": np.array(float(k / T), dtype=np.float32),
    }


class _UniformSelfAttention(nn.Module):
    """
    Drop-in replacement for TransformerEncoderLayer.self_attn that applies uniform
    attention over source positions (keys), while keeping the linear projections
    and output projection consistent with the original MultiheadAttention.

    It returns ONLY attn_output (no weights), matching the caller expectations.
    """

    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        if not isinstance(mha, nn.MultiheadAttention):
            raise TypeError("Expected nn.MultiheadAttention.")
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        # Torch TransformerEncoder inspects this attribute for fast-path decisions.
        # Our encoder uses seq-first tensors, so batch_first is False.
        self.batch_first = getattr(mha, "batch_first", False)
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads.")

        # Copy parameters (share by reference to keep exact weights).
        self.in_proj_weight = mha.in_proj_weight
        self.in_proj_bias = mha.in_proj_bias
        self.out_proj = mha.out_proj
        self.bias_k = mha.bias_k
        self.bias_v = mha.bias_v
        self.add_zero_attn = mha.add_zero_attn
        self.dropout = mha.dropout
        self.training = False

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> torch.Tensor:
        # Expect (S, B, E) since TransformerEncoder uses seq-first.
        # We ignore masks and causality for this minimal ablation.
        if query.dim() != 3:
            raise ValueError("Expected query shape (S,B,E).")
        S, B, E = query.shape
        if E != self.embed_dim:
            raise ValueError(f"Expected embed dim {self.embed_dim}, got {E}.")

        # Compute V only (Q/K irrelevant for uniform weights, but keep projection pathway).
        # Use in-proj to compute q,k,v like PyTorch does: [q; k; v] projections.
        # We'll compute v_proj = linear(value, Wv, bv).
        W = self.in_proj_weight
        b = self.in_proj_bias
        # W shape: (3E, E). V block is last E rows.
        Wv = W[2 * E : 3 * E, :]
        bv = None if b is None else b[2 * E : 3 * E]
        v_proj = F.linear(value, Wv, bv)  # (S, B, E)

        # Reshape to (B, H, S, head_dim)
        v = v_proj.permute(1, 0, 2).contiguous().view(B, S, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)  # (B, H, S, head_dim)

        # Uniform attention => each query gets mean(V over src positions).
        v_mean = v.mean(dim=2, keepdim=True)  # (B, H, 1, head_dim)
        out = v_mean.expand(B, self.num_heads, S, self.head_dim)  # (B, H, S, head_dim)

        # Merge heads back: (S, B, E)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, S, E).permute(1, 0, 2)
        out = self.out_proj(out)
        return out


@contextmanager
def _uniform_attention_ablation(model: nn.Module) -> Iterable[None]:
    """
    Temporarily replace each encoder layer's self-attention with uniform attention.
    """
    if not hasattr(model, "transformer_encoder"):
        raise AttributeError("Model must have attribute 'transformer_encoder'.")
    enc = getattr(model, "transformer_encoder")
    layers = list(getattr(enc, "layers", []))
    if not layers:
        raise AttributeError("model.transformer_encoder.layers not found or empty.")

    originals: List[Tuple[nn.Module, nn.Module]] = []
    try:
        for layer in layers:
            mha = layer.self_attn
            uniform = _UniformSelfAttention(mha)
            originals.append((layer, mha))
            layer.self_attn = uniform  # type: ignore[assignment]
        yield
    finally:
        for layer, mha in originals:
            layer.self_attn = mha  # type: ignore[assignment]


def _nearest_centroid_labels(
    emb: np.ndarray, labels_ref: np.ndarray, emb_query: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign query embeddings to nearest centroid of reference clusters.

    - Excludes noise label -1 from centroid construction.
    - Returns (assigned_labels, valid_mask) where valid_mask indicates points in ref
      with non-noise label (used for reassignment rate).
    """
    X = emb.astype(np.float32)
    Y = emb_query.astype(np.float32)
    labels_ref = labels_ref.astype(int)

    valid = labels_ref != -1
    uniq = np.unique(labels_ref[valid])
    if uniq.size == 0:
        return np.full((Y.shape[0],), -1, dtype=int), valid

    centroids = np.stack([X[labels_ref == c].mean(axis=0) for c in uniq], axis=0)  # (K, D)
    # cosine distance to centroids (normalized)
    def l2norm(a):
        n = np.linalg.norm(a, axis=1, keepdims=True)
        return a / np.clip(n, 1e-12, None)

    Yn = l2norm(Y)
    Cn = l2norm(centroids)
    sims = Yn @ Cn.T  # (N, K)
    idx = sims.argmax(axis=1)
    return uniq[idx].astype(int), valid


def uniform_attention_ablation_test(
    model: nn.Module,
    x: torch.Tensor,
    labels: np.ndarray,
    embeddings_baseline: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Test B: ablate attention by enforcing uniform weights, then measure:
    - embedding cosine distance shift distribution
    - percent cluster change (nearest-centroid reassignment, ignoring noise in baseline)

    Args:
        model: eval-mode transformer
        x: (B, T, C) input on device
        labels: (B,) baseline cluster labels
        embeddings_baseline: (B, D) baseline embeddings (CPU numpy)

    Returns:
        dict with:
        - "cosine_distance_shift": (B,) float32
        - "percent_cluster_change": scalar float32 (computed on non-noise points)
    """
    device = x.device
    model.eval()

    with torch.no_grad():
        with _uniform_attention_ablation(model):
            emb_u = model(x).detach().float().cpu().numpy().astype(np.float32)  # (B, D)

    # Embedding shift distribution
    emb0 = torch.tensor(embeddings_baseline, device=device)
    emb1 = torch.tensor(emb_u, device=device)
    shift = _cosine_distance(emb0, emb1).detach().float().cpu().numpy().astype(np.float32)

    # Cluster reassignment rate via nearest-centroid mapping
    assigned, valid_mask = _nearest_centroid_labels(embeddings_baseline, labels, emb_u)
    labels = labels.astype(int)

    valid_idx = np.where(valid_mask)[0]
    if valid_idx.size == 0:
        pct_change = np.array(np.nan, dtype=np.float32)
    else:
        pct_change = np.array(float((assigned[valid_idx] != labels[valid_idx]).mean() * 100.0), dtype=np.float32)

    return {
        "cosine_distance_shift": shift,
        "percent_cluster_change": pct_change,
    }

