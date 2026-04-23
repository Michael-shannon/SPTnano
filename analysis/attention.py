"""
attention.py

Minimal utilities to extract embeddings + attention weights from the trained
TransformerMotionEncoder during inference.

Key tensor shapes
-----------------
Let:
- B = batch size
- T = input sequence length (timesteps in a window, e.g. 60)
- S = T + 1 (CLS token + timesteps)
- L = num_layers
- H = num_heads
- D = embedding dimension (model's output dim; often 64 or 128)

Model input:                x              : (B, T, 3)  [dx, dy, dtheta] per timestep
Model embedding output:     embeddings     : (B, D)
Extracted attention weights attentions     : (B, L, H, S, S)

Notes
-----
- This code assumes the model is frozen and in eval mode.
- Uses a lightweight monkeypatch to force attention weight computation from
  each `nn.MultiheadAttention` inside `nn.TransformerEncoderLayer`.
- GPU compatible: all tensors stay on the chosen device during forward; results
  are moved to CPU NumPy arrays at the end for analysis/plotting.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def _as_input_tensor(batch: Any) -> torch.Tensor:
    """
    Make this tolerant to common dataset outputs.

    Accepts:
    - Tensor: (B, T, C)
    - dict with "features" or "features_t": (B, T, C)
    - tuple/list: first element is the tensor
    """
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, dict):
        if "features" in batch:
            return batch["features"]
        if "features_t" in batch:
            return batch["features_t"]
        raise KeyError("Batch dict must contain 'features' or 'features_t'.")
    if isinstance(batch, (tuple, list)) and len(batch) > 0:
        return _as_input_tensor(batch[0])
    raise TypeError(f"Unsupported batch type for model input: {type(batch)!r}")


@contextmanager
def _capture_transformer_encoder_attn(
    model: nn.Module,
) -> Iterable[List[torch.Tensor]]:
    """
    Context manager that forces `nn.MultiheadAttention` inside each encoder layer
    to compute and expose attention weights, without changing the layer outputs.

    Returns a list `per_layer_weights` that will be filled during forward with:
      per_layer_weights[layer_idx] = attn_weights tensor of shape (B, H, S, S)

    Implementation detail:
    - We monkeypatch each `layer.self_attn.forward` to always request
      `need_weights=True` and `average_attn_weights=False`, then *store* the
      weights and return only the attention output so residual paths remain intact.
    """
    if not hasattr(model, "transformer_encoder"):
        raise AttributeError("Model must have attribute 'transformer_encoder'.")
    enc = getattr(model, "transformer_encoder")
    if not hasattr(enc, "layers"):
        raise AttributeError("model.transformer_encoder must have attribute 'layers'.")

    layers = list(enc.layers)
    per_layer_weights: List[Optional[torch.Tensor]] = [None] * len(layers)
    originals: List[Tuple[nn.Module, Any]] = []

    def make_patched_forward(layer_idx: int, mha: nn.Module, orig_forward):
        def patched_forward(*args, **kwargs):
            # Force attention weight computation.
            kwargs = dict(kwargs)
            kwargs["need_weights"] = True
            # Ensure per-head weights; required to return (B, H, S, S)
            kwargs["average_attn_weights"] = False

            out = orig_forward(*args, **kwargs)
            # PyTorch MultiheadAttention returns:
            # - (attn_output, attn_weights) if need_weights=True
            # - attn_output if need_weights=False
            if isinstance(out, tuple) and len(out) == 2:
                attn_output, attn_weights = out
            else:
                # Very old / custom MHA; can't extract weights.
                attn_output, attn_weights = out, None

            if attn_weights is None:
                raise RuntimeError(
                    "Failed to capture attention weights. "
                    "Your MultiheadAttention did not return weights."
                )

            # Normalize shape to (B, H, S, S).
            # Common shapes:
            # - (B, H, S, S) when average_attn_weights=False
            # - (B, S, S) if averaged (shouldn't happen here)
            if attn_weights.dim() == 3:
                attn_weights = attn_weights.unsqueeze(1)
            per_layer_weights[layer_idx] = attn_weights

            # IMPORTANT: return only the attention output so TransformerEncoderLayer
            # behavior stays identical.
            return attn_output

        return patched_forward

    try:
        for i, layer in enumerate(layers):
            if not hasattr(layer, "self_attn"):
                raise AttributeError("Encoder layer missing 'self_attn' module.")
            mha = layer.self_attn
            orig_forward = mha.forward
            originals.append((mha, orig_forward))
            mha.forward = make_patched_forward(i, mha, orig_forward)  # type: ignore[method-assign]

        yield per_layer_weights  # filled after a forward call
    finally:
        # Restore original forwards
        for (mha, orig_forward) in originals:
            mha.forward = orig_forward  # type: ignore[method-assign]


def extract_embeddings_and_attention(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Run inference to extract embeddings and per-layer attention weights.

    Args:
        model: Transformer model (frozen, eval mode)
        dataloader: yields batches containing input sequences shaped (B, T, 3)
        device: torch device (cuda or cpu)

    Returns:
        dict with:
        - "embeddings": (N, D) float32
        - "attentions": (N, L, H, S, S) float32
    """
    model.eval()

    all_emb: List[np.ndarray] = []
    all_attn: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            x = _as_input_tensor(batch).to(device, non_blocking=True)

            # Capture attention weights during the same forward used for embeddings.
            with _capture_transformer_encoder_attn(model) as per_layer:
                emb = model(x)  # (B, D)

            # Stack per-layer attention: list length L, each (B, H, S, S)
            if any(w is None for w in per_layer):
                missing = [i for i, w in enumerate(per_layer) if w is None]
                raise RuntimeError(f"Missing attention weights for layers: {missing}")

            attn_layers = torch.stack([w for w in per_layer if w is not None], dim=1)
            # attn_layers: (B, L, H, S, S)

            all_emb.append(emb.detach().float().cpu().numpy())
            all_attn.append(attn_layers.detach().float().cpu().numpy())

    embeddings = np.concatenate(all_emb, axis=0) if all_emb else np.zeros((0, 0), dtype=np.float32)
    attentions = np.concatenate(all_attn, axis=0) if all_attn else np.zeros((0, 0, 0, 0, 0), dtype=np.float32)

    return {"embeddings": embeddings.astype(np.float32), "attentions": attentions.astype(np.float32)}


def cluster_average_attention(attentions: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Compute cluster-averaged attention maps.

    Args:
        attentions: (N, L, H, S, S)
        labels: (N,) int (e.g. HDBSCAN labels, may include -1 for noise)

    Returns:
        dict mapping cluster_label -> mean attention (L, H, S, S)
    """
    if attentions.ndim != 5:
        raise ValueError(f"attentions must have shape (N,L,H,S,S), got {attentions.shape}")
    if labels.ndim != 1 or labels.shape[0] != attentions.shape[0]:
        raise ValueError("labels must be (N,) and match attentions[0].")

    out: Dict[int, np.ndarray] = {}
    for lab in np.unique(labels):
        mask = labels == lab
        if mask.sum() == 0:
            continue
        out[int(lab)] = attentions[mask].mean(axis=0)
    return out


def attention_entropy(attn: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute per-query attention entropy.

    Args:
        attn: attention weights with shape (..., S) where the last dim is the
              distribution over keys for each query.
              For example: (N, L, H, S, S) or (L, H, S, S)
        eps: numerical stability

    Returns:
        entropy with shape attn.shape[:-1]
    """
    p = np.clip(attn, eps, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    return -(p * np.log(p)).sum(axis=-1)

