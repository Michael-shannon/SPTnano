"""
Per-frame (position-wise) attention utilities for TransformerMotionEncoder.

When ``use_raw_attention=True`` and ``return_raw_attention=True``, each layer's
tensor has shape ``[B, H, L, L]`` (batch, heads, query positions, key positions).
Sequence layout matches ``transformer.forward``: position 0 is CLS, positions
``1 .. L-1`` are the T motion frames (``L = T + 1``).

Canonical CLS→frame attention: for each head, take query row ``cls_index`` (0)
and key columns ``1..L-1`` — i.e. how much CLS attends to each real frame.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import re

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def _ensure_torch():
    if torch is None:
        raise ImportError("attention.py requires PyTorch.")


def cls_query_attention_to_all_keys(
    attn: "torch.Tensor",
    cls_index: int = 0,
) -> "torch.Tensor":
    """
    Attention weights from the CLS query row to every key position.

    Parameters
    ----------
    attn : Tensor
        One layer's weights, shape ``[B, H, L, L]`` (query last dim, key last).
    cls_index : int
        Index of the CLS token (default 0).

    Returns
    -------
    Tensor of shape ``[B, H, L]`` — attention from CLS to each key index.
    """
    _ensure_torch()
    if attn.dim() != 4:
        raise ValueError(f"Expected attention [B,H,L,L], got shape {tuple(attn.shape)}")
    return attn[:, :, cls_index, :]


def cls_frame_attention(
    attn: "torch.Tensor",
    cls_index: int = 0,
) -> "torch.Tensor":
    """
    CLS→frame attention only (exclude CLS-as-key self column).

    Parameters
    ----------
    attn : Tensor
        Shape ``[B, H, L, L]`` with ``L = T + 1`` (CLS + T frames).

    Returns
    -------
    Tensor of shape ``[B, H, T]`` where ``T = L - 1``, keys aligned with
    frame order in the window (0 = first frame after CLS).
    """
    _ensure_torch()
    full = cls_query_attention_to_all_keys(attn, cls_index=cls_index)
    # keys 1..L-1 are frames; key 0 is CLS
    if full.shape[-1] < 2:
        raise ValueError("Sequence too short: need CLS + at least one frame.")
    return full[:, :, 1:]


def reduce_heads(
    attn_bh_t: "torch.Tensor",
    how: str = "mean",
) -> "torch.Tensor":
    """
    Reduce head dimension.

    Parameters
    ----------
    attn_bh_t : Tensor
        Shape ``[B, H, T]``.
    how : ``'mean'`` | ``'max'`` | ``'sum'`` | ``'none'``
        If ``'none'``, returns input unchanged (caller must handle heads).
    """
    _ensure_torch()
    how = how.lower()
    if how == "none":
        return attn_bh_t
    if how == "mean":
        return attn_bh_t.mean(dim=1)
    if how == "max":
        return attn_bh_t.max(dim=1).values
    if how == "sum":
        return attn_bh_t.sum(dim=1)
    raise ValueError(f"Unknown head reduction: {how!r}")


def layer_frame_attention_to_numpy(
    attn: "torch.Tensor",
    head_reduce: str = "mean",
    cls_index: int = 0,
) -> np.ndarray:
    """
    Single layer: ``[B,H,L,L]`` -> numpy ``[B, T]`` after CLS→frame + head reduce.
    """
    _ensure_torch()
    af = cls_frame_attention(attn, cls_index=cls_index)
    # [B, H, T] -> [B, T] or [B, H, T] if none
    reduced = reduce_heads(af, how=head_reduce)
    return reduced.detach().float().cpu().numpy()


def per_frame_attention_numpy(
    attn_list: Sequence[Optional["torch.Tensor"]],
    head_reduce: str = "mean",
    cls_index: int = 0,
) -> List[np.ndarray]:
    """
    Full stack: one numpy array per layer, each ``[B, T]`` (or ``[B, H, T]`` if
    ``head_reduce`` is ``'none'``).
    """
    out: List[np.ndarray] = []
    for a in attn_list:
        if a is None:
            out.append(np.array([]))
            continue
        out.append(layer_frame_attention_to_numpy(a, head_reduce=head_reduce, cls_index=cls_index))
    return out


def flatten_batch_to_per_frame_row_dicts(
    attn_list: Sequence[Optional["torch.Tensor"]],
    head_reduce: str = "mean",
    cls_index: int = 0,
    prefix: str = "raw_attn",
) -> List[Dict[str, float]]:
    """
    Build one dict per batch row with keys
    ``{prefix}_L{layer}_frame_{t}`` for t = 0..T-1.

    Suitable for ``pd.DataFrame`` rows merged on ``window_uid``.

    Parameters
    ----------
    attn_list : list of Tensor or None
        Length = num_layers; each tensor ``[B, H, L, L]`` (same B).
    head_reduce : str
        Passed to :func:`reduce_heads`.
    cls_index : int
        CLS query index (default 0).
    prefix : str
        Column name prefix (default ``raw_attn``).

    Returns
    -------
    List of length B dicts mapping column name -> float.
    """
    _ensure_torch()
    if not attn_list:
        return []

    arrs: List[Tuple[int, np.ndarray]] = []
    for li, a in enumerate(attn_list):
        if a is None:
            continue
        arr = layer_frame_attention_to_numpy(
            a, head_reduce=head_reduce, cls_index=cls_index
        )
        arrs.append((li, arr))

    if not arrs:
        return []

    b = int(arrs[0][1].shape[0])
    rows: List[Dict[str, float]] = [dict() for _ in range(b)]

    for li, arr in arrs:
        if arr.ndim == 3:
            # [B, H, T] when head_reduce == 'none' — flatten heads into columns
            # raw_attn_L0_h0_frame_t, ... or we average here — user should not pass none
            # for flatten; document: use mean/max for flat columns.
            raise ValueError(
                "flatten_batch_to_per_frame_row_dicts expects head_reduce != 'none' "
                f"when using flat frame columns; got shape {arr.shape}"
            )
        # [B, T]
        _, t = arr.shape
        for bi in range(b):
            for ti in range(t):
                key = f"{prefix}_L{li}_frame_{ti}"
                rows[bi][key] = float(arr[bi, ti])

    return rows


def per_frame_attention_scalar_summaries(
    frame_weights: np.ndarray,
    prefix: str = "raw_attn",
    layer_index: int = 0,
) -> Dict[str, float]:
    """
    Summarize a single window's per-frame weights (1D, length T) with scalars
    (entropy, argmax, mass in thirds, etc.) for compact storage.

    Parameters
    ----------
    frame_weights : ndarray
        1D, non-negative; normalized internally for entropy.
    prefix : str
        Key prefix.
    layer_index : int
        Layer id for column names.
    """
    w = np.asarray(frame_weights, dtype=np.float64).ravel()
    if w.size == 0:
        return {}
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s <= 0:
        p = np.ones_like(w) / (w.size or 1)
    else:
        p = w / s
    ent = float(-(p * np.log(p + 1e-12)).sum())
    max_entropy = float(np.log(max(w.size, 1)))
    ent_norm = ent / max_entropy if max_entropy > 0 else 0.0
    t = int(w.size)
    third = max(1, t // 3)
    out: Dict[str, float] = {
        f"{prefix}_L{layer_index}_frame_entropy": ent,
        f"{prefix}_L{layer_index}_frame_entropy_norm": ent_norm,
        f"{prefix}_L{layer_index}_frame_argmax": float(np.argmax(w)),
        f"{prefix}_L{layer_index}_frame_max": float(w.max()),
        f"{prefix}_L{layer_index}_frame_mass_first_third": float(p[:third].sum()),
        f"{prefix}_L{layer_index}_frame_mass_mid_third": float(p[third : 2 * third].sum()),
        f"{prefix}_L{layer_index}_frame_mass_last_third": float(p[2 * third :].sum()),
    }
    return out


def flatten_batch_to_scalar_summary_row_dicts(
    attn_list: Sequence[Optional["torch.Tensor"]],
    head_reduce: str = "mean",
    cls_index: int = 0,
    prefix: str = "raw_attn",
) -> List[Dict[str, float]]:
    """
    One dict per batch row with scalar summaries per layer (no per-frame columns).
    """
    per_layer = per_frame_attention_numpy(
        attn_list, head_reduce=head_reduce, cls_index=cls_index
    )
    if not per_layer or per_layer[0].size == 0:
        return []

    b = int(per_layer[0].shape[0])
    rows: List[Dict[str, float]] = [dict() for _ in range(b)]
    for li, arr in enumerate(per_layer):
        if arr.size == 0:
            continue
        for bi in range(b):
            summ = per_frame_attention_scalar_summaries(
                arr[bi], prefix=prefix, layer_index=li
            )
            rows[bi].update(summ)
    return rows


def column_names_per_frame(
    n_layers: int,
    n_frames: int,
    prefix: str = "raw_attn",
) -> List[str]:
    """Ordered column names for ``flatten_batch_to_per_frame_row_dicts``."""
    names: List[str] = []
    for li in range(n_layers):
        for ti in range(n_frames):
            names.append(f"{prefix}_L{li}_frame_{ti}")
    return names


_PER_FRAME_CURVE_COL = re.compile(r"^raw_attn_L\d+_frame_\d+$")


def discover_per_frame_curve_column_names(
    columns: Sequence[Any],
) -> List[str]:
    """
    Names of per-frame CLS→frame attention **curve** columns from EVAL Cell 6+
    (``PER_FRAME_MODE=''full''``): ``raw_attn_L<layer>_frame_<t>``.

    Does **not** match scalar-summary names like ``raw_attn_L0_frame_entropy``.
    """
    found = [str(c) for c in columns if _PER_FRAME_CURVE_COL.match(str(c))]

    def _key(name: str) -> Tuple[int, int]:
        m = re.match(r"^raw_attn_L(\d+)_frame_(\d+)$", name)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    return sorted(found, key=_key)


def diagnostic_raw_attn_column_sample(columns: Sequence[Any], limit: int = 30) -> List[str]:
    """For debugging: ``raw_attn`` column names that are not per-frame curves."""
    out: List[str] = []
    for c in columns:
        s = str(c)
        if "raw_attn" not in s:
            continue
        if _PER_FRAME_CURVE_COL.match(s):
            continue
        out.append(s)
        if len(out) >= limit:
            break
    return out


def correlate_embedding_dim_with_columns(
    emb: np.ndarray,
    df: Any,
    column_names: Sequence[str],
    method: str = "spearman",
) -> np.ndarray:
    """
    Matrix of correlations: each embedding dimension vs each column in ``column_names``.

    ``df`` must be pandas DataFrame or polars DataFrame with ``column_names``;
    rows aligned with ``emb`` (same order).

    Returns
    -------
    ndarray of shape ``(D, len(column_names))`` with correlations; NaN if constant.
    """
    from scipy import stats as scipy_stats

    if hasattr(df, "to_pandas"):
        pdf = df.to_pandas()
    else:
        pdf = df

    d = emb.shape[1]
    c = len(column_names)
    out = np.full((d, c), np.nan, dtype=np.float64)
    for j in range(d):
        x = emb[:, j]
        ok = np.isfinite(x)
        for ci, col in enumerate(column_names):
            if col not in pdf.columns:
                continue
            y = pdf[col].to_numpy()
            m = ok & np.isfinite(y)
            if m.sum() < 3:
                continue
            xx, yy = x[m], y[m]
            if method == "pearson":
                r, _ = scipy_stats.pearsonr(xx, yy)
            elif method == "spearman":
                r, _ = scipy_stats.spearmanr(xx, yy)
            else:
                raise ValueError(method)
            out[j, ci] = r
    return out


__all__ = [
    "cls_query_attention_to_all_keys",
    "cls_frame_attention",
    "reduce_heads",
    "layer_frame_attention_to_numpy",
    "per_frame_attention_numpy",
    "flatten_batch_to_per_frame_row_dicts",
    "flatten_batch_to_scalar_summary_row_dicts",
    "per_frame_attention_scalar_summaries",
    "column_names_per_frame",
    "discover_per_frame_curve_column_names",
    "diagnostic_raw_attn_column_sample",
    "correlate_embedding_dim_with_columns",
]
