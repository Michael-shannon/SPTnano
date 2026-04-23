"""
signature.py

Minimal "signature" clustering used in `Simple_SIGNATUREMAPPING_3_17_2026.ipynb`.

Implements a simple Mahalanobis/probabilistic classifier:
1) For each ground-truth category (e.g. `final_population`), fit a Gaussian model
   in embedding space (diagonal variance or Ledoit-Wolf shrunk covariance).
2) For each embedding, compute distance to every category Gaussian.
3) Convert distances to probabilities via softmax over negative distances.
4) Assign the most probable category as the "signature_cluster" id.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.special import softmax


def signature_cluster(
    emb: np.ndarray,
    categories: List[str],
    *,
    cov_mode: str = "shrunk",
    temperature: float = 0.5,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Args:
        emb: (N, D) embeddings
        categories: length-N list/array of category labels (strings)
        cov_mode: 'diagonal' or 'shrunk'
        temperature: softmax temperature on negative distances

    Returns:
        labels: (N,) int cluster ids 0..K-1
        uncertainty: (N,) float32 where 0 is most certain (1 - max_prob)
        margin: (N,) float32 gap between top-1 and top-2 probs
        entropy_norm: (N,) float32 normalized entropy in [0,1]
        category_lookup: {cluster_id: category_name}
    """
    emb = np.asarray(emb, dtype=np.float64)
    categories = np.asarray(categories)
    if emb.ndim != 2:
        raise ValueError(f"emb must be (N,D), got shape {emb.shape}")
    if categories.shape[0] != emb.shape[0]:
        raise ValueError("categories length must match emb rows")

    # Build a stable cluster id ordering (matches your signature notebook convention).
    unique_cats = sorted({c for c in categories.tolist() if c is not None})
    K = len(unique_cats)
    if K < 2:
        raise RuntimeError(f"Need at least 2 unique categories for signature clustering, got K={K}")

    cat2id = {c: i for i, c in enumerate(unique_cats)}
    category_lookup = {i: c for c, i in cat2id.items()}

    y = np.array([cat2id[c] for c in categories.tolist()], dtype=int)

    N, D = emb.shape
    cat_means = np.zeros((K, D), dtype=np.float64)

    if cov_mode == "diagonal":
        cat_vars = np.zeros((K, D), dtype=np.float64)
    elif cov_mode == "shrunk":
        cat_cov_invs: List[np.ndarray] = [None] * K  # type: ignore[list-item]
    else:
        raise ValueError("cov_mode must be 'diagonal' or 'shrunk'.")

    for cid in range(K):
        mask = y == cid
        cat_emb = emb[mask]
        cat_means[cid] = cat_emb.mean(axis=0)
        if cov_mode == "diagonal":
            v = cat_emb.var(axis=0)
            cat_vars[cid] = np.where(v < eps, eps, v)
        else:
            from sklearn.covariance import LedoitWolf

            lw = LedoitWolf()
            lw.fit(cat_emb)
            cat_cov_invs[cid] = lw.precision_

    # Compute squared Mahalanobis distances (N,K)
    mahal_sq = np.zeros((N, K), dtype=np.float64)
    for cid in range(K):
        diff = emb - cat_means[cid]
        if cov_mode == "diagonal":
            mahal_sq[:, cid] = (diff**2 / cat_vars[cid]).sum(axis=1)
        else:
            mahal_sq[:, cid] = np.sum(diff * (diff @ cat_cov_invs[cid]), axis=1)

    mahal_dist = np.sqrt(np.maximum(mahal_sq, 0))
    log_scores = -mahal_dist / float(temperature)
    probs = softmax(log_scores, axis=1)  # (N,K)

    labels = probs.argmax(axis=1).astype(int)

    max_probs = probs.max(axis=1)
    uncertainty = (1.0 - max_probs).astype(np.float32)

    entropy = -np.sum(probs * np.log(probs + eps), axis=1)
    entropy_norm = (entropy / np.log(K)).astype(np.float32)

    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    margin = (sorted_probs[:, 0] - sorted_probs[:, 1]).astype(np.float32)

    return labels, uncertainty, margin, entropy_norm, category_lookup

