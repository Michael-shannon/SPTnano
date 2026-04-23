"""
clustering.py

Minimal clustering utilities for transformer embeddings.

Dependencies:
- numpy
- scikit-learn (for silhouette)
- hdbscan (optional but expected for `run_hdbscan`)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


def normalize_embeddings(emb: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize embeddings row-wise.

    Args:
        emb: (N, D)

    Returns:
        (N, D) float32
    """
    emb = np.asarray(emb, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.clip(norms, eps, None)


def run_hdbscan(
    emb: np.ndarray,
    min_cluster_size: int = 30,
    min_samples: Optional[int] = None,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Run HDBSCAN clustering.

    Args:
        emb: (N, D)
        min_cluster_size: HDBSCAN hyperparameter
        min_samples: HDBSCAN hyperparameter (None -> default)
        metric: distance metric

    Returns:
        labels: (N,) int (may contain -1 for noise)
        metrics: dict with basic info
    """
    try:
        import hdbscan  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "hdbscan is required for run_hdbscan(). Install with `pip install hdbscan`."
        ) from e

    X = np.asarray(emb, dtype=np.float32)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        prediction_data=False,
    )
    labels = clusterer.fit_predict(X)

    n_noise = int((labels == -1).sum())
    n_clusters = int(len(set(labels)) - (1 if -1 in set(labels) else 0))
    metrics = {
        "n_points": float(X.shape[0]),
        "n_clusters": float(n_clusters),
        "n_noise": float(n_noise),
        "noise_frac": float(n_noise / max(1, X.shape[0])),
    }
    return labels.astype(int), metrics


def compute_silhouette(emb: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute silhouette score on non-noise points (labels != -1) if possible.

    Returns:
        dict with "silhouette" (nan if not computable) and "silhouette_n"
    """
    from sklearn.metrics import silhouette_score

    X = np.asarray(emb, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)

    mask = labels != -1
    if mask.sum() < 3:
        return {"silhouette": float("nan"), "silhouette_n": float(mask.sum())}

    labs = labels[mask]
    if len(np.unique(labs)) < 2:
        return {"silhouette": float("nan"), "silhouette_n": float(mask.sum())}

    score = float(silhouette_score(X[mask], labs))
    return {"silhouette": score, "silhouette_n": float(mask.sum())}

