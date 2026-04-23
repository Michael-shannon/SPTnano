"""
Window-level scalar features aligned with ``ParticleMetrics`` / ``features.py``.

Used for Option B transformer inputs: recompute four values from **actual** (x, y) paths,
including after augmentation, using the same definitions as time-windowed metrics:

- ``cum_displacement_um``: sum of segment lengths between consecutive positions
- ``avg_speed_um_s``: mean of (segment_len / delta_t) with constant ``delta_t = time_between_frames``
- ``self_intersections``: segment–segment intersection count (same geometry as ``calculate_self_intersections``)
- ``anomalous_exponent``: from MSD fit ``4 D t^α`` (``fit_method='r2_threshold'`` path, ``bad_fit_strategy='flag'``)

See ``features.py`` — sliding window block around lines 1156–1280.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.optimize
from tqdm.auto import tqdm

from . import config

__all__ = [
    "positions_from_stored_deltas",
    "compute_cum_displacement_um",
    "compute_avg_speed_um_s",
    "compute_self_intersections_count",
    "compute_anomalous_exponent_msd_r2_threshold",
    "compute_four_window_scalars_numpy",
    "broadcast_four_to_time",
    "zscore_numpy",
    "precompute_scalar_mean_std_recompute",
]


def positions_from_stored_deltas(dx: np.ndarray, dy: np.ndarray, n_valid: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct positions consistent with training tensors and ``shuffle_scale_angle`` augmentation
    (``x_coords = cumsum(dx)`` over the window).

    Uses the first ``n_valid`` frames (ignores padding).
    """
    n = int(max(0, min(n_valid, len(dx))))
    if n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64)[:n]
    dy = np.asarray(dy, dtype=np.float64)[:n]
    x = np.cumsum(dx)
    y = np.cumsum(dy)
    return x, y


def compute_cum_displacement_um(x_um: np.ndarray, y_um: np.ndarray) -> float:
    """Match ``features.py`` sliding-window block (``segment_len_um`` sum)."""
    x_um = np.asarray(x_um, dtype=np.float64)
    y_um = np.asarray(y_um, dtype=np.float64)
    if len(x_um) < 2:
        return 0.0
    dx = np.diff(x_um, prepend=x_um[0])
    dy = np.diff(y_um, prepend=y_um[0])
    seg = np.sqrt(dx**2 + dy**2)
    seg[0] = 0.0
    return float(np.nansum(seg))


def compute_avg_speed_um_s(
    x_um: np.ndarray,
    y_um: np.ndarray,
    time_between_frames: float | None = None,
) -> float:
    """
    Mean speed over frames with valid segments — same as ``window_data['speed_um_s'].mean()``
    when speeds are ``segment_len / delta_t`` with uniform ``delta_t``.
    """
    dt = float(
        time_between_frames
        if time_between_frames is not None
        else config.TIME_BETWEEN_FRAMES
    )
    if dt <= 0:
        dt = config.TIME_BETWEEN_FRAMES
    if len(x_um) < 2:
        return 0.0
    x_um = np.asarray(x_um, dtype=np.float64)
    y_um = np.asarray(y_um, dtype=np.float64)
    dx = np.diff(x_um, prepend=x_um[0])
    dy = np.diff(y_um, prepend=y_um[0])
    seg = np.sqrt(dx**2 + dy**2)
    # First "segment" from prepend matches features.py convention (first step may be 0)
    speeds = seg / dt
    return float(np.nanmean(speeds))


def segments_intersect(p1, p2, p3, p4) -> bool:
    """Copied from ``ParticleMetrics.segments_intersect``."""

    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def compute_self_intersections_count(x_um: np.ndarray, y_um: np.ndarray) -> int:
    """Same algorithm as ``ParticleMetrics.calculate_self_intersections``."""
    points = np.column_stack([np.asarray(x_um, dtype=np.float64), np.asarray(y_um, dtype=np.float64)])
    n_points = len(points)
    if n_points < 4:
        return 0
    intersections = 0
    for i in range(n_points - 1):
        p1, p2 = points[i], points[i + 1]
        for j in range(i + 2, n_points - 1):
            p3, p4 = points[j], points[j + 1]
            if segments_intersect(p1, p2, p3, p4):
                intersections += 1
    return int(intersections)


def msd_model(t, D, alpha):
    """``ParticleMetrics.msd_model``."""
    return 4 * D * t**alpha


def compute_anomalous_exponent_msd_r2_threshold(
    x_um: np.ndarray,
    y_um: np.ndarray,
    time_between_frames: float | None = None,
    *,
    tolerance: float = 0.1,
    r2_threshold: float = 0.95,
    anomalous_r2_threshold: float = 0.80,
    bad_fit_strategy: str = "flag",
) -> float:
    """
    MSD curve + single anomalous fit as in ``calculate_msd_for_track`` (``r2_threshold`` branch).

    Returns ``alpha`` (float); on hard failure returns ``nan`` (caller may replace with 0.0).
    """
    track_data = pd.DataFrame(
        {
            "x_um": np.asarray(x_um, dtype=np.float64),
            "y_um": np.asarray(y_um, dtype=np.float64),
            "unique_id": np.zeros(len(x_um), dtype=np.int32),
        }
    )
    n_frames = len(track_data)
    dt = float(
        time_between_frames
        if time_between_frames is not None
        else config.TIME_BETWEEN_FRAMES
    )
    if n_frames < 3:
        return float("nan")

    lag_times = np.arange(1, n_frames) * dt
    msd_values = np.zeros(len(lag_times))
    for lag in range(1, len(lag_times) + 1):
        displacements = (
            track_data[["x_um", "y_um"]].iloc[lag:].values
            - track_data[["x_um", "y_um"]].iloc[:-lag].values
        ) ** 2
        msd_values[lag - 1] = np.mean(np.sum(displacements, axis=1))

    if len(lag_times) < 3:
        return float("nan")

    log_obs = np.log10(msd_values)

    try:
        popt, _pcov = scipy.optimize.curve_fit(msd_model, lag_times, msd_values)
        D, alpha = popt[0], popt[1]
        predicted = msd_model(lag_times, D, alpha)
    except (RuntimeError, ValueError):
        return float("nan")

    log_pred = np.log10(predicted)
    ss_res = np.sum((log_obs - log_pred) ** 2)
    ss_tot = np.sum((log_obs - np.mean(log_obs)) ** 2)
    chosen_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    threshold = (
        anomalous_r2_threshold
        if (alpha < 1 - tolerance or alpha > 1 + tolerance)
        else r2_threshold
    )

    if chosen_r2 < threshold and bad_fit_strategy == "flag":
        # Still return alpha (flag path in original returns alpha anyway)
        pass
    elif chosen_r2 < threshold:
        return float("nan")

    return float(alpha)


def compute_four_window_scalars_numpy(
    x_um: np.ndarray,
    y_um: np.ndarray,
    *,
    time_between_frames: float | None = None,
) -> np.ndarray:
    """
    Shape ``(4,)`` order:
    ``[self_intersections, cum_displacement_um, anomalous_exponent, avg_speed_um_s]``
    (same column order as ``WINDOW_SCALAR_FEATURE_KEYS_DEFAULT`` in ``transformer.py``).
    """
    x_um = np.asarray(x_um, dtype=np.float64)
    y_um = np.asarray(y_um, dtype=np.float64)
    inter = float(compute_self_intersections_count(x_um, y_um))
    cum_d = compute_cum_displacement_um(x_um, y_um)
    alpha = compute_anomalous_exponent_msd_r2_threshold(
        x_um, y_um, time_between_frames=time_between_frames
    )
    if np.isnan(alpha):
        alpha = 0.0
    vavg = compute_avg_speed_um_s(x_um, y_um, time_between_frames=time_between_frames)
    return np.array([inter, cum_d, alpha, vavg], dtype=np.float32)


def broadcast_four_to_time(
    motion_t3: np.ndarray,
    scalars_z: np.ndarray,
) -> np.ndarray:
    """``(T, 3)`` + z-scored ``(4,)`` → ``(T, 7)``."""
    t = motion_t3.shape[0]
    s = np.asarray(scalars_z, dtype=np.float32).reshape(4)
    tail = np.broadcast_to(s, (t, 4))
    return np.concatenate([motion_t3.astype(np.float32), tail], axis=1)


def zscore_numpy(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.maximum(np.asarray(std, dtype=np.float64), 1e-6)
    return ((np.asarray(x, dtype=np.float64) - mean) / std).astype(np.float32)


def precompute_scalar_mean_std_recompute(
    samples: list[dict],
    *,
    time_between_frames: float | None = None,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One pass over training samples (``features`` = (T,3), ``n_valid_frames``) to compute
    mean/std of the four **raw** scalars (before z-score), for ``recompute`` mode.

    Parameters
    ----------
    show_progress
        If True, show a tqdm bar over training windows (can disable for non-interactive logs).
    """
    rows = []
    it = samples
    if show_progress:
        it = tqdm(
            samples,
            desc="Precompute scalar mean/std (recompute, train)",
            unit="win",
            leave=True,
        )
    for s in it:
        f = np.asarray(s["features"], dtype=np.float32)
        if f.ndim != 2 or f.shape[1] < 3:
            continue
        nv = int(s.get("n_valid_frames", f.shape[0]))
        dx, dy = f[:, 0], f[:, 1]
        x_um, y_um = positions_from_stored_deltas(dx, dy, nv)
        if len(x_um) < 2:
            continue
        raw = compute_four_window_scalars_numpy(
            x_um, y_um, time_between_frames=time_between_frames
        )
        rows.append(raw)
    if len(rows) < 2:
        raise ValueError("Need at least 2 valid training windows for scalar statistics.")
    mat = np.stack(rows, axis=0)
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    std = np.maximum(std, 1e-6).astype(np.float32)
    return mean.astype(np.float32), std.astype(np.float32)
