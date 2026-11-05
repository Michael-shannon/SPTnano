"""Spatial metrics measured inside cell contours."""

from __future__ import annotations

import ast
import warnings
from dataclasses import dataclass
from typing import Iterable

import matplotlib.path as mpltPath
import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm


def _to_pandas(df):
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    if isinstance(df, pd.DataFrame):
        return df.copy()
    raise TypeError("Expected a pandas.DataFrame or polars.DataFrame.")


def _values_to_array(values):
    if values is None:
        return np.array([], dtype=float)
    if isinstance(values, str):
        try:
            values = ast.literal_eval(values)
        except (SyntaxError, ValueError):
            return np.array([], dtype=float)
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


@dataclass
class _ContainmentBackend:
    use_shapely: bool
    polygon: object | None
    path: mpltPath.Path


def _make_containment_backend(contour_x, contour_y):
    path = mpltPath.Path(np.column_stack([contour_x, contour_y]))
    try:
        from shapely.geometry import Polygon

        polygon = Polygon(np.column_stack([contour_x, contour_y]))
        if polygon.is_valid and not polygon.is_empty:
            return _ContainmentBackend(use_shapely=True, polygon=polygon, path=path)
    except Exception:
        pass
    return _ContainmentBackend(use_shapely=False, polygon=None, path=path)


def _box_is_inside_contour(
    backend: _ContainmentBackend,
    x0: float,
    y0: float,
    box_size_px: float,
    validation_points_per_side: int = 7,
):
    x1 = x0 + box_size_px
    y1 = y0 + box_size_px

    if backend.use_shapely:
        from shapely.geometry import box

        return bool(backend.polygon.covers(box(x0, y0, x1, y1)))

    # Conservative fallback: test a grid of points across the square, including
    # all edges/corners. Exact polygon/square containment needs Shapely.
    n = max(2, int(validation_points_per_side))
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    xx, yy = np.meshgrid(xs, ys)
    test_points = np.column_stack([xx.ravel(), yy.ravel()])
    return bool(backend.path.contains_points(test_points, radius=1e-9).all())


def _generate_inside_boxes(
    contour_x,
    contour_y,
    box_size_px,
    grid_offset_steps=1,
    validation_points_per_side=7,
):
    backend = _make_containment_backend(contour_x, contour_y)
    if not backend.use_shapely:
        warnings.warn(
            "Shapely is not available or polygon was invalid; using sampled box-containment checks. "
            "Increase validation_points_per_side for stricter checking.",
            RuntimeWarning,
            stacklevel=2,
        )

    min_x, max_x = np.nanmin(contour_x), np.nanmax(contour_x)
    min_y, max_y = np.nanmin(contour_y), np.nanmax(contour_y)
    if (max_x - min_x) < box_size_px or (max_y - min_y) < box_size_px:
        return []

    offset_steps = max(1, int(grid_offset_steps))
    offsets = np.linspace(0, box_size_px, offset_steps, endpoint=False)
    best_boxes = []

    for offset_x in offsets:
        for offset_y in offsets:
            boxes = []
            x_starts = np.arange(min_x + offset_x, max_x - box_size_px + 1e-9, box_size_px)
            y_starts = np.arange(min_y + offset_y, max_y - box_size_px + 1e-9, box_size_px)
            for x0 in x_starts:
                for y0 in y_starts:
                    if _box_is_inside_contour(
                        backend,
                        float(x0),
                        float(y0),
                        box_size_px,
                        validation_points_per_side=validation_points_per_side,
                    ):
                        boxes.append((float(x0), float(y0), float(x0 + box_size_px), float(y0 + box_size_px)))
            if len(boxes) > len(best_boxes):
                best_boxes = boxes

    return best_boxes


def measure_track_counts_in_contour_boxes(
    contours_df,
    points_df,
    box_size_px,
    contour_key_col=None,
    points_key_col=None,
    contour_x_col="contour_x",
    contour_y_col="contour_y",
    point_x_col="x",
    point_y_col="y",
    track_col="unique_id",
    frame_col="frame",
    metadata_cols=None,
    grid_offset_steps=1,
    validation_points_per_side=7,
    return_polars=True,
    show_progress=True,
):
    """
    Count points and unique tracks in non-overlapping boxes fully inside contours.

    Parameters
    ----------
    contours_df, points_df : pandas.DataFrame or polars.DataFrame
        Contour-level dataframe and point/frame-level dataframe.
    box_size_px : float
        Side length of each square box in pixel units.
    contour_key_col, points_key_col : str, optional
        Shared cell key. If omitted, the function uses cell_key when available
        in both dataframes, otherwise filename.
    grid_offset_steps : int
        Number of grid offsets to try along each axis. The offset that yields
        the most fully-contained non-overlapping boxes is used per contour.
    validation_points_per_side : int
        Used only when Shapely is unavailable. Higher values more strictly test
        that the whole box lies inside the contour.

    Returns
    -------
    tuple
        (box_metrics_df, cell_metrics_df)
    """
    contours_pd = _to_pandas(contours_df)
    points_pd = _to_pandas(points_df)

    if contour_key_col is None or points_key_col is None:
        if "cell_key" in contours_pd.columns and "cell_key" in points_pd.columns:
            contour_key_col = contour_key_col or "cell_key"
            points_key_col = points_key_col or "cell_key"
        else:
            contour_key_col = contour_key_col or "filename"
            points_key_col = points_key_col or "filename"

    required_contour_cols = [contour_key_col, contour_x_col, contour_y_col]
    required_point_cols = [points_key_col, point_x_col, point_y_col, track_col]
    missing_contour_cols = [col for col in required_contour_cols if col not in contours_pd.columns]
    missing_point_cols = [col for col in required_point_cols if col not in points_pd.columns]
    if missing_contour_cols:
        raise ValueError(f"Missing contour columns: {missing_contour_cols}")
    if missing_point_cols:
        raise ValueError(f"Missing point columns: {missing_point_cols}")

    if metadata_cols is None:
        metadata_cols = [
            col
            for col in [
                "filename",
                "foldername",
                "cell_key",
                "contour_id",
                "geno",
                "group",
                "mol",
                "type",
                "location",
                "replicate",
                "condition",
            ]
            if col in contours_pd.columns
        ]

    dedupe_cols = [points_key_col, track_col]
    if frame_col in points_pd.columns:
        dedupe_cols.append(frame_col)
    points_pd = points_pd.dropna(subset=[points_key_col, point_x_col, point_y_col]).drop_duplicates(subset=dedupe_cols)
    points_by_key = {key: group.copy() for key, group in points_pd.groupby(points_key_col, dropna=False)}

    box_records = []
    cell_records = []
    iterator = contours_pd.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(contours_pd), desc="Measuring contour box track density")

    for _, contour_row in iterator:
        cell_key = contour_row[contour_key_col]
        contour_x = _values_to_array(contour_row[contour_x_col])
        contour_y = _values_to_array(contour_row[contour_y_col])
        n_contour_points = min(len(contour_x), len(contour_y))
        if n_contour_points < 3:
            continue
        contour_x = contour_x[:n_contour_points]
        contour_y = contour_y[:n_contour_points]
        cell_points = points_by_key.get(cell_key, pd.DataFrame(columns=points_pd.columns))

        boxes = _generate_inside_boxes(
            contour_x,
            contour_y,
            box_size_px=float(box_size_px),
            grid_offset_steps=grid_offset_steps,
            validation_points_per_side=validation_points_per_side,
        )

        metadata = {col: contour_row[col] for col in metadata_cols if col in contour_row.index}
        point_counts = []
        track_counts = []
        tracks_seen = set()

        for box_idx, (x0, y0, x1, y1) in enumerate(boxes):
            if cell_points.empty:
                points_in_box = cell_points
            else:
                points_in_box = cell_points[
                    (cell_points[point_x_col] >= x0)
                    & (cell_points[point_x_col] < x1)
                    & (cell_points[point_y_col] >= y0)
                    & (cell_points[point_y_col] < y1)
                ]

            n_points = int(len(points_in_box))
            n_tracks = int(points_in_box[track_col].nunique()) if n_points else 0
            if n_points:
                tracks_seen.update(points_in_box[track_col].dropna().unique())
            point_counts.append(n_points)
            track_counts.append(n_tracks)

            box_area_px2 = float(box_size_px) ** 2
            box_records.append(
                {
                    **metadata,
                    "_cell_key": cell_key,
                    "box_id": box_idx,
                    "box_size_px": float(box_size_px),
                    "box_x_min": x0,
                    "box_y_min": y0,
                    "box_x_max": x1,
                    "box_y_max": y1,
                    "box_area_px2": box_area_px2,
                    "n_points_in_box": n_points,
                    "n_tracks_in_box": n_tracks,
                    "point_density_per_px2": n_points / box_area_px2,
                    "track_density_per_px2": n_tracks / box_area_px2,
                }
            )

        n_boxes = len(boxes)
        box_area_px2 = float(box_size_px) ** 2
        cell_records.append(
            {
                **metadata,
                "_cell_key": cell_key,
                "box_size_px": float(box_size_px),
                "n_boxes_inside_contour": n_boxes,
                "total_box_area_px2": n_boxes * box_area_px2,
                "total_points_in_boxes": int(np.sum(point_counts)) if point_counts else 0,
                "total_track_hits_in_boxes": int(np.sum(track_counts)) if track_counts else 0,
                "unique_tracks_in_any_box": len(tracks_seen),
                "mean_points_per_box": float(np.mean(point_counts)) if point_counts else np.nan,
                "median_points_per_box": float(np.median(point_counts)) if point_counts else np.nan,
                "mean_tracks_per_box": float(np.mean(track_counts)) if track_counts else np.nan,
                "median_tracks_per_box": float(np.median(track_counts)) if track_counts else np.nan,
                "mean_point_density_per_px2": float(np.mean(point_counts) / box_area_px2) if point_counts else np.nan,
                "mean_track_density_per_px2": float(np.mean(track_counts) / box_area_px2) if track_counts else np.nan,
                "unique_track_density_per_total_box_area_px2": (
                    len(tracks_seen) / (n_boxes * box_area_px2) if n_boxes else np.nan
                ),
            }
        )

    box_metrics_df = pd.DataFrame(box_records)
    cell_metrics_df = pd.DataFrame(cell_records)
    if return_polars:
        return pl.from_pandas(box_metrics_df), pl.from_pandas(cell_metrics_df)
    return box_metrics_df, cell_metrics_df
