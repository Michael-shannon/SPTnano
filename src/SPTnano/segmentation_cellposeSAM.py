"""
Cellpose-SAM projection segmentation helpers.

This module is intentionally separate from the existing track-footprint contour
workflow. It segments max/sum/mean raw movie projections with Cellpose-SAM,
stores one row per contour, and maps track ``unique_id`` values back to contour
IDs by testing track coordinates against the saved contour polygons.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import skimage.io as skio
from matplotlib.collections import LineCollection
from skimage import measure


def _as_pandas_copy(df):
    if isinstance(df, pl.DataFrame):
        try:
            return df.to_pandas()
        except ModuleNotFoundError as exc:
            if exc.name != "pyarrow":
                raise
            return pd.DataFrame(df.to_dicts())
    return df.copy()


def _pandas_to_polars(df):
    try:
        return pl.from_pandas(df)
    except ModuleNotFoundError as exc:
        if exc.name != "pyarrow":
            raise
        return pl.DataFrame(df.to_dict(orient="list"))


def _single_value_from_df(df, column, default=None):
    if column not in df.columns or df.empty:
        return default
    values = pd.Series(df[column]).dropna().unique()
    if len(values) == 0:
        return default
    return values[0]


def select_single_cell_df(df, *, filename, group_col="filename"):
    """
    Return rows for one filename while preserving pandas/polars input type.

    This local helper keeps the Cellpose-SAM workflow independent of the
    visualization module, which may not import in minimal Cellpose environments.
    """
    if isinstance(df, pl.DataFrame):
        if group_col not in df.columns:
            raise ValueError(f"Column {group_col!r} not found")
        result = df.filter(pl.col(group_col) == filename)
        if result.is_empty():
            raise ValueError(f"No rows found for {group_col}={filename!r}")
        return result

    if group_col not in df.columns:
        raise ValueError(f"Column {group_col!r} not found")
    result = df[df[group_col] == filename].copy()
    if result.empty:
        raise ValueError(f"No rows found for {group_col}={filename!r}")
    return result


def resolve_raw_movie_path(single_cell_df, master_dir=None, filename=None, condition=None):
    """
    Resolve the raw TIFF path from a selected cell dataframe.
    """
    df = _as_pandas_copy(single_cell_df)
    filename = filename or _single_value_from_df(df, "filename")
    condition = condition or _single_value_from_df(df, "condition")
    if filename is None:
        raise ValueError("Could not infer filename from dataframe")

    image_filename = str(filename).replace("_tracked", "") + ".tif"
    if master_dir is None:
        master_dir = _single_value_from_df(df, "foldername")
    if master_dir is None:
        raise ValueError("master_dir was not provided and foldername was not found")

    candidates = []
    if condition is not None:
        candidates.append(os.path.join(master_dir, "data", str(condition), image_filename))
        candidates.append(os.path.join(master_dir, str(condition), image_filename))
    candidates.append(os.path.join(master_dir, "data", image_filename))
    candidates.append(os.path.join(master_dir, image_filename))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def load_raw_frame_indices(raw_movie_path, frame_indices):
    """
    Load only requested TIFF frames when possible.
    """
    frame_indices = np.asarray(frame_indices, dtype=int)
    if frame_indices.size == 0:
        raise ValueError("No frame indices requested")

    try:
        import tifffile

        with tifffile.TiffFile(raw_movie_path) as tif:
            image = tif.asarray(key=frame_indices.tolist())
    except Exception:
        full_image = skio.imread(raw_movie_path)
        image = full_image[frame_indices]

    image = np.asarray(image)
    if image.ndim == 2 and frame_indices.size > 1:
        image = image[np.newaxis, ...]
    return image


def make_raw_projection(
    single_cell_df,
    *,
    raw_movie_path=None,
    master_dir=None,
    frame_range=None,
    projection="max",
    max_frames=None,
):
    """
    Make a raw image projection for one cell/movie.

    Parameters
    ----------
    projection:
        ``"max"``, ``"sum"``, or ``"mean"``.
    max_frames:
        If not None, sample this many frames evenly across the requested range.
    """
    df = _as_pandas_copy(single_cell_df)
    raw_movie_path = raw_movie_path or resolve_raw_movie_path(df, master_dir=master_dir)

    if frame_range is None:
        if "frame" not in df.columns:
            raise ValueError("frame_range is required when dataframe has no frame column")
        frame_start = int(df["frame"].min())
        frame_end = int(df["frame"].max())
    else:
        frame_start, frame_end = map(int, frame_range)

    frame_indices = np.arange(frame_start, frame_end + 1)
    if max_frames is not None and len(frame_indices) > int(max_frames):
        frame_indices = np.linspace(frame_start, frame_end, int(max_frames), dtype=int)

    stack = load_raw_frame_indices(raw_movie_path, frame_indices)
    if stack.ndim == 2:
        image = stack
    elif projection == "max":
        image = np.nanmax(stack, axis=0)
    elif projection == "sum":
        image = np.nansum(stack, axis=0)
    elif projection == "mean":
        image = np.nanmean(stack, axis=0)
    else:
        raise ValueError("projection must be 'max', 'sum', or 'mean'")

    return {
        "image": image,
        "raw_movie_path": raw_movie_path,
        "frame_range": (frame_start, frame_end),
        "frame_indices": frame_indices,
        "projection": projection,
    }


def normalize_image(image, contrast_limits=None, invert=False):
    image = image.astype(float, copy=False)
    if contrast_limits is None:
        lower = np.nanmin(image)
        upper = np.nanmax(image)
    else:
        lower, upper = contrast_limits
    if upper <= lower:
        norm = np.zeros_like(image, dtype=float)
    else:
        norm = np.clip((image - lower) / (upper - lower), 0, 1)
    if invert:
        norm = 1 - norm
    return norm


def load_cellpose_sam_model(*, gpu=True, pretrained_model="cpsam", **model_kwargs):
    """
    Load Cellpose-SAM from Cellpose 4.x.

    Cellpose is imported lazily so this module can be imported in environments
    that do not have Cellpose installed.
    """
    try:
        from cellpose import models
    except ImportError as exc:
        raise ImportError(
            "Cellpose is not installed. Create/activate the separate "
            "cellpose-SAM environment before running segmentation."
        ) from exc

    return models.CellposeModel(
        gpu=bool(gpu),
        pretrained_model=pretrained_model,
        **model_kwargs,
    )


def run_cellpose_sam_on_projection(
    image,
    *,
    model=None,
    gpu=True,
    pretrained_model="cpsam",
    diameter=None,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    normalize=True,
    invert=False,
    contrast_limits=None,
    **eval_kwargs,
):
    """
    Segment one 2D projection with Cellpose-SAM.
    """
    if model is None:
        model = load_cellpose_sam_model(gpu=gpu, pretrained_model=pretrained_model)

    eval_image = image
    if normalize or invert or contrast_limits is not None:
        eval_image = normalize_image(
            image,
            contrast_limits=contrast_limits,
            invert=invert,
        )

    eval_result = model.eval(
        eval_image,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        **eval_kwargs,
    )
    if len(eval_result) == 4:
        masks, flows, styles, diams = eval_result
    elif len(eval_result) == 3:
        masks, flows, styles = eval_result
        diams = diameter
    else:
        raise ValueError(
            "Unexpected CellposeModel.eval return signature: "
            f"expected 3 or 4 values, got {len(eval_result)}"
        )
    return {
        "masks": np.asarray(masks),
        "flows": flows,
        "styles": styles,
        "diams": diams,
        "model": model,
        "eval_image": eval_image,
    }


def _polygon_area_perimeter(contour):
    x = contour[:, 0]
    y = contour[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    perimeter = float(np.sum(np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))))
    return float(area), perimeter


def masks_to_contours_dataframe(
    masks,
    *,
    filename,
    foldername=None,
    metadata=None,
    max_contours=2,
    min_area_pixels=100,
    pixel_size_um=None,
    contour_id_start=1,
):
    """
    Convert a labeled Cellpose mask image to a one-row-per-contour dataframe.
    """
    masks = np.asarray(masks)
    metadata = dict(metadata or {})
    labels = [label for label in np.unique(masks) if label != 0]
    components = []
    for label in labels:
        mask = masks == label
        area_pixels = int(mask.sum())
        if area_pixels < int(min_area_pixels):
            continue
        props = measure.regionprops(mask.astype(np.uint8))
        if not props:
            continue
        contours = measure.find_contours(mask.astype(float), level=0.5)
        if not contours:
            continue
        contour_rc = max(contours, key=len)
        # skimage gives row, col. Store as x, y.
        contour_xy = np.column_stack([contour_rc[:, 1], contour_rc[:, 0]])
        polygon_area, perimeter_pixels = _polygon_area_perimeter(contour_xy)
        centroid_y, centroid_x = props[0].centroid
        components.append(
            {
                "cellpose_label": int(label),
                "mask_area_pixels": area_pixels,
                "polygon_area_pixels": polygon_area,
                "perimeter_pixels": perimeter_pixels,
                "centroid_x": float(centroid_x),
                "centroid_y": float(centroid_y),
                "contour_x": contour_xy[:, 0].astype(float).tolist(),
                "contour_y": contour_xy[:, 1].astype(float).tolist(),
                "n_contour_points": int(len(contour_xy)),
            }
        )

    components = sorted(
        components,
        key=lambda row: row["mask_area_pixels"],
        reverse=True,
    )[: int(max_contours)]

    rows = []
    for offset, row in enumerate(components):
        contour_id = int(contour_id_start) + offset
        out = {
            "filename": filename,
            "foldername": foldername,
            "contour_id": contour_id,
            "segmentation_method": "cellpose_sam_projection",
            **row,
            **metadata,
        }
        if pixel_size_um is not None:
            out["mask_area_um2"] = row["mask_area_pixels"] * float(pixel_size_um) ** 2
            out["polygon_area_um2"] = row["polygon_area_pixels"] * float(pixel_size_um) ** 2
            out["perimeter_um"] = row["perimeter_pixels"] * float(pixel_size_um)
        rows.append(out)
    return pd.DataFrame(rows)


def assign_unique_ids_to_contours(
    tracks_df,
    contours_df,
    *,
    group_col="filename",
    track_id_col="unique_id",
    x_col="x",
    y_col="y",
    contour_id_col="contour_id",
    min_points_inside=1,
    min_fraction_inside=0.5,
):
    """
    Assign each track unique_id to the contour containing most of its points.
    """
    tracks = _as_pandas_copy(tracks_df)
    contours = _as_pandas_copy(contours_df)
    for col in [group_col, track_id_col, x_col, y_col]:
        if col not in tracks.columns:
            raise ValueError(f"Column {col!r} not found in tracks_df")
    for col in [group_col, contour_id_col, "contour_x", "contour_y"]:
        if col not in contours.columns:
            raise ValueError(f"Column {col!r} not found in contours_df")

    rows = []
    for filename, cell_tracks in tracks.groupby(group_col):
        cell_contours = contours[contours[group_col] == filename]
        if cell_contours.empty:
            continue
        for unique_id, track in cell_tracks.groupby(track_id_col):
            points = track[[x_col, y_col]].dropna().to_numpy()
            if len(points) == 0:
                continue
            best = None
            for _, contour_row in cell_contours.iterrows():
                polygon = np.column_stack(
                    [
                        np.asarray(contour_row["contour_x"], dtype=float),
                        np.asarray(contour_row["contour_y"], dtype=float),
                    ]
                )
                inside = measure.points_in_poly(points, polygon)
                n_inside = int(inside.sum())
                fraction_inside = n_inside / len(points)
                candidate = {
                    group_col: filename,
                    track_id_col: unique_id,
                    contour_id_col: int(contour_row[contour_id_col]),
                    "n_points_total": int(len(points)),
                    "n_points_inside": n_inside,
                    "fraction_inside": float(fraction_inside),
                }
                if best is None or candidate["n_points_inside"] > best["n_points_inside"]:
                    best = candidate
            if best is None:
                continue
            if best["n_points_inside"] < int(min_points_inside):
                best[contour_id_col] = None
            elif best["fraction_inside"] < float(min_fraction_inside):
                best[contour_id_col] = None
            rows.append(best)
    return pd.DataFrame(rows)


def add_cellpose_contour_ids_to_tracks(
    df,
    mapping_df,
    *,
    group_col="filename",
    track_id_col="unique_id",
    contour_id_col="cellpose_contour_id",
    source_contour_id_col="contour_id",
):
    """
    Join a unique_id-to-contour mapping onto instant/windowed dataframes.
    """
    input_is_polars = isinstance(df, pl.DataFrame)
    mapping = _as_pandas_copy(mapping_df)[
        [group_col, track_id_col, source_contour_id_col]
    ].rename(columns={source_contour_id_col: contour_id_col})

    if input_is_polars:
        result = df.join(_pandas_to_polars(mapping), on=[group_col, track_id_col], how="left")
        return result

    result = _as_pandas_copy(df)
    return result.merge(mapping, on=[group_col, track_id_col], how="left")


def segment_cell_projection_to_contours(
    single_cell_df,
    *,
    model=None,
    raw_movie_path=None,
    master_dir=None,
    frame_range=None,
    projection="max",
    max_frames=None,
    max_contours=2,
    min_area_pixels=100,
    pixel_size_um=None,
    gpu=True,
    pretrained_model="cpsam",
    diameter=None,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    normalize=True,
    invert=False,
    contrast_limits=None,
    metadata_cols=None,
):
    """
    Run the full projection -> Cellpose-SAM -> contour dataframe path for one cell.
    """
    cell_df = _as_pandas_copy(single_cell_df)
    filename = _single_value_from_df(cell_df, "filename")
    foldername = _single_value_from_df(cell_df, "foldername")
    metadata = {}
    for col in metadata_cols or []:
        metadata[col] = _single_value_from_df(cell_df, col)

    projection_result = make_raw_projection(
        cell_df,
        raw_movie_path=raw_movie_path,
        master_dir=master_dir,
        frame_range=frame_range,
        projection=projection,
        max_frames=max_frames,
    )
    segmentation_result = run_cellpose_sam_on_projection(
        projection_result["image"],
        model=model,
        gpu=gpu,
        pretrained_model=pretrained_model,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize=normalize,
        invert=invert,
        contrast_limits=contrast_limits,
    )
    contours_df = masks_to_contours_dataframe(
        segmentation_result["masks"],
        filename=filename,
        foldername=foldername,
        metadata={
            **metadata,
            "projection": projection,
            "projection_frame_start": projection_result["frame_range"][0],
            "projection_frame_end": projection_result["frame_range"][1],
            "cellpose_diameter": diameter,
            "cellpose_flow_threshold": flow_threshold,
            "cellpose_cellprob_threshold": cellprob_threshold,
            "cellpose_pretrained_model": pretrained_model,
        },
        max_contours=max_contours,
        min_area_pixels=min_area_pixels,
        pixel_size_um=pixel_size_um,
    )
    return {
        "projection": projection_result,
        "segmentation": segmentation_result,
        "contours_df": contours_df,
        "model": segmentation_result["model"],
    }


def segment_cells_dataframe(
    tracks_df,
    *,
    model=None,
    group_col="filename",
    selection_method=None,
    n_cells=None,
    track_id_col="unique_id",
    random_state=42,
    return_backend="polars",
    **segment_kwargs,
):
    """
    Segment projections for many cells and return one row per contour.
    """
    df = _as_pandas_copy(tracks_df)
    if group_col not in df.columns:
        raise ValueError(f"Column {group_col!r} not found")

    summary = (
        df.groupby(group_col)
        .agg(
            n_datapoints=(group_col, "size"),
            n_tracks=(track_id_col, "nunique")
            if track_id_col in df.columns
            else (group_col, "size"),
        )
        .reset_index()
    )
    if selection_method == "most_tracks":
        summary = summary.sort_values(["n_tracks", "n_datapoints"], ascending=False)
    elif selection_method == "most_points":
        summary = summary.sort_values(["n_datapoints", "n_tracks"], ascending=False)
    elif selection_method == "random":
        summary = summary.sample(frac=1, random_state=random_state).reset_index(drop=True)
    elif selection_method is not None:
        raise ValueError(
            "selection_method must be None, 'most_tracks', 'most_points', or 'random'"
        )
    if n_cells is not None:
        summary = summary.head(int(n_cells))

    rows = []
    for filename in summary[group_col].tolist():
        cell_df = df[df[group_col] == filename].copy()
        result = segment_cell_projection_to_contours(
            cell_df,
            model=model,
            **segment_kwargs,
        )
        if not result["contours_df"].empty:
            rows.append(result["contours_df"])
        model = result["model"]

    contours = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if return_backend == "pandas":
        return contours
    if return_backend == "polars":
        return _pandas_to_polars(contours)
    raise ValueError("return_backend must be 'polars' or 'pandas'")


def plot_cellpose_segmentation_preview(
    projection,
    masks,
    contours_df=None,
    *,
    tracks_df=None,
    track_id_col="unique_id",
    x_col="x",
    y_col="y",
    invert_display=False,
    cmap="gray",
    figsize=(5, 5),
    dpi=150,
):
    """
    Preview projection, Cellpose masks, contours, and optional tracks.
    """
    image = normalize_image(projection, invert=invert_display)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(image, cmap=cmap, origin="upper")
    masked = np.ma.masked_where(np.asarray(masks) == 0, masks)
    ax.imshow(masked, alpha=0.25, cmap="tab20", origin="upper")

    if contours_df is not None and len(contours_df) > 0:
        contour_pdf = _as_pandas_copy(contours_df)
        for _, row in contour_pdf.iterrows():
            contour = np.column_stack(
                [
                    np.asarray(row["contour_x"], dtype=float),
                    np.asarray(row["contour_y"], dtype=float),
                ]
            )
            ax.plot(contour[:, 0], contour[:, 1], lw=1.5, color="yellow")
            ax.text(
                row["centroid_x"],
                row["centroid_y"],
                str(row["contour_id"]),
                color="yellow",
                weight="bold",
                ha="center",
                va="center",
            )

    if tracks_df is not None:
        tracks = _as_pandas_copy(tracks_df)
        if {track_id_col, x_col, y_col}.issubset(tracks.columns):
            segments = []
            for _, track in tracks.groupby(track_id_col):
                if len(track) >= 2:
                    segments.append(track[[x_col, y_col]].to_numpy())
            if segments:
                ax.add_collection(
                    LineCollection(
                        segments,
                        colors="cyan",
                        linewidths=0.5,
                        alpha=0.5,
                    )
                )

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    plt.tight_layout()
    return {"fig": fig, "ax": ax}


def plot_cellpose_contour_gallery(
    tracks_df,
    contours_df,
    *,
    n_cells=16,
    selection_method="most_tracks",
    group_col="filename",
    track_id_col="unique_id",
    x_col="x",
    y_col="y",
    frame_col="frame",
    n_cols=4,
    figsize_per_cell=(2.5, 2.5),
    show_raw_projection=True,
    raw_projection="max",
    raw_frame_range=None,
    raw_max_frames=250,
    raw_contrast_limits=None,
    invert_raw=False,
    raw_cmap="gray",
    show_tracks=True,
    track_color="cyan",
    track_line_width=0.4,
    track_alpha=0.55,
    contour_id_col="contour_id",
    contour_color="yellow",
    contour_line_width=1.1,
    show_contour_ids=True,
    contour_id_fontsize=8,
    show_panel_numbers=True,
    panel_number_start=1,
    dpi=150,
    save_path=None,
    show_plot=True,
):
    """
    Multi-subplot QC gallery for Cellpose-SAM contours with optional raw underlay
    and overlaid tracks.
    """
    tracks = _as_pandas_copy(tracks_df)
    contours = _as_pandas_copy(contours_df)
    if contours.empty:
        raise ValueError("contours_df is empty")
    for col in [group_col, contour_id_col, "contour_x", "contour_y"]:
        if col not in contours.columns:
            raise ValueError(f"Column {col!r} not found in contours_df")
    for col in [group_col, track_id_col, x_col, y_col]:
        if col not in tracks.columns:
            raise ValueError(f"Column {col!r} not found in tracks_df")

    summary = (
        tracks.groupby(group_col)
        .agg(
            n_datapoints=(group_col, "size"),
            n_tracks=(track_id_col, "nunique"),
        )
        .reset_index()
    )
    if selection_method == "most_tracks":
        summary = summary.sort_values(["n_tracks", "n_datapoints"], ascending=False)
    elif selection_method == "most_points":
        summary = summary.sort_values(["n_datapoints", "n_tracks"], ascending=False)
    elif selection_method == "random":
        summary = summary.sample(frac=1, random_state=42)
    elif selection_method is not None:
        raise ValueError(
            "selection_method must be None, 'most_tracks', 'most_points', or 'random'"
        )

    available_cells = set(contours[group_col].dropna().unique())
    selected_cells = [cell for cell in summary[group_col].tolist() if cell in available_cells]
    if n_cells is not None:
        selected_cells = selected_cells[: int(n_cells)]
    if not selected_cells:
        raise ValueError("No selected cells have Cellpose contours")

    n = len(selected_cells)
    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        squeeze=False,
        dpi=dpi,
    )
    for ax in axes.ravel():
        ax.axis("off")

    lookup_rows = []
    for idx, filename in enumerate(selected_cells):
        ax = axes.ravel()[idx]
        cell_tracks = tracks[tracks[group_col] == filename].copy()
        cell_contours = contours[contours[group_col] == filename].copy()
        panel_number = int(panel_number_start) + idx

        if raw_frame_range is not None and frame_col in cell_tracks.columns:
            frame_start, frame_end = map(int, raw_frame_range)
            cell_tracks_for_raw = cell_tracks[
                (cell_tracks[frame_col] >= frame_start)
                & (cell_tracks[frame_col] <= frame_end)
            ]
        else:
            cell_tracks_for_raw = cell_tracks

        if show_raw_projection and not cell_tracks_for_raw.empty:
            try:
                projection_result = make_raw_projection(
                    cell_tracks_for_raw,
                    frame_range=raw_frame_range,
                    projection=raw_projection,
                    max_frames=raw_max_frames,
                )
                ax.imshow(
                    normalize_image(
                        projection_result["image"],
                        contrast_limits=raw_contrast_limits,
                        invert=invert_raw,
                    ),
                    cmap=raw_cmap,
                    origin="upper",
                    zorder=0,
                )
            except Exception as exc:
                ax.text(
                    0.5,
                    0.5,
                    f"raw load failed\\n{exc}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=6,
                )

        if show_tracks and not cell_tracks.empty:
            if raw_frame_range is not None and frame_col in cell_tracks.columns:
                frame_start, frame_end = map(int, raw_frame_range)
                cell_tracks = cell_tracks[
                    (cell_tracks[frame_col] >= frame_start)
                    & (cell_tracks[frame_col] <= frame_end)
                ]
            segments = []
            sort_col = frame_col if frame_col in cell_tracks.columns else x_col
            for _, track in cell_tracks.sort_values(sort_col).groupby(track_id_col):
                if len(track) >= 2:
                    segments.append(track[[x_col, y_col]].to_numpy())
            if segments:
                ax.add_collection(
                    LineCollection(
                        segments,
                        colors=track_color,
                        linewidths=track_line_width,
                        alpha=track_alpha,
                        zorder=1,
                    )
                )

        for _, contour_row in cell_contours.iterrows():
            contour = np.column_stack(
                [
                    np.asarray(contour_row["contour_x"], dtype=float),
                    np.asarray(contour_row["contour_y"], dtype=float),
                ]
            )
            ax.plot(
                contour[:, 0],
                contour[:, 1],
                color=contour_color,
                lw=contour_line_width,
                zorder=2,
            )
            if show_contour_ids:
                label_x = contour_row.get("centroid_x", np.nanmean(contour[:, 0]))
                label_y = contour_row.get("centroid_y", np.nanmean(contour[:, 1]))
                ax.text(
                    label_x,
                    label_y,
                    str(contour_row[contour_id_col]),
                    color=contour_color,
                    weight="bold",
                    fontsize=contour_id_fontsize,
                    ha="center",
                    va="center",
                    zorder=3,
                )

        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.axis("off")
        if show_panel_numbers:
            ax.set_title(str(panel_number), fontsize=8)
        else:
            ax.set_title(str(filename), fontsize=7)

        lookup_rows.append(
            {
                "gallery_number": panel_number,
                "filename": filename,
                "foldername": _single_value_from_df(cell_tracks, "foldername", ""),
                "n_contours": int(cell_contours[contour_id_col].nunique()),
                "n_tracks": int(cell_tracks[track_id_col].nunique())
                if track_id_col in cell_tracks.columns
                else np.nan,
            }
        )

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", transparent=True)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        "fig": fig,
        "axes": axes,
        "selected_cells": selected_cells,
        "cell_lookup_df": pd.DataFrame(lookup_rows),
    }


def save_contours_dataframe(contours_df, output_path):
    """
    Save contours as parquet or csv based on file extension.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(contours_df, pl.DataFrame):
        if output_path.suffix.lower() == ".csv":
            contours_df.write_csv(output_path)
        else:
            contours_df.write_parquet(output_path)
        return output_path

    if output_path.suffix.lower() == ".csv":
        contours_df.to_csv(output_path, index=False)
    else:
        _pandas_to_polars(contours_df).write_parquet(output_path)
    return output_path

