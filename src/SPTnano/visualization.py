import io
import math
import os
import random
import re

# from matplotlib import path as mpltPath
import colorcet
import datashader as ds
import datashader.transfer_functions as tf
import imageio
import matplotlib as mpl
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
import pims
import seaborn as sns
import skimage.io as skio
import xarray as xr  # for converting arrays to DataArray
from .config import FEATURES2  # import additional features from your config
from datashader.reductions import mean
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, is_color_like
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from napari.utils.colormaps import Colormap, ensure_colormap
from napari_animation import Animation
from scipy.interpolate import splev, splprep
from scipy.ndimage import gaussian_filter
from scipy.stats import linregress, sem
from skimage import img_as_float
from skimage.io import imread

# import minmaxscaler
from sklearn.preprocessing import MinMaxScaler

from . import config
from .helper_scripts import *

# Set up fonts and SVG text handling so that text remains editable in Illustrator.
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "sans-serif"
# Try Helvetica; if not available, fallback to Arial
import matplotlib.font_manager as fm

if any("Helvetica" in f.name for f in fm.fontManager.ttflist):
    mpl.rcParams["font.sans-serif"] = ["Helvetica"]
else:
    mpl.rcParams["font.sans-serif"] = ["Arial"]


# Helper function to handle Location/location column case sensitivity
def _get_location_column(df):
    """
    Helper function to detect whether dataframe has 'Location' or 'location' column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check for location column

    Returns
    -------
    str
        The correct column name ('Location' or 'location')

    Raises
    ------
    ValueError
        If neither 'Location' nor 'location' column is found

    """
    if "Location" in df.columns:
        return "Location"
    elif "location" in df.columns:
        return "location"
    else:
        raise ValueError(
            "Neither 'Location' nor 'location' column found in dataframe. Available columns: "
            + str(list(df.columns))
        )


def overlay_tracks_with_movie(tracks_df, movie_path, colormap=None):
    # Load the raw movie
    frames = pims.open(movie_path)

    # Create a new folder to save the PNG images
    output_folder = os.path.splitext(movie_path)[0]
    os.makedirs(output_folder, exist_ok=True)

    if colormap:
        # Get the colormap if specified
        cmap = cm.get_cmap(colormap)
        # Get unique track IDs and assign colors from the colormap
        unique_tracks = tracks_df["unique_id"].unique()
        colors = {
            track_id: cmap(i / len(unique_tracks))
            for i, track_id in enumerate(unique_tracks)
        }
    else:
        # Use the default color cycle
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        unique_tracks = tracks_df["unique_id"].unique()
        colors = {
            track_id: color_cycle[i % len(color_cycle)]
            for i, track_id in enumerate(unique_tracks)
        }

    # Get the last frame for each track
    last_frame = tracks_df.groupby("unique_id")["frame"].max()

    # Iterate over each frame in the movie
    for frame_index, frame in enumerate(frames):
        # Debug: Print frame index and frame shape
        print(f"Processing frame {frame_index} with shape {frame.shape}")

        # Create a figure and axis with a larger size
        fig, ax = plt.subplots(figsize=(12, 12))

        # Display the current frame
        ax.imshow(frame, cmap="gray", origin="upper")

        # Set the plot limits to match the image dimensions
        ax.set_xlim(0, frame.shape[1])
        ax.set_ylim(frame.shape[0], 0)

        # Iterate over each track in the DataFrame
        for track_id, track in tracks_df.groupby("unique_id"):
            # Only plot the track if the current frame is less than or equal to the last frame of the track
            if frame_index <= last_frame[track_id]:
                # Get the x and y coordinates of the track for the current frame
                x = track.loc[track["frame"] <= frame_index, "x"]
                y = track.loc[track["frame"] <= frame_index, "y"]

                # Plot the track as a line with slightly thicker lines and consistent color
                ax.plot(
                    x,
                    y,
                    label=f"Track {track_id}",
                    linewidth=2.0,
                    color=colors[track_id],
                )

        # Remove the axis labels, ticks, and grid
        ax.axis("off")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Save the figure as a PNG image in the output folder
        output_path = os.path.join(output_folder, f"frame_{frame_index:04d}.png")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)

        # Close the figure to free up memory
        plt.close(fig)


def plot_histograms_seconds(traj_df, bins=100, coltoseparate="tracker", xlimit=None):
    """
    Plot histograms of track lengths in seconds for each tracker, with consistent binning.

    Parameters
    ----------
    traj_df : DataFrame
        DataFrame containing track data with columns 'tracker', 'unique_id', 'time_s_zeroed', and 'filename'.
    bins : int, optional
        Number of bins for the histogram. Default is 100.
    coltoseparate : str, optional
        Column to separate the data by. Default is 'tracker'.
    xlimit : float, optional
        Upper limit for the x-axis. Default is None.

    """
    plt.figure(figsize=(20, 12))
    size = 10
    multiplier = 2
    sns.set_context(
        "notebook",
        rc={"xtick.labelsize": size * multiplier, "ytick.labelsize": size * multiplier},
    )

    max_track_length = traj_df.groupby("unique_id")["time_s_zeroed"].max().max()
    bin_edges = np.linspace(0, max_track_length, bins + 1)

    for i, tracker in enumerate(traj_df[coltoseparate].unique()):
        subset = traj_df[traj_df[coltoseparate] == tracker]
        subsetvalues = subset.groupby("unique_id")["time_s_zeroed"].max()

        # Calculate percentage counts
        counts, _ = np.histogram(subsetvalues, bins=bin_edges)
        percentage_counts = (counts / counts.sum()) * 100

        # Plot histogram
        sns.histplot(
            subsetvalues,
            bins=bin_edges,
            kde=True,
            label=tracker,
            alpha=0.5,
            stat="percent",
        )

        subset_mean = subsetvalues.mean()
        subset_median = subsetvalues.median()
        subset_number_of_tracks = len(subset["unique_id"].unique())
        shift = i * 0.05
        plt.text(
            0.4,
            0.6 - shift,
            f"{tracker}: mean: {subset_mean:.2f} seconds from {subset_number_of_tracks} tracks",
            transform=plt.gca().transAxes,
            fontsize=10 * multiplier,
        )

    plt.xlabel("Track length (seconds)", fontsize=size * multiplier)
    plt.ylabel("Percentage", fontsize=size * multiplier)
    plt.legend(title="", fontsize=size * multiplier)
    ax = plt.gca()
    if xlimit is not None:
        ax.set_xlim(0, xlimit)
    else:
        ax.set_xlim(0, max_track_length)
    plt.show()


def plot_histograms(
    data_df,
    feature,
    bins=100,
    separate=None,
    xlimit=None,
    small_multiples=False,
    palette="colorblind",
    use_kde=False,
    kde_fill=True,
    show_plot=True,
    master_dir=None,
    tick_interval=5,
    average="mean",
    order=None,
    grid=False,
    background="white",
    transparent=False,
    condition_colors=None,
    line_color="black",
    font_size=9,
    showavg=True,
    export_format="png",
    return_svg=False,
    x_range=None,
    y_range=None,
    percentage=True,
    log_scale=False,
    log_base=10,
    alpha=1,
    log_axis_label="log",
    save_folder=None,
    figsize=(3, 3),
):
    """
    Modified function to allow removing KDE fill and moving the legend outside the plot.
    """
    if master_dir is None:
        master_dir = "plots"

    baseline_width = 3.0
    scale_factor = figsize[0] / baseline_width
    scaled_font = font_size * scale_factor
    plt.rcParams.update(
        {
            "font.size": scaled_font,
            "axes.titlesize": scaled_font,
            "axes.labelsize": scaled_font,
            "xtick.labelsize": scaled_font,
            "ytick.labelsize": scaled_font,
        }
    )

    if log_scale:
        new_feature = "log_" + feature
        if (data_df[feature] <= 0).any():
            raise ValueError(f"All values of {feature} must be positive for log scale.")
        if log_base == 10:
            data_df[new_feature] = np.log10(data_df[feature])
        elif log_base == 2:
            data_df[new_feature] = np.log2(data_df[feature])
        else:
            data_df[new_feature] = np.log(data_df[feature]) / np.log(log_base)
        feature_to_plot = new_feature
        x_label = f"log{log_base}({feature})" if log_axis_label != "actual" else feature
    else:
        feature_to_plot = feature
        x_label = feature

    figure_background = (
        "none" if transparent else background if is_color_like(background) else "white"
    )
    axis_background = figure_background

    if separate is not None:
        if not pd.api.types.is_categorical_dtype(data_df[separate]):
            data_df[separate] = data_df[separate].astype("category")
        if order is not None:
            data_df[separate] = pd.Categorical(
                data_df[separate], categories=order, ordered=True
            )
        unique_categories = data_df[separate].cat.categories
    else:
        unique_categories = [None]

    color_palette = sns.color_palette(palette, len(unique_categories))

    # # Create a color mapping for conditions based on the provided dictionary
    # color_mapping = {}
    # for i, category in enumerate(unique_categories):
    #     if category in condition_colors:
    #         color_mapping[category] = condition_colors[category]
    #     else:
    #         color_mapping[category] = color_palette[i % len(color_palette)]  # Assign fallback colors if not in the dict
    #
    # Ensure condition_colors is not None
    if condition_colors is None:
        condition_colors = {}

    color_mapping = {}
    for i, category in enumerate(unique_categories):
        if category in condition_colors:
            color_mapping[category] = condition_colors[category]
        else:
            color_mapping[category] = color_palette[i % len(color_palette)]

    if x_range is not None:
        global_lower_bound, global_upper_bound = x_range
    else:
        global_lower_bound = data_df[feature_to_plot].min()
        global_upper_bound = (
            xlimit if xlimit is not None else data_df[feature_to_plot].max()
        )

    # fig, ax = plt.subplots(figsize=figsize, facecolor=figure_background)
    if small_multiples and separate is not None:
        fig, axes = plt.subplots(
            len(unique_categories),
            1,
            figsize=(figsize[0], figsize[1] * len(unique_categories)),
            sharex=True,
            facecolor=figure_background,
        )
        if len(unique_categories) == 1:
            axes = [axes]
    else:
        fig, ax = plt.subplots(figsize=figsize, facecolor=figure_background)
        axes = [ax]

    for i, category in enumerate(unique_categories):
        ax = axes[i] if (small_multiples and separate) else axes[0]
        subsetvalues = (
            data_df[data_df[separate] == category][feature_to_plot]
            if category
            else data_df[feature_to_plot]
        )
        subsetvalues = subsetvalues[
            (subsetvalues >= global_lower_bound) & (subsetvalues <= global_upper_bound)
        ]

        ax.set_facecolor(axis_background)
        bin_edges = np.linspace(global_lower_bound, global_upper_bound, bins + 1)

        if use_kde:
            sns.kdeplot(
                subsetvalues,
                fill=kde_fill,
                ax=ax,
                color=color_mapping[category],
                linewidth=1.5,
                label=category,
                alpha=alpha,
            )
        else:
            counts, _ = np.histogram(subsetvalues, bins=bin_edges)
            if percentage:
                counts = 100 * counts / counts.sum() if counts.sum() > 0 else counts
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.bar(
                bin_centers,
                counts,
                width=np.diff(bin_edges),
                color=color_mapping[category],
                alpha=alpha,
                label=category,
            )

        avg_value = subsetvalues.mean() if average == "mean" else subsetvalues.median()
        annotation_value = (
            log_base**avg_value
            if log_scale and log_axis_label == "actual"
            else avg_value
        )

        if showavg:
            ax.axvline(avg_value, color=line_color, linestyle="--")

    ax.set_xlabel(x_label, fontsize=scaled_font, color=line_color)
    ax.set_ylabel(
        "Percentage" if percentage else "Count", fontsize=scaled_font, color=line_color
    )
    ax.tick_params(
        axis="both",
        which="both",
        color=line_color,
        labelcolor=line_color,
        labelsize=scaled_font,
    )

    if grid:
        ax.grid(
            True,
            linestyle="--",
            linewidth=0.5,
            color=line_color if not transparent else (0, 0, 0, 0.5),
            alpha=0.7,
            axis="y",
        )

    for spine in ax.spines.values():
        spine.set_edgecolor(line_color)
        if transparent:
            spine.set_alpha(0.9)

    ax.set_xlim(global_lower_bound, global_upper_bound)
    xticks = np.arange(
        global_lower_bound, global_upper_bound + tick_interval, tick_interval
    )
    ax.set_xticks(xticks)
    if y_range is not None:
        ax.set_ylim(y_range)
    if log_scale:
        ax.set_xscale("linear")
        if log_axis_label == "actual":
            formatter = FuncFormatter(lambda val, pos: f"{log_base ** val:.2g}")
            ax.xaxis.set_major_formatter(formatter)

    # legend = ax.legend(title=separate, fontsize=scaled_font, title_fontsize=scaled_font, loc='upper left', bbox_to_anchor=(1, 1))
    # plt.gca().add_artist(legend)
    if small_multiples:
        fig.legend(
            title=separate,
            fontsize=scaled_font,
            title_fontsize=scaled_font,
            loc="upper right",
            bbox_to_anchor=(1, 1),
        )
    else:
        legend = ax.legend(
            title=separate,
            fontsize=scaled_font,
            title_fontsize=scaled_font,
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )
        plt.gca().add_artist(legend)

    if save_folder is None:
        save_folder = os.path.join(master_dir, "plots", "histograms")
    os.makedirs(save_folder, exist_ok=True)
    ext = export_format.lower()
    if ext not in ["png", "svg"]:
        print("Invalid export format specified. Defaulting to 'png'.")
        ext = "png"
    out_filename = f"{feature}_histogram.{ext}"
    full_save_path = os.path.join(save_folder, out_filename)
    plt.savefig(
        full_save_path, bbox_inches="tight", transparent=transparent, format=ext
    )

    svg_data = None
    if ext == "svg":
        with open(full_save_path, encoding="utf-8") as f:
            svg_data = f.read()
        svg_data = re.sub(
            r'<clipPath id="[^"]*">.*?</clipPath>', "", svg_data, flags=re.DOTALL
        )
        svg_data = re.sub(r'\s*clip-path="url\([^)]*\)"', "", svg_data)
        svg_data = re.sub(r"<metadata>.*?</metadata>", "", svg_data, flags=re.DOTALL)
        svg_data = re.sub(r"<\?xml[^>]*\?>", "", svg_data, flags=re.DOTALL)
        svg_data = re.sub(r"<!DOCTYPE[^>]*>", "", svg_data, flags=re.DOTALL)
        svg_data = svg_data.strip()
        with open(full_save_path, "w", encoding="utf-8") as f:
            f.write(svg_data)

    if show_plot:
        plt.show()
    else:
        plt.close()

    if ext == "svg" and return_svg:
        return svg_data


def annotate_threshold_simple(data_df, feature, threshold, separate=None, sigfigs=3):
    """
    Adds an 'above_threshold' flag and computes per-category (or overall) % above/below.
    Returns (df_with_flag, percentages_dict).
    """
    df = data_df.copy()
    df["above_threshold"] = df[feature] > threshold
    if separate and separate in df.columns:
        percentages = {}
        for name, group in df.groupby(separate):
            total = len(group)
            above = int(group["above_threshold"].sum())
            below = total - above
            percentages[name] = {
                "above": round(100 * above / total, sigfigs) if total else 0,
                "below": round(100 * below / total, sigfigs) if total else 0,
            }
    else:
        total = len(df)
        above = int(df["above_threshold"].sum())
        below = total - above
        percentages = {
            "above": round(100 * above / total, sigfigs) if total else 0,
            "below": round(100 * below / total, sigfigs) if total else 0,
        }
    return df, percentages


def plot_histograms_threshold(
    data_df,
    feature,
    bins=100,
    separate=None,
    xlimit=None,
    small_multiples=False,
    palette="colorblind",
    use_kde=False,
    kde_fill=True,
    show_plot=True,
    master_dir=None,
    tick_interval=5,
    average="mean",
    order=None,
    grid=False,
    background="white",
    transparent=False,
    condition_colors=None,
    line_color="black",
    font_size=9,
    showavg=True,
    export_format="png",
    return_svg=False,
    x_range=None,
    y_range=None,
    percentage=True,
    log_scale=False,
    log_base=10,
    alpha=1,
    log_axis_label="log",
    save_folder=None,
    figsize=(3, 3),
    threshold=None,
):
    """
    Modified function to allow removing KDE fill, moving the legend,
    and now adding correct % above/below threshold annotations.
    """
    # ——— NEW: compute threshold flags & percentages ———
    if threshold is not None:
        data_df, threshold_pct = annotate_threshold_simple(
            data_df, feature, threshold, separate
        )
    else:
        threshold_pct = None

    # ——— existing master_dir default ———
    if master_dir is None:
        master_dir = "plots"

    # ——— existing font scaling ———
    baseline_width = 3.0
    scale_factor = figsize[0] / baseline_width
    scaled_font = font_size * scale_factor
    plt.rcParams.update(
        {
            "font.size": scaled_font,
            "axes.titlesize": scaled_font,
            "axes.labelsize": scaled_font,
            "xtick.labelsize": scaled_font,
            "ytick.labelsize": scaled_font,
        }
    )

    # ——— existing log‐scale handling ———
    if log_scale:
        new_feature = "log_" + feature
        if (data_df[feature] <= 0).any():
            raise ValueError(f"All values of {feature} must be positive for log scale.")
        if log_base == 10:
            data_df[new_feature] = np.log10(data_df[feature])
        elif log_base == 2:
            data_df[new_feature] = np.log2(data_df[feature])
        else:
            data_df[new_feature] = np.log(data_df[feature]) / np.log(log_base)
        feature_to_plot = new_feature
        x_label = f"log{log_base}({feature})" if log_axis_label != "actual" else feature
        # ——— NEW: threshold in log‐space ———
        threshold_plot = (
            (np.log(threshold) / np.log(log_base)) if threshold is not None else None
        )
    else:
        feature_to_plot = feature
        x_label = feature
        threshold_plot = threshold

    # ——— existing background colors ———
    figure_background = (
        "none" if transparent else background if is_color_like(background) else "white"
    )
    axis_background = figure_background

    # ——— existing category setup ———
    if separate is not None:
        if not pd.api.types.is_categorical_dtype(data_df[separate]):
            data_df[separate] = data_df[separate].astype("category")
        if order is not None:
            data_df[separate] = pd.Categorical(
                data_df[separate], categories=order, ordered=True
            )
        unique_categories = data_df[separate].cat.categories
    else:
        unique_categories = [None]

    # ——— existing color mapping ———
    color_palette = sns.color_palette(palette, len(unique_categories))
    if condition_colors is None:
        condition_colors = {}
    color_mapping = {}
    for i, category in enumerate(unique_categories):
        color_mapping[category] = condition_colors.get(
            category, color_palette[i % len(color_palette)]
        )

    # ——— existing x‐range ———
    if x_range is not None:
        global_lower_bound, global_upper_bound = x_range
    else:
        global_lower_bound = data_df[feature_to_plot].min()
        global_upper_bound = (
            xlimit if xlimit is not None else data_df[feature_to_plot].max()
        )

    # ——— existing figure/axes setup ———
    if small_multiples and separate is not None:
        fig, axes = plt.subplots(
            len(unique_categories),
            1,
            figsize=(figsize[0], figsize[1] * len(unique_categories)),
            sharex=True,
            facecolor=figure_background,
        )
        if len(unique_categories) == 1:
            axes = [axes]
    else:
        fig, ax = plt.subplots(figsize=figsize, facecolor=figure_background)
        axes = [ax]

    # ——— plot each category ———
    for i, category in enumerate(unique_categories):
        ax = axes[i] if (small_multiples and separate) else axes[0]
        subset = data_df[data_df[separate] == category] if category else data_df
        subsetvalues = subset[feature_to_plot]
        subsetvalues = subsetvalues[
            (subsetvalues >= global_lower_bound) & (subsetvalues <= global_upper_bound)
        ]

        ax.set_facecolor(axis_background)
        bin_edges = np.linspace(global_lower_bound, global_upper_bound, bins + 1)

        if use_kde:
            sns.kdeplot(
                subsetvalues,
                fill=kde_fill,
                ax=ax,
                color=color_mapping[category],
                linewidth=1.5,
                label=category,
                alpha=alpha,
            )
        else:
            counts, _ = np.histogram(subsetvalues, bins=bin_edges)
            if percentage:
                counts = 100 * counts / counts.sum() if counts.sum() > 0 else counts
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.bar(
                bin_centers,
                counts,
                width=np.diff(bin_edges),
                color=color_mapping[category],
                alpha=alpha,
                label=category,
            )

        avg_value = subsetvalues.mean() if average == "mean" else subsetvalues.median()
        if showavg:
            ax.axvline(avg_value, color=line_color, linestyle="--")

        # ——— NEW: small‐multiples threshold annotation ———
        if threshold is not None and small_multiples:
            ax.axvline(threshold_plot, color="gray", linestyle="--", linewidth=1.2)
            y_max = ax.get_ylim()[1]
            offset = 0.03 * (global_upper_bound - global_lower_bound)
            pct = (
                threshold_pct.get(category, threshold_pct)
                if isinstance(threshold_pct, dict)
                else threshold_pct
            )
            ax.text(
                threshold_plot - offset,
                y_max * 0.95,
                f"{pct['below']}% <",
                ha="right",
                va="top",
                fontsize=scaled_font,
                color=color_mapping[category],
            )
            ax.text(
                threshold_plot + offset,
                y_max * 0.95,
                f"{pct['above']}% >",
                ha="left",
                va="top",
                fontsize=scaled_font,
                color=color_mapping[category],
            )

    # ——— existing axes labels & styling ———
    ax.set_xlabel(x_label, fontsize=scaled_font, color=line_color)
    ax.set_ylabel(
        "Percentage" if percentage else "Count", fontsize=scaled_font, color=line_color
    )
    ax.tick_params(
        axis="both",
        which="both",
        color=line_color,
        labelcolor=line_color,
        labelsize=scaled_font,
    )

    if grid:
        ax.grid(
            True,
            linestyle="--",
            linewidth=0.5,
            color=line_color if not transparent else (0, 0, 0, 0.5),
            alpha=0.7,
            axis="y",
        )

    for spine in ax.spines.values():
        spine.set_edgecolor(line_color)
        if transparent:
            spine.set_alpha(0.9)

    ax.set_xlim(global_lower_bound, global_upper_bound)
    xticks = np.arange(
        global_lower_bound, global_upper_bound + tick_interval, tick_interval
    )
    ax.set_xticks(xticks)
    if y_range is not None:
        ax.set_ylim(y_range)

    if log_scale:
        ax.set_xscale("linear")
        if log_axis_label == "actual":
            formatter = FuncFormatter(lambda val, pos: f"{log_base ** val:.2g}")
            ax.xaxis.set_major_formatter(formatter)

    # ——— NEW: shared‐axis threshold annotation (once) ———
    if threshold is not None and not (small_multiples and separate):
        ax.axvline(threshold_plot, color="gray", linestyle="--", linewidth=1.2)
        y_max = ax.get_ylim()[1]
        offset = 0.03 * (global_upper_bound - global_lower_bound)
        for j, cat in enumerate(unique_categories):
            pct = (
                threshold_pct.get(cat, threshold_pct)
                if isinstance(threshold_pct, dict)
                else threshold_pct
            )
            vert = y_max * (0.95 - j * 0.08)
            ax.text(
                threshold_plot - offset,
                vert,
                f"{pct['below']}% <",
                ha="right",
                va="top",
                fontsize=scaled_font,
                color=color_mapping[cat],
            )
            ax.text(
                threshold_plot + offset,
                vert,
                f"{pct['above']}% >",
                ha="left",
                va="top",
                fontsize=scaled_font,
                color=color_mapping[cat],
            )

    # ——— existing legend placement ———
    if small_multiples:
        fig.legend(
            title=separate,
            fontsize=scaled_font,
            title_fontsize=scaled_font,
            loc="upper right",
            bbox_to_anchor=(1, 1),
        )
    else:
        legend = ax.legend(
            title=separate,
            fontsize=scaled_font,
            title_fontsize=scaled_font,
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )
        plt.gca().add_artist(legend)

    # ——— existing saving logic (PNG/SVG + cleanup) ———
    if save_folder is None:
        save_folder = os.path.join(master_dir, "plots", "histograms")
    os.makedirs(save_folder, exist_ok=True)

    ext = export_format.lower()
    if ext not in ["png", "svg"]:
        print("Invalid export format specified. Defaulting to 'png'.")
        ext = "png"
    out_filename = f"{feature}_histogram.{ext}"
    full_save_path = os.path.join(save_folder, out_filename)

    plt.savefig(
        full_save_path, bbox_inches="tight", transparent=transparent, format=ext
    )

    svg_data = None
    if ext == "svg":
        with open(full_save_path, encoding="utf-8") as f:
            svg_data = f.read()
        # strip problematic clipPaths & metadata
        svg_data = re.sub(
            r'<clipPath id="[^"]*">.*?</clipPath>', "", svg_data, flags=re.DOTALL
        )
        svg_data = re.sub(r'\s*clip-path="url\([^)]*\)"', "", svg_data, flags=re.DOTALL)
        svg_data = re.sub(r"<metadata>.*?</metadata>", "", svg_data, flags=re.DOTALL)
        svg_data = re.sub(r"<\?xml[^>]*\?>", "", svg_data, flags=re.DOTALL)
        svg_data = re.sub(r"<!DOCTYPE[^>]*>", "", svg_data, flags=re.DOTALL)
        svg_data = svg_data.strip()
        with open(full_save_path, "w", encoding="utf-8") as f:
            f.write(svg_data)

    if show_plot:
        plt.show()
    else:
        plt.close()

    if ext == "svg" and return_svg:
        return svg_data

        #### To be refined::: #############


def plot_trajectory(
    traj,
    colorby="particle",
    mpp=None,
    label=False,
    superimpose=None,
    cmap=None,
    ax=None,
    t_column=None,
    pos_columns=None,
    plot_style={},
    **kwargs,
):
    """
    Plot traces of trajectories for each particle.
    Optionally superimpose it on a frame from the video.

    Parameters
    ----------
    traj : DataFrame
        The DataFrame should include time and spatial coordinate columns.
    colorby : {'particle', 'frame'}, optional
    mpp : float, optional
        Microns per pixel. If omitted, the labels will have units of pixels.
    label : boolean, optional
        Set to True to write particle ID numbers next to trajectories.
    superimpose : ndarray, optional
        Background image, default None
    cmap : colormap, optional
        This is only used in colorby='frame' mode. Default = mpl.cm.winter
    ax : matplotlib axes object, optional
        Defaults to current axes
    t_column : string, optional
        DataFrame column name for time coordinate. Default is 'frame'.
    pos_columns : list of strings, optional
        Dataframe column names for spatial coordinates. Default is ['x', 'y'].
    plot_style : dictionary
        Keyword arguments passed through to the `Axes.plot(...)` command

    Returns
    -------
    Axes object

    """
    if cmap is None:
        cmap = plt.cm.winter
    if t_column is None:
        t_column = "frame"
    if pos_columns is None:
        pos_columns = ["x", "y"]
    if len(traj) == 0:
        raise ValueError("DataFrame of trajectories is empty.")

    _plot_style = dict(linewidth=1)
    _plot_style.update(**plot_style)

    if ax is None:
        ax = plt.gca()

    # Axes labels
    if mpp is None:
        ax.set_xlabel(f"{pos_columns[0]} [px]")
        ax.set_ylabel(f"{pos_columns[1]} [px]")
        mpp = 1.0  # for computations of image extent below
    else:
        ax.set_xlabel(f"{pos_columns[0]} [μm]")
        ax.set_ylabel(f"{pos_columns[1]} [μm]")

    # Background image
    if superimpose is not None:
        ax.imshow(
            superimpose,
            cmap=plt.cm.gray,
            origin="lower",
            interpolation="nearest",
            vmin=kwargs.get("vmin"),
            vmax=kwargs.get("vmax"),
        )
        ax.set_xlim(-0.5 * mpp, (superimpose.shape[1] - 0.5) * mpp)
        ax.set_ylim(-0.5 * mpp, (superimpose.shape[0] - 0.5) * mpp)

    # Trajectories
    if colorby == "particle":
        # Unstack particles into columns.
        unstacked = traj.set_index(["particle", t_column])[pos_columns].unstack()
        for i, trajectory in unstacked.iterrows():
            ax.plot(
                mpp * trajectory[pos_columns[0]],
                mpp * trajectory[pos_columns[1]],
                **_plot_style,
            )
    elif colorby == "frame":
        # Read http://www.scipy.org/Cookbook/Matplotlib/MulticoloredLine
        x = traj.set_index([t_column, "particle"])[pos_columns[0]].unstack()
        y = traj.set_index([t_column, "particle"])[pos_columns[1]].unstack()
        color_numbers = traj[t_column].values / float(traj[t_column].max())
        for particle in x:
            points = np.array([x[particle].values, y[particle].values]).T.reshape(
                -1, 1, 2
            )
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap)
            lc.set_array(color_numbers)
            ax.add_collection(lc)
            ax.set_xlim(x.apply(np.min).min(), x.apply(np.max).max())
            ax.set_ylim(y.apply(np.min).min(), y.apply(np.max).max())

    if label:
        unstacked = traj.set_index([t_column, "particle"])[pos_columns].unstack()
        first_frame = int(traj[t_column].min())
        coords = unstacked.fillna(method="backfill").stack().loc[first_frame]
        for particle_id, coord in coords.iterrows():
            ax.text(
                *coord.tolist(),
                s="%d" % particle_id,
                horizontalalignment="center",
                verticalalignment="center",
            )

    ax.invert_yaxis()
    return ax


def batch_plot_trajectories(
    master_folder,
    traj_df,
    batch=True,
    filename=None,
    colorby="particle",
    mpp=None,
    label=False,
    cmap=None,
):
    """
    Batch plot trajectories for all replicates across several conditions.

    Parameters
    ----------
    master_folder : str
        Path to the master folder containing 'data' and 'saved_data' folders.
    traj_df : DataFrame
        The DataFrame containing trajectory data.
    batch : bool, optional
        If True, plots trajectories for all replicates in batch mode.
        If False, plots trajectory for the specified filename.
    filename : str, optional
        Filename of interest when batch is False.
    colorby : str, optional
        Color by 'particle' or 'frame'.
    mpp : float, optional
        Microns per pixel.
    label : bool, optional
        Set to True to write particle ID numbers next to trajectories.
    cmap : colormap, optional
        Colormap to use for coloring tracks.

    """
    data_folder = os.path.join(master_folder, "data")
    vis_folder = os.path.join(master_folder, "visualization/trajectories")
    os.makedirs(vis_folder, exist_ok=True)

    if batch:
        for condition in os.listdir(data_folder):
            condition_folder = os.path.join(data_folder, condition)
            if os.path.isdir(condition_folder):
                for file in os.listdir(condition_folder):
                    if file.endswith(".tif"):
                        filepath = os.path.join(condition_folder, file)
                        subset_traj_df = traj_df[traj_df["filename"] == file]
                        if not subset_traj_df.empty:
                            frames = pims.open(filepath)
                            frame = frames[0]
                            fig, ax = plt.subplots()
                            plot_trajectory(
                                subset_traj_df,
                                colorby=colorby,
                                mpp=mpp,
                                label=label,
                                superimpose=frame,
                                cmap=cmap,
                                ax=ax,
                            )
                            plt.savefig(
                                os.path.join(vis_folder, f"{condition}_{file}.png")
                            )
                            plt.close(fig)
    elif filename is not None:
        filepath = os.path.join(data_folder, filename)
        subset_traj_df = traj_df[traj_df["filename"] == filename]
        if not subset_traj_df.empty:
            frames = pims.open(filepath)
            frame = frames[0]
            fig, ax = plt.subplots()
            plot_trajectory(
                subset_traj_df,
                colorby=colorby,
                mpp=mpp,
                label=label,
                superimpose=frame,
                cmap=cmap,
                ax=ax,
            )
            plt.show()
    else:
        print("Please provide a filename when batch is set to False.")


# Usage example
# master_folder = 'path_to__master_folder'
# traj_df = pd.read_csv('path_to_dataframe.csv')
# batch_plot_trajectories(master_folder, traj_df, batch=True)
# batch_plot_trajectories(master_folder, traj_df, batch=False, filename='file.tif')


def plot_particle_trajectory(ax, particle_df, particle_id, condition, plot_size=None):
    x_min, x_max = particle_df["x_um"].min(), particle_df["x_um"].max()
    y_min, y_max = particle_df["y_um"].min(), particle_df["y_um"].max()

    if plot_size is None:
        max_range = max(x_max - x_min, y_max - y_min)
        plot_size = max_range * 1.1  # Add 10% padding

    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    xlim = (x_center - plot_size / 2, x_center + plot_size / 2)
    ylim = (y_center + plot_size / 2, y_center - plot_size / 2)  # Inverted y-axis

    scatter = ax.scatter(
        particle_df["x_um"],
        particle_df["y_um"],
        c=particle_df["time_s"],
        cmap="viridis",
        s=30,
    )
    ax.plot(particle_df["x_um"], particle_df["y_um"], "-", linewidth=1, alpha=0.5)

    ax.set_xlabel("X position (µm)", fontsize=8)
    ax.set_ylabel("Y position (µm)", fontsize=8)
    ax.set_title(f"{condition}: Particle {particle_id}", fontsize=10)

    ax.invert_yaxis()
    ax.set_aspect("equal")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=8)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_label("Time (s)", fontsize=8)
    cbar.ax.tick_params(labelsize=6)

    return scatter


def plot_multiple_particles(combined_df, particles_per_condition=2, plot_size=None):
    conditions = combined_df["condition"].unique()
    num_conditions = len(conditions)
    total_particles = num_conditions * particles_per_condition

    # Calculate the grid size
    grid_size = int(np.ceil(np.sqrt(total_particles)))

    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(4 * grid_size, 4 * grid_size)
    )
    fig.suptitle("Particle Trajectories by Condition", fontsize=16)

    # Flatten the axes array for easier iteration
    axes_flat = axes.flatten()

    plot_index = 0
    for condition in conditions:
        condition_df = combined_df[combined_df["condition"] == condition]
        particles = np.random.choice(
            condition_df["unique_id"].unique(), particles_per_condition, replace=False
        )

        for particle_id in particles:
            if plot_index >= len(axes_flat):
                break

            ax = axes_flat[plot_index]
            particle_df = condition_df[condition_df["unique_id"] == particle_id]

            plot_particle_trajectory(ax, particle_df, particle_id, condition, plot_size)
            plot_index += 1

    # Remove extra subplots
    for i in range(plot_index, len(axes_flat)):
        fig.delaxes(axes_flat[i])

    plt.tight_layout()
    plt.show()

    # Build this into a function


def load_image(file_path):
    return skio.imread(file_path)  # changed to skio to avoid shadowing


def load_tracks(df, filename):
    tracks = df[df["filename"] == filename]
    return tracks


def get_condition_from_filename(df, filename):
    try:
        condition = df[df["filename"] == filename]["condition"].iloc[0]
    except IndexError:
        print(f"Error: Filename '{filename}' not found in the dataframe.")
        raise
    return condition


def save_movie(viewer, tracks, feature="particle", save_path="movie.mov", steps=None):
    animation = Animation(viewer)

    # Set the display to 2D
    viewer.dims.ndisplay = 2

    # Automatically set the keyframes for the start, middle, and end
    num_frames = len(tracks["frame"].unique())

    # If steps not provided, use the default (num_frames - 1)
    if steps is None:
        steps = num_frames - 1
        print(f"Using default steps: {steps}")

    # Start keyframe (frame 0)
    viewer.dims.set_point(0, 0)
    animation.capture_keyframe(steps=steps)

    # Middle keyframe (middle frame)
    middle_frame = num_frames // 2
    viewer.dims.set_point(0, middle_frame)
    animation.capture_keyframe(steps=steps)

    # End keyframe (last frame)
    viewer.dims.set_point(0, num_frames - 1)
    animation.capture_keyframe(steps=steps)

    # Save the animation to the specified path
    animation.animate(
        save_path, canvas_only=True
    )  # canvas_only=True to exclude controls


def napari_visualize_image_with_tracksdev(
    tracks_df,
    master_dir=config.MASTER,
    condition=None,
    cell=None,
    location=None,
    save_movie_flag=False,
    feature="particle",
):
    master_dir = config.MASTER + "data"
    movie_dir = config.MASTER + "movies"

    print("The master directory is:", master_dir)
    if save_movie_flag:
        print("The movie directory is:", movie_dir)

    # Handle location input
    location_col = _get_location_column(tracks_df)
    locationlist = tracks_df[location_col].unique()
    if isinstance(location, int):
        location = locationlist[location]
    elif isinstance(location, str):
        if location not in locationlist:
            raise ValueError(
                f"Location '{location}' not found in available locations: {locationlist}"
            )
    elif location is None:
        location = np.random.choice(locationlist)
    else:
        raise ValueError("Location must be a string, integer, or None.")

    # Filter the dataframe by the selected location
    filtered_tracks_df = tracks_df[tracks_df[location_col] == location]

    # Handle condition input
    conditionlist = filtered_tracks_df["condition"].unique()
    if isinstance(condition, int):
        condition = conditionlist[condition]
    elif isinstance(condition, str):
        if condition not in conditionlist:
            raise ValueError(
                f"Condition '{condition}' not found in available conditions for location '{location}': {conditionlist}"
            )
    elif condition is None:
        condition = np.random.choice(conditionlist)
    else:
        raise ValueError("Condition must be a string, integer, or None.")

    # Handle cell input
    celllist = filtered_tracks_df[filtered_tracks_df["condition"] == condition][
        "filename"
    ].unique()
    if isinstance(cell, int):
        cell = celllist[cell]
    elif isinstance(cell, str):
        if cell not in celllist:
            raise ValueError(
                f"Cell '{cell}' not found in available cells for condition '{condition}' and location '{location}': {celllist}"
            )
    elif cell is None:
        cell = np.random.choice(celllist)
    else:
        raise ValueError("Cell must be a string, integer, or None.")

    # Construct the full file path by removing '_tracked' and adding '.tif'
    image_filename = cell.replace("_tracked", "") + ".tif"
    image_path = os.path.join(master_dir, condition, image_filename)

    # Load the image
    image = load_image(image_path)

    # Load the tracks
    tracks = load_tracks(filtered_tracks_df, cell)

    print(tracks.columns)

    # Prepare the tracks DataFrame for Napari
    tracks_new_df = tracks[["particle", "frame", "y", "x"]]

    # Include 'particle' and all features from config.FEATURES
    features_dict = {"particle": tracks["particle"].values}
    features_dict.update(
        {
            feature: tracks[feature].values
            for feature in config.FEATURES
            if feature in tracks.columns
        }
    )

    # Start Napari viewer
    viewer = napari.Viewer()

    # Add image layer
    viewer.add_image(image, name=f"Raw {cell}")

    # Add tracks layer, using 'particle' for coloring, with additional features
    viewer.add_tracks(
        tracks_new_df.to_numpy(),
        features=features_dict,
        name=f"Tracks {cell}",
        color_by=feature,
    )

    # Save the movie if specified
    if save_movie_flag:
        movies_dir = os.path.join(movie_dir, "movies")
        os.makedirs(movies_dir, exist_ok=True)
        movie_path = os.path.join(movies_dir, f"{condition}_{cell}.mov")
        save_movie(viewer, tracks_new_df, feature=feature, save_path=movie_path)

    napari.run()


def napari_visualize_image_with_tracksdev2(
    tracks_df,
    master_dir=config.MASTER,
    condition=None,
    cell=None,
    location=None,
    save_movie_flag=False,
    feature="particle",
    steps=None,
):
    master_dir = master_dir + "data"
    movie_dir = master_dir + "movies"

    print("The master directory is:", master_dir)
    if save_movie_flag:
        print("The movie directory is:", movie_dir)

    # Handle location input
    location_col = _get_location_column(tracks_df)
    locationlist = tracks_df[location_col].unique()
    if isinstance(location, int):
        location = locationlist[location]
    elif isinstance(location, str):
        if location not in locationlist:
            raise ValueError(
                f"Location '{location}' not found in available locations: {locationlist}"
            )
    elif location is None:
        np.random.shuffle(locationlist)  # Shuffle the list to make random selection
        for loc in locationlist:
            if loc in locationlist:
                location = loc
                break
        if location is None:
            raise ValueError(
                f"No valid location found in available locations: {locationlist}"
            )
    else:
        raise ValueError("Location must be a string, integer, or None.")

    # Filter the dataframe by the selected location
    filtered_tracks_df = tracks_df[tracks_df[location_col] == location]

    # Handle condition input
    conditionlist = filtered_tracks_df["condition"].unique()
    if isinstance(condition, int):
        condition = conditionlist[condition]
    elif isinstance(condition, str):
        if condition not in conditionlist:
            raise ValueError(
                f"Condition '{condition}' not found in available conditions for location '{location}': {conditionlist}"
            )
    elif condition is None:
        np.random.shuffle(conditionlist)  # Shuffle the list to make random selection
        for cond in conditionlist:
            if cond in conditionlist:
                condition = cond
                break
        if condition is None:
            raise ValueError(
                f"No valid condition found for location '{location}': {conditionlist}"
            )
    else:
        raise ValueError("Condition must be a string, integer, or None.")

    # Handle cell input
    celllist = filtered_tracks_df[filtered_tracks_df["condition"] == condition][
        "filename"
    ].unique()
    if isinstance(cell, int):
        cell = celllist[cell]
    elif isinstance(cell, str):
        if cell not in celllist:
            raise ValueError(
                f"Cell '{cell}' not found in available cells for condition '{condition}' and location '{location}': {celllist}"
            )
    elif cell is None:
        np.random.shuffle(celllist)  # Shuffle the list to make random selection
        for c in celllist:
            if c in celllist:
                cell = c
                break
        if cell is None:
            raise ValueError(
                f"No valid cell found for condition '{condition}' and location '{location}': {celllist}"
            )
    else:
        raise ValueError("Cell must be a string, integer, or None.")

    # Construct the full file path by removing '_tracked' and adding '.tif'
    image_filename = cell.replace("_tracked", "") + ".tif"
    image_path = os.path.join(master_dir, condition, image_filename)

    # Load the image
    image = load_image(image_path)

    # Load the tracks
    tracks = load_tracks(filtered_tracks_df, cell)

    print(tracks.columns)

    # Prepare the tracks DataFrame for Napari
    tracks_new_df = tracks[["particle", "frame", "y", "x"]]

    # Include 'particle' and all features from config.FEATURES2
    features_dict = {"particle": tracks["particle"].values}
    features_dict.update(
        {
            feature: tracks[feature].values
            for feature in config.FEATURES2
            if feature in tracks.columns
        }
    )

    # Start Napari viewer
    viewer = napari.Viewer()

    # Add image layer
    viewer.add_image(image, name=f"Raw {cell}")

    # Add tracks layer, using 'particle' for coloring, with additional features
    viewer.add_tracks(
        tracks_new_df.to_numpy(),
        features=features_dict,
        name=f"Tracks {cell}",
        color_by=feature,
    )

    # Save the movie if specified
    if save_movie_flag:
        # If steps is not provided, define it based on data (here, maximum frame + 1)
        if steps is None:
            steps = int(tracks_new_df["frame"].max()) + 1
            print(f"Number of steps for the movie automatically set to: {steps}")
        movies_dir = os.path.join(movie_dir, "movies")
        os.makedirs(movies_dir, exist_ok=True)
        movie_path = os.path.join(movies_dir, f"{condition}_{cell}.mov")
        save_movie(
            viewer, tracks_new_df, feature=feature, save_path=movie_path, steps=steps
        )

    napari.run()


def bootstrap_ci_mean(data, num_samples=1000, alpha=0.05):
    n = len(data)
    samples = np.random.choice(data, size=(num_samples, n), replace=True)
    means = np.mean(samples, axis=1)
    lower_bound = np.percentile(means, 100 * alpha / 2)
    upper_bound = np.percentile(means, 100 * (1 - alpha / 2))
    return upper_bound - lower_bound


### Vectorized bootstrap median


def bootstrap_ci_median(data, num_samples=1000, alpha=0.05):
    n = len(data)
    samples = np.random.choice(data, size=(num_samples, n), replace=True)
    medians = np.median(samples, axis=1)
    lower_bound = np.percentile(medians, 100 * alpha / 2)
    upper_bound = np.percentile(medians, 100 * (1 - alpha / 2))
    return upper_bound - lower_bound


def plot_time_series(
    data_df,
    factor_col="speed_um_s",
    absolute=True,
    separate_by="condition",
    palette="colorblind",
    meanormedian="mean",
    multiplot=False,
    talk=False,
    bootstrap=True,
    show_plot=True,
    master_dir=None,
    order=None,
    grid=True,
    custom_yrange=None,
):
    """
    Plot time series of a specified factor, with mean/median as a line and confidence intervals as shaded areas.

    Parameters
    ----------
    data_df : DataFrame
        DataFrame containing the time series data.
    factor_col : str, optional
        The column representing the factor to be plotted on the y-axis. Default is 'speed_um_s'.
    absolute : bool, optional
        Whether to use absolute time values or time zeroed values. Default is True.
    separate_by : str, optional
        Column to separate the data by, for coloring. If None, all data will be plotted together. Default is None.
    palette : str, optional
        Color palette for the plot. Default is 'colorblind'.
    meanormedian : str, optional
        Whether to use mean or median for aggregation. Default is 'mean'.
    multiplot : bool, optional
        Whether to generate separate small multiple plots for each category. Default is False.
    talk : bool, optional
        Whether to set the figure size to the original large size or a smaller size. Default is False.
    bootstrap : bool, optional
        Whether to use bootstrapping for confidence intervals. Default is True.
    show_plot : bool, optional
        Whether to display the plot in the notebook. Default is True.
    master_dir : str, optional
        The directory where the plots folder will be created and the plot will be saved. Default is None.
    order : list, optional
        Specific order for the conditions. Default is None.
    grid : bool, optional
        Whether to display grid lines. Default is True.
    custom_yrange : tuple, optional
        Custom y-axis range as (ymin, ymax). Default is None, which sets limits based on data.

    """
    xmin = 0.2  # A FIX FOR NOW because really this should be the same as the shortest track (filtered to 0.2 s during filterstubs)

    if master_dir is None:
        master_dir = "."  # Use current directory if not provided

    if separate_by is not None and order is not None:
        # Ensure the data is ordered according to the specified order
        data_df[separate_by] = pd.Categorical(
            data_df[separate_by], categories=order, ordered=True
        )

    if not absolute:
        time_col = "time_s_zeroed"
        max_time_zeroed = data_df["time_s_zeroed"].max()
        x_label = "Time zeroed (s)"
        xmax = max_time_zeroed
    else:
        time_col = "time_s"
        max_time = data_df["time_s"].max()
        x_label = "Time (s)"
        xmax = max_time

    # Use the categories attribute to maintain the specified order
    if separate_by is not None:
        # Convert to categorical if not already
        if not pd.api.types.is_categorical_dtype(data_df[separate_by]):
            data_df[separate_by] = pd.Categorical(
                data_df[separate_by], categories=order, ordered=True
            )
        unique_categories = data_df[separate_by].cat.categories
    else:
        unique_categories = [None]

    color_palette = sns.color_palette(palette, len(unique_categories))

    # Set figure size and font size based on the `talk` and `multiplot` parameters
    if talk:
        base_fig_size = (30, 12)
        font_size = 35
    else:
        base_fig_size = (10, 4)
        font_size = 14

    # Adjust figure size if multiplot is true
    if multiplot and separate_by:
        fig_size = (base_fig_size[0], base_fig_size[1] * len(unique_categories))
    else:
        fig_size = base_fig_size

    sns.set_context(
        "notebook",
        rc={
            "lines.linewidth": 2.5,
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
        },
    )

    if multiplot and separate_by:
        num_categories = len(unique_categories)
        fig, axes = plt.subplots(num_categories, 1, figsize=fig_size, sharex=True)

        if num_categories == 1:
            axes = [axes]  # To handle the case with only one subplot

        for i, category in enumerate(unique_categories):
            if pd.isna(category):
                continue
            ax = axes[i] if len(unique_categories) > 1 else axes
            subset = data_df[data_df[separate_by] == category]
            times = subset[time_col]
            factors = subset[factor_col]

            if meanormedian == "mean":
                avg_factors = subset.groupby(time_col)[factor_col].mean()
                ci_func = bootstrap_ci_mean if bootstrap else lambda x: sem(x) * 1.96
            else:
                avg_factors = subset.groupby(time_col)[factor_col].median()
                ci_func = bootstrap_ci_median if bootstrap else lambda x: sem(x) * 1.96

            ci = subset.groupby(time_col)[factor_col].apply(ci_func)

            color = color_palette[i]
            label = category

            # Exclude the first time point (time zero)
            valid_indices = avg_factors.index > 0

            ax.plot(
                avg_factors.index[valid_indices],
                avg_factors.values[valid_indices],
                label=label,
                color=color,
                linewidth=2.5,
            )
            ax.fill_between(
                avg_factors.index[valid_indices],
                np.maximum(
                    (avg_factors - ci)[valid_indices], 0
                ),  # Ensure lower bound is not below zero
                (avg_factors + ci)[valid_indices],
                color=color,
                alpha=0.3,
            )
            ax.set_xlabel(x_label, fontsize=font_size)
            ax.set_ylabel(factor_col, fontsize=font_size, labelpad=20)
            ax.legend(fontsize=font_size, loc="upper left", bbox_to_anchor=(1, 1))
            ax.set_xlim(xmin, xmax)

            # Set custom or automatic y-limits
            if custom_yrange:
                ax.set_ylim(custom_yrange)
            else:
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(0, ymax * 1.1)  # Add padding if using automatic limits

            if grid:
                ax.grid(
                    True,
                    which="both",
                    linestyle="--",
                    linewidth=0.5,
                    color="gray",
                    alpha=0.7,
                )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_title(f"{category}", fontsize=font_size)

        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=fig_size)

        for i, category in enumerate(unique_categories):
            if pd.isna(category):
                continue
            subset = (
                data_df
                if category is None
                else data_df[data_df[separate_by] == category]
            )
            times = subset[time_col]
            factors = subset[factor_col]

            if meanormedian == "mean":
                avg_factors = subset.groupby(time_col)[factor_col].mean()
                ci_func = bootstrap_ci_mean if bootstrap else lambda x: sem(x) * 1.96
            else:
                avg_factors = subset.groupby(time_col)[factor_col].median()
                ci_func = bootstrap_ci_median if bootstrap else lambda x: sem(x) * 1.96

            ci = subset.groupby(time_col)[factor_col].apply(ci_func)

            color = color_palette[i]
            label = "Overall" if category is None else category

            # Exclude the first time point (time zero)
            valid_indices = avg_factors.index > 0

            ax.plot(
                avg_factors.index[valid_indices],
                avg_factors.values[valid_indices],
                label=label,
                color=color,
                linewidth=2.5,
            )
            ax.fill_between(
                avg_factors.index[valid_indices],
                np.maximum(
                    (avg_factors - ci)[valid_indices], 0
                ),  # Ensure lower bound is not below zero
                (avg_factors + ci)[valid_indices],
                color=color,
                alpha=0.3,
            )

        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(factor_col, fontsize=font_size, labelpad=20)
        ax.legend(fontsize=font_size, loc="upper left", bbox_to_anchor=(1.05, 1))
        if grid:
            ax.grid(
                True,
                which="both",
                linestyle="--",
                linewidth=0.5,
                color="gray",
                alpha=0.7,
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(xmin, xmax)

        # Set custom or automatic y-limits
        if custom_yrange:
            ax.set_ylim(custom_yrange)
        else:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(0, ymax * 1.1)  # Add padding if using automatic limits

        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to fit legend

    # Create directory for plots if it doesn't exist
    plots_dir = os.path.join(master_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Generate filename
    time_type = "absolute" if absolute else "time_zeroed"
    bootstrap_text = "bootstrapped" if bootstrap else "nonbootstrapped"
    multiplot_text = "multiplot" if multiplot else "singleplot"
    filename = f"{plots_dir}/{factor_col}_{time_type}_{meanormedian}_{bootstrap_text}_{multiplot_text}.png"

    # Save plot
    plt.savefig(filename, bbox_inches="tight")

    # Show plot if specified
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_barplots(
    data_df,
    factor_col="speed_um_s",
    separate_by="condition",
    palette="colorblind",
    meanormedian="mean",
    talk=False,
):
    """
    Plot bar plots of a specified factor, with bootstrapped confidence intervals.

    Parameters
    ----------
    data_df : DataFrame
        DataFrame containing the data.
    factor_col : str, optional
        The column representing the factor to be plotted on the y-axis. Default is 'speed_um_s'.
    separate_by : str, optional
        Column to separate the data by, for coloring. If None, all data will be plotted together. Default is 'condition'.
    palette : str, optional
        Color palette for the plot. Default is 'colorblind'.
    meanormedian : str, optional
        Whether to use mean or median for aggregation. Default is 'mean'.
    talk : bool, optional
        Whether to set the figure size to the original large size or a smaller size. Default is False.

    """
    unique_categories = data_df[separate_by].unique() if separate_by else [None]
    color_palette = sns.color_palette(palette, len(unique_categories))

    # Set figure size based on the `talk` parameter
    if talk:
        fig_size = (20, 12)
        font_size = 35
    else:
        fig_size = (5, 3)
        font_size = 14

    fig, ax = plt.subplots(figsize=fig_size)
    sns.set_context(
        "notebook",
        rc={
            "lines.linewidth": 2.5,
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
        },
    )

    avg_factors_list = []
    ci_intervals = []

    for i, category in enumerate(unique_categories):
        subset = (
            data_df if category is None else data_df[data_df[separate_by] == category]
        )

        if meanormedian == "mean":
            avg_factors = subset[factor_col].mean()
            ci_interval = bootstrap_ci_mean(
                subset[factor_col], num_samples=1000, alpha=0.05
            )
        else:
            avg_factors = subset[factor_col].median()
            ci_interval = bootstrap_ci_median(
                subset[factor_col], num_samples=1000, alpha=0.05
            )

        avg_factors_list.append(avg_factors)
        ci_intervals.append(ci_interval)

    categories = unique_categories if separate_by else ["Overall"]
    ax.bar(
        categories,
        avg_factors_list,
        yerr=ci_intervals,
        color=color_palette,
        capsize=5,
        edgecolor="black",
    )

    # Remove 'Condition_' prefix from x tick labels
    new_labels = [label.replace("Condition_", "") for label in categories]
    if talk:
        ax.set_xticklabels(new_labels, fontsize=font_size)
    else:
        ax.set_xticklabels(new_labels, fontsize=font_size, rotation=90)

    ax.set_ylabel(factor_col, fontsize=font_size)
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    plt.tight_layout()

    plt.show()


def plot_violinplots(
    data_df,
    factor_col="speed_um_s",
    separate_by="condition",
    palette="colorblind",
    talk=False,
    orderin=None,
):
    """
    Plot violin plots of a specified factor, with data separated by categories.

    Parameters
    ----------
    data_df : DataFrame
        DataFrame containing the data.
    factor_col : str, optional
        The column representing the factor to be plotted on the y-axis. Default is 'speed_um_s'.
    separate_by : str, optional
        Column to separate the data by, for coloring. If None, all data will be plotted together. Default is 'condition'.
    palette : str, optional
        Color palette for the plot. Default is 'colorblind'.
    talk : bool, optional
        Whether to set the figure size to the original large size or a smaller size. Default is False.
    orderin : list, optional
        Custom order for the categories in the violin plot. Default is None.

    """
    unique_categories = data_df[separate_by].unique() if separate_by else [None]
    color_palette = sns.color_palette(palette, len(unique_categories))

    # Set figure size based on the `talk` parameter
    if talk:
        fig_size = (20, 12)
        font_size = 35
    else:
        fig_size = (5, 3)
        font_size = 14

    fig, ax = plt.subplots(figsize=fig_size)
    sns.set_context(
        "notebook",
        rc={
            "lines.linewidth": 2.5,
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
        },
    )

    # Plot violin plot with custom order
    sns.violinplot(
        x=separate_by,
        y=factor_col,
        hue=separate_by,
        data=data_df,
        palette=color_palette,
        ax=ax,
        legend=False,
        alpha=0.79,
        order=orderin,
    )

    # If orderin is provided, update x-tick labels accordingly
    if orderin is not None:
        new_labels = [label.replace("Condition_", "") for label in orderin]
    else:
        new_labels = [label.replace("Condition_", "") for label in unique_categories]

    ax.set_xticks(range(len(new_labels)))
    ax.set_xticklabels(new_labels, fontsize=font_size)

    ax.set_ylabel(factor_col, fontsize=font_size, labelpad=20)
    ax.set_xlabel(None)
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    plt.tight_layout()

    plt.show()


def plot_metric_for_selected_particles(
    time_windowed_df,
    feature="avg_msd",
    num_particles=5,
    global_xlim=True,
    subplot_size=5,
):
    # Get unique motion classes
    motion_classes = time_windowed_df["motion_class"].unique()

    # Randomly select a set of particles for each motion class
    selected_particles = {}
    global_max_time = 0  # To store the global maximum time across selected particles
    max_feature_value = (
        0  # To store the global max feature value across selected particles
    )

    for motion_class in motion_classes:
        particles = time_windowed_df[time_windowed_df["motion_class"] == motion_class][
            "unique_id"
        ].unique()
        selected_particles[motion_class] = random.sample(
            list(particles), min(num_particles, len(particles))
        )

        # Calculate the maximum time_s for the current motion class
        for unique_id in selected_particles[motion_class]:
            data = time_windowed_df[time_windowed_df["unique_id"] == unique_id]
            global_max_time = max(global_max_time, data["time_s"].max())
            max_feature_value = max(max_feature_value, data[feature].max())

    # Add padding to the maximum feature value
    padding = 0.1
    max_feature_value *= 1 + padding

    # Determine the total number of plots
    total_plots = sum(len(particles) for particles in selected_particles.values())

    # Set up subplots with fixed subplot size
    ncols = num_particles  # Number of particles per row
    nrows = len(motion_classes)  # One row per motion class

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(subplot_size * ncols, subplot_size * nrows)
    )
    fig.suptitle(f"{feature} vs. Time for Selected Particles in Each Motion Class")

    # Plot each selected particle in its subplot
    plot_idx = 0
    for i, motion_class in enumerate(motion_classes):
        for j, unique_id in enumerate(selected_particles[motion_class]):
            data = time_windowed_df[time_windowed_df["unique_id"] == unique_id]
            ax = axes[i, j]  # Access subplot at row i and column j
            ax.plot(
                data["time_s"],
                data[feature],
                label=f"Particle {unique_id}",
                color=get_color(motion_class),
            )
            ax.set_title(f"Particle {unique_id} ({motion_class})")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(rf"{feature} ($\mu m^2$)")
            ax.grid(True)

            # Set limits for x-axis and y-axis
            if global_xlim:
                ax.set_xlim(0, global_max_time)
            else:
                ax.set_xlim(0, data["time_s"].max())

            ax.set_ylim(0, max_feature_value)

            plot_idx += 1

    # Turn off any empty subplots
    for i in range(plot_idx, nrows * ncols):
        fig.delaxes(axes.flatten()[i])

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def get_color(motion_class):
    # Assign colors using Colorblind colormap
    colorblind_colors = plt.get_cmap("tab10")
    if motion_class == "subdiffusive":
        return colorblind_colors(0)  # Blue
    elif motion_class == "normal":
        return colorblind_colors(1)  # Orange
    elif motion_class == "superdiffusive":
        return colorblind_colors(2)  # Green
    else:
        return "black"


def plot_single_particle_msd(msd_lagtime_df):
    """
    This thing basically takes 3 example particle tracks, one from each motion class
    """
    # Ensure we have data for each motion class
    motion_classes = ["subdiffusive", "normal", "superdiffusive"]

    # Randomly select one unique_id from each motion class
    selected_particles = {}
    for motion_class in motion_classes:
        particles = msd_lagtime_df[msd_lagtime_df["motion_class"] == motion_class][
            "unique_id"
        ].unique()
        if len(particles) > 0:
            selected_particles[motion_class] = random.choice(particles)

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Plot MSD for each selected particle
    for motion_class, unique_id in selected_particles.items():
        data = msd_lagtime_df[msd_lagtime_df["unique_id"] == unique_id]
        plt.plot(
            data["lag_time"],
            data["msd"],
            label=f"{motion_class} (Particle {unique_id})",
            color=get_color(motion_class),
        )

    # Set log scales
    plt.xscale("log")
    plt.yscale("log")

    # Add labels and title
    plt.xlabel("Time Lag (s)")
    plt.ylabel(r"MSD ($\mu m^2$)")
    plt.title("MSD vs. Time Lag for Selected Particles")
    plt.legend()
    plt.grid(True)
    plt.show()


def get_color(motion_class):
    # Assign colors using Colorblind colormap
    colorblind_colors = plt.get_cmap("tab10")
    if motion_class == "subdiffusive":
        return colorblind_colors(0)  # Blue
    elif motion_class == "normal":
        return colorblind_colors(1)  # Orange
    elif motion_class == "superdiffusive":
        return colorblind_colors(2)  # Green
    else:
        return "black"


def plot_classification_pie_charts(
    df,
    group_by=None,
    colormap_name="Dark2",
    order=None,
    figsize=(15, 10),
    font_size=12,
    label_font_size=8,
):
    # Handle group_by parameter with location column detection
    if group_by is None:
        group_by = _get_location_column(df)

    # Get unique categories for grouping
    categories = df[group_by].unique()

    # Determine the layout for subplots
    n_categories = len(categories)
    ncols = 3  # Number of columns
    nrows = (n_categories + ncols - 1) // ncols  # Calculate the number of rows needed

    # Create subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()  # Flatten to iterate easily

    colormap = cm.get_cmap(colormap_name)

    # If an order is provided, ensure the colormap colors align with the specified order
    if order is not None:
        unique_classes = order
    else:
        unique_classes = df["motion_class"].unique()

    # Create a mapping of motion_class to specific colors
    color_map = {
        cls: colormap(i / (len(unique_classes) - 1))
        for i, cls in enumerate(unique_classes)
    }

    # Plot each category as a separate pie chart
    for i, category in enumerate(categories):
        ax = axes[i]
        subset_df = df[df[group_by] == category]
        classification_counts = subset_df["motion_class"].value_counts()
        total_count = classification_counts.sum()
        percentages = classification_counts / total_count * 100

        # Reorder classification_counts based on the order
        if order is not None:
            classification_counts = classification_counts.reindex(order, fill_value=0)

        # Define labels for outside the pie
        outside_labels = [
            f"{cls} ({count})"
            for cls, count in zip(
                classification_counts.index, classification_counts.values, strict=False
            )
        ]

        # Colors for pie slices
        colors = [color_map[cls] for cls in classification_counts.index]

        # Plot pie chart with percentages inside
        wedges, texts, autotexts = ax.pie(
            classification_counts,
            labels=outside_labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            textprops={"fontsize": label_font_size},
        )

        # Set the color and size of the percentage text
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(label_font_size)

        ax.set_title(f"{category} ({total_count} tracks)", fontsize=font_size)
        ax.axis("equal")  # Equal aspect ratio ensures the pie chart is circular.

    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(
        f"Classification of Time Windowed Tracks by {group_by}", fontsize=font_size + 2
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_boxplots(
    data_df,
    feature,
    x_category,
    font_size=12,
    order=None,
    palette="colorblind",
    background="white",
    transparent=False,
    line_color="black",
    show_plot=True,
    master_dir=None,
    grid=True,
    bw=False,
    strip=False,
    y_max=None,
    figsize=(10, 8),
    annotate_median=False,
    rotation=90,
    dotsize=3,
):
    """
    Plot boxplots for the specified feature against a categorical x_category with custom order and styling options.

    Parameters
    ----------
    data_df : DataFrame
        DataFrame containing the data.
    feature : str
        The feature to plot on the y-axis.
    x_category : str
        The categorical feature to plot on the x-axis.
    font_size : int, optional
        Font size for the plot text. Default is 12.
    order : list, optional
        Specific order for the categories. Default is None.
    palette : str, optional
        Color palette for the plot. Default is 'colorblind'.
    background : str, tuple, optional
        Background color as color name, RGB tuple, or hex code. Default is 'white'.
    transparent : bool, optional
        If True, makes the plot fully transparent except for box plots and axes. Default is False.
    line_color : str, optional
        Color of all plot lines, including box edges, axis lines, grid lines, labels, and strip plot dots. Default is 'black'.
    show_plot : bool, optional
        Whether to display the plot in the notebook. Default is True.
    master_dir : str, optional
        Directory to save the plot. Default is None.
    grid : bool, optional
        Whether to display grid lines. Default is True.
    bw : bool, optional
        Whether to use black-and-white styling. Default is False.
    strip : bool, optional
        Whether to overlay a stripplot on the boxplot. Default is False.
    y_max : float, optional
        Maximum value for the y-axis. Default is None.
    figsize : tuple, optional
        Size of the plot (width, height) in inches. Default is (10, 8).

    """
    # Validate and apply background color
    if is_color_like(background):
        figure_background = background
    else:
        print("Invalid color provided for background. Defaulting to white.")
        figure_background = "white"

    # Create figure and set background color
    fig, ax = plt.subplots(figsize=figsize, facecolor=figure_background)
    if not transparent:
        ax.set_facecolor(figure_background)  # Set plot area background color
    else:
        fig.patch.set_alpha(0)  # Make the figure background transparent
        ax.set_facecolor((0, 0, 0, 0))  # Make the plot area transparent

    sns.set_context(
        "notebook", rc={"xtick.labelsize": font_size, "ytick.labelsize": font_size}
    )

    if bw:
        boxplot = sns.boxplot(
            x=x_category,
            y=feature,
            data=data_df,
            linewidth=1.5,
            showfliers=False,
            color="white",
            order=order,
        )
        for patch in boxplot.patches:
            patch.set_edgecolor(line_color)
            patch.set_linewidth(1.5)
        for element in ["boxes", "whiskers", "medians", "caps"]:
            plt.setp(boxplot.artists, color=line_color)
            plt.setp(boxplot.lines, color=line_color)
    else:
        boxplot = sns.boxplot(
            x=x_category,
            y=feature,
            data=data_df,
            palette=palette,
            order=order,
            showfliers=False,
            linewidth=1.5,
        )
        for patch in boxplot.patches:
            patch.set_edgecolor(line_color)
            patch.set_linewidth(1.5)
        for line in (
            boxplot.lines
        ):  # Apply color to all boxplot lines, including medians and quartiles
            line.set_color(line_color)
            line.set_linewidth(1.5)

    if strip:
        sns.stripplot(
            x=x_category,
            y=feature,
            data=data_df,
            color=line_color,
            size=dotsize,
            order=order,
            jitter=True,
        )

    plt.xlabel("", fontsize=font_size, color=line_color)
    plt.ylabel(feature, fontsize=font_size, color=line_color)
    # plt.title(f'{feature} by {x_category}', fontsize=font_size, color=line_color)

    # Set tick and label colors
    ax.tick_params(
        axis="both",
        which="both",
        color=line_color,
        labelcolor=line_color,
        labelsize=font_size,
        rotation=rotation,
    )

    # Set grid and axis line colors
    if grid:
        # ax.grid(True, linestyle='--', linewidth=0.5, color=line_color if not transparent else (0, 0, 0, 0.5), alpha=0.7, axis='y')
        ax.grid(
            True, linestyle="--", linewidth=0.5, color=line_color, alpha=0.7, axis="y"
        )

    # Set maximum y-axis limit if specified
    if y_max is not None:
        plt.ylim(top=y_max)

    # Customize spines
    for spine in ax.spines.values():
        spine.set_edgecolor(line_color)
        spine.set_linewidth(1.5)
        if transparent:
            spine.set_alpha(0.5)

    if annotate_median:
        medians = data_df.groupby(x_category)[feature].median()
        y_max = plt.ylim()[1]
        sorted_medians = medians.reindex(order) if order else medians
        for i, median in enumerate(sorted_medians):
            plt.text(
                i,
                y_max * 0.965,
                f"{median:.2f}",
                horizontalalignment="center",
                size=font_size,
                color=line_color,
                weight="bold",
            )

    plt.tight_layout()

    if master_dir is None:
        master_dir = "plots"
    os.makedirs(master_dir, exist_ok=True)
    filename = f"{master_dir}/{feature}_by_{x_category}.png"
    plt.savefig(filename, bbox_inches="tight", transparent=transparent)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_boxplots_svg(
    data_df,
    feature,
    x_category,
    font_size=12,
    order=None,
    palette="colorblind",
    background="white",
    transparent=False,
    line_color="black",
    show_plot=True,
    master_dir=None,
    grid=True,
    bw=False,
    strip=False,
    y_max=None,
    y_min=None,
    figsize=(10, 8),
    annotate_median=False,
    rotation=90,
    dotsize=3,
    custom="_",
    export_format="png",
    return_svg=False,
    annotatemultiplier=0.95,
):
    """
    Plot boxplots for the specified feature against a categorical x_category with custom order and styling options.

    Parameters
    ----------
    data_df : DataFrame
        DataFrame containing the data.
    feature : str
        The feature to plot on the y-axis.
    x_category : str
        The categorical feature to plot on the x-axis.
    font_size : int, optional
        Font size for the plot text. Default is 12.
    order : list, optional
        Specific order for the categories. Default is None.
    palette : str, optional
        Color palette for the plot. Default is 'colorblind'.
    background : str, tuple, optional
        Background color as a color name, RGB tuple, or hex code. Default is 'white'.
    transparent : bool, optional
        If True, makes the figure background transparent. Default is False.
    line_color : str, optional
        Color of all plot lines, including box edges, axis lines, grid lines, labels, and strip plot dots. Default is 'black'.
    show_plot : bool, optional
        Whether to display the plot in the notebook. Default is True.
    master_dir : str, optional
        Directory to save the plot. If not provided, defaults to "plots".
    grid : bool, optional
        Whether to display grid lines. Default is True.
    bw : bool, optional
        Whether to use black-and-white styling. Default is False.
    strip : bool, optional
        Whether to overlay a stripplot on the boxplot. Default is False.
    y_max : float, optional
        Maximum value for the y-axis. Default is None.
    figsize : tuple, optional
        Size of the plot (width, height) in inches. Default is (10, 8).
    annotate_median : bool, optional
        If True, annotate each boxplot with its median value. Default is False.
    rotation : int, optional
        Rotation angle for tick labels. Default is 90.
    dotsize : int, optional
        Size of the dots in the strip plot. Default is 3.
    export_format : str, optional
        File format to export the figure ('png' or 'svg'). Default is 'png'.
    return_svg : bool, optional
        If True and export_format is 'svg', returns the post-processed SVG image data as a string.

    Returns
    -------
    str or None
        When export_format is 'svg' and return_svg is True, returns the cleaned SVG data as a string.
        Otherwise, returns None.

    """
    # Validate and apply background color
    if is_color_like(background):
        figure_background = background
    else:
        print("Invalid color provided for background. Defaulting to white.")
        figure_background = "white"

    # Create figure and set background color
    fig, ax = plt.subplots(figsize=figsize, facecolor=figure_background)
    if not transparent:
        ax.set_facecolor(figure_background)
    else:
        fig.patch.set_alpha(0)
        ax.set_facecolor((0, 0, 0, 0))

    sns.set_context(
        "notebook", rc={"xtick.labelsize": font_size, "ytick.labelsize": font_size}
    )

    if bw:
        boxplot = sns.boxplot(
            x=x_category,
            y=feature,
            data=data_df,
            linewidth=1.5,
            showfliers=False,
            color="white",
            order=order,
        )
        for patch in boxplot.patches:
            patch.set_edgecolor(line_color)
            patch.set_linewidth(1.5)
        for element in ["boxes", "whiskers", "medians", "caps"]:
            plt.setp(boxplot.artists, color=line_color)
            plt.setp(boxplot.lines, color=line_color)
    else:
        boxplot = sns.boxplot(
            x=x_category,
            y=feature,
            data=data_df,
            palette=palette,
            order=order,
            showfliers=False,
            linewidth=1.5,
        )
        for patch in boxplot.patches:
            patch.set_edgecolor(line_color)
            patch.set_linewidth(1.5)
        for line in boxplot.lines:
            line.set_color(line_color)
            line.set_linewidth(1.5)

    if strip:
        sns.stripplot(
            x=x_category,
            y=feature,
            data=data_df,
            color=line_color,
            size=dotsize,
            order=order,
            jitter=True,
        )

    plt.xlabel("", fontsize=font_size, color=line_color)
    plt.ylabel(feature, fontsize=font_size, color=line_color)
    ax.tick_params(
        axis="both",
        which="both",
        color=line_color,
        labelcolor=line_color,
        labelsize=font_size,
        rotation=rotation,
    )

    if grid:
        ax.grid(
            True, linestyle="--", linewidth=0.5, color=line_color, alpha=0.7, axis="y"
        )

    if y_max is not None:
        plt.ylim(top=y_max)

    if y_min is not None:
        plt.ylim(bottom=y_min)

    for spine in ax.spines.values():
        spine.set_edgecolor(line_color)
        spine.set_linewidth(1.5)
        if transparent:
            spine.set_alpha(0.5)

    if annotate_median:
        medians = data_df.groupby(x_category)[feature].median()
        current_y_max = plt.ylim()[1]
        sorted_medians = medians.reindex(order) if order else medians
        for i, median in enumerate(sorted_medians):
            # plt.text(i, current_y_max * 0.965, f'{median:.2f}',
            plt.text(
                i,
                current_y_max * annotatemultiplier,
                f"{median:.2f}",
                horizontalalignment="center",
                size=font_size,
                color=line_color,
                weight="bold",
            )

    plt.tight_layout()

    if master_dir is None:
        master_dir = "plots"
    os.makedirs(master_dir, exist_ok=True)

    # condsindf = data_df['condition'].unique()

    ext = export_format.lower()
    if ext not in ["png", "svg"]:
        print("Invalid export format specified. Defaulting to 'png'.")
        ext = "png"
    filename = f"{master_dir}/{feature}_by_{x_category}_{custom}.{ext}"

    # Save the figure to file in the requested format
    plt.savefig(filename, bbox_inches="tight", transparent=transparent, format=ext)

    svg_data = None
    # If exporting to SVG, post-process the file to remove clipping paths.
    if ext == "svg":
        with open(filename, encoding="utf-8") as f:
            svg_data = f.read()
        # Remove any <clipPath> definitions and clip-path attributes.
        svg_data = re.sub(
            r'<clipPath id="[^"]*">.*?</clipPath>', "", svg_data, flags=re.DOTALL
        )
        svg_data = re.sub(r'\s*clip-path="url\(#.*?\)"', "", svg_data)
        # Write the cleaned SVG back to the file.
        with open(filename, "w", encoding="utf-8") as f:
            f.write(svg_data)

    if show_plot:
        plt.show()
    else:
        plt.close()

    if ext == "svg" and return_svg:
        return svg_data


def plot_stacked_bar(
    df,
    x_category,
    order=None,
    font_size=16,
    colormap="Dark2",
    figsize=(10, 8),
    background="white",
    transparent=False,
    line_color="black",
):
    """
    Plot a stacked bar chart showing the percentage of motion classes for each category on the x-axis.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the data with a 'motion_class' column and specified category.
    x_category : str
        The column name of the category to plot on the x-axis.
    order : list, optional
        Custom order for the categories on the x-axis. Default is None.
    font_size : int, optional
        Font size for the plot. Default is 16.
    colormap : str, optional
        Colormap for the plot. Default is 'Dark2'.
    figsize : tuple, optional
        Size of the plot (width, height) in inches. Default is (10, 5).
    background : str, optional
        Background color for the plot. Default is 'white'.
    transparent : bool, optional
        If True, makes the plot fully transparent except for the bars and text elements. Default is False.
    line_color : str, optional
        Color of all lines, including axis lines, bar outlines, and gridlines. Default is 'black'.

    """
    # Apply background and transparency
    if transparent:
        figure_background = "none"
        axis_background = (0, 0, 0, 0)  # Transparent background for axes
    else:
        figure_background = background
        axis_background = background

    # Apply custom order if provided
    if order is not None:
        df[x_category] = pd.Categorical(df[x_category], categories=order, ordered=True)

    # Calculate the percentage of each motion class within each category
    percentage_data = (
        df.groupby([x_category, "motion_class"])
        .size()
        .unstack(fill_value=0)
        .apply(lambda x: x / x.sum() * 100, axis=1)
    )

    # Determine the unique motion classes and assign colors
    motion_classes = df["motion_class"].unique()
    if colormap == "colorblind":
        colors = sns.color_palette("colorblind", len(motion_classes))
    elif colormap == "Dark2":
        cmap = cm.get_cmap("Dark2", len(motion_classes))
        colors = cmap(np.linspace(0, 1, len(motion_classes)))
    else:
        colors = plt.get_cmap(colormap, len(motion_classes)).colors

    # Plotting
    fig, ax = plt.subplots(figsize=figsize, facecolor=figure_background)
    ax.set_facecolor(axis_background)
    percentage_data.plot(
        kind="bar", stacked=True, ax=ax, color=colors, edgecolor=line_color
    )

    # Add black outlines to the bars
    for patch in ax.patches:
        patch.set_edgecolor(line_color)

    # Annotate percentages on the bars
    for patch in ax.patches:
        width, height = patch.get_width(), patch.get_height()
        x, y = patch.get_xy()
        if height > 0:  # Only annotate if there's a height to show
            ax.annotate(
                f"{height:.1f}%",
                (x + width / 2, y + height / 2),
                ha="center",
                va="center",
                fontsize=font_size,
                color=line_color,
            )

    # Customize text elements
    ax.set_title("Distribution of Motion Classes", fontsize=font_size, color=line_color)
    ax.set_xlabel("", fontsize=font_size, color=line_color)
    ax.set_ylabel("Percentage (%)", fontsize=font_size, color=line_color)
    ax.tick_params(
        axis="both",
        which="both",
        color=line_color,
        labelcolor=line_color,
        labelsize=font_size,
    )

    # Rotate x-tick labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Move the legend outside the plot
    legend = plt.legend(
        title="Motion Type",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        title_fontsize=font_size,
        prop={"size": font_size},
        frameon=False,
    )

    # Set legend text color
    for text in legend.get_texts():
        text.set_color(line_color)

    # Set title color
    legend.get_title().set_color(line_color)

    # Add grid with line color
    ax.grid(
        True, which="both", linestyle="--", linewidth=0.5, color=line_color, zorder=0
    )

    # Customize axis spines
    for spine in ax.spines.values():
        spine.set_edgecolor(line_color)
        if transparent:
            spine.set_alpha(0.5)

    plt.tight_layout()
    plt.show()


def plot_stacked_bar_svg(
    df,
    x_category,
    order=None,
    font_size=16,
    colormap="Dark2",
    figsize=(10, 8),
    background="white",
    transparent=False,
    line_color="black",
    export_format="png",
    master_dir=None,
    custom="_",
    show_plot=True,
    return_svg=False,
):
    """
    Plot a stacked bar chart showing the percentage of motion classes for each category on the x-axis.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the data with a 'motion_class' column and specified category.
    x_category : str
        The column name of the category to plot on the x-axis.
    order : list, optional
        Custom order for the categories on the x-axis. Default is None.
    font_size : int, optional
        Font size for the plot text. Default is 16.
    colormap : str, optional
        Colormap for the plot. Default is 'Dark2'. (If 'colorblind', uses seaborn's colorblind palette.)
    figsize : tuple, optional
        Size of the plot (width, height) in inches. Default is (10, 8).
    background : str, optional
        Background color for the figure. Default is 'white'.
    transparent : bool, optional
        If True, makes the plot background transparent. Default is False.
    line_color : str, optional
        Color of all lines (axis lines, bar outlines, gridlines, and text). Default is 'black'.
    export_format : str, optional
        File format to export the figure ('png' or 'svg'). Default is 'png'.
    master_dir : str, optional
        Directory to save the plot. If not provided, defaults to "plots".
    show_plot : bool, optional
        Whether to display the plot in the notebook. Default is True.
    return_svg : bool, optional
        If True and export_format is 'svg', returns the post-processed SVG image data as a string.

    Returns
    -------
    str or None
        When export_format is 'svg' and return_svg is True, returns the cleaned SVG data as a string.
        Otherwise, returns None.

    """
    # Apply background and transparency settings.
    if transparent:
        figure_background = "none"
        axis_background = (0, 0, 0, 0)  # Transparent background for axes
    else:
        figure_background = background
        axis_background = background

    # Apply custom order if provided.
    if order is not None:
        df[x_category] = pd.Categorical(df[x_category], categories=order, ordered=True)

    # Calculate percentage data for each motion class.
    percentage_data = (
        df.groupby([x_category, "motion_class"])
        .size()
        .unstack(fill_value=0)
        .apply(lambda x: x / x.sum() * 100, axis=1)
    )

    # Determine unique motion classes and assign colors.
    motion_classes = df["motion_class"].unique()
    if colormap == "colorblind":
        colors = sns.color_palette("colorblind", len(motion_classes))
    elif colormap == "Dark2":
        cmap = cm.get_cmap("Dark2", len(motion_classes))
        colors = cmap(np.linspace(0, 1, len(motion_classes)))
    else:
        colors = plt.get_cmap(colormap, len(motion_classes)).colors

    # Create figure and axes.
    fig, ax = plt.subplots(figsize=figsize, facecolor=figure_background)
    ax.set_facecolor(axis_background)

    # Plot the stacked bar chart.
    percentage_data.plot(
        kind="bar", stacked=True, ax=ax, color=colors, edgecolor=line_color
    )

    # Add black outlines to each bar.
    for patch in ax.patches:
        patch.set_edgecolor(line_color)

    # Annotate percentages on the bars.
    for patch in ax.patches:
        width, height = patch.get_width(), patch.get_height()
        x, y = patch.get_xy()
        if height > 0:  # Only annotate if there's a height to show.
            # ax.annotate(f'{height:.1f}%', (x + width / 2, y + height / 2),
            ax.annotate(
                f"{height:.1f}",
                (x + width / 2, y + height / 2),
                # ha='center', va='center', fontsize=font_size, color=line_color)
                ha="center",
                va="center",
                fontsize=font_size * 0.75,
                color=line_color,
            )

    # Customize text elements.
    ax.set_title("Distribution of Motion Classes", fontsize=font_size, color=line_color)
    ax.set_xlabel("", fontsize=font_size, color=line_color)
    ax.set_ylabel("Percentage (%)", fontsize=font_size, color=line_color)
    ax.tick_params(
        axis="both",
        which="both",
        color=line_color,
        labelcolor=line_color,
        labelsize=font_size,
    )

    # Rotate x-tick labels for readability.
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Move the legend outside the plot.
    legend = plt.legend(
        title="Motion Type",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        title_fontsize=font_size,
        prop={"size": font_size},
        frameon=False,
    )
    for text in legend.get_texts():
        text.set_color(line_color)
    legend.get_title().set_color(line_color)

    # Add grid.
    ax.grid(
        True, which="both", linestyle="--", linewidth=0.5, color=line_color, zorder=0
    )

    # Customize axis spines.
    for spine in ax.spines.values():
        spine.set_edgecolor(line_color)
        if transparent:
            spine.set_alpha(0.5)

    plt.tight_layout()

    # Saving/export functionality.
    svg_data = None
    master_dir = master_dir or "plots"
    os.makedirs(master_dir, exist_ok=True)
    ext = export_format.lower()
    if ext not in ["png", "svg"]:
        print("Invalid export format specified. Defaulting to 'png'.")
        ext = "png"
    filename = f"{master_dir}/stacked_bar_{x_category}_{custom}.{ext}"
    plt.savefig(filename, bbox_inches="tight", transparent=transparent, format=ext)

    if ext == "svg":
        with open(filename, encoding="utf-8") as f:
            svg_data = f.read()
        # Remove any <clipPath> definitions and clip-path attributes.
        svg_data = re.sub(
            r'<clipPath id="[^"]*">.*?</clipPath>', "", svg_data, flags=re.DOTALL
        )
        svg_data = re.sub(r'\s*clip-path="url\([^)]*\)"', "", svg_data)
        # Remove the <metadata> section.
        svg_data = re.sub(r"<metadata>.*?</metadata>", "", svg_data, flags=re.DOTALL)
        # Remove XML declaration and DOCTYPE.
        svg_data = re.sub(r"<\?xml[^>]*\?>", "", svg_data, flags=re.DOTALL)
        svg_data = re.sub(r"<!DOCTYPE[^>]*>", "", svg_data, flags=re.DOTALL)
        svg_data = svg_data.strip()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(svg_data)

    if show_plot:
        plt.show()
    else:
        plt.close()

    if ext == "svg" and return_svg:
        return svg_data


from PIL import Image


def plot_single_particle(particle_df, threshold_col, thresholds, animate=True):
    """
    Plots the trajectory of a single particle, either as an animation or as a static image,
    with each segment colored according to its speed category.

    Parameters
    - particle_df: DataFrame containing the particle's data with columns 'x', 'y', and 'speed_um_s'.
    - threshold_col: The column used to determine the thresholds for coloring.
    - thresholds: List or tuple of three numbers defining the boundaries for low, medium, and high categories.
    - animate: Boolean, if True creates an animation, if False creates a static plot.

    """
    # Get the unique ID of the particle
    unique_id = particle_df["unique_id"].unique()[0]

    # Ensure the thresholds are sorted
    thresholds = sorted(thresholds)

    # Assign categories based on thresholds
    conditions = [
        (particle_df[threshold_col] >= thresholds[0])
        & (particle_df[threshold_col] < thresholds[1]),
        (particle_df[threshold_col] >= thresholds[1])
        & (particle_df[threshold_col] < thresholds[2]),
        (particle_df[threshold_col] >= thresholds[2]),
    ]
    choices = ["low", "medium", "high"]
    factorcategory = f"{threshold_col}_category"
    particle_df[factorcategory] = np.select(conditions, choices, default="unknown")

    # Define a colormap for the categories
    # colormap = {
    #     'low': 'blue',
    #     'medium': 'green',
    #     'high': 'red'
    # }

    colormap = {
        "low": "#1F77B4",  # Hex color for 'low'
        "medium": "#FF7F0E",  # Hex color for 'medium'
        "high": "#2CA02C",  # Hex color for 'high'
    }

    # Calculate center and range for square plot
    center_x = (particle_df["x"].max() + particle_df["x"].min()) / 2
    center_y = (particle_df["y"].max() + particle_df["y"].min()) / 2
    range_extent = (
        max(
            particle_df["x"].max() - particle_df["x"].min(),
            particle_df["y"].max() - particle_df["y"].min(),
        )
        / 2
    )
    range_extent *= 1.1  # Add some padding

    # Create directory for saving video or images
    # dir_name = f'{config.MASTER}visualization/particle_{unique_id}_cat_{threshold_col}'
    # os.makedirs(dir_name, exist_ok=True)

    # if animate:
    #     dir_name = f'{config.MASTER}visualization/particle_{unique_id}_cat_{threshold_col}'
    #     os.makedirs(dir_name, exist_ok=True)
    # else:
    dir_name = rf"{config.MASTER}visualization\singleparticleplots"
    os.makedirs(dir_name, exist_ok=True)

    fontsizes = 16
    plt.rcParams.update({"font.size": fontsizes})
    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)

    ax.set_title(
        f"Particle Trajectory with {factorcategory} Categories: {unique_id}",
        fontsize=fontsizes,
    )
    ax.set_xlabel("X Position", fontsize=fontsizes)
    ax.set_ylabel("Y Position", fontsize=fontsizes)
    ax.set_xlim(center_x - range_extent, center_x + range_extent)
    ax.set_ylim(center_y - range_extent, center_y + range_extent)

    # Adjust the font size of the axis ticks
    ax.tick_params(axis="both", which="major", labelsize=fontsizes)

    # List to store frames
    frames = []

    def update_plot(i):
        ax.clear()
        ax.set_title(
            f"Particle Trajectory with {factorcategory} Categories: {unique_id}",
            fontsize=fontsizes,
        )
        ax.set_xlabel("X Position", fontsize=fontsizes)
        ax.set_ylabel("Y Position", fontsize=fontsizes)
        ax.set_xlim(center_x - range_extent, center_x + range_extent)
        ax.set_ylim(center_y - range_extent, center_y + range_extent)

        # Adjust the font size of the axis ticks in the update function as well
        ax.tick_params(axis="both", which="major", labelsize=fontsizes)

        # Plot the trajectory up to the current point, changing colors according to category
        for j in range(1, i + 1):
            x_values = particle_df["x"].iloc[j - 1 : j + 1]
            y_values = particle_df["y"].iloc[j - 1 : j + 1]
            fac_category = particle_df[factorcategory].iloc[j - 1]
            color = colormap.get(fac_category, "black")
            ax.plot(x_values, y_values, color=color, linewidth=2)

        ax.legend(
            handles=[
                plt.Line2D([0], [0], color=color, label=f"{category.capitalize()}")
                for category, color in colormap.items()
            ],
            fontsize=fontsizes,
        )

        # Save frame to list for GIF creation
        fig.canvas.draw()
        image = Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        frames.append(image)

    if animate:
        # Create animation frame by frame
        for i in range(1, len(particle_df)):
            update_plot(i)

        # Save the frames as a GIF
        gif_path = os.path.join(
            dir_name, f"particle_{unique_id}_cat_{threshold_col}.gif"
        )
        frames[0].save(
            gif_path, save_all=True, append_images=frames[1:], duration=50, loop=0
        )

        # frames[0].save('output.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)
    else:
        # Create static plot of the entire trajectory
        for i in range(1, len(particle_df)):
            x_values = particle_df["x"].iloc[i - 1 : i + 1]
            y_values = particle_df["y"].iloc[i - 1 : i + 1]
            fac_category = particle_df[factorcategory].iloc[i - 1]
            color = colormap.get(fac_category, "black")
            ax.plot(x_values, y_values, color=color, linewidth=2)

        ax.legend(
            handles=[
                plt.Line2D([0], [0], color=color, label=f"{category.capitalize()}")
                for category, color in colormap.items()
            ]
        )
        static_path = os.path.join(
            dir_name, f"static_particle_{unique_id}_cat_{threshold_col}.png"
        )
        plt.savefig(static_path)
        plt.show()

    plt.close(fig)


def plot_single_particle_wrapper(
    time_windowed_df,
    metrics_df,
    filter_col,
    low=None,
    high=None,
    condition=None,
    location=None,
    threshold_col="speed_um_s",
    thresholds=None,
    animate=False,
):
    """
    Wrapper function to filter dataframes, extract a single particle, and plot its track.

    Parameters
    - time_windowed_df (pd.DataFrame): The dataframe containing time-windowed data.
    - metrics_df (pd.DataFrame): The dataframe containing particle metrics.
    - filter_col (str): The column in time_windowed_df to filter by.
    - low (float, optional): The lower bound of the filter range. Default is None.
    - high (float, optional): The upper bound of the filter range. Default is None.
    - condition (Union[str, int], optional): The specific condition to filter by, or an index to choose from unique conditions. Default is None.
    - location (Union[str, int], optional): The specific location to filter by, or an index to choose from unique locations. Default is None.
    - threshold_col (str): The column to use for setting thresholds in the plot. Default is 'speed_um_s'.
    - thresholds (list, optional): List of thresholds for color-coding the plot. Default is None.
    - animate (bool, optional): Whether to animate the plot. Default is False.

    Returns
    - None: Displays and saves the plot.

    """
    # Step 1: Filter the time_windowed_df and extract unique IDs
    filtered_df, unique_ids = generalized_filter(
        time_windowed_df, filter_col, low, high, condition, location
    )

    # Step 2: Filter the metrics_df by the extracted unique IDs
    metrics_df_filtered = metrics_df[metrics_df["unique_id"].isin(unique_ids)]

    # Step 3: Extract a single particle track
    single_particle_df = extract_single_particle_df(metrics_df_filtered)
    if thresholds is None:
        thresholds = [0, 10, 15, 10000]

    # Step 4: Plot the single particle track
    plot_single_particle(single_particle_df, threshold_col, thresholds, animate)

    # Optional: Save the plot
    # plt.savefig(f'single_particle_plot_{single_particle_df["unique_id"].iloc[0]}.png')
    # plt.show()


def plot_jointplot(
    data_df,
    x_var,
    y_var,
    font_size=12,
    palette="colorblind",
    separate=None,
    show_plot=True,
    master_dir=None,
    grid=True,
    bw=False,
    y_min=None,
    y_max=None,
    x_min=None,
    x_max=None,
    figsize=(10, 8),
    order=None,
    kind="reg",
    height=7,
    color=None,
    point_size=50,
    small_multiples=False,
):
    sns.set_theme(style="darkgrid" if not bw else "whitegrid")

    # Set defaults for x_min and y_min if not provided
    x_min = x_min if x_min is not None else np.floor(data_df[x_var].min())
    y_min = y_min if y_min is not None else np.floor(data_df[y_var].min())

    # Set defaults for x_max and y_max if not provided
    x_max = x_max if x_max is not None else np.ceil(data_df[x_var].max())
    y_max = y_max if y_max is not None else np.ceil(data_df[y_var].max())

    # Calculate tick intervals based on the specified limits
    x_tick_interval = (x_max - x_min) / 10
    y_tick_interval = (y_max - y_min) / 10

    def add_r_squared(ax, x, y):
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r_squared = r_value**2
        ax.text(
            0.05,
            0.95,
            f"$R^2 = {r_squared:.2f}$",
            transform=ax.transAxes,
            fontsize=font_size,
            verticalalignment="top",
        )

    if separate is not None:
        if order:
            data_df[separate] = pd.Categorical(
                data_df[separate], categories=order, ordered=True
            )
        else:
            data_df[separate] = pd.Categorical(data_df[separate])

    if small_multiples and separate:
        unique_categories = data_df[separate].cat.categories
        colors = sns.color_palette(palette, len(unique_categories))

        for i, category in enumerate(unique_categories):
            subset = data_df[data_df[separate] == category]

            if kind == "hex":
                g = sns.jointplot(
                    x=x_var,
                    y=y_var,
                    data=subset,
                    kind=kind,
                    height=height,
                    color=colors[i],
                )
            else:
                g = sns.jointplot(
                    x=x_var,
                    y=y_var,
                    data=subset,
                    kind=kind,
                    height=height,
                    color=colors[i],
                    scatter_kws={"s": point_size},
                )

            g.fig.suptitle(f"{category}", fontsize=font_size + 2)
            g.ax_joint.set_xlabel(x_var, fontsize=font_size)
            g.ax_joint.set_ylabel(y_var, fontsize=font_size)

            g.ax_joint.set_xlim(left=x_min, right=x_max)
            g.ax_joint.set_ylim(bottom=y_min, top=y_max)
            g.ax_joint.set_xticks(
                np.arange(x_min, x_max + x_tick_interval, x_tick_interval)
            )
            g.ax_joint.set_yticks(
                np.arange(y_min, y_max + y_tick_interval, y_tick_interval)
            )
            add_r_squared(g.ax_joint, subset[x_var], subset[y_var])

            plt.tight_layout()

            if master_dir is None:
                master_dir = "plots"

            os.makedirs(master_dir, exist_ok=True)
            filename = f"{master_dir}/{category}_{y_var}_vs_{x_var}_jointplot.png"
            plt.savefig(filename, bbox_inches="tight")

            if show_plot:
                plt.show()
            else:
                plt.close()

    else:
        if separate:
            g = sns.FacetGrid(
                data_df, hue=separate, palette=palette, height=height, aspect=1.5
            )
            g.map(sns.regplot, x_var, y_var, scatter_kws={"s": point_size}, ci=None)
            g.add_legend()
            for ax in g.axes.flatten():
                for category in data_df[separate].unique():
                    subset = data_df[data_df[separate] == category]
                    add_r_squared(ax, subset[x_var], subset[y_var])

        else:
            g = sns.jointplot(
                x=x_var,
                y=y_var,
                data=data_df,
                kind=kind,
                height=height,
                color=color if color else sns.color_palette(palette, 1)[0],
                scatter_kws={"s": point_size},
            )

        g.ax_joint.set_xlim(left=x_min, right=x_max)
        g.ax_joint.set_ylim(bottom=y_min, top=y_max)
        g.ax_joint.set_xticks(
            np.arange(x_min, x_max + x_tick_interval, x_tick_interval)
        )
        g.ax_joint.set_yticks(
            np.arange(y_min, y_max + y_tick_interval, y_tick_interval)
        )

        add_r_squared(g.ax_joint, data_df[x_var], data_df[y_var])

        g.set_axis_labels(x_var, y_var, fontsize=font_size)
        plt.suptitle(f"{y_var} vs {x_var}", fontsize=font_size, y=1.02)

        plt.tight_layout()

        if master_dir is None:
            master_dir = "plots"

        os.makedirs(master_dir, exist_ok=True)
        filename = f"{master_dir}/{y_var}_vs_{x_var}_jointplot.png"
        plt.savefig(filename, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()


def plot_joint_with_fit(
    data_df,
    x_var,
    y_var,
    font_size=12,
    palette="colorblind",
    separate=None,
    show_plot=True,
    master_dir=None,
    grid=True,
    bw=False,
    y_max=None,
    x_max=None,
    figsize=(10, 8),
    tick_interval=5,
    order=None,
    fit_type="linear",
    scatter=True,
    kind="reg",
    height=7,
    color=None,
    point_size=50,
):
    """
    Plot a joint plot with a line of best fit (linear or other) for the specified x and y variables, with options for scatter and fit type.

    Parameters
    ----------
    data_df : DataFrame
        DataFrame containing the data.
    x_var : str
        The variable to plot on the x-axis.
    y_var : str
        The variable to plot on the y-axis.
    font_size : int, optional
        Font size for the plot text. Default is 12.
    palette : str, optional
        Color palette for the plot. Default is 'colorblind'.
    separate : str, optional
        Column to separate the data by. If None, all data will be plotted together. Default is None.
    show_plot : bool, optional
        Whether to display the plot in the notebook. Default is True.
    master_dir : str, optional
        Directory to save the plot. Default is None.
    grid : bool, optional
        Whether to display grid lines. Default is True.
    bw : bool, optional
        Whether to use black-and-white styling. Default is False.
    y_max : float, optional
        Maximum value for the y-axis. Default is None.
    x_max : float, optional
        Maximum value for the x-axis. Default is None.
    figsize : tuple, optional
        Size of the plot (width, height) in inches. Default is (10, 8).
    tick_interval : int, optional
        Interval for x-axis ticks. Default is 5.
    order : list, optional
        Specific order for the categories in the separate variable. Default is None.
    fit_type : str, optional
        Type of fit to apply ('linear', 'polynomial', 'exponential'). Default is 'linear'.
    scatter : bool, optional
        Whether to include the scatter plot along with the line of best fit. Default is True.
    kind : str, optional
        The kind of plot to draw ('scatter', 'reg', 'resid', 'kde', 'hex'). Default is 'reg'.
    height : float, optional
        Size of the figure (it will be square). Default is 7.
    color : str, optional
        Color of the regression line. Default is None.
    point_size : int, optional
        Size of the points in the scatter plot. Default is 50.

    """
    sns.set_theme(style="darkgrid")

    scatter_kws = {"s": point_size}

    # Apply the order if specified
    if separate and order:
        data_df[separate] = pd.Categorical(
            data_df[separate], categories=order, ordered=True
        )

    if separate:
        unique_categories = (
            data_df[separate].cat.categories if order else data_df[separate].unique()
        )
        colors = sns.color_palette(palette, len(unique_categories))

        for i, category in enumerate(unique_categories):
            subset = data_df[data_df[separate] == category]
            reg_color = colors[i] if color is None else color

            g = sns.jointplot(
                x=x_var,
                y=y_var,
                data=subset,
                kind=kind,
                height=height,
                color=reg_color,
                scatter_kws=scatter_kws,
            )

            if fit_type == "linear":
                sns.regplot(
                    x=x_var,
                    y=y_var,
                    data=subset,
                    scatter=scatter,
                    color=reg_color,
                    ci=None,
                    ax=g.ax_joint,
                )
            elif fit_type == "polynomial":
                sns.regplot(
                    x=x_var,
                    y=y_var,
                    data=subset,
                    scatter=scatter,
                    color=reg_color,
                    ci=None,
                    order=2,
                    ax=g.ax_joint,
                )
            elif fit_type == "exponential":
                log_y = np.log(subset[y_var])
                sns.regplot(
                    x=x_var,
                    y=log_y,
                    data=subset,
                    scatter=scatter,
                    color=reg_color,
                    ci=None,
                    ax=g.ax_joint,
                )
                g.ax_joint.set_ylabel(f"log({y_var})")

            if y_max is not None:
                g.ax_joint.set_ylim(top=y_max)
            if x_max is not None:
                g.ax_joint.set_xlim(right=x_max)
                g.ax_joint.set_xticks(np.arange(0, x_max + 1, tick_interval))

            g.ax_joint.set_xlabel(x_var, fontsize=font_size)
            g.ax_joint.set_ylabel(y_var, fontsize=font_size)
            g.ax_joint.set_title(
                f"{category}: {y_var} vs {x_var}", fontsize=font_size, pad=20
            )

            plt.tight_layout()

            if master_dir is None:
                master_dir = "plots"  # Default directory

            os.makedirs(master_dir, exist_ok=True)
            filename = f"{master_dir}/{category}_{y_var}_vs_{x_var}_jointplot_fit_{fit_type}.png"
            plt.savefig(filename, bbox_inches="tight")

            if show_plot:
                plt.show()
            else:
                plt.close()

    else:
        g = sns.jointplot(
            x=x_var,
            y=y_var,
            data=data_df,
            kind=kind,
            height=height,
            color=color if color else sns.color_palette(palette, 1)[0],
            scatter_kws=scatter_kws,
        )

        if fit_type == "linear":
            sns.regplot(
                x=x_var,
                y=y_var,
                data=data_df,
                scatter=scatter,
                color=color if color else sns.color_palette(palette, 1)[0],
                ci=None,
                ax=g.ax_joint,
            )
        elif fit_type == "polynomial":
            sns.regplot(
                x=x_var,
                y=y_var,
                data=data_df,
                scatter=scatter,
                color=color if color else sns.color_palette(palette, 1)[0],
                ci=None,
                order=2,
                ax=g.ax_joint,
            )
        elif fit_type == "exponential":
            log_y = np.log(data_df[y_var])
            sns.regplot(
                x=x_var,
                y=log_y,
                data=data_df,
                scatter=scatter,
                color=color if color else sns.color_palette(palette, 1)[0],
                ci=None,
                ax=g.ax_joint,
            )
            g.ax_joint.set_ylabel(f"log({y_var})")

        if y_max is not None:
            g.ax_joint.set_ylim(top=y_max)
        if x_max is not None:
            g.ax_joint.set_xlim(right=x_max)
            g.ax_joint.set_xticks(np.arange(0, x_max + 1, tick_interval))

        g.ax_joint.set_xlabel(x_var, fontsize=font_size)
        g.ax_joint.set_ylabel(y_var, fontsize=font_size)
        g.ax_joint.set_title(f"{y_var} vs {x_var}", fontsize=font_size, pad=20)

        plt.tight_layout()

        if master_dir is None:
            master_dir = "plots"  # Default directory

        os.makedirs(master_dir, exist_ok=True)
        filename = f"{master_dir}/{y_var}_vs_{x_var}_jointplot_fit_{fit_type}.png"
        plt.savefig(filename, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()


def plot_combo_hist_scatter_kde(
    data_df,
    x_var,
    y_var,
    font_size=12,
    palette="mako",
    scatter_color=".15",
    hist_bins=50,
    kde_levels=5,
    figsize=(6, 6),
    separate=None,
    order=None,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    horizontal=False,
):
    """
    Draw a combination of histogram, scatterplot, and density contours with optional separation into subplots.

    Parameters
    ----------
    data_df : DataFrame
        DataFrame containing the data.
    x_var : str
        The feature to plot on the x-axis.
    y_var : str
        The feature to plot on the y-axis.
    font_size : int, optional
        Font size for the plot text, including axis labels, ticks, and titles. Default is 12.
    palette : str, optional
        Color palette for the histogram and KDE plot. Default is 'mako'.
    scatter_color : str, optional
        Color for the scatter plot points. Default is ".15".
    hist_bins : int, optional
        Number of bins for the histogram. Default is 50.
    kde_levels : int, optional
        Number of contour levels for the KDE plot. Default is 5.
    figsize : tuple, optional
        Size of the plot (width, height) in inches. Default is (6, 6).
    separate : str, optional
        Column name to separate the data by. Creates subplots for each category. Default is None.
    order : list, optional
        Specific order for the categories in the subplots. Default is None.
    x_min : float, optional
        Minimum value for the x-axis. Default is None.
    x_max : float, optional
        Maximum value for the x-axis. Default is None.
    y_min : float, optional
        Minimum value for the y-axis. Default is None.
    y_max : float, optional
        Maximum value for the y-axis. Default is None.
    horizontal : bool, optional
        If True, the subplots will be arranged horizontally. Default is False.

    """
    if separate is not None:
        if order:
            data_df[separate] = pd.Categorical(
                data_df[separate], categories=order, ordered=True
            )
        else:
            data_df[separate] = pd.Categorical(data_df[separate])

        unique_categories = data_df[separate].cat.categories
        num_categories = len(unique_categories)

        # Create subplots horizontally or vertically based on the 'horizontal' flag
        if horizontal:
            fig, axes = plt.subplots(
                ncols=num_categories, figsize=(figsize[0] * num_categories, figsize[1])
            )
        else:
            fig, axes = plt.subplots(
                nrows=num_categories, figsize=(figsize[0], figsize[1] * num_categories)
            )

        if num_categories == 1:
            axes = [axes]  # Ensure axes is iterable if there's only one subplot

        for i, category in enumerate(unique_categories):
            subset = data_df[data_df[separate] == category]
            ax = axes[i]

            # Scatterplot
            sns.scatterplot(
                x=x_var, y=y_var, data=subset, s=5, color=scatter_color, ax=ax
            )

            # 2D Histogram
            sns.histplot(
                x=x_var,
                y=y_var,
                data=subset,
                bins=hist_bins,
                pthresh=0.1,
                cmap=palette,
                ax=ax,
            )

            # KDE Plot
            sns.kdeplot(
                x=x_var,
                y=y_var,
                data=subset,
                levels=kde_levels,
                color="w",
                linewidths=1,
                ax=ax,
            )

            ax.set_xlabel(x_var, fontsize=font_size)
            ax.set_ylabel(y_var, fontsize=font_size)
            ax.set_title(f"{category}", fontsize=font_size + 2)

            ax.tick_params(axis="both", labelsize=font_size)

            if x_min is not None and x_max is not None:
                ax.set_xlim(x_min, x_max)
            if y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)

        plt.tight_layout()
        plt.show()

    else:
        f, ax = plt.subplots(figsize=figsize)

        # Scatterplot
        sns.scatterplot(x=x_var, y=y_var, data=data_df, s=5, color=scatter_color, ax=ax)

        # 2D Histogram
        sns.histplot(
            x=x_var,
            y=y_var,
            data=data_df,
            bins=hist_bins,
            pthresh=0.1,
            cmap=palette,
            ax=ax,
        )

        # KDE Plot
        sns.kdeplot(
            x=x_var,
            y=y_var,
            data=data_df,
            levels=kde_levels,
            color="w",
            linewidths=1,
            ax=ax,
        )

        ax.set_xlabel(x_var, fontsize=font_size)
        ax.set_ylabel(y_var, fontsize=font_size)
        ax.set_title(f"{y_var} vs {x_var} with Hist and KDE", fontsize=font_size + 2)

        ax.tick_params(axis="both", labelsize=font_size)

        if x_min is not None and x_max is not None:
            ax.set_xlim(x_min, x_max)
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)

        plt.tight_layout()
        plt.show()


def plot_tracks_by_motion_class(
    time_windowed_df,
    metrics_df,
    num_tracks=10,
    colormap="Dark2",
    axis_range=None,
    show_annotations=False,
    order=None,
    transparent_background=False,
    annotation_color="white",
    text_size=10,
    figsizemultiplier=5,  # Overall figure size multiplier for adaptable subplot size
    time_window=config.TIME_WINDOW,
    overlap=config.OVERLAP,
):
    # Enforce numeric data types to avoid memory-related inconsistencies. Added this in because of a weird bug where this would work on pd read dataframes but not newly created ones.
    for col in ["x_um_start", "y_um_start", "x_um", "y_um"]:
        if col in time_windowed_df:
            time_windowed_df[col] = pd.to_numeric(
                time_windowed_df[col], errors="coerce"
            )
        if col in metrics_df:
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")

    # Reset indices to ensure consistency
    time_windowed_df = time_windowed_df.reset_index(drop=True)
    metrics_df = metrics_df.reset_index(drop=True)

    # Use the specified order for motion classes, or get unique classes from the data
    motion_classes = order if order else time_windowed_df["motion_class"].unique()
    print(f"Plotting motion classes in the following order: {motion_classes}")

    # Assign colors based on the order of motion classes
    if colormap == "colorblind":
        colors = sns.color_palette("colorblind", len(motion_classes))
    elif colormap == "Dark2":
        cmap = cm.get_cmap("Dark2", len(motion_classes))
        colors = cmap(np.linspace(0, 1, len(motion_classes)))
    else:
        colors = plt.get_cmap(colormap, len(motion_classes)).colors

    motion_color_map = {
        motion_class: colors[i] for i, motion_class in enumerate(motion_classes)
    }

    # Collect all selected track segments for range calculation
    track_segments = []
    track_info = []  # To store unique_id, time_window, and anomalous_exponent for annotations
    for motion_class in motion_classes:
        # Filter by motion class and pick unique IDs
        class_df = time_windowed_df[time_windowed_df["motion_class"] == motion_class]
        unique_ids = class_df["unique_id"].unique()

        # Randomly select num_tracks unique IDs for this motion class
        selected_ids = random.sample(list(unique_ids), min(num_tracks, len(unique_ids)))

        for unique_id in selected_ids:
            # Filter to find the time windows for this unique ID and motion class
            track_df = class_df[class_df["unique_id"] == unique_id].sample(
                n=1
            )  # Pick one random time window

            # Extract starting x, y, time window, and anomalous exponent for the selected track segment
            x_start = track_df["x_um_start"].values[0]
            y_start = track_df["y_um_start"].values[0]
            time_window_id = track_df["time_window"].values[0]
            anomalous_exponent = track_df["anomalous_exponent"].values[0]

            # Find the starting point in metrics_df
            start_index = metrics_df[
                (metrics_df["unique_id"] == unique_id)
                & (metrics_df["x_um"] == x_start)
                & (metrics_df["y_um"] == y_start)
            ].index

            # Skip if no matching start index is found
            if len(start_index) == 0:
                continue

            # Extract the segment based on the time_window length
            start_index = start_index[0]
            metrics_track_segment = metrics_df.iloc[
                start_index : start_index + time_window
            ]

            # Check if we have exactly time_window frames; if not, skip this track
            if len(metrics_track_segment) < time_window:
                continue

            # Append to the track segments list
            track_segments.append(metrics_track_segment[["x_um", "y_um"]])
            track_info.append(
                (
                    motion_class,
                    unique_id,
                    time_window_id,
                    anomalous_exponent,
                    x_start,
                    y_start,
                )
            )  # Store info for annotations

    # Determine global axis range if not provided
    if axis_range is None:
        x_ranges = [
            segment["x_um"].max() - segment["x_um"].min() for segment in track_segments
        ]
        y_ranges = [
            segment["y_um"].max() - segment["y_um"].min() for segment in track_segments
        ]
        max_range = max(max(x_ranges), max(y_ranges))
        axis_range = max_range  # Use this as the global axis range

    # Adjust layout to keep each motion class in columns with a maximum of 10 tracks per column
    columns_per_class = math.ceil(num_tracks / 10)  # Number of columns per motion class
    fig_cols = len(motion_classes) * columns_per_class
    fig_rows = min(num_tracks, 10)

    # Set up figure with optional transparency and adaptable subplot size
    figure_background = "none" if transparent_background else "white"
    axis_background = (0, 0, 0, 0) if transparent_background else "white"

    # Calculate adaptable figsize based on rows and columns
    fig, axes = plt.subplots(
        fig_rows,
        fig_cols,
        figsize=(figsizemultiplier * fig_cols, figsizemultiplier * fig_rows / 2),
        facecolor=figure_background,
    )
    fig.subplots_adjust(wspace=0.0001, hspace=0.0001)  # Tighter spacing

    if len(motion_classes) == 1:
        axes = [axes]

    # Plot each track segment
    for idx, (segment, info) in enumerate(
        zip(track_segments, track_info, strict=False)
    ):
        (
            motion_class,
            unique_id,
            time_window_id,
            anomalous_exponent,
            x_start,
            y_start,
        ) = info
        x_coords = segment["x_um"].values
        y_coords = segment["y_um"].values

        # Calculate centroid for the current track segment
        x_centroid = x_coords.mean()
        y_centroid = y_coords.mean()

        # Calculate subplot column and row index
        j = (
            list(motion_classes).index(motion_class) * columns_per_class
            + (idx // fig_rows) % columns_per_class
        )
        i = idx % fig_rows

        # Plot the track on the subplot
        ax = axes[i, j] if len(motion_classes) > 1 else axes[j]
        ax.plot(x_coords, y_coords, color=motion_color_map[motion_class], linewidth=1.5)

        # Set axis limits centered around the centroid with the global range
        ax.set_xlim(x_centroid - axis_range / 2, x_centroid + axis_range / 2)
        ax.set_ylim(y_centroid - axis_range / 2, y_centroid + axis_range / 2)

        # Set transparent axis background if requested
        ax.set_facecolor(axis_background)  # Transparent axis background

        # Remove all plot aesthetics
        ax.axis("off")  # Turn off the axis, including ticks and lines

        # Add motion class label only to the first plot in each set of columns
        if i == 0 and (idx // fig_rows) % columns_per_class == 0:
            ax.text(
                0.5,
                1.15,
                motion_class,
                ha="center",
                va="top",
                transform=ax.transAxes,
                fontsize=text_size,
                weight="bold",
                color=annotation_color,
            )

        # Optionally add annotations for unique_id, time window, and anomalous exponent
        if show_annotations:
            ax.text(
                0.5,
                1.05,
                f"{unique_id}\nTW: {time_window_id}\nα: {anomalous_exponent:.2f}",
                ha="center",
                va="top",
                transform=ax.transAxes,
                fontsize=text_size,
                color=annotation_color,
            )

    # Prepare the DataFrame with plotted tracks info
    plotted_info_df = pd.DataFrame(
        track_info,
        columns=[
            "motion_class",
            "unique_id",
            "time_window",
            "anomalous_exponent",
            "x_start",
            "y_start",
        ],
    )

    # Show the minimalist plot
    plt.show()

    return plotted_info_df


def plot_tracks_static(
    tracks_df,
    filename=None,
    file_id=None,
    location=None,
    condition=None,
    time_start=None,
    time_end=None,
    color_by="particle",
    motion_type=None,  # New parameter
    overlay_image=False,
    master_dir=config.MASTER,
    scale_bar_length=5,
    scale_bar_position=(0.9, 0.1),
    scale_bar_color="white",
    transparent_background=True,
    save_path=None,
    display_final_frame=True,
    max_projection=False,
    contrast_limits=None,  # Tuple: (lower, upper) or None for auto
    invert_image=False,
    pixel_size_um=config.PIXELSIZE_MICRONS,  # Conversion factor: microns per pixel
    frame_interval=config.TIME_BETWEEN_FRAMES,
    gradient=False,  # Frame interval in seconds
    colorway="tab20",
    order=None,  # New parameter: Order for categorical coloring
    plot_size_px=150,
    dpi=100,
):
    """
    Create a static plot of tracks filtered by file and condition within a time range.
    Adds:
    - Support for `order` to specify a custom ordering of colors for categorical `color_by`.
    """
    # Default figure size and font scaling
    figsize = (8, 8)
    # base_figsize = (8, 8)
    # figsize = (base_figsize[0] * figsize_multiplier, base_figsize[1] * figsize_multiplier)

    # # Update font sizes
    # plt.rcParams.update({
    #     'font.size': 10 * figsize_multiplier,
    #     'axes.titlesize': 12 * figsize_multiplier,
    #     'axes.labelsize': 12 * figsize_multiplier,
    #     'xtick.labelsize': 10 * figsize_multiplier,
    #     'ytick.labelsize': 10 * figsize_multiplier,
    # })

    # Map `color_by` to numeric if it contains strings or is categorical
    if tracks_df[color_by].dtype == object or pd.api.types.is_categorical_dtype(
        tracks_df[color_by]
    ):
        # Apply `order` if provided
        if order is None:
            unique_classes = tracks_df[color_by].unique()
        else:
            unique_classes = order
        class_to_int = {cls: i for i, cls in enumerate(unique_classes)}
        tracks_df["segment_color"] = tracks_df[color_by].map(
            class_to_int
        )  # New numeric column
    else:
        tracks_df["segment_color"] = tracks_df[
            color_by
        ]  # Use original column if numeric

    # Pre-map motion types to consistent colors if color_by is motion_class
    if color_by == "motion_class":
        if order is None:
            unique_classes = tracks_df[color_by].unique()
        else:
            unique_classes = order
        class_to_color = {
            cls: plt.cm.get_cmap(colorway)(i / max(len(unique_classes) - 1, 1))
            for i, cls in enumerate(unique_classes)
        }
        tracks_df["motion_color"] = tracks_df["motion_class"].map(class_to_color)

        # Pre-map motion types to consistent colors if `motion_type` filtering is enabled
    if motion_type is not None or color_by == "motion_class":
        if order is None:
            unique_classes = [
                "subdiffusive",
                "normal",
                "superdiffusive",
            ]  # Default order
        else:
            unique_classes = order
        class_to_color = {
            cls: plt.cm.get_cmap(colorway)(i / max(len(unique_classes) - 1, 1))
            for i, cls in enumerate(unique_classes)
        }
        tracks_df["motion_color"] = tracks_df["motion_class"].map(class_to_color)

    # Filter by motion type
    if motion_type is not None:
        if "motion_color" not in tracks_df.columns:
            raise ValueError(
                "motion_color column not pre-mapped; ensure color_by='motion_class'."
            )
        tracks_df = tracks_df[tracks_df["motion_class"] == motion_type]

    # Filter by location
    if location is None:
        location_col = _get_location_column(tracks_df)
        location = np.random.choice(tracks_df[location_col].unique())
    else:
        location_col = _get_location_column(tracks_df)
    tracks_df = tracks_df[tracks_df[location_col] == location]

    # Filter by condition
    if condition is None:
        condition = np.random.choice(tracks_df["condition"].unique())
    tracks_df = tracks_df[tracks_df["condition"] == condition]

    # Filter by filename or file_id
    if filename is None and file_id is None:
        filename = np.random.choice(tracks_df["filename"].unique())
    elif file_id is not None:
        filename = tracks_df[tracks_df["file_id"] == file_id]["filename"].iloc[0]
    tracks_df = tracks_df[tracks_df["filename"] == filename]

    # Convert time_start and time_end from seconds to frames if provided
    min_frame = tracks_df["frame"].min()
    max_frame = tracks_df["frame"].max()

    if time_start is not None:
        time_start_frames = int(time_start / frame_interval)
        time_start_frames = max(min_frame, min(time_start_frames, max_frame))
    else:
        time_start_frames = min_frame

    if time_end is not None:
        time_end_frames = int(time_end / frame_interval)
        time_end_frames = max(min_frame, min(time_end_frames, max_frame))
    else:
        time_end_frames = max_frame

    # Add this immediately after
    time_start_sec = time_start_frames * frame_interval
    time_end_sec = time_end_frames * frame_interval

    # Filter by adjusted time_start and time_end
    tracks_df = tracks_df[
        (tracks_df["frame"] >= time_start_frames)
        & (tracks_df["frame"] <= time_end_frames)
    ]

    # Check if any data is left after filtering
    if tracks_df.empty:
        raise ValueError(
            "No valid data available for plotting after filtering by filename and time range."
        )

    # Set figure and axis background based on transparency setting
    figure_background = "none" if transparent_background else "white"
    axis_background = (0, 0, 0, 0) if transparent_background else "white"

    # Create a plot
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    fig.patch.set_facecolor(figure_background)
    ax.set_facecolor(axis_background)

    # Overlay the image if requested
    if overlay_image:
        image_filename = filename.replace("_tracked", "") + ".tif"
        image_path = os.path.join(master_dir, "data", condition, image_filename)
        overlay_data = imread(image_path)  # Load image as 3D array

        if max_projection:
            # Maximum intensity projection
            overlay_data = np.max(overlay_data, axis=0)
        elif display_final_frame:
            # Display the final frame
            overlay_data = overlay_data[-1, :, :]

        # Normalize the image for contrast limits
        overlay_data = img_as_float(overlay_data)  # Convert to float for scaling
        if contrast_limits:
            lower, upper = contrast_limits
            overlay_data = np.clip((overlay_data - lower) / (upper - lower), 0, 1)
        else:
            overlay_data = (overlay_data - overlay_data.min()) / (
                overlay_data.max() - overlay_data.min()
            )

        if invert_image:
            overlay_data = 1 - overlay_data  # Invert image intensity

        # Compute image extent in microns
        height, width = overlay_data.shape
        extent = [0, width * pixel_size_um, 0, height * pixel_size_um]

        # Display the image with correct scaling
        ax.imshow(overlay_data, cmap="gray", origin="lower", extent=extent)

    # Plot tracks colored by the specified column and add directionality
    unique_ids = tracks_df["particle"].unique()

    # Plot tracks with transparency gradient
    for uid in unique_ids:
        track = tracks_df[tracks_df["particle"] == uid]
        n_points = len(track)
        alphas = np.linspace(0.4, 1.0, n_points)  # Transparency gradient: 40% to 100%

        for i in range(n_points - 1):
            # line_color = (
            #     plt.cm.inferno(i / n_points) if gradient else
            #     plt.cm.get_cmap(colorway)(track['segment_color'].iloc[i] % 20 / 20)
            # )
            # line_color = (
            #     plt.cm.inferno(i / n_points) if gradient else
            #     plt.cm.get_cmap(colorway)(track['segment_color'].iloc[i] / max(len(unique_classes) - 1, 1))
            # )
            if tracks_df[color_by].dtype == object or pd.api.types.is_categorical_dtype(
                tracks_df[color_by]
            ):
                num_classes = max(len(unique_classes) - 1, 1)
            else:
                num_classes = max(
                    tracks_df["segment_color"].max() - tracks_df["segment_color"].min(),
                    1,
                )

            # line_color = (
            #     plt.cm.inferno(i / n_points) if gradient else
            #     plt.cm.get_cmap(colorway)(track['segment_color'].iloc[i] / num_classes)
            # )
            # Normalize `segment_color` values to the range [0, 1]
            # normalized_color = (track['segment_color'].iloc[i] - tracks_df['segment_color'].min()) / num_classes

            # line_color = (
            #     plt.cm.inferno(i / n_points) if gradient else
            #     plt.cm.get_cmap(colorway)(normalized_color)
            # )

            ####################
            if color_by == "motion_class":
                line_color = track["motion_color"].iloc[i]
            else:
                normalized_color = (
                    track["segment_color"].iloc[i] - tracks_df["segment_color"].min()
                ) / num_classes
                line_color = (
                    plt.cm.inferno(i / n_points)
                    if gradient
                    else plt.cm.get_cmap(colorway)(normalized_color)
                )

            ax.plot(
                track.iloc[i : i + 2]["x_um"],
                track.iloc[i : i + 2]["y_um"],
                color=line_color,
                alpha=alphas[i],  # Transparency gradient
                linewidth=0.1 + i * 0.1
                if gradient
                else 1,  # Tapered width for gradient
            )
    #######################

    # Remove axes if no overlay image
    if not overlay_image:
        ax.axis("off")

    if scale_bar_length:
        # Define relative position in the axes space
        bar_x_end = 0.95  # 95% from the left
        bar_x_start = bar_x_end - (
            scale_bar_length / (tracks_df["x_um"].max() - tracks_df["x_um"].min())
        )
        bar_y = 0.05  # 5% from the bottom

        ax.plot(
            [bar_x_start, bar_x_end],
            [bar_y, bar_y],
            transform=ax.transAxes,
            color=scale_bar_color,
            lw=3,
        )

        text_x = (bar_x_start + bar_x_end) / 2
        text_y = bar_y - 0.025
        ax.text(
            text_x,
            text_y,
            f"{scale_bar_length} µm",
            transform=ax.transAxes,
            color=scale_bar_color,
            ha="center",
            va="top",
            fontsize=10,
        )

    # Add time range annotation
    ax.annotate(
        f"Time: {time_start_sec:.2f}s - {time_end_sec:.2f}s",
        xy=(0.5, 1.02),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=12,
        color=scale_bar_color,
    )

    # Set plot limits based on data
    ax.set_xlim([tracks_df["x_um"].min(), tracks_df["x_um"].max()])
    ax.set_ylim([tracks_df["y_um"].min(), tracks_df["y_um"].max()])

    # Standardize plot size to 150x150 pixels (converted to microns)
    plot_size_microns = plot_size_px * pixel_size_um
    ax.set_xlim(0, plot_size_microns)
    ax.set_ylim(0, plot_size_microns)

    # Ensure the aspect ratio is square
    ax.set_aspect("equal", adjustable="datalim")

    # Add a legend only for categorical or string-based coloring
    if tracks_df[color_by].dtype == object or pd.api.types.is_categorical_dtype(
        tracks_df[color_by]
    ):
        handles = [
            plt.Line2D(
                [0],
                [0],
                color=plt.cm.get_cmap(colorway)(i / max(len(unique_classes) - 1, 1)),
                lw=2,
                label=cls,
            )
            for i, cls in enumerate(unique_classes)
        ]
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0.0,
            title=f"Legend: {color_by}",
        )

    # No legend for numeric `color_by` like 'particle'
    # Add a colorbar for continuous `color_by` values
    if not (
        tracks_df[color_by].dtype == object
        or pd.api.types.is_categorical_dtype(tracks_df[color_by])
    ):
        color_min = tracks_df[color_by].min()
        color_max = tracks_df[color_by].max()

        print(
            f"Colorbar range for '{color_by}': Min={round(color_min, 2)}, Max={round(color_max, 2)}"
        )

        sm = plt.cm.ScalarMappable(
            cmap=colorway, norm=plt.Normalize(vmin=color_min, vmax=color_max)
        )
        sm.set_array([])

        # Add the colorbar outside the plot area
        cbar = plt.colorbar(
            sm,
            ax=ax,
            orientation="vertical",
            pad=0.1,  # Distance from the plot
            fraction=0.03,  # Width of the colorbar as a fraction of the plot
            shrink=0.25,  # Length of the colorbar as a fraction of the plot height
        )
        cbar.set_label(
            f"{color_by} (range: {round(color_min, 2)} - {round(color_max, 2)})",
            color=scale_bar_color,
        )
        cbar.ax.yaxis.set_tick_params(color=scale_bar_color)
        cbar.ax.yaxis.set_tick_params(labelsize=10)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=scale_bar_color)

        # Move colorbar outside plot
        cbar.ax.set_position([0.85, 0.35, 0.05, 0.3])

    # Save or show plot
    if save_path:
        # make the directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        #

        plt.savefig(save_path + filename, transparent=transparent_background, dpi=dpi)
    else:
        plt.show()

    return tracks_df


def plot_tracks_static_svg(
    tracks_df,
    filename=None,
    file_id=None,
    location=None,
    condition=None,
    time_start=None,
    time_end=None,
    color_by="particle",
    motion_type=None,  # New parameter
    overlay_image=False,
    master_dir=config.MASTER,
    scale_bar_length=2,  # in microns
    scale_bar_color="black",
    scale_bar_thickness=2,  # thickness of the scale bar
    transparent_background=True,
    save_path=None,
    display_final_frame=True,
    max_projection=False,
    contrast_limits=None,  # Tuple: (lower, upper) or None for auto
    invert_image=False,
    pixel_size_um=config.PIXELSIZE_MICRONS,  # microns per pixel conversion factor
    frame_interval=config.TIME_BETWEEN_FRAMES,
    gradient=False,  # (gradient effect not applied when drawing a single polyline)
    colorway="tab20",
    order=[
        "subdiffusive",
        "normal",
        "superdiffusive",
    ],  # New parameter: Order for categorical coloring
    figsize=(3, 3),  # figure size in inches
    plot_size_um=10,  # final data range (in microns)
    line_thickness=0.6,  # thickness of track lines
    dpi=200,
    export_format="svg",  # 'png' or 'svg'
    return_svg=False,  # if True and exporting as SVG, return the SVG string
    show_plot=True,  # whether to show the plot after saving/exporting
):
    """
    Create a static plot of tracks filtered by file and condition within a time range.

    Features added:
      - Custom ordering of colors via the 'order' parameter.
      - Export as SVG with post-processing to remove clipping paths/metadata.
      - Each particle's track is plotted as a single polyline (with a unique gid) so that in Illustrator it appears as one object.
      - The scale bar is drawn in data coordinates so that its length is accurate.
      - Plot size is defined in microns (plot_size_um) rather than pixels.
      - Line thickness (for tracks) and scale bar thickness can be customized.
      - **Text scaling:** The default text size is 12 when figsize is (8,8) inches; when you change figsize, the text sizes scale proportionally.
    """
    # --- Text Scaling Block ---
    # Define the baseline figure width and font size.
    baseline_width = 8.0  # inches
    baseline_font = 14  # default text size for an 8x8 figure
    scale_factor = figsize[0] / baseline_width
    default_font = baseline_font * scale_factor  # scaled default font size

    # Optionally, update rcParams (this update is global)
    plt.rcParams.update(
        {
            "font.size": default_font,
            "axes.titlesize": default_font,
            "axes.labelsize": default_font,
            "xtick.labelsize": default_font,
            "ytick.labelsize": default_font,
        }
    )

    # Make a copy to avoid SettingWithCopyWarning
    tracks_df = tracks_df.copy()

    # Ensure 'frame' is numeric for proper filtering.
    tracks_df["frame"] = pd.to_numeric(tracks_df["frame"], errors="coerce")

    # Map `color_by` to numeric if it contains strings or is categorical.
    # Use consistent cluster ordering logic across all functions
    is_cluster_column = (
        color_by in ["cluster", "cluster_id"] or "cluster" in color_by.lower()
    )
    is_categorical = (
        tracks_df[color_by].dtype == object
        or pd.api.types.is_categorical_dtype(tracks_df[color_by])
        or is_cluster_column
    )

    if is_categorical:
        # Get unique values and sort them for consistent ordering
        all_unique_values = sorted(
            [v for v in tracks_df[color_by].unique() if pd.notna(v)]
        )

        if order is None:
            unique_classes = all_unique_values
        else:
            # Use provided order, but only include values that exist in data
            unique_classes = [c for c in order if c in all_unique_values]
            # Add any remaining values not in the provided order
            remaining_values = [c for c in all_unique_values if c not in unique_classes]
            unique_classes.extend(remaining_values)

        class_to_int = {cls: i for i, cls in enumerate(unique_classes)}
        tracks_df["segment_color"] = tracks_df[color_by].map(class_to_int)
    else:
        tracks_df["segment_color"] = tracks_df[color_by]

    # Pre-map motion types to consistent colors if color_by is 'motion_class'.
    if color_by == "motion_class":
        if order is None:
            unique_classes = tracks_df[color_by].unique()
        else:
            unique_classes = order
        class_to_color = {
            cls: plt.cm.get_cmap(colorway)(i / max(len(unique_classes) - 1, 1))
            for i, cls in enumerate(unique_classes)
        }
        tracks_df["motion_color"] = tracks_df["motion_class"].map(class_to_color)

    # If motion_type filtering is enabled, re-map colors using a default order.
    if motion_type is not None or color_by == "motion_class":
        if order is None:
            unique_classes = [
                "subdiffusive",
                "normal",
                "superdiffusive",
            ]  # Default order
        else:
            unique_classes = order
        class_to_color = {
            cls: plt.cm.get_cmap(colorway)(i / max(len(unique_classes) - 1, 1))
            for i, cls in enumerate(unique_classes)
        }
        tracks_df["motion_color"] = tracks_df["motion_class"].map(class_to_color)

    # Filter by motion type.
    if motion_type is not None:
        if "motion_color" not in tracks_df.columns:
            raise ValueError(
                "motion_color column not pre-mapped; ensure color_by='motion_class'."
            )
        tracks_df = tracks_df[tracks_df["motion_class"] == motion_type]

    # Filter by location.
    if location is None:
        location_col = _get_location_column(tracks_df)
        location = np.random.choice(tracks_df[location_col].unique())
    else:
        location_col = _get_location_column(tracks_df)
    tracks_df = tracks_df[tracks_df[location_col] == location]

    # Filter by condition.
    if condition is None:
        condition = np.random.choice(tracks_df["condition"].unique())
    tracks_df = tracks_df[tracks_df["condition"] == condition]

    # Filter by filename or file_id.
    if filename is None and file_id is None:
        filename = np.random.choice(tracks_df["filename"].unique())
    elif file_id is not None:
        filename = tracks_df[tracks_df["file_id"] == file_id]["filename"].iloc[0]
    tracks_df = tracks_df[tracks_df["filename"] == filename]

    # Determine frame range (convert time to frames).
    min_frame = tracks_df["frame"].min()
    max_frame = tracks_df["frame"].max()
    if time_start is not None:
        time_start_frames = int(time_start / frame_interval)
        time_start_frames = max(min_frame, min(time_start_frames, max_frame))
    else:
        time_start_frames = min_frame
    if time_end is not None:
        time_end_frames = int(time_end / frame_interval)
        time_end_frames = max(min_frame, min(time_end_frames, max_frame))
    else:
        time_end_frames = max_frame
    # Calculate time (in seconds) for annotation.
    time_start_sec = time_start_frames * frame_interval
    time_end_sec = time_end_frames * frame_interval
    # Filter tracks by frame range.
    tracks_df = tracks_df[
        (tracks_df["frame"] >= time_start_frames)
        & (tracks_df["frame"] <= time_end_frames)
    ]

    if tracks_df.empty:
        raise ValueError(
            "No valid data available for plotting after filtering by filename and time range."
        )

    # Set figure and axis backgrounds.
    figure_background = "none" if transparent_background else "white"
    axis_background = (0, 0, 0, 0) if transparent_background else "white"

    # Create the figure.
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(figure_background)
    ax.set_facecolor(axis_background)

    # Overlay image if requested.
    if overlay_image:
        image_filename = filename.replace("_tracked", "") + ".tif"
        image_path = os.path.join(master_dir, "data", condition, image_filename)
        overlay_data = imread(image_path)
        if max_projection:
            overlay_data = np.max(overlay_data, axis=0)
        elif display_final_frame:
            overlay_data = overlay_data[-1, :, :]
        overlay_data = img_as_float(overlay_data)
        if contrast_limits:
            lower, upper = contrast_limits
            overlay_data = np.clip((overlay_data - lower) / (upper - lower), 0, 1)
        else:
            overlay_data = (overlay_data - overlay_data.min()) / (
                overlay_data.max() - overlay_data.min()
            )
        if invert_image:
            overlay_data = 1 - overlay_data
        height, width = overlay_data.shape
        extent = [0, width * pixel_size_um, 0, height * pixel_size_um]
        ax.imshow(overlay_data, cmap="gray", origin="lower", extent=extent)

    # --- Plot Tracks with Segment-by-Segment Coloring ---
    unique_ids = tracks_df["unique_id"].unique()
    plotted_unique_ids = []  # Track which unique_ids were actually plotted

    for uid in unique_ids:
        track = tracks_df[tracks_df["unique_id"] == uid].sort_values("frame")

        # For cluster-based coloring, we need to handle segments that might change cluster
        if color_by in ["cluster", "cluster_id"] or "cluster" in color_by.lower():
            # Plot track segment by segment based on cluster changes
            cluster_changes = track[color_by].ne(track[color_by].shift()).cumsum()

            for segment_id in cluster_changes.unique():
                segment = track[cluster_changes == segment_id]
                if len(segment) < 2:  # Skip single-point segments
                    continue

                # Get color for this segment's cluster
                if color_by == "motion_class":
                    line_color_plot = segment["motion_color"].iloc[0]
                elif is_categorical:
                    cluster_value = segment[color_by].iloc[0]
                    if cluster_value in unique_classes:
                        color_idx = list(unique_classes).index(cluster_value)
                        line_color_plot = plt.cm.get_cmap(colorway)(
                            color_idx / max(len(unique_classes) - 1, 1)
                        )
                    else:
                        line_color_plot = "gray"  # Default for unknown clusters
                else:
                    num_classes = max(
                        tracks_df["segment_color"].max()
                        - tracks_df["segment_color"].min(),
                        1,
                    )
                    normalized_color = (
                        segment["segment_color"].iloc[0]
                        - tracks_df["segment_color"].min()
                    ) / num_classes
                    line_color_plot = plt.cm.get_cmap(colorway)(normalized_color)

                # Plot this segment
                line_obj = ax.plot(
                    segment["x_um"],
                    segment["y_um"],
                    color=line_color_plot,
                    linewidth=line_thickness,
                )[0]
                line_obj.set_gid(f"track_{uid}_segment_{segment_id}")

                # else:
            # For non-cluster coloring, plot entire track as one polyline
            if color_by == "motion_class":
                line_color_plot = track["motion_color"].iloc[0]
            else:
                if is_categorical:
                    num_classes = max(len(unique_classes) - 1, 1)
                else:
                    num_classes = max(
                        tracks_df["segment_color"].max()
                        - tracks_df["segment_color"].min(),
                        1,
                    )
                normalized_color = (
                    track["segment_color"].iloc[0] - tracks_df["segment_color"].min()
                ) / num_classes
                line_color_plot = plt.cm.get_cmap(colorway)(normalized_color)

            line_obj = ax.plot(
                track["x_um"],
                track["y_um"],
                color=line_color_plot,
                linewidth=line_thickness,
            )[0]
            line_obj.set_gid(f"track_{uid}")

        plotted_unique_ids.append(uid)

    # Remove axes if no overlay image.
    if not overlay_image:
        ax.axis("off")

    # --- Draw Scale Bar in Data Coordinates ---
    # Set the data limits based on the desired plot size (in microns).
    ax.set_xlim(0, plot_size_um)
    ax.set_ylim(0, plot_size_um)
    ax.set_aspect("equal", adjustable="datalim")

    # Define a margin (e.g., 5% of plot_size_um) for placing the scale bar.
    margin = plot_size_um * 0.05
    x_end = plot_size_um - margin
    x_start = x_end - scale_bar_length  # exactly 'scale_bar_length' microns long
    y_bar = margin
    ax.plot(
        [x_start, x_end],
        [y_bar, y_bar],
        color=scale_bar_color,
        lw=scale_bar_thickness,
        solid_capstyle="butt",
    )
    # Place a text label centered below the scale bar.
    ax.text(
        (x_start + x_end) / 2,
        y_bar - margin * 0.3,
        f"{scale_bar_length} µm",
        ha="center",
        va="top",
        fontsize=10 * scale_factor,
        color=scale_bar_color,
    )

    # --- Add Time Range Annotation ---
    ax.annotate(
        f"Time: {time_start_sec:.2f}s - {time_end_sec:.2f}s",
        xy=(0.5, 1.02),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=default_font,
        color=scale_bar_color,
    )

    # --- Add Legend / Colorbar ---
    # Treat cluster columns as categorical even if they're numeric
    is_cluster_column = (
        color_by in ["cluster", "cluster_id"] or "cluster" in color_by.lower()
    )
    is_categorical = (
        tracks_df[color_by].dtype == object
        or pd.api.types.is_categorical_dtype(tracks_df[color_by])
        or is_cluster_column
    )

    if is_categorical:
        # Only show colors for classes actually present in the filtered data
        present_classes = [
            cls for cls in unique_classes if cls in tracks_df[color_by].unique()
        ]
        handles = []
        for cls in present_classes:
            color_idx = list(unique_classes).index(cls)
            color = plt.cm.get_cmap(colorway)(
                color_idx / max(len(unique_classes) - 1, 1)
            )
            handles.append(plt.Line2D([0], [0], color=color, lw=2, label=f"{cls}"))

        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0.0,
            title=f"Legend: {color_by}",
            fontsize=default_font,
            title_fontsize=default_font,
        )
    else:
        # Only show colorbar range for actual data in the plot
        color_min = tracks_df[color_by].min()
        color_max = tracks_df[color_by].max()
        sm = plt.cm.ScalarMappable(
            cmap=colorway, norm=plt.Normalize(vmin=color_min, vmax=color_max)
        )
        sm.set_array([])
        cbar = plt.colorbar(
            sm, ax=ax, orientation="vertical", pad=0.1, fraction=0.03, shrink=0.25
        )
        cbar.set_label(
            f"{color_by} (range: {round(color_min, 2)} - {round(color_max, 2)})",
            color=scale_bar_color,
            fontsize=default_font,
        )
        cbar.ax.yaxis.set_tick_params(color=scale_bar_color, labelsize=default_font)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=scale_bar_color)

    plt.tight_layout()

    # --- Saving/Exporting ---
    if save_path:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ext = export_format.lower()
        if ext not in ["png", "svg"]:
            print("Invalid export format specified. Defaulting to 'png'.")
            ext = "png"
        base_name = filename.split(".")[0] if filename else "tracks"
        out_filename = f"{base_name}_tracks.{ext}"
        full_save_path = os.path.join(save_path, out_filename)
        plt.savefig(
            full_save_path, transparent=transparent_background, dpi=dpi, format=ext
        )

        svg_data = None
        if ext == "svg":
            with open(full_save_path, encoding="utf-8") as f:
                svg_data = f.read()
            # Remove <clipPath> definitions, metadata, XML declaration, and DOCTYPE.
            svg_data = re.sub(
                r'<clipPath id="[^"]*">.*?</clipPath>', "", svg_data, flags=re.DOTALL
            )
            svg_data = re.sub(r'\s*clip-path="url\([^)]*\)"', "", svg_data)
            svg_data = re.sub(
                r"<metadata>.*?</metadata>", "", svg_data, flags=re.DOTALL
            )
            svg_data = re.sub(r"<\?xml[^>]*\?>", "", svg_data, flags=re.DOTALL)
            svg_data = re.sub(r"<!DOCTYPE[^>]*>", "", svg_data, flags=re.DOTALL)
            svg_data = svg_data.strip()
            with open(full_save_path, "w", encoding="utf-8") as f:
                f.write(svg_data)

        if show_plot:
            plt.show()
        else:
            plt.close()

        if ext == "svg" and return_svg:
            return {
                "svg_data": svg_data,
                "plotted_unique_ids": plotted_unique_ids,
                "tracks_df": tracks_df,
            }
        else:
            return {"plotted_unique_ids": plotted_unique_ids, "tracks_df": tracks_df}
    else:
        plt.show()
        return {"plotted_unique_ids": plotted_unique_ids, "tracks_df": tracks_df}


def plot_single_track_by_cluster(
    tracks_df,
    unique_id,
    cluster_col="cluster_id",
    filename=None,
    file_id=None,
    location=None,
    condition=None,
    time_start=None,
    time_end=None,
    master_dir=config.MASTER,
    scale_bar_length=2,  # in microns
    scale_bar_color="black",
    scale_bar_thickness=2,  # thickness of the scale bar
    transparent_background=True,
    save_path=None,
    pixel_size_um=config.PIXELSIZE_MICRONS,
    frame_interval=config.TIME_BETWEEN_FRAMES,
    colorway="tab20",
    order=None,  # Order for cluster coloring
    figsize=(4, 4),  # Consistent figure size for comparison
    plot_size_um=10,  # Consistent plot size in microns
    line_thickness=1.2,
    dpi=200,
    export_format="svg",
    return_svg=False,
    show_plot=True,
    show_legend=True,
    show_start_marker=True,
    start_marker_size=4,
):
    """
    Plot a single track colored by cluster ID changes.

    This function displays one track with consistent sizing and scale bars,
    making it easy to compare tracks across different conditions or time periods.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing track data with cluster assignments
    unique_id : str
        The unique track identifier to plot
    cluster_col : str, default 'cluster_id'
        Column name containing cluster IDs
    ... (other parameters same as plot_tracks_static_svg)
    show_legend : bool, default True
        Whether to show the cluster legend
    show_start_marker : bool, default True
        Whether to show a marker at the track start
    start_marker_size : float, default 4
        Size of the start marker

    Returns
    -------
    dict : Dictionary containing track info, cluster changes, and plot data

    """
    # Filter to single track
    if unique_id not in tracks_df["unique_id"].unique():
        available_ids = tracks_df["unique_id"].unique()
        raise ValueError(
            f"Unique ID '{unique_id}' not found. Available IDs: {available_ids}"
        )

    # Apply same filtering logic as plot_tracks_static_svg
    df = tracks_df.copy()

    # Filter by location
    if location is not None:
        location_col = _get_location_column(df)
        df = df[df[location_col] == location]

    # Filter by condition
    if condition is not None:
        df = df[df["condition"] == condition]

    # Filter by filename
    if filename is not None:
        df = df[df["filename"] == filename]
    elif file_id is not None:
        df = df[df["file_id"] == file_id]

    # Filter by time range
    if time_start is not None or time_end is not None:
        df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
        min_frame = df["frame"].min()
        max_frame = df["frame"].max()

        if time_start is not None:
            time_start_frames = int(time_start / frame_interval)
            time_start_frames = max(min_frame, min(time_start_frames, max_frame))
        else:
            time_start_frames = min_frame

        if time_end is not None:
            time_end_frames = int(time_end / frame_interval)
            time_end_frames = max(min_frame, min(time_end_frames, max_frame))
        else:
            time_end_frames = max_frame

        df = df[(df["frame"] >= time_start_frames) & (df["frame"] <= time_end_frames)]

    # Get single track data
    track_data = df[df["unique_id"] == unique_id].sort_values("frame")

    if track_data.empty:
        raise ValueError(f"No data found for unique ID '{unique_id}' after filtering")

    # Get unique clusters and set up colors (consistent with other functions)
    all_unique_clusters = sorted([c for c in df[cluster_col].unique() if pd.notna(c)])
    if order is None:
        ordered_clusters = all_unique_clusters
    else:
        ordered_clusters = [c for c in order if c in all_unique_clusters]
        remaining_clusters = [
            c for c in all_unique_clusters if c not in ordered_clusters
        ]
        ordered_clusters.extend(remaining_clusters)

    # Create cluster to color mapping
    cluster_to_color = {}
    for i, cluster_id in enumerate(ordered_clusters):
        cluster_to_color[cluster_id] = plt.cm.get_cmap(colorway)(
            i / max(len(ordered_clusters) - 1, 1)
        )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Set backgrounds
    figure_background = "none" if transparent_background else "white"
    axis_background = (0, 0, 0, 0) if transparent_background else "white"
    fig.patch.set_facecolor(figure_background)
    ax.set_facecolor(axis_background)

    # Convert coordinates to microns
    track_data = track_data.copy()
    track_data["x_um"] = track_data["x"] * pixel_size_um
    track_data["y_um"] = track_data["y"] * pixel_size_um

    # Center the track in the plot
    x_center = track_data["x_um"].mean()
    y_center = track_data["y_um"].mean()
    track_data["x_um"] -= x_center - plot_size_um / 2
    track_data["y_um"] -= y_center - plot_size_um / 2

    # Plot track segments by cluster
    cluster_changes = (
        track_data[cluster_col].ne(track_data[cluster_col].shift()).cumsum()
    )
    segment_info = []

    for segment_id in cluster_changes.unique():
        segment = track_data[cluster_changes == segment_id]
        if len(segment) < 2:  # Skip single-point segments
            continue

        cluster_id = segment[cluster_col].iloc[0]
        if pd.notna(cluster_id) and cluster_id in cluster_to_color:
            color = cluster_to_color[cluster_id]
        else:
            color = "gray"

        # Plot segment
        ax.plot(
            segment["x_um"],
            segment["y_um"],
            color=color,
            linewidth=line_thickness,
            alpha=0.8,
            solid_capstyle="round",
        )

        # Store segment info
        segment_info.append(
            {
                "segment_id": segment_id,
                "cluster_id": cluster_id,
                "start_frame": segment["frame"].min(),
                "end_frame": segment["frame"].max(),
                "n_points": len(segment),
                "color": color,
            }
        )

    # Add start marker
    if show_start_marker and len(track_data) > 0:
        start_point = track_data.iloc[0]
        ax.plot(
            start_point["x_um"],
            start_point["y_um"],
            "o",
            color="red",
            markersize=start_marker_size,
            markeredgecolor="white",
            markeredgewidth=0.5,
            zorder=10,
            alpha=0.9,
        )

    # Set axis properties
    ax.set_xlim(0, plot_size_um)
    ax.set_ylim(0, plot_size_um)
    ax.set_aspect("equal")
    ax.axis("off")

    # Add scale bar
    margin = plot_size_um * 0.05
    x_end = plot_size_um - margin
    x_start = x_end - scale_bar_length
    y_bar = margin
    ax.plot(
        [x_start, x_end],
        [y_bar, y_bar],
        color=scale_bar_color,
        lw=scale_bar_thickness,
        solid_capstyle="butt",
    )
    ax.text(
        (x_start + x_end) / 2,
        y_bar - margin * 0.3,
        f"{scale_bar_length} µm",
        ha="center",
        va="top",
        fontsize=10,
        color=scale_bar_color,
    )

    # Add title
    title_parts = [f"Track {unique_id}"]
    if condition:
        title_parts.append(f"Condition: {condition}")
    if location:
        title_parts.append(f"Location: {location}")
    if filename:
        title_parts.append(f"File: {filename}")

    ax.set_title(" | ".join(title_parts), pad=20, fontsize=12)

    # Add legend
    used_clusters = [
        info["cluster_id"] for info in segment_info if pd.notna(info["cluster_id"])
    ]
    unique_used_clusters = sorted(set(used_clusters))

    if show_legend and unique_used_clusters:
        handles = []
        for cluster_id in unique_used_clusters:
            color = cluster_to_color.get(cluster_id, "gray")
            handles.append(
                plt.Line2D([0], [0], color=color, lw=2, label=f"Cluster {cluster_id}")
            )

        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0.0,
            title="Clusters",
            fontsize=10,
            title_fontsize=10,
        )

    plt.tight_layout()

    # Save or show
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        ext = export_format.lower()
        if ext not in ["png", "svg"]:
            ext = "png"

        out_filename = f"track_{unique_id}_{cluster_col}.{ext}"
        if condition:
            out_filename = f"track_{unique_id}_{condition}_{cluster_col}.{ext}"

        full_save_path = os.path.join(save_path, out_filename)
        plt.savefig(
            full_save_path, transparent=transparent_background, dpi=dpi, format=ext
        )

        svg_data = None
        if ext == "svg":
            with open(full_save_path, encoding="utf-8") as f:
                svg_data = f.read()

        if show_plot:
            plt.show()
        else:
            plt.close()

        result = {
            "unique_id": unique_id,
            "track_data": track_data,
            "segment_info": segment_info,
            "cluster_changes": len(segment_info),
            "clusters_used": unique_used_clusters,
            "save_path": full_save_path,
        }

        if ext == "svg" and return_svg:
            result["svg_data"] = svg_data

        return result
    else:
        plt.show()
        return {
            "unique_id": unique_id,
            "track_data": track_data,
            "segment_info": segment_info,
            "cluster_changes": len(segment_info),
            "clusters_used": unique_used_clusters,
        }


def plot_tracks_grid_by_cluster(
    tracks_df,
    cluster_col="cluster_id",
    segments_per_cluster=5,
    window_size=config.OVERLAP,  # Number of frames per window segment
    filename=None,
    file_id=None,
    location=None,
    condition=None,
    time_start=None,
    time_end=None,
    motion_type=None,
    master_dir=config.MASTER,
    transparent_background=True,
    save_path=None,
    pixel_size_um=config.PIXELSIZE_MICRONS,
    frame_interval=config.TIME_BETWEEN_FRAMES,
    colorway="tab20",
    order=None,  # Order for cluster coloring, defaults to low-to-high cluster numbers
    figsize=(12, 8),
    track_size_um=5,  # Size of each track's bounding box in microns
    line_thickness=0.6,
    dpi=200,
    export_format="svg",
    return_svg=False,
    show_plot=True,
    grid_spacing_um=2,  # Spacing between columns in microns
    track_spacing_um=1,  # Spacing between tracks within a column
    random_seed=42,
    show_cluster_labels=True,
    label_fontsize=12,
):
    """
    Create a grid visualization of track window segments organized by cluster ID.

    Each column represents a different cluster, with multiple window segments from that cluster
    displayed in a vertical arrangement. Each segment shows a fixed number of frames (window_size)
    to ensure consistent cluster ID throughout the segment. Segments are centered and normalized
    rather than showing absolute spatial positions.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing track data with cluster assignments
    cluster_col : str, default 'cluster_id'
        Column name containing cluster IDs
    segments_per_cluster : int, default 5
        Number of window segments to display per cluster
    window_size : int, default 30
        Number of consecutive frames per window segment (ensures consistent cluster ID)
    filename, file_id, location, condition : str, optional
        Filters for selecting specific data
    time_start, time_end : float, optional
        Time range to display (in seconds)
    motion_type : str, optional
        Filter by motion type if specified
    track_size_um : float, default 5
        Size of each track segment's display area in microns
    grid_spacing_um : float, default 2
        Horizontal spacing between cluster columns in microns
    track_spacing_um : float, default 1
        Vertical spacing between segments within a column in microns
    random_seed : int, default 42
        Seed for reproducible segment selection
    show_cluster_labels : bool, default True
        Whether to show cluster labels at the top of each column
    label_fontsize : int, default 12
        Font size for cluster labels
    order : list, optional
        Custom order for cluster coloring. If None, uses low-to-high cluster numbers
    ... (other parameters same as plot_tracks_static_svg)

    Returns
    -------
    dict : Dictionary containing segment information and layout details

    """
    # Set random seed for reproducible results
    np.random.seed(random_seed)

    # Apply same filtering logic as plot_tracks_static_svg
    df = tracks_df.copy()

    # Filter by location
    if location is not None:
        location_col = _get_location_column(df)
        if location not in df[location_col].unique():
            available_locations = df[location_col].unique()
            raise ValueError(
                f"Location '{location}' not found. Available: {available_locations}"
            )
        df = df[df[location_col] == location]

    # Filter by condition
    if condition is not None:
        if condition not in df["condition"].unique():
            available_conditions = df["condition"].unique()
            raise ValueError(
                f"Condition '{condition}' not found. Available: {available_conditions}"
            )
        df = df[df["condition"] == condition]

    # Filter by filename
    if filename is not None:
        if filename not in df["filename"].unique():
            available_files = df["filename"].unique()
            raise ValueError(
                f"Filename '{filename}' not found. Available: {available_files}"
            )
        df = df[df["filename"] == filename]

    # Filter by file_id
    if file_id is not None:
        if "file_id" in df.columns:
            df = df[df["file_id"] == file_id]

    # Filter by motion type
    if motion_type is not None:
        if "motion_class" in df.columns:
            df = df[df["motion_class"] == motion_type]

    # Time filtering
    if time_start is not None or time_end is not None:
        if "time_s" not in df.columns:
            df["time_s"] = df["frame"] * frame_interval

        if time_start is not None:
            df = df[df["time_s"] >= time_start]
        if time_end is not None:
            df = df[df["time_s"] <= time_end]

    # Check if cluster column exists
    if cluster_col not in df.columns:
        raise ValueError(
            f"Cluster column '{cluster_col}' not found in dataframe. Available columns: {list(df.columns)}"
        )

    # Get unique clusters, excluding NaN values (use same logic as other functions)
    all_unique_clusters = sorted([c for c in df[cluster_col].unique() if pd.notna(c)])
    n_clusters = len(all_unique_clusters)

    if n_clusters == 0:
        raise ValueError(
            "No valid clusters found after filtering (all values are NaN or missing)"
        )

    # Set up cluster ordering for left-to-right positioning (consistent with other functions)
    if order is None:
        ordered_clusters = all_unique_clusters  # Low to high cluster numbers
    else:
        # Use provided order, but only include clusters that exist in data
        ordered_clusters = [c for c in order if c in all_unique_clusters]
        # Add any remaining clusters not in the provided order
        remaining_clusters = [
            c for c in all_unique_clusters if c not in ordered_clusters
        ]
        ordered_clusters.extend(remaining_clusters)

    # Find valid window segments for each cluster
    selected_segments = {}
    segment_info = []

    for cluster_id in all_unique_clusters:
        cluster_df = df[(df[cluster_col] == cluster_id) & pd.notna(df[cluster_col])]

        # Find valid window segments within this cluster
        segment_candidates = []
        for track_id in cluster_df["unique_id"].unique():
            track_data = cluster_df[cluster_df["unique_id"] == track_id].sort_values(
                "frame"
            )
            frames = track_data["frame"].values

            # Find consecutive frame runs of at least window_size length
            if len(frames) < window_size:
                continue

            # Check for consecutive frames
            frame_diffs = np.diff(frames)
            split_points = np.where(frame_diffs != 1)[0] + 1
            frame_runs = np.split(frames, split_points)

            for run in frame_runs:
                if len(run) >= window_size:
                    # Find all possible window starts in this run
                    for start_idx in range(len(run) - window_size + 1):
                        start_frame = run[start_idx]
                        end_frame = run[start_idx + window_size - 1]

                        # Verify all frames in window have the same cluster ID
                        window_data = track_data[
                            (track_data["frame"] >= start_frame)
                            & (track_data["frame"] <= end_frame)
                        ]
                        if (
                            len(window_data) == window_size
                            and len(window_data[cluster_col].unique()) == 1
                        ):
                            segment_candidates.append(
                                {
                                    "track_id": track_id,
                                    "start_frame": start_frame,
                                    "end_frame": end_frame,
                                    "cluster_id": cluster_id,
                                }
                            )

        # Sample segments for this cluster
        n_available = len(segment_candidates)
        if n_available == 0:
            selected_segments[cluster_id] = []
            continue

        if n_available >= segments_per_cluster:
            sampled_indices = np.random.choice(
                n_available, segments_per_cluster, replace=False
            )
        else:
            # Use all available and pad with repeats if needed
            sampled_indices = list(range(n_available))
            while len(sampled_indices) < segments_per_cluster:
                additional_needed = min(
                    segments_per_cluster - len(sampled_indices), n_available
                )
                sampled_indices.extend(
                    np.random.choice(n_available, additional_needed, replace=False)
                )

        selected_segments[cluster_id] = [
            segment_candidates[i] for i in sampled_indices[:segments_per_cluster]
        ]

        # Store segment info
        for i, segment in enumerate(selected_segments[cluster_id]):
            segment_info.append(
                {
                    "cluster_id": cluster_id,
                    "track_id": segment["track_id"],
                    "segment_index": i,
                    "start_frame": segment["start_frame"],
                    "end_frame": segment["end_frame"],
                    "n_frames": window_size,
                }
            )

    # Set up the plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Calculate grid layout using ordered clusters
    max_segments = (
        max(len(segments) for segments in selected_segments.values())
        if selected_segments
        else 0
    )
    if max_segments == 0:
        raise ValueError("No valid segments found for any cluster")

    # Use ordered clusters for width calculation
    n_ordered_clusters = len(ordered_clusters)
    total_width = (
        n_ordered_clusters * track_size_um + (n_ordered_clusters - 1) * grid_spacing_um
    )
    total_height = max_segments * track_size_um + (max_segments - 1) * track_spacing_um

    # Set up cluster-based coloring (same logic as plot_tracks_static_svg)

    cluster_to_color = {}
    for i, cluster_id in enumerate(ordered_clusters):
        cluster_to_color[cluster_id] = plt.cm.get_cmap(colorway)(
            i / max(len(ordered_clusters) - 1, 1)
        )

    # FIRST PASS: Calculate global coordinate range for consistent scaling
    all_coord_ranges = []
    all_segments_data = []

    for cluster_id in ordered_clusters:
        if (
            cluster_id not in selected_segments
            or len(selected_segments[cluster_id]) == 0
        ):
            continue

        for segment in selected_segments[cluster_id]:
            # Get segment data (window frames only)
            segment_data = df[
                (df["unique_id"] == segment["track_id"])
                & (df["frame"] >= segment["start_frame"])
                & (df["frame"] <= segment["end_frame"])
            ].sort_values("frame")

            if len(segment_data) == 0:
                continue

            # Convert to microns and center
            x_coords = segment_data["x"].values * pixel_size_um
            y_coords = segment_data["y"].values * pixel_size_um

            # Center the segment
            x_coords = x_coords - np.mean(x_coords)
            y_coords = y_coords - np.mean(y_coords)

            # Store data for second pass
            all_segments_data.append(
                {
                    "cluster_id": cluster_id,
                    "segment": segment,
                    "x_coords": x_coords,
                    "y_coords": y_coords,
                }
            )

            # Calculate range for this segment
            coord_range = max(np.ptp(x_coords), np.ptp(y_coords))
            if coord_range > 0:
                all_coord_ranges.append(coord_range)

    # Calculate global scale factor
    if all_coord_ranges:
        global_max_range = max(all_coord_ranges)
        global_scale_factor = (
            track_size_um * 0.8
        ) / global_max_range  # Use 80% of available space
    else:
        global_scale_factor = 1.0

    # SECOND PASS: Plot segments using global scale
    for segment_data in all_segments_data:
        cluster_id = segment_data["cluster_id"]
        segment = segment_data["segment"]
        x_coords = segment_data["x_coords"].copy()
        y_coords = segment_data["y_coords"].copy()

        # Find this segment's position in the grid
        cluster_idx = ordered_clusters.index(cluster_id)
        segment_idx = selected_segments[cluster_id].index(segment)

        cluster_x_center = (
            cluster_idx * (track_size_um + grid_spacing_um) + track_size_um / 2
        )
        segment_y_center = (
            segment_idx * (track_size_um + track_spacing_um) + track_size_um / 2
        )
        cluster_color = cluster_to_color[cluster_id]

        # Apply global scale factor
        x_coords = x_coords * global_scale_factor
        y_coords = y_coords * global_scale_factor

        # Position in grid
        x_coords = x_coords + cluster_x_center
        y_coords = y_coords + segment_y_center

        # Plot segment with cluster color
        ax.plot(
            x_coords, y_coords, color=cluster_color, linewidth=line_thickness, alpha=0.8
        )

        # Add a small dot at the start
        ax.plot(
            x_coords[0],
            y_coords[0],
            "o",
            color=cluster_color,
            markersize=line_thickness * 2,
            alpha=0.6,
        )

    # Add cluster labels using ordered clusters
    if show_cluster_labels:
        for cluster_idx, cluster_id in enumerate(ordered_clusters):
            cluster_x_center = (
                cluster_idx * (track_size_um + grid_spacing_um) + track_size_um / 2
            )
            label_y = total_height + 0.5
            ax.text(
                cluster_x_center,
                label_y,
                f"Cluster {cluster_id}",
                ha="center",
                va="bottom",
                fontsize=label_fontsize,
                fontweight="bold",
            )

    # Set axis properties
    ax.set_xlim(-0.5, total_width + 0.5)
    ax.set_ylim(-0.5, total_height + (1 if show_cluster_labels else 0.5))
    ax.set_aspect("equal")
    ax.axis("off")

    # Add title
    title_parts = []
    if condition:
        title_parts.append(f"Condition: {condition}")
    if location:
        title_parts.append(f"Location: {location}")
    if filename:
        title_parts.append(f"File: {filename}")
    if time_start is not None or time_end is not None:
        time_str = f"Time: {time_start or 0:.1f}-{time_end or 'end':.1f}s"
        title_parts.append(time_str)

    if title_parts:
        ax.set_title(" | ".join(title_parts), pad=20, fontsize=label_fontsize)

    # Set background
    if transparent_background:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

    # Save or show
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Create filename
        save_filename = f"segment_grid_{cluster_col}"
        if condition:
            save_filename += f"_{condition}"
        if location:
            save_filename += f"_{location}"
        if filename:
            save_filename += f"_{filename}"
        save_filename += f".{export_format}"

        full_path = os.path.join(save_path, save_filename)

        if export_format.lower() == "svg":
            plt.savefig(
                full_path,
                format="svg",
                bbox_inches="tight",
                transparent=transparent_background,
                dpi=dpi,
            )
            if return_svg:
                with open(full_path) as f:
                    svg_content = f.read()
        else:
            plt.savefig(
                full_path,
                format=export_format,
                bbox_inches="tight",
                transparent=transparent_background,
                dpi=dpi,
            )

    if show_plot:
        plt.show()
    else:
        plt.close()

    # Return information
    result = {
        "segment_info": segment_info,
        "clusters": all_unique_clusters,
        "ordered_clusters": ordered_clusters,  # The actual left-to-right order used
        "selected_segments": selected_segments,
        "n_clusters": n_clusters,
        "segments_per_cluster": segments_per_cluster,
        "window_size": window_size,
        "grid_dimensions": (total_width, total_height),
    }

    if save_path and export_format.lower() == "svg" and return_svg:
        result["svg_content"] = svg_content

    return result


def visualize_track_changes_with_filtering(
    original_df,
    cleaned_df,
    removed_ids=set(),
    filename=None,
    time_start=None,
    time_end=None,
    time_between_frames=0.1,
    plot_size_px=150,
    dpi=100,
    pixel_size_um=0.1,
    figsize=(18, 6),
    line_width=1.2,
    alpha_range=(0.3, 1.0),
    transparent_background=True,
    overlay_image=False,
    master_dir=None,
    condition=None,
    max_projection=False,
    display_final_frame=True,
    contrast_limits=None,
    invert_image=False,
):
    """
    Visualizes track changes with three side-by-side subplots:
    1. Original tracks (one color)
    2. New & Removed tracks (distinct colors)
    3. Combined view of all tracks

    Tracks are properly **aligned to the raw image**, ensuring correct time filtering.

    Parameters
    - overlay_image: bool, toggles display of raw image underneath tracks.
    - master_dir: str, dataset directory for images.
    - condition: str, experimental condition (used for file path).
    - max_projection, display_final_frame: Controls which image frame to show.
    - contrast_limits: tuple (low, high), scales image contrast.
    - invert_image: bool, inverts grayscale image.

    """
    # **Restrict to a single filename** for efficiency
    if filename is None:
        filename = original_df["filename"].iloc[0]
    original_df = original_df[original_df["filename"] == filename]
    cleaned_df = cleaned_df[cleaned_df["filename"] == filename]

    # Convert time_start and time_end from seconds to frames
    min_frame = original_df["frame"].min()
    max_frame = original_df["frame"].max()

    if time_start is not None:
        time_start_frame = max(
            min_frame, min(int(time_start / time_between_frames), max_frame)
        )
        time_start_str = f"{time_start:.2f}s"
    else:
        time_start_frame = min_frame
        time_start_str = "Start"

    if time_end is not None:
        time_end_frame = max(
            min_frame, min(int(time_end / time_between_frames), max_frame)
        )
        time_end_str = f"{time_end:.2f}s"
    else:
        time_end_frame = max_frame
        time_end_str = "End"

    # **Filter data by time range**
    original_df = original_df[
        (original_df["frame"] >= time_start_frame)
        & (original_df["frame"] <= time_end_frame)
    ]
    cleaned_df = cleaned_df[
        (cleaned_df["frame"] >= time_start_frame)
        & (cleaned_df["frame"] <= time_end_frame)
    ]

    # **Define Colors (Dark2 colormap)**
    colors = plt.cm.Dark2.colors
    old_track_color = colors[0]
    new_track_color = colors[1]
    removed_track_color = colors[2]

    # **Set figure background transparency**
    figure_background = "none" if transparent_background else "white"
    axis_background = (0, 0, 0, 0) if transparent_background else "white"

    # **Set up figure with 3 subplots**
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(figure_background)

    titles = ["Original Tracks", "New & Removed Tracks", "Combined View"]

    # **Load and process the image if overlay_image is enabled**
    if overlay_image:
        image_filename = filename.replace("_tracked", "") + ".tif"
        if condition is not None:
            image_path = os.path.join(master_dir, "data", condition, image_filename)
        else:
            # filter the df by filename first
            filtered_df = original_df[original_df["filename"] == filename]
            condition = filtered_df["condition"].iloc[0]
            image_path = os.path.join(master_dir, "data", condition, image_filename)

        try:
            overlay_data = imread(image_path)

            if max_projection:
                overlay_data = np.max(overlay_data, axis=0)
            elif display_final_frame:
                overlay_data = overlay_data[-1, :, :]

            overlay_data = img_as_float(overlay_data)
            if contrast_limits:
                lower, upper = contrast_limits
                overlay_data = np.clip((overlay_data - lower) / (upper - lower), 0, 1)
            else:
                overlay_data = (overlay_data - overlay_data.min()) / (
                    overlay_data.max() - overlay_data.min()
                )

            if invert_image:
                overlay_data = 1 - overlay_data

            height, width = overlay_data.shape
            extent = [
                original_df["x_um"].min(),
                original_df["x_um"].max(),
                original_df["y_um"].min(),
                original_df["y_um"].max(),
            ]

        except Exception as e:
            print(f"Error loading image: {e}")
            overlay_image = False

    # **Helper function to plot tracks**
    def plot_tracks(ax, df, color):
        unique_tracks = df["unique_id"].unique()
        for unique_id in unique_tracks:
            track = df[df["unique_id"] == unique_id]
            n_points = len(track)
            alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)

            for i in range(n_points - 1):
                ax.plot(
                    track.iloc[i : i + 2]["x_um"],
                    track.iloc[i : i + 2]["y_um"],
                    color=color,
                    alpha=alphas[i],
                    linewidth=line_width,
                )

    # **Helper function to plot removed tracks (now connected to previous points)**
    def plot_removed_tracks(ax, df, removed_ids, color):
        for unique_id in removed_ids:
            track = df[df["unique_id"] == unique_id].copy()
            track["x_um_prev"] = track["x_um"].shift(1)
            track["y_um_prev"] = track["y_um"].shift(1)

            for _, row in track.iterrows():
                if not np.isnan(row["x_um_prev"]):
                    ax.plot(
                        [row["x_um_prev"], row["x_um"]],
                        [row["y_um_prev"], row["y_um"]],
                        color=color,
                        alpha=0.8,
                        linewidth=line_width,
                    )

    # **Plot 1: Original Tracks Only**
    axes[0].set_facecolor(axis_background)
    if overlay_image:
        axes[0].imshow(overlay_data, cmap="gray", origin="lower", extent=extent)
    plot_tracks(axes[0], original_df, old_track_color)
    axes[0].set_title(titles[0], fontsize=14)

    # **Plot 2: New Tracks + Removed Tracks**
    axes[1].set_facecolor(axis_background)
    if overlay_image:
        axes[1].imshow(overlay_data, cmap="gray", origin="lower", extent=extent)
    plot_tracks(axes[1], cleaned_df, new_track_color)
    plot_removed_tracks(axes[1], original_df, removed_ids, removed_track_color)
    axes[1].set_title(titles[1], fontsize=14)

    # **Plot 3: Combined View**
    axes[2].set_facecolor(axis_background)
    if overlay_image:
        axes[2].imshow(overlay_data, cmap="gray", origin="lower", extent=extent)
    plot_tracks(axes[2], original_df, old_track_color)
    plot_tracks(axes[2], cleaned_df, new_track_color)
    plot_removed_tracks(axes[2], original_df, removed_ids, removed_track_color)

    # **Legend for Final Plot**
    axes[2].set_title(titles[2], fontsize=14)
    legend_labels = ["Original", "New", "Removed"]
    legend_colors = [old_track_color, new_track_color, removed_track_color]
    legend_patches = [
        plt.Line2D([0], [0], color=color, lw=2, label=label)
        for color, label in zip(legend_colors, legend_labels, strict=False)
    ]
    axes[2].legend(handles=legend_patches, loc="upper right", fontsize=12)

    fig.suptitle(
        f"Track Changes | File: {filename} | Time: {time_start_str} - {time_end_str}",
        fontsize=16,
    )

    plt.show()


### The following for new movie format with Napari ##


# -------------------------------
# CSV Reading Function
# -------------------------------
def read_csv_file(csv_path):
    """
    Read a CSV file into a pandas DataFrame.
    """
    df = pd.read_csv(csv_path)
    return df


def load_image(file_path):
    return skio.imread(file_path)  # changed to skio to avoid conflict with imageio


def load_tracks(df, filename):
    tracks = df[df["filename"] == filename]
    return tracks


def get_condition_from_filename(df, filename):
    try:
        condition = df[df["filename"] == filename]["condition"].iloc[0]
    except IndexError:
        print(f"Error: Filename '{filename}' not found in the dataframe.")
        raise
    return condition


def invert_colormap(cmap):
    """
    Inverts a napari Colormap by extracting its RGBA color array,
    inverting the RGB channels, and returning a new Colormap.

    Parameters
        cmap (napari.utils.colormaps.Colormap): The original colormap.

    Returns
        napari.utils.colormaps.Colormap: A new colormap with inverted RGB values.

    """
    # Ensure the colormap has a 'colors' attribute (an array of shape (N, 4))
    if not hasattr(cmap, "colors"):
        raise ValueError("Provided colormap does not have a 'colors' attribute.")

    # Copy the original colors array
    orig_colors = cmap.colors.copy()
    # Invert the RGB channels (first three columns) but leave the alpha channel unchanged
    inverted_colors = orig_colors.copy()
    inverted_colors[:, :3] = 1 - inverted_colors[:, :3]

    # Create a new name for the inverted colormap
    new_name = cmap.name + "_inverted"

    # Create and return a new napari Colormap object with the inverted colors
    from napari.utils.colormaps import Colormap

    new_cmap = Colormap(inverted_colors, name=new_name)
    return new_cmap


def save_movie(
    viewer,
    tracks,
    feature="particle",
    save_path="movie.mov",
    steps=None,
    timer_overlay=False,
    timer_format="{time:.2f}s",
    fps=100,
):
    import imageio
    import numpy as np
    import pandas as pd

    # Build a complete timer mapping if timer overlay is enabled and 'time_s' exists.
    if timer_overlay and "time_s" in tracks.columns:
        min_frame = tracks["frame"].min()
        max_frame = tracks["frame"].max()
        # Since frames are floats, use np.arange with a 1.0 step.
        timer_df = pd.DataFrame({"frame": np.arange(min_frame, max_frame + 1, 1.0)})
        # Merge with available time data (dropping duplicate frames if any)
        time_data = tracks[["frame", "time_s"]].drop_duplicates()
        timer_df = timer_df.merge(time_data, on="frame", how="left")
        # Interpolate missing time_s values
        timer_df["time_s"] = timer_df["time_s"].interpolate()
    else:
        timer_df = None

    # Ensure the viewer is in 2D mode.
    viewer.dims.ndisplay = 2

    # Get all unique frames from the track data, sorted in order.
    unique_frames = np.sort(tracks["frame"].unique())
    total_frames = len(unique_frames)

    # If steps is not specified, capture every frame.
    if steps is None:
        steps = total_frames
    elif steps < total_frames:
        indices = np.linspace(0, total_frames - 1, steps, dtype=int)
        unique_frames = unique_frames[indices]
    else:
        steps = total_frames

    frames_list = []

    # Loop through each frame, update the viewer, and capture a screenshot.
    for frame in unique_frames:
        # Set the current frame in the viewer (assuming the time dimension is 0)
        viewer.dims.set_point(0, frame)
        # Update timer overlay if enabled.
        if timer_overlay and timer_df is not None:
            # update_timer_text should update the viewer overlay based on the current frame.
            update_timer_text(viewer, timer_df, frame, timer_format)
        # Take a screenshot of the canvas (set canvas_only=True to exclude UI elements)
        frame_img = viewer.screenshot(canvas_only=True)
        frames_list.append(frame_img)

    # Write the frames to a movie file with the specified frames per second.
    # At 100 fps, each frame displays for 10 ms.
    imageio.mimwrite(save_path, frames_list, fps=fps)
    print(f"Movie saved to {save_path} with {len(frames_list)} frames at {fps} fps.")


def load_lut_from_file(path):
    """
    Load a Fiji LUT file and return a (256, 4) float32 RGBA array for Napari.
    """
    with open(path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)

    r, g, b = raw[:256], raw[256:512], raw[512:768]
    rgba_uint8 = np.stack([r, g, b, np.full_like(r, 255)], axis=1)
    rgba_float = rgba_uint8.astype(np.float32) / 255.0
    return rgba_float


def special_colormap_for_feature(lut_filepath=None):
    """
    Returns a Napari Colormap object from a Fiji-style LUT file.
    """
    if lut_filepath is not None:
        lut = load_lut_from_file(lut_filepath)
    else:
        print(f"[LUT] COULD NOT FIND a matching LUT file: {lut_filepath}")
        lut = np.linspace(0, 1, 256).reshape(-1, 1) * np.array(
            [[1.0, 0.84, 0.0]]
        )  # golden fallback
        lut = np.hstack([lut, np.ones((256, 1))])  # add alpha

    return Colormap(lut, name="klein_gold")


def update_timer_text(viewer, timer_df, frame, timer_format):
    # Round the float frame to match the timer_df frame values (or convert as needed)
    frame_val = round(frame, 0)
    time_series = timer_df.loc[timer_df["frame"] == frame_val, "time_s"]
    if not time_series.empty:
        current_time = time_series.iloc[0]
    else:
        current_time = frame  # fallback
        # print(f"No time found for frame {frame}. Using frame number as fallback.")
    viewer.text_overlay.text = timer_format.format(time=current_time)


def napari_visualizer(
    tracks_df,
    master_dir=config.MASTER,
    condition=None,
    cell=None,
    location=None,
    save_movie_flag=False,
    save_as_png=False,
    feature="particle",  # feature to use for coloring
    steps=None,
    tail_length=10,
    tail_width=2,
    smoothing=False,
    smoothing_window=5,
    colormap="viridis",  # can be a matplotlib colormap name, dict, or Colormap object
    path_to_alt_lut=None,
    invert_raw=False,
    invert_colors=False,
    time_coloring=False,
    # time_window=None,
    frame_range=None,
    show_raw=True,
    timer_overlay=False,
    timer_format="{time:.2f}s",
    track_symbol="cross",
    show_track_head=False,
    head_size=1,
    head_symbol="o",  # available are:
    head_color="red",
    background="dark",
    scale_bar=True,
):
    """
    Visualize an image with overlaid tracks in napari with customizable colormaps.

    Parameters
      - tracks_df: DataFrame containing track information.
      - master_dir: Directory containing the master folder. Can be encoded in df.
      - condition, cell, location: Identifiers for filtering tracks and images.
      - save_movie_flag: If True, save a movie of the visualization in the movies folder.
      - feature: Column name to use for track coloring.
      - steps: Number of steps/frames for the movie.
      - track_thickness, tail_length, tail_width: Track rendering options.
      - smoothing, smoothing_window: Options for smoothing track positions.
      - colormap: A matplotlib colormap name, a dict defining a custom colormap, or a napari Colormap object.
      - invert_raw: If True, invert the raw image.
      - invert_colors: If True, invert the colormap.
      - time_coloring: If True, use time-based coloring (normalized per track).
      - time_window, frame_range: Options for frame selection.
      - show_raw: If True, display the raw image.
      - timer_overlay, timer_format: Timer overlay options.
      - track_symbol: arrow, clobber cross diamond disc hbar ring square star tailed_arrow triangle_down vbar x SPECIAL ONE IS: shape_arrow
      - show_track_head: If True, highlight the head of each track.
      - background: 'dark' (default) or 'light'.
      - scale_bar_length: Length of a scalebar overlay (in microns).

    """
    # --- Step 1: Filter by location, condition, and cell ---
    location_col = _get_location_column(tracks_df)
    locationlist = tracks_df[location_col].unique()
    if isinstance(location, int):
        location = locationlist[location]
    elif isinstance(location, str):
        if location not in locationlist:
            raise ValueError(
                f"Location '{location}' not found in available locations: {locationlist}"
            )
    elif location is None:
        np.random.shuffle(locationlist)
        location = locationlist[0]
    else:
        raise ValueError("Location must be a string, integer, or None.")

    filtered_tracks_df = tracks_df[tracks_df[location_col] == location]

    conditionlist = filtered_tracks_df["condition"].unique()
    if isinstance(condition, int):
        condition = conditionlist[condition]
    elif isinstance(condition, str):
        if condition not in conditionlist:
            raise ValueError(
                f"Condition '{condition}' not found for location '{location}': {conditionlist}"
            )
    elif condition is None:
        np.random.shuffle(conditionlist)
        condition = conditionlist[0]
    else:
        raise ValueError("Condition must be a string, integer, or None.")

    celllist = filtered_tracks_df[filtered_tracks_df["condition"] == condition][
        "filename"
    ].unique()
    if isinstance(cell, int):
        cell = celllist[cell]
    elif isinstance(cell, str):
        if cell not in celllist:
            raise ValueError(
                f"Cell '{cell}' not found for condition '{condition}' and location '{location}': {celllist}"
            )
    elif cell is None:
        np.random.shuffle(celllist)
        cell = celllist[0]
    else:
        raise ValueError("Cell must be a string, integer, or None.")

    # --- Step 2: Load the image ---
    master_data_dir = os.path.join(master_dir, "data")
    image_filename = cell.replace("_tracked", "") + ".tif"
    image_path = os.path.join(master_data_dir, condition, image_filename)
    image = load_image(image_path)
    if invert_raw:
        image = image.max() - image

    # --- Step 3: Load and prepare track data ---
    tracks = load_tracks(filtered_tracks_df, cell)
    tracks = tracks.sort_values("frame").reset_index(drop=True)

    frame_min = int(tracks["frame"].min())
    frame_max = int(tracks["frame"].max())

    # Crop the raw image to match the track time span
    image = image[frame_min : frame_max + 1]

    # And reassign the frame numbers so they match this new range
    tracks["frame"] = tracks["frame"] - frame_min

    if timer_overlay and "time_s" in tracks.columns:
        min_frame = tracks["frame"].min()
        max_frame = tracks["frame"].max()
        timer_df = pd.DataFrame({"frame": np.arange(min_frame, max_frame + 1, 1.0)})
        time_data = tracks[["frame", "time_s"]].drop_duplicates()
        timer_df = timer_df.merge(time_data, on="frame", how="left")
        timer_df["time_s"] = timer_df["time_s"].interpolate()
        print("Timer DataFrame created with time_s values interpolated.")

    if frame_range is not None:
        start_frame, end_frame = frame_range
        tracks = tracks[
            (tracks["frame"] >= start_frame) & (tracks["frame"] <= end_frame)
        ]
        tracks["frame"] = tracks["frame"] - start_frame
        image = image[start_frame : end_frame + 1]

    if smoothing:

        def smooth_series(s):
            return s.rolling(window=smoothing_window, center=True, min_periods=1).mean()

        tracks["x"] = tracks.groupby("particle")["x"].transform(smooth_series)
        tracks["y"] = tracks.groupby("particle")["y"].transform(smooth_series)

    # --- Step 4: Determine the coloring feature ---
    # If time_coloring is enabled, use the normalized time; else use the specified feature.
    if time_coloring and feature not in ["frame", "time_s"]:
        tracks["time_norm"] = tracks.groupby("particle")["frame"].transform(
            lambda s: (s - s.min()) / (s.max() - s.min())
        )
        color_feature = "time_norm"
        is_categorical = False  # normalized time is continuous

    else:
        color_feature = feature
        if not pd.api.types.is_numeric_dtype(tracks[feature]):
            is_categorical = True
            unique_vals = sorted(tracks[feature].unique())
            mapping = {val: i for i, val in enumerate(unique_vals)}
            mapped_key = feature + "_numeric"
            tracks[mapped_key] = tracks[feature].map(mapping)
            color_feature = mapped_key
        else:
            is_categorical = False
            # 💡 NEW: normalize feature to [0, 1]
            norm_key = feature + "_norm"
            tracks[norm_key] = MinMaxScaler().fit_transform(tracks[[feature]])
            color_feature = norm_key

    # --- Step 5: Build the features dictionary ---
    features_dict = {"particle": tracks["particle"].values}
    # IMPORTANT: use the mapped key as the dictionary key.
    if color_feature in tracks.columns:
        features_dict[color_feature] = tracks[color_feature].values
    else:
        print(f"[warning] Color feature '{color_feature}' not found in tracks columns.")
    # Also include any additional features from config.FEATURES2.
    for feat in FEATURES2:
        if feat in tracks.columns:
            features_dict[feat] = tracks[feat].values

    custom_colormap = None

    # 1. Handle override by path_to_alt_lut if it's set and valid
    if path_to_alt_lut is not None and os.path.exists(path_to_alt_lut):
        try:
            rgba_lut = load_lut_from_file(path_to_alt_lut)
            custom_colormap = Colormap(
                rgba_lut, name=os.path.basename(path_to_alt_lut).replace(".lut", "")
            )
            print(f"[LUT] Loaded custom LUT from: {path_to_alt_lut}")
        except Exception as e:
            print(f"[LUT ERROR] Failed to load LUT from '{path_to_alt_lut}': {e}")

    # 2. Fallback: special built-in LUTs like 'klein_gold'
    elif isinstance(colormap, str) and colormap.lower() == "klein_gold":
        custom_colormap = special_colormap_for_feature(
            lut_filepath="D:/customLUTs/KTZ_Klein_Gold.lut"
        )

    # 3. Fallback: categorical logic
    elif is_categorical:
        num_categories = len(unique_vals)
        if isinstance(colormap, str):
            cmap = plt.get_cmap(colormap, num_categories)
        else:
            cmap = colormap
        discrete_colors = np.array(
            [cmap(i) for i in range(num_categories)], dtype=np.float32
        )
        custom_colormap = Colormap(discrete_colors, name=str(colormap))

    # 4. Fallback: normal continuous colormap
    else:
        custom_colormap = ensure_colormap(colormap)

    # 5. Invert if needed
    if invert_colors:
        # custom_colormap = custom_colormap.reversed()
        custom_colormap = invert_colormap(custom_colormap)

    # 6. Register colormap if not already
    from napari.utils.colormaps import AVAILABLE_COLORMAPS

    if custom_colormap.name not in AVAILABLE_COLORMAPS:
        AVAILABLE_COLORMAPS[custom_colormap.name] = custom_colormap

    colormap_name = custom_colormap.name
    custom_colormaps_dict = {color_feature: custom_colormap}

    # --- Step 7: Prepare track data for napari ---
    # Expected columns: [particle, frame, time_s, y, x]
    # tracks_new_df = tracks[["particle", "frame", "time_s", "y", "x"]]
    tracks_new_df = tracks[["particle", "frame", "y", "x"]]  # CARDAMOMMMMMMM

    # --- Step 8: Initialize the napari viewer and add layers ---
    viewer = napari.Viewer()
    viewer.dims.axis_labels = ("time", "y", "x")  # ADDED CARDAMOM
    viewer.theme = "light" if background == "light" else "dark"

    if timer_overlay:
        from napari.components._viewer_constants import CanvasPosition

        viewer.text_overlay.visible = True
        viewer.text_overlay.text = ""
        viewer.text_overlay.font_size = 16
        viewer.text_overlay.color = "white" if background == "dark" else "black"
        viewer.text_overlay.position = CanvasPosition.TOP_RIGHT

        def on_dims_change(event):
            current_frame = viewer.dims.current_step[0]
            update_timer_text(viewer, timer_df, current_frame, timer_format)

        viewer.dims.events.current_step.connect(on_dims_change)

    if show_raw:
        viewer.add_image(image, name=f"Raw {cell}", scale=(1, 0.065, 0.065))  # optional

        print("Image shape:", image.shape)
    # tracks_new_df["frame"] = tracks_new_df["frame"].astype(int) #new wee bit remove if problem
    tracks_new_df.loc[:, "frame"] = tracks_new_df["frame"].astype(int)

    tracks_layer = viewer.add_tracks(
        tracks_new_df.to_numpy(),
        features=features_dict,
        name=f"Tracks {cell}",
        scale=(1, 0.065, 0.065),  # Assuming the image is in (time, y, x) format
        color_by=color_feature,
        colormap=colormap_name,
        colormaps_dict=custom_colormaps_dict,
        tail_length=tail_length,
        tail_width=tail_width,
    )

    # 🎬 Sync animation for both raw image and tracks
    if frame_range is not None:
        # Make sure the current time is within the filtered range.
        viewer.dims.current_step = (start_frame, 0, 0)
    else:
        viewer.dims.current_step = (0, 0, 0)
    # viewer.dims.axis_labels = ('time', 'y', 'x')  # just for clarity/debugging # removed cardamom

    if show_track_head:
        import matplotlib.colors as mcolors

        # --- Special Case: Using Shapes layer if head_symbol is 'shape_arrow' ---
        if head_symbol == "shape_arrow":

            def compute_arrow_polygon_for_particle(particle_df, frame, s):
                valid = particle_df[particle_df["frame"] <= frame]
                if valid.empty:
                    return None, None, None
                sorted_valid = valid.sort_values("frame")
                head_row = sorted_valid.iloc[-1]
                head_coord = head_row[["y", "x"]].to_numpy()
                head_frame = head_row["frame"]
                feat_val = head_row.get(color_feature, None)
                if len(sorted_valid) >= 2:
                    r1 = sorted_valid.iloc[-2]
                    r2 = sorted_valid.iloc[-1]
                    angle = np.degrees(np.arctan2(r2["y"] - r1["y"], r2["x"] - r1["x"]))
                else:
                    angle = 0
                base_arrow = np.array(
                    [[0, -s], [s / 2, s / 2], [0, s / 4], [-s / 2, s / 2]]
                )
                theta = np.radians(angle)
                R = np.array(
                    [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
                )
                rotated_arrow = base_arrow @ R.T
                arrow_poly = rotated_arrow + head_coord
                return arrow_poly, feat_val, head_frame

            current_frame = viewer.dims.current_step[0]
            arrow_polys = []
            arrow_colors = []
            fade_duration = tail_length
            for particle, df in tracks.groupby("particle"):
                poly, feat_val, head_frame = compute_arrow_polygon_for_particle(
                    df, current_frame, head_size
                )
                if poly is not None:
                    arrow_polys.append(poly)
                    if head_color is None:
                        if feat_val is not None:
                            base_color = np.ravel(custom_colormap.map(feat_val))
                        else:
                            base_color = np.ravel(mcolors.to_rgba("white"))
                    else:
                        base_color = np.ravel(mcolors.to_rgba(head_color))
                    alpha = 1 - (current_frame - head_frame) / fade_duration
                    alpha = np.clip(alpha, 0, 1)
                    base_rgb = list(base_color[:3])
                    arrow_colors.append(base_rgb + [float(alpha)])
            if len(arrow_polys) == 0:
                arrow_polys = np.empty((0, 0, 2))
                arrow_colors = "white"
            else:
                arrow_polys = np.array(arrow_polys)
                arrow_colors = np.array(arrow_colors, dtype=np.float32)
            shapes_layer = viewer.add_shapes(
                arrow_polys,
                name="Track Heads",
                face_color=arrow_colors,
                edge_color="transparent",
                opacity=1,
                shape_type="polygon",
            )

            def update_arrow_heads(event):
                frame = viewer.dims.current_step[0]
                new_polys = []
                new_colors = []
                for particle, df in tracks.groupby("particle"):
                    poly, feat_val, head_frame = compute_arrow_polygon_for_particle(
                        df, frame, head_size
                    )
                    if poly is not None:
                        new_polys.append(poly)
                        if head_color is None:
                            if feat_val is not None:
                                base_color = np.ravel(custom_colormap.map(feat_val))
                            else:
                                base_color = np.ravel(mcolors.to_rgba("white"))
                        else:
                            base_color = np.ravel(mcolors.to_rgba(head_color))
                        alpha = 1 - (frame - head_frame) / fade_duration
                        alpha = np.clip(alpha, 0, 1)
                        base_rgb = list(base_color[:3])
                        new_colors.append(base_rgb + [float(alpha)])
                if len(new_polys) == 0:
                    new_polys = np.empty((0, 0, 2))
                    new_colors = "white"
                else:
                    new_polys = np.array(new_polys)
                    new_colors = np.array(new_colors, dtype=np.float32)
                shapes_layer.data = new_polys
                shapes_layer.face_color = new_colors

            viewer.dims.events.current_step.connect(update_arrow_heads)

        # --- Else: Use Points layer for track heads ---
        else:
            fixed_head_color = (
                mcolors.to_rgba(head_color) if head_color is not None else None
            )
            fade_duration = tail_length

            def compute_track_heads_with_feature(frame):
                heads = []
                for particle, df in tracks.groupby("particle"):
                    valid = df[df["frame"] <= frame]
                    if not valid.empty:
                        row = valid.loc[valid["frame"].idxmax()]
                        y, x = row[["y", "x"]].to_numpy()
                        head_frame = row["frame"]
                        feat_val = row.get(color_feature, None)
                        heads.append([y, x, head_frame, feat_val])
                if heads:
                    return np.array(heads)
                else:
                    return np.empty((0, 4))

            def safe_rgb(c):
                c_arr = np.ravel(c)
                if c_arr.size < 3:
                    return list(c_arr) + [0] * (3 - c_arr.size)
                else:
                    return list(c_arr[:3])

            current_frame = viewer.dims.current_step[0]
            initial_heads = compute_track_heads_with_feature(current_frame)
            if initial_heads.shape[0] > 0:
                coords = initial_heads[:, :2]
                if fixed_head_color is None and all(
                    feat is not None for feat in initial_heads[:, 3]
                ):
                    dyn_colors = np.array(
                        [
                            np.ravel(custom_colormap.map(feat))
                            for feat in initial_heads[:, 3]
                        ],
                        dtype=np.float32,
                    )
                elif fixed_head_color is not None:
                    dyn_colors = np.tile(fixed_head_color, (initial_heads.shape[0], 1))
                else:
                    dyn_colors = np.array(
                        [
                            mcolors.to_rgba("white")
                            for _ in range(initial_heads.shape[0])
                        ],
                        dtype=np.float32,
                    )
                alphas = 1 - (current_frame - initial_heads[:, 2]) / fade_duration
                alphas = np.clip(alphas, 0, 1)
                face_colors = np.array(
                    [
                        safe_rgb(c) + [float(a)]
                        for c, a in zip(dyn_colors, alphas, strict=False)
                    ],
                    dtype=np.float32,
                )
            else:
                coords = np.empty((0, 2))
                face_colors = np.empty((0, 4), dtype=np.float32)
            head_points_layer = viewer.add_points(
                coords,
                name="Track Heads",
                face_color=face_colors,
                edge_width=0,
                size=head_size,
                symbol=head_symbol,
            )

            def update_track_heads(event):
                frame = viewer.dims.current_step[0]
                new_heads = compute_track_heads_with_feature(frame)
                if new_heads.shape[0] > 0:
                    new_coords = new_heads[:, :2]
                    if fixed_head_color is None and all(
                        feat is not None for feat in new_heads[:, 3]
                    ):
                        new_dyn_colors = np.array(
                            [
                                np.ravel(custom_colormap.map(feat))
                                for feat in new_heads[:, 3]
                            ],
                            dtype=np.float32,
                        )
                    elif fixed_head_color is not None:
                        new_dyn_colors = np.tile(
                            fixed_head_color, (new_heads.shape[0], 1)
                        )
                    else:
                        new_dyn_colors = np.array(
                            [
                                mcolors.to_rgba("white")
                                for _ in range(new_heads.shape[0])
                            ],
                            dtype=np.float32,
                        )
                    new_alphas = 1 - (frame - new_heads[:, 2]) / fade_duration
                    new_alphas = np.clip(new_alphas, 0, 1)
                    new_face_colors = np.array(
                        [
                            safe_rgb(c) + [float(a)]
                            for c, a in zip(new_dyn_colors, new_alphas, strict=False)
                        ],
                        dtype=np.float32,
                    )
                else:
                    new_coords = np.empty((0, 2))
                    new_face_colors = np.empty((0, 4), dtype=np.float32)
                head_points_layer.data = new_coords
                head_points_layer.face_color = new_face_colors

            viewer.dims.events.current_step.connect(update_track_heads)

    # --- Step 9: Add scalebar overlay if requested ---
    if scale_bar:
        # from napari._vispy.overlays.scale_bar import VispyScaleBarOverlay
        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = "micron"
        # viewer.scale_bar.length = scale_bar_length

        class DummyEvent:
            def connect(self, callback):
                pass

        class DummyEvents:
            def __init__(self):
                self.visible = DummyEvent()
                self.opacity = DummyEvent()
                self.blending = DummyEvent()
                self.box = DummyEvent()
                self.box_color = DummyEvent()
                self.color = DummyEvent()
                self.colored = DummyEvent()
                self.font_size = DummyEvent()
                self.ticks = DummyEvent()
                self.unit = DummyEvent()
                self.length = DummyEvent()
                self.position = DummyEvent()

    # --- Step 10: Save the movie if requested ---
    if save_movie_flag:
        if frame_range is not None:
            start_frame, end_frame = frame_range
            steps = end_frame - start_frame + 1
        elif steps is None:
            steps = int(tracks_new_df["frame"].max()) + 1
            print(f"Number of steps for the movie automatically set to: {steps}")
        movie_dir = os.path.join(master_dir, "movies")
        print(f"Saving movie to: {movie_dir}")
        if not os.path.exists(movie_dir):
            os.makedirs(movie_dir, exist_ok=True)

        os.makedirs(movie_dir, exist_ok=True)
        movie_path = os.path.join(movie_dir, f"{condition}_{cell}.mov")
        save_movie(
            viewer,
            tracks_new_df,
            feature=color_feature,
            save_path=movie_path,
            steps=steps,
            timer_overlay=timer_overlay,
            timer_format=timer_format,
        )

    if save_as_png:
        # Here you may use a different steps logic if desired
        if frame_range is not None:
            start_frame, end_frame = frame_range
            steps = end_frame - start_frame + 1
        elif steps is None:
            steps = int(tracks_new_df["frame"].max()) + 1
            print(f"Number of steps for PNG capture automatically set to: {steps}")
        png_dir = os.path.join(master_dir, "png_frames")
        print(f"Saving PNG frames to: {png_dir}")
        save_frames_as_png(
            viewer,
            tracks_new_df,
            feature=color_feature,
            output_dir=png_dir,
            steps=steps,
            timer_overlay=timer_overlay,
            timer_format=timer_format,
        )
        # Optionally bypass the interactive viewer by returning early:
        return

    napari.run()


###########################################

# SAVE AS PNG
###########################################


def save_frames_as_png(
    viewer,
    tracks,
    feature="particle",
    output_dir="frames",
    steps=None,
    timer_overlay=False,
    timer_format="{time:.2f}s",
):
    """
    Save a screenshot (PNG) of the napari canvas for every frame (or subset of frames)
    by updating the viewer dims point.

    Parameters
      viewer: The napari viewer instance.
      tracks: DataFrame or array with track information (must include a 'frame' column).
      feature: (Unused here; kept for consistency with save_movie if needed later.)
      output_dir: Directory to save the PNG files.
      steps: Number of frames to capture. If None, capture every frame.
      timer_overlay: If True, update the timer overlay.
      timer_format: Format string for the timer overlay.

    """
    # Build a timer dataframe if needed
    if timer_overlay and "time_s" in tracks.columns:
        min_frame = tracks["frame"].min()
        max_frame = tracks["frame"].max()
        timer_df = pd.DataFrame({"frame": np.arange(min_frame, max_frame + 1, 1.0)})
        time_data = tracks[["frame", "time_s"]].drop_duplicates()
        timer_df = timer_df.merge(time_data, on="frame", how="left")
        timer_df["time_s"] = timer_df["time_s"].interpolate()
    else:
        timer_df = None

    # Ensure the viewer is in 2D mode.
    viewer.dims.ndisplay = 2

    # Determine unique frames and sample if steps is provided
    unique_frames = np.sort(tracks["frame"].unique())
    total_frames = len(unique_frames)
    if steps is None:
        steps = total_frames
    elif steps < total_frames:
        indices = np.linspace(0, total_frames - 1, steps, dtype=int)
        unique_frames = unique_frames[indices]
    else:
        steps = total_frames

    # Create output directory if it does not exist.
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each frame, update the viewer and save a screenshot.
    for i, frame in enumerate(unique_frames):
        viewer.dims.set_point(0, frame)
        if timer_overlay and timer_df is not None:
            update_timer_text(viewer, timer_df, frame, timer_format)
        # Capture the screenshot. (canvas_only=True omits UI elements.)
        frame_img = viewer.screenshot(canvas_only=True)
        output_path = os.path.join(output_dir, f"frame_{i:03d}.png")
        imageio.imwrite(output_path, frame_img)
    print(f"PNG frames saved to {output_dir} with {len(unique_frames)} frames.")


### Alright, here's the datashader stuff, for visualizing as heatmaps on cells ###


def compute_cell_edge_angle_binning(
    points, centroid, edge_angle_bins=180, edge_padding=0.05, edge_smoothing=0.1
):
    """
    Compute the cell edge using an angle–binning approach.

    The circle (0 to 2π) is divided into edge_angle_bins; for each bin, the maximum
    radial distance from the centroid is computed. Missing bins are filled by circular
    linear interpolation. The radii are then inflated by edge_padding and optionally
    smoothed using a periodic spline (controlled by edge_smoothing).

    Parameters
      points (np.ndarray): Array of shape (N,2) of [x, y] coordinates.
      centroid (np.ndarray): [x, y] coordinates of the centroid.
      edge_angle_bins (int): Number of angular bins.
      edge_padding (float): Fractional padding (e.g., 0.05 increases each radius by 5%).
      edge_smoothing (float): Spline smoothing parameter (0 means no smoothing).

    Returns
      np.ndarray: An (M,2) array with the [x, y] coordinates outlining the cell edge.

    """
    diffs = points - centroid
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    angles = np.mod(angles, 2 * np.pi)
    radii = np.sqrt(np.sum(diffs**2, axis=1))

    bin_edges = np.linspace(0, 2 * np.pi, edge_angle_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    max_radii = np.full_like(bin_centers, np.nan)

    for i in range(edge_angle_bins):
        mask = (angles >= bin_edges[i]) & (angles < bin_edges[i + 1])
        if np.any(mask):
            max_radii[i] = np.max(radii[mask])

    if np.any(np.isnan(max_radii)):
        valid = ~np.isnan(max_radii)
        if np.sum(valid) < 2:
            max_radii = np.zeros_like(bin_centers)
        else:
            angles_ext = np.concatenate(
                (bin_centers[valid], np.array([bin_centers[valid][0] + 2 * np.pi]))
            )
            radii_ext = np.concatenate(
                (max_radii[valid], np.array([max_radii[valid][0]]))
            )
            max_radii = np.interp(bin_centers, angles_ext, radii_ext)

    padded_radii = max_radii * (1 + edge_padding)
    x_edge = centroid[0] + padded_radii * np.cos(bin_centers)
    y_edge = centroid[1] + padded_radii * np.sin(bin_centers)
    edge_points = np.vstack((x_edge, y_edge)).T

    if edge_smoothing > 0:
        try:
            tck, u = splprep(
                [edge_points[:, 0], edge_points[:, 1]], s=edge_smoothing, per=True
            )
            u_fine = np.linspace(0, 1, 200)
            smooth_edge = np.array(splev(u_fine, tck)).T
            return smooth_edge
        except Exception as e:
            print("Smoothing failed:", e)
            return edge_points
    else:
        return edge_points


# Default constants; adjust as needed.
DEFAULT_BOX_SIZE_PIXELS = 150
# DEFAULT_MICRONS_PER_PIXEL = 0.065

# def plot_contour_timelapse_datashader(
#     df,
#     time_col='time_s',
#     value_col='diffusion_coefficient',
#     time_bin=0.6,                 # Duration of each time window
#     smooth_time_bins=0.3,         # Overlap between windows
#     spatial_unit='microns',      # 'pixels' or 'microns'
#     canvas_resolution=600,       # Resolution (in pixels) for Datashader Canvas
#     output_format='gif',         # 'gif' or 'png'
#     export_path=config.SAVED_DATA,
#     cmap=colorcet.fire,          # Default colormap (from colorcet)
#     box_size_pixels=DEFAULT_BOX_SIZE_PIXELS,
#     microns_per_pixel=config.PIXELSIZE_MICRONS,
#     gif_playback=None,           # Frame duration override (in seconds)
#     time_between_frames=config.TIME_BETWEEN_FRAMES,    # Default frame duration if gif_playback is None
#     overlay_contour_lines=False,  # Only used in the matplotlib branch
#     show_colorbar=True,          # Only used with matplotlib's contouring branch
#     contour_levels=200,          # Number of contour levels (for matplotlib branch)
#     spatial_smoothing_sigma=1.0, # Gaussian sigma for spatial smoothing (0 disables)
#     edge_angle_bins=180,         # For computing the cell edge
#     edge_padding=0.05,           # Padding for cell edge computation
#     edge_smoothing=0.1,          # Smoothing factor for cell edge computation
#     remove_axes=False,           # If True, remove axes from the figure
#     use_log_scale=False,         # If True, use logarithmic scaling
#     use_datashader=True          # If True, use datashader native shading; otherwise, use Matplotlib contourf
# ):
#     """
#     Generates a time-lapse map by aggregating data via Datashader and then rendering the
#     result either using datashader’s native shading or Matplotlib’s contouring.

#     - When `use_datashader` is True:
#       The function converts the aggregated array into an xarray DataArray and calls
#       ds.tf.shade with:
#           how='log'    if use_log_scale is True,
#           how='linear' otherwise.

#     - When `use_datashader` is False:
#       The function renders the image with Matplotlib’s contourf. If use_log_scale is True,
#       it uses LogNorm (after ensuring vmin is positive); otherwise, it uses a linear norm.

#     In both cases, a cell-edge contour (computed via helper function) is overlaid.
#     """
#     # Use the first unique filename from the dataframe.
#     filename = df.filename.unique()[0]
#     import colorcet

#     # Set coordinate columns.
#     if spatial_unit == 'pixels':
#         x_col, y_col = 'x', 'y'
#         box_size = box_size_pixels
#     elif spatial_unit == 'microns':
#         x_col, y_col = 'x_um', 'y_um'
#         box_size = box_size_pixels * microns_per_pixel
#     else:
#         raise ValueError("spatial_unit must be either 'pixels' or 'microns'")

#     # Define the fixed coordinate system.
#     global_x_min, global_y_min = 0, 0
#     global_x_max, global_y_max = box_size, box_size

#     # Create export folder if needed.
#     if export_path is None:
#         export_path = "./saved_data"
#     if not os.path.isdir(export_path):
#         os.makedirs(export_path)

#     # Determine global color scale.
#     if use_log_scale:
#         # For log scaling, vmin must be > 0.
#         positive_vals = df[df[value_col] > 0][value_col]
#         if positive_vals.empty:
#             raise ValueError("No positive values found for log scale!")
#         global_vmin = positive_vals.min()
#     else:
#         global_vmin = 0
#     global_vmax = df[value_col].max()

#     # Only needed for the matplotlib contouring branch.
#     if not use_datashader:
#         norm = LogNorm(vmin=global_vmin, vmax=global_vmax) if use_log_scale else Normalize(vmin=global_vmin, vmax=global_vmax)
#         fixed_ticks = np.linspace(global_vmin, global_vmax, 5)

#     # Convert cmap to a Matplotlib colormap if provided as a list.
#     if isinstance(cmap, list):
#         cmap = LinearSegmentedColormap.from_list('custom_cmap', cmap)

#     # Compute the cell edge from all data (assumes helper function defined elsewhere).
#     all_points = df[[x_col, y_col]].dropna().values
#     cell_edge_points = None
#     if len(all_points) >= 3:
#         centroid = np.mean(all_points, axis=0)
#         cell_edge_points = compute_cell_edge_angle_binning(
#             all_points,
#             centroid,
#             edge_angle_bins=edge_angle_bins,
#             edge_padding=edge_padding,
#             edge_smoothing=edge_smoothing
#         )

#     def create_frame(window_df, title):
#         """
#         Creates one frame.
#             - Aggregates the data with Datashader.
#             - Optionally smoothes and masks the data using the cell edge.
#             - Renders using either datashader’s native shading or Matplotlib’s contourf.
#             - Overlays the cell-edge contour.
#         """
#         # Create Datashader Canvas.
#         cvs = ds.Canvas(
#             plot_width=canvas_resolution,
#             plot_height=canvas_resolution,
#             x_range=(global_x_min, global_x_max),
#             y_range=(global_y_min, global_y_max)
#         )
#         agg = cvs.points(window_df, x=x_col, y=y_col, agg=mean(value_col))
#         agg_array = agg.values
#         agg_array = np.where(np.isnan(agg_array), global_vmin, agg_array)

#         if spatial_smoothing_sigma > 0:
#             agg_array = gaussian_filter(agg_array, sigma=spatial_smoothing_sigma)

#         # Generate grid coordinates.
#         x_lin = np.linspace(global_x_min, global_x_max, num=canvas_resolution)
#         y_lin = np.linspace(global_y_min, global_y_max, num=canvas_resolution)
#         X, Y = np.meshgrid(x_lin, y_lin)

#         # Apply cell edge mask.
#         if cell_edge_points is not None:
#             poly_path = mpltPath.Path(cell_edge_points)
#             grid_points = np.vstack((X.ravel(), Y.ravel())).T
#             mask = poly_path.contains_points(grid_points).reshape(X.shape)
#             agg_array[~mask] = global_vmin

#         # Create the figure.
#         fig, ax = plt.subplots(figsize=(6, 6))
#         if use_datashader:
#             # Convert aggregated array into an xarray DataArray.
#             data = xr.DataArray(agg_array, dims=("y", "x"), coords={"y": y_lin, "x": x_lin})
#             how_setting = 'log' if use_log_scale else 'linear'
#             shaded = ds.tf.shade(data, cmap=cmap, how=how_setting)
#             ax.imshow(shaded.to_pil(), extent=(global_x_min, global_x_max, global_y_min, global_y_max))
#         else:
#             # Matplotlib contourf branch.
#             levels_global = np.linspace(global_vmin, global_vmax, contour_levels)
#             cf = ax.contourf(X, Y, agg_array, levels=levels_global, cmap=cmap, norm=norm, alpha=0.8)
#             if overlay_contour_lines:
#                 ax.contour(X, Y, agg_array, levels=levels_global, colors='white', linewidths=0.1)
#             if show_colorbar:
#                 fig.subplots_adjust(right=0.85)
#                 cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
#                 fig.colorbar(cf, cax=cbar_ax, norm=norm, ticks=fixed_ticks, label=value_col)

#         ax.set_xlim(global_x_min, global_x_max)
#         ax.set_ylim(global_y_min, global_y_max)
#         ax.set_aspect('equal')
#         ax.set_title(title)
#         ax.set_xlabel(x_col)
#         ax.set_ylabel(y_col)
#         if remove_axes:
#             ax.axis('off')

#         # Determine edge color: white if using datashader (for better contrast), else black.
#         edge_color = 'white' if use_datashader else 'black'
#         if cell_edge_points is not None:
#             cell_edge_points_arr = np.array(cell_edge_points)
#             ax.plot(cell_edge_points_arr[:, 0], cell_edge_points_arr[:, 1], color=edge_color, lw=0.5, linestyle=':', alpha = 0.95)
#         return fig, ax

#     # Single-frame case.
#     if time_bin is None:
#         title = "Contour Map: All Time"
#         fig, ax = create_frame(df, title)
#         if output_format.lower() == 'gif':
#             buf = io.BytesIO()
#             fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', transparent=True)
#             buf.seek(0)
#             img = imageio.v2.imread(buf)
#             gif_file = os.path.join(export_path, f"{filename}.gif")
#             duration = time_between_frames if gif_playback is None else gif_playback
#             imageio.mimsave(gif_file, [img], duration=duration)
#             plt.close(fig)
#             print(f"Animated GIF saved to {gif_file}")
#         else:
#             folder_path = os.path.join(export_path, filename)
#             if not os.path.exists(folder_path):
#                 os.makedirs(folder_path)
#             png_file = os.path.join(folder_path, f"{filename}_T000.png")
#             fig.savefig(png_file, dpi=150, bbox_inches='tight', transparent=True)
#             plt.close(fig)
#             print(f"PNG saved to {png_file}")
#         return

#     # Multi-frame case.
#     t_min = df[time_col].min()
#     t_max = df[time_col].max()
#     step = smooth_time_bins if smooth_time_bins is not None else time_bin

#     if output_format.lower() == 'gif':
#         frames = []
#         n_frames = 0
#         current_t = t_min
#         while current_t <= t_max:
#             t_end = current_t + time_bin
#             window_df = df[(df[time_col] >= current_t) & (df[time_col] < t_end)]
#             title = f"Time: {current_t:.2f}"
#             fig, ax = create_frame(window_df, title)
#             buf = io.BytesIO()
#             fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', transparent=True)
#             plt.close(fig)
#             buf.seek(0)
#             img = imageio.v2.imread(buf)
#             frames.append(img)
#             n_frames += 1
#             current_t += step
#         gif_file = os.path.join(export_path, f"{filename}.gif")
#         duration = time_between_frames if gif_playback is None else gif_playback
#         imageio.mimsave(gif_file, frames, duration=duration)
#         print(f"Generated {n_frames} frames with frame duration = {duration} sec.")
#         print(f"Animated GIF saved to {gif_file}")
#     else:
#         folder_path = os.path.join(export_path, filename)
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)
#         n_frames = 0
#         current_t = t_min
#         while current_t <= t_max:
#             t_end = current_t + time_bin
#             window_df = df[(df[time_col] >= current_t) & (df[time_col] < t_end)]
#             title = f"Time: {current_t:.2f}"
#             fig, ax = create_frame(window_df, title)
#             png_file = os.path.join(folder_path, f"{filename}_T{n_frames:03d}.png")
#             fig.savefig(png_file, dpi=150, bbox_inches='tight', transparent=True)
#             plt.close(fig)
#             n_frames += 1
#             current_t += step
#         print(f"Generated {n_frames} PNG files saved in folder {folder_path}")


def plot_contour_timelapse_datashader(
    df,
    time_col="time_s",
    value_col="diffusion_coefficient",
    time_bin=0.6,  # Duration of each time window
    smooth_time_bins=0.3,  # Overlap between windows
    spatial_unit="microns",  # 'pixels' or 'microns'
    canvas_resolution=600,  # Resolution (in pixels) for Datashader Canvas
    output_format="gif",  # 'gif' or 'png'
    export_path=config.SAVED_DATA,
    cmap=colorcet.fire,  # Default colormap (from colorcet)
    box_size_pixels=DEFAULT_BOX_SIZE_PIXELS,
    microns_per_pixel=config.PIXELSIZE_MICRONS,
    gif_playback=None,  # Frame duration override (in seconds)
    time_between_frames=config.TIME_BETWEEN_FRAMES,  # Default frame duration if gif_playback is None
    overlay_contour_lines=False,  # Only used in the matplotlib branch
    show_colorbar=True,  # Only used with matplotlib's contouring branch
    contour_levels=200,  # Number of contour levels (for matplotlib branch)
    spatial_smoothing_sigma=1.0,  # Gaussian sigma for spatial smoothing (0 disables)
    edge_angle_bins=180,  # For computing the cell edge
    edge_padding=0.05,  # Padding for cell edge computation
    edge_smoothing=0.1,  # Smoothing factor for cell edge computation
    remove_axes=False,  # If True, remove axes from the figure
    use_log_scale=False,  # If True, use logarithmic scaling
    use_datashader=True,  # If True, use datashader native shading; otherwise, use Matplotlib contourf
    vmin=None,  # Minimum value for color scaling
    vmax=None,  # Maximum value for color scaling
):
    """
    Generates a time-lapse map by aggregating data via Datashader and then rendering the
    result either using datashader’s native shading or Matplotlib’s contouring.

    - When `use_datashader` is True:
      The function converts the aggregated array into an xarray DataArray and calls
      ds.tf.shade with:
          how='log'    if use_log_scale is True,
          how='linear' otherwise.

    - When `use_datashader` is False:
      The function renders the image with Matplotlib’s contourf. If use_log_scale is True,
      it uses LogNorm (after ensuring vmin is positive); otherwise, it uses a linear norm.

    In both cases, a cell-edge contour (computed via helper function) is overlaid.
    """
    # Use the first unique filename from the dataframe.
    filename = df.filename.unique()[0]

    # Set coordinate columns.
    if spatial_unit == "pixels":
        x_col, y_col = "x", "y"
        box_size = box_size_pixels
    elif spatial_unit == "microns":
        x_col, y_col = "x_um", "y_um"
        box_size = box_size_pixels * microns_per_pixel
    else:
        raise ValueError("spatial_unit must be either 'pixels' or 'microns'")

    # Define the fixed coordinate system.
    global_x_min, global_y_min = 0, 0
    global_x_max, global_y_max = box_size, box_size

    # Create export folder if needed.
    if export_path is None:
        export_path = "./saved_data"
    if not os.path.isdir(export_path):
        os.makedirs(export_path)

    # Determine global color scale.
    # if use_log_scale:
    #     # For log scaling, vmin must be > 0.
    #     positive_vals = df[df[value_col] > 0][value_col]
    #     if positive_vals.empty:
    #         raise ValueError("No positive values found for log scale!")
    #     global_vmin = positive_vals.min()
    # else:
    #     global_vmin = 0
    # global_vmax = df[value_col].max()

    # Determine global color scale (allow override via vmin/vmax).
    if vmin is not None:
        global_vmin = vmin
    elif use_log_scale:
        # For log scaling, vmin must be > 0.
        pos = df[df[value_col] > 0][value_col]
        if pos.empty:
            raise ValueError("No positive values found for log scale!")
        global_vmin = pos.min()
    else:
        global_vmin = 0

    if vmax is not None:
        global_vmax = vmax
    else:
        global_vmax = df[value_col].max()

    # Only needed for the matplotlib contouring branch.
    if not use_datashader:
        norm = (
            LogNorm(vmin=global_vmin, vmax=global_vmax)
            if use_log_scale
            else Normalize(vmin=global_vmin, vmax=global_vmax)
        )
        fixed_ticks = np.linspace(global_vmin, global_vmax, 5)

    # Convert cmap to a Matplotlib colormap if provided as a list.
    if isinstance(cmap, list):
        cmap = LinearSegmentedColormap.from_list("custom_cmap", cmap)

    # Compute the cell edge from all data (assumes helper function defined elsewhere).
    all_points = df[[x_col, y_col]].dropna().values
    cell_edge_points = None
    if len(all_points) >= 3:
        centroid = np.mean(all_points, axis=0)
        cell_edge_points = compute_cell_edge_angle_binning(
            all_points,
            centroid,
            edge_angle_bins=edge_angle_bins,
            edge_padding=edge_padding,
            edge_smoothing=edge_smoothing,
        )

    def create_frame(window_df, title):
        """
        Creates one frame.
            - Aggregates the data with Datashader.
            - Optionally smoothes and masks the data using the cell edge.
            - Renders using either datashader’s native shading or Matplotlib’s contourf.
            - Overlays the cell-edge contour.
        """
        # Create Datashader Canvas.
        cvs = ds.Canvas(
            plot_width=canvas_resolution,
            plot_height=canvas_resolution,
            x_range=(global_x_min, global_x_max),
            y_range=(global_y_min, global_y_max),
        )
        agg = cvs.points(window_df, x=x_col, y=y_col, agg=mean(value_col))
        agg_array = agg.values
        agg_array = np.where(np.isnan(agg_array), global_vmin, agg_array)

        if spatial_smoothing_sigma > 0:
            agg_array = gaussian_filter(agg_array, sigma=spatial_smoothing_sigma)

        # Generate grid coordinates.
        x_lin = np.linspace(global_x_min, global_x_max, num=canvas_resolution)
        y_lin = np.linspace(global_y_min, global_y_max, num=canvas_resolution)
        X, Y = np.meshgrid(x_lin, y_lin)

        # Apply cell edge mask.
        if cell_edge_points is not None:
            poly_path = mpltPath.Path(cell_edge_points)
            grid_points = np.vstack((X.ravel(), Y.ravel())).T
            mask = poly_path.contains_points(grid_points).reshape(X.shape)
            agg_array[~mask] = global_vmin

        # Create the figure.
        fig, ax = plt.subplots(figsize=(6, 6))
        if use_datashader:
            # Convert aggregated array into an xarray DataArray.
            data = xr.DataArray(
                agg_array, dims=("y", "x"), coords={"y": y_lin, "x": x_lin}
            )
            how_setting = "log" if use_log_scale else "linear"
            # shaded = ds.tf.shade(data, cmap=cmap, how=how_setting)
            shaded = ds.tf.shade(
                data, cmap=cmap, how=how_setting, span=(global_vmin, global_vmax)
            )
            ax.imshow(
                shaded.to_pil(),
                extent=(global_x_min, global_x_max, global_y_min, global_y_max),
            )
        else:
            # Matplotlib contourf branch.
            levels_global = np.linspace(global_vmin, global_vmax, contour_levels)
            cf = ax.contourf(
                X, Y, agg_array, levels=levels_global, cmap=cmap, norm=norm, alpha=0.8
            )
            if overlay_contour_lines:
                ax.contour(
                    X,
                    Y,
                    agg_array,
                    levels=levels_global,
                    colors="white",
                    linewidths=0.1,
                )
            if show_colorbar:
                fig.subplots_adjust(right=0.85)
                cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
                fig.colorbar(
                    cf, cax=cbar_ax, norm=norm, ticks=fixed_ticks, label=value_col
                )

        ax.set_xlim(global_x_min, global_x_max)
        ax.set_ylim(global_y_min, global_y_max)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        if remove_axes:
            ax.axis("off")

        # Determine edge color: white if using datashader (for better contrast), else black.
        edge_color = "white" if use_datashader else "black"
        if cell_edge_points is not None:
            cell_edge_points_arr = np.array(cell_edge_points)
            ax.plot(
                cell_edge_points_arr[:, 0],
                cell_edge_points_arr[:, 1],
                color=edge_color,
                lw=0.5,
                linestyle=":",
                alpha=0.95,
            )
        return fig, ax

    # Single-frame case.
    if time_bin is None:
        title = "Contour Map: All Time"
        fig, ax = create_frame(df, title)
        if output_format.lower() == "gif":
            buf = io.BytesIO()
            fig.savefig(
                buf, format="png", dpi=150, bbox_inches="tight", transparent=True
            )
            buf.seek(0)
            img = imageio.v2.imread(buf)
            gif_file = os.path.join(export_path, f"{filename}.gif")
            duration = time_between_frames if gif_playback is None else gif_playback
            imageio.mimsave(gif_file, [img], duration=duration)
            plt.close(fig)
            print(f"Animated GIF saved to {gif_file}")
        else:
            folder_path = os.path.join(export_path, filename)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            png_file = os.path.join(folder_path, f"{filename}_T000.png")
            fig.savefig(png_file, dpi=150, bbox_inches="tight", transparent=True)
            plt.close(fig)
            print(f"PNG saved to {png_file}")
        return

    # Multi-frame case.
    t_min = df[time_col].min()
    t_max = df[time_col].max()
    step = smooth_time_bins if smooth_time_bins is not None else time_bin

    if output_format.lower() == "gif":
        frames = []
        n_frames = 0
        current_t = t_min
        while current_t <= t_max:
            t_end = current_t + time_bin
            window_df = df[(df[time_col] >= current_t) & (df[time_col] < t_end)]
            title = f"Time: {current_t:.2f}"
            fig, ax = create_frame(window_df, title)
            buf = io.BytesIO()
            fig.savefig(
                buf, format="png", dpi=150, bbox_inches="tight", transparent=True
            )
            plt.close(fig)
            buf.seek(0)
            img = imageio.v2.imread(buf)
            frames.append(img)
            n_frames += 1
            current_t += step
        gif_file = os.path.join(export_path, f"{filename}.gif")
        duration = time_between_frames if gif_playback is None else gif_playback
        imageio.mimsave(gif_file, frames, duration=duration)
        print(f"Generated {n_frames} frames with frame duration = {duration} sec.")
        print(f"Animated GIF saved to {gif_file}")
    else:
        folder_path = os.path.join(export_path, filename)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        n_frames = 0
        current_t = t_min
        while current_t <= t_max:
            t_end = current_t + time_bin
            window_df = df[(df[time_col] >= current_t) & (df[time_col] < t_end)]
            title = f"Time: {current_t:.2f}"
            fig, ax = create_frame(window_df, title)
            png_file = os.path.join(folder_path, f"{filename}_T{n_frames:03d}.png")
            fig.savefig(png_file, dpi=150, bbox_inches="tight", transparent=True)
            plt.close(fig)
            n_frames += 1
            current_t += step
        print(f"Generated {n_frames} PNG files saved in folder {folder_path}")


def plot_contour_panel_timelapse(
    df,
    time_col="time_s",
    value_col="diffusion_coefficient",
    time_bin=0.6,
    smooth_time_bins=0.3,
    spatial_unit="microns",
    canvas_resolution=300,
    export_path="./",
    cmap="viridis",
    box_size_pixels=100,
    microns_per_pixel=0.1,
    overlay_contour_lines=False,
    show_colorbar=True,
    contour_levels=50,
    spatial_smoothing_sigma=1.0,
    edge_angle_bins=180,
    edge_padding=0.05,
    edge_smoothing=0.1,
    remove_axes=False,
    use_log_scale=False,
    use_datashader=True,
    vmin=None,
    vmax=None,
    figsize_per_subplot=(3, 3),
    downsample=1.0,
    dpi=150,
    output_format="png",
    output_fname="panel",
):
    """
    Creates a grid of subplots: rows = unique df.filename, columns = time windows.
    Saves either PNG (with dpi) or SVG (vector) as specified.
    """
    # Prepare export folder
    os.makedirs(export_path, exist_ok=True)

    # Determine x,y columns and box size
    if spatial_unit == "pixels":
        x_col, y_col = "x", "y"
        box_size = box_size_pixels
    elif spatial_unit == "microns":
        x_col, y_col = "x_um", "y_um"
        box_size = box_size_pixels * microns_per_pixel
    else:
        raise ValueError("spatial_unit must be 'pixels' or 'microns'")

    # Build time windows
    t0, t1 = df[time_col].min(), df[time_col].max()
    step = smooth_time_bins if smooth_time_bins is not None else time_bin
    windows = []
    t = t0
    while t <= t1:
        windows.append((t, t + time_bin, f"{t:.2f}–{t+time_bin:.2f}s"))
        t += step

    # List of cells
    cells = sorted(df.filename.unique())
    n_rows, n_cols = len(cells), len(windows)

    # Compute global color range
    if vmin is not None:
        gmin = vmin
    else:
        gmin = df[df[value_col] > 0][value_col].min() if use_log_scale else 0
    gmax = vmax if vmax is not None else df[value_col].max()

    # Prepare norm & cmap
    norm = (
        LogNorm(vmin=gmin, vmax=gmax)
        if use_log_scale
        else Normalize(vmin=gmin, vmax=gmax)
    )
    if isinstance(cmap, list):
        cmap = LinearSegmentedColormap.from_list("custom", cmap)

    # Figure size with downsampling
    fig_w = n_cols * figsize_per_subplot[0] * downsample
    fig_h = n_rows * figsize_per_subplot[1] * downsample
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    # Precompute cell edge for each
    cell_edges = {}
    for cell in cells:
        pts = df[df.filename == cell][[x_col, y_col]].dropna().values
        if len(pts) >= 3:
            c0 = pts.mean(axis=0)
            edge = compute_cell_edge_angle_binning(
                pts,
                c0,
                edge_angle_bins=edge_angle_bins,
                edge_padding=edge_padding,
                edge_smoothing=edge_smoothing,
            )
            cell_edges[cell] = np.array(edge)
        else:
            cell_edges[cell] = None

    # Fill in each subplot
    for i, cell in enumerate(cells):
        sub = df[df.filename == cell]
        for j, (t_start, t_end, title) in enumerate(windows):
            ax = axes[i][j]
            win = sub[(sub[time_col] >= t_start) & (sub[time_col] < t_end)]

            # Aggregate with Datashader
            cvs = ds.Canvas(
                plot_width=canvas_resolution,
                plot_height=canvas_resolution,
                x_range=(0, box_size),
                y_range=(0, box_size),
            )
            agg = cvs.points(win, x=x_col, y=y_col, agg=ds.mean(value_col))
            arr = np.nan_to_num(agg.values, nan=gmin)
            if spatial_smoothing_sigma > 0:
                arr = gaussian_filter(arr, sigma=spatial_smoothing_sigma)

            # Mask outside cell
            edge_pts = cell_edges[cell]
            xs = np.linspace(0, box_size, canvas_resolution)
            ys = np.linspace(0, box_size, canvas_resolution)
            if edge_pts is not None:
                X, Y = np.meshgrid(xs, ys)
                mask = (
                    mpltPath.Path(edge_pts)
                    .contains_points(np.vstack((X.ravel(), Y.ravel())).T)
                    .reshape(X.shape)
                )
                arr[~mask] = gmin

            # Render
            if use_datashader:
                da_xr = xr.DataArray(arr, dims=("y", "x"), coords={"y": ys, "x": xs})
                how = "log" if use_log_scale else "linear"
                img = tf.shade(da_xr, cmap=cmap, how=how, span=(gmin, gmax))
                ax.imshow(img.to_pil(), extent=(0, box_size, 0, box_size))
            else:
                X, Y = np.meshgrid(xs, ys)
                levels = np.linspace(gmin, gmax, contour_levels)
                cf = ax.contourf(X, Y, arr, levels=levels, cmap=cmap, norm=norm)
                if overlay_contour_lines:
                    ax.contour(X, Y, arr, levels=levels, colors="white", linewidths=0.2)
                if show_colorbar and (i == 0 and j == n_cols - 1):
                    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
                    fig.colorbar(cf, cax=cax, norm=norm, label=value_col)

            # Axes tweaks
            ax.set_title(title, fontsize=8)
            if remove_axes:
                ax.axis("off")
            else:
                ax.set_xticks([])
                ax.set_yticks([])

    plt.tight_layout(rect=(0, 0, 0.9, 1))

    # Determine file extension & save
    ext = output_format.lower()
    out_file = os.path.join(export_path, f"{output_fname}.{ext}")
    if ext == "svg":
        fig.savefig(out_file, format="svg", bbox_inches="tight")
    else:
        fig.savefig(out_file, dpi=dpi, format=ext, bbox_inches="tight")

    plt.close(fig)
    print(f"Panel figure saved to {out_file}")


# # Quick dummy test (with random data)
# if __name__ == "__main__":
#     # Create a fake dataset with 2 cells, random coords & times
#     np.random.seed(0)
#     test_df = pd.DataFrame({
#         'filename': np.repeat(['cellA', 'cellB'], 500),
#         'x_um': np.random.rand(1000)*50,
#         'y_um': np.random.rand(1000)*50,
#         'time_s': np.random.rand(1000)*100,
#         'diffusion_coefficient': np.random.rand(1000)*2
#     })
#     plot_contour_panel_timelapse(
#         test_df,
#         time_bin=20,
#         smooth_time_bins=10,
#         spatial_unit='microns',
#         canvas_resolution=100,
#         figsize_per_subplot=(2,2),
#         downsample=0.5,
#         output_format='svg',
#         output_fname='test_panel',
#         export_path='.'
#     )


###################################
# BACKUP HEXBIN FUNCTIONS
###################################

# def plot_heatmap_timeseries(
#     df,
#     time_col='frame',
#     value_col='diffusion_coefficient',
#     time_bin=60,
#     spatial_unit='microns',
#     spatial_gridsize=80,
#     output_format='gif',
#     export_path=None,
#     hexbin_color='plasma',
#     box_size_pixels=DEFAULT_BOX_SIZE_PIXELS,
#     microns_per_pixel=DEFAULT_MICRONS_PER_PIXEL,
#     plot_method="hexbin",         # Options: "hexbin" or "contour"
#     gif_frame_duration=0.01,      # Seconds per frame (e.g., 0.01 = 10 ms)
#     edge_angle_bins=180,
#     edge_padding=0.05,
#     edge_smoothing=0.1,
#     overlay_contour_lines=True,   # For contour plots: whether to overlay white contour lines.
#     show_colorbar=True            # If False, do not display a colorbar.
# ):
#     """
#     Generates a fixed-size, square heatmap time series from single-particle tracking data.

#     The x- and y-axes are fixed to span from 0 to box_size, where:
#        - For pixel data: box_size = box_size_pixels (default 150).
#        - For micron data: box_size = box_size_pixels * microns_per_pixel.

#     The cell edge is computed via angle–binning (using edge_angle_bins, edge_padding,
#     and edge_smoothing).

#     Two plot methods are available:
#       - "hexbin": Uses hexbin with a fixed gridsize.
#       - "contour": Interpolates the scattered data onto a fixed grid (via np.mgrid) over the full
#          plot extent and displays a filled contour heatmap with an option to overlay white contour lines.

#     The global color scale is fixed from 0 (vmin=0) to the global maximum (vmax) computed from the
#     entire dataset. A Normalize object is created to enforce this scale.

#     The function uses a fixed position for the colorbar—if enabled—so that its size and tick range remain constant.

#     Parameters:
#       df (pd.DataFrame): Contains at least the time, spatial, and value columns.
#       time_col (str): Name of the time column.
#       value_col (str): Name of the value column.
#       time_bin (int/float or None): Duration of each time window; if None, a single plot is produced.
#       spatial_unit (str): "pixels" (using "x", "y") or "microns" (using "x_um", "y_um").
#       spatial_gridsize (int): Gridsize for hexbin plots.
#       output_format (str): "gif" for an animated GIF; "png" for separate image files.
#       export_path (str): Directory for output (defaults to "saved_data" if defined).
#       hexbin_color (str): Colormap name.
#       box_size_pixels (int): Fixed side length (in pixels).
#       microns_per_pixel (float): Conversion factor (if spatial_unit=="microns").
#       plot_method (str): "hexbin" (default) or "contour".
#       gif_frame_duration (float): Time per frame (seconds) for the GIF.
#       edge_angle_bins (int): Number of angle bins for cell edge computation.
#       edge_padding (float): Padding fraction for the cell edge.
#       edge_smoothing (float): Spline smoothing parameter for the cell edge.
#       overlay_contour_lines (bool): Whether to overlay white contour lines (only used in "contour" mode).
#       show_colorbar (bool): If True, display a colorbar at a fixed position; if False, omit it.
#     """
#     # Determine spatial coordinate columns and fixed box size.
#     if spatial_unit == 'pixels':
#         x_col, y_col = 'x', 'y'
#         box_size = box_size_pixels
#     elif spatial_unit == 'microns':
#         x_col, y_col = 'x_um', 'y_um'
#         box_size = box_size_pixels * microns_per_pixel
#     else:
#         raise ValueError("spatial_unit must be either 'pixels' or 'microns'")

#     # Fixed coordinate system.
#     global_x_min, global_y_min = 0, 0
#     global_x_max, global_y_max = box_size, box_size

#     # Set export folder.
#     if export_path is None:
#         try:
#             saved_data
#         except NameError:
#             export_path = "./saved_data"
#         else:
#             export_path = saved_data
#     if not os.path.isdir(export_path):
#         os.makedirs(export_path)

#     # Fixed global color scale: vmin = 0 and vmax = global maximum from the entire dataset.
#     global_vmin = 0
#     global_vmax = df[value_col].max()
#     norm = Normalize(vmin=global_vmin, vmax=global_vmax)
#     # Fixed tick locations.
#     fixed_ticks = np.linspace(global_vmin, global_vmax, 5)

#     # Compute the cell edge.
#     all_points = df[[x_col, y_col]].dropna().values
#     cell_edge_points = None
#     if len(all_points) >= 3:
#         centroid = np.mean(all_points, axis=0)
#         cell_edge_points = compute_cell_edge_angle_binning(
#             all_points,
#             centroid,
#             edge_angle_bins=edge_angle_bins,
#             edge_padding=edge_padding,
#             edge_smoothing=edge_smoothing
#         )

#     # def create_plot(data, title):
#     #     """
#     #     Creates a fixed, square plot with x and y from 0 to box_size.
#     #     Depending on plot_method, produces a hexbin or contour plot using the constant global norm.
#     #     A fixed-position colorbar is added if show_colorbar is True.
#     #     The cell edge is overlaid.
#     #     """
#     #     fig, ax = plt.subplots(figsize=(6, 6))

#     #     if plot_method == "hexbin":
#     #         coll = ax.hexbin(
#     #             data[x_col], data[y_col],
#     #             C=data[value_col],
#     #             reduce_C_function=np.nanmean,
#     #             gridsize=spatial_gridsize,
#     #             cmap=hexbin_color,
#     #             mincnt=1,
#     #             norm=norm,
#     #             extent=[global_x_min, global_x_max, global_y_min, global_y_max]
#     #         )
#     #         if show_colorbar:
#     #             # Use fixed position for colorbar.
#     #             fig.subplots_adjust(right=0.85)
#     #             cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
#     #             fig.colorbar(coll, cax=cbar_ax, norm=norm, ticks=fixed_ticks, label=value_col)
#     #     elif plot_method == "contour":
#     #         grid_x, grid_y = np.mgrid[global_x_min:global_x_max:200j, global_y_min:global_y_max:200j]
#     #         if len(data) < 3:
#     #             grid_z = np.full(grid_x.shape, global_vmin)
#     #         else:
#     #             grid_z = griddata(
#     #                 (data[x_col].values, data[y_col].values),
#     #                 data[value_col].values,
#     #                 (grid_x, grid_y),
#     #                 method='cubic'
#     #             )
#     #             if grid_z is None or np.all(np.isnan(grid_z)):
#     #                 grid_z = np.full(grid_x.shape, global_vmin)
#     #             else:
#     #                 grid_z = np.nan_to_num(grid_z, nan=global_vmin)
#     #         levels = np.linspace(global_vmin, global_vmax, 10)
#     #         contf = ax.contourf(
#     #             grid_x, grid_y, grid_z, levels=100, cmap=hexbin_color,
#     #             norm=norm, alpha=0.8
#     #         )
#     #         if overlay_contour_lines:
#     #             ax.contour(
#     #                 grid_x, grid_y, grid_z, levels=levels, colors='white', linewidths=1
#     #             )
#     #         if show_colorbar:
#     #             fig.subplots_adjust(right=0.85)
#     #             cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
#     #             fig.colorbar(contf, cax=cbar_ax, norm=norm, ticks=fixed_ticks, label=value_col)
#     #     else:
#     #         raise ValueError("plot_method must be either 'hexbin' or 'contour'")

#     #     ax.set_xlim(global_x_min, global_x_max)
#     #     ax.set_ylim(global_y_min, global_y_max)
#     #     ax.set_aspect('equal')
#     #     ax.set_title(title)
#     #     ax.set_xlabel(x_col)
#     #     ax.set_ylabel(y_col)
#     #     if cell_edge_points is not None:
#     #         ax.plot(cell_edge_points[:, 0], cell_edge_points[:, 1], color='black', lw=2)
#     #     return fig, ax

#     def create_plot(data, title):
#         """
#         Creates a fixed, square plot with x and y from 0 to box_size.
#         Depending on plot_method, produces a hexbin or contour plot using the constant global norm.
#         A fixed-position colorbar is added if show_colorbar is True.
#         The cell edge is overlaid.
#         """
#         fig, ax = plt.subplots(figsize=(6, 6))

#         if plot_method == "hexbin":
#             coll = ax.hexbin(
#                 data[x_col], data[y_col],
#                 C=data[value_col],
#                 reduce_C_function=np.nanmean,
#                 gridsize=spatial_gridsize,
#                 cmap=hexbin_color,
#                 mincnt=1,
#                 norm=norm,
#                 extent=[global_x_min, global_x_max, global_y_min, global_y_max]
#             )
#             if show_colorbar:
#                 # Use fixed position for colorbar.
#                 fig.subplots_adjust(right=0.85)
#                 cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
#                 fig.colorbar(coll, cax=cbar_ax, norm=norm, ticks=fixed_ticks, label=value_col)

#         elif plot_method == "contour":
#             grid_x, grid_y = np.mgrid[global_x_min:global_x_max:200j, global_y_min:global_y_max:200j]
#             if len(data) < 3:
#                 grid_z = np.full(grid_x.shape, global_vmin)
#             else:
#                 grid_z = griddata(
#                     (data[x_col].values, data[y_col].values),
#                     data[value_col].values,
#                     (grid_x, grid_y),
#                     method='cubic'
#                 )
#                 if grid_z is None or np.all(np.isnan(grid_z)):
#                     grid_z = np.full(grid_x.shape, global_vmin)
#                 else:
#                     grid_z = np.nan_to_num(grid_z, nan=global_vmin)
#             # Explicitly define levels so that the global color range is used in every frame.
#             levels = np.linspace(global_vmin, global_vmax, 100)
#             contf = ax.contourf(
#                 grid_x, grid_y, grid_z, levels=levels, cmap=hexbin_color,
#                 norm=norm, alpha=0.8
#             )
#             if overlay_contour_lines:
#                 ax.contour(
#                     grid_x, grid_y, grid_z, levels=levels, colors='white', linewidths=1
#                 )
#             if show_colorbar:
#                 fig.subplots_adjust(right=0.85)
#                 cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
#                 fig.colorbar(contf, cax=cbar_ax, norm=norm, ticks=fixed_ticks, label=value_col)

#         else:
#             raise ValueError("plot_method must be either 'hexbin' or 'contour'")

#         ax.set_xlim(global_x_min, global_x_max)
#         ax.set_ylim(global_y_min, global_y_max)
#         ax.set_aspect('equal')
#         ax.set_title(title)
#         ax.set_xlabel(x_col)
#         ax.set_ylabel(y_col)
#         if cell_edge_points is not None:
#             ax.plot(cell_edge_points[:, 0], cell_edge_points[:, 1], color='black', lw=2)
#         return fig, ax


#     # Either a single plot (if time_bin is None) or multiple frames.
#     if time_bin is None:
#         title = "Heatmap: All times"
#         fig, ax = create_plot(df, title)
#         file_name = os.path.join(export_path, "heatmap_whole.png")
#         fig.savefig(file_name, dpi=150)
#         plt.close(fig)
#         n_frames = 1
#     else:
#         t_min = df[time_col].min()
#         t_max = df[time_col].max()
#         n_frames = 0
#         image_files = []
#         current_t = t_min
#         while current_t <= t_max:
#             t_end = current_t + time_bin
#             window_df = df[(df[time_col] >= current_t) & (df[time_col] < t_end)]
#             title = f"Time window: {current_t:.2f} to {t_end:.2f}"
#             fig, ax = create_plot(window_df, title)
#             file_name = os.path.join(export_path, f"heatmap_{n_frames:03d}.png")
#             fig.savefig(file_name, dpi=150)
#             plt.close(fig)
#             image_files.append(file_name)
#             n_frames += 1
#             current_t = t_end

#         if output_format.lower() == 'gif':
#             gif_file = os.path.join(export_path, "heatmap_animation.gif")
#             images = [imageio.imread(fn) for fn in image_files]
#             try:
#                 imageio.mimsave(gif_file, images, duration=gif_frame_duration)
#             except ValueError as ve:
#                 print("Error while creating GIF:", ve)
#             summary_message = (
#                 f"Generated {n_frames} frames (GIF frame duration = {gif_frame_duration} s) using "
#                 f"time_bin = {time_bin}, spatial_unit = {spatial_unit}, plot_method = {plot_method}, "
#                 f"hexbin_color = {hexbin_color}, gridsize = {spatial_gridsize}, box_size = {box_size}."
#             )
#             print(summary_message)
#             print(f"Animated GIF saved to {gif_file}")
#         else:
#             summary_message = (
#                 f"Generated {n_frames} frames using time_bin = {time_bin}, spatial_unit = {spatial_unit}, "
#                 f"plot_method = {plot_method}, hexbin_color = {hexbin_color}, gridsize = {spatial_gridsize}, "
#                 f"box_size = {box_size}."
#             )
#             print(summary_message)
#             print(f"Frames saved as individual PNG files in {export_path}")

#     if time_bin is None:
#         summary_message = (
#             f"Generated a single plot using spatial_unit = {spatial_unit}, plot_method = {plot_method}, "
#             f"hexbin_color = {hexbin_color}, gridsize = {spatial_gridsize}, box_size = {box_size}."
#         )
#         print(summary_message)
