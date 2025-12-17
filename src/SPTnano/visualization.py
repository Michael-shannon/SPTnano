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
import polars as pl
import pims
import seaborn as sns
import skimage.io as skio
import xarray as xr  # for converting arrays to DataArray
from scipy.stats import gaussian_kde
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
from .helper_scripts import center_scale_data

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
def _process_palette(palette, n_colors):
    """
    Process palette parameter that can be either a string (palette name) or a list of colors.
    
    Parameters
    ----------
    palette : str or list
        Either a palette name ('colorblind', 'Dark2', etc.) or a list of colors (hex codes, named colors, RGB tuples)
    n_colors : int
        Number of colors needed
        
    Returns
    -------
    list
        List of colors, cycling if needed
    """
    if isinstance(palette, (list, tuple)):
        # Custom colors provided - cycle if needed
        return [palette[i % len(palette)] for i in range(n_colors)]
    elif palette == "colorblind":
        # Use Wong 2011 colorblind-friendly palette
        colorblind_colors = [
            '#0173B2',  # Blue
            '#DE8F05',  # Orange  
            '#029E73',  # Green
            '#CC78BC',  # Purple
            '#CA9161',  # Brown
            '#FBAFE4',  # Pink
            '#949494',  # Gray
            '#ECE133',  # Yellow
            '#56B4E9',  # Sky blue
            '#D55E00'   # Vermillion
        ]
        return [colorblind_colors[i % len(colorblind_colors)] for i in range(n_colors)]
    else:
        # Use seaborn/matplotlib palette
        import seaborn as sns
        return sns.color_palette(palette, n_colors)


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


# Helper functions for dataframe type detection and operations
def _is_polars(df):
    """Check if dataframe is a Polars DataFrame."""
    return isinstance(df, pl.DataFrame)


def _is_pandas(df):
    """Check if dataframe is a Pandas DataFrame."""
    return isinstance(df, pd.DataFrame)


def _get_column(df, column):
    """Get a column from either Polars or Pandas dataframe."""
    if _is_polars(df):
        return df[column]
    else:
        return df[column]


def _get_unique_categories(df, column, order=None):
    """Get unique categories from a column, handling both Polars and Pandas."""
    if _is_polars(df):
        if order is not None:
            return order
        return df[column].unique().to_list()
    else:
        # Pandas logic with categorical
        if not pd.api.types.is_categorical_dtype(df[column]):
            df[column] = df[column].astype("category")
        if order is not None:
            df[column] = pd.Categorical(df[column], categories=order, ordered=True)
        return df[column].cat.categories.tolist()


def _filter_dataframe(df, column, value):
    """Filter dataframe by column value, handling both Polars and Pandas."""
    if _is_polars(df):
        return df.filter(pl.col(column) == value)
    else:
        return df[df[column] == value]


def _filter_range(df, column, lower, upper):
    """Filter dataframe by range, handling both Polars and Pandas."""
    if _is_polars(df):
        return df.filter((pl.col(column) >= lower) & (pl.col(column) <= upper))
    else:
        return df[(df[column] >= lower) & (df[column] <= upper)]


def _column_min(df, column):
    """Get column minimum, handling both Polars and Pandas."""
    if _is_polars(df):
        return df[column].min()
    else:
        return df[column].min()


def _column_max(df, column):
    """Get column maximum, handling both Polars and Pandas."""
    if _is_polars(df):
        return df[column].max()
    else:
        return df[column].max()


def _column_mean(df, column):
    """Get column mean, handling both Polars and Pandas."""
    if _is_polars(df):
        return df[column].mean()
    else:
        return df[column].mean()


def _column_median(df, column):
    """Get column median, handling both Polars and Pandas."""
    if _is_polars(df):
        return df[column].median()
    else:
        return df[column].median()


def _column_sum(df, column):
    """Get column sum, handling both Polars and Pandas."""
    if _is_polars(df):
        return df[column].sum()
    else:
        return df[column].sum()


def _column_any(df, column):
    """Check if any value in column is True, handling both Polars and Pandas."""
    if _is_polars(df):
        return df[column].any()
    else:
        return df[column].any()


def _create_log_column(df, feature, new_feature, log_base):
    """Create a log-transformed column, handling both Polars and Pandas."""
    if _is_polars(df):
        if log_base == 10:
            return df.with_columns(pl.col(feature).log10().alias(new_feature))
        elif log_base == 2:
            return df.with_columns(pl.col(feature).log().alias(new_feature) / np.log(2))
        else:
            return df.with_columns(pl.col(feature).log().alias(new_feature) / np.log(log_base))
    else:
        # Pandas - modify in place
        if log_base == 10:
            df[new_feature] = np.log10(df[feature])
        elif log_base == 2:
            df[new_feature] = np.log2(df[feature])
        else:
            df[new_feature] = np.log(df[feature]) / np.log(log_base)
        return df


def _to_numpy(series):
    """Convert a series to numpy array, handling both Polars and Pandas."""
    if isinstance(series, pl.Series):
        return series.to_numpy()
    elif isinstance(series, pl.DataFrame):
        # If single column dataframe
        return series.to_numpy().flatten()
    else:
        # Pandas
        return series.values


def _groupby_agg(df, groupby_col, agg_col, agg_func):
    """Perform groupby aggregation, handling both Polars and Pandas."""
    if _is_polars(df):
        if agg_func == 'median':
            result = df.group_by(groupby_col).agg(pl.col(agg_col).median())
        elif agg_func == 'mean':
            result = df.group_by(groupby_col).agg(pl.col(agg_col).mean())
        else:
            raise ValueError(f"Unsupported aggregation function: {agg_func}")
        # Return as dict for consistency
        return dict(zip(result[groupby_col].to_list(), result[agg_col].to_list()))
    else:
        if agg_func == 'median':
            return df.groupby(groupby_col)[agg_col].median()
        elif agg_func == 'mean':
            return df.groupby(groupby_col)[agg_col].mean()
        else:
            raise ValueError(f"Unsupported aggregation function: {agg_func}")


def _get_unique_values(df, column):
    """Get unique values from a column, handling both Polars and Pandas."""
    if _is_polars(df):
        return df[column].unique().to_list()
    else:
        return df[column].unique()


def plot_histograms(
    data_df,
    feature,
    bins=100,
    separate=None,
    xlimit=None,
    small_multiples=False,
    palette="colorblind",  # Can be string (palette name) or list of colors
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
    Plot histograms with flexible color options and customization.
    
    Compatible with both Polars and Pandas DataFrames.
    
    Parameters
    ----------
    data_df : DataFrame (Polars or Pandas)
        Input dataframe containing the data to plot.
    palette : str or list, default "colorblind"
        Color palette for categories. Can be:
        - String: palette name ('colorblind' uses Wong 2011, 'Dark2', 'Set2', etc.)
        - List: custom colors as hex codes, named colors, or RGB tuples
        Example: ['#0173B2', '#DE8F05', '#029E73'] or ['red', 'blue', 'green']
    
    (Other parameters documented elsewhere)
    """
    # Detect dataframe type
    is_polars_df = _is_polars(data_df)
    
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
        # Check for non-positive values
        if is_polars_df:
            has_nonpositive = (data_df[feature] <= 0).any()
            if has_nonpositive:
                num_negative = (data_df[feature] <= 0).sum()
                min_val = data_df[feature].min()
                max_val = data_df[feature].max()
        else:
            has_nonpositive = (data_df[feature] <= 0).any()
            if has_nonpositive:
                num_negative = (data_df[feature] <= 0).sum()
                min_val = data_df[feature].min()
                max_val = data_df[feature].max()
                
        if has_nonpositive:
            print(f"⚠️  Skipping histogram for '{feature}': Cannot use log scale with {num_negative} non-positive values (log requires all positive values).")
            print(f"   Min value: {min_val}, Max value: {max_val}")
            print(f"   Consider using log_scale=False or filtering data to positive values only.")
            plt.close('all')  # Clean up any matplotlib artifacts
            return None
            
        data_df = _create_log_column(data_df, feature, new_feature, log_base)
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
        unique_categories = _get_unique_categories(data_df, separate, order)
    else:
        unique_categories = [None]

    color_palette = _process_palette(palette, len(unique_categories))

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
        global_lower_bound = _column_min(data_df, feature_to_plot)
        global_upper_bound = (
            xlimit if xlimit is not None else _column_max(data_df, feature_to_plot)
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
        
        # Get subset values
        if category:
            subset_df = _filter_dataframe(data_df, separate, category)
            if is_polars_df:
                subsetvalues_col = subset_df[feature_to_plot]
            else:
                subsetvalues_col = subset_df[feature_to_plot]
        else:
            subsetvalues_col = data_df[feature_to_plot]
        
        # Filter by range
        if is_polars_df:
            filtered_df = _filter_range(
                pl.DataFrame({feature_to_plot: subsetvalues_col}),
                feature_to_plot,
                global_lower_bound,
                global_upper_bound
            )
            subsetvalues = _to_numpy(filtered_df[feature_to_plot])
        else:
            mask = (subsetvalues_col >= global_lower_bound) & (subsetvalues_col <= global_upper_bound)
            subsetvalues = subsetvalues_col[mask].values

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

        # Calculate average
        if average == "mean":
            avg_value = np.mean(subsetvalues)
        else:
            avg_value = np.median(subsetvalues)
            
        annotation_value = (
            log_base**avg_value
            if log_scale and log_axis_label == "actual"
            else avg_value
        )

        if showavg:
            ax.axvline(avg_value, color=line_color, linestyle="--")

        # Apply line_color styling to each subplot individually
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
            num_negative = (data_df[feature] <= 0).sum()
            print(f"⚠️  Skipping histogram for '{feature}': Cannot use log scale with {num_negative} non-positive values (log requires all positive values).")
            print(f"   Min value: {data_df[feature].min()}, Max value: {data_df[feature].max()}")
            print(f"   Consider using log_scale=False or filtering data to positive values only.")
            plt.close('all')  # Clean up any matplotlib artifacts
            return None
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

        # Apply line_color styling to each subplot individually
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
    annotation_decimals=2,
):
    """
    Plot boxplots for the specified feature against a categorical x_category with custom order and styling options.

    Compatible with both Polars and Pandas DataFrames.

    Parameters
    ----------
    data_df : DataFrame (Polars or Pandas)
        DataFrame containing the data.
    feature : str
        The feature to plot on the y-axis.
    x_category : str
        The categorical feature to plot on the x-axis.
    font_size : int, optional
        Font size for the plot text. Default is 12.
    order : list, optional
        Specific order for the categories. Default is None.
    palette : str or list, optional
        Color palette for the plot. Can be:
        - String: palette name ('colorblind' uses Wong 2011, 'Dark2', 'Set2', etc.)
        - List: custom colors as hex codes, named colors, or RGB tuples
        Example: ['#0173B2', '#DE8F05', '#029E73'] or ['red', 'blue', 'green']
        Default is 'colorblind'.
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
    annotation_decimals : int, optional
        Number of decimal places to display for annotated median values. Default is 2.

    Returns
    -------
    str or None
        When export_format is 'svg' and return_svg is True, returns the cleaned SVG data as a string.
        Otherwise, returns None.

    """
    # Detect dataframe type and convert if needed
    is_polars_df = _is_polars(data_df)
    
    # For seaborn plotting, we need pandas DataFrame
    # Store the original for median calculation if using polars
    original_df = data_df
    if is_polars_df:
        # Convert to pandas for seaborn - only select needed columns for efficiency
        plot_df = data_df.select([x_category, feature]).to_pandas()
    else:
        plot_df = data_df
    
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

    # Get number of categories
    if order is not None:
        n_categories = len(order)
    else:
        if is_polars_df:
            n_categories = original_df[x_category].n_unique()
        else:
            n_categories = len(plot_df[x_category].unique())
    
    # Process palette to handle both string names and custom color lists
    processed_palette = _process_palette(palette, n_categories)

    if bw:
        boxplot = sns.boxplot(
            x=x_category,
            y=feature,
            data=plot_df,
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
            data=plot_df,
            palette=processed_palette,
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
            data=plot_df,
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
        # Calculate medians using the helper function
        medians_dict = _groupby_agg(original_df, x_category, feature, 'median')
        current_y_max = plt.ylim()[1]
        
        # Sort medians by order if provided
        if order:
            sorted_medians_list = [medians_dict[cat] for cat in order if cat in medians_dict]
        else:
            sorted_medians_list = list(medians_dict.values())
            
        for i, median in enumerate(sorted_medians_list):
            plt.text(
                i,
                current_y_max * annotatemultiplier,
                f"{median:.{annotation_decimals}f}",
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
    stack_category=None,
    order=None,
    stack_order=None,
    title=None,
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
    show_percentages=True,
    percentage_threshold=5.0,
    ylabel="Percentage (%)",
    legend_title=None,
    grid=False,
):
    """
    Plot a generalized stacked bar chart showing the percentage distribution of any categorical variable.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the data with the specified categorical columns.
    x_category : str
        The column name of the category to plot on the x-axis.
    stack_category : str, optional
        The column name of the category to stack. If None, defaults to 'motion_class' for backward compatibility.
    order : list, optional
        Custom order for the categories on the x-axis. Default is None.
    stack_order : list, optional
        Custom order for the stacked categories. Default is None.
    title : str, optional
        Title for the plot. If None, generates a default title.
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
    custom : str, optional
        Custom string to add to filename. Default is "_".
    show_plot : bool, optional
        Whether to display the plot in the notebook. Default is True.
    return_svg : bool, optional
        If True and export_format is 'svg', returns the post-processed SVG image data as a string.
    show_percentages : bool, optional
        Whether to annotate percentages on bars. Default is True.
    percentage_threshold : float, optional
        Only show percentage labels on bars with height >= this threshold. Default is 5.0.
    ylabel : str, optional
        Label for y-axis. Default is "Percentage (%)".
    legend_title : str, optional
        Title for the legend. If None, uses the stack_category name.
    grid : bool, optional
        Whether to show grid lines. Default is False.

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

    # Set default stack_category for backward compatibility
    if stack_category is None:
        stack_category = "motion_class"
        
    # Check if required columns exist
    if x_category not in df.columns:
        raise ValueError(f"Column '{x_category}' not found in dataframe")
    if stack_category not in df.columns:
        raise ValueError(f"Column '{stack_category}' not found in dataframe")

    # Apply custom order if provided
    if order is not None:
        df[x_category] = pd.Categorical(df[x_category], categories=order, ordered=True)
    
    if stack_order is not None:
        df[stack_category] = pd.Categorical(df[stack_category], categories=stack_order, ordered=True)

    # Calculate percentage data for each stack category
    percentage_data = (
        df.groupby([x_category, stack_category])
        .size()
        .unstack(fill_value=0)
        .apply(lambda x: x / x.sum() * 100, axis=1)
    )

    # Determine unique stack categories and assign colors
    stack_categories = df[stack_category].unique() if stack_order is None else stack_order
    if colormap == "colorblind":
        colors = sns.color_palette("colorblind", len(stack_categories))
    elif colormap == "Dark2":
        cmap = cm.get_cmap("Dark2", len(stack_categories))
        colors = cmap(np.linspace(0, 1, len(stack_categories)))
    else:
        colors = plt.get_cmap(colormap, len(stack_categories)).colors

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

    # Annotate percentages on the bars if enabled
    if show_percentages:
        for patch in ax.patches:
            width, height = patch.get_width(), patch.get_height()
            x, y = patch.get_xy()
            if height >= percentage_threshold:  # Only annotate if above threshold
                ax.annotate(
                    f"{height:.1f}%",
                    (x + width / 2, y + height / 2),
                    ha="center",
                    va="center",
                    fontsize=max(8, font_size - 2),  # Slightly smaller font for percentages
                    color=line_color,
                    weight="bold",
                )

    # Set title (generate default if not provided)
    if title is None:
        title = f"Distribution of {stack_category.replace('_', ' ').title()}"
    
    # Customize text elements
    ax.set_title(title, fontsize=font_size, color=line_color, pad=20)
    ax.set_xlabel(x_category.replace('_', ' ').title(), fontsize=font_size, color=line_color)
    ax.set_ylabel(ylabel, fontsize=font_size, color=line_color)
    ax.tick_params(
        axis="both",
        which="both",
        color=line_color,
        labelcolor=line_color,
        labelsize=font_size,
    )

    # Rotate x-tick labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Set legend title
    if legend_title is None:
        legend_title = stack_category.replace('_', ' ').title()

    # Move the legend outside the plot
    legend = plt.legend(
        title=legend_title,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        title_fontsize=font_size,
        prop={"size": font_size},
        frameon=False,
    )
    for text in legend.get_texts():
        text.set_color(line_color)
    legend.get_title().set_color(line_color)

    # Add grid if requested
    if grid:
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


def plot_ratio_analysis_svg(
    data_df,
    x_category,
    numerator_category="high",
    denominator_category="low",
    ratio_column=None,
    calculate_ratio=True,
    order=None,
    title=None,
    font_size=12,
    palette="colorblind",
    background="white",
    transparent=False,
    line_color="black",
    show_plot=True,
    master_dir=None,
    grid=True,
    y_max=None,
    y_min=None,
    figsize=(10, 8),
    annotate_values=True,
    rotation=45,
    custom="_",
    export_format="svg",
    return_svg=False,
    plot_type="bar",
    add_reference_line=True,
    reference_value=1.0,
):
    """
    Plot ratio analysis comparing two categories (e.g., high:low diffusion coefficients).
    
    Parameters
    ----------
    data_df : DataFrame
        DataFrame containing the data for ratio analysis.
    x_category : str
        The column name of the category to plot on the x-axis.
    numerator_category : str, optional
        Value in the ratio column to use as numerator. Default is "high".
    denominator_category : str, optional
        Value in the ratio column to use as denominator. Default is "low".
    ratio_column : str, optional
        Column containing the categories to calculate ratios from (e.g., 'simple_threshold').
        If None and calculate_ratio=True, will look for 'simple_threshold'.
    calculate_ratio : bool, optional
        Whether to calculate ratios from counts or use pre-calculated values. Default is True.
    order : list, optional
        Custom order for the categories on the x-axis. Default is None.
    title : str, optional
        Title for the plot. If None, generates a default title.
    font_size : int, optional
        Font size for the plot text. Default is 12.
    palette : str, optional
        Color palette for the plot. Default is "colorblind".
    background : str, optional
        Background color for the figure. Default is "white".
    transparent : bool, optional
        If True, makes the plot background transparent. Default is False.
    line_color : str, optional
        Color of all lines, text, and borders. Default is "black".
    show_plot : bool, optional
        Whether to display the plot. Default is True.
    master_dir : str, optional
        Directory to save the plot. Default is None.
    grid : bool, optional
        Whether to show grid lines. Default is True.
    y_max : float, optional
        Maximum y-axis value. Default is None (auto).
    y_min : float, optional
        Minimum y-axis value. Default is None (auto).
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (10, 8).
    annotate_values : bool, optional
        Whether to annotate values on bars/points. Default is True.
    rotation : int, optional
        Rotation angle for x-axis labels. Default is 45.
    custom : str, optional
        Custom string for filename. Default is "_".
    export_format : str, optional
        Export format ('png' or 'svg'). Default is "svg".
    return_svg : bool, optional
        Whether to return SVG string. Default is False.
    plot_type : str, optional
        Type of plot ('bar', 'point', 'violin'). Default is "bar".
    add_reference_line : bool, optional
        Whether to add horizontal reference line. Default is True.
    reference_value : float, optional
        Y-value for reference line. Default is 1.0.
        
    Returns
    -------
    str or None
        SVG string if return_svg=True and format='svg', otherwise None.
    """
    
    figure_background = "none" if transparent else background
    axis_background = (0, 0, 0, 0) if transparent else background
    
    # Set defaults for ratio calculation
    if ratio_column is None and calculate_ratio:
        ratio_column = "simple_threshold"
    
    if calculate_ratio:
        # Calculate ratios from counts
        if ratio_column not in data_df.columns:
            raise ValueError(f"Column '{ratio_column}' not found for ratio calculation")
        
        # Group and count
        count_data = data_df.groupby([x_category, ratio_column]).size().reset_index(name='count')
        pivot_data = count_data.pivot_table(
            index=x_category, 
            columns=ratio_column, 
            values='count', 
            fill_value=0
        )
        
        # Calculate ratio (add small constant to avoid division by zero)
        if numerator_category not in pivot_data.columns or denominator_category not in pivot_data.columns:
            available_categories = list(pivot_data.columns)
            raise ValueError(f"Categories '{numerator_category}' or '{denominator_category}' not found. Available: {available_categories}")
        
        ratio_data = pivot_data[numerator_category] / (pivot_data[denominator_category] + 0.1)
        plot_data = ratio_data.reset_index()
        plot_data.columns = [x_category, 'ratio']
        y_column = 'ratio'
        y_label = f"Ratio ({numerator_category}:{denominator_category})"
    else:
        # Use pre-calculated ratio column
        plot_data = data_df.copy()
        y_column = ratio_column or 'ratio'
        y_label = f"Ratio"
    
    # Apply order if provided
    if order is not None:
        plot_data[x_category] = pd.Categorical(plot_data[x_category], categories=order, ordered=True)
        plot_data = plot_data.sort_values(x_category)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor=figure_background)
    ax.set_facecolor(axis_background)
    
    # Get colors
    colors = sns.color_palette(palette, len(plot_data))
    
    # Create plot based on type
    if plot_type == "bar":
        # Use individual colors for each bar if enough colors available
        bar_colors = colors if len(colors) >= len(plot_data) else [colors[i % len(colors)] for i in range(len(plot_data))]
        
        bars = ax.bar(plot_data[x_category], plot_data[y_column], 
                     color=bar_colors, edgecolor=line_color, linewidth=1)
        
        if annotate_values:
            for bar in bars:
                height = bar.get_height()
                # Place annotation in the middle of the bar (both horizontally and vertically)
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                           ha='center', va='center',
                           fontsize=font_size-1, color='k', weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                                   edgecolor='k', alpha=0.8))
                           
    elif plot_type == "point":
        ax.scatter(plot_data[x_category], plot_data[y_column], 
                  color=colors[0], s=100, edgecolors=line_color, linewidth=1)
        ax.plot(plot_data[x_category], plot_data[y_column], 
               color=colors[0], alpha=0.7, linewidth=2)
        
        if annotate_values:
            for i, (x, y) in enumerate(zip(plot_data[x_category], plot_data[y_column])):
                ax.annotate(f'{y:.2f}', xy=(x, y), xytext=(5, 5),
                           textcoords="offset points", ha='left',
                           fontsize=font_size-2, color=line_color)
    
    # Add reference line
    if add_reference_line:
        ax.axhline(y=reference_value, color=line_color, linestyle='--', 
                  alpha=0.7, label=f'Reference ({reference_value})')
    
    # Set title and labels
    if title is None:
        title = f"{numerator_category.title()}:{denominator_category.title()} Ratio by {x_category.replace('_', ' ').title()}"
    
    ax.set_title(title, fontsize=font_size, color=line_color, pad=20)
    ax.set_xlabel(x_category.replace('_', ' ').title(), fontsize=font_size, color=line_color)
    ax.set_ylabel(y_label, fontsize=font_size, color=line_color)
    
    # Set y-limits if provided
    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)
    
    # Customize ticks
    ax.tick_params(axis="both", which="both", color=line_color, 
                  labelcolor=line_color, labelsize=font_size)
    
    # Rotate x-labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha="right")
    
    # Add grid
    if grid:
        ax.grid(True, linestyle="--", linewidth=0.5, color=line_color, alpha=0.7, axis="y")
    
    # Customize spines
    for spine in ax.spines.values():
        spine.set_edgecolor(line_color)
        if transparent:
            spine.set_alpha(0.9)
    
    plt.tight_layout()
    
    # Save file
    svg_data = None
    master_dir = master_dir or "plots"
    os.makedirs(master_dir, exist_ok=True)
    ext = export_format.lower()
    if ext not in ["png", "svg"]:
        print("Invalid export format specified. Defaulting to 'png'.")
        ext = "png"
    filename = f"{master_dir}/ratio_analysis_{x_category}_{custom}.{ext}"
    plt.savefig(filename, bbox_inches="tight", transparent=transparent, format=ext)
    
    # Process SVG if needed
    if ext == "svg":
        with open(filename, encoding="utf-8") as f:
            svg_data = f.read()
        # Clean SVG similar to other functions
        svg_data = re.sub(r'<clipPath id="[^"]*">.*?</clipPath>', "", svg_data, flags=re.DOTALL)
        svg_data = re.sub(r'\s*clip-path="url\([^)]*\)"', "", svg_data)
        svg_data = re.sub(r"<metadata>.*?</metadata>", "", svg_data, flags=re.DOTALL)
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

        else:
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
    save_path=None,  # Optional custom save path for movies
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
      - save_movie_flag: If True, save a movie of the visualization.
      - save_path: Optional custom path for saving movies. If None, uses default path with format "{condition}_{cell}.mov".
                   Supported formats: .mov, .mp4, .avi, .gif, .mpg, .mpeg, .mkv, .wmv
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
        if save_path is None:
            # Use default path if no custom path provided
            movie_dir = os.path.join(master_dir, "movies")
            print(f"Saving movie to: {movie_dir}")
            if not os.path.exists(movie_dir):
                os.makedirs(movie_dir, exist_ok=True)
            movie_path = os.path.join(movie_dir, f"{condition}_{cell}.mov")
        else:
            # Use custom save path
            movie_path = save_path
            # Create directory if it doesn't exist
            movie_dir = os.path.dirname(movie_path)
            if movie_dir and not os.path.exists(movie_dir):
                os.makedirs(movie_dir, exist_ok=True)
            print(f"Saving movie to custom path: {movie_path}")
        
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


def auto_select_time_period(tracks_df, target_tracks=30, frame_interval=0.01, start_offset=1.0):
    """
    Automatically select a time period based on the number of unique tracks.
    
    Starting from start_offset seconds into the data, this function finds the first
    time window where at least target_tracks unique tracks are present.
    
    Parameters
    ----------
    tracks_df : polars.DataFrame or pandas.DataFrame
        DataFrame containing track data with 'unique_id' and 'frame' columns
    target_tracks : int, default 30
        Target number of unique tracks to find in the time window
    frame_interval : float, default 0.01
        Time between frames in seconds
    start_offset : float, default 1.0
        Starting time offset in seconds (to skip initial frames)
        
    Returns
    -------
    tuple
        (time_start, time_end) in seconds, or (None, None) if insufficient tracks found
    """
    import polars as pl
    
    # Convert to polars if pandas
    if hasattr(tracks_df, 'to_polars'):
        df = tracks_df.to_polars()
    elif not hasattr(tracks_df, 'lazy'):
        # It's pandas, convert to polars
        df = pl.from_pandas(tracks_df)
    else:
        df = tracks_df
    
    # Get total unique tracks in the dataset
    total_unique_tracks = df['unique_id'].n_unique()
    
    # If we don't have enough tracks overall, adjust target or return fallback
    if total_unique_tracks < target_tracks:
        # Use a fallback: target 80% of available tracks or at least 5 tracks
        adjusted_target = max(5, int(total_unique_tracks * 0.8))
        print(f"Dataset has only {total_unique_tracks} total tracks, adjusting target to {adjusted_target}")
    else:
        adjusted_target = target_tracks
    
    # Get frame range
    min_frame = int(df['frame'].min())
    max_frame = int(df['frame'].max())
    
    # Convert start_offset to frame number
    start_frame = int(start_offset / frame_interval)
    start_frame = max(min_frame, start_frame)
    
    # Search for suitable time window, incrementing by 1 second (1/frame_interval frames)
    frames_per_second = int(1.0 / frame_interval)
    
    # Try progressively longer time windows
    for current_start_frame in range(start_frame, max_frame, frames_per_second):
        # Try different window sizes: 5s, 10s, 15s, 20s, 30s, 45s, 60s
        for window_duration in [5, 10, 15, 20, 30, 45, 60]:
            window_frames = int(window_duration / frame_interval)
            end_frame = current_start_frame + window_frames
            
            if end_frame > max_frame:
                continue
                
            # Count unique tracks in this window
            window_df = df.filter(
                (pl.col('frame') >= current_start_frame) & 
                (pl.col('frame') <= end_frame)
            )
            
            unique_tracks = window_df['unique_id'].n_unique()
            
            if unique_tracks >= adjusted_target:
                time_start = current_start_frame * frame_interval
                time_end = end_frame * frame_interval
                return time_start, time_end
    
    # If no suitable window found with adjusted target, try a very permissive fallback
    # Just find any window with at least 5 tracks
    if adjusted_target > 5:
        for current_start_frame in range(start_frame, max_frame, frames_per_second):
            for window_duration in [10, 20, 30, 60]:
                window_frames = int(window_duration / frame_interval)
                end_frame = current_start_frame + window_frames
                
                if end_frame > max_frame:
                    continue
                    
                window_df = df.filter(
                    (pl.col('frame') >= current_start_frame) & 
                    (pl.col('frame') <= end_frame)
                )
                
                unique_tracks = window_df['unique_id'].n_unique()
                
                if unique_tracks >= 5:
                    time_start = current_start_frame * frame_interval
                    time_end = end_frame * frame_interval
                    print(f"Fallback: found {unique_tracks} tracks (less than target {target_tracks})")
                    return time_start, time_end
    
    # If still no suitable window found, return None
    print(f"Warning: Could not find time window with sufficient tracks (target: {target_tracks}, total: {total_unique_tracks})")
    return None, None


def plot_tracks_polars_static(
    tracks_df,
    color_by="particle",
    time_start=None,
    time_end=None,
    overlay_image=False,
    master_dir=None,
    display_final_frame=True,
    max_projection=False,
    contrast_limits=None,
    invert_image=False,
    scale_bar_length=2,  # in microns
    scale_bar_color="black",
    scale_bar_thickness=2,
    transparent_background=True,
    save_path=None,
    figsize=(3, 3),  # figure size in inches
    plot_size_um=10,  # final data range (in microns)
    line_thickness=0.8,  # thickness of track lines
    dpi=200,
    export_format="svg",  # 'png' or 'svg'
    show_plot=True,  # whether to show the plot after saving/exporting
    colorway="tab20",
    frame_interval=0.01,  # seconds per frame
    pixel_size_um=0.065,  # microns per pixel conversion factor
    gradient=False,
    order=None,  # Order for categorical coloring
    motion_type=None,  # Filter by motion type if column exists
    return_svg=False,  # if True and exporting as SVG, return the SVG string
    auto_select_tracks=30,  # target number of tracks for auto time selection (disabled if time_start/end specified)
    filename=None,  # filename to display on the plot
):
    """
    Create a static plot of tracks using Polars DataFrame directly with full functionality.
    
    Enhanced version with image overlay, proper timing annotations, and all original features
    while maintaining streamlined Polars input without complex filtering logic.
    
    Parameters
    ----------
    tracks_df : polars.DataFrame
        Pre-filtered DataFrame containing tracks to plot
    color_by : str
        Column name to use for coloring tracks
    time_start : float, optional
        Start time in seconds for temporal filtering
    time_end : float, optional  
        End time in seconds for temporal filtering
    overlay_image : bool
        Whether to overlay tracks on microscopy image
    master_dir : str, optional
        Master directory containing image data (required if overlay_image=True)
    display_final_frame : bool
        Whether to show final frame of image (vs max projection)
    max_projection : bool
        Whether to use max projection of image stack
    contrast_limits : tuple, optional
        (lower, upper) contrast limits for image display
    invert_image : bool
        Whether to invert image colors
    scale_bar_length : float
        Length of scale bar in microns
    scale_bar_color : str
        Color of scale bar and labels
    scale_bar_thickness : float
        Thickness of scale bar line
    transparent_background : bool
        Whether to use transparent background
    save_path : str, optional
        Directory to save plot (if None, won't save)
    figsize : tuple
        Figure size in inches (width, height)
    plot_size_um : float
        Size of plot area in microns
    line_thickness : float
        Thickness of track lines
    dpi : int
        Resolution for saving
    export_format : str
        Format for saving ('png' or 'svg')
    show_plot : bool
        Whether to display the plot
    colorway : str
        Matplotlib colormap name
    frame_interval : float
        Time between frames in seconds
    pixel_size_um : float
        Microns per pixel conversion factor
    gradient : bool
        Whether to apply gradient effect (for future enhancement)
    order : list, optional
        Order for categorical coloring
    motion_type : str, optional
        Filter by specific motion type if motion_class column exists
    return_svg : bool
        If True and export_format='svg', return SVG string
    auto_select_tracks : int, default 30
        Target number of tracks for automatic time period selection.
        Only used when both time_start and time_end are None.
    filename : str, optional
        Filename to display on the plot as an annotation
        
    Returns
    -------
    dict
        Dictionary with plotted track info, filtered data, and optional SVG string
    """
    import polars as pl
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import re
    from skimage.io import imread
    from skimage.util import img_as_float
    
    # Convert to pandas for matplotlib compatibility
    # (Polars -> Pandas conversion for plotting only)
    df_pandas = tracks_df.to_pandas()
    
    # Filter by motion type if specified
    if motion_type is not None and 'motion_class' in df_pandas.columns:
        df_pandas = df_pandas[df_pandas['motion_class'] == motion_type]
    
    # Text scaling based on figure size
    baseline_width = 6.0
    baseline_font = 12
    scale_factor = figsize[0] / baseline_width
    default_font = baseline_font * scale_factor
    
    plt.rcParams.update({
        "font.size": default_font,
        "axes.titlesize": default_font,
        "axes.labelsize": default_font,
        "xtick.labelsize": default_font,
        "ytick.labelsize": default_font,
    })
    
    # Ensure required columns exist
    required_cols = ['unique_id', 'frame', 'x_um', 'y_um']
    missing_cols = [col for col in required_cols if col not in df_pandas.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Temporal filtering with auto-selection support
    min_frame = df_pandas['frame'].min()
    max_frame = df_pandas['frame'].max()
    
    # Auto-select time period if both time_start and time_end are None
    if time_start is None and time_end is None and auto_select_tracks is not None:
        auto_time_start, auto_time_end = auto_select_time_period(
            tracks_df, 
            target_tracks=auto_select_tracks, 
            frame_interval=frame_interval
        )
        if auto_time_start is not None and auto_time_end is not None:
            time_start, time_end = auto_time_start, auto_time_end
            print(f"Auto-selected time period: {time_start:.2f}s to {time_end:.2f}s "
                  f"(targeting {auto_select_tracks} tracks)")
        else:
            print(f"Auto-selection failed, using full time range (all {tracks_df.shape[0]} data points)")
    
    if time_start is not None:
        start_frame = int(time_start / frame_interval)
        start_frame = max(min_frame, min(start_frame, max_frame))
    else:
        start_frame = min_frame
        
    if time_end is not None:
        end_frame = int(time_end / frame_interval)
        end_frame = max(min_frame, min(end_frame, max_frame))
    else:
        end_frame = max_frame
    
    if time_start is not None or time_end is not None:
        df_pandas = df_pandas[
            (df_pandas['frame'] >= start_frame) & 
            (df_pandas['frame'] <= end_frame)
        ]
    
    # Calculate time range for annotation
    time_start_sec = start_frame * frame_interval
    time_end_sec = end_frame * frame_interval
    
    if df_pandas.empty:
        raise ValueError("No data remaining after filtering")
    
    # Enhanced color mapping with order support
    if color_by in df_pandas.columns:
        # Handle categorical vs continuous coloring
        is_categorical = (
            df_pandas[color_by].dtype == 'object' or 
            pd.api.types.is_categorical_dtype(df_pandas[color_by]) or
            color_by in ['cluster', 'cluster_id', 'motion_class'] or 
            'cluster' in color_by.lower()
        )
        
        if is_categorical:
            # Get unique values and apply order if provided
            all_unique_values = sorted([v for v in df_pandas[color_by].unique() if pd.notna(v)])
            
            if order is None:
                unique_classes = all_unique_values
            else:
                # Use provided order, but only include values that exist in data
                unique_classes = [c for c in order if c in all_unique_values]
                # Add any remaining values not in the provided order
                remaining_values = [c for c in all_unique_values if c not in unique_classes]
                unique_classes.extend(remaining_values)
            
            # Create color mapping
            n_colors = len(unique_classes)
            if n_colors > 0:
                colors = plt.cm.get_cmap(colorway)(np.linspace(0, 1, max(n_colors, 1)))
                color_map = dict(zip(unique_classes, colors))
                df_pandas['plot_color'] = df_pandas[color_by].map(color_map)
            else:
                df_pandas['plot_color'] = 'blue'
        else:
            # Continuous coloring
            norm = plt.Normalize(vmin=df_pandas[color_by].min(), 
                               vmax=df_pandas[color_by].max())
            df_pandas['plot_color'] = df_pandas[color_by].apply(
                lambda x: plt.cm.get_cmap(colorway)(norm(x))
            )
            unique_classes = None  # For continuous data
    else:
        # Default color if column doesn't exist
        df_pandas['plot_color'] = 'blue'
        unique_classes = None
    
    # Create figure
    figure_background = "none" if transparent_background else "white"
    axis_background = (0, 0, 0, 0) if transparent_background else "white"
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(figure_background)
    ax.set_facecolor(axis_background)
    
    # Overlay image if requested
    if overlay_image:
        if master_dir is None:
            raise ValueError("master_dir must be provided when overlay_image=True")
            
        # Get filename and condition from data
        if 'filename' in df_pandas.columns and 'condition' in df_pandas.columns:
            filename = df_pandas['filename'].iloc[0]
            condition = df_pandas['condition'].iloc[0]
            
            # Construct image path
            image_filename = filename.replace("_tracked", "") + ".tif"
            image_path = os.path.join(master_dir, "data", condition, image_filename)
            
            try:
                overlay_data = imread(image_path)
                
                # Process image based on settings
                if max_projection:
                    overlay_data = np.max(overlay_data, axis=0)
                elif display_final_frame:
                    overlay_data = overlay_data[-1, :, :]
                    
                overlay_data = img_as_float(overlay_data)
                
                # Apply contrast limits
                if contrast_limits:
                    lower, upper = contrast_limits
                    overlay_data = np.clip((overlay_data - lower) / (upper - lower), 0, 1)
                else:
                    overlay_data = (overlay_data - overlay_data.min()) / (
                        overlay_data.max() - overlay_data.min()
                    )
                
                # Invert if requested
                if invert_image:
                    overlay_data = 1 - overlay_data
                
                # Set image extent in microns
                height, width = overlay_data.shape
                extent = [0, width * pixel_size_um, 0, height * pixel_size_um]
                ax.imshow(overlay_data, cmap="gray", origin="lower", extent=extent, alpha=0.8)
                
            except Exception as e:
                print(f"Warning: Could not load image from {image_path}: {e}")
        else:
            print("Warning: filename and condition columns required for image overlay")
    
    # Plot tracks
    unique_tracks = df_pandas['unique_id'].unique()
    plotted_tracks = []
    
    for track_id in unique_tracks:
        track_data = df_pandas[df_pandas['unique_id'] == track_id].sort_values('frame')
        
        # Skip single-point tracks
        if len(track_data) < 2:
            continue
            
        # Get color for this track
        if 'plot_color' in track_data.columns:
            track_color = track_data['plot_color'].iloc[0]
        else:
            track_color = 'blue'
        
        # Plot track as line
        line = ax.plot(
            track_data['x_um'],
            track_data['y_um'],
            color=track_color,
            linewidth=line_thickness,
            alpha=0.8
        )[0]
        
        # Set unique identifier for SVG
        line.set_gid(f"track_{track_id}")
        plotted_tracks.append(track_id)
    
    # Set plot limits and aspect
    ax.set_xlim(0, plot_size_um)
    ax.set_ylim(0, plot_size_um)
    ax.set_aspect("equal", adjustable="datalim")
    
    # Remove axes if no overlay image
    if not overlay_image:
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
    
    # Scale bar label
    ax.text(
        (x_start + x_end) / 2,
        y_bar - margin * 0.3,
        f"{scale_bar_length} µm",
        ha="center",
        va="top",
        fontsize=10 * scale_factor,
        color=scale_bar_color,
    )
    
    # Add Time Range Annotation (the "timer" at the top)
    ax.annotate(
        f"Time: {time_start_sec:.2f}s - {time_end_sec:.2f}s",
        xy=(0.5, 1.02),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=default_font,
        color=scale_bar_color,
    )
    
    # Add filename annotation at the bottom if provided
    if filename is not None:
        ax.annotate(
            filename,
            xy=(0.5, -0.08),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=default_font * 0.8,  # Slightly smaller than time annotation
            color=scale_bar_color,
        )
    
    # Add colorbar or legend if needed
    is_categorical = (
        color_by in df_pandas.columns and 
        (df_pandas[color_by].dtype == 'object' or 
         pd.api.types.is_categorical_dtype(df_pandas[color_by]) or
         color_by in ['cluster', 'cluster_id', 'motion_class'] or 
         'cluster' in color_by.lower())
    )
    
    if color_by in df_pandas.columns and unique_classes is not None:
        if is_categorical:
            # Only show colors for classes actually present in the filtered data
            present_classes = [
                cls for cls in unique_classes if cls in df_pandas[color_by].unique()
            ]
            handles = []
            for cls in present_classes[:10]:  # Limit to 10 for readability
                if 'color_map' in locals():
                    color = color_map.get(cls, 'gray')
                    handles.append(plt.Line2D([0], [0], color=color, lw=2, label=f"{cls}"))

            if handles:
                ax.legend(
                    handles=handles,
                    loc="upper left",
                    bbox_to_anchor=(1.05, 1),
                    borderaxespad=0.0,
                    title=f"Legend: {color_by}",
                    fontsize=default_font * 0.8,
                    title_fontsize=default_font * 0.8,
                )
        else:
            # Continuous colorbar
            color_min = df_pandas[color_by].min()
            color_max = df_pandas[color_by].max()
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
    
    # Save if requested
    svg_data = None
    if save_path:
        save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else save_path
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        ext = export_format.lower()
        if ext not in ["png", "svg"]:
            print("Invalid export format specified. Defaulting to 'png'.")
            ext = "png"
            
        # Create filename based on available data
        if 'filename' in df_pandas.columns:
            base_name = df_pandas['filename'].iloc[0].split(".")[0]
        else:
            base_name = f"tracks_plot_{len(plotted_tracks)}tracks"
            
        out_filename = f"{base_name}_tracks.{ext}"
        full_save_path = os.path.join(save_dir, out_filename)
        
        plt.savefig(
            full_save_path, 
            transparent=transparent_background, 
            dpi=dpi, 
            format=ext,
            bbox_inches='tight'
        )
        
        # SVG post-processing (like in original function)
        if ext == "svg":
            try:
                with open(full_save_path, encoding="utf-8") as f:
                    svg_data = f.read()
                # Remove <clipPath> definitions, metadata, XML declaration, and DOCTYPE
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
            except Exception as e:
                print(f"Warning: SVG post-processing failed: {e}")
        
        print(f"Plot saved to: {full_save_path}")
    
    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Prepare return data
    result = {
        "plotted_tracks": plotted_tracks,
        "n_tracks_plotted": len(plotted_tracks),
        "filtered_data_shape": df_pandas.shape,
        "coordinate_range": {
            "x_min": df_pandas['x_um'].min(),
            "x_max": df_pandas['x_um'].max(),
            "y_min": df_pandas['y_um'].min(), 
            "y_max": df_pandas['y_um'].max()
        },
        "tracks_df": df_pandas  # Include filtered data like original function
    }
    
    # Add SVG data if requested and available
    if export_format.lower() == "svg" and return_svg and svg_data:
        result["svg_data"] = svg_data
    
    return result


def gallery_of_tracks(
    instant_df,
    color_by="simple_threshold",
    num_tracks=20,
    colormap="Dark2",
    custom_colors=None,
    figsize=(12, 12),
    transparent_background=True,
    show_annotations=False,
    annotation_color="white",
    text_size=10,
    export_format="svg",
    save_path=None,
    show_plot=True,
    order=None,
    track_length_frames=60,
    spacing_factor=1.2,
    line_width=1.5,
    grid_cols=None,
    dpi=200,
    subplot_size_um=None,
):
    """
    Create a gallery of tracks distributed in a grid layout with consistent scaling.
    
    This function creates a visual gallery of particle tracks, where each track is 
    displayed in its own subplot with consistent scaling across all tracks for 
    visual size comparison.
    
    Parameters
    ----------
    instant_df : pandas.DataFrame or polars.DataFrame
        Instant trajectory dataframe containing track data
    color_by : str, default "simple_threshold"
        Column name to use for categorical coloring of tracks
    num_tracks : int, default 20
        Number of tracks to display per category
    colormap : str, default "Dark2"
        Colormap for coloring categories. Options:
        - 'colorblind': Wong 2011 colorblind-friendly palette (10 colors, cycles if needed)
        - 'Dark2': Matplotlib Dark2 colormap
        - Any other matplotlib colormap name
        Ignored if custom_colors is provided.
    custom_colors : list of str or None, default None
        Custom list of colors (hex codes, named colors, or RGB tuples).
        If provided, overrides colormap parameter.
        Example: ['#0173B2', '#DE8F05', '#029E73'] or ['red', 'blue', 'green']
    figsize : tuple or None, default (12, 12)
        Figure size in inches (width, height). If None, automatically calculated based on grid size
    transparent_background : bool, default True
        Whether to use transparent background
    show_annotations : bool, default False
        Whether to show track annotations
    annotation_color : str, default "white"
        Color for annotations
    text_size : int, default 10
        Font size for text annotations
    export_format : str, default "svg"
        Export format ('svg' or 'png')
    save_path : str or None, default None
        Path to save the figure. If None, saves to current directory
    show_plot : bool, default True
        Whether to display the plot
    order : list or None, default None
        Order of categories to display
    track_length_frames : int, default 60
        Number of consecutive frames to include per track
    spacing_factor : float, default 1.2
        Factor to multiply the maximum track extent for spacing between tracks
    line_width : float, default 1.5
        Width of track lines
    grid_cols : int or None, default None
        Number of columns in the grid. If None, automatically calculated
    dpi : int, default 200
        Resolution for saved figure
    subplot_size_um : float or None, default None
        Manual subplot size in microns. If None, automatically calculated from largest track extent
        
    Returns
    -------
    dict
        Dictionary containing track information and figure details
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    import random
    from matplotlib import cm
    import seaborn as sns
    
    # Auto-detect dataframe type and convert to pandas for compatibility
    is_polars = hasattr(instant_df, 'schema')
    if is_polars:
        import polars as pl
        df = instant_df.to_pandas()
    else:
        df = instant_df.copy()
    
    # Ensure required columns exist
    required_cols = ['unique_id', 'x_um', 'y_um', 'frame']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if color_by not in df.columns:
        raise ValueError(f"Column '{color_by}' not found in dataframe")
    
    # Get categories for coloring
    categories = order if order else sorted(df[color_by].unique())
    print(f"Gallery categories in order: {categories}")
    
    # Set up color mapping
    if custom_colors is not None:
        # Use custom colors provided by user
        # Cycle through colors if more categories than colors
        colors = [custom_colors[i % len(custom_colors)] for i in range(len(categories))]
    elif colormap == "colorblind":
        # Use Wong 2011 colorblind-friendly palette (same as plot_pca_3d)
        colorblind_colors = [
            '#0173B2',  # Blue
            '#DE8F05',  # Orange  
            '#029E73',  # Green
            '#CC78BC',  # Purple
            '#CA9161',  # Brown
            '#FBAFE4',  # Pink
            '#949494',  # Gray
            '#ECE133',  # Yellow
            '#56B4E9',  # Sky blue
            '#D55E00'   # Vermillion
        ]
        # Cycle through colors if more categories than colors
        colors = [colorblind_colors[i % len(colorblind_colors)] for i in range(len(categories))]
    elif colormap == "Dark2":
        cmap = cm.get_cmap("Dark2", len(categories))
        colors = cmap(np.linspace(0, 1, len(categories)))
    else:
        colors = plt.get_cmap(colormap, len(categories)).colors
    
    category_color_map = {cat: colors[i] for i, cat in enumerate(categories)}
    
    # Collect track segments for each category
    all_track_segments = []
    track_info = []
    
    for category in categories:
        # Filter by category
        cat_df = df[df[color_by] == category]
        unique_ids = cat_df['unique_id'].unique()
        
        # Randomly select tracks for this category
        selected_ids = random.sample(
            list(unique_ids), 
            min(num_tracks, len(unique_ids))
        )
        
        for unique_id in selected_ids:
            # Get all data for this track
            track_data = cat_df[cat_df['unique_id'] == unique_id].sort_values('frame')
            
            # Take a consecutive segment of the specified length
            if len(track_data) >= track_length_frames:
                # Randomly choose a starting point that allows for full segment
                max_start = len(track_data) - track_length_frames
                start_idx = random.randint(0, max_start) if max_start > 0 else 0
                track_segment = track_data.iloc[start_idx:start_idx + track_length_frames]
            else:
                # Use all available data if track is shorter
                track_segment = track_data
            
            if len(track_segment) >= 10:  # Minimum track length for visualization
                all_track_segments.append(track_segment)
                track_info.append((category, unique_id, len(track_segment)))
    
    if not all_track_segments:
        raise ValueError("No valid track segments found")
    
    print(f"Collected {len(all_track_segments)} tracks total")
    
    # Calculate subplot size
    if subplot_size_um is not None:
        # Use manually specified subplot size
        subplot_size = subplot_size_um
        print(f"Using manual subplot size: {subplot_size:.2f} μm")
    else:
        # Calculate global scale based on largest track extent
        max_extent = 0
        for segment in all_track_segments:
            x_range = segment['x_um'].max() - segment['x_um'].min()
            y_range = segment['y_um'].max() - segment['y_um'].min()
            extent = max(x_range, y_range)
            max_extent = max(max_extent, extent)
        
        # Add spacing between tracks
        subplot_size = max_extent * spacing_factor
        print(f"Using calculated subplot size: {subplot_size:.2f} μm")
    
    # Calculate grid layout
    total_tracks = len(all_track_segments)
    if grid_cols is None:
        grid_cols = int(np.ceil(np.sqrt(total_tracks)))
    grid_rows = int(np.ceil(total_tracks / grid_cols))
    
    print(f"Grid layout: {grid_rows} rows × {grid_cols} columns")
    
    # Calculate dynamic figsize if None
    if figsize is None:
        # Base size per subplot in inches (adjust as needed)
        subplot_size_inches = 2.0
        figsize = (grid_cols * subplot_size_inches, grid_rows * subplot_size_inches)
        print(f"Using dynamic figsize: {figsize}")
    
    # Create figure
    figure_background = "none" if transparent_background else "white"
    axis_background = (0, 0, 0, 0) if transparent_background else "white"
    
    fig, axes = plt.subplots(
        grid_rows, grid_cols, 
        figsize=figsize,
        facecolor=figure_background
    )
    
    # Handle single subplot case
    if grid_rows == 1 and grid_cols == 1:
        axes = [axes]
    elif grid_rows == 1 or grid_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Remove spacing between subplots for seamless grid
    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    
    # Plot each track
    plotted_tracks = []
    for idx, (segment, (category, unique_id, segment_length)) in enumerate(zip(all_track_segments, track_info)):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Extract coordinates
        x_coords = segment['x_um'].values
        y_coords = segment['y_um'].values
        
        # Calculate centroid for centering
        x_center = x_coords.mean()
        y_center = y_coords.mean()
        
        # Plot the track
        color = category_color_map[category]
        ax.plot(x_coords, y_coords, color=color, linewidth=line_width, alpha=0.8)
        
        # Set consistent axis limits centered on track
        half_size = subplot_size / 2
        ax.set_xlim(x_center - half_size, x_center + half_size)
        ax.set_ylim(y_center - half_size, y_center + half_size)
        ax.set_aspect('equal')
        
        # Remove all axis elements
        ax.axis('off')
        ax.set_facecolor(axis_background)
        
        # Add category annotation if requested
        if show_annotations:
            ax.text(
                0.5, 0.95, f"{category}\n{unique_id}",
                transform=ax.transAxes,
                ha='center', va='top',
                fontsize=text_size,
                color=annotation_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7)
            )
        
        plotted_tracks.append({
            'category': category,
            'unique_id': unique_id,
            'segment_length': segment_length,
            'x_center': x_center,
            'y_center': y_center,
            'subplot_index': idx
        })
    
    # Hide any unused subplots
    for idx in range(len(all_track_segments), len(axes)):
        axes[idx].axis('off')
        axes[idx].set_facecolor(axis_background)
    
    # Add title with category information
    category_counts = {}
    for category, _, _ in track_info:
        category_counts[category] = category_counts.get(category, 0) + 1
    
    title_text = f"Gallery of Tracks (colored by {color_by})\n"
    title_text += " | ".join([f"{cat}: {count}" for cat, count in category_counts.items()])
    
    fig.suptitle(
        title_text,
        fontsize=text_size + 2,
        color=annotation_color if transparent_background else "black",
        y=0.98
    )
    
    # Save figure
    if save_path is None:
        save_path = f"gallery_of_tracks_{color_by}.{export_format}"
    
    plt.savefig(
        save_path,
        format=export_format,
        dpi=dpi,
        bbox_inches='tight',
        transparent=transparent_background,
        facecolor=figure_background
    )
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Return summary information
    return {
        'plotted_tracks': plotted_tracks,
        'category_counts': category_counts,
        'total_tracks': len(all_track_segments),
        'subplot_size_um': subplot_size,
        'grid_dimensions': (grid_rows, grid_cols),
        'save_path': save_path,
        'categories': categories
    }


def gallery_of_tracks_v4(
    instant_df,
    color_by="simple_threshold",
    num_tracks=20,
    colormap="Dark2",
    custom_colors=None,
    figsize=(12, 12),
    transparent_background=True,
    show_annotations=False,
    annotation="default",  # NEW: str | list[str] | callable | "default"
    annotation_color="white",
    text_size=10,
    export_format="svg",
    save_path=None,
    show_plot=True,
    order=None,
    track_length_frames=60,
    spacing_factor=1.2,
    line_width=1.5,
    grid_cols=None,
    dpi=200,
    subplot_size_um=None,
    show_scale_bar=False,
    scale_bar_length_um=None,
    scale_bar_color='white',
    scale_bar_linewidth=3,
):
    """
    Create a gallery of tracks distributed in a grid layout with consistent scaling.
    ...
    Parameters
    ----------
    ...
    show_annotations : bool, default False
        Whether to show track annotations
    annotation : str | list[str] | callable | "default", default "default"
        What to annotate inside each subplot.
        - If a format string, it will be formatted with variables:
          {category}, {unique_id}, {window_uid} (if present), {segment_length},
          {start_frame}, {end_frame}.
        - If a list/tuple of column names, their values (from the segment) are joined with ' | '.
        - If a callable, it must be: fn(segment_dataframe, meta_dict) -> str
        - "default" shows "{category}\n{unique_id}" (previous behavior).
    annotation_color : str, default "white"
        Color for annotations
    text_size : int, default 10
        Font size for annotations and scale bar
    subplot_size_um : float or None, default None
        Manual subplot size in microns. If None, automatically calculated from largest track extent
    show_scale_bar : bool, default False
        Whether to show a scale bar in the bottom-right subplot
    scale_bar_length_um : float or None, default None
        Length of scale bar in microns. If None, defaults to subplot_size_um
    scale_bar_color : str, default 'white'
        Color of the scale bar and label
    scale_bar_linewidth : float, default 3
        Line width of the scale bar
    
    Returns
    -------
    dict
        Dictionary with keys: 'plotted_tracks', 'category_counts', 'total_tracks',
        'subplot_size_um', 'grid_dimensions', 'save_path', 'categories'
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    from matplotlib import cm

    # Auto-detect dataframe type and convert to pandas for compatibility
    is_polars = hasattr(instant_df, 'schema')
    if is_polars:
        import polars as pl
        df = instant_df.to_pandas()
    else:
        df = instant_df.copy()

    # Ensure required columns exist
    required_cols = ['unique_id', 'x_um', 'y_um', 'frame']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if color_by not in df.columns:
        raise ValueError(f"Column '{color_by}' not found in dataframe")

    # Get categories for coloring
    categories = order if order else sorted(df[color_by].unique())
    print(f"Gallery categories in order: {categories}")

    # Set up color mapping
    if custom_colors is not None:
        colors = [custom_colors[i % len(custom_colors)] for i in range(len(categories))]
    elif colormap == "colorblind":
        colorblind_colors = [
            '#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161',
            '#FBAFE4', '#949494', '#ECE133', '#56B4E9', '#D55E00'
        ]
        colors = [colorblind_colors[i % len(colorblind_colors)] for i in range(len(categories))]
    elif colormap == "Dark2":
        cmap = cm.get_cmap("Dark2", len(categories))
        colors = cmap(np.linspace(0, 1, len(categories)))
    else:
        colors = plt.get_cmap(colormap, len(categories)).colors

    category_color_map = {cat: colors[i] for i, cat in enumerate(categories)}

    # Collect track segments for each category
    all_track_segments = []
    track_info = []

    for category in categories:
        cat_df = df[df[color_by] == category]
        unique_ids = cat_df['unique_id'].unique()

        selected_ids = random.sample(list(unique_ids), min(num_tracks, len(unique_ids)))

        for unique_id in selected_ids:
            track_data = cat_df[cat_df['unique_id'] == unique_id].sort_values('frame')

            if len(track_data) >= track_length_frames:
                max_start = len(track_data) - track_length_frames
                start_idx = random.randint(0, max_start) if max_start > 0 else 0
                track_segment = track_data.iloc[start_idx:start_idx + track_length_frames]
            else:
                track_segment = track_data

            if len(track_segment) >= 10:
                all_track_segments.append(track_segment)
                track_info.append((category, unique_id, len(track_segment)))

    if not all_track_segments:
        raise ValueError("No valid track segments found")

    print(f"Collected {len(all_track_segments)} tracks total")

    # Calculate subplot size
    if subplot_size_um is not None:
        subplot_size = subplot_size_um
        print(f"Using manual subplot size: {subplot_size:.2f} μm")
    else:
        max_extent = 0
        for segment in all_track_segments:
            x_range = segment['x_um'].max() - segment['x_um'].min()
            y_range = segment['y_um'].max() - segment['y_um'].min()
            extent = max(x_range, y_range)
            max_extent = max(max_extent, extent)
        subplot_size = max_extent * spacing_factor
        print(f"Using calculated subplot size: {subplot_size:.2f} μm")

    # Calculate grid layout
    total_tracks = len(all_track_segments)
    if grid_cols is None:
        grid_cols = int(np.ceil(np.sqrt(total_tracks)))
    grid_rows = int(np.ceil(total_tracks / grid_cols))
    print(f"Grid layout: {grid_rows} rows × {grid_cols} columns")

    # Dynamic figsize
    if figsize is None:
        subplot_size_inches = 2.0
        figsize = (grid_cols * subplot_size_inches, grid_rows * subplot_size_inches)
        print(f"Using dynamic figsize: {figsize}")

    # Create figure
    figure_background = "none" if transparent_background else "white"
    axis_background = (0, 0, 0, 0) if transparent_background else "white"

    fig, axes = plt.subplots(
        grid_rows, grid_cols,
        figsize=figsize,
        facecolor=figure_background
    )

    # Normalize axes to 1D list
    if grid_rows == 1 and grid_cols == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    fig.subplots_adjust(wspace=0.02, hspace=0.02)

    plotted_tracks = []

    # --- Annotation setup (NEW) ---
    def _make_annotation_text(segment, meta):
        """
        segment: pd.DataFrame for this plotted segment
        meta: dict with keys: category, unique_id, segment_length, start_frame, end_frame, window_uid (if present)
        """
        if not show_annotations:
            return None

        # Callable user function
        if callable(annotation):
            try:
                return str(annotation(segment, meta))
            except Exception as e:
                return f"(annotation error: {e})"

        # List/tuple of columns -> join their values from the segment (first non-null)
        if isinstance(annotation, (list, tuple)):
            vals = []
            for col in annotation:
                if col in segment.columns:
                    val = segment[col].iloc[0]
                else:
                    val = meta.get(col, None)
                vals.append(f"{col}={val}")
            return " | ".join(vals)

        # Format string or "default"
        if isinstance(annotation, str):
            if annotation == "default":
                return f"{meta['category']}\n{meta['unique_id']}"
            # Build context for formatting
            ctx = dict(meta)  # start with meta
            # Allow direct column access if user wrote {some_col}
            # Pull first element for each column name present
            for col in segment.columns:
                if col not in ctx:
                    try:
                        ctx[col] = segment[col].iloc[0]
                    except Exception:
                        pass
            try:
                return annotation.format(**ctx)
            except KeyError as e:
                return f"(missing {e.args[0]} in annotation)"
            except Exception as e:
                return f"(annotation error: {e})"

        # Fallback
        return f"{meta['category']}\n{meta['unique_id']}"

    # Plot each track
    for idx, (segment, (category, unique_id, segment_length)) in enumerate(zip(all_track_segments, track_info)):
        if idx >= len(axes):
            break

        ax = axes[idx]

        x_coords = segment['x_um'].values
        y_coords = segment['y_um'].values

        x_center = x_coords.mean()
        y_center = y_coords.mean()

        color = category_color_map[category]
        ax.plot(x_coords, y_coords, color=color, linewidth=line_width, alpha=0.8)

        half_size = subplot_size / 2
        ax.set_xlim(x_center - half_size, x_center + half_size)
        ax.set_ylim(y_center - half_size, y_center + half_size)
        ax.set_aspect('equal')

        ax.axis('off')
        ax.set_facecolor(axis_background)

        # Build meta for annotation formatting
        start_frame = int(segment['frame'].min())
        end_frame = int(segment['frame'].max())
        # window_uid = None
        # if 'window_uid' in segment.columns:
        #     try:
        #         window_uid = segment['window_uid'].iloc[0]
        #     except Exception:
        #         pass
        # --- NEW: support multiple window_uids if segment crosses windows ---
        window_uid = None
        window_uids = None
        if 'window_uid' in segment.columns:
            vals = segment['window_uid'].dropna().astype(str)
            if len(vals) > 0:
                window_uids = sorted(vals.unique())
                window_uid = ", ".join(window_uids)   # show all windows in this segment


        meta = {
            'category': category,
            'unique_id': unique_id,
            'segment_length': int(segment_length),
            'start_frame': start_frame,
            'end_frame': end_frame,
            'window_uid': window_uid,
            'window_uids': window_uids
        }

        ann_text = _make_annotation_text(segment, meta)
        if show_annotations and ann_text:
            ax.text(
                0.5, 0.95, ann_text,
                transform=ax.transAxes,
                ha='center', va='top',
                fontsize=text_size,
                color=annotation_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7)
            )

        plotted_tracks.append({
            'category': category,
            'unique_id': unique_id,
            'segment_length': segment_length,
            'x_center': x_center,
            'y_center': y_center,
            'subplot_index': idx,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'window_uid': window_uid
        })

    # Hide any unused subplots
    for idx in range(len(all_track_segments), len(axes)):
        axes[idx].axis('off')
        axes[idx].set_facecolor(axis_background)

    # Add scale bar to bottom-right plotted subplot
    if show_scale_bar and len(all_track_segments) > 0:
        # Default scale bar length to subplot size
        if scale_bar_length_um is None:
            scale_bar_length_um = subplot_size
        
        # Get the last plotted subplot (bottom-right with content)
        last_plot_idx = len(all_track_segments) - 1
        scale_ax = axes[last_plot_idx]
        
        # Get the axis limits
        xlim = scale_ax.get_xlim()
        ylim = scale_ax.get_ylim()
        
        # Position scale bar at bottom-right (with padding)
        padding_fraction = 0.1  # 10% padding from edges
        x_padding = (xlim[1] - xlim[0]) * padding_fraction
        y_padding = (ylim[1] - ylim[0]) * padding_fraction
        
        # Scale bar endpoints
        scale_bar_x_end = xlim[1] - x_padding
        scale_bar_x_start = scale_bar_x_end - scale_bar_length_um
        scale_bar_y = ylim[0] + y_padding
        
        # Draw scale bar line
        scale_ax.plot(
            [scale_bar_x_start, scale_bar_x_end],
            [scale_bar_y, scale_bar_y],
            color=scale_bar_color,
            linewidth=scale_bar_linewidth,
            solid_capstyle='butt'
        )
        
        # Add end caps for scale bar
        cap_height = (ylim[1] - ylim[0]) * 0.02
        scale_ax.plot(
            [scale_bar_x_start, scale_bar_x_start],
            [scale_bar_y - cap_height, scale_bar_y + cap_height],
            color=scale_bar_color,
            linewidth=scale_bar_linewidth * 0.7
        )
        scale_ax.plot(
            [scale_bar_x_end, scale_bar_x_end],
            [scale_bar_y - cap_height, scale_bar_y + cap_height],
            color=scale_bar_color,
            linewidth=scale_bar_linewidth * 0.7
        )
        
        # Add scale bar label
        scale_ax.text(
            (scale_bar_x_start + scale_bar_x_end) / 2,
            scale_bar_y + y_padding * 0.5,
            f'{scale_bar_length_um:.1f} μm',
            ha='center',
            va='bottom',
            color=scale_bar_color,
            fontsize=text_size,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7)
        )

    # Title
    category_counts = {}
    for category, _, _ in track_info:
        category_counts[category] = category_counts.get(category, 0) + 1

    title_text = f"Gallery of Tracks (colored by {color_by})\n"
    title_text += " | ".join([f"{cat}: {count}" for cat, count in category_counts.items()])

    fig.suptitle(
        title_text,
        fontsize=text_size + 2,
        color=annotation_color if transparent_background else "black",
        y=0.98
    )

    # Save
    if save_path is None:
        save_path = f"gallery_of_tracks_{color_by}.{export_format}"

    plt.savefig(
        save_path,
        format=export_format,
        dpi=dpi,
        bbox_inches='tight',
        transparent=transparent_background,
        facecolor=figure_background
    )

    if show_plot:
        plt.show()
    else:
        plt.close()

    return {
        'plotted_tracks': plotted_tracks,
        'category_counts': category_counts,
        'total_tracks': len(all_track_segments),
        'subplot_size_um': subplot_size,
        'grid_dimensions': (grid_rows, grid_cols),
        'save_path': save_path,
        'categories': categories
    }


def gallery_of_tracks_v5(
    instant_df,
    order_by="superwindow_id",
    segment_color_by="final_population",
    num_tracks=20,
    segment_colors=None,
    segment_colormap="colorblind",
    figsize=(12, 12),
    transparent_background=True,
    show_annotations=False,
    annotation="default",
    annotation_color="white",  # Deprecated: use text_color instead
    text_color=None,  # NEW: Single variable for all text/annotation colors
    text_size=10,
    export_format="svg",
    save_path=None,
    show_plot=True,
    order=None,
    track_length_frames=60,
    spacing_factor=1.2,
    line_width=1.5,
    grid_cols=None,
    dpi=200,
    subplot_size_um=None,
    show_scale_bar=False,
    scale_bar_length_um=None,
    scale_bar_color='white',
    scale_bar_linewidth=3,
    # DEPRECATED: use show_subplot_border instead
    draw_box=False,
    box_colormap="tab10",
    box_colors=None,
    box_linewidth=3,
    # Subplot border (bounding box around each subplot)
    show_subplot_border=False,
    subplot_border_color=None,  # Solid color for all borders (if not using colormap)
    subplot_border_linewidth=2,
    subplot_border_color_by=None,  # 'cluster' to color by cluster, None for solid color
    subplot_border_colormap='colorblind',  # Colormap for borders when color_by is set
    subplot_border_colors=None,  # Custom color dict for borders {cluster_id: color}
    # NEW: Legend for segment colors
    show_legend=True,
    legend_title=None,  # Defaults to segment_color_by column name
    legend_loc='right',  # 'right', 'bottom', or specific matplotlib location
    legend_order=None,  # Custom order for legend items (e.g., state_order from plot_representative_sequences)
    # NEW: Structured grid layout (matching plot_representative_sequences)
    group_by=None,
    cluster_col=None,
    n_per_cluster=None,
    cluster_order=None,
    group_order=None,
    random_seed=42,
    # NEW: Use exact track order from plot_representative_sequences
    superwindow_ids=None,  # List of superwindow_ids in EXACT plotting order from plot_representative_sequences
    superwindow_id_col='superwindow_id',  # Column name for superwindow_id in the dataframe
):
    """
    Create a gallery of tracks with WITHIN-TRACK coloring by a segment attribute.
    
    This version allows tracks to be colored segment-by-segment based on a column
    like 'final_population', so you can see state transitions directly on the XY plot.
    
    Parameters
    ----------
    instant_df : pd.DataFrame or pl.DataFrame
        DataFrame with frame-level data. Must have columns: unique_id, x_um, y_um, frame
    order_by : str, default='superwindow_id'
        Column to use for ORDERING/GROUPING tracks in the gallery (e.g., 'superwindow_id', 
        'superwindow_cluster'). Tracks will be collected per unique value of this column.
    segment_color_by : str, default='final_population'
        Column to use for coloring WITHIN each track. Each unique value gets a different color.
    num_tracks : int, default=20
        Maximum number of tracks to show per order_by category
    segment_colors : dict, optional
        Custom color mapping for segment_color_by values. {value: color}
    segment_colormap : str, default='colorblind'
        Colormap to use for segment colors if segment_colors is not provided
    figsize : tuple, default=(12, 12)
        Figure size
    transparent_background : bool, default=True
        Transparent background
    show_annotations : bool, default=False
        Whether to show track annotations (displayed BELOW each subplot)
    annotation : str | list[str] | callable | "default", default="default"
        Annotation format. Use "{superwindow_id}" to show superwindow IDs.
    annotation_color : str, default="white"
        DEPRECATED: Use text_color instead. Color for annotations.
    text_color : str, optional
        Single variable to control ALL text colors (annotations, labels, legend, title).
        If None, defaults to annotation_color for backwards compatibility.
    text_size : int, default=10
        Font size for annotations
    export_format : str, default="svg"
        Export format
    save_path : str, optional
        Save path for the figure
    show_plot : bool, default=True
        Whether to display the plot
    order : list, optional
        Custom order for order_by values (e.g., list of superwindow_ids in desired order)
    track_length_frames : int, default=60
        Maximum number of frames to show per track
    spacing_factor : float, default=1.2
        Spacing factor for subplot sizing
    line_width : float, default=1.5
        Line width for track plots
    grid_cols : int, optional
        Number of columns in the grid. Auto-calculated if None.
    dpi : int, default=200
        DPI for saving
    subplot_size_um : float, optional
        Manual subplot size in microns
    show_scale_bar : bool, default=False
        Whether to show a scale bar
    scale_bar_length_um : float, optional
        Length of scale bar in microns
    scale_bar_color : str, default='white'
        Color of the scale bar
    scale_bar_linewidth : float, default=3
        Line width of the scale bar
    draw_box : bool, default=False
        Whether to draw a colored box/border around each subplot based on order_by value
    box_colormap : str, default='tab10'
        Colormap to use for box colors (e.g., 'tab10', 'colorblind', 'viridis')
    box_colors : dict, optional
        Custom color mapping for box colors. {order_by_value: color}
    box_linewidth : float, default=3
        Line width for the box border
    show_subplot_border : bool, default=False
        Whether to draw a bounding box/border around each subplot.
    subplot_border_color : str, optional
        Solid color for ALL subplot borders (used when subplot_border_color_by=None).
        Defaults to text_color if not specified.
    subplot_border_linewidth : float, default=2
        Line width for subplot borders.
    subplot_border_color_by : str, optional
        What to color the borders by. Options:
        - None: Use solid color (subplot_border_color)
        - 'cluster': Color borders by cluster (requires cluster_col to be set)
    subplot_border_colormap : str, default='colorblind'
        Colormap to use for border colors when subplot_border_color_by is set.
        Options: 'colorblind', 'tab10', 'viridis', etc.
    subplot_border_colors : dict, optional
        Custom color mapping for borders when using subplot_border_color_by.
        Example: {0: '#FF0000', 1: '#00FF00', 2: '#0000FF'}
    show_legend : bool, default=True
        Whether to show a legend mapping segment colors to their values.
        Legend appears OUTSIDE the subplot grid.
    legend_title : str, optional
        Title for the legend. Defaults to segment_color_by column name.
    legend_loc : str, default='right'
        Location for the legend. Options: 'right', 'bottom', or matplotlib location strings.
    legend_order : list, optional
        Custom order for legend items. If None, uses the order of keys in segment_colors dict
        (if provided), otherwise uses sorted order. Use this to match state_order from
        plot_representative_sequences.
    group_by : str, optional
        Column to use for ROW grouping in structured grid layout (e.g., 'mol').
        When specified along with cluster_col and n_per_cluster, creates a grid
        matching plot_representative_sequences layout:
        - Rows = unique values from group_by column
        - Columns = n_clusters * n_per_cluster
    cluster_col : str, optional
        Column to use for COLUMN grouping in structured grid layout (e.g., 'superwindow_cluster').
        Clusters are arranged left-to-right, with n_per_cluster tracks per cluster per row.
    n_per_cluster : int, optional
        Number of tracks to show per cluster per group (row).
        Similar to n_superwindows_per_cluster in plot_representative_sequences.
    cluster_order : list, optional
        Custom order for clusters. If None, uses sorted unique values.
    group_order : list, optional
        Custom order for groups (rows). If None, uses sorted unique values.
    random_seed : int, default=42
        Random seed for reproducible track selection.
    superwindow_ids : list, optional
        List of superwindow_ids in EXACT plotting order from plot_representative_sequences.
        When provided, skips random sampling and uses these IDs directly to match
        the exact order from plot_representative_sequences. Pass the `plotted_sw_ids`
        output from plot_representative_sequences.
    superwindow_id_col : str, default='superwindow_id'
        Column name for superwindow_id in the dataframe.
        
    Returns
    -------
    dict
        Dictionary with keys: 'plotted_tracks', 'category_counts', 'total_tracks',
        'subplot_size_um', 'grid_dimensions', 'save_path', 'order_by_values', 
        'segment_color_map', 'box_color_map'
        When using structured layout, also includes: 'groups', 'clusters', 'n_per_cluster'
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    from matplotlib import cm

    # Auto-detect dataframe type and convert to pandas for compatibility
    is_polars = hasattr(instant_df, 'schema')
    if is_polars:
        import polars as pl
        df = instant_df.to_pandas()
    else:
        df = instant_df.copy()

    # Ensure required columns exist
    required_cols = ['unique_id', 'x_um', 'y_um', 'frame']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if order_by not in df.columns:
        raise ValueError(f"Order column '{order_by}' not found in dataframe")
    
    if segment_color_by not in df.columns:
        raise ValueError(f"Segment color column '{segment_color_by}' not found in dataframe")

    # Handle text_color (single variable for all text colors)
    # If text_color is not specified, fall back to annotation_color for backwards compatibility
    if text_color is None:
        text_color = annotation_color
    
    # Set subplot_border_color to text_color if not specified
    if subplot_border_color is None:
        subplot_border_color = text_color

    # Set up border color mapping (for subplot borders colored by cluster)
    border_color_map = {}

    # Get order_by categories
    order_by_values = order if order else sorted(df[order_by].dropna().unique())
    print(f"Order by '{order_by}': {len(order_by_values)} unique values")

    # Set up box color mapping (for order_by values)
    box_color_map = {}
    if draw_box:
        if box_colors is not None:
            box_color_map = box_colors
        elif box_colormap == "colorblind":
            colorblind_colors = [
                '#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161',
                '#FBAFE4', '#949494', '#ECE133', '#56B4E9', '#D55E00'
            ]
            box_color_map = {val: colorblind_colors[i % len(colorblind_colors)] 
                            for i, val in enumerate(order_by_values)}
        else:
            cmap = plt.get_cmap(box_colormap, len(order_by_values))
            box_color_map = {val: cmap(i / max(1, len(order_by_values) - 1)) 
                            for i, val in enumerate(order_by_values)}
        print(f"Box colors for {len(order_by_values)} order_by values")

    # Get all unique segment colors
    all_segment_values = sorted(df[segment_color_by].dropna().unique())
    print(f"Segment color by '{segment_color_by}': {all_segment_values}")

    # Set up segment color mapping
    if segment_colors is not None:
        segment_color_map = segment_colors
    elif segment_colormap == "colorblind":
        colorblind_colors = [
            '#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161',
            '#FBAFE4', '#949494', '#ECE133', '#56B4E9', '#D55E00'
        ]
        segment_color_map = {val: colorblind_colors[i % len(colorblind_colors)] 
                            for i, val in enumerate(all_segment_values)}
    else:
        cmap = plt.get_cmap(segment_colormap, len(all_segment_values))
        segment_color_map = {val: cmap(i / len(all_segment_values)) 
                            for i, val in enumerate(all_segment_values)}

    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Check if using structured grid layout (matching plot_representative_sequences)
    use_structured_layout = (group_by is not None and cluster_col is not None and n_per_cluster is not None)
    
    all_track_segments = []
    track_info = []
    grid_position_map = {}  # Maps (row_idx, col_idx) to track info for structured layout
    
    if use_structured_layout:
        # ========== STRUCTURED GRID LAYOUT MODE ==========
        # Rows = groups (e.g., mol values)
        # Columns = clusters * n_per_cluster (same as plot_representative_sequences)
        
        if group_by not in df.columns:
            raise ValueError(f"group_by column '{group_by}' not found in dataframe")
        if cluster_col not in df.columns:
            raise ValueError(f"cluster_col column '{cluster_col}' not found in dataframe")
        
        # Get groups and clusters
        groups = group_order if group_order else sorted(df[group_by].dropna().unique())
        clusters = cluster_order if cluster_order else sorted(df[cluster_col].dropna().unique())
        n_clusters = len(clusters)
        
        print(f"\n=== STRUCTURED GRID LAYOUT (matching plot_representative_sequences) ===")
        print(f"Groups (rows): {groups}")
        print(f"Clusters (cols): {clusters}")
        print(f"Tracks per cluster per group: {n_per_cluster}")
        
        # Calculate grid dimensions: rows = groups, cols = clusters * n_per_cluster
        grid_rows = len(groups)
        grid_cols = n_clusters * n_per_cluster
        
        print(f"Grid layout: {grid_rows} rows × {grid_cols} columns")
        
        # Check if using exact superwindow_ids from plot_representative_sequences
        use_exact_order = superwindow_ids is not None and len(superwindow_ids) > 0
        
        if use_exact_order:
            print(f"✅ Using EXACT order from plot_representative_sequences ({len(superwindow_ids)} superwindow_ids)")
            # superwindow_ids is already in the correct order: 
            # [group0_cluster0_ex0, group0_cluster0_ex1, ..., group0_cluster1_ex0, ...]
            sw_idx = 0  # Index into superwindow_ids list
        
        # Collect tracks in structured order (matching plot_representative_sequences)
        for group_idx, group_val in enumerate(groups):
            group_data = df[df[group_by] == group_val]
            
            col_idx = 0
            for cluster_id in clusters:
                cluster_data = group_data[group_data[cluster_col] == cluster_id]
                
                # Collect track segments for this cluster
                for ex_idx in range(n_per_cluster):
                    if use_exact_order:
                        # Use EXACT superwindow_id from plot_representative_sequences
                        if sw_idx < len(superwindow_ids):
                            target_sw_id = superwindow_ids[sw_idx]
                            sw_idx += 1
                            
                            # Find track data for this superwindow_id
                            if superwindow_id_col in df.columns:
                                track_data = df[df[superwindow_id_col] == target_sw_id].sort_values('frame')
                            else:
                                # Fall back to unique_id if superwindow_id_col not found
                                track_data = df[df['unique_id'] == target_sw_id].sort_values('frame')
                            
                            if len(track_data) == 0:
                                all_track_segments.append(None)
                                track_info.append((None, None, 0))
                                grid_position_map[(group_idx, col_idx)] = len(all_track_segments) - 1
                                col_idx += 1
                                continue
                            
                            # Get cluster_id from the track data for border coloring
                            if cluster_col in track_data.columns:
                                actual_cluster = track_data[cluster_col].iloc[0]
                            else:
                                actual_cluster = cluster_id
                            
                            # When using superwindow_ids, use ALL frames for that superwindow
                            # (NO random sampling - this ensures exact alignment with sequences!)
                            track_segment = track_data  # Use ALL frames belonging to this superwindow_id
                            
                            if len(track_segment) >= 10:
                                all_track_segments.append(track_segment)
                                track_info.append((actual_cluster, target_sw_id, len(track_segment)))
                            else:
                                all_track_segments.append(None)
                                track_info.append((None, None, 0))
                        else:
                            # No more superwindow_ids, fill with empty
                            all_track_segments.append(None)
                            track_info.append((None, None, 0))
                        
                        grid_position_map[(group_idx, col_idx)] = len(all_track_segments) - 1
                        col_idx += 1
                    
                    else:
                        # SAMPLING MODE (original behavior when superwindow_ids not provided)
                        if ex_idx == 0:
                            # Only do this once per cluster
                            if len(cluster_data) == 0:
                                # Fill all slots for this cluster with empty
                                for _ in range(n_per_cluster):
                                    all_track_segments.append(None)
                                    track_info.append((None, None, 0))
                                    grid_position_map[(group_idx, col_idx)] = len(all_track_segments) - 1
                                    col_idx += 1
                                break  # Skip remaining ex_idx iterations
                            
                            # Sample tracks
                            unique_ids = cluster_data['unique_id'].unique()
                            n_available = len(unique_ids)
                            n_to_sample = min(n_per_cluster, n_available)
                            
                            if n_to_sample > 0:
                                unique_ids_df = pd.DataFrame({'unique_id': unique_ids})
                                sampled_df = unique_ids_df.sample(n=n_to_sample, random_state=42)
                                selected_ids = sampled_df['unique_id'].tolist()
                            else:
                                selected_ids = []
                        
                        if ex_idx < len(selected_ids):
                            unique_id = selected_ids[ex_idx]
                            track_data = cluster_data[cluster_data['unique_id'] == unique_id].sort_values('frame')
                            
                            if len(track_data) >= track_length_frames:
                                max_start = len(track_data) - track_length_frames
                                start_idx = random.randint(0, max_start) if max_start > 0 else 0
                                track_segment = track_data.iloc[start_idx:start_idx + track_length_frames]
                            else:
                                track_segment = track_data
                            
                            if len(track_segment) >= 10:
                                all_track_segments.append(track_segment)
                                track_info.append((cluster_id, unique_id, len(track_segment)))
                            else:
                                all_track_segments.append(None)
                                track_info.append((None, None, 0))
                        else:
                            # Not enough tracks for this slot
                            all_track_segments.append(None)
                            track_info.append((None, None, 0))
                        
                        grid_position_map[(group_idx, col_idx)] = len(all_track_segments) - 1
                        col_idx += 1
        
        # Update box_color_map to use cluster values (DEPRECATED - use border_color_map)
        if draw_box:
            if box_colors is not None:
                box_color_map = box_colors
            elif box_colormap == "colorblind":
                colorblind_colors = [
                    '#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161',
                    '#FBAFE4', '#949494', '#ECE133', '#56B4E9', '#D55E00'
                ]
                box_color_map = {val: colorblind_colors[i % len(colorblind_colors)] 
                                for i, val in enumerate(clusters)}
            else:
                cmap = plt.get_cmap(box_colormap, len(clusters))
                box_color_map = {val: cmap(i / max(1, len(clusters) - 1)) 
                                for i, val in enumerate(clusters)}
            print(f"Box colors based on clusters: {list(box_color_map.keys())}")
        
        # Set up border_color_map for subplot borders colored by cluster
        if show_subplot_border and subplot_border_color_by == 'cluster':
            if subplot_border_colors is not None:
                border_color_map = subplot_border_colors
            elif subplot_border_colormap == "colorblind":
                colorblind_colors = [
                    '#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161',
                    '#FBAFE4', '#949494', '#ECE133', '#56B4E9', '#D55E00'
                ]
                border_color_map = {val: colorblind_colors[i % len(colorblind_colors)] 
                                   for i, val in enumerate(clusters)}
            else:
                cmap = plt.get_cmap(subplot_border_colormap, len(clusters))
                border_color_map = {val: cmap(i / max(1, len(clusters) - 1)) 
                                   for i, val in enumerate(clusters)}
            print(f"Border colors based on clusters: {list(border_color_map.keys())}")
        
        valid_tracks = sum(1 for t in all_track_segments if t is not None)
        print(f"Collected {valid_tracks} valid tracks (out of {len(all_track_segments)} grid slots)")
        
    else:
        # ========== ORIGINAL FLAT LAYOUT MODE ==========
        for order_val in order_by_values:
            # Filter to this order_by value (e.g., superwindow_id)
            order_df = df[df[order_by] == order_val]
            
            if len(order_df) == 0:
                continue
            
            # Get unique tracks in this order_by value
            unique_ids = order_df['unique_id'].unique()
            
            # Sample if needed
            selected_ids = random.sample(list(unique_ids), min(num_tracks, len(unique_ids)))

            for unique_id in selected_ids:
                track_data = order_df[order_df['unique_id'] == unique_id].sort_values('frame')

                if len(track_data) >= track_length_frames:
                    max_start = len(track_data) - track_length_frames
                    start_idx = random.randint(0, max_start) if max_start > 0 else 0
                    track_segment = track_data.iloc[start_idx:start_idx + track_length_frames]
                else:
                    track_segment = track_data

                if len(track_segment) >= 10:
                    all_track_segments.append(track_segment)
                    track_info.append((order_val, unique_id, len(track_segment)))

        if not all_track_segments:
            raise ValueError("No valid track segments found")

        print(f"Collected {len(all_track_segments)} tracks total")

        # Calculate grid layout for flat mode
        total_tracks = len(all_track_segments)
        if grid_cols is None:
            grid_cols = int(np.ceil(np.sqrt(total_tracks)))
        grid_rows = int(np.ceil(total_tracks / grid_cols))
        print(f"Grid layout: {grid_rows} rows × {grid_cols} columns")

    # Calculate subplot size
    valid_segments = [s for s in all_track_segments if s is not None]
    if not valid_segments:
        raise ValueError("No valid track segments found")
    
    if subplot_size_um is not None:
        subplot_size = subplot_size_um
        print(f"Using manual subplot size: {subplot_size:.2f} μm")
    else:
        max_extent = 0
        for segment in valid_segments:
            x_range = segment['x_um'].max() - segment['x_um'].min()
            y_range = segment['y_um'].max() - segment['y_um'].min()
            extent = max(x_range, y_range)
            max_extent = max(max_extent, extent)
        subplot_size = max_extent * spacing_factor
        print(f"Using calculated subplot size: {subplot_size:.2f} μm")

    # Dynamic figsize
    if figsize is None:
        subplot_size_inches = 2.0
        figsize = (grid_cols * subplot_size_inches, grid_rows * subplot_size_inches)
        print(f"Using dynamic figsize: {figsize}")

    # Create figure
    figure_background = "none" if transparent_background else "white"
    axis_background = (0, 0, 0, 0) if transparent_background else "white"

    fig, axes = plt.subplots(
        grid_rows, grid_cols,
        figsize=figsize,
        facecolor=figure_background,
        squeeze=False  # Always return 2D array for structured layout
    )
    
    # Keep 2D axes reference for structured layout
    axes_2d = axes
    
    # Also create 1D axes list for iteration
    axes_flat = np.array(axes).flatten()

    fig.subplots_adjust(wspace=0.02, hspace=0.02)

    plotted_tracks = []

    # --- Annotation setup ---
    def _make_annotation_text(segment, meta):
        if not show_annotations:
            return None

        if callable(annotation):
            try:
                return str(annotation(segment, meta))
            except Exception as e:
                return f"(annotation error: {e})"

        if isinstance(annotation, (list, tuple)):
            vals = []
            for col in annotation:
                if col in segment.columns:
                    val = segment[col].iloc[0]
                else:
                    val = meta.get(col, None)
                vals.append(f"{col}={val}")
            return " | ".join(vals)

        if isinstance(annotation, str):
            if annotation == "default":
                return f"{meta['order_val']}\n{meta['unique_id']}"
            ctx = dict(meta)
            for col in segment.columns:
                if col not in ctx:
                    try:
                        ctx[col] = segment[col].iloc[0]
                    except Exception:
                        pass
            try:
                return annotation.format(**ctx)
            except KeyError as e:
                return f"(missing {e.args[0]} in annotation)"
            except Exception as e:
                return f"(annotation error: {e})"

        return f"{meta['order_val']}\n{meta['unique_id']}"

    # Plot each track with WITHIN-TRACK segment coloring
    for idx, (segment, (order_val, unique_id, segment_length)) in enumerate(zip(all_track_segments, track_info)):
        if idx >= len(axes_flat):
            break

        ax = axes_flat[idx]
        
        # Handle empty/None segments (structured layout may have empty slots)
        if segment is None or segment_length == 0:
            ax.axis('off')
            ax.set_facecolor(axis_background)
            continue

        x_coords = segment['x_um'].values
        y_coords = segment['y_um'].values
        segment_values = segment[segment_color_by].values

        x_center = x_coords.mean()
        y_center = y_coords.mean()

        # Plot segment-by-segment with different colors
        # Group consecutive frames by their segment_color_by value
        i = 0
        while i < len(x_coords) - 1:
            current_val = segment_values[i]
            color = segment_color_map.get(current_val, 'gray')
            
            # Find the end of this contiguous segment
            j = i + 1
            while j < len(segment_values) and segment_values[j] == current_val:
                j += 1
            
            # Plot this segment (include one point overlap for continuity)
            end_idx = min(j + 1, len(x_coords))
            ax.plot(x_coords[i:end_idx], y_coords[i:end_idx], 
                   color=color, linewidth=line_width, alpha=0.8)
            
            i = j

        half_size = subplot_size / 2
        ax.set_xlim(x_center - half_size, x_center + half_size)
        ax.set_ylim(y_center - half_size, y_center + half_size)
        ax.set_aspect('equal')

        # Hide ticks and labels but KEEP SPINES AVAILABLE for borders
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_facecolor(axis_background)

        # Configure subplot borders
        if show_subplot_border:
            # Determine border color for this subplot
            if subplot_border_color_by == 'cluster' and order_val is not None:
                # Color by cluster - order_val contains the cluster ID in structured mode
                border_color = border_color_map.get(order_val, subplot_border_color or text_color)
            else:
                # Solid color for all borders
                border_color = subplot_border_color or text_color
            
            # Make spines visible with the determined color
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(border_color)
                spine.set_linewidth(subplot_border_linewidth)
        else:
            # Hide all spines when borders are disabled
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        # Legacy draw_box support (DEPRECATED - use show_subplot_border with subplot_border_color_by='cluster')
        if draw_box and not show_subplot_border and order_val in box_color_map:
            box_color = box_color_map[order_val]
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(box_color)
                spine.set_linewidth(box_linewidth)

        # Build meta for annotation
        start_frame = int(segment['frame'].min())
        end_frame = int(segment['frame'].max())
        
        window_uid = None
        window_uids = None
        if 'window_uid' in segment.columns:
            vals = segment['window_uid'].dropna().astype(str)
            if len(vals) > 0:
                window_uids = sorted(vals.unique())
                window_uid = ", ".join(window_uids)
        
        # Get superwindow_id if available
        superwindow_id = None
        if 'superwindow_id' in segment.columns:
            superwindow_id = segment['superwindow_id'].iloc[0]

        meta = {
            'order_val': order_val,
            'unique_id': unique_id,
            'segment_length': int(segment_length),
            'start_frame': start_frame,
            'end_frame': end_frame,
            'window_uid': window_uid,
            'window_uids': window_uids,
            'superwindow_id': superwindow_id
        }

        ann_text = _make_annotation_text(segment, meta)
        if show_annotations and ann_text:
            # Place annotation BELOW the subplot (like plot_representative_sequences)
            ax.text(
                0.5, -0.12, ann_text,
                transform=ax.transAxes,
                ha='center', va='top',
                fontsize=text_size,
                color=text_color,
                style='italic'  # Match plot_representative_sequences style
            )

        plotted_tracks.append({
            'order_val': order_val,
            'unique_id': unique_id,
            'segment_length': segment_length,
            'x_center': x_center,
            'y_center': y_center,
            'subplot_index': idx,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'window_uid': window_uid
        })

    # Hide any unused subplots
    for idx in range(len(all_track_segments), len(axes_flat)):
        axes_flat[idx].axis('off')
        axes_flat[idx].set_facecolor(axis_background)

    # Add row labels and column headers for STRUCTURED LAYOUT
    if use_structured_layout:
        groups = group_order if group_order else sorted(df[group_by].dropna().unique())
        clusters = cluster_order if cluster_order else sorted(df[cluster_col].dropna().unique())
        n_clusters = len(clusters)
        
        # Add column headers (cluster labels) - on first row only
        for cluster_idx, cluster_id in enumerate(clusters):
            # Position at the center of each cluster's columns
            center_col = cluster_idx * n_per_cluster + n_per_cluster // 2
            ax = axes_2d[0, center_col]
            ax.set_title(f'Cluster {cluster_id}', fontsize=text_size + 2, fontweight='bold',
                        color=text_color, pad=5)
        
        # Add row labels (group names) - on first column only
        for group_idx, group_val in enumerate(groups):
            ax = axes_2d[group_idx, 0]
            ax.set_ylabel(f'{group_val}', fontsize=text_size + 2, fontweight='bold',
                         color=text_color, rotation=90, labelpad=10)

    # Add scale bar to bottom-right plotted subplot
    valid_indices = [i for i, s in enumerate(all_track_segments) if s is not None]
    if show_scale_bar and len(valid_indices) > 0:
        if scale_bar_length_um is None:
            scale_bar_length_um = subplot_size
        
        last_plot_idx = valid_indices[-1]
        scale_ax = axes_flat[last_plot_idx]
        
        xlim = scale_ax.get_xlim()
        ylim = scale_ax.get_ylim()
        
        padding_fraction = 0.1
        x_padding = (xlim[1] - xlim[0]) * padding_fraction
        y_padding = (ylim[1] - ylim[0]) * padding_fraction
        
        scale_bar_x_end = xlim[1] - x_padding
        scale_bar_x_start = scale_bar_x_end - scale_bar_length_um
        scale_bar_y = ylim[0] + y_padding
        
        scale_ax.plot(
            [scale_bar_x_start, scale_bar_x_end],
            [scale_bar_y, scale_bar_y],
            color=scale_bar_color,
            linewidth=scale_bar_linewidth,
            solid_capstyle='butt'
        )
        
        cap_height = (ylim[1] - ylim[0]) * 0.02
        scale_ax.plot(
            [scale_bar_x_start, scale_bar_x_start],
            [scale_bar_y - cap_height, scale_bar_y + cap_height],
            color=scale_bar_color,
            linewidth=scale_bar_linewidth * 0.7
        )
        scale_ax.plot(
            [scale_bar_x_end, scale_bar_x_end],
            [scale_bar_y - cap_height, scale_bar_y + cap_height],
            color=scale_bar_color,
            linewidth=scale_bar_linewidth * 0.7
        )
        
        scale_ax.text(
            (scale_bar_x_start + scale_bar_x_end) / 2,
            scale_bar_y + y_padding * 0.5,
            f'{scale_bar_length_um:.1f} μm',
            ha='center',
            va='bottom',
            color=scale_bar_color,
            fontsize=text_size,
            fontweight='bold'
        )

    # Count tracks per order_by value (or cluster for structured layout)
    category_counts = {}
    for order_val, _, _ in track_info:
        if order_val is not None:
            category_counts[order_val] = category_counts.get(order_val, 0) + 1

    # Title
    valid_count = sum(1 for s in all_track_segments if s is not None)
    if use_structured_layout:
        groups = group_order if group_order else sorted(df[group_by].dropna().unique())
        clusters = cluster_order if cluster_order else sorted(df[cluster_col].dropna().unique())
        title_text = f"Gallery of Tracks (rows={group_by}, cols={cluster_col}, colored by {segment_color_by})\n"
        title_text += f"{valid_count} tracks ({len(groups)} groups × {len(clusters)} clusters × {n_per_cluster} per)"
    else:
        title_text = f"Gallery of Tracks (ordered by {order_by}, colored by {segment_color_by})\n"
        title_text += f"{valid_count} tracks"

    fig.suptitle(
        title_text,
        fontsize=text_size + 2,
        color=text_color,
        y=0.98
    )

    # Add legend for segment colors (OUTSIDE the subplot grid)
    if show_legend and segment_color_map:
        from matplotlib.patches import Patch
        
        # Determine legend order:
        # 1. Use legend_order if provided explicitly
        # 2. Otherwise, use the order of keys in segment_colors dict (if user provided it)
        # 3. Otherwise, use sorted order
        if legend_order is not None:
            ordered_values = [v for v in legend_order if v in segment_color_map]
        elif segment_colors is not None:
            # Use the order from the user-provided segment_colors dict (preserves insertion order)
            ordered_values = list(segment_colors.keys())
        else:
            # Fall back to sorted order
            ordered_values = sorted(segment_color_map.keys(), key=lambda x: str(x))
        
        # Create legend handles in the determined order
        legend_handles = []
        for value in ordered_values:
            if value in segment_color_map:
                color = segment_color_map[value]
                patch = Patch(facecolor=color, edgecolor='none', label=str(value))
                legend_handles.append(patch)
        
        # Determine legend title
        leg_title = legend_title if legend_title else segment_color_by
        
        # Position legend outside the subplots
        if legend_loc == 'right':
            legend = fig.legend(
                handles=legend_handles,
                title=leg_title,
                loc='center left',
                bbox_to_anchor=(1.01, 0.5),
                frameon=False,
                fontsize=text_size,
                title_fontsize=text_size + 1
            )
        elif legend_loc == 'bottom':
            legend = fig.legend(
                handles=legend_handles,
                title=leg_title,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.02),
                ncol=len(legend_handles),
                frameon=False,
                fontsize=text_size,
                title_fontsize=text_size + 1
            )
        else:
            # Use matplotlib's built-in location string
            legend = fig.legend(
                handles=legend_handles,
                title=leg_title,
                loc=legend_loc,
                frameon=False,
                fontsize=text_size,
                title_fontsize=text_size + 1
            )
        
        # Set legend text color
        legend.get_title().set_color(text_color)
        for text in legend.get_texts():
            text.set_color(text_color)

    # Save
    if save_path is None:
        if use_structured_layout:
            save_path = f"gallery_of_tracks_v5_{group_by}_{cluster_col}_{segment_color_by}.{export_format}"
        else:
            save_path = f"gallery_of_tracks_v5_{order_by}_{segment_color_by}.{export_format}"

    plt.savefig(
        save_path,
        format=export_format,
        dpi=dpi,
        bbox_inches='tight',
        transparent=transparent_background,
        facecolor=figure_background
    )

    if show_plot:
        plt.show()
    else:
        plt.close()

    # Build return dictionary
    result = {
        'plotted_tracks': plotted_tracks,
        'category_counts': category_counts,
        'total_tracks': valid_count,
        'subplot_size_um': subplot_size,
        'grid_dimensions': (grid_rows, grid_cols),
        'save_path': save_path,
        'order_by_values': order_by_values,
        'segment_color_map': segment_color_map,
        'box_color_map': box_color_map
    }
    
    # Add structured layout info if applicable
    if use_structured_layout:
        groups = group_order if group_order else sorted(df[group_by].dropna().unique())
        clusters = cluster_order if cluster_order else sorted(df[cluster_col].dropna().unique())
        result['groups'] = groups
        result['clusters'] = clusters
        result['n_per_cluster'] = n_per_cluster
        result['group_by'] = group_by
        result['cluster_col'] = cluster_col
    
    return result


def plot_stacked_bar_percentage(
    df,
    x_category,
    stack_category,
    unique_id_col='unique_id',
    x_order=None,
    stack_order=None,
    palette='colorblind',
    figsize=(10, 6),
    title=None,
    xlabel=None,
    ylabel='Percentage (%)',
    show_percentages=True,
    percentage_threshold=5.0,
    font_size=12,
    transparent_background=False,
    line_color='black',
    grid=True,
    save_path=None,
    export_format='png',
    return_svg=False,
    show_plot=True,
    dpi=300,
):
    """
    Create a 100% stacked bar chart showing distribution of stack_category within each x_category.
    
    Automatically handles counting unique tracks and calculating percentages.
    
    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Input dataframe (accepts both pandas and polars).
    x_category : str
        Column name for x-axis categories (e.g., 'mol', 'condition').
    stack_category : str
        Column name for stacking categories (e.g., 'gate_name', 'cluster_id').
    unique_id_col : str, optional
        Column name containing unique track IDs. Default is 'unique_id'.
    x_order : list, optional
        Order of categories on x-axis. Default is None (sorted).
    stack_order : list, optional
        Order of stacking categories (bottom to top). Default is None (sorted).
    palette : str or list, optional
        Color palette. Can be 'colorblind', 'raiders'/'raiders_colors2', seaborn palette name,
        or list of colors. Default is 'colorblind'.
    figsize : tuple, optional
        Figure size (width, height). Default is (10, 6).
    title : str, optional
        Plot title. Auto-generated if None.
    xlabel : str, optional
        X-axis label. Uses x_category if None.
    ylabel : str, optional
        Y-axis label. Default is 'Percentage (%)'.
    show_percentages : bool, optional
        Whether to show percentage labels on bars. Default is True.
    percentage_threshold : float, optional
        Minimum percentage to show label (avoids cluttering). Default is 5.0.
    font_size : int, optional
        Base font size. Default is 12.
    transparent_background : bool, optional
        Transparent background. Default is False.
    line_color : str, optional
        Color for text and lines. Default is 'black'.
    grid : bool, optional
        Whether to show y-axis gridlines. Default is True.
    save_path : str, optional
        Path to save figure. Default is None (no save).
    export_format : str, optional
        Export format ('png' or 'svg'). Default is 'png'.
    return_svg : bool, optional
        Return SVG string if export_format='svg'. Default is False.
    show_plot : bool, optional
        Whether to display plot. Default is True.
    dpi : int, optional
        Resolution for saved figure. Default is 300.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    summary_df : pd.DataFrame
        Summary dataframe with percentages and counts.
    svg_data : str (optional)
        SVG string if return_svg=True and export_format='svg'.
    
    Examples
    --------
    >>> # Basic usage
    >>> fig, ax, summary = spt.plot_stacked_bar_percentage(
    ...     df, x_category='mol', stack_category='gate_name'
    ... )
    
    >>> # With custom colors and order
    >>> fig, ax, summary = spt.plot_stacked_bar_percentage(
    ...     df,
    ...     x_category='condition',
    ...     stack_category='cluster_id',
    ...     stack_order=['0', '1', '2', '3'],
    ...     palette=['#0173B2', '#DE8F05', '#029E73', '#CC78BC']
    ... )
    """
    # Detect and convert Polars DataFrames
    try:
        import polars as pl
        is_polars_input = isinstance(df, pl.DataFrame)
    except ImportError:
        is_polars_input = False
    
    if is_polars_input:
        # Use polars for efficient computation
        # Get unique tracks per category combination
        counts = (
            df
            .group_by([x_category, stack_category, unique_id_col])
            .agg(pl.len().alias('n_windows'))
            .group_by([x_category, stack_category])
            .agg([
                pl.col(unique_id_col).n_unique().alias('n_tracks'),
                pl.col('n_windows').sum().alias('total_windows')
            ])
        )
        
        # Calculate totals per x_category
        totals = (
            counts
            .group_by(x_category)
            .agg(pl.col('n_tracks').sum().alias('total_tracks'))
        )
        
        # Join and calculate percentages
        summary_df = (
            counts
            .join(totals, on=x_category, how='left')
            .with_columns([
                (pl.col('n_tracks') / pl.col('total_tracks') * 100).alias('percentage')
            ])
            .sort([x_category, stack_category])
            .to_pandas()
        )
    else:
        # Use pandas
        # Get unique tracks per category combination
        counts = (
            df.groupby([x_category, stack_category])[unique_id_col]
            .nunique()
            .reset_index()
            .rename(columns={unique_id_col: 'n_tracks'})
        )
        
        # Calculate totals per x_category
        totals = (
            counts.groupby(x_category)['n_tracks']
            .sum()
            .reset_index()
            .rename(columns={'n_tracks': 'total_tracks'})
        )
        
        # Join and calculate percentages
        summary_df = counts.merge(totals, on=x_category)
        summary_df['percentage'] = (summary_df['n_tracks'] / summary_df['total_tracks'] * 100)
        summary_df = summary_df.sort_values([x_category, stack_category])
    
    # Determine orders
    if x_order is None:
        x_order = sorted(summary_df[x_category].unique())
    if stack_order is None:
        stack_order = sorted(summary_df[stack_category].unique())
    
    # Get colors
    if palette == 'colorblind':
        colors = [
            '#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161',
            '#FBAFE4', '#949494', '#ECE133', '#56B4E9', '#D55E00'
        ]
    elif palette == 'raiders' or palette == 'raiders_colors2':
        colors = ['#ed1e24', '#DE8F05', '#6cc176', '#95d6d7']
    elif isinstance(palette, (list, tuple)):
        colors = palette
    else:
        # Try seaborn palette
        try:
            import seaborn as sns
            colors = sns.color_palette(palette, n_colors=len(stack_order))
        except:
            colors = plt.cm.tab10(np.linspace(0, 1, len(stack_order)))
    
    # Ensure enough colors
    while len(colors) < len(stack_order):
        colors = colors * 2
    colors = colors[:len(stack_order)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    if transparent_background:
        fig.patch.set_facecolor('none')
        ax.set_facecolor((0, 0, 0, 0))
    
    # Prepare data for stacking
    x_pos = np.arange(len(x_order))
    width = 0.6
    bottom = np.zeros(len(x_order))
    
    # Create stacked bars
    for i, stack_val in enumerate(stack_order):
        stack_data = summary_df[summary_df[stack_category] == stack_val]
        
        # Get percentages for each x category
        percentages = []
        for x_val in x_order:
            match = stack_data[stack_data[x_category] == x_val]
            pct = match['percentage'].values[0] if len(match) > 0 else 0
            percentages.append(pct)
        
        # Plot bar segment
        bars = ax.bar(
            x_pos, percentages, width,
            bottom=bottom,
            label=str(stack_val),
            color=colors[i],
            edgecolor='white',
            linewidth=0.5
        )
        
        # Add percentage labels
        if show_percentages:
            for j, (x_val, pct) in enumerate(zip(x_order, percentages)):
                if pct > percentage_threshold:
                    ax.text(
                        x_pos[j],
                        bottom[j] + pct/2,
                        f'{pct:.1f}%',
                        ha='center',
                        va='center',
                        fontsize=font_size - 2,
                        fontweight='bold',
                        color='white'
                    )
        
        bottom += percentages
    
    # Styling
    if xlabel is None:
        xlabel = x_category
    if title is None:
        title = f'{stack_category} Distribution by {x_category} (100% Stacked)'
    
    ax.set_xlabel(xlabel, fontsize=font_size + 2, fontweight='bold', color=line_color)
    ax.set_ylabel(ylabel, fontsize=font_size + 2, fontweight='bold', color=line_color)
    ax.set_title(title, fontsize=font_size + 4, fontweight='bold', pad=20, color=line_color)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_order, fontsize=font_size, color=line_color)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(colors=line_color)
    
    # Legend
    ax.legend(
        title=stack_category,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=font_size - 2,
        title_fontsize=font_size
    )
    
    # Grid
    if grid:
        ax.grid(True, alpha=0.3, axis='y', linestyle='--', color=line_color)
    
    # Spines
    for spine in ax.spines.values():
        spine.set_edgecolor(line_color)
    
    plt.tight_layout()
    
    # Save if requested
    svg_data = None
    if save_path is not None:
        ext = export_format.lower()
        if ext not in ['png', 'svg']:
            print("Invalid export format. Defaulting to 'png'.")
            ext = 'png'
        
        # Generate filename if directory provided
        if os.path.isdir(save_path):
            filename = f"{x_category}_by_{stack_category}_stacked.{ext}"
            full_path = os.path.join(save_path, filename)
        else:
            # Use as full path, adjust extension
            base, _ = os.path.splitext(save_path)
            full_path = f"{base}.{ext}"
        
        if ext == 'svg':
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='svg', dpi=dpi, bbox_inches='tight',
                       transparent=transparent_background)
            buf.seek(0)
            svg_data = buf.read().decode('utf-8')
            
            with open(full_path, 'w') as f:
                f.write(svg_data)
        else:
            plt.savefig(full_path, dpi=dpi, bbox_inches='tight',
                       transparent=transparent_background)
        
        print(f"✅ Saved to: {full_path}")
    
    if show_plot:
        plt.show()
    
    if return_svg and svg_data is not None:
        return fig, ax, summary_df, svg_data
    else:
        return fig, ax, summary_df


def create_test_track_data_polars(n_tracks=10, track_length=50, plot_size=8):
    """
    Create simulated track data in Polars format for testing.
    
    Parameters
    ----------
    n_tracks : int
        Number of tracks to generate
    track_length : int
        Number of points per track
    plot_size : float
        Size of the plotting area in microns
        
    Returns
    -------
    polars.DataFrame
        Simulated track data
    """
    import polars as pl
    import numpy as np
    
    data = []
    
    for track_id in range(n_tracks):
        # Random starting position
        start_x = np.random.uniform(1, plot_size - 1)
        start_y = np.random.uniform(1, plot_size - 1)
        
        # Random walk parameters
        step_size = 0.1
        drift_x = np.random.uniform(-0.02, 0.02)
        drift_y = np.random.uniform(-0.02, 0.02)
        
        x_pos, y_pos = start_x, start_y
        
        for frame in range(track_length):
            # Random walk with slight drift
            x_pos += np.random.normal(drift_x, step_size)
            y_pos += np.random.normal(drift_y, step_size)
            
            # Keep within bounds
            x_pos = np.clip(x_pos, 0.1, plot_size - 0.1)
            y_pos = np.clip(y_pos, 0.1, plot_size - 0.1)
            
            # Add simple categorical for testing gallery function
            simple_threshold = 'high' if track_id % 2 == 0 else 'low'
            motion_type = ['subdiffusive', 'normal', 'superdiffusive'][track_id % 3]
            
            data.append({
                'unique_id': f'track_{track_id:03d}',
                'frame': frame,
                'x_um': x_pos,
                'y_um': y_pos,
                'particle': track_id,
                'filename': 'test_data',
                'condition': 'simulated',
                'simple_threshold': simple_threshold,
                'motion_type': motion_type
            })
    
    return pl.DataFrame(data)


def plot_xy_heatmap(
    df,
    x_col,
    y_col,
    plot_type='hexbin',
    color_by=None,
    order=None,
    small_multiples=False,
    shared_scale=True,
    gridsize=50,
    figsize=(8, 6),
    cmap=None,
    alpha=0.6,
    s=10,
    title=None,
    xlabel=None,
    ylabel=None,
    scale_data=False,
    scale_method='standard',
    scaler=None,
    scale_group_by=None,
    xlim=None,
    ylim=None,
    save_path=None,
    dpi=150,
    log_scale=True,
    transparent_background=False,
    line_color='black',
    export_format='png',
    return_svg=False,
    contour_levels=None,
    contour_cmap='viridis',
    gates=None,
    gate_order=None,
    gate_colors='black',
    gate_linewidth=2,
    gate_alpha=0.1,
    gate_edge_alpha=1.0,
    gate_text_size=10,
    gate_label_border=True,
    gate_linestyle='--',
    gate_fill_color=None,
    gate_label_position='auto',
    **kwargs
):
    """
    Create flexible x vs y plots with hexbin or scatter options.
    
    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        DataFrame containing the data to plot. Accepts both pandas and polars.
    x_col : str
        Column name for x-axis.
    y_col : str
        Column name for y-axis.
    plot_type : str, optional
        Type of plot. Options are:
        - 'hexbin': Hexagonal binning (good for medium-large datasets)
        - 'scatter': Individual points (good for small datasets or categorical data)
        Default is 'hexbin'.
    color_by : str, optional
        Column name to color/separate data by. If None, pools all data together.
        If specified, colors scatter plots by category or creates small multiples
        when small_multiples=True. Default is None.
    order : list, optional
        Custom order for categories when using color_by parameter. If None, 
        categories will be sorted alphabetically. If provided, should be a list 
        of category values in the desired display order. Only categories present 
        in both the order list and the data will be plotted. Default is None.
    small_multiples : bool, optional
        If True and color_by is provided, creates separate subplots for each category.
        If False, plots all data together (colored by category for scatter).
        Default is False.
    shared_scale : bool, optional
        If True (default), all subplots share the same x/y scale (max of all data).
        If False, each subplot uses different colormaps (for hexbin/hist2d).
        Only applies when small_multiples=True.
    gridsize : int, optional
        Grid size for hexbin plots. Default is 50.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (8, 6).
    cmap : str or list of str, optional
        Colormap(s) for plots. 
        - For hexbin plots:
          * Single hexbin: string colormap name (default: 'inferno')
          * Small multiples hexbin: list of colormaps, one per subplot
            (default: ['inferno', 'viridis', 'plasma', 'magma', 'cividis', 'hot'])
        - For scatter plots:
          * Continuous color_by (numeric, >20 unique values): colormap name 
            (default: 'viridis'), displays with colorbar
          * Categorical color_by: converts colormap to discrete colors, or pass 
            list of colors directly
          * Special value 'colorblind': Wong 2011 colorblind-friendly palette
          * No color_by: ignored (uses default color)
        Good colormaps: 'colorblind' (recommended!), 'viridis', 'plasma', 
        'inferno', 'magma', 'cividis', 'coolwarm', 'RdYlBu', 'tab10', 'Set2'.
    alpha : float, optional
        Transparency for scatter plots. Default is 0.6.
    s : float or int, optional
        Marker size for scatter plots. Default is 10.
    title : str, optional
        Overall figure title. Default is None.
    xlabel : str, optional
        X-axis label. Default is x_col.
    ylabel : str, optional
        Y-axis label. Default is y_col.
    scale_data : bool, optional
        If True, applies scaling to x_col and y_col before plotting. Default is False.
    scale_method : str, optional
        Scaling method to use if scale_data=True and scaler is None. Default is 'standard'.
        Options: 'standard', 'minmax', 'robust', 'standard_minmax', etc.
    scaler : sklearn scaler object, optional
        Pre-fitted scaler to use for transforming data. If provided, this scaler
        will be used instead of fitting a new one (CRITICAL for Option A gating).
        The scaler must have a .transform() method and be already fitted.
        Default is None.
    scale_group_by : str or list of str, optional
        Group by column(s) for scaling. Only used if scaler is None. Default is None.
    xlim : tuple, optional
        X-axis limits (xmin, xmax). Default is None (auto).
    ylim : tuple, optional
        Y-axis limits (ymin, ymax). Default is None (auto).
    save_path : str, optional
        Path to save the figure. Can be:
        - A directory path: filename will be auto-generated as 
          "{x_col}_vs_{y_col}_{plot_type}_by_{color_by}_{modifiers}.{ext}"
        - A full file path: will be used as-is (extension adjusted to match export_format)
        Default is None (no save).
    dpi : int, optional
        DPI for saved figure. Default is 150.
    log_scale : bool, optional
        If True, use logarithmic color scaling for hexbin plots to better
        highlight areas of low and high density. Default is True.
    transparent_background : bool, optional
        If True, makes the background transparent. Useful for presentations
        and publications. Default is False.
    line_color : str, optional
        Color for axis lines, ticks, and labels. Default is 'black'.
        Use 'white' for dark backgrounds in presentations.
    export_format : str, optional
        File format to export ('png' or 'svg'). Default is 'png'.
    return_svg : bool, optional
        If True and export_format is 'svg', returns the cleaned SVG data as a string.
        Default is False.
    contour_levels : int or None, optional
        If specified, adds density contour lines on top of scatter/hexbin plots.
        Number indicates how many contour levels to draw. Default is None (no contours).
    contour_cmap : str, optional
        Colormap for contour lines when contour_levels is specified.
        Default is 'viridis'.
    gates : ROIManager or list, optional
        Gates to draw on the plot. Can be:
        - An ROIManager object (e.g., from interactive_roi_gating_with_capture)
        - A list of gate dictionaries with 'coordinates', 'type', and optional 'name'
        Default is None (no gates drawn).
    gate_order : list of str, optional
        Order of gate names for color assignment. If provided, gates will be colored
        in this order. Only gates present in both gate_order and the data will be drawn.
        This ensures consistent color mapping across multiple plots. Default is None
        (uses order from ROIManager or alphabetical by gate name).
    gate_colors : str or list of str, optional
        Color(s) for gate edges/borders. Can be a single color or list of colors.
        Good options for dashed lines: 'black', 'darkgray', 'navy', 'darkgreen',
        'darkred', 'purple'. Default is 'black'.
    gate_linewidth : float, optional
        Line width for gate borders. Default is 2.
    gate_alpha : float, optional
        Transparency for gate fill (0-1). Set to 0 for no fill. Default is 0.1.
        This does NOT affect edge transparency - edges use gate_edge_alpha.
    gate_edge_alpha : float, optional
        Transparency for gate edges/borders (0-1). Default is 1.0 (fully opaque).
        Keep at 1.0 for clear, visible gate boundaries.
    gate_text_size : float, optional
        Font size for gate labels. Default is 10.
    gate_label_border : bool, optional
        Whether to draw a border around gate label text boxes. Default is True.
    gate_linestyle : str, optional
        Line style for gate borders. Options: '-' (solid), '--' (dashed), 
        '-.' (dash-dot), ':' (dotted). Default is '--' (dashed).
    gate_fill_color : str or None, optional
        Fill color for gates. If None, uses same color as edge with alpha.
        Set to 'none' for transparent fill. Good options: 'lightgray', 'lightblue',
        'lightyellow', 'white', or 'none'. Default is None (uses edge color).
    gate_label_position : str, optional
        Label position strategy. Options:
        - 'auto': Smart positioning based on gate location (recommended)
          * Gates closer to left side (0) → label at top-left corner
          * Gates closer to right side (1) → label at top-right corner
          * Always positions at top of gate for visibility
        - 'top-left': Always top-left corner
        - 'top-right': Always top-right corner
        - 'center': Center of gate
        Default is 'auto'.
    **kwargs
        Additional keyword arguments passed to plotting functions.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : matplotlib.axes.Axes or np.ndarray of Axes
        The axes object(s).
    svg_data : str (optional)
        If export_format='svg' and return_svg=True, returns cleaned SVG string.
    
    Examples
    --------
    >>> # Simple hexbin plot (all data pooled)
    >>> fig, ax = plot_xy_heatmap(df, 'x_um', 'y_um')
    
    >>> # Scatter plot colored by category (automatic colors)
    >>> fig, ax = plot_xy_heatmap(df, 'speed', 'D', plot_type='scatter', color_by='mol')
    
    >>> # Scatter plot with colorblind-friendly palette (recommended!)
    >>> fig, ax = plot_xy_heatmap(df, 'speed', 'D', plot_type='scatter', 
    ...                            color_by='mol', cmap='colorblind')
    
    >>> # Scatter plot with other custom colormap for categories
    >>> fig, ax = plot_xy_heatmap(df, 'speed', 'D', plot_type='scatter', 
    ...                            color_by='mol', cmap='Set2')
    
    >>> # Scatter plot with continuous colormap (for numeric color_by)
    >>> fig, ax = plot_xy_heatmap(df, 'x_um', 'y_um', plot_type='scatter',
    ...                            color_by='speed', cmap='plasma')
    
    >>> # Small multiples by condition
    >>> fig, axes = plot_xy_heatmap(df, 'x_um', 'y_um', 
    ...                              color_by='condition',
    ...                              small_multiples=True)
    
    >>> # Small multiples with custom order
    >>> fig, axes = plot_xy_heatmap(df, 'x_um', 'y_um', 
    ...                              color_by='condition',
    ...                              order=['control', 'drug1', 'drug2'],
    ...                              small_multiples=True)
    
    >>> # With data scaling
    >>> fig, ax = plot_xy_heatmap(df, 'x_um', 'y_um',
    ...                            scale_data=True,
    ...                            scale_method='standard_minmax')
    
    >>> # With gates (dashed lines, no fill, opaque edges)
    >>> fig, ax = plot_xy_heatmap(df, 'x_um', 'y_um',
    ...                            gates=roi_manager,
    ...                            gate_colors='black',
    ...                            gate_linestyle='--',
    ...                            gate_fill_color='none',
    ...                            gate_linewidth=2,
    ...                            gate_edge_alpha=1.0)  # Opaque edges (default)
    
    >>> # With gates (transparent fill, smart auto label positioning, separate alphas)
    >>> fig, ax = plot_xy_heatmap(df, 'x_um', 'y_um',
    ...                            gates=roi_manager,
    ...                            gate_colors='navy',
    ...                            gate_fill_color='lightblue',
    ...                            gate_alpha=0.15,          # Fill transparency
    ...                            gate_edge_alpha=1.0,      # Opaque edges
    ...                            gate_label_position='auto')  # Smart: left gates→top-left, right gates→top-right
    
    >>> # With gates (no label borders for clean look)
    >>> fig, ax = plot_xy_heatmap(df, 'x_um', 'y_um',
    ...                            gates=roi_manager,
    ...                            gate_colors='black',
    ...                            gate_linestyle='--',
    ...                            gate_fill_color='none',
    ...                            gate_label_border=False)  # No borders around labels
    
    >>> # With gates (consistent color mapping across plots)
    >>> gate_names = ['Low_Speed', 'Medium_Speed', 'High_Speed', 'Very_High']
    >>> gate_colors = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC']
    >>> fig, ax = plot_xy_heatmap(df, 'x_um', 'y_um',
    ...                            gates=roi_manager,
    ...                            gate_order=gate_names,  # Consistent order
    ...                            gate_colors=gate_colors,  # Matches order
    ...                            gate_linestyle='--')
    
    """
    # Detect and convert Polars DataFrames to Pandas for plotting
    # Matplotlib requires pandas/numpy for plotting operations
    try:
        import polars as pl
        is_polars_input = isinstance(df, pl.DataFrame)
    except ImportError:
        is_polars_input = False
    
    if is_polars_input:
        # Convert to pandas for plotting (matplotlib requires pandas/numpy)
        plot_df = df.to_pandas()
    else:
        # Create a copy of the dataframe to avoid modifying the original
        plot_df = df.copy()
    
    # Apply scaling if requested
    if scale_data:
        if scaler is not None:
            # Use the provided scaler (OPTION A - for population-level gating)
            plot_df[[x_col, y_col]] = scaler.transform(plot_df[[x_col, y_col]])
        else:
            # Fit a new scaler on this data (default behavior)
            plot_df = center_scale_data(
                plot_df,
                columns=[x_col, y_col],
                method=scale_method,
                group_by=scale_group_by,
                copy=False
            )
    
    # Set default labels
    if xlabel is None:
        xlabel = x_col
    if ylabel is None:
        ylabel = y_col
    
    def add_density_contours(ax, x, y, levels=5, cmap_name='viridis', alpha_contour=0.6):
        """Add 2D density contours."""
        # Calculate 2D density
        try:
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            
            # Create grid for contours
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            # Add padding
            x_min -= x_range * 0.1
            x_max += x_range * 0.1
            y_min -= y_range * 0.1
            y_max += y_range * 0.1
            
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            f = np.reshape(gaussian_kde(xy)(positions).T, xx.shape)
            
            # Draw contours
            contours = ax.contour(xx, yy, f, levels=levels, cmap=cmap_name, 
                                 alpha=alpha_contour, linewidths=1.5)
            ax.clabel(contours, inline=True, fontsize=8)
        except (ValueError, np.linalg.LinAlgError):
            # Skip if KDE fails (too few points or singular matrix)
            pass
    
    # Set default colormaps
    default_hexbin_cmaps = ['inferno', 'viridis', 'plasma', 'magma', 'cividis', 'hot']
    if cmap is None:
        if plot_type == 'hexbin':
            cmap = 'inferno'  # Single plot default
        else:
            cmap = None  # Scatter uses categorical colors
    
    # Set backgrounds
    figure_background = 'none' if transparent_background else 'white'
    axis_background = (0, 0, 0, 0) if transparent_background else 'white'
    
    # Determine if we need subplots based on color_by and small_multiples
    if color_by is not None and small_multiples:
        if order is not None:
            # Use custom order, filtering to only include categories present in data
            unique_vals = set(plot_df[color_by].unique())
            categories = [cat for cat in order if cat in unique_vals]
        else:
            # Default: sort alphabetically
            categories = sorted(plot_df[color_by].unique())
        n_cats = len(categories)
        
        # Always create vertical stack of plots for small multiples
        fig, axes = plt.subplots(
            n_cats, 1,
            figsize=(figsize[0], figsize[1] * n_cats)
        )
        fig.patch.set_facecolor(figure_background)
        if n_cats == 1:
            axes = [axes]
        
        # Calculate shared scale if needed
        if shared_scale:
            x_min, x_max = plot_df[x_col].min(), plot_df[x_col].max()
            y_min, y_max = plot_df[y_col].min(), plot_df[y_col].max()
            # Use max range for both axes
            data_range = max(x_max - x_min, y_max - y_min)
            x_center = (x_max + x_min) / 2
            y_center = (y_max + y_min) / 2
            xlim_shared = (x_center - data_range/2, x_center + data_range/2)
            ylim_shared = (y_center - data_range/2, y_center + data_range/2)
        
        # Handle colormap list for hexbin plots
        cmap_list = cmap
        if plot_type == 'hexbin':
            if isinstance(cmap, str):
                # Single colormap provided - use different ones for each subplot
                cmap_list = default_hexbin_cmaps
            if isinstance(cmap_list, list):
                # Extend list if needed
                if len(cmap_list) < n_cats:
                    cmap_list = (cmap_list * (n_cats // len(cmap_list) + 1))[:n_cats]
            else:
                cmap_list = [cmap_list] * n_cats
        
        # Plot each category
        for idx, cat in enumerate(categories):
            ax = axes[idx]
            ax.set_facecolor(axis_background)
            cat_df = plot_df[plot_df[color_by] == cat]
            
            if plot_type == 'hexbin':
                # Use different colormap for each subplot
                current_cmap = cmap_list[idx] if isinstance(cmap_list, list) else cmap_list
                norm = LogNorm() if log_scale else None
                
                hb = ax.hexbin(
                    cat_df[x_col],
                    cat_df[y_col],
                    gridsize=gridsize,
                    cmap=current_cmap,
                    mincnt=1,
                    norm=norm,
                    **kwargs
                )
                cbar = plt.colorbar(hb, ax=ax, label='Count (log scale)' if log_scale else 'Count')
                cbar.ax.yaxis.set_tick_params(color=line_color)
                cbar.ax.yaxis.label.set_color(line_color)
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=line_color)
                
                # Add contours if requested
                if contour_levels is not None:
                    add_density_contours(ax, cat_df[x_col].values, cat_df[y_col].values, 
                                        levels=contour_levels, cmap_name=contour_cmap)
                
            elif plot_type == 'scatter':
                # For scatter in small multiples, use uniform single color per subplot
                # Get color from specified palette or use default
                if cmap == 'colorblind':
                    # Use Wong 2011 colorblind-friendly palette
                    colorblind_colors = [
                        '#0173B2',  # Blue
                        '#DE8F05',  # Orange  
                        '#029E73',  # Green
                        '#CC78BC',  # Purple
                        '#CA9161',  # Brown
                        '#FBAFE4',  # Pink
                        '#949494',  # Gray
                        '#ECE133',  # Yellow
                        '#56B4E9',  # Sky blue
                        '#D55E00'   # Vermillion
                    ]
                    color_palette = colorblind_colors[idx % len(colorblind_colors)]
                elif cmap is not None and isinstance(cmap, (list, tuple)):
                    # Custom list of colors
                    color_palette = cmap[idx % len(cmap)]
                elif cmap is not None and isinstance(cmap, str):
                    # Try to use as colormap
                    try:
                        colormap = plt.cm.get_cmap(cmap)
                        color_palette = colormap(idx / n_cats)
                    except:
                        color_palette = plt.cm.tab10(idx % 10)
                else:
                    # Default to tab10
                    color_palette = plt.cm.tab10(idx % 10)
                
                ax.scatter(cat_df[x_col], cat_df[y_col], s=s, alpha=alpha, 
                          color=color_palette, **kwargs)
                
                # Add contours if requested
                if contour_levels is not None:
                    add_density_contours(ax, cat_df[x_col].values, cat_df[y_col].values,
                                        levels=contour_levels, cmap_name=contour_cmap)
            else:
                raise ValueError(f"plot_type must be 'hexbin' or 'scatter', got '{plot_type}'")
            
            # Set limits
            if shared_scale:
                ax.set_xlim(xlim_shared if xlim is None else xlim)
                ax.set_ylim(ylim_shared if ylim is None else ylim)
            else:
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
            
            # Labels and styling
            ax.set_xlabel(xlabel, color=line_color)
            ax.set_ylabel(ylabel, color=line_color)
            ax.set_title(f"{color_by}: {cat}", color=line_color)
            ax.grid(alpha=0.3, color=line_color)
            ax.tick_params(colors=line_color)
            for spine in ax.spines.values():
                spine.set_edgecolor(line_color)
        
        # No unused axes to hide in vertical layout
        
    else:
        # Single plot (pool all data or color by category)
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(figure_background)
        ax.set_facecolor(axis_background)
        
        if plot_type == 'hexbin':
            # Density plot - pool all data
            norm = LogNorm() if log_scale else None
            hb = ax.hexbin(
                plot_df[x_col],
                plot_df[y_col],
                gridsize=gridsize,
                cmap=cmap,
                mincnt=1,
                norm=norm,
                **kwargs
            )
            cbar = plt.colorbar(hb, ax=ax, label='Count (log scale)' if log_scale else 'Count')
            cbar.ax.yaxis.set_tick_params(color=line_color)
            cbar.ax.yaxis.label.set_color(line_color)
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=line_color)
            
            # Add contours if requested
            if contour_levels is not None:
                add_density_contours(ax, plot_df[x_col].values, plot_df[y_col].values,
                                    levels=contour_levels, cmap_name=contour_cmap)
            
        elif plot_type == 'scatter':
            if color_by is not None and color_by in plot_df.columns:
                # Check if color_by is numeric (continuous) or categorical
                is_numeric_color = np.issubdtype(plot_df[color_by].dtype, np.number)
                n_unique = plot_df[color_by].nunique()
                
                # Treat as continuous if numeric and many unique values
                if is_numeric_color and n_unique > 20:
                    # Continuous coloring with colormap
                    scatter_cmap = cmap if cmap is not None else 'viridis'
                    scatter = ax.scatter(
                        plot_df[x_col],
                        plot_df[y_col],
                        c=plot_df[color_by],
                        s=s,
                        alpha=alpha,
                        cmap=scatter_cmap,
                        **kwargs
                    )
                    cbar = plt.colorbar(scatter, ax=ax, label=color_by)
                    cbar.ax.yaxis.set_tick_params(color=line_color)
                    cbar.ax.yaxis.label.set_color(line_color)
                    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=line_color)
                else:
                    # Categorical coloring
                    if order is not None:
                        # Use custom order, filtering to only include categories present in data
                        unique_vals = set(plot_df[color_by].unique())
                        unique_colors = [cat for cat in order if cat in unique_vals]
                    else:
                        # Default: sort alphabetically
                        unique_colors = sorted(plot_df[color_by].unique())
                    
                    # Get colors from colormap if provided, otherwise use default palette
                    if cmap is not None:
                        # Handle special palette names
                        if cmap == 'colorblind':
                            # Use Wong 2011 colorblind-friendly palette
                            colors = [
                                '#0173B2',  # Blue
                                '#DE8F05',  # Orange  
                                '#029E73',  # Green
                                '#CC78BC',  # Purple
                                '#CA9161',  # Brown
                                '#FBAFE4',  # Pink
                                '#949494',  # Gray
                                '#ECE133',  # Yellow
                                '#56B4E9',  # Sky blue
                                '#D55E00'   # Vermillion
                            ]
                        elif isinstance(cmap, str):
                            # Try to get it as a matplotlib colormap
                            try:
                                colormap = plt.cm.get_cmap(cmap)
                                colors = [colormap(i / len(unique_colors)) for i in range(len(unique_colors))]
                            except ValueError:
                                # If not a valid colormap, try as seaborn palette
                                try:
                                    import seaborn as sns
                                    colors = sns.color_palette(cmap, n_colors=len(unique_colors))
                                except:
                                    # Fall back to default
                                    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                        else:
                            colors = cmap  # Assume it's a list of colors
                    else:
                        # Use matplotlib's default color cycle
                        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                    
                    for idx, color_val in enumerate(unique_colors):
                        color_df = plot_df[plot_df[color_by] == color_val]
                        ax.scatter(
                            color_df[x_col],
                            color_df[y_col],
                            s=s,
                            alpha=alpha,
                            color=colors[idx % len(colors)],
                            label=str(color_val),
                            **kwargs
                        )
                    ax.legend(title=color_by, bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # Add contours if requested (on all data)
                if contour_levels is not None:
                    add_density_contours(ax, plot_df[x_col].values, plot_df[y_col].values,
                                        levels=contour_levels, cmap_name=contour_cmap)
            else:
                # Single color scatter (pool all data)
                # For single scatter without color_by, cmap is not used
                ax.scatter(plot_df[x_col], plot_df[y_col], s=s, alpha=alpha, **kwargs)
                
                # Add contours if requested
                if contour_levels is not None:
                    add_density_contours(ax, plot_df[x_col].values, plot_df[y_col].values,
                                        levels=contour_levels, cmap_name=contour_cmap)
        else:
            raise ValueError(f"plot_type must be 'hexbin' or 'scatter', got '{plot_type}'")
        
        # Set limits
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        
        # Labels and styling
        ax.set_xlabel(xlabel, color=line_color)
        ax.set_ylabel(ylabel, color=line_color)
        ax.grid(alpha=0.3, color=line_color)
        ax.tick_params(colors=line_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(line_color)
        axes = ax
    
    # Draw gates if provided
    if gates is not None:
        # Handle both ROIManager object and list of gate dictionaries
        gate_list = []
        if hasattr(gates, 'rois'):  # ROIManager object
            # If scale_data is False and ROIManager has a scaler, we need to inverse transform
            # the gate coordinates to match the unscaled plot
            if not scale_data and hasattr(gates, 'scaler') and gates.scaler is not None:
                gate_list = gates.get_inverse_transformed_gates()
            else:
                gate_list = gates.rois
        elif isinstance(gates, list):  # List of gate dicts
            gate_list = gates
        else:
            print(f"Warning: gates parameter type not recognized. Expected ROIManager or list.")
        
        # Apply gate_order if provided
        if gate_order is not None:
            # Create mapping of gate names to gates
            gate_dict = {gate.get('name', f"Gate {i}"): gate for i, gate in enumerate(gate_list)}
            # Reorder gates according to gate_order, filtering to only include present gates
            ordered_gates = []
            for gate_name in gate_order:
                if gate_name in gate_dict:
                    ordered_gates.append(gate_dict[gate_name])
            gate_list = ordered_gates
        
        # Draw gates on all axes
        axes_to_draw = [axes] if not isinstance(axes, np.ndarray) else axes.flatten()
        
        from matplotlib.patches import Rectangle
        from matplotlib.patches import Polygon as MplPolygon
        import matplotlib.colors as mcolors
        
        for ax_obj in axes_to_draw:
            # Get axis limits for smart positioning
            x_lim = ax_obj.get_xlim()
            y_lim = ax_obj.get_ylim()
            x_range = x_lim[1] - x_lim[0]
            y_range = y_lim[1] - y_lim[0]
            x_mid = (x_lim[0] + x_lim[1]) / 2
            y_mid = (y_lim[0] + y_lim[1]) / 2
            
            for i, gate in enumerate(gate_list):
                coords = gate.get('coordinates', [])
                gate_name = gate.get('name', f"Gate {i}")
                
                if not coords:
                    continue
                
                # Determine gate edge color
                if isinstance(gate_colors, str):
                    edge_color_base = gate_colors
                elif isinstance(gate_colors, list):
                    edge_color_base = gate_colors[i % len(gate_colors)]
                else:
                    edge_color_base = 'black'
                
                # Convert edge color to RGBA with edge_alpha baked in
                edge_rgba = mcolors.to_rgba(edge_color_base, alpha=gate_edge_alpha)
                
                # Determine gate fill color with fill_alpha baked in
                if gate_fill_color == 'none':
                    fill_rgba = 'none'
                elif gate_fill_color is not None:
                    # Convert fill color to RGBA with gate_alpha baked in
                    fill_rgba = mcolors.to_rgba(gate_fill_color, alpha=gate_alpha)
                else:
                    # Use edge color with gate_alpha for fill
                    fill_rgba = mcolors.to_rgba(edge_color_base, alpha=gate_alpha)
                
                # Draw based on gate type
                if gate.get('type') == 'rect':
                    # Rectangle gate
                    x0, y0 = coords[0]
                    x1, y1 = coords[2]  # Opposite corner
                    width = x1 - x0
                    height = y1 - y0
                    
                    rect = Rectangle(
                        (x0, y0), width, height,
                        linewidth=gate_linewidth,
                        edgecolor=edge_rgba,  # RGBA with edge_alpha baked in
                        facecolor=fill_rgba,  # RGBA with fill_alpha baked in (or 'none')
                        linestyle=gate_linestyle,
                        label=gate_name
                    )
                    # Don't set alpha parameter - it would multiply with the baked-in alphas!
                    ax_obj.add_patch(rect)
                    
                    # Smart label positioning
                    gate_center_x = x0 + width / 2
                    gate_center_y = y0 + height / 2
                    
                    if gate_label_position == 'auto':
                        # Smart positioning based on gate proximity to 0 and 1 on axes
                        # Calculate how close gate is to min (0-like) vs max (1-like) on each axis
                        
                        # For X-axis: determine if gate is closer to left (0) or right (1)
                        dist_to_left = gate_center_x - x_lim[0]
                        dist_to_right = x_lim[1] - gate_center_x
                        
                        # For Y-axis: we always want labels at TOP of gates for visibility
                        # but adjust horizontal position based on X proximity
                        
                        if dist_to_left < dist_to_right:
                            # Gate closer to left/0 side → label at top-left
                            label_x = x0 + width * 0.05
                            ha = 'left'
                        else:
                            # Gate closer to right/1 side → label at top-right
                            label_x = x1 - width * 0.05
                            ha = 'right'
                        
                        # Always put labels at top of gate for better visibility
                        label_y = y1 - height * 0.05
                        va = 'top'
                    elif gate_label_position == 'top-left':
                        label_x = x0 + width * 0.05
                        label_y = y1 - height * 0.05
                        ha = 'left'
                        va = 'top'
                    elif gate_label_position == 'top-right':
                        label_x = x1 - width * 0.05
                        label_y = y1 - height * 0.05
                        ha = 'right'
                        va = 'top'
                    elif gate_label_position == 'center':
                        label_x = gate_center_x
                        label_y = gate_center_y
                        ha = 'center'
                        va = 'center'
                    else:
                        # Default to top-left
                        label_x = x0 + width * 0.05
                        label_y = y1 - height * 0.05
                        ha = 'left'
                        va = 'top'
                    
                    # Add label
                    if gate_label_border:
                        label_bbox = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, 
                                         edgecolor=edge_color_base, linewidth=1)
                    else:
                        label_bbox = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, 
                                         edgecolor='none', linewidth=0)
                    
                    ax_obj.text(
                        label_x,
                        label_y,
                        gate_name,
                        fontsize=gate_text_size,
                        color=edge_color_base,  # Use base color for text
                        fontweight='bold',
                        ha=ha,
                        va=va,
                        bbox=label_bbox
                    )
                else:
                    # Polygon gate
                    polygon = MplPolygon(
                        coords,
                        linewidth=gate_linewidth,
                        edgecolor=edge_rgba,  # RGBA with edge_alpha baked in
                        facecolor=fill_rgba,  # RGBA with fill_alpha baked in (or 'none')
                        linestyle=gate_linestyle,
                        label=gate_name
                    )
                    # Don't set alpha parameter - it would multiply with the baked-in alphas!
                    ax_obj.add_patch(polygon)
                    
                    # Add label for polygon
                    xs, ys = zip(*coords)
                    cx, cy = np.mean(xs), np.mean(ys)
                    
                    # Smart positioning for polygon
                    if gate_label_position == 'auto':
                        # Use same logic as rectangles: proximity to left vs right
                        dist_to_left = cx - x_lim[0]
                        dist_to_right = x_lim[1] - cx
                        
                        if dist_to_left < dist_to_right:
                            # Polygon closer to left/0 side → position label to left
                            ha = 'right'
                            label_offset_x = -0.02 * x_range
                        else:
                            # Polygon closer to right/1 side → position label to right
                            ha = 'left'
                            label_offset_x = 0.02 * x_range
                        
                        # Position slightly above centroid
                        va = 'bottom'
                        label_offset_y = 0.02 * y_range
                        
                        label_x = cx + label_offset_x
                        label_y = cy + label_offset_y
                    else:
                        label_x = cx
                        label_y = cy
                        ha = 'center'
                        va = 'center'
                    
                    if gate_label_border:
                        label_bbox = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, 
                                         edgecolor=edge_color_base, linewidth=1)
                    else:
                        label_bbox = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, 
                                         edgecolor='none', linewidth=0)
                    
                    ax_obj.text(
                        label_x,
                        label_y,
                        gate_name,
                        fontsize=gate_text_size,
                        color=edge_color_base,  # Use base color for text
                        fontweight='bold',
                        ha=ha,
                        va=va,
                        bbox=label_bbox
                    )
    
    # Set overall title
    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight='bold', color=line_color)
    
    plt.tight_layout()
    
    # Save if requested
    svg_data = None
    if save_path is not None:
        ext = export_format.lower()
        if ext not in ['png', 'svg']:
            print("Invalid export format specified. Defaulting to 'png'.")
            ext = 'png'
        
        # Check if save_path is a directory or a full file path
        if os.path.isdir(save_path) or (not os.path.exists(save_path) and not save_path.endswith(('.png', '.svg'))):
            # It's a directory - generate filename automatically
            save_dir = save_path
            
            # Build descriptive filename
            filename_parts = [f"{x_col}_vs_{y_col}", plot_type]
            if color_by is not None:
                filename_parts.append(f"by_{color_by}")
            if small_multiples:
                filename_parts.append("multiples")
            if scale_data:
                filename_parts.append(scale_method)
            
            filename = "_".join(filename_parts) + f".{ext}"
            save_path = os.path.join(save_dir, filename)
        else:
            # It's a full file path - use as is but ensure correct extension
            save_dir = os.path.dirname(save_path)
            if not save_path.endswith(f'.{ext}'):
                save_path = save_path.rsplit('.', 1)[0] + f'.{ext}' if '.' in save_path else f'{save_path}.{ext}'
        
        # Create directory if it doesn't exist
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        fig.savefig(
            save_path,
            transparent=transparent_background,
            dpi=dpi,
            format=ext,
            bbox_inches='tight'
        )
        
        # SVG post-processing
        if ext == 'svg':
            try:
                with open(save_path, encoding='utf-8') as f:
                    svg_data = f.read()
                # Remove <clipPath> definitions, metadata, XML declaration, and DOCTYPE
                svg_data = re.sub(
                    r'<clipPath id="[^"]*">.*?</clipPath>', '', svg_data, flags=re.DOTALL
                )
                svg_data = re.sub(r'\s*clip-path="url\([^)]*\)"', '', svg_data)
                svg_data = re.sub(
                    r'<metadata>.*?</metadata>', '', svg_data, flags=re.DOTALL
                )
                svg_data = re.sub(r'<\?xml[^>]*\?>', '', svg_data, flags=re.DOTALL)
                svg_data = re.sub(r'<!DOCTYPE[^>]*>', '', svg_data, flags=re.DOTALL)
                svg_data = svg_data.strip()
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(svg_data)
            except Exception as e:
                print(f"Warning: SVG post-processing failed: {e}")
        
        print(f"Figure saved to: {save_path}")
    
    if return_svg and svg_data:
        return fig, axes, svg_data
    else:
        return fig, axes


def interactive_roi_gating(
    df,
    x_col,
    y_col,
    color_by=None,
    scale_data=False,
    scale_method='standard',
    title=None,
    point_size=3,
    opacity=0.6,
    contour_density=True,
    n_contours=10,
    contour_colorscale='Viridis',
    height=700,
    width=900,
    show_legend=True,
    sample_frac=None,
    colorscale='Plotly',
):
    """
    Interactive FACS-style gating plot using Plotly with polygonal ROI selection.
    
    This function creates an interactive scatter plot where you can draw polygonal 
    regions of interest (ROIs) to gate/filter your data, similar to FACS analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to plot.
    x_col : str
        Column name for x-axis.
    y_col : str
        Column name for y-axis.
    color_by : str, optional
        Column name to color points by. If None, all points are the same color.
        Can be categorical or continuous. Default is None.
    scale_data : bool, optional
        If True, applies scaling to x_col and y_col before plotting. Default is False.
    scale_method : str, optional
        Scaling method if scale_data=True. Options: 'standard', 'minmax', 
        'robust', 'standard_minmax'. Default is 'standard'.
    title : str, optional
        Plot title. Default is auto-generated from column names.
    point_size : int, optional
        Size of scatter points. Default is 3.
    opacity : float, optional
        Opacity of scatter points (0-1). Default is 0.6.
    contour_density : bool, optional
        If True, adds density contours to the plot. Default is True.
    n_contours : int, optional
        Number of contour levels to show if contour_density=True. Default is 10.
    contour_colorscale : str, optional
        Colorscale for contours. Default is 'Viridis'.
    height : int, optional
        Plot height in pixels. Default is 700.
    width : int, optional
        Plot width in pixels. Default is 900.
    show_legend : bool, optional
        Whether to show the legend. Default is True.
    sample_frac : float, optional
        Fraction of data to plot (for very large datasets). If None, plots all data.
        Example: 0.1 plots 10% of the data. Default is None.
    colorscale : str, optional
        Colorscale for continuous color_by values. Options: 'Plotly', 'Viridis', 
        'Cividis', 'Inferno', 'Magma', 'Plasma', 'Turbo'. Default is 'Plotly'.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The interactive Plotly figure. Use fig.show() to display.
        
    Notes
    -----
    How to use:
    1. Run the function to display the interactive plot
    2. Click the "Draw a closed shape" tool in the toolbar (lasso/polygon icon)
    3. Click points to draw your ROI polygon
    4. Double-click to close the polygon
    5. The selected points are automatically highlighted
    6. Use fig.data to access the ROI coordinates and selected data
    
    To extract ROI data programmatically:
    ```python
    # After drawing ROIs, extract the selection:
    from shapely.geometry import Point, Polygon
    
    # Get ROI coordinates from the drawn shape
    # You can access selectedData from the figure or export via browser
    
    # Or use the companion function: extract_roi_data(df, roi_coords, x_col, y_col)
    ```
    
    Examples
    --------
    >>> # Basic interactive plot
    >>> fig = interactive_roi_gating(df, 'PC1', 'PC2')
    >>> fig.show()
    
    >>> # Colored by molecule type with contours
    >>> fig = interactive_roi_gating(df, 'diffusion_coefficient', 'radius_of_gyration',
    ...                               color_by='mol', contour_density=True)
    >>> fig.show()
    
    >>> # With data scaling
    >>> fig = interactive_roi_gating(df, 'x_um', 'y_um',
    ...                               scale_data=True, scale_method='standard')
    >>> fig.show()
    
    >>> # For very large datasets, sample the data
    >>> fig = interactive_roi_gating(df, 'PC1', 'PC2', sample_frac=0.1)
    >>> fig.show()
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError(
            "Plotly is required for interactive gating. "
            "Install with: pip install plotly"
        )
    
    # Create a copy of the dataframe
    plot_df = df.copy()
    
    # Sample data if requested (for performance with large datasets)
    if sample_frac is not None:
        if 0 < sample_frac < 1:
            plot_df = plot_df.sample(frac=sample_frac, random_state=42)
            print(f"📊 Sampled {len(plot_df):,} points ({sample_frac*100:.1f}% of data)")
    
    # Apply scaling if requested
    if scale_data:
        plot_df = center_scale_data(
            plot_df,
            columns=[x_col, y_col],
            method=scale_method,
            copy=False
        )
        print(f"✓ Data scaled using '{scale_method}' method")
    
    # Set default title
    if title is None:
        title = f"{y_col} vs {x_col}"
        if color_by:
            title += f" (colored by {color_by})"
    
    # Create the figure
    fig = go.Figure()
    
    # Determine if color_by is categorical or continuous
    is_categorical = False
    if color_by is not None:
        if plot_df[color_by].dtype == 'object' or plot_df[color_by].nunique() < 20:
            is_categorical = True
    
    # Add scatter plot
    if color_by is None:
        # Single color scatter
        fig.add_trace(go.Scattergl(
            x=plot_df[x_col],
            y=plot_df[y_col],
            mode='markers',
            marker=dict(
                size=point_size,
                color='#636EFA',  # Plotly default blue
                opacity=opacity,
                line=dict(width=0)
            ),
            name='Data',
            showlegend=show_legend,
            hovertemplate=(
                f'{x_col}: %{{x:.3f}}<br>'
                f'{y_col}: %{{y:.3f}}<br>'
                '<extra></extra>'
            )
        ))
    elif is_categorical:
        # Categorical coloring - one trace per category
        categories = sorted(plot_df[color_by].unique())
        colors = px.colors.qualitative.Plotly
        if len(categories) > len(colors):
            colors = px.colors.qualitative.Dark24
        
        for i, cat in enumerate(categories):
            cat_df = plot_df[plot_df[color_by] == cat]
            fig.add_trace(go.Scattergl(
                x=cat_df[x_col],
                y=cat_df[y_col],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=colors[i % len(colors)],
                    opacity=opacity,
                    line=dict(width=0)
                ),
                name=str(cat),
                showlegend=show_legend,
                hovertemplate=(
                    f'{color_by}: {cat}<br>'
                    f'{x_col}: %{{x:.3f}}<br>'
                    f'{y_col}: %{{y:.3f}}<br>'
                    '<extra></extra>'
                )
            ))
    else:
        # Continuous coloring
        fig.add_trace(go.Scattergl(
            x=plot_df[x_col],
            y=plot_df[y_col],
            mode='markers',
            marker=dict(
                size=point_size,
                color=plot_df[color_by],
                colorscale=colorscale,
                opacity=opacity,
                showscale=True,
                colorbar=dict(title=color_by),
                line=dict(width=0)
            ),
            name='Data',
            showlegend=False,
            hovertemplate=(
                f'{x_col}: %{{x:.3f}}<br>'
                f'{y_col}: %{{y:.3f}}<br>'
                f'{color_by}: %{{marker.color:.3f}}<br>'
                '<extra></extra>'
            )
        ))
    
    # Add density contours if requested
    if contour_density:
        try:
            from scipy.stats import gaussian_kde
            
            # Calculate 2D density
            x_data = plot_df[x_col].values
            y_data = plot_df[y_col].values
            
            # Remove NaN values
            mask = ~(np.isnan(x_data) | np.isnan(y_data))
            x_data = x_data[mask]
            y_data = y_data[mask]
            
            if len(x_data) > 10:
                # Create grid for contours
                x_min, x_max = x_data.min(), x_data.max()
                y_min, y_max = y_data.min(), y_data.max()
                x_range = x_max - x_min
                y_range = y_max - y_min
                
                # Add padding
                x_min -= x_range * 0.05
                x_max += x_range * 0.05
                y_min -= y_range * 0.05
                y_max += y_range * 0.05
                
                # Create meshgrid
                x_grid = np.linspace(x_min, x_max, 100)
                y_grid = np.linspace(y_min, y_max, 100)
                xx, yy = np.meshgrid(x_grid, y_grid)
                
                # Calculate density
                positions = np.vstack([xx.ravel(), yy.ravel()])
                kernel = gaussian_kde(np.vstack([x_data, y_data]))
                density = np.reshape(kernel(positions).T, xx.shape)
                
                # Add contour
                fig.add_trace(go.Contour(
                    x=x_grid,
                    y=y_grid,
                    z=density,
                    colorscale=contour_colorscale,
                    opacity=0.4,
                    ncontours=n_contours,
                    showscale=False,
                    hoverinfo='skip',
                    line=dict(width=1),
                    name='Density'
                ))
        except Exception as e:
            print(f"Warning: Could not add density contours: {e}")
    
    # Update layout with drawing tools
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis=dict(
            title=x_col,
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray'
        ),
        yaxis=dict(
            title=y_col,
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray'
        ),
        height=height,
        width=width,
        hovermode='closest',
        dragmode='drawclosedpath',  # Enable polygon drawing by default
        newshape=dict(
            line=dict(color='red', width=3),
            fillcolor='rgba(255, 0, 0, 0.1)',
            opacity=0.5
        ),
        modebar_add=[
            'drawclosedpath',
            'drawopenpath', 
            'drawrect',
            'eraseshape'
        ],
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Add config for better interactivity
    config = {
        'modeBarButtonsToAdd': ['drawclosedpath', 'eraseshape'],
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'{x_col}_vs_{y_col}_gating',
            'height': height,
            'width': width,
            'scale': 2
        }
    }
    
    print("\n" + "="*80)
    print("🎨 INTERACTIVE GATING PLOT")
    print("="*80)
    print(f"📊 Plotting {len(plot_df):,} points")
    print(f"📈 X-axis: {x_col}")
    print(f"📈 Y-axis: {y_col}")
    if color_by:
        print(f"🎨 Colored by: {color_by}")
    print("\n" + "─"*80)
    print("🖱️  HOW TO USE:")
    print("─"*80)
    print("1. Click 'Draw a closed shape' button in the toolbar (top-right)")
    print("2. Click points on the plot to draw your ROI polygon")
    print("3. Double-click to close the polygon")
    print("4. Draw multiple ROIs if needed")
    print("5. Use 'Erase active shape' to remove ROIs")
    print("6. Use extract_roi_data() to get the gated data (see below)")
    print("="*80 + "\n")
    
    # Store the dataframe in the figure for later extraction
    fig._df = plot_df
    fig._x_col = x_col
    fig._y_col = y_col
    
    return fig


def extract_roi_data(fig=None, df=None, roi_coords=None, x_col=None, y_col=None):
    """
    Extract data points within drawn ROI polygons.
    
    Parameters
    ----------
    fig : plotly.graph_objects.Figure, optional
        The figure object returned by interactive_roi_gating(). 
        If provided, uses the stored dataframe and coordinates.
    df : pd.DataFrame, optional
        DataFrame to filter. Required if fig is not provided.
    roi_coords : list of tuples, optional
        List of (x, y) coordinates defining the ROI polygon.
        Required if fig is not provided.
    x_col : str, optional
        Column name for x-axis. Required if fig is not provided.
    y_col : str, optional
        Column name for y-axis. Required if fig is not provided.
    
    Returns
    -------
    pd.DataFrame
        Filtered dataframe containing only points within the ROI(s).
    dict
        Dictionary with ROI statistics and metadata.
    
    Examples
    --------
    >>> # Method 1: After drawing ROIs in the plot, manually provide coordinates
    >>> roi_coords = [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]
    >>> filtered_df, info = extract_roi_data(df=my_df, roi_coords=roi_coords, 
    ...                                       x_col='PC1', y_col='PC2')
    
    >>> # Method 2: With saved ROI coordinates from multiple gates
    >>> gate1 = [(0, 0), (1, 0), (1, 1), (0, 1)]
    >>> gate2 = [(2, 2), (3, 2), (3, 3), (2, 3)]
    >>> df_gate1, _ = extract_roi_data(df=my_df, roi_coords=gate1, x_col='PC1', y_col='PC2')
    >>> df_gate2, _ = extract_roi_data(df=my_df, roi_coords=gate2, x_col='PC1', y_col='PC2')
    """
    try:
        from shapely.geometry import Point, Polygon
    except ImportError:
        raise ImportError(
            "Shapely is required for ROI extraction. "
            "Install with: pip install shapely"
        )
    
    # Get data from figure if provided
    if fig is not None and hasattr(fig, '_df'):
        df = fig._df
        x_col = fig._x_col
        y_col = fig._y_col
    
    # Validate inputs
    if df is None or x_col is None or y_col is None:
        raise ValueError(
            "Must provide either 'fig' from interactive_roi_gating(), "
            "or all of: df, x_col, y_col"
        )
    
    if roi_coords is None:
        print("⚠️  No ROI coordinates provided.")
        print("    After drawing ROIs in the plot, you need to:")
        print("    1. Manually copy the coordinates from the drawn shapes, or")
        print("    2. Use Plotly's relayout_data callback to capture the shape coordinates")
        print("\n    Example:")
        print("    roi_coords = [(x1, y1), (x2, y2), (x3, y3), ...]")
        print("    filtered_df, info = extract_roi_data(df=df, roi_coords=roi_coords,")
        print("                                          x_col='PC1', y_col='PC2')")
        return df.iloc[:0], {}  # Return empty dataframe
    
    # Create polygon from coordinates
    try:
        polygon = Polygon(roi_coords)
    except Exception as e:
        raise ValueError(f"Invalid ROI coordinates: {e}")
    
    # Filter points within polygon
    points = [Point(x, y) for x, y in zip(df[x_col], df[y_col])]
    mask = [polygon.contains(point) for point in points]
    
    filtered_df = df[mask].copy()
    
    # Calculate statistics
    n_total = len(df)
    n_selected = len(filtered_df)
    percent_selected = (n_selected / n_total * 100) if n_total > 0 else 0
    
    info = {
        'n_total': n_total,
        'n_selected': n_selected,
        'percent_selected': percent_selected,
        'roi_coords': roi_coords,
        'x_col': x_col,
        'y_col': y_col
    }
    
    print(f"✅ ROI Filtering Complete")
    print(f"   Total points: {n_total:,}")
    print(f"   Selected points: {n_selected:,} ({percent_selected:.2f}%)")
    
    return filtered_df, info


def save_roi_coordinates(roi_coords, filename, save_path=None):
    """
    Save ROI coordinates to a file for later use.
    
    Parameters
    ----------
    roi_coords : list of tuples
        List of (x, y) coordinates defining the ROI polygon.
    filename : str
        Name for the saved file (without extension).
    save_path : str, optional
        Directory to save the file. If None, saves to current directory.
    
    Examples
    --------
    >>> roi = [(0, 0), (1, 0), (1, 1), (0, 1)]
    >>> save_roi_coordinates(roi, 'my_gate_1', save_path='./gates/')
    """
    import json
    
    if save_path is None:
        save_path = '.'
    
    os.makedirs(save_path, exist_ok=True)
    
    filepath = os.path.join(save_path, f'{filename}.json')
    
    with open(filepath, 'w') as f:
        json.dump(roi_coords, f, indent=2)
    
    print(f"✅ ROI coordinates saved to: {filepath}")


def load_roi_coordinates(filename, save_path=None):
    """
    Load ROI coordinates from a saved file.
    
    Parameters
    ----------
    filename : str
        Name of the saved file (with or without .json extension).
    save_path : str, optional
        Directory where the file is saved. If None, looks in current directory.
    
    Returns
    -------
    list of tuples
        List of (x, y) coordinates defining the ROI polygon.
    
    Examples
    --------
    >>> roi = load_roi_coordinates('my_gate_1', save_path='./gates/')
    >>> filtered_df, _ = extract_roi_data(df=df, roi_coords=roi, x_col='PC1', y_col='PC2')
    """
    import json
    
    if save_path is None:
        save_path = '.'
    
    if not filename.endswith('.json'):
        filename += '.json'
    
    filepath = os.path.join(save_path, filename)
    
    with open(filepath, 'r') as f:
        roi_coords = json.load(f)
    
    print(f"✅ ROI coordinates loaded from: {filepath}")
    print(f"   {len(roi_coords)} vertices")
    
    return roi_coords


def interactive_roi_gating_with_capture(
    df,
    x_col,
    y_col,
    color_by=None,
    scale_data=False,
    scale_method='standard',
    scaler=None,
    roi_manager=None,
    title=None,
    point_size=3,
    opacity=0.6,
    contour_density=True,
    n_contours=10,
    contour_colorscale='Viridis',
    height=700,
    width=900,
    show_legend=True,
    sample_frac=None,
    colorscale='Plotly',
):
    """
    Interactive FACS-style gating plot with automatic ROI coordinate capture.
    
    This enhanced version uses FigureWidget to automatically capture drawn ROI coordinates
    in real-time, making it easy to extract gated data without manual coordinate entry.
    
    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        DataFrame containing the data to plot. Accepts both pandas and polars.
    x_col : str
        Column name for x-axis.
    y_col : str
        Column name for y-axis.
    color_by : str, optional
        Column name to color points by. If None, all points are the same color.
    scale_data : bool, optional
        If True, applies scaling to x_col and y_col before plotting. Default is False.
    scale_method : str, optional
        Scaling method if scale_data=True. Options: 'standard', 'minmax', 
        'robust', 'standard_minmax'. Default is 'standard'.
    scaler : sklearn scaler object, optional
        Pre-fitted scaler to use for transforming data. If provided, this scaler
        will be stored in the ROIManager and used for on-the-fly transformations
        when classifying new data. This allows gates defined in scaled space to
        be applied to unscaled data efficiently.
    roi_manager : ROIManager, optional
        Pre-initialized ROIManager object to use. If provided, this manager will
        be used instead of creating a new one. This allows you to set up gates
        programmatically before displaying the plot, or to reuse the same manager
        across multiple plots. Default is None (creates a new manager).
    title : str, optional
        Plot title. Default is auto-generated.
    point_size : int, optional
        Size of scatter points. Default is 3.
    opacity : float, optional
        Opacity of scatter points (0-1). Default is 0.6.
    contour_density : bool, optional
        If True, adds density contours. Default is True.
    n_contours : int, optional
        Number of contour levels if contour_density=True. Default is 10.
    contour_colorscale : str, optional
        Colorscale for contours. Default is 'Viridis'.
    height : int, optional
        Plot height in pixels. Default is 700.
    width : int, optional
        Plot width in pixels. Default is 900.
    show_legend : bool, optional
        Whether to show legend. Default is True.
    sample_frac : float, optional
        Fraction of data to plot (for large datasets). Default is None.
    colorscale : str, optional
        Colorscale for continuous color_by values. Default is 'Plotly'.
    
    Returns
    -------
    fig : plotly.graph_objects.FigureWidget
        Interactive FigureWidget with ROI capture capabilities.
    roi_manager : ROIManager
        Manager object containing captured ROI coordinates and extraction methods.
        Access via: roi_manager.get_all_rois(), roi_manager.extract_data()
    
    Examples
    --------
    >>> # Example 1: Basic usage with automatic ROI manager creation
    >>> fig, roi_manager = spt.interactive_roi_gating_with_capture(
    ...     df, 'x', 'y', color_by='mol', scale_data=True
    ... )
    >>> display(fig)
    >>> 
    >>> # Example 2: With fitted scaler
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.pipeline import Pipeline
    >>> scaler = Pipeline([('std', StandardScaler()), ('minmax', MinMaxScaler())])
    >>> scaler.fit(df[['x', 'y']])
    >>> fig, roi_manager = spt.interactive_roi_gating_with_capture(
    ...     df, 'x', 'y', color_by='mol', scale_data=True, scaler=scaler
    ... )
    >>> display(fig)
    >>> 
    >>> # Example 3: Pre-create ROI manager and add gates programmatically
    >>> # Create manager separately
    >>> roi_manager = spt.ROIManager(df, 'x', 'y', scaler=scaler)
    >>> # Add gates before plotting
    >>> roi_manager.add_rectangle_gate(x_min=0.0, x_max=0.3, y_min=0.0, y_max=0.2, name='Gate1')
    >>> roi_manager.add_rectangle_gate(x_min=0.5, x_max=1.0, y_min=0.5, y_max=1.0, name='Gate2')
    >>> # Now create the plot with the pre-configured manager
    >>> fig, roi_manager = spt.interactive_roi_gating_with_capture(
    ...     df, 'x', 'y', color_by='mol', scale_data=True, roi_manager=roi_manager
    ... )
    >>> display(fig)
    >>> 
    >>> # After drawing (or using pre-defined gates), get all ROIs
    >>> rois = roi_manager.get_all_rois()
    >>> print(f"Captured {len(rois)} ROIs")
    >>> 
    >>> # Extract data for a specific ROI
    >>> gated_df = roi_manager.extract_data(roi_index=0)
    >>> 
    >>> # Or extract data for all ROIs
    >>> all_gated_data = roi_manager.extract_all_data()
    >>> 
    >>> # Apply gates to NEW data (scaler transforms automatically!)
    >>> df_classified = roi_manager.classify_data(new_df)
    >>> 
    >>> # Save ROIs
    >>> roi_manager.save_all_rois('my_gates', save_path='./gates/')
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        raise ImportError(
            "Plotly is required for interactive gating. "
            "Install with: pip install plotly"
        )
    
    # Detect and convert Polars DataFrames
    try:
        import polars as pl
        is_polars_input = isinstance(df, pl.DataFrame)
    except ImportError:
        is_polars_input = False
    
    if is_polars_input:
        # Convert to pandas for Plotly
        df_pandas = df.to_pandas()
    else:
        df_pandas = df
    
    # Create or use provided ROI manager
    if roi_manager is None:
        # Create ROI manager to store captured coordinates
        # If a scaler is provided, store it for on-the-fly transformations
        # Pass the original df (can be polars or pandas) - ROIManager handles both
        roi_manager = ROIManager(df, x_col, y_col, scaler=scaler)
    else:
        # Use provided ROI manager, but update its dataframe and columns
        roi_manager.df = df
        roi_manager.x_col = x_col
        roi_manager.y_col = y_col
        # Update scaler if one was provided (otherwise keep the manager's scaler)
        if scaler is not None:
            roi_manager.scaler = scaler
    
    # Create a copy and apply scaling if requested
    plot_df = df_pandas.copy()
    
    if sample_frac is not None:
        if 0 < sample_frac < 1:
            plot_df = plot_df.sample(frac=sample_frac, random_state=42)
            print(f"📊 Sampled {len(plot_df):,} points ({sample_frac*100:.1f}% of data)")
    
    if scale_data:
        # If a scaler was provided, use it; otherwise use center_scale_data
        if scaler is not None:
            plot_df[[x_col, y_col]] = scaler.transform(plot_df[[x_col, y_col]])
            print(f"✓ Data scaled using provided scaler")
        else:
            plot_df = center_scale_data(
                plot_df,
                columns=[x_col, y_col],
                method=scale_method,
                copy=False
            )
            print(f"✓ Data scaled using '{scale_method}' method")
    
    roi_manager._plot_df = plot_df  # Store for later extraction
    
    # Set default title
    if title is None:
        title = f"{y_col} vs {x_col}"
        if color_by:
            title += f" (colored by {color_by})"
    
    # Create FigureWidget instead of Figure for callbacks
    fig = go.FigureWidget()
    
    # Determine if color_by is categorical or continuous
    is_categorical = False
    if color_by is not None:
        if plot_df[color_by].dtype == 'object' or plot_df[color_by].nunique() < 20:
            is_categorical = True
    
    # Add scatter plot (same as before)
    if color_by is None:
        fig.add_trace(go.Scattergl(
            x=plot_df[x_col],
            y=plot_df[y_col],
            mode='markers',
            marker=dict(
                size=point_size,
                color='#636EFA',
                opacity=opacity,
                line=dict(width=0)
            ),
            name='Data',
            showlegend=show_legend,
            hovertemplate=(
                f'{x_col}: %{{x:.3f}}<br>'
                f'{y_col}: %{{y:.3f}}<br>'
                '<extra></extra>'
            )
        ))
    elif is_categorical:
        categories = sorted(plot_df[color_by].unique())
        colors = px.colors.qualitative.Plotly
        if len(categories) > len(colors):
            colors = px.colors.qualitative.Dark24
        
        for i, cat in enumerate(categories):
            cat_df = plot_df[plot_df[color_by] == cat]
            fig.add_trace(go.Scattergl(
                x=cat_df[x_col],
                y=cat_df[y_col],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=colors[i % len(colors)],
                    opacity=opacity,
                    line=dict(width=0)
                ),
                name=str(cat),
                showlegend=show_legend,
                hovertemplate=(
                    f'{color_by}: {cat}<br>'
                    f'{x_col}: %{{x:.3f}}<br>'
                    f'{y_col}: %{{y:.3f}}<br>'
                    '<extra></extra>'
                )
            ))
    else:
        fig.add_trace(go.Scattergl(
            x=plot_df[x_col],
            y=plot_df[y_col],
            mode='markers',
            marker=dict(
                size=point_size,
                color=plot_df[color_by],
                colorscale=colorscale,
                opacity=opacity,
                showscale=True,
                colorbar=dict(title=color_by),
                line=dict(width=0)
            ),
            name='Data',
            showlegend=False,
            hovertemplate=(
                f'{x_col}: %{{x:.3f}}<br>'
                f'{y_col}: %{{y:.3f}}<br>'
                f'{color_by}: %{{marker.color:.3f}}<br>'
                '<extra></extra>'
            )
        ))
    
    # Add density contours if requested
    if contour_density:
        try:
            from scipy.stats import gaussian_kde
            
            x_data = plot_df[x_col].values
            y_data = plot_df[y_col].values
            
            mask = ~(np.isnan(x_data) | np.isnan(y_data))
            x_data = x_data[mask]
            y_data = y_data[mask]
            
            if len(x_data) > 10:
                x_min, x_max = x_data.min(), x_data.max()
                y_min, y_max = y_data.min(), y_data.max()
                x_range = x_max - x_min
                y_range = y_max - y_min
                
                x_min -= x_range * 0.05
                x_max += x_range * 0.05
                y_min -= y_range * 0.05
                y_max += y_range * 0.05
                
                x_grid = np.linspace(x_min, x_max, 100)
                y_grid = np.linspace(y_min, y_max, 100)
                xx, yy = np.meshgrid(x_grid, y_grid)
                
                positions = np.vstack([xx.ravel(), yy.ravel()])
                kernel = gaussian_kde(np.vstack([x_data, y_data]))
                density = np.reshape(kernel(positions).T, xx.shape)
                
                fig.add_trace(go.Contour(
                    x=x_grid,
                    y=y_grid,
                    z=density,
                    colorscale=contour_colorscale,
                    opacity=0.4,
                    ncontours=n_contours,
                    showscale=False,
                    hoverinfo='skip',
                    line=dict(width=1),
                    name='Density'
                ))
        except Exception as e:
            print(f"Warning: Could not add density contours: {e}")
    
    # Update layout with drawing tools
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis=dict(
            title=x_col,
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray'
        ),
        yaxis=dict(
            title=y_col,
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray'
        ),
        height=height,
        width=width,
        hovermode='closest',
        dragmode='drawclosedpath',
        newshape=dict(
            line=dict(color='red', width=3),
            fillcolor='rgba(255, 0, 0, 0.1)',
            opacity=0.5
        ),
        modebar_add=[
            'drawclosedpath',
            'drawopenpath', 
            'drawrect',
            'eraseshape'
        ],
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Add callback to capture drawn shapes
    def on_relayout(layout, shapes_data):
        """Callback function to capture ROI coordinates when shapes are drawn."""
        if 'shapes' in layout:
            roi_manager.update_rois(layout['shapes'])
    
    fig.layout.on_change(on_relayout, 'shapes')
    
    # Store figure in manager for manual capture
    roi_manager._fig = fig
    
    print("\n" + "="*80)
    print("🎨 INTERACTIVE GATING PLOT WITH AUTO-CAPTURE")
    print("="*80)
    print(f"📊 Plotting {len(plot_df):,} points")
    print(f"📈 X-axis: {x_col}")
    print(f"📈 Y-axis: {y_col}")
    if color_by:
        print(f"🎨 Colored by: {color_by}")
    print("\n" + "─"*80)
    print("🖱️  HOW TO USE:")
    print("─"*80)
    print("1. Click 'Draw rectangle' or 'Draw closed shape' in the toolbar")
    print("2. Draw your ROI on the plot:")
    print("   - Rectangle: Click and drag")
    print("   - Polygon: Click points, double-click to close")
    print("3. After drawing, run: roi_manager.capture_rois()")
    print("4. Then extract data: roi_manager.extract_all_data()")
    print("─"*80)
    print("💡 TIP: If auto-capture doesn't work, use roi_manager.capture_rois()")
    print("="*80 + "\n")
    
    return fig, roi_manager


class ROIManager:
    """
    Manager class for storing and manipulating captured ROI coordinates.
    
    Supports on-the-fly data scaling using a fitted scaler object. This allows
    gates to be defined in scaled space while working with unscaled data, without
    needing to store scaled columns in the dataframe.
    """
    
    def __init__(self, df, x_col, y_col, scaler=None):
        """
        Initialize ROIManager.
        
        Parameters
        ----------
        df : pd.DataFrame or pl.DataFrame
            Input dataframe
        x_col : str
            Column name for x-axis
        y_col : str
            Column name for y-axis
        scaler : sklearn scaler object, optional
            Pre-fitted scaler for transforming data (e.g., fitted StandardScaler
            or custom pipeline). If provided, gates are assumed to be in scaled
            space, and data will be transformed before checking gate membership.
            The scaler must have a .transform() method and be already fitted.
        """
        self.df = df
        self.x_col = x_col
        self.y_col = y_col
        self.scaler = scaler  # Store the fitted scaler
        self._plot_df = df  # Will be updated with scaled/sampled data
        self._fig = None  # Will store the figure for manual capture
        self.rois = []
    
    def capture_rois(self):
        """
        Manually capture ROIs from the figure.
        
        Use this method after drawing your ROIs if automatic capture doesn't work.
        This is the most reliable method to ensure your ROIs are captured.
        
        Returns
        -------
        int
            Number of ROIs captured.
        
        Examples
        --------
        >>> # After drawing ROIs on the plot
        >>> roi_manager.capture_rois()
        ✅ Captured 2 ROI(s)
        """
        if self._fig is None:
            print("❌ No figure available for capture.")
            print("   Make sure you ran the cell that creates the figure first.")
            return 0
        
        # Access shapes from figure layout
        try:
            shapes = self._fig.layout.shapes
            
            # Convert tuple to list if needed
            if shapes is None:
                shapes_list = []
            elif isinstance(shapes, tuple):
                shapes_list = list(shapes)
            else:
                shapes_list = shapes
                
            if len(shapes_list) > 0:
                print(f"🔍 Found {len(shapes_list)} shape(s) in the figure...")
                self.update_rois(shapes_list)
                return len(self.rois)
            else:
                print("⚠️  No shapes found in the figure yet.")
                print("\n📝 INSTRUCTIONS:")
                print("   1. Look at the plot above")
                print("   2. Find the toolbar at the TOP-RIGHT")
                print("   3. Click the 'Draw rectangle' button (looks like a box 📦)")
                print("   4. Click and DRAG on the plot to draw a rectangle")
                print("   5. The rectangle should appear (might be faint)")
                print("   6. Run this cell again to capture it")
                print("\n💡 TIP: The shapes are stored in fig.layout.shapes")
                print(f"   Current shapes: {self._fig.layout.shapes}")
                return 0
                
        except Exception as e:
            print(f"❌ Error accessing shapes: {e}")
            print(f"   Figure type: {type(self._fig)}")
            print(f"   Has layout: {hasattr(self._fig, 'layout')}")
            if hasattr(self._fig, 'layout'):
                print(f"   Layout type: {type(self._fig.layout)}")
                print(f"   Has shapes: {hasattr(self._fig.layout, 'shapes')}")
                if hasattr(self._fig.layout, 'shapes'):
                    print(f"   Shapes: {self._fig.layout.shapes}")
            return 0
        
    def update_rois(self, shapes):
        """Update ROIs from Plotly shapes."""
        self.rois = []
        for shape in shapes:
            if shape['type'] in ['rect', 'path']:
                coords = self._extract_coordinates(shape)
                if coords:
                    self.rois.append({
                        'type': shape['type'],
                        'coordinates': coords,
                        'shape': shape
                    })
        print(f"✅ Captured {len(self.rois)} ROI(s)")
    
    def _extract_coordinates(self, shape):
        """Extract coordinates from a Plotly shape."""
        if shape['type'] == 'rect':
            # Rectangle: convert to polygon coordinates
            x0, y0 = shape['x0'], shape['y0']
            x1, y1 = shape['x1'], shape['y1']
            return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        
        elif shape['type'] == 'path':
            # Parse SVG path string
            path_str = shape['path']
            coords = []
            
            # Simple parser for M (move) and L (line) commands
            import re
            # Extract M and L commands with coordinates
            commands = re.findall(r'[ML]\s*([-\d.]+)[,\s]+([-\d.]+)', path_str)
            for cmd, x, y in commands:
                coords.append((float(x), float(y)))
            
            return coords if coords else None
        
        return None
    
    def get_all_rois(self):
        """Get list of all captured ROI coordinates."""
        return [roi['coordinates'] for roi in self.rois]
    
    def get_roi(self, roi_index):
        """Get specific ROI by index."""
        if 0 <= roi_index < len(self.rois):
            return self.rois[roi_index]['coordinates']
        else:
            raise IndexError(f"ROI index {roi_index} out of range (0-{len(self.rois)-1})")
    
    def extract_data(self, roi_index=0, use_original_df=False):
        """
        Extract data points within a specific ROI.
        
        Parameters
        ----------
        roi_index : int
            Index of the ROI to extract (0-based). Default is 0.
        use_original_df : bool
            If True, applies ROI to original unscaled/unsampled df.
            If False, applies to the plot dataframe. Default is False.
        
        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Filtered dataframe with points inside the ROI. Returns same type as input.
        """
        try:
            from shapely.geometry import Point, Polygon
        except ImportError:
            raise ImportError("Shapely required: pip install shapely")
        
        roi_coords = self.get_roi(roi_index)
        polygon = Polygon(roi_coords)
        
        target_df = self.df if use_original_df else self._plot_df
        
        # Detect if target is Polars DataFrame
        try:
            import polars as pl
            is_polars = isinstance(target_df, pl.DataFrame)
        except ImportError:
            is_polars = False
        
        # Get x and y values
        if is_polars:
            x_vals = target_df[self.x_col].to_list()
            y_vals = target_df[self.y_col].to_list()
        else:
            x_vals = target_df[self.x_col]
            y_vals = target_df[self.y_col]
        
        # Create mask
        points = [Point(x, y) for x, y in zip(x_vals, y_vals)]
        mask = [polygon.contains(point) for point in points]
        
        # Filter based on dataframe type
        if is_polars:
            # Polars requires .filter() method with a Series
            import polars as pl
            mask_series = pl.Series("mask", mask)
            filtered_df = target_df.filter(mask_series)
        else:
            # Pandas supports boolean indexing
            filtered_df = target_df[mask].copy()
        
        print(f"✅ ROI {roi_index}: {len(filtered_df):,} / {len(target_df):,} points "
              f"({len(filtered_df)/len(target_df)*100:.1f}%)")
        
        return filtered_df
    
    def extract_all_data(self, use_original_df=False):
        """
        Extract data for all ROIs.
        
        Returns
        -------
        list of pd.DataFrame
            List of filtered dataframes, one per ROI.
        """
        return [self.extract_data(i, use_original_df) for i in range(len(self.rois))]
    
    def save_all_rois(self, base_filename, save_path=None):
        """Save all ROI coordinates to separate files."""
        if save_path is None:
            save_path = '.'
        
        os.makedirs(save_path, exist_ok=True)
        
        for i, roi_coords in enumerate(self.get_all_rois()):
            filename = f"{base_filename}_roi{i}"
            save_roi_coordinates(roi_coords, filename, save_path)
        
        print(f"✅ Saved {len(self.rois)} ROI(s) to {save_path}")
    
    def show_shapes(self):
        """
        Debug method to show what shapes are currently in the figure.
        
        Use this to troubleshoot if capture_rois() isn't finding your shapes.
        """
        if self._fig is None:
            print("No figure stored.")
            return
        
        print("\n" + "="*80)
        print("DEBUG: Figure Shapes Information")
        print("="*80)
        print(f"Figure type: {type(self._fig)}")
        print(f"Has layout: {hasattr(self._fig, 'layout')}")
        
        if hasattr(self._fig, 'layout'):
            print(f"Has shapes attribute: {hasattr(self._fig.layout, 'shapes')}")
            if hasattr(self._fig.layout, 'shapes'):
                shapes = self._fig.layout.shapes
                print(f"Shapes type: {type(shapes)}")
                print(f"Number of shapes: {len(shapes) if shapes else 0}")
                if shapes:
                    print("\nShapes content:")
                    for i, shape in enumerate(shapes):
                        print(f"\n  Shape {i}:")
                        print(f"    Type: {shape.get('type', 'unknown')}")
                        if shape.get('type') == 'rect':
                            print(f"    Coordinates: ({shape.get('x0')}, {shape.get('y0')}) to ({shape.get('x1')}, {shape.get('y1')})")
                        elif shape.get('type') == 'path':
                            print(f"    Path: {shape.get('path', 'N/A')[:100]}...")
                else:
                    print("  Shapes is None or empty")
        print("="*80 + "\n")
    
    def add_rectangle_gate(self, x_min, x_max, y_min, y_max, name=None):
        """
        Manually add a rectangular gate by specifying its bounds.
        
        This is the SIMPLE, WORKING alternative to interactive drawing.
        
        Parameters
        ----------
        x_min : float
            Minimum x coordinate
        x_max : float
            Maximum x coordinate  
        y_min : float
            Minimum y coordinate
        y_max : float
            Maximum y coordinate
        name : str, optional
            Name for this gate
            
        Examples
        --------
        >>> # Look at your plot and identify the rectangle bounds
        >>> roi_manager.add_rectangle_gate(x_min=0.0, x_max=0.3, y_min=0.0, y_max=0.2)
        >>> # Add another gate
        >>> roi_manager.add_rectangle_gate(x_min=0.5, x_max=1.0, y_min=0.5, y_max=1.0)
        >>> # Extract data
        >>> all_data = roi_manager.extract_all_data()
        """
        coords = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        
        gate = {
            'type': 'rect',
            'coordinates': coords,
            'shape': {
                'type': 'rect',
                'x0': x_min,
                'y0': y_min,
                'x1': x_max,
                'y1': y_max
            }
        }
        
        if name:
            gate['name'] = name
            
        self.rois.append(gate)
        
        print(f"✅ Added rectangular gate: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}]")
        print(f"   Total gates: {len(self.rois)}")
        return len(self.rois) - 1  # Return index of added gate
    
    def clear_gates(self):
        """Remove all gates."""
        n = len(self.rois)
        self.rois = []
        print(f"🗑️  Cleared {n} gate(s)")
    
    def add_polygon_gate(self, coordinates, name=None):
        """
        Manually add a polygonal gate by specifying its vertices.
        
        This allows you to define arbitrary polygon shapes programmatically.
        
        Parameters
        ----------
        coordinates : list of tuples
            List of (x, y) coordinate tuples defining the polygon vertices.
            Example: [(0.1, 0.2), (0.5, 0.2), (0.5, 0.8), (0.1, 0.8)]
        name : str, optional
            Name for this gate
            
        Examples
        --------
        >>> # Triangle gate
        >>> roi_manager.add_polygon_gate([(0, 0), (1, 0), (0.5, 1)], name='triangle')
        >>> 
        >>> # Complex polygon
        >>> coords = [(0.1, 0.2), (0.5, 0.1), (0.8, 0.5), (0.4, 0.9), (0.1, 0.6)]
        >>> roi_manager.add_polygon_gate(coords, name='pentagonal_gate')
        """
        if len(coordinates) < 3:
            raise ValueError("Polygon must have at least 3 vertices")
        
        gate = {
            'type': 'path',
            'coordinates': coordinates,
            'shape': {
                'type': 'path',
                'path': self._coords_to_svg_path(coordinates)
            }
        }
        
        if name:
            gate['name'] = name
            
        self.rois.append(gate)
        
        print(f"✅ Added polygon gate with {len(coordinates)} vertices")
        if name:
            print(f"   Name: {name}")
        print(f"   Total gates: {len(self.rois)}")
        return len(self.rois) - 1  # Return index of added gate
    
    def _coords_to_svg_path(self, coordinates):
        """Convert coordinate list to SVG path string."""
        if not coordinates:
            return ""
        path_parts = [f"M {coordinates[0][0]},{coordinates[0][1]}"]
        for x, y in coordinates[1:]:
            path_parts.append(f"L {x},{y}")
        path_parts.append("Z")  # Close the path
        return " ".join(path_parts)
    
    def add_diagonal_separator(self, point1, point2, xlim, ylim, 
                                buffer=0.01, names=None):
        """
        Add two ROIs separated by a diagonal line.
        
        This creates two polygonal regions: one "below/left" and one "above/right" 
        of the diagonal line you specify. Perfect for separating two populations!
        
        Parameters
        ----------
        point1 : tuple
            (x, y) coordinates of first point on the line
        point2 : tuple
            (x, y) coordinates of second point on the line
        xlim : tuple
            (xmin, xmax) limits of your plot
        ylim : tuple
            (ymin, ymax) limits of your plot
        buffer : float, optional
            Small buffer distance perpendicular to the line (prevents overlap).
            Default is 0.01. Set to 0 for no gap.
        names : tuple of str, optional
            (name_below, name_above) - Names for the two gates.
            Default is ('below_line', 'above_line')
            
        Returns
        -------
        tuple
            (index_below, index_above) - Indices of the two created gates
            
        Examples
        --------
        >>> # Simple diagonal from bottom-left to top-right
        >>> roi_manager.add_diagonal_separator(
        ...     point1=(0.0, 0.0), 
        ...     point2=(1.0, 1.0),
        ...     xlim=(0, 1),
        ...     ylim=(0, 1),
        ...     names=('low_pop', 'high_pop')
        ... )
        >>> 
        >>> # Custom diagonal after looking at your plot
        >>> roi_manager.add_diagonal_separator(
        ...     point1=(0.2, 0.1), 
        ...     point2=(0.8, 0.9),
        ...     xlim=(0, 1),
        ...     ylim=(0, 1),
        ...     buffer=0.02,
        ...     names=('immobile', 'mobile')
        ... )
        >>> 
        >>> # Extract the separated populations
        >>> pop_below = roi_manager.extract_data(roi_index=0)
        >>> pop_above = roi_manager.extract_data(roi_index=1)
        """
        x1, y1 = point1
        x2, y2 = point2
        xmin, xmax = xlim
        ymin, ymax = ylim
        
        if names is None:
            names = ('below_line', 'above_line')
        
        # Calculate perpendicular offset vector for buffer
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        if length == 0:
            raise ValueError("point1 and point2 must be different")
        
        # Perpendicular unit vector (rotated 90 degrees)
        perp_x = -dy / length
        perp_y = dx / length
        
        # Create buffered line points
        # Line offset "below" (negative perpendicular direction)
        line_below = [
            (x1 - buffer * perp_x, y1 - buffer * perp_y),
            (x2 - buffer * perp_x, y2 - buffer * perp_y)
        ]
        
        # Line offset "above" (positive perpendicular direction)
        line_above = [
            (x1 + buffer * perp_x, y1 + buffer * perp_y),
            (x2 + buffer * perp_x, y2 + buffer * perp_y)
        ]
        
        # Create polygon for "below" region
        # Strategy: Find where line intersects the plot boundaries and build polygon
        below_coords = self._create_half_plane_polygon(
            line_below, xlim, ylim, side='below', 
            perp_vector=(-perp_x, -perp_y)
        )
        
        # Create polygon for "above" region
        above_coords = self._create_half_plane_polygon(
            line_above, xlim, ylim, side='above',
            perp_vector=(perp_x, perp_y)
        )
        
        # Add both gates
        idx_below = self.add_polygon_gate(below_coords, name=names[0])
        idx_above = self.add_polygon_gate(above_coords, name=names[1])
        
        print(f"\n📐 Diagonal separator created:")
        print(f"   Line from ({x1:.3f}, {y1:.3f}) to ({x2:.3f}, {y2:.3f})")
        print(f"   Gate '{names[0]}': {len(below_coords)} vertices (index {idx_below})")
        print(f"   Gate '{names[1]}': {len(above_coords)} vertices (index {idx_above})")
        
        return idx_below, idx_above
    
    def _create_half_plane_polygon(self, line_points, xlim, ylim, side, perp_vector):
        """
        Create polygon for half-plane on one side of a line.
        
        Parameters
        ----------
        line_points : list of 2 tuples
            [(x1, y1), (x2, y2)] defining the separator line
        xlim : tuple
            (xmin, xmax)
        ylim : tuple
            (ymin, ymax)
        side : str
            'below' or 'above'
        perp_vector : tuple
            (px, py) perpendicular vector pointing toward the half-plane
        """
        xmin, xmax = xlim
        ymin, ymax = ylim
        (x1, y1), (x2, y2) = line_points
        px, py = perp_vector
        
        # Get corners of plot area
        corners = [
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax)
        ]
        
        # Determine which side of the line each corner is on
        # Using cross product: (point - line_start) × (line_end - line_start)
        def point_side(px, py):
            """Positive if left of line, negative if right"""
            return (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        
        # Collect corners on the correct side
        polygon_points = []
        
        # Add line endpoints first
        polygon_points.append((x1, y1))
        polygon_points.append((x2, y2))
        
        # Add corners that are on the correct side
        target_sign = 1 if side == 'above' else -1
        
        for corner in corners:
            side_val = point_side(corner[0], corner[1])
            if (side_val * target_sign) >= 0:  # Same sign or zero
                polygon_points.append(corner)
        
        # Sort points to form a proper polygon (counterclockwise)
        # Calculate centroid
        cx = sum(p[0] for p in polygon_points) / len(polygon_points)
        cy = sum(p[1] for p in polygon_points) / len(polygon_points)
        
        # Sort by angle from centroid
        def angle_from_centroid(point):
            return np.arctan2(point[1] - cy, point[0] - cx)
        
        polygon_points.sort(key=angle_from_centroid)
        
        return polygon_points
    
    def _transform_data(self, df_pd, x_col, y_col):
        """
        Apply scaler transformation to x and y columns if scaler is available.
        
        Parameters
        ----------
        df_pd : pd.DataFrame
            Input dataframe (pandas)
        x_col : str
            X column name
        y_col : str
            Y column name
            
        Returns
        -------
        x_transformed, y_transformed : arrays
            Transformed x and y values (or original if no scaler)
        """
        if self.scaler is None:
            return df_pd[x_col].values, df_pd[y_col].values
        
        # Transform using the scaler
        # Scaler expects 2D array
        data_to_transform = df_pd[[x_col, y_col]].values
        transformed = self.scaler.transform(data_to_transform)
        
        return transformed[:, 0], transformed[:, 1]
    
    def get_inverse_transformed_gates(self):
        """
        Get gate coordinates in original (unscaled) space.
        
        This is useful for plotting gates on unscaled data. If no scaler is set,
        returns gates as-is.
        
        Returns
        -------
        list of dict
            List of gate dictionaries with inverse-transformed coordinates.
            
        Examples
        --------
        >>> # Get gates in original space for plotting on unscaled data
        >>> unscaled_gates = roi_manager.get_inverse_transformed_gates()
        >>> fig = spt.plot_xy_heatmap(df, 'x', 'y', gates=unscaled_gates, scale_data=False)
        """
        if self.scaler is None:
            # No transformation needed
            return self.rois
        
        # Check if scaler has inverse_transform method
        if not hasattr(self.scaler, 'inverse_transform'):
            print("Warning: scaler does not have inverse_transform method. Returning gates as-is.")
            return self.rois
        
        # Inverse transform each gate's coordinates
        inverse_gates = []
        for gate in self.rois:
            coords = gate.get('coordinates', [])
            if not coords:
                inverse_gates.append(gate)
                continue
            
            # Convert coordinates to array and inverse transform
            coords_array = np.array(coords)
            coords_inverse = self.scaler.inverse_transform(coords_array)
            
            # Create new gate dict with inverse coordinates
            inverse_gate = gate.copy()
            inverse_gate['coordinates'] = [(x, y) for x, y in coords_inverse]
            
            # Update shape if it's a rectangle
            if gate.get('type') == 'rect' and len(coords_inverse) >= 4:
                inverse_gate['shape'] = {
                    'type': 'rect',
                    'x0': coords_inverse[0, 0],
                    'y0': coords_inverse[0, 1],
                    'x1': coords_inverse[2, 0],
                    'y1': coords_inverse[2, 1]
                }
            
            inverse_gates.append(inverse_gate)
        
        return inverse_gates
    
    def classify_data(self, df, x_col=None, y_col=None, gate_col_name='gate_id', 
                     gate_name_col='gate_name', return_type=None,
                     parent_gate_col=None, parent_gate_ids=None, 
                     not_applicable_label='parent_gate_filtered'):
        """
        Classify each row of a dataframe by which gate it belongs to.
        
        Adds two new columns with gate classifications:
        - gate_col_name: integer gate ID (0, 1, 2, ...) or -1 if not in any gate
        - gate_name_col: gate name (string) or 'ungated' if not in any gate
        
        If a scaler is set, data will be transformed on-the-fly before checking
        gate membership (no need to pre-scale the data).
        
        For sequential/hierarchical gating, use different column names for each round
        (e.g., 'gate_id', 'gate2_id', 'gate3_id') and specify parent gates to 
        distinguish between "ungated" and "not applicable".
        
        Parameters
        ----------
        df : pd.DataFrame or pl.DataFrame
            Input dataframe to classify (in original/unscaled space if using scaler)
        x_col : str, optional
            Column name for x coordinate. If None, uses self.x_col
        y_col : str, optional
            Column name for y coordinate. If None, uses self.y_col
        gate_col_name : str, optional
            Name for the gate ID column. Default is 'gate_id'.
            Use different names for multiple rounds: 'gate2_id', 'gate3_id', etc.
        gate_name_col : str, optional
            Name for the gate name column. Default is 'gate_name'.
            Use different names for multiple rounds: 'gate2_name', 'gate3_name', etc.
        return_type : {'pandas', 'polars', None}, optional
            Return type. If None, returns same type as input.
        parent_gate_col : str, optional
            Column name of parent gate (e.g., 'gate_id') for hierarchical gating.
            If specified, only rows matching parent_gate_ids will be evaluated.
            Others get not_applicable_label. Default is None (no hierarchical gating).
        parent_gate_ids : int, list of int, or None, optional
            Parent gate ID(s) to filter by. Only rows with these parent gate values
            will be evaluated for current gates. Can be single int or list.
            Default is None (evaluate all rows).
        not_applicable_label : str, optional
            Label for rows that don't match parent gate filter. Default is 
            'parent_gate_filtered'. Use this to distinguish from 'ungated'.
            
        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Dataframe with added gate classification columns
            
        Examples
        --------
        >>> # Round 1: Gate on speed vs intersections
        >>> df = roi_manager1.classify_data(df)
        >>> # Adds: gate_id (0, 1, -1), gate_name
        >>> 
        >>> # Round 2: Hierarchical gating ONLY on gate_id==0 population
        >>> df = roi_manager2.classify_data(
        ...     df, 
        ...     gate_col_name='gate2_id',
        ...     gate_name_col='gate2_name',
        ...     parent_gate_col='gate_id',    # ← Check parent gate
        ...     parent_gate_ids=0              # ← Only gate rows where gate_id==0
        ... )
        >>> # Result:
        >>> #   - Rows with gate_id==0: get gate2_id (0, 1, -1) and gate2_name
        >>> #   - Rows with gate_id!=0: get gate2_id=-1, gate2_name='parent_gate_filtered'
        >>> 
        >>> # Round 2 alternative: Gate on multiple parent populations
        >>> df = roi_manager2.classify_data(
        ...     df,
        ...     gate_col_name='gate2_id',
        ...     gate_name_col='gate2_name', 
        ...     parent_gate_col='gate_id',
        ...     parent_gate_ids=[0, 1]        # ← Gate both populations 0 and 1
        ... )
        >>> 
        >>> # Now analyze:
        >>> # These are in parent gate 0 but not in any gate2
        >>> ungated_in_parent = df[df['gate2_name'] == 'ungated']
        >>> # These were filtered out by parent gate
        >>> not_applicable = df[df['gate2_name'] == 'parent_gate_filtered']
        """
        from shapely.geometry import Point, Polygon
        
        # Detect input type
        try:
            import polars as pl
            is_polars = isinstance(df, pl.DataFrame)
        except ImportError:
            is_polars = False
        
        # Convert to pandas for processing
        if is_polars:
            df_pd = df.to_pandas()
        else:
            df_pd = df.copy()
        
        # Use provided columns or fall back to stored ones
        x_col = x_col or self.x_col
        y_col = y_col or self.y_col
        
        if x_col not in df_pd.columns or y_col not in df_pd.columns:
            raise ValueError(f"Columns '{x_col}' and/or '{y_col}' not found in dataframe")
        
        # Handle parent gate filtering for hierarchical gating
        if parent_gate_col is not None:
            if parent_gate_col not in df_pd.columns:
                raise ValueError(f"Parent gate column '{parent_gate_col}' not found in dataframe")
            
            # Normalize parent_gate_ids to list
            if parent_gate_ids is None:
                raise ValueError("parent_gate_ids must be specified when using parent_gate_col")
            if not isinstance(parent_gate_ids, (list, tuple)):
                parent_gate_ids = [parent_gate_ids]
            
            # Create mask for rows that match parent gate
            in_parent_mask = df_pd[parent_gate_col].isin(parent_gate_ids)
            
            # Initialize columns with appropriate defaults
            df_pd[gate_col_name] = -1
            df_pd[gate_name_col] = not_applicable_label  # Default for rows NOT in parent
            
            # Update only rows in parent gate to 'ungated'
            df_pd.loc[in_parent_mask, gate_name_col] = 'ungated'
            
            print(f"🔍 Hierarchical gating:")
            print(f"   Parent gate: {parent_gate_col} in {parent_gate_ids}")
            print(f"   Eligible rows: {in_parent_mask.sum():,} / {len(df_pd):,}")
            print(f"   Filtered rows labeled as: '{not_applicable_label}'")
        else:
            # Standard gating: initialize all as ungated
            df_pd[gate_col_name] = -1
            df_pd[gate_name_col] = 'ungated'
            in_parent_mask = None  # Will gate all rows
        
        # Transform data if scaler is available
        x_transformed, y_transformed = self._transform_data(df_pd, x_col, y_col)
        
        # For each gate, check which points are inside
        for gate_idx, gate in enumerate(self.rois):
            coords = gate.get('coordinates', [])
            if not coords:
                continue
            
            # Create polygon from gate coordinates (gates are in transformed space if scaler exists)
            poly = Polygon(coords)
            
            # Check each point using transformed coordinates
            for idx in range(len(df_pd)):
                # Skip if hierarchical gating and not in parent gate
                if in_parent_mask is not None and not in_parent_mask.iloc[idx]:
                    continue
                
                point = Point(x_transformed[idx], y_transformed[idx])
                if poly.contains(point):
                    df_pd.iloc[idx, df_pd.columns.get_loc(gate_col_name)] = gate_idx
                    df_pd.iloc[idx, df_pd.columns.get_loc(gate_name_col)] = gate.get('name', f'Gate {gate_idx}')
        
        # Convert back to polars if requested or if input was polars
        if return_type == 'polars' or (return_type is None and is_polars):
            try:
                import polars as pl
                return pl.from_pandas(df_pd)
            except ImportError:
                print("Warning: polars not available, returning pandas DataFrame")
                return df_pd
        else:
            return df_pd
    
    def get_gate_summary(self, df, x_col=None, y_col=None):
        """
        Get a summary of how many points fall into each gate.
        
        If a scaler is set, data will be transformed on-the-fly before checking
        gate membership.
        
        Parameters
        ----------
        df : pd.DataFrame or pl.DataFrame
            Dataframe to analyze (in original/unscaled space if using scaler)
        x_col : str, optional
            Column name for x coordinate. If None, uses self.x_col
        y_col : str, optional
            Column name for y coordinate. If None, uses self.y_col
            
        Returns
        -------
        dict
            Summary statistics for each gate
            
        Examples
        --------
        >>> summary = roi_manager.get_gate_summary(df)
        >>> print(summary)
        {
            'total_points': 1000,
            'ungated': 250,
            'gates': [
                {'id': 0, 'name': 'low_speed', 'count': 300, 'percent': 30.0},
                {'id': 1, 'name': 'high_speed', 'count': 450, 'percent': 45.0}
            ]
        }
        """
        from shapely.geometry import Point, Polygon
        
        # Detect input type and convert to pandas if needed
        try:
            import polars as pl
            is_polars = isinstance(df, pl.DataFrame)
            if is_polars:
                df_pd = df.to_pandas()
            else:
                df_pd = df
        except ImportError:
            df_pd = df
        
        # Use provided columns or fall back to stored ones
        x_col = x_col or self.x_col
        y_col = y_col or self.y_col
        
        total_points = len(df_pd)
        gate_counts = []
        
        # Transform data if scaler is available
        x_transformed, y_transformed = self._transform_data(df_pd, x_col, y_col)
        
        # Initialize counters for each gate
        for gate_idx, gate in enumerate(self.rois):
            coords = gate.get('coordinates', [])
            if not coords:
                continue
            
            poly = Polygon(coords)
            count = 0
            
            # Count points in this gate using transformed coordinates
            for idx in range(len(df_pd)):
                point = Point(x_transformed[idx], y_transformed[idx])
                if poly.contains(point):
                    count += 1
            
            gate_counts.append({
                'id': gate_idx,
                'name': gate.get('name', f'Gate {gate_idx}'),
                'count': count,
                'percent': (count / total_points * 100) if total_points > 0 else 0
            })
        
        # Calculate ungated
        gated_total = sum(g['count'] for g in gate_counts)
        ungated_count = total_points - gated_total
        
        return {
            'total_points': total_points,
            'ungated': ungated_count,
            'ungated_percent': (ungated_count / total_points * 100) if total_points > 0 else 0,
            'gates': gate_counts
        }
    
    def __repr__(self):
        return f"ROIManager({len(self.rois)} ROIs captured)"



# ============================================================================
# STATE TRANSITION VISUALIZATION FUNCTIONS
# ============================================================================


def plot_transition_matrix(
    transition_result,
    figsize=(10, 8),
    cmap='Blues',
    annot=True,
    fmt='.2f',
    cbar_label='Transition Probability',
    save_path=None,
    export_format='svg',
    dpi=300,
    transparent_background=True,
    title=None,
    show_counts=False,
    state_order=None,
    invert_yaxis=False,
    invert_xaxis=False
):
    """
    Visualize state transition matrix as a heatmap.
    
    Parameters
    ----------
    transition_result : dict
        Result from analyze_state_transitions()
    figsize : tuple, default=(10, 8)
        Figure size
    cmap : str, default='Blues'
        Colormap for heatmap
    annot : bool, default=True
        Whether to annotate cells with values
    fmt : str, default='.2f'
        Format string for annotations
    cbar_label : str, default='Transition Probability'
        Label for colorbar
    save_path : str, optional
        Directory to save figure
    export_format : str, default='svg'
        Export format
    dpi : int, default=300
        DPI for raster formats
    transparent_background : bool, default=True
        Whether to make background transparent
    title : str, optional
        Custom title for plot
    show_counts : bool, default=False
        Show transition counts instead of probabilities
    state_order : list, optional
        List of states in desired order for axes. If None, uses default order.
    invert_yaxis : bool, default=False
        Whether to invert the y-axis (reverse row order)
    invert_xaxis : bool, default=False
        Whether to invert the x-axis (reverse column order)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get the appropriate matrix
    if show_counts:
        matrix = transition_result['transition_counts']
        if title is None:
            title = 'State Transition Counts'
        cbar_label = 'Number of Transitions'
        fmt = 'd'
    else:
        matrix = transition_result['transition_probabilities']
        if title is None:
            title = 'State Transition Probabilities'
    
    # Reorder matrix if state_order is provided
    if state_order is not None:
        # Filter to only states that exist in the matrix
        available_states = [s for s in state_order if s in matrix.index and s in matrix.columns]
        if len(available_states) > 0:
            matrix = matrix.loc[available_states, available_states]
        else:
            print(f"⚠️  Warning: None of the states in state_order found in matrix. Using default order.")
    
    # Add group name to title if available
    if 'group_name' in transition_result and transition_result['group_name'] != 'all':
        title = f"{title} - {transition_result['group_name']}"
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap with larger annotations for publication
    sns.heatmap(matrix, 
                annot=annot,
                fmt=fmt,
                cmap=cmap,
                cbar_kws={'label': cbar_label, 'shrink': 0.8},
                square=True,
                linewidths=0.5,
                linecolor='gray',
                ax=ax,
                vmin=0,
                vmax=1 if not show_counts else None,
                annot_kws={'size': 14})  # Larger annotation text
    
    # Larger axis labels for publication
    ax.set_xlabel('To State', fontsize=16, fontweight='bold')
    ax.set_ylabel('From State', fontsize=16, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    
    # Rotate labels with larger font
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=14)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=14)
    
    # Larger colorbar label
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(cbar_label, size=14, weight='bold')
    
    # Add statistics text with larger font
    stats = transition_result['stats']
    stats_text = (f"Tracks: {stats['n_tracks']:,}\n"
                 f"Transitions: {stats['n_transitions']:,}\n"
                 f"Avg/track: {stats['avg_transitions_per_track']:.1f}")
    
    ax.text(1.15, 0.5, stats_text, transform=ax.transAxes,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=13)
    
    plt.tight_layout()
    
    # Invert axes if requested
    if invert_yaxis:
        ax.invert_yaxis()
    if invert_xaxis:
        ax.invert_xaxis()
    
    if transparent_background:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    
    # Save
    if save_path:
        group_suffix = f"_{transition_result['group_name']}" if 'group_name' in transition_result else ""
        matrix_type = "counts" if show_counts else "probabilities"
        filename = f"transition_matrix_{matrix_type}{group_suffix}.{export_format}"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight', 
                   transparent=transparent_background)
        print(f"✅ Saved transition matrix to: {full_path}")
    
    return fig, ax


def plot_transition_sankey(
    transition_result,
    min_transition_count=5,
    figsize=(12, 8),
    save_path=None,
    export_format='svg',
    dpi=300,
    title=None
):
    """
    Create a Sankey diagram showing state transitions.
    
    Parameters
    ----------
    transition_result : dict
        Result from analyze_state_transitions()
    min_transition_count : int, default=5
        Minimum number of transitions to show
    figsize : tuple, default=(12, 8)
        Figure size
    save_path : str, optional
        Directory to save figure
    export_format : str, default='svg'
        Export format
    dpi : int, default=300
        DPI for raster formats
    title : str, optional
        Custom title
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Sankey diagram requires plotly. Install with: pip install plotly")
    
    # Get transition data
    transitions = transition_result['transitions_summary']
    transitions = transitions[transitions['count'] >= min_transition_count].copy()
    
    # Get unique states
    all_states = sorted(set(transitions['from_state'].unique()) | set(transitions['to_state'].unique()))
    state_to_idx = {state: idx for idx, state in enumerate(all_states)}
    
    # Prepare Sankey data
    source = [state_to_idx[s] for s in transitions['from_state']]
    target = [state_to_idx[s] + len(all_states) for s in transitions['to_state']]  # Offset target indices
    values = transitions['count'].tolist()
    
    # Create labels (source + target)
    labels = list(all_states) + list(all_states)
    
    # Create color palette
    import matplotlib.cm as cm
    colors_norm = plt.Normalize(vmin=0, vmax=len(all_states)-1)
    colors = [f"rgba{tuple(int(x*255) for x in cm.Set3(colors_norm(i))[:3]) + (0.6,)}" 
              for i in range(len(all_states))]
    
    # Assign colors to links based on source state
    link_colors = [colors[src] for src in source]
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=colors + colors
        ),
        link=dict(
            source=source,
            target=target,
            value=values,
            color=link_colors
        )
    )])
    
    # Update layout
    if title is None:
        title = 'State Transition Flow'
    if 'group_name' in transition_result and transition_result['group_name'] != 'all':
        title = f"{title} - {transition_result['group_name']}"
    
    fig.update_layout(
        title=title,
        font=dict(size=12),
        height=figsize[1]*100,
        width=figsize[0]*100
    )
    
    # Save
    if save_path:
        group_suffix = f"_{transition_result['group_name']}" if 'group_name' in transition_result else ""
        filename = f"transition_sankey{group_suffix}.{export_format}"
        full_path = os.path.join(save_path, filename)
        
        if export_format == 'html':
            fig.write_html(full_path)
        else:
            fig.write_image(full_path, format=export_format)
        
        print(f"✅ Saved Sankey diagram to: {full_path}")
    
    return fig


def plot_dwell_time_distributions(
    dwell_result,
    figsize=(12, 6),
    color_palette=None,
    save_path=None,
    export_format='svg',
    dpi=300,
    transparent_background=True,
    bins=None
):
    """
    Plot distributions of dwell times for each state.
    
    Parameters
    ----------
    dwell_result : dict
        Result from analyze_state_dwell_times()
    figsize : tuple, default=(12, 6)
        Figure size
    color_palette : list, optional
        Colors for different states
    save_path : str, optional
        Directory to save figure
    export_format : str, default='svg'
        Export format
    dpi : int, default=300
        DPI for raster formats
    transparent_background : bool, default=True
        Whether to make background transparent
    bins : int or list, optional
        Histogram bins
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    dwell_df = dwell_result['dwell_times']
    summary_stats = dwell_result['summary_stats']
    
    if len(dwell_df) == 0:
        print("No dwell time data to plot")
        return None, None
    
    # Get unique states
    states = sorted(dwell_df['state'].unique())
    
    # Set colors
    if color_palette is None:
        color_palette = sns.color_palette("colorblind", len(states))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot distributions
    for idx, state in enumerate(states):
        state_data = dwell_df[dwell_df['state'] == state]['dwell_windows']
        
        ax.hist(state_data, bins=bins if bins else 'auto',
                alpha=0.6, label=state, color=color_palette[idx],
                edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Dwell Time (windows)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('State Dwell Time Distributions', fontsize=14, fontweight='bold')
    ax.legend(title='State', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if transparent_background:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    
    # Save
    if save_path:
        group_suffix = f"_{dwell_result['group_name']}" if 'group_name' in dwell_result else ""
        filename = f"dwell_time_distributions{group_suffix}.{export_format}"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight', 
                   transparent=transparent_background)
        print(f"✅ Saved dwell time distributions to: {full_path}")
    
    return fig, ax


def plot_transition_counts_comparison(
    transition_counts_df,
    x='mol',
    hue='n_unique_states',
    y='n_transitions',
    figsize=(10, 6),
    palette=None,
    save_path=None,
    export_format='svg',
    dpi=300,
    transparent_background=True
):
    """
    Compare transition counts across groups (e.g., molecules).
    
    Parameters
    ----------
    transition_counts_df : pd.DataFrame or pl.DataFrame
        Result from count_state_transitions_per_track()
    x : str, default='mol'
        Column for x-axis grouping
    hue : str, default='n_unique_states'
        Column for color grouping
    y : str, default='n_transitions'
        Column for y-axis values
    figsize : tuple, default=(10, 6)
        Figure size
    palette : str or list, optional
        Color palette
    save_path : str, optional
        Directory to save figure
    export_format : str, default='svg'
        Export format
    dpi : int, default=300
        DPI for raster formats
    transparent_background : bool, default=True
        Whether to make background transparent
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Convert to pandas if needed
    if isinstance(transition_counts_df, pl.DataFrame):
        df = transition_counts_df.to_pandas()
    else:
        df = transition_counts_df
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create boxplot
    sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax, palette=palette)
    
    ax.set_xlabel(x.capitalize(), fontsize=12, fontweight='bold')
    ax.set_ylabel(y.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(f'{y.replace("_", " ").title()} by {x.capitalize()}', 
                fontsize=14, fontweight='bold')
    ax.legend(title=hue.replace('_', ' ').title(), fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if transparent_background:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    
    # Save
    if save_path:
        filename = f"transition_counts_comparison_{x}.{export_format}"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight', 
                   transparent=transparent_background)
        print(f"✅ Saved transition counts comparison to: {full_path}")
    
    return fig, ax


# ============================================================================
# SIMILARITY ANALYSIS VISUALIZATIONS
# ============================================================================


def plot_between_molecule_similarity_heatmap(
    between_mol_result,
    figsize=(8, 7),
    cmap='RdYlGn',
    annot=True,
    save_path=None,
    export_format='svg',
    dpi=300,
    transparent_background=True
):
    """
    Plot heatmap of between-molecule similarity scores.
    
    Parameters
    ----------
    between_mol_result : dict
        Result from calculate_between_molecule_similarity()
    figsize : tuple, default=(8, 7)
        Figure size
    cmap : str, default='RdYlGn'
        Colormap
    annot : bool, default=True
        Annotate cells with values
    save_path : str, optional
        Directory to save figure
    export_format : str, default='svg'
        Export format
    dpi : int, default=300
        DPI
    transparent_background : bool, default=True
        Transparent background
        
    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    similarity_df = between_mol_result['similarity_matrix']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(similarity_df, 
                annot=annot,
                fmt='.3f',
                cmap=cmap,
                vmin=0,
                vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Similarity Score'},
                ax=ax)
    
    ax.set_title(f"Between-Molecule Similarity\n({between_mol_result['stats']['method']})",
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if transparent_background:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    
    if save_path:
        filename = f"between_molecule_similarity_heatmap.{export_format}"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight', transparent=transparent_background)
        print(f"✅ Saved to: {full_path}")
    
    return fig, ax


def plot_similarity_dendrogram(
    similarity_result,
    method='average',
    metric='precomputed',
    figsize=(10, 6),
    save_path=None,
    export_format='svg',
    dpi=300,
    transparent_background=True
):
    """
    Plot dendrogram showing hierarchical clustering based on similarity.
    
    Parameters
    ----------
    similarity_result : dict
        Result from calculate_between_molecule_similarity() or calculate_track_similarity()
    method : str, default='average'
        Linkage method ('single', 'complete', 'average', 'ward')
    metric : str, default='precomputed'
        Distance metric
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : str, optional
        Directory to save
    export_format : str, default='svg'
        Export format
    dpi : int, default=300
        DPI
    transparent_background : bool, default=True
        Transparent background
        
    Returns
    -------
    fig, ax, linkage_matrix
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    import matplotlib.pyplot as plt
    
    # Get distance matrix
    distance_matrix = similarity_result['distance_matrix'].values
    
    # Convert to condensed distance matrix for linkage
    condensed_dist = squareform(distance_matrix, checks=False)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method=method)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    dendrogram(linkage_matrix,
              labels=similarity_result['distance_matrix'].index.tolist(),
              ax=ax)
    
    ax.set_title(f"Hierarchical Clustering Dendrogram\n(method={method})",
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Molecule / Superwindow', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    
    plt.tight_layout()
    
    if transparent_background:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    
    if save_path:
        filename = f"similarity_dendrogram_{method}.{export_format}"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight', transparent=transparent_background)
        print(f"✅ Saved to: {full_path}")
    
    return fig, ax, linkage_matrix


def plot_similarity_mds(
    similarity_result,
    n_components=2,
    figsize=(10, 8),
    color_by=None,
    palette=None,
    save_path=None,
    export_format='svg',
    dpi=300,
    transparent_background=True
):
    """
    Plot MDS (Multidimensional Scaling) of similarity matrix.
    
    Visualizes similarities in 2D space where similar items are close together.
    
    Parameters
    ----------
    similarity_result : dict
        Result from calculate_between_molecule_similarity() or calculate_track_similarity()
    n_components : int, default=2
        Number of dimensions (2 or 3)
    figsize : tuple, default=(10, 8)
        Figure size
    color_by : pd.Series or list, optional
        Color points by this variable
    palette : str or list, optional
        Color palette
    save_path : str, optional
        Directory to save
    export_format : str, default='svg'
        Export format
    dpi : int, default=300
        DPI
    transparent_background : bool, default=True
        Transparent background
        
    Returns
    -------
    fig, ax, embedding
    """
    from sklearn.manifold import MDS
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get distance matrix
    distance_matrix = similarity_result['distance_matrix'].values
    labels = similarity_result['distance_matrix'].index.tolist()
    
    # Perform MDS
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
    embedding = mds.fit_transform(distance_matrix)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if color_by is not None:
        if isinstance(color_by, str) and color_by in ['mol', 'molecule']:
            # Extract molecule names from labels
            colors = [label.split('_')[0] if '_' in label else label for label in labels]
        else:
            colors = color_by
        
        unique_colors = list(set(colors))
        if palette is None:
            palette = sns.color_palette("husl", len(unique_colors))
        
        color_map = {col: palette[i] for i, col in enumerate(unique_colors)}
        point_colors = [color_map[c] for c in colors]
        
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                            c=point_colors, s=100, alpha=0.7, edgecolors='k')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[c], label=c) for c in unique_colors]
        ax.legend(handles=legend_elements, title='Molecule', loc='best')
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=100, alpha=0.7, edgecolors='k')
    
    # Annotate points
    for i, label in enumerate(labels):
        ax.annotate(label, (embedding[i, 0], embedding[i, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)
    
    ax.set_xlabel('MDS Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('MDS Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title(f'MDS Plot of Molecular Similarity\n(Stress: {mds.stress_:.3f})',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if transparent_background:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    
    if save_path:
        filename = f"similarity_mds.{export_format}"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight', transparent=transparent_background)
        print(f"✅ Saved to: {full_path}")
    
    return fig, ax, embedding


def plot_similarity_umap(
    similarity_result,
    n_neighbors=15,
    min_dist=0.1,
    figsize=(10, 8),
    color_by=None,
    palette=None,
    save_path=None,
    export_format='svg',
    dpi=300,
    transparent_background=True
):
    """
    Plot UMAP of similarity matrix.
    
    UMAP (Uniform Manifold Approximation and Projection) preserves both
    local and global structure better than MDS.
    
    Parameters
    ----------
    similarity_result : dict
        Result from similarity calculation
    n_neighbors : int, default=15
        UMAP n_neighbors parameter
    min_dist : float, default=0.1
        UMAP min_dist parameter
    figsize : tuple, default=(10, 8)
        Figure size
    color_by : pd.Series or list or str, optional
        Color points by this variable. If 'mol', extracts from labels.
    palette : str or list, optional
        Color palette
    save_path : str, optional
        Directory to save
    export_format : str, default='svg'
        Export format
    dpi : int, default=300
        DPI
    transparent_background : bool, default=True
        Transparent background
        
    Returns
    -------
    fig, ax, embedding
    """
    try:
        import umap
    except ImportError:
        raise ImportError("UMAP requires 'umap-learn' package. Install with: pip install umap-learn")
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get distance matrix
    distance_matrix = similarity_result['distance_matrix'].values
    labels = similarity_result['distance_matrix'].index.tolist()
    
    # Perform UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, 
                       metric='precomputed', random_state=42)
    embedding = reducer.fit_transform(distance_matrix)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if color_by is not None:
        if isinstance(color_by, str) and color_by in ['mol', 'molecule']:
            # Extract molecule names from labels
            colors = [label.split('_')[0] if '_' in label else label for label in labels]
        else:
            colors = color_by
        
        unique_colors = list(set(colors))
        if palette is None:
            palette = sns.color_palette("husl", len(unique_colors))
        
        color_map = {col: palette[i] for i, col in enumerate(unique_colors)}
        point_colors = [color_map[c] for c in colors]
        
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                            c=point_colors, s=100, alpha=0.7, edgecolors='k')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[c], label=c) for c in unique_colors]
        ax.legend(handles=legend_elements, title='Molecule', loc='best')
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=100, alpha=0.7, edgecolors='k')
    
    # Optionally annotate points (can be crowded)
    if len(labels) <= 50:  # Only annotate if not too many points
        for i, label in enumerate(labels):
            ax.annotate(label, (embedding[i, 0], embedding[i, 1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
    
    ax.set_xlabel('UMAP Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title(f'UMAP Plot of Molecular Similarity\n(n_neighbors={n_neighbors}, min_dist={min_dist})',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if transparent_background:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    
    if save_path:
        filename = f"similarity_umap.{export_format}"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight', transparent=transparent_background)
        print(f"✅ Saved to: {full_path}")
    
    return fig, ax, embedding


def plot_similarity_pca(
    similarity_result,
    n_components=2,
    figsize=(10, 8),
    color_by=None,
    palette=None,
    save_path=None,
    export_format='svg',
    dpi=300,
    transparent_background=True
):
    """
    Plot PCA of similarity matrix.
    
    PCA finds principal components that explain variance in similarity.
    
    Parameters
    ----------
    similarity_result : dict
        Result from similarity calculation
    n_components : int, default=2
        Number of components
    figsize : tuple, default=(10, 8)
        Figure size
    color_by : pd.Series or list or str, optional
        Color points
    palette : str or list, optional
        Color palette
    save_path : str, optional
        Directory to save
    export_format : str, default='svg'
        Export format
    dpi : int, default=300
        DPI
    transparent_background : bool, default=True
        Transparent background
        
    Returns
    -------
    fig, ax, pca
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get similarity matrix (use similarity, not distance, for PCA)
    similarity_matrix = similarity_result['similarity_matrix'].values
    labels = similarity_result['similarity_matrix'].index.tolist()
    
    # Perform PCA
    pca = PCA(n_components=n_components, random_state=42)
    embedding = pca.fit_transform(similarity_matrix)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if color_by is not None:
        if isinstance(color_by, str) and color_by in ['mol', 'molecule']:
            colors = [label.split('_')[0] if '_' in label else label for label in labels]
        else:
            colors = color_by
        
        unique_colors = list(set(colors))
        if palette is None:
            palette = sns.color_palette("husl", len(unique_colors))
        
        color_map = {col: palette[i] for i, col in enumerate(unique_colors)}
        point_colors = [color_map[c] for c in colors]
        
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                            c=point_colors, s=100, alpha=0.7, edgecolors='k')
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[c], label=c) for c in unique_colors]
        ax.legend(handles=legend_elements, title='Molecule', loc='best')
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=100, alpha=0.7, edgecolors='k')
    
    # Optionally annotate
    if len(labels) <= 50:
        for i, label in enumerate(labels):
            ax.annotate(label, (embedding[i, 0], embedding[i, 1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
    
    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% variance)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% variance)', fontsize=12, fontweight='bold')
    ax.set_title(f'PCA of Molecular Similarity\n(Total variance: {var_explained.sum()*100:.1f}%)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if transparent_background:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    
    if save_path:
        filename = f"similarity_pca.{export_format}"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight', transparent=transparent_background)
        print(f"✅ Saved to: {full_path}")
    
    return fig, ax, pca


def plot_molecule_distance_comparison(
    between_mol_result,
    reference_molecule='HTT',
    figsize=(8, 6),
    palette=None,
    save_path=None,
    export_format='svg',
    dpi=300,
    transparent_background=True
):
    """
    Plot "Who does X resemble?" - distance from reference molecule to others.
    
    Parameters
    ----------
    between_mol_result : dict
        Result from calculate_between_molecule_similarity()
    reference_molecule : str, default='HTT'
        Reference molecule to compare others to
    figsize : tuple, default=(8, 6)
        Figure size
    palette : list, optional
        Colors
    save_path : str, optional
        Directory to save
    export_format : str, default='svg'
        Export format
    dpi : int, default=300
        DPI
    transparent_background : bool, default=True
        Transparent background
        
    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt
    
    similarity_df = between_mol_result['similarity_matrix']
    
    if reference_molecule not in similarity_df.index:
        raise ValueError(f"{reference_molecule} not in similarity matrix")
    
    # Get similarities to reference molecule
    similarities = similarity_df.loc[reference_molecule]
    similarities = similarities[similarities.index != reference_molecule]  # Exclude self
    similarities = similarities.sort_values(ascending=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = palette if palette else ['#0173B2', '#DE8F05', '#029E73', '#CC78BC']
    
    bars = ax.barh(range(len(similarities)), similarities.values, 
                  color=colors[:len(similarities)], alpha=0.7, edgecolor='k')
    
    ax.set_yticks(range(len(similarities)))
    ax.set_yticklabels(similarities.index)
    ax.set_xlabel('Similarity Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Who Does {reference_molecule} Resemble?\n(Higher = More Similar)',
                fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (mol, sim) in enumerate(similarities.items()):
        ax.text(sim + 0.02, i, f'{sim:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if transparent_background:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    
    if save_path:
        filename = f"molecule_distance_{reference_molecule}.{export_format}"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight', transparent=transparent_background)
        print(f"✅ Saved to: {full_path}")
    
    return fig, ax


def plot_state_trajectory_blobs(
    transition_result,
    superwindow_length=10,
    n_trajectories=5,
    figsize=(14, 8),
    state_colors=None,
    state_order=None,
    line_width_scale=5,
    alpha_base=0.7,
    save_path=None,
    export_format='svg',
    dpi=300
):
    """
    Plot example trajectories with blobs and transition-weighted lines.
    
    Parameters
    ----------
    transition_result : dict
        Result from analyze_state_transitions()
    superwindow_length : int, default=10
        Length of superwindow
    n_trajectories : int, default=5
        Number of example trajectories to show
    figsize : tuple, default=(14, 8)
        Figure size
    state_colors : dict, optional
        State name -> color mapping
    state_order : list, optional
        Custom order for states on y-axis
    line_width_scale : float, default=5
        Scale factor for line widths based on transition probability
    alpha_base : float, default=0.7
        Base transparency
    save_path : str, optional
        Directory to save
    export_format : str, default='svg'
        Export format
    dpi : int, default=300
        DPI
        
    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    prob_matrix = transition_result['transition_probabilities']
    
    if state_order is None:
        states = prob_matrix.index.tolist()
    else:
        states = state_order
    n_states = len(states)
    
    if state_colors is None:
        cmap = plt.cm.Set3
        state_colors = {state: cmap(i / n_states) for i, state in enumerate(states)}
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw background blobs
    for t in range(superwindow_length):
        for i, state in enumerate(states):
            prob = prob_matrix.loc[:, state].mean() if t > 0 else 1.0 / n_states
            size = prob * 3000
            ax.scatter(t, i, s=size, c=[state_colors[state]], 
                      alpha=0.3, edgecolors='k', linewidths=0.5, zorder=1)
    
    # Generate and plot example trajectories
    np.random.seed(42)
    for traj_idx in range(n_trajectories):
        current_state = np.random.choice(states)
        trajectory = [current_state]
        
        for t in range(superwindow_length - 1):
            probs = prob_matrix.loc[current_state].values
            next_state = np.random.choice(states, p=probs)
            trajectory.append(next_state)
            
            # Get transition probability for line thickness
            trans_prob = prob_matrix.loc[current_state, next_state]
            linewidth = 1 + trans_prob * line_width_scale
            alpha = alpha_base * (0.5 + 0.5 * trans_prob)
            
            # Draw line segment
            y_curr = states.index(current_state)
            y_next = states.index(next_state)
            ax.plot([t, t+1], [y_curr, y_next], '-', 
                   linewidth=linewidth, alpha=alpha, color='black', zorder=5)
            
            current_state = next_state
        
        # Draw trajectory blobs
        y_positions = [states.index(s) for s in trajectory]
        x_positions = list(range(superwindow_length))
        ax.scatter(x_positions, y_positions, s=150, c='black', 
                  alpha=0.8, edgecolors='white', linewidths=2, zorder=10)
    
    ax.set_yticks(range(n_states))
    ax.set_yticklabels(states, fontsize=10)
    ax.set_xlabel('Window Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('State', fontsize=12, fontweight='bold')
    ax.set_title(f'Example State Trajectories (n={n_trajectories})\nLine thickness = transition probability',
                fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, superwindow_length - 0.5)
    ax.set_ylim(-0.5, n_states - 0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        filename = f"state_trajectory_examples.{export_format}"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Saved to: {full_path}")
    
    return fig, ax


def plot_similarity_space_pca(
    sequence_df,
    pca_embedding_col='pca_embedding',
    color_by='mol',
    split_by=None,
    palette='colorblind',
    figsize=(10, 8),
    s=50,
    alpha=0.7,
    save_path=None,
    export_format='svg',
    dpi=300,
    transparent_background=True
):
    """
    Plot PCA embedding of DTW similarity space.
    
    Parameters
    ----------
    sequence_df : pd.DataFrame
        DataFrame with PCA embeddings
    pca_embedding_col : str, default='pca_embedding'
        Column containing PCA coordinates (array of shape (n_components,))
    color_by : str, default='mol'
        Column to color points by
    split_by : str, optional
        Column to create small multiples (e.g., 'mol')
    palette : str, dict, or list, default='colorblind'
        Color palette
    figsize : tuple, default=(10, 8)
        Figure size
    s : float, default=50
        Point size
    alpha : float, default=0.7
        Point transparency
    save_path : str, optional
        Save directory
    export_format : str, default='svg'
        Export format
    dpi : int, default=300
        DPI
    transparent_background : bool, default=True
        Transparent background
        
    Returns
    -------
    fig, ax or axes
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract PCA coordinates
    pca_coords = np.vstack(sequence_df[pca_embedding_col].values)
    
    # Handle colors
    if palette == 'colorblind':
        colorblind_palette = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161']
    else:
        colorblind_palette = None
    
    unique_vals = sequence_df[color_by].unique()
    
    if isinstance(palette, dict):
        color_map = palette
    elif isinstance(palette, list):
        color_map = {val: palette[i % len(palette)] for i, val in enumerate(unique_vals)}
    elif palette == 'colorblind':
        color_map = {val: colorblind_palette[i % len(colorblind_palette)] 
                    for i, val in enumerate(unique_vals)}
    else:
        cmap = plt.cm.get_cmap(palette)
        color_map = {val: cmap(i / len(unique_vals)) for i, val in enumerate(unique_vals)}
    
    colors = [color_map[val] for val in sequence_df[color_by]]
    
    # Create plot(s)
    if split_by is None:
        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(pca_coords[:, 0], pca_coords[:, 1], 
                            c=colors, s=s, alpha=alpha, edgecolors='k', linewidths=0.5)
        ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
        ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
        ax.set_title(f'PCA of DTW Similarity Space\n(colored by {color_by})',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[val], label=val) 
                          for val in unique_vals]
        ax.legend(handles=legend_elements, title=color_by, loc='best')
        
        axes = None
    else:
        # Small multiples
        split_vals = sequence_df[split_by].unique()
        n_plots = len(split_vals)
        ncols = min(3, n_plots)
        nrows = int(np.ceil(n_plots / ncols))
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        for i, split_val in enumerate(split_vals):
            ax = axes[i]
            mask = sequence_df[split_by] == split_val
            coords = pca_coords[mask]
            cols = [colors[j] for j, m in enumerate(mask) if m]
            
            ax.scatter(coords[:, 0], coords[:, 1], 
                      c=cols, s=s, alpha=alpha, edgecolors='k', linewidths=0.5)
            ax.set_xlabel('PC1', fontweight='bold')
            ax.set_ylabel('PC2', fontweight='bold')
            ax.set_title(f'{split_val}', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        fig.suptitle(f'PCA of DTW Similarity (colored by {color_by})',
                    fontsize=14, fontweight='bold', y=1.0)
    
    if transparent_background:
        fig.patch.set_alpha(0)
        if axes is None:
            ax.patch.set_alpha(0)
        else:
            for a in axes:
                a.patch.set_alpha(0)
    
    plt.tight_layout()
    
    if save_path:
        suffix = f"_{split_by}" if split_by else ""
        filename = f"pca_similarity_space{suffix}.{export_format}"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight', 
                   transparent=transparent_background)
        print(f"✅ Saved to: {full_path}")
    
    return fig, ax if axes is None else axes


def plot_similarity_space_umap(
    sequence_df,
    umap_embedding_col='umap_embedding',
    color_by='mol',
    split_by=None,
    palette='colorblind',
    figsize=(10, 8),
    s=50,
    alpha=0.7,
    save_path=None,
    export_format='svg',
    dpi=300,
    transparent_background=True
):
    """
    Plot UMAP embedding of DTW similarity space.
    
    Parameters
    ----------
    sequence_df : pd.DataFrame
        DataFrame with UMAP embeddings
    umap_embedding_col : str, default='umap_embedding'
        Column containing UMAP coordinates
    color_by : str, default='mol'
        Column to color points by
    split_by : str, optional
        Column to create small multiples
    palette : str, dict, or list, default='colorblind'
        Color palette
    figsize : tuple, default=(10, 8)
        Figure size
    s : float, default=50
        Point size
    alpha : float, default=0.7
        Point transparency
    save_path : str, optional
        Save directory
    export_format : str, default='svg'
        Export format
    dpi : int, default=300
        DPI
    transparent_background : bool, default=True
        Transparent background
        
    Returns
    -------
    fig, ax or axes
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    umap_coords = np.vstack(sequence_df[umap_embedding_col].values)
    
    # Handle colors (same logic as PCA)
    if palette == 'colorblind':
        colorblind_palette = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161']
    else:
        colorblind_palette = None
    
    unique_vals = sequence_df[color_by].unique()
    
    if isinstance(palette, dict):
        color_map = palette
    elif isinstance(palette, list):
        color_map = {val: palette[i % len(palette)] for i, val in enumerate(unique_vals)}
    elif palette == 'colorblind':
        color_map = {val: colorblind_palette[i % len(colorblind_palette)] 
                    for i, val in enumerate(unique_vals)}
    else:
        cmap = plt.cm.get_cmap(palette)
        color_map = {val: cmap(i / len(unique_vals)) for i, val in enumerate(unique_vals)}
    
    colors = [color_map[val] for val in sequence_df[color_by]]
    
    if split_by is None:
        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                            c=colors, s=s, alpha=alpha, edgecolors='k', linewidths=0.5)
        ax.set_xlabel('UMAP1', fontsize=12, fontweight='bold')
        ax.set_ylabel('UMAP2', fontsize=12, fontweight='bold')
        ax.set_title(f'UMAP of DTW Similarity Space\n(colored by {color_by})',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[val], label=val) 
                          for val in unique_vals]
        ax.legend(handles=legend_elements, title=color_by, loc='best')
        
        axes = None
    else:
        split_vals = sequence_df[split_by].unique()
        n_plots = len(split_vals)
        ncols = min(3, n_plots)
        nrows = int(np.ceil(n_plots / ncols))
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        for i, split_val in enumerate(split_vals):
            ax = axes[i]
            mask = sequence_df[split_by] == split_val
            coords = umap_coords[mask]
            cols = [colors[j] for j, m in enumerate(mask) if m]
            
            ax.scatter(coords[:, 0], coords[:, 1], 
                      c=cols, s=s, alpha=alpha, edgecolors='k', linewidths=0.5)
            ax.set_xlabel('UMAP1', fontweight='bold')
            ax.set_ylabel('UMAP2', fontweight='bold')
            ax.set_title(f'{split_val}', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        fig.suptitle(f'UMAP of DTW Similarity (colored by {color_by})',
                    fontsize=14, fontweight='bold', y=1.0)
    
    if transparent_background:
        fig.patch.set_alpha(0)
        if axes is None:
            ax.patch.set_alpha(0)
        else:
            for a in axes:
                a.patch.set_alpha(0)
    
    plt.tight_layout()
    
    if save_path:
        suffix = f"_{split_by}" if split_by else ""
        filename = f"umap_similarity_space{suffix}.{export_format}"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight', 
                   transparent=transparent_background)
        print(f"✅ Saved to: {full_path}")
    
    return fig, ax if axes is None else axes


def plot_similarity_space_phate(
    sequence_df,
    phate_embedding_col='phate_embedding',
    color_by='mol',
    split_by=None,
    palette='colorblind',
    figsize=(10, 8),
    s=50,
    alpha=0.7,
    save_path=None,
    export_format='svg',
    dpi=300,
    transparent_background=True,
    # PHATE computation parameters (used if embedding column doesn't exist)
    compute_phate=False,
    distance_matrix_col='similarity_distance',
    n_components=2,
    knn=5,
    decay=40,
    t='auto',
    random_state=42
):
    """
    Plot PHATE embedding of similarity space.
    
    PHATE (Potential of Heat-diffusion for Affinity-based Trajectory Embedding)
    is excellent for visualizing trajectories and continuous processes.
    
    Parameters
    ----------
    sequence_df : pd.DataFrame
        DataFrame with PHATE embeddings or distance data
    phate_embedding_col : str, default='phate_embedding'
        Column containing PHATE coordinates (array of shape (n_components,))
    color_by : str, default='mol'
        Column to color points by
    split_by : str, optional
        Column to create small multiples (e.g., 'mol')
    palette : str, dict, or list, default='colorblind'
        Color palette
    figsize : tuple, default=(10, 8)
        Figure size
    s : float, default=50
        Point size
    alpha : float, default=0.7
        Point transparency
    save_path : str, optional
        Save directory
    export_format : str, default='svg'
        Export format
    dpi : int, default=300
        DPI
    transparent_background : bool, default=True
        Transparent background
    compute_phate : bool, default=False
        If True and phate_embedding_col doesn't exist, compute PHATE from distance_matrix_col
    distance_matrix_col : str, default='similarity_distance'
        Column containing distance values (used if compute_phate=True)
    n_components : int, default=2
        Number of PHATE dimensions
    knn : int, default=5
        Number of nearest neighbors for PHATE
    decay : int, default=40
        Decay parameter for PHATE kernel
    t : int or 'auto', default='auto'
        Number of diffusion steps
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    fig, ax or axes
        Figure and axes objects
    phate_coords : np.ndarray (optional)
        If compute_phate=True, also returns the computed PHATE coordinates
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Check if PHATE embedding exists or needs to be computed
    if phate_embedding_col in sequence_df.columns:
        phate_coords = np.vstack(sequence_df[phate_embedding_col].values)
        computed = False
    elif compute_phate:
        try:
            import phate
        except ImportError:
            raise ImportError("PHATE requires 'phate' package. Install with: pip install phate")
        
        print("Computing PHATE embedding...")
        
        # Get distance data
        if distance_matrix_col in sequence_df.columns:
            distances = np.vstack(sequence_df[distance_matrix_col].values)
        else:
            raise ValueError(f"Column '{distance_matrix_col}' not found and no embedding column exists")
        
        # Compute PHATE
        phate_op = phate.PHATE(
            n_components=n_components,
            knn=knn,
            decay=decay,
            t=t,
            random_state=random_state,
            verbose=0
        )
        phate_coords = phate_op.fit_transform(distances)
        computed = True
        print(f"✅ PHATE embedding computed: {phate_coords.shape}")
    else:
        raise ValueError(f"Column '{phate_embedding_col}' not found. Set compute_phate=True to compute PHATE.")
    
    # Handle colors
    if palette == 'colorblind':
        colorblind_palette = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161',
                              '#FBAFE4', '#949494', '#ECE133', '#56B4E9', '#D55E00']
    else:
        colorblind_palette = None
    
    unique_vals = sequence_df[color_by].unique()
    
    if isinstance(palette, dict):
        color_map = palette
    elif isinstance(palette, list):
        color_map = {val: palette[i % len(palette)] for i, val in enumerate(unique_vals)}
    elif palette == 'colorblind':
        color_map = {val: colorblind_palette[i % len(colorblind_palette)] 
                    for i, val in enumerate(unique_vals)}
    else:
        cmap = plt.cm.get_cmap(palette)
        color_map = {val: cmap(i / len(unique_vals)) for i, val in enumerate(unique_vals)}
    
    colors = [color_map[val] for val in sequence_df[color_by]]
    
    # Create plot(s)
    if split_by is None:
        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(phate_coords[:, 0], phate_coords[:, 1], 
                            c=colors, s=s, alpha=alpha, edgecolors='k', linewidths=0.5)
        ax.set_xlabel('PHATE1', fontsize=12, fontweight='bold')
        ax.set_ylabel('PHATE2', fontsize=12, fontweight='bold')
        ax.set_title(f'PHATE of Similarity Space\n(colored by {color_by})',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[val], label=val) 
                          for val in unique_vals]
        ax.legend(handles=legend_elements, title=color_by, loc='best')
        
        axes = None
    else:
        # Small multiples
        split_vals = sequence_df[split_by].unique()
        n_plots = len(split_vals)
        ncols = min(3, n_plots)
        nrows = int(np.ceil(n_plots / ncols))
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        for i, split_val in enumerate(split_vals):
            ax = axes[i]
            mask = sequence_df[split_by] == split_val
            coords = phate_coords[mask]
            cols = [colors[j] for j, m in enumerate(mask) if m]
            
            ax.scatter(coords[:, 0], coords[:, 1], 
                      c=cols, s=s, alpha=alpha, edgecolors='k', linewidths=0.5)
            ax.set_xlabel('PHATE1', fontweight='bold')
            ax.set_ylabel('PHATE2', fontweight='bold')
            ax.set_title(f'{split_val}', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        fig.suptitle(f'PHATE of Similarity Space (colored by {color_by})',
                    fontsize=14, fontweight='bold', y=1.0)
    
    if transparent_background:
        fig.patch.set_alpha(0)
        if axes is None:
            ax.patch.set_alpha(0)
        else:
            for a in axes:
                a.patch.set_alpha(0)
    
    plt.tight_layout()
    
    if save_path:
        suffix = f"_{split_by}" if split_by else ""
        filename = f"phate_similarity_space{suffix}.{export_format}"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight', 
                   transparent=transparent_background)
        print(f"✅ Saved to: {full_path}")
    
    if computed:
        return fig, ax if axes is None else axes, phate_coords
    else:
        return fig, ax if axes is None else axes


def plot_representative_sequences(
    clustered_df,
    sequence_col='state_sequence',
    cluster_col='superwindow_cluster',
    group_by='mol',
    n_superwindows_per_cluster=3,
    state_order=None,
    state_colors='colorblind',
    figsize=(18, 12),
    save_path=None,
    export_format='svg',
    dpi=300,
    transparent_background=True,
    show_superwindow_id=False,
    superwindow_id_col='superwindow_id',
    id_fontsize=6,
    id_color='gray'
):
    """
    Plot representative sequences from each cluster to verify similarity.
    
    Parameters
    ----------
    clustered_df : pd.DataFrame
        DataFrame with cluster labels (from add_dtw_clusters)
    sequence_col : str, default='state_sequence'
        Column with state sequences
    cluster_col : str, default='superwindow_cluster'
        Column with cluster labels
    group_by : str, default='mol'
        Column to group by (e.g., 'mol')
    n_superwindows_per_cluster : int, default=3
        Number of example sequences to show per cluster
    state_order : list, optional
        Custom state order
    state_colors : str, dict, or list, default='colorblind'
        State colors
    figsize : tuple, default=(18, 12)
        Figure size
    save_path : str, optional
        Save directory
    export_format : str, default='svg'
        Export format
    dpi : int, default=300
        DPI
    transparent_background : bool, default=True
        Transparent background
    show_superwindow_id : bool, default=False
        Whether to annotate each subplot with the superwindow_id
    superwindow_id_col : str, default='superwindow_id'
        Column name for superwindow_id
    id_fontsize : int, default=6
        Font size for superwindow_id annotation
    id_color : str, default='gray'
        Color for superwindow_id annotation
        
    Returns
    -------
    tuple
        (fig, axes, plotted_superwindow_ids, state_colors_dict)
        - fig: matplotlib figure
        - axes: matplotlib axes array
        - plotted_superwindow_ids: list of superwindow_ids that were plotted
        - state_colors_dict: dict mapping state -> color
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Track superwindow_ids that are plotted
    plotted_superwindow_ids = []
    
    # Get all states
    all_states = set()
    for seq in clustered_df[sequence_col]:
        if isinstance(seq, list):
            all_states.update(seq)
        else:
            all_states.update(list(seq))
    
    if state_order is None:
        states = sorted(all_states)
    else:
        states = state_order
    
    n_states = len(states)
    
    # Handle colors
    if state_colors == 'colorblind':
        colorblind_palette = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161', 
                             '#FBAFE4', '#949494', '#ECE133', '#56B4E9', '#D55E00']
        state_colors = {state: colorblind_palette[i % len(colorblind_palette)] 
                       for i, state in enumerate(states)}
    elif isinstance(state_colors, list):
        state_colors = {state: state_colors[i % len(state_colors)] 
                       for i, state in enumerate(states)}
    elif state_colors is None:
        cmap = plt.cm.Set3
        state_colors = {state: cmap(i / n_states) for i, state in enumerate(states)}
    
    # Get groups and clusters
    groups = sorted(clustered_df[group_by].unique())
    n_clusters = clustered_df[cluster_col].nunique()
    
    # Create subplot grid: rows = groups, cols = clusters * n_examples
    n_cols = n_clusters * n_superwindows_per_cluster
    fig, axes = plt.subplots(len(groups), n_cols, 
                            figsize=figsize, squeeze=False)
    
    for group_idx, group_val in enumerate(groups):
        group_data = clustered_df[clustered_df[group_by] == group_val]
        
        col_idx = 0
        for cluster_id in range(n_clusters):
            cluster_data = group_data[group_data[cluster_col] == cluster_id]
            
            # Sample n examples from this cluster
            n_available = len(cluster_data)
            n_to_show = min(n_superwindows_per_cluster, n_available)
            
            if n_to_show > 0:
                examples = cluster_data.sample(n=n_to_show, random_state=42)
            else:
                examples = cluster_data
            
            for ex_idx, (_, row) in enumerate(examples.iterrows()):
                ax = axes[group_idx, col_idx]
                sequence = row[sequence_col]
                
                # Track this superwindow_id
                sw_id = None
                if superwindow_id_col in row.index:
                    sw_id = row[superwindow_id_col]
                    plotted_superwindow_ids.append(sw_id)
                
                # Convert to list if needed
                if not isinstance(sequence, list):
                    sequence = list(sequence)
                
                # Plot sequence
                for t, state in enumerate(sequence):
                    if state in states:
                        y_pos = states.index(state)
                        ax.scatter(t, y_pos, s=200, c=[state_colors[state]], 
                                 edgecolors='k', linewidths=1.5, zorder=5)
                        
                        if t < len(sequence) - 1:
                            next_state = sequence[t + 1]
                            if next_state in states:
                                y_next = states.index(next_state)
                                ax.plot([t, t+1], [y_pos, y_next], 'k-', 
                                       linewidth=1.5, alpha=0.5, zorder=3)
                
                # Formatting
                ax.set_yticks(range(n_states))
                ax.set_yticklabels(states if col_idx == 0 else [], fontsize=8)
                ax.set_xlim(-0.5, len(sequence) - 0.5)
                ax.set_ylim(-0.5, n_states - 0.5)
                ax.grid(True, alpha=0.2, axis='x')
                
                # Add superwindow_id annotation
                if show_superwindow_id and sw_id is not None:
                    ax.text(0.5, -0.15, sw_id, transform=ax.transAxes,
                           fontsize=id_fontsize, color=id_color,
                           ha='center', va='top', style='italic')
                
                # Title only on first row
                if group_idx == 0:
                    if ex_idx == 0:
                        ax.set_title(f'Cluster {cluster_id}\n(n={n_available})', 
                                   fontsize=9, fontweight='bold')
                    else:
                        ax.set_title(f'Example {ex_idx+1}', fontsize=8)
                
                # Y-label on first column
                if col_idx == 0:
                    ax.set_ylabel(f'{group_val}', fontsize=10, fontweight='bold')
                
                col_idx += 1
            
            # Fill remaining columns for this cluster if not enough examples
            for _ in range(n_to_show, n_superwindows_per_cluster):
                axes[group_idx, col_idx].axis('off')
                col_idx += 1
    
    plt.suptitle(f'Representative Sequences per Cluster\n({n_superwindows_per_cluster} examples each)', 
                fontsize=14, fontweight='bold', y=0.995)
    
    if transparent_background:
        fig.patch.set_alpha(0)
        for ax_row in axes:
            for ax in ax_row:
                ax.patch.set_alpha(0)
    
    plt.tight_layout()
    
    if save_path:
        filename = f"representative_sequences_n{n_superwindows_per_cluster}.{export_format}"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight', 
                   transparent=transparent_background)
        print(f"✅ Saved to: {full_path}")
    
    print(f"\n✅ Plotted {len(plotted_superwindow_ids)} superwindows")
    print(f"   Use these IDs to filter instant_df for matching gallery visualization")
    
    return fig, axes, plotted_superwindow_ids, state_colors


def plot_transition_probabilities_stacked(
    transition_results_dict,
    state_order=None,
    figsize=(12, 6),
    state_colors='colorblind',
    save_path=None,
    export_format='svg',
    dpi=300,
    transparent_background=True
):
    """
    Stacked bar chart showing transition probabilities for each molecule.
    
    Parameters
    ----------
    transition_results_dict : dict
        Dict of molecule -> transition_result
    state_order : list, optional
        Custom state order
    figsize : tuple, default=(12, 6)
        Figure size
    state_colors : str, dict, or list, default='colorblind'
        'colorblind' for colorblind-friendly palette,
        dict mapping state -> color,
        or list of hex codes
    save_path : str, optional
        Save directory
    export_format : str, default='svg'
        Export format
    dpi : int, default=300
        DPI
    transparent_background : bool, default=True
        Transparent background
        
    Returns
    -------
    fig, axes
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get all unique states
    all_states = set()
    for result in transition_results_dict.values():
        all_states.update(result['transition_probabilities'].index)
    
    if state_order is None:
        states = sorted(all_states)
    else:
        states = state_order
    
    n_states = len(states)
    molecules = list(transition_results_dict.keys())
    
    # Handle color specification
    if state_colors == 'colorblind':
        colorblind_palette = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161', 
                             '#FBAFE4', '#949494', '#ECE133', '#56B4E9', '#D55E00']
        state_colors = {state: colorblind_palette[i % len(colorblind_palette)] 
                       for i, state in enumerate(states)}
    elif isinstance(state_colors, list):
        state_colors = {state: state_colors[i % len(state_colors)] 
                       for i, state in enumerate(states)}
    elif state_colors is None:
        cmap = plt.cm.Set3
        state_colors = {state: cmap(i / n_states) for i, state in enumerate(states)}
    
    # Create figure
    fig, axes = plt.subplots(1, len(molecules), figsize=figsize, sharey=True)
    if len(molecules) == 1:
        axes = [axes]
    
    for mol_idx, (mol, result) in enumerate(transition_results_dict.items()):
        ax = axes[mol_idx]
        prob_matrix = result['transition_probabilities']
        
        bar_width = 0.8
        x_pos = np.arange(n_states)
        
        bottom = np.zeros(n_states)
        for to_state in states:
            values = [prob_matrix.loc[from_state, to_state] 
                     if from_state in prob_matrix.index and to_state in prob_matrix.columns 
                     else 0 for from_state in states]
            ax.bar(x_pos, values, bar_width, bottom=bottom, 
                  label=to_state, color=state_colors[to_state], 
                  edgecolor='white', linewidth=1)
            bottom += values
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(states, rotation=45, ha='right', fontsize=9)
        ax.set_title(mol, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        
        if mol_idx == 0:
            ax.set_ylabel('Transition Probability', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('From State', fontsize=10, fontweight='bold')
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='To State', 
              bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=9)
    
    fig.suptitle('State Transition Probabilities (Stacked)', 
                fontsize=14, fontweight='bold', y=1.02)
    
    if transparent_background:
        fig.patch.set_alpha(0)
        for ax in axes:
            ax.patch.set_alpha(0)
    
    plt.tight_layout()
    
    if save_path:
        filename = f"transition_probabilities_stacked.{export_format}"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight', 
                   transparent=transparent_background)
        print(f"✅ Saved to: {full_path}")
    
    return fig, axes


