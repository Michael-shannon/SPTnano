import os
import pims
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np

import pandas as pd
import skimage.io as io
import napari
import numpy as np
from scipy.stats import sem
import random
from scipy.stats import linregress

from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.collections import LineCollection


from scipy.stats import sem
from matplotlib.colors import is_color_like

import config
from napari_animation import Animation

from .helper_scripts import *


from matplotlib.ticker import FixedLocator



def overlay_tracks_with_movie(tracks_df, movie_path, colormap=None):
    # Load the raw movie
    frames = pims.open(movie_path)
    
    # Create a new folder to save the PNG images
    output_folder = os.path.splitext(movie_path)[0]
    os.makedirs(output_folder, exist_ok=True)
    
    if colormap:
        # Get the colormap if specified
        cmap = cm.get_cmap(colormap)
        # Get unique particle IDs and assign colors from the colormap
        unique_particles = tracks_df['particle'].unique()
        colors = {particle: cmap(i / len(unique_particles)) for i, particle in enumerate(unique_particles)}
    else:
        # Use the default color cycle
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        unique_particles = tracks_df['particle'].unique()
        colors = {particle: color_cycle[i % len(color_cycle)] for i, particle in enumerate(unique_particles)}
    
    # Get the last frame for each particle
    last_frame = tracks_df.groupby('particle')['frame'].max()
    
    # Iterate over each frame in the movie
    for frame_index, frame in enumerate(frames):
        # Debug: Print frame index and frame shape
        print(f"Processing frame {frame_index} with shape {frame.shape}")
        
        # Create a figure and axis with a larger size
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Display the current frame
        ax.imshow(frame, cmap='gray', origin='upper')
        
        # Set the plot limits to match the image dimensions
        ax.set_xlim(0, frame.shape[1])
        ax.set_ylim(frame.shape[0], 0)
        
        # Iterate over each track in the DataFrame
        for particle_id, track in tracks_df.groupby('particle'):
            # Only plot the track if the current frame is less than or equal to the last frame of the particle
            if frame_index <= last_frame[particle_id]:
                # Get the x and y coordinates of the track for the current frame
                x = track.loc[track['frame'] <= frame_index, 'x']
                y = track.loc[track['frame'] <= frame_index, 'y']
                
                # Plot the track as a line with slightly thicker lines and consistent color
                ax.plot(x, y, label=f'Track {particle_id}', linewidth=2.0, color=colors[particle_id])
        
        # Remove the axis labels, ticks, and grid
        ax.axis('off')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Save the figure as a PNG image in the output folder
        output_path = os.path.join(output_folder, f'frame_{frame_index:04d}.png')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        
        # Close the figure to free up memory
        plt.close(fig)




def plot_histograms_seconds(traj_df, bins=100, coltoseparate='tracker', xlimit=None):
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
    sns.set_context("notebook", rc={"xtick.labelsize": size*multiplier, "ytick.labelsize": size*multiplier})
    
    max_track_length = traj_df.groupby('unique_id')['time_s_zeroed'].max().max()
    bin_edges = np.linspace(0, max_track_length, bins + 1)
    
    for i, tracker in enumerate(traj_df[coltoseparate].unique()):
        subset = traj_df[traj_df[coltoseparate] == tracker]
        subsetvalues = subset.groupby('unique_id')['time_s_zeroed'].max()
        
        # Calculate percentage counts
        counts, _ = np.histogram(subsetvalues, bins=bin_edges)
        percentage_counts = (counts / counts.sum()) * 100
        
        # Plot histogram
        sns.histplot(subsetvalues, bins=bin_edges, kde=True, label=tracker, alpha=0.5, stat="percent")
        
        subset_mean = subsetvalues.mean()
        subset_median = subsetvalues.median()
        subset_number_of_tracks = len(subset['unique_id'].unique())
        shift = i * 0.05
        plt.text(0.4, 0.6 - shift, f"{tracker}: mean: {subset_mean:.2f} seconds from {subset_number_of_tracks} tracks", transform=plt.gca().transAxes, fontsize=10 * multiplier)
    
    plt.xlabel('Track length (seconds)', fontsize=size * multiplier)
    plt.ylabel('Percentage', fontsize=size * multiplier)
    plt.legend(title='', fontsize=size * multiplier)
    ax = plt.gca()
    if xlimit is not None:
        ax.set_xlim(0, xlimit)
    else:
        ax.set_xlim(0, max_track_length)
    plt.show()



    




###########################################################

# def plot_histograms(data_df, feature, bins=100, separate=None, xlimit=None, small_multiples=False, palette='colorblind',
#                     use_kde=False, show_plot=True, master_dir=None, tick_interval=5, average='mean', order=None, grid=False):
#     """
#     Plot histograms or KDEs of a specified feature for each category in `separate`, with consistent binning.

#     Parameters
#     ----------
#     data_df : DataFrame
#         DataFrame containing track data with the specified feature and optionally a separating column.
#     feature : str
#         The feature to plot histograms for.
#     bins : int, optional
#         Number of bins for the histogram. Default is 100.
#     separate : str, optional
#         Column to separate the data by. If None, all data will be plotted together. Default is None.
#     xlimit : float, optional
#         Upper limit for the x-axis. Default is None.
#     small_multiples : bool, optional
#         Whether to plot each category separately as small multiples. Default is False.
#     palette : str, optional
#         Color palette for the plot. Default is 'colorblind'.
#     use_kde : bool, optional
#         Whether to use KDE plot instead of histogram. Default is False.
#     show_plot : bool, optional
#         Whether to display the plot in the notebook. Default is True.
#     master_dir : str, optional
#         The directory where the plots folder will be created and the plot will be saved. Default is None.
#     tick_interval : int, optional
#         Interval for x-axis ticks. Default is 5.
#     average : str, optional
#         Whether to draw 'mean' or 'median' line on the plot. Default is 'mean'.
#     order : list, optional
#         Specific order for the conditions. Default is None.
#     """
#     if master_dir is None:
#         master_dir = "plots"  # Use a default directory if not provided

#     if separate is not None:
#         # Ensure the column used for separation is of categorical type
#         if not pd.api.types.is_categorical_dtype(data_df[separate]):
#             data_df[separate] = data_df[separate].astype('category')

#         if order is not None:
#             # Ensure the data is ordered according to the specified order
#             data_df[separate] = pd.Categorical(data_df[separate], categories=order, ordered=True)

#     textpositionx = 0.6
#     textpositiony = 0.8

#     # Use the categories attribute to maintain the specified order
#     if separate is not None:
#         unique_categories = data_df[separate].cat.categories
#     else:
#         unique_categories = [None]

#     color_palette = sns.color_palette(palette, len(unique_categories))

#     # Determine global maximum y-value for consistent y-axis limits
#     global_max_y = 0

#     if small_multiples and separate is not None:
#         num_categories = len(unique_categories)
#         fig, axes = plt.subplots(num_categories, 1, figsize=(20, 6 * num_categories), sharex=True)

#         if num_categories == 1:
#             axes = [axes]  # To handle the case with only one subplot

#         for i, category in enumerate(unique_categories):
#             if pd.isna(category):
#                 continue
#             subset = data_df[data_df[separate] == category]
#             subsetvalues = subset[feature]

#             # Determine bin edges for the entire data range, including negative values
#             min_value = data_df[feature].min()
#             max_value = data_df[feature].max() if xlimit is None else xlimit
#             bin_edges = np.linspace(min_value, max_value, bins + 1)

#             # Plot histogram or KDE
#             if use_kde:
#                 sns.kdeplot(subsetvalues, fill=True, ax=axes[i], color=color_palette[i])
#                 current_max_y = axes[i].get_ylim()[1]  # Get the current maximum y-value from the plot
#             else:
#                 plot = sns.histplot(subsetvalues, bins=bin_edges, kde=False, ax=axes[i], stat="percent", color=color_palette[i])
#                 current_max_y = plot.get_ylim()[1]

#             # Update global maximum y-value
#             if current_max_y > global_max_y:
#                 global_max_y = current_max_y

#             # Plot average line
#             if average == 'mean':
#                 avg_value = subsetvalues.mean()
#                 axes[i].axvline(avg_value, color='black', linestyle='--')
#                 axes[i].text(textpositionx, textpositiony, f"Mean: {avg_value:.2f}", transform=axes[i].transAxes, fontsize=16)
#             elif average == 'median':
#                 avg_value = subsetvalues.median()
#                 axes[i].axvline(avg_value, color='black', linestyle='--')
#                 axes[i].text(textpositionx, textpositiony, f"Median: {avg_value:.2f}", transform=axes[i].transAxes, fontsize=16)

#             axes[i].set_title(f'{category}', fontsize=16)
#             axes[i].tick_params(axis='both', which='major', labelsize=16)
#             axes[i].set_xlabel(f'{feature}', fontsize=16)
#             axes[i].set_ylabel('Percentage', fontsize=16)

#             # Apply x-axis limits
#             x_lower = min_value if min_value < 0 else 0
#             axes[i].set_xlim(x_lower, max_value)

#         # Set common y-axis limits for all subplots
#         for ax in axes:
#             ax.set_ylim(0, global_max_y)

#         plt.tight_layout()

#     else:
#         plt.figure(figsize=(20, 12))
#         sns.set_context("notebook", rc={"xtick.labelsize": 16, "ytick.labelsize": 16})

#         min_value = data_df[feature].min()
#         max_value = data_df[feature].max() if xlimit is None else xlimit
#         bin_edges = np.linspace(min_value, max_value, bins + 1)

#         if separate is None:
#             subsetvalues = data_df[feature]

#             # Plot histogram or KDE
#             if use_kde:
#                 sns.kdeplot(subsetvalues, fill=True, alpha=0.5, color=color_palette[0])
#             else:
#                 sns.histplot(subsetvalues, bins=bin_edges, kde=False, alpha=0.5, stat="percent", color=color_palette[0])

#             # Plot average line
#             if average == 'mean':
#                 avg_value = subsetvalues.mean()
#                 plt.axvline(avg_value, color='r', linestyle='--')
#                 plt.text(textpositionx, textpositiony, f"Overall Mean: {avg_value:.2f}", transform=plt.gca().transAxes, fontsize=16)
#             elif average == 'median':
#                 avg_value = subsetvalues.median()
#                 plt.axvline(avg_value, color='b', linestyle='--')
#                 plt.text(0.4, 0.6, f"Overall Median: {avg_value:.2f}", transform=plt.gca().transAxes, fontsize=16)

#         else:
#             for i, category in enumerate(unique_categories):
#                 if pd.isna(category):
#                     continue
#                 subset = data_df[data_df[separate] == category]
#                 subsetvalues = subset[feature]

#                 # Plot histogram or KDE
#                 if use_kde:
#                     sns.kdeplot(subsetvalues, fill=True, label=category, alpha=0.5, color=color_palette[i])
#                 else:
#                     sns.histplot(subsetvalues, bins=bin_edges, kde=False, label=category, alpha=0.5, stat="percent", color=color_palette[i])

#                 # Plot average line
#                 if average == 'mean':
#                     avg_value = subsetvalues.mean()
#                     plt.axvline(avg_value, color=color_palette[i], linestyle='--')
#                 elif average == 'median':
#                     avg_value = subsetvalues.median()
#                     plt.axvline(avg_value, color=color_palette[i], linestyle='--')

#                 number_of_tracks = len(subset['unique_id'].unique())
#                 shift = i * 0.05
#                 plt.text(textpositionx, textpositiony - shift, f"{category}: {average.capitalize()}: {avg_value:.2f} from {number_of_tracks} tracks", transform=plt.gca().transAxes, fontsize=16)

#         plt.xlabel(f'{feature}', fontsize=16)
#         plt.ylabel('Percentage', fontsize=16)
#         plt.legend(title='', fontsize=16)
#         ax = plt.gca()

#         # Apply x-axis limits
#         x_lower = min_value if min_value < 0 else 0
#         ax.set_xlim(x_lower, max_value)

#         ax.set_xticks(np.arange(0, max_value + 1, tick_interval))  # Ensure ticks are at integer intervals
#         if grid:
#             ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)  # Add faint gridlines

#     # Create directory for plots if it doesn't exist
#     plots_dir = os.path.join(master_dir, 'plots')
#     os.makedirs(plots_dir, exist_ok=True)
#     histodir = os.path.join(plots_dir, 'histograms')
#     os.makedirs(histodir, exist_ok=True)

#     # Generate filename
#     kde_text = 'kde' if use_kde else 'histogram'
#     average_text = f'{average}'
#     multitext = 'small_multiples' if small_multiples else 'single_plot'

#     filename = f"{histodir}/{kde_text}_{feature}_{average_text}_{multitext}.png"
    
#     # Save plot
#     plt.savefig(filename, bbox_inches='tight')
    
#     # Show plot if specified
#     if show_plot:
#         plt.show()
#     else:
#         plt.close()


def plot_histograms(data_df, feature, bins=100, separate=None, xlimit=None, small_multiples=False, palette='colorblind',
                    use_kde=False, show_plot=True, master_dir=None, tick_interval=5, average='mean', order=None, 
                    grid=False, background='white', transparent=False, line_color='black', font_size=16):
    """
    Plot histograms or KDEs of a specified feature for each category in `separate`, with consistent binning.

    Parameters
    ----------
    data_df : DataFrame
        DataFrame containing track data with the specified feature and optionally a separating column.
    feature : str
        The feature to plot histograms for.
    bins : int, optional
        Number of bins for the histogram. Default is 100.
    separate : str, optional
        Column to separate the data by. If None, all data will be plotted together. Default is None.
    xlimit : float, optional
        Upper limit for the x-axis. Default is None.
    small_multiples : bool, optional
        Whether to plot each category separately as small multiples. Default is False.
    palette : str, optional
        Color palette for the plot. Default is 'colorblind'.
    use_kde : bool, optional
        Whether to use KDE plot instead of histogram. Default is False.
    show_plot : bool, optional
        Whether to display the plot in the notebook. Default is True.
    master_dir : str, optional
        The directory where the plots folder will be created and the plot will be saved. Default is None.
    tick_interval : int, optional
        Interval for x-axis ticks. Default is 5.
    average : str, optional
        Whether to draw 'mean' or 'median' line on the plot. Default is 'mean'.
    order : list, optional
        Specific order for the conditions. Default is None.
    background : str, tuple, optional
        Background color as color name, RGB tuple, or hex code. Default is 'white'.
    transparent : bool, optional
        If True, makes the plot fully transparent except for histogram bars and lines. Default is False.
    line_color : str, optional
        Color of all plot lines, including axis lines, grid lines, labels, and average lines. Default is 'black'.
    font_size : int, optional
        Font size for all text in the plot. Default is 16.
    """
    if master_dir is None:
        master_dir = "plots"  # Use a default directory if not provided

    # Set up background and transparency
    if transparent:
        figure_background = 'none'
        axis_background = (0, 0, 0, 0)  # Transparent RGBA for axes
    else:
        figure_background = background if is_color_like(background) else 'white'
        axis_background = figure_background

    # Prepare data for separation by category if specified
    if separate is not None:
        if not pd.api.types.is_categorical_dtype(data_df[separate]):
            data_df[separate] = data_df[separate].astype('category')
        if order is not None:
            data_df[separate] = pd.Categorical(data_df[separate], categories=order, ordered=True)
        unique_categories = data_df[separate].cat.categories
    else:
        unique_categories = [None]

    # Create a color palette for the categories
    color_palette = sns.color_palette(palette, len(unique_categories))

    # Set up figure and axes for small multiples or single plot
    if small_multiples and separate is not None:
        fig, axes = plt.subplots(len(unique_categories), 1, figsize=(20, 6 * len(unique_categories)), 
                                 sharex=True, facecolor=figure_background)
        if len(unique_categories) == 1:
            axes = [axes]  # Handle single subplot case
    else:
        fig, ax = plt.subplots(figsize=(20, 12), facecolor=figure_background)
        axes = [ax]  # Use a single axis in a list for consistency

    # Initial vertical offset for median text
    text_position_y = 0.8

    for i, category in enumerate(unique_categories):
        ax = axes[i] if small_multiples and separate else axes[0]
        subsetvalues = data_df[data_df[separate] == category][feature] if category is not None else data_df[feature]

        # Set axis background color
        ax.set_facecolor(axis_background)

        # Determine bin edges for consistent binning
        min_value = data_df[feature].min()
        max_value = data_df[feature].max() if xlimit is None else xlimit
        bin_edges = np.linspace(min_value, max_value, bins + 1)

        # Plot histogram or KDE
        if use_kde:
            sns.kdeplot(subsetvalues, fill=True, ax=ax, color=color_palette[i], linewidth=1.5, label=category)
        else:
            sns.histplot(subsetvalues, bins=bin_edges, kde=False, ax=ax, color=color_palette[i], alpha=0.5, 
                         stat="percent", label=category)

        # Plot average line
        avg_value = subsetvalues.mean() if average == 'mean' else subsetvalues.median()
        ax.axvline(avg_value, color=line_color, linestyle='--')

        # Display category and value below each other, adjusting vertical position for each category
        ax.text(0.8, text_position_y - (i * 0.05), f"{category}: {avg_value:.2f}", 
                transform=ax.transAxes, fontsize=font_size, color=line_color, ha='center')

        # Customize x and y labels, and tick parameters
        ax.set_xlabel(feature, fontsize=font_size, color=line_color)
        ax.set_ylabel('Percentage', fontsize=font_size, color=line_color)
        ax.tick_params(axis='both', which='both', color=line_color, labelcolor=line_color, labelsize=font_size)
        
        # Set grid and axis line colors
        if grid:
            ax.grid(True, linestyle='--', linewidth=0.5, color=line_color if not transparent else (0, 0, 0, 0.5), alpha=0.7, axis='y')
        
        # Set axis spine colors
        for spine in ax.spines.values():
            spine.set_edgecolor(line_color)
            if transparent:
                spine.set_alpha(0.5)
        
        # Set x-axis limits
        ax.set_xlim(0, max_value)
        ax.set_xticks(np.arange(0, max_value + 1, tick_interval))

        # Title for each subplot if small multiples
        if small_multiples and separate:
            ax.set_title(f'{category}', fontsize=font_size, color=line_color)

    # Legend and title handling for non-multiples
    if separate and not small_multiples:
        ax.legend(title=separate, fontsize=font_size, title_fontsize=font_size)
    elif separate:
        fig.suptitle(f"{feature} distribution by {separate}", fontsize=font_size + 4, color=line_color)

    # Save and show plot
    plot_dir = os.path.join(master_dir, 'plots', 'histograms')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f"{plot_dir}/{feature}_histogram.png", bbox_inches='tight', transparent=transparent)

    if show_plot:
        plt.show()
    else:
        plt.close()







        #### To be refined::: #############



def plot_trajectory(traj, colorby='particle', mpp=None, label=False,
                    superimpose=None, cmap=None, ax=None, t_column=None,
                    pos_columns=None, plot_style={}, **kwargs):
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
        t_column = 'frame'
    if pos_columns is None:
        pos_columns = ['x', 'y']
    if len(traj) == 0:
        raise ValueError("DataFrame of trajectories is empty.")
    
    _plot_style = dict(linewidth=1)
    _plot_style.update(**plot_style)

    if ax is None:
        ax = plt.gca()
        
    # Axes labels
    if mpp is None:
        ax.set_xlabel(f'{pos_columns[0]} [px]')
        ax.set_ylabel(f'{pos_columns[1]} [px]')
        mpp = 1.  # for computations of image extent below
    else:
        ax.set_xlabel(f'{pos_columns[0]} [μm]')
        ax.set_ylabel(f'{pos_columns[1]} [μm]')
        
    # Background image
    if superimpose is not None:
        ax.imshow(superimpose, cmap=plt.cm.gray,
                  origin='lower', interpolation='nearest',
                  vmin=kwargs.get('vmin'), vmax=kwargs.get('vmax'))
        ax.set_xlim(-0.5 * mpp, (superimpose.shape[1] - 0.5) * mpp)
        ax.set_ylim(-0.5 * mpp, (superimpose.shape[0] - 0.5) * mpp)
    
    # Trajectories
    if colorby == 'particle':
        # Unstack particles into columns.
        unstacked = traj.set_index(['particle', t_column])[pos_columns].unstack()
        for i, trajectory in unstacked.iterrows():
            ax.plot(mpp * trajectory[pos_columns[0]], mpp * trajectory[pos_columns[1]], **_plot_style)
    elif colorby == 'frame':
        # Read http://www.scipy.org/Cookbook/Matplotlib/MulticoloredLine
        x = traj.set_index([t_column, 'particle'])[pos_columns[0]].unstack()
        y = traj.set_index([t_column, 'particle'])[pos_columns[1]].unstack()
        color_numbers = traj[t_column].values / float(traj[t_column].max())
        for particle in x:
            points = np.array([x[particle].values, y[particle].values]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap)
            lc.set_array(color_numbers)
            ax.add_collection(lc)
            ax.set_xlim(x.apply(np.min).min(), x.apply(np.max).max())
            ax.set_ylim(y.apply(np.min).min(), y.apply(np.max).max())
    
    if label:
        unstacked = traj.set_index([t_column, 'particle'])[pos_columns].unstack()
        first_frame = int(traj[t_column].min())
        coords = unstacked.fillna(method='backfill').stack().loc[first_frame]
        for particle_id, coord in coords.iterrows():
            ax.text(*coord.tolist(), s="%d" % particle_id,
                    horizontalalignment='center',
                    verticalalignment='center')

    ax.invert_yaxis()
    return ax

def batch_plot_trajectories(master_folder, traj_df, batch=True, filename=None, colorby='particle', mpp=None, label=False, cmap=None):
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
    data_folder = os.path.join(master_folder, 'data')
    vis_folder = os.path.join(master_folder, 'visualization/trajectories')
    os.makedirs(vis_folder, exist_ok=True)

    if batch:
        for condition in os.listdir(data_folder):
            condition_folder = os.path.join(data_folder, condition)
            if os.path.isdir(condition_folder):
                for file in os.listdir(condition_folder):
                    if file.endswith('.tif'):
                        filepath = os.path.join(condition_folder, file)
                        subset_traj_df = traj_df[traj_df['filename'] == file]
                        if not subset_traj_df.empty:
                            frames = pims.open(filepath)
                            frame = frames[0]
                            fig, ax = plt.subplots()
                            plot_trajectory(subset_traj_df, colorby=colorby, mpp=mpp, label=label, superimpose=frame, cmap=cmap, ax=ax)
                            plt.savefig(os.path.join(vis_folder, f'{condition}_{file}.png'))
                            plt.close(fig)
    else:
        if filename is not None:
            filepath = os.path.join(data_folder, filename)
            subset_traj_df = traj_df[traj_df['filename'] == filename]
            if not subset_traj_df.empty:
                frames = pims.open(filepath)
                frame = frames[0]
                fig, ax = plt.subplots()
                plot_trajectory(subset_traj_df, colorby=colorby, mpp=mpp, label=label, superimpose=frame, cmap=cmap, ax=ax)
                plt.show()
        else:
            print("Please provide a filename when batch is set to False.")

# Usage example
# master_folder = 'path_to__master_folder'
# traj_df = pd.read_csv('path_to_dataframe.csv')
# batch_plot_trajectories(master_folder, traj_df, batch=True)
# batch_plot_trajectories(master_folder, traj_df, batch=False, filename='file.tif')


def plot_particle_trajectory(ax, particle_df, particle_id, condition, plot_size=None):
    
    x_min, x_max = particle_df['x_um'].min(), particle_df['x_um'].max()
    y_min, y_max = particle_df['y_um'].min(), particle_df['y_um'].max()
    
    if plot_size is None:
        max_range = max(x_max - x_min, y_max - y_min)
        plot_size = max_range * 1.1  # Add 10% padding
    
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    xlim = (x_center - plot_size/2, x_center + plot_size/2)
    ylim = (y_center + plot_size/2, y_center - plot_size/2)  # Inverted y-axis
    
    scatter = ax.scatter(particle_df['x_um'], particle_df['y_um'], 
                         c=particle_df['time_s'], cmap='viridis', s=30)
    ax.plot(particle_df['x_um'], particle_df['y_um'], '-', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('X position (µm)', fontsize=8)
    ax.set_ylabel('Y position (µm)', fontsize=8)
    ax.set_title(f'{condition}: Particle {particle_id}', fontsize=10)
    
    ax.invert_yaxis()
    ax.set_aspect('equal')
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_label('Time (s)', fontsize=8)
    cbar.ax.tick_params(labelsize=6)
    
    return scatter


def plot_multiple_particles(combined_df, particles_per_condition=2, plot_size=None):
    conditions = combined_df['condition'].unique()
    num_conditions = len(conditions)
    total_particles = num_conditions * particles_per_condition
    
    # Calculate the grid size
    grid_size = int(np.ceil(np.sqrt(total_particles)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(4*grid_size, 4*grid_size))
    fig.suptitle('Particle Trajectories by Condition', fontsize=16)
    
    # Flatten the axes array for easier iteration
    axes_flat = axes.flatten()
    
    plot_index = 0
    for condition in conditions:
        condition_df = combined_df[combined_df['condition'] == condition]
        particles = np.random.choice(condition_df['unique_id'].unique(), particles_per_condition, replace=False)
        
        for particle_id in particles:
            if plot_index >= len(axes_flat):
                break
            
            ax = axes_flat[plot_index]
            particle_df = condition_df[condition_df['unique_id'] == particle_id]
            
            plot_particle_trajectory(ax, particle_df, particle_id, condition, plot_size)
            plot_index += 1
    
    # Remove extra subplots
    for i in range(plot_index, len(axes_flat)):
        fig.delaxes(axes_flat[i])
    
    plt.tight_layout()
    plt.show()


    # Build this into a function



def load_image(file_path):
    return io.imread(file_path)

def load_tracks(df, filename):
    tracks = df[df['filename'] == filename]
    return tracks

def get_condition_from_filename(df, filename):
    try:
        condition = df[df['filename'] == filename]['condition'].iloc[0]
    except IndexError:
        print(f"Error: Filename '{filename}' not found in the dataframe.")
        raise
    return condition




# def napari_visualize_image_with_tracks(tracked_filename, tracks_df, master_dir):
#     # Get the condition from the dataframe based on the tracked filename
#     condition = get_condition_from_filename(tracks_df, tracked_filename)
    
#     # Construct the full file path by removing '_tracked' and adding '.tif'
#     image_filename = tracked_filename.replace('_tracked', '') + '.tif'
#     image_path = os.path.join(master_dir, condition, image_filename)
    
#     # Load the image
#     image = load_image(image_path)

#     # Load the tracks
#     tracks = load_tracks(tracks_df, tracked_filename)

#     print(tracks.columns)

#     tracks_new_df = tracks[["particle", "frame", "y", "x"]]

#     # Extract x, y, and frame information for the tracks
#     coords = np.array([tracks['x'].values, tracks['y'].values, tracks['frame'].values]).T
    
#     # Start Napari viewer
#     viewer = napari.Viewer()

#     # Add image layer
#     viewer.add_image(image, name=f'Raw {tracked_filename}')

#     # Add tracks layer
#     viewer.add_tracks(tracks_new_df, name=f'Tracks{tracked_filename}')

#     napari.run()



# import os
# import numpy as np
# import napari

# def napari_visualize_image_with_tracksdev(tracked_filename, tracks_df, master_dir):
#     # Get the condition from the dataframe based on the tracked filename
#     condition = get_condition_from_filename(tracks_df, tracked_filename)
    
#     # Construct the full file path by removing '_tracked' and adding '.tif'
#     image_filename = tracked_filename.replace('_tracked', '') + '.tif'
#     image_path = os.path.join(master_dir, condition, image_filename)
    
#     # Load the image
#     image = load_image(image_path)

#     # Load the tracks
#     tracks = load_tracks(tracks_df, tracked_filename)

#     print(tracks.columns)

#     # Prepare the tracks DataFrame for Napari
#     tracks_new_df = tracks[["particle", "frame", "y", "x"]]

#     # Ensure 'speed_um_s' is in the DataFrame
#     if 'speed_um_s' not in tracks.columns:
#         raise ValueError("The tracks DataFrame does not contain the 'speed_um_s' column.")

#     # Extract coordinates
#     coords = np.array([tracks['x'].values, tracks['y'].values, tracks['frame'].values]).T

#     # Prepare the features dictionary
#     features = {'speed_um_s': tracks['speed_um_s'].values}

#     # Start Napari viewer
#     viewer = napari.Viewer()

#     # Add image layer
#     viewer.add_image(image, name=f'Raw {tracked_filename}')

#     # Add tracks layer, specifying features to use speed for coloring
#     viewer.add_tracks(tracks_new_df.to_numpy(), features=features, name=f'Tracks{tracked_filename}', color_by='speed_um_s')

#     napari.run()


# def napari_visualize_image_with_tracksdev(tracks_df, master_dir=config.MASTER + 'data', condition=None, cell=None):
    
#     print('The master directory is:', master_dir)
    
#     if condition is None:
#         conditionlist = tracks_df['condition'].unique()
#         # make a random choice of condition from the list
#         condition = np.random.choice(conditionlist)
#     else: 
#         condition = condition

#     if cell is None:
#         celllist = tracks_df[tracks_df['condition'] == condition]['filename'].unique()
#         cell = np.random.choice(celllist)
#     else:
#         cell = cell

#     # Get the condition from the dataframe based on the tracked filename
#     # condition = get_condition_from_filename(tracks_df, tracked_filename)
    
#     # Construct the full file path by removing '_tracked' and adding '.tif'
#     image_filename = cell.replace('_tracked', '') + '.tif'
#     image_path = os.path.join(master_dir, condition, image_filename)
    
#     # Load the image
#     image = load_image(image_path)

#     # Load the tracks
#     tracks = load_tracks(tracks_df, cell)

#     print(tracks.columns)

#     # Prepare the tracks DataFrame for Napari
#     tracks_new_df = tracks[["particle", "frame", "y", "x"]]

#     # Ensure 'speed_um_s' is in the DataFrame
#     if 'speed_um_s' not in tracks.columns:
#         raise ValueError("The tracks DataFrame does not contain the 'speed_um_s' column.")

#     # Extract coordinates
#     coords = np.array([tracks['x'].values, tracks['y'].values, tracks['frame'].values]).T

#     # Prepare the features dictionary
#     features = {'speed_um_s': tracks['speed_um_s'].values}

#     # Start Napari viewer
#     viewer = napari.Viewer()

#     # Add image layer
#     viewer.add_image(image, name=f'Raw {cell}')

#     # Add tracks layer, specifying features to use speed for coloring
#     viewer.add_tracks(tracks_new_df.to_numpy(), features=features, name=f'Tracks{cell}', color_by='speed_um_s')

#     napari.run()



# from napari_animation import Animation

# def save_movie(viewer, tracks, feature='particle', save_path='movie.mov'):
#     animation = Animation(viewer)

#     # Set the display to 2D
#     viewer.dims.ndisplay = 2
    
#     # Automatically set the keyframes for the start, middle, and end
#     num_frames = len(tracks['frame'].unique())
    
#     # Start keyframe (frame 0)
#     viewer.dims.set_point(0, 0)
#     animation.capture_keyframe()

#     # Middle keyframe (middle frame)
#     middle_frame = num_frames // 2
#     viewer.dims.set_point(0, middle_frame)
#     animation.capture_keyframe()

#     # End keyframe (last frame)
#     viewer.dims.set_point(0, num_frames - 1)
#     animation.capture_keyframe()

#     # Save the animation to the specified path
#     animation.animate(save_path, canvas_only=True)  # canvas_only=True to exclude controls


# def napari_visualize_image_with_tracksdev(tracks_df, master_dir=config.MASTER, condition=None, cell=None, location=None, save_movie_flag=False, feature='particle'):
    
#     master_dir = config.MASTER + 'data'
#     movie_dir = config.MASTER + 'movies'
    

#     print('The master directory is:', master_dir)
#     if save_movie_flag:
#         print('The movie directory is:', movie_dir)
    
#     # Handle location input
#     locationlist = tracks_df['Location'].unique()
#     if isinstance(location, int):
#         location = locationlist[location]
#     elif isinstance(location, str):
#         if location not in locationlist:
#             raise ValueError(f"Location '{location}' not found in available locations: {locationlist}")
#     elif location is None:
#         location = np.random.choice(locationlist)
#     else:
#         raise ValueError("Location must be a string, integer, or None.")
    
#     # Filter the dataframe by the selected location
#     filtered_tracks_df = tracks_df[tracks_df['Location'] == location]
    
#     # Handle condition input
#     conditionlist = filtered_tracks_df['condition'].unique()
#     if isinstance(condition, int):
#         condition = conditionlist[condition]
#     elif isinstance(condition, str):
#         if condition not in conditionlist:
#             raise ValueError(f"Condition '{condition}' not found in available conditions for location '{location}': {conditionlist}")
#     elif condition is None:
#         condition = np.random.choice(conditionlist)
#     else:
#         raise ValueError("Condition must be a string, integer, or None.")
    
#     # Handle cell input
#     celllist = filtered_tracks_df[filtered_tracks_df['condition'] == condition]['filename'].unique()
#     if isinstance(cell, int):
#         cell = celllist[cell]
#     elif isinstance(cell, str):
#         if cell not in celllist:
#             raise ValueError(f"Cell '{cell}' not found in available cells for condition '{condition}' and location '{location}': {celllist}")
#     elif cell is None:
#         cell = np.random.choice(celllist)
#     else:
#         raise ValueError("Cell must be a string, integer, or None.")

#     # Construct the full file path by removing '_tracked' and adding '.tif'
#     image_filename = cell.replace('_tracked', '') + '.tif'
#     image_path = os.path.join(master_dir, condition, image_filename)
    
#     # Load the image
#     image = load_image(image_path)

#     # Load the tracks
#     tracks = load_tracks(filtered_tracks_df, cell)

#     print(tracks.columns)

#     # Prepare the tracks DataFrame for Napari
#     tracks_new_df = tracks[["particle", "frame", "y", "x"]]

#     # Start Napari viewer
#     viewer = napari.Viewer()

#     # Add image layer
#     viewer.add_image(image, name=f'Raw {cell}')

#     features = {'speed_um_s': tracks['speed_um_s'].values, 'particle': tracks['particle'].values}

   

#     # Add tracks layer, using 'particle' for coloring
#     viewer.add_tracks(tracks_new_df.to_numpy(), features=features, name=f'Tracks {cell}', color_by='particle')


#     #####

#     ### This is a placeholder for a kymograph function, if required (time on x axis, feature as colour)

#     #####

#     # Save the movie if specified
#     if save_movie_flag:
#         movies_dir = os.path.join(movie_dir, 'movies')
#         os.makedirs(movies_dir, exist_ok=True)
#         movie_path = os.path.join(movies_dir, f'{condition}_{cell}.mov')
#         save_movie(viewer, tracks_new_df, feature='particle', save_path=movie_path)
    
#     napari.run()



def save_movie(viewer, tracks, feature='particle', save_path='movie.mov'):
    animation = Animation(viewer)

    # Set the display to 2D
    viewer.dims.ndisplay = 2
    
    # Automatically set the keyframes for the start, middle, and end
    num_frames = len(tracks['frame'].unique())
    
    # Start keyframe (frame 0)
    viewer.dims.set_point(0, 0)
    animation.capture_keyframe()

    # Middle keyframe (middle frame)
    middle_frame = num_frames // 2
    viewer.dims.set_point(0, middle_frame)
    animation.capture_keyframe()

    # End keyframe (last frame)
    viewer.dims.set_point(0, num_frames - 1)
    animation.capture_keyframe()

    # Save the animation to the specified path
    animation.animate(save_path, canvas_only=True)  # canvas_only=True to exclude controls


def napari_visualize_image_with_tracksdev(tracks_df, master_dir=config.MASTER, condition=None, cell=None, location=None, save_movie_flag=False, feature='particle'):
    
    master_dir = config.MASTER + 'data'
    movie_dir = config.MASTER + 'movies'

    print('The master directory is:', master_dir)
    if save_movie_flag:
        print('The movie directory is:', movie_dir)
    
    # Handle location input
    locationlist = tracks_df['Location'].unique()
    if isinstance(location, int):
        location = locationlist[location]
    elif isinstance(location, str):
        if location not in locationlist:
            raise ValueError(f"Location '{location}' not found in available locations: {locationlist}")
    elif location is None:
        location = np.random.choice(locationlist)
    else:
        raise ValueError("Location must be a string, integer, or None.")
    
    # Filter the dataframe by the selected location
    filtered_tracks_df = tracks_df[tracks_df['Location'] == location]
    
    # Handle condition input
    conditionlist = filtered_tracks_df['condition'].unique()
    if isinstance(condition, int):
        condition = conditionlist[condition]
    elif isinstance(condition, str):
        if condition not in conditionlist:
            raise ValueError(f"Condition '{condition}' not found in available conditions for location '{location}': {conditionlist}")
    elif condition is None:
        condition = np.random.choice(conditionlist)
    else:
        raise ValueError("Condition must be a string, integer, or None.")
    
    # Handle cell input
    celllist = filtered_tracks_df[filtered_tracks_df['condition'] == condition]['filename'].unique()
    if isinstance(cell, int):
        cell = celllist[cell]
    elif isinstance(cell, str):
        if cell not in celllist:
            raise ValueError(f"Cell '{cell}' not found in available cells for condition '{condition}' and location '{location}': {celllist}")
    elif cell is None:
        cell = np.random.choice(celllist)
    else:
        raise ValueError("Cell must be a string, integer, or None.")

    # Construct the full file path by removing '_tracked' and adding '.tif'
    image_filename = cell.replace('_tracked', '') + '.tif'
    image_path = os.path.join(master_dir, condition, image_filename)
    
    # Load the image
    image = load_image(image_path)

    # Load the tracks
    tracks = load_tracks(filtered_tracks_df, cell)

    print(tracks.columns)

    # Prepare the tracks DataFrame for Napari
    tracks_new_df = tracks[["particle", "frame", "y", "x"]]

    # Include 'particle' and all features from config.FEATURES
    features_dict = {'particle': tracks['particle'].values}
    features_dict.update({feature: tracks[feature].values for feature in config.FEATURES if feature in tracks.columns})

    # Start Napari viewer
    viewer = napari.Viewer()

    # Add image layer
    viewer.add_image(image, name=f'Raw {cell}')

    # Add tracks layer, using 'particle' for coloring, with additional features
    viewer.add_tracks(tracks_new_df.to_numpy(), features=features_dict, name=f'Tracks {cell}', color_by=feature)

    # Save the movie if specified
    if save_movie_flag:
        movies_dir = os.path.join(movie_dir, 'movies')
        os.makedirs(movies_dir, exist_ok=True)
        movie_path = os.path.join(movies_dir, f'{condition}_{cell}.mov')
        save_movie(viewer, tracks_new_df, feature=feature, save_path=movie_path)
    
    napari.run()




def napari_visualize_image_with_tracksdev2(tracks_df, master_dir=config.MASTER, condition=None, cell=None, location=None, save_movie_flag=False, feature='particle'):
    
    master_dir = config.MASTER + 'data'
    movie_dir = config.MASTER + 'movies'

    print('The master directory is:', master_dir)
    if save_movie_flag:
        print('The movie directory is:', movie_dir)
    
    # Handle location input
    locationlist = tracks_df['Location'].unique()
    if isinstance(location, int):
        location = locationlist[location]
    elif isinstance(location, str):
        if location not in locationlist:
            raise ValueError(f"Location '{location}' not found in available locations: {locationlist}")
    elif location is None:
        np.random.shuffle(locationlist)  # Shuffle the list to make random selection
        for loc in locationlist:
            if loc in locationlist:
                location = loc
                break
        if location is None:
            raise ValueError(f"No valid location found in available locations: {locationlist}")
    else:
        raise ValueError("Location must be a string, integer, or None.")
    
    # Filter the dataframe by the selected location
    filtered_tracks_df = tracks_df[tracks_df['Location'] == location]
    
    # Handle condition input
    conditionlist = filtered_tracks_df['condition'].unique()
    if isinstance(condition, int):
        condition = conditionlist[condition]
    elif isinstance(condition, str):
        if condition not in conditionlist:
            raise ValueError(f"Condition '{condition}' not found in available conditions for location '{location}': {conditionlist}")
    elif condition is None:
        np.random.shuffle(conditionlist)  # Shuffle the list to make random selection
        for cond in conditionlist:
            if cond in conditionlist:
                condition = cond
                break
        if condition is None:
            raise ValueError(f"No valid condition found for location '{location}': {conditionlist}")
    else:
        raise ValueError("Condition must be a string, integer, or None.")
    
    # Handle cell input
    celllist = filtered_tracks_df[filtered_tracks_df['condition'] == condition]['filename'].unique()
    if isinstance(cell, int):
        cell = celllist[cell]
    elif isinstance(cell, str):
        if cell not in celllist:
            raise ValueError(f"Cell '{cell}' not found in available cells for condition '{condition}' and location '{location}': {celllist}")
    elif cell is None:
        np.random.shuffle(celllist)  # Shuffle the list to make random selection
        for c in celllist:
            if c in celllist:
                cell = c
                break
        if cell is None:
            raise ValueError(f"No valid cell found for condition '{condition}' and location '{location}': {celllist}")
    else:
        raise ValueError("Cell must be a string, integer, or None.")

    # Construct the full file path by removing '_tracked' and adding '.tif'
    image_filename = cell.replace('_tracked', '') + '.tif'
    image_path = os.path.join(master_dir, condition, image_filename)
    
    # Load the image
    image = load_image(image_path)

    # Load the tracks
    tracks = load_tracks(filtered_tracks_df, cell)

    print(tracks.columns)

    # Prepare the tracks DataFrame for Napari
    tracks_new_df = tracks[["particle", "frame", "y", "x"]]

    # Include 'particle' and all features from config.FEATURES
    features_dict = {'particle': tracks['particle'].values}
    features_dict.update({feature: tracks[feature].values for feature in config.FEATURES if feature in tracks.columns})

    # Start Napari viewer
    viewer = napari.Viewer()

    # Add image layer
    viewer.add_image(image, name=f'Raw {cell}')

    # Add tracks layer, using 'particle' for coloring, with additional features
    viewer.add_tracks(tracks_new_df.to_numpy(), features=features_dict, name=f'Tracks {cell}', color_by=feature)

    # Save the movie if specified
    if save_movie_flag:
        movies_dir = os.path.join(movie_dir, 'movies')
        os.makedirs(movies_dir, exist_ok=True)
        movie_path = os.path.join(movies_dir, f'{condition}_{cell}.mov')
        save_movie(viewer, tracks_new_df, feature=feature, save_path=movie_path)
    
    napari.run





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




# def plot_time_series(data_df, factor_col='speed_um_s', absolute=True, separate_by='condition', palette='colorblind', meanormedian='mean', multiplot=False, talk=False, bootstrap=True, show_plot=True, master_dir=None):
#     """
#     Plot time series of a specified factor, with mean/median as a line and confidence intervals as shaded areas.
    
#     Parameters
#     ----------
#     data_df : DataFrame
#         DataFrame containing the time series data.
#     factor_col : str, optional
#         The column representing the factor to be plotted on the y-axis. Default is 'speed_um_s'.
#     absolute : bool, optional
#         Whether to use absolute time values or time zeroed values. Default is True.
#     separate_by : str, optional
#         Column to separate the data by, for coloring. If None, all data will be plotted together. Default is None.
#     palette : str, optional
#         Color palette for the plot. Default is 'colorblind'.
#     meanormedian : str, optional
#         Whether to use mean or median for aggregation. Default is 'mean'.
#     multiplot : bool, optional
#         Whether to generate separate small multiple plots for each category. Default is False.
#     talk : bool, optional
#         Whether to set the figure size to the original large size or a smaller size. Default is False.
#     bootstrap : bool, optional
#         Whether to use bootstrapping for confidence intervals. Default is True.
#     show_plot : bool, optional
#         Whether to display the plot in the notebook. Default is True.
#     master_dir : str, optional
#         The directory where the plots folder will be created and the plot will be saved. Default is None.
#     """
#     # max_time = data_df['time_s'].max()
#     # max_time_zeroed = data_df['time_s_zeroed'].max()
#     xmin=0.2 #### A FIX FOR NOW because really this should be the same as the shortest track (filteres to 0.2 s during filterstubs)
    
#     if master_dir is None:
#         master_dir = config.master  # Use the master directory from config if not provided

#     if not absolute:
#         time_col = 'time_s_zeroed'
#         max_time_zeroed = data_df['time_s_zeroed'].max()
#         x_label = 'Time zeroed (s)'
#         xmax = max_time_zeroed
#     else:
#         time_col = 'time_s'
#         max_time = data_df['time_s'].max()
#         x_label = 'Time (s)'
#         xmax = max_time

#     unique_categories = data_df[separate_by].unique() if separate_by else [None]
#     color_palette = sns.color_palette(palette, len(unique_categories))
    
#     # Set figure size and font size based on the `talk` parameter
#     if talk:
#         fig_size = (40, 12)
#         font_size = 35
#     else:
#         if multiplot and separate_by:
#             fig_size = (10, 5 * len(unique_categories))
#         else:
#             fig_size = (5, 3)
#         font_size = 14
    
#     sns.set_context("notebook", rc={"lines.linewidth": 2.5, "font.size": font_size, "axes.titlesize": font_size, "axes.labelsize": font_size, "xtick.labelsize": font_size, "ytick.labelsize": font_size})
    
#     if multiplot and separate_by:
#         fig, axes = plt.subplots(len(unique_categories), 1, figsize=fig_size, sharex=True)
        
#         for i, category in enumerate(unique_categories):
#             ax = axes[i] if len(unique_categories) > 1 else axes
#             subset = data_df[data_df[separate_by] == category]
#             times = subset[time_col]
#             factors = subset[factor_col]

#             if meanormedian == 'mean':
#                 avg_factors = subset.groupby(time_col)[factor_col].mean()
#                 ci_func = bootstrap_ci_mean if bootstrap else lambda x: sem(x) * 1.96
#             else:
#                 avg_factors = subset.groupby(time_col)[factor_col].median()
#                 ci_func = bootstrap_ci_median if bootstrap else lambda x: sem(x) * 1.96

#             ci = subset.groupby(time_col)[factor_col].apply(ci_func)

#             color = color_palette[i]
#             label = category

#             # Exclude the first time point (time zero)
#             valid_indices = avg_factors.index > 0

#             ax.plot(avg_factors.index[valid_indices], avg_factors.values[valid_indices], label=label, color=color, linewidth=2.5)
#             ax.fill_between(avg_factors.index[valid_indices], (avg_factors - ci)[valid_indices], (avg_factors + ci)[valid_indices], color=color, alpha=0.3)
#             ax.set_xlabel(x_label, fontsize=font_size)
#             ax.set_ylabel(factor_col, fontsize=font_size, labelpad=20)
#             ax.legend(fontsize=font_size, loc='upper left', bbox_to_anchor=(1, 1))
#             # ax.set_title(f'Time Series of {factor_col} - {category}', fontsize=font_size)
#             ax.set_xlim(xmin, xmax)
#             # Add faint gridlines
#             ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
#             ax.spines['top'].set_visible(False)
#             ax.spines['right'].set_visible(False)

#         plt.tight_layout()
#     else:
#         fig, ax = plt.subplots(figsize=fig_size)
        
#         for i, category in enumerate(unique_categories):
#             subset = data_df if category is None else data_df[data_df[separate_by] == category]
#             times = subset[time_col]
#             factors = subset[factor_col]

#             if meanormedian == 'mean':
#                 avg_factors = subset.groupby(time_col)[factor_col].mean()
#                 ci_func = bootstrap_ci_mean if bootstrap else lambda x: sem(x) * 1.96
#             else:
#                 avg_factors = subset.groupby(time_col)[factor_col].median()
#                 ci_func = bootstrap_ci_median if bootstrap else lambda x: sem(x) * 1.96

#             ci = subset.groupby(time_col)[factor_col].apply(ci_func)

#             color = color_palette[i]
#             label = 'Overall' if category is None else category

#             # Exclude the first time point (time zero)
#             valid_indices = avg_factors.index > 0

#             ax.plot(avg_factors.index[valid_indices], avg_factors.values[valid_indices], label=label, color=color, linewidth=2.5)
#             ax.fill_between(avg_factors.index[valid_indices], (avg_factors - ci)[valid_indices], (avg_factors + ci)[valid_indices], color=color, alpha=0.3)

#         ax.set_xlabel(x_label, fontsize=font_size)
#         ax.set_ylabel(factor_col, fontsize=font_size, labelpad=20)
#         ax.legend(fontsize=font_size, loc='upper left', bbox_to_anchor=(1.05, 1))
#         # Add faint gridlines
#         ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)


#         # ax.set_title(f'Time Series of {factor_col}', fontsize=font_size)
#         # get the max value of the x

#         #

#         ax.set_xlim(xmin, xmax)
#         plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to fit legend

#     # Create directory for plots if it doesn't exist
#     plots_dir = os.path.join(master_dir, 'plots')
#     os.makedirs(plots_dir, exist_ok=True)
    
#     # Generate filename
#     time_type = 'absolute' if absolute else 'time_zeroed'
#     bootstrap_text = 'bootstrapped' if bootstrap else 'nonbootstrapped'
#     multiplot_text = 'multiplot' if multiplot else 'singleplot'
#     filename = f"{plots_dir}/{factor_col}_{time_type}_{meanormedian}_{bootstrap_text}_{multiplot_text}.png"
    
#     # Save plot
#     plt.savefig(filename, bbox_inches='tight')
#     #set xlim
   
    
#     # Show plot if specified
#     if show_plot:
#         plt.show()
#     else:
#         plt.close()

######################## saved #########################################
# def plot_time_series(data_df, factor_col='speed_um_s', absolute=True, separate_by='condition', palette='colorblind', 
#                      meanormedian='mean', multiplot=False, talk=False, bootstrap=True, show_plot=True, 
#                      master_dir=None, order=None, grid=True):
#     """
#     Plot time series of a specified factor, with mean/median as a line and confidence intervals as shaded areas.
    
#     Parameters
#     ----------
#     data_df : DataFrame
#         DataFrame containing the time series data.
#     factor_col : str, optional
#         The column representing the factor to be plotted on the y-axis. Default is 'speed_um_s'.
#     absolute : bool, optional
#         Whether to use absolute time values or time zeroed values. Default is True.
#     separate_by : str, optional
#         Column to separate the data by, for coloring. If None, all data will be plotted together. Default is None.
#     palette : str, optional
#         Color palette for the plot. Default is 'colorblind'.
#     meanormedian : str, optional
#         Whether to use mean or median for aggregation. Default is 'mean'.
#     multiplot : bool, optional
#         Whether to generate separate small multiple plots for each category. Default is False.
#     talk : bool, optional
#         Whether to set the figure size to the original large size or a smaller size. Default is False.
#     bootstrap : bool, optional
#         Whether to use bootstrapping for confidence intervals. Default is True.
#     show_plot : bool, optional
#         Whether to display the plot in the notebook. Default is True.
#     master_dir : str, optional
#         The directory where the plots folder will be created and the plot will be saved. Default is None.
#     order : list, optional
#         Specific order for the conditions. Default is None.
#     grid : bool, optional
#         Whether to display grid lines. Default is True.
#     """
#     xmin = 0.2  # A FIX FOR NOW because really this should be the same as the shortest track (filtered to 0.2 s during filterstubs)
    
#     if master_dir is None:
#         master_dir = config.MASTER  # Use the master directory from config if not provided

#     if separate_by is not None and order is not None:
#         # Ensure the data is ordered according to the specified order
#         data_df[separate_by] = pd.Categorical(data_df[separate_by], categories=order, ordered=True)

#     if not absolute:
#         time_col = 'time_s_zeroed'
#         max_time_zeroed = data_df['time_s_zeroed'].max()
#         x_label = 'Time zeroed (s)'
#         xmax = max_time_zeroed
#     else:
#         time_col = 'time_s'
#         max_time = data_df['time_s'].max()
#         x_label = 'Time (s)'
#         xmax = max_time

#     # Use the categories attribute to maintain the specified order
#     if separate_by is not None:
#         # Convert to categorical if not already
#         if not pd.api.types.is_categorical_dtype(data_df[separate_by]):
#             data_df[separate_by] = pd.Categorical(data_df[separate_by], categories=order, ordered=True)
#         unique_categories = data_df[separate_by].cat.categories
#     else:
#         unique_categories = [None]

#     color_palette = sns.color_palette(palette, len(unique_categories))
    
#     # # Set figure size and font size based on the `talk` parameter
#     # if talk:
#     #     fig_size = (40, 12)
#     #     font_size = 35
#     # else:
#     #     if multiplot and separate_by:
#     #         fig_size = (4, 8 * len(unique_categories))
#     #     else:
#     #         fig_size = (5, 3)
#     #     font_size = 14

# # Set figure size and font size based on the `talk` and `multiplot` parameters
#     if talk:
#         base_fig_size = (30, 12)
#         font_size = 35
#     else:
#         base_fig_size = (10, 4)
#         font_size = 14

#     # Adjust figure size if multiplot is true
#     if multiplot and separate_by:
#         fig_size = (base_fig_size[0], base_fig_size[1] * len(unique_categories))
#     else:
#         fig_size = base_fig_size
    
#     sns.set_context("notebook", rc={"lines.linewidth": 2.5, "font.size": font_size, "axes.titlesize": font_size, 
#                                     "axes.labelsize": font_size, "xtick.labelsize": font_size, "ytick.labelsize": font_size})
    
#     if multiplot and separate_by:
#         num_categories = len(unique_categories)
#         fig, axes = plt.subplots(num_categories, 1, figsize=fig_size, sharex=True)
        
#         if num_categories == 1:
#             axes = [axes]  # To handle the case with only one subplot
        
#         for i, category in enumerate(unique_categories):
#             if pd.isna(category):
#                 continue
#             ax = axes[i] if len(unique_categories) > 1 else axes
#             subset = data_df[data_df[separate_by] == category]
#             times = subset[time_col]
#             factors = subset[factor_col]

#             if meanormedian == 'mean':
#                 avg_factors = subset.groupby(time_col)[factor_col].mean()
#                 ci_func = bootstrap_ci_mean if bootstrap else lambda x: sem(x) * 1.96
#             else:
#                 avg_factors = subset.groupby(time_col)[factor_col].median()
#                 ci_func = bootstrap_ci_median if bootstrap else lambda x: sem(x) * 1.96

#             ci = subset.groupby(time_col)[factor_col].apply(ci_func)

#             color = color_palette[i]
#             label = category

#             # Exclude the first time point (time zero)
#             valid_indices = avg_factors.index > 0

#             ax.plot(avg_factors.index[valid_indices], avg_factors.values[valid_indices], label=label, color=color, linewidth=2.5)
#             ax.fill_between(avg_factors.index[valid_indices], (avg_factors - ci)[valid_indices], (avg_factors + ci)[valid_indices], color=color, alpha=0.3)
#             ax.set_xlabel(x_label, fontsize=font_size)
#             ax.set_ylabel(factor_col, fontsize=font_size, labelpad=20)
#             ax.legend(fontsize=font_size, loc='upper left', bbox_to_anchor=(1, 1))
#             ax.set_xlim(xmin, xmax)
#             if grid:
#                 ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
#             ax.spines['top'].set_visible(False)
#             ax.spines['right'].set_visible(False)
#             ax.set_title(f'{category}', fontsize=font_size)

#         plt.tight_layout()
#     else:
#         fig, ax = plt.subplots(figsize=fig_size)
        
#         for i, category in enumerate(unique_categories):
#             if pd.isna(category):
#                 continue
#             subset = data_df if category is None else data_df[data_df[separate_by] == category]
#             times = subset[time_col]
#             factors = subset[factor_col]

#             if meanormedian == 'mean':
#                 avg_factors = subset.groupby(time_col)[factor_col].mean()
#                 ci_func = bootstrap_ci_mean if bootstrap else lambda x: sem(x) * 1.96
#             else:
#                 avg_factors = subset.groupby(time_col)[factor_col].median()
#                 ci_func = bootstrap_ci_median if bootstrap else lambda x: sem(x) * 1.96

#             ci = subset.groupby(time_col)[factor_col].apply(ci_func)

#             color = color_palette[i]
#             label = 'Overall' if category is None else category

#             # Exclude the first time point (time zero)
#             valid_indices = avg_factors.index > 0

#             ax.plot(avg_factors.index[valid_indices], avg_factors.values[valid_indices], label=label, color=color, linewidth=2.5)
#             ax.fill_between(avg_factors.index[valid_indices], (avg_factors - ci)[valid_indices], (avg_factors + ci)[valid_indices], color=color, alpha=0.3)

#         ax.set_xlabel(x_label, fontsize=font_size)
#         ax.set_ylabel(factor_col, fontsize=font_size, labelpad=20)
#         ax.legend(fontsize=font_size, loc='upper left', bbox_to_anchor=(1.05, 1))
#         if grid:
#             ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.set_xlim(xmin, xmax)
#         plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to fit legend

#     # Create directory for plots if it doesn't exist
#     plots_dir = os.path.join(master_dir, 'plots')
#     os.makedirs(plots_dir, exist_ok=True)
    
#     # Generate filename
#     time_type = 'absolute' if absolute else 'time_zeroed'
#     bootstrap_text = 'bootstrapped' if bootstrap else 'nonbootstrapped'
#     multiplot_text = 'multiplot' if multiplot else 'singleplot'
#     filename = f"{plots_dir}/{factor_col}_{time_type}_{meanormedian}_{bootstrap_text}_{multiplot_text}.png"
    
#     # Save plot
#     plt.savefig(filename, bbox_inches='tight')
    
#     # Show plot if specified
#     if show_plot:
#         plt.show()
#     else:
#         plt.close()
################################### saved ##################################




# # Placeholder bootstrap functions
# def bootstrap_ci_mean(data, n_bootstraps=1000):
#     boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstraps)]
#     lower = np.percentile(boot_means, 2.5)
#     upper = np.percentile(boot_means, 97.5)
#     return upper - lower

# def bootstrap_ci_median(data, n_bootstraps=1000):
#     boot_medians = [np.median(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstraps)]
#     lower = np.percentile(boot_medians, 2.5)
#     upper = np.percentile(boot_medians, 97.5)
#     return upper - lower

def plot_time_series(data_df, factor_col='speed_um_s', absolute=True, separate_by='condition', palette='colorblind', 
                     meanormedian='mean', multiplot=False, talk=False, bootstrap=True, show_plot=True, 
                     master_dir=None, order=None, grid=True, custom_yrange=None):
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
        master_dir = '.'  # Use current directory if not provided

    if separate_by is not None and order is not None:
        # Ensure the data is ordered according to the specified order
        data_df[separate_by] = pd.Categorical(data_df[separate_by], categories=order, ordered=True)

    if not absolute:
        time_col = 'time_s_zeroed'
        max_time_zeroed = data_df['time_s_zeroed'].max()
        x_label = 'Time zeroed (s)'
        xmax = max_time_zeroed
    else:
        time_col = 'time_s'
        max_time = data_df['time_s'].max()
        x_label = 'Time (s)'
        xmax = max_time

    # Use the categories attribute to maintain the specified order
    if separate_by is not None:
        # Convert to categorical if not already
        if not pd.api.types.is_categorical_dtype(data_df[separate_by]):
            data_df[separate_by] = pd.Categorical(data_df[separate_by], categories=order, ordered=True)
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
    
    sns.set_context("notebook", rc={"lines.linewidth": 2.5, "font.size": font_size, "axes.titlesize": font_size, 
                                    "axes.labelsize": font_size, "xtick.labelsize": font_size, "ytick.labelsize": font_size})
    
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

            if meanormedian == 'mean':
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

            ax.plot(avg_factors.index[valid_indices], avg_factors.values[valid_indices], label=label, color=color, linewidth=2.5)
            ax.fill_between(avg_factors.index[valid_indices], 
                            np.maximum((avg_factors - ci)[valid_indices], 0),  # Ensure lower bound is not below zero
                            (avg_factors + ci)[valid_indices], 
                            color=color, alpha=0.3)
            ax.set_xlabel(x_label, fontsize=font_size)
            ax.set_ylabel(factor_col, fontsize=font_size, labelpad=20)
            ax.legend(fontsize=font_size, loc='upper left', bbox_to_anchor=(1, 1))
            ax.set_xlim(xmin, xmax)
            
            # Set custom or automatic y-limits
            if custom_yrange:
                ax.set_ylim(custom_yrange)
            else:
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(0, ymax * 1.1)  # Add padding if using automatic limits

            if grid:
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_title(f'{category}', fontsize=font_size)

        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=fig_size)
        
        for i, category in enumerate(unique_categories):
            if pd.isna(category):
                continue
            subset = data_df if category is None else data_df[data_df[separate_by] == category]
            times = subset[time_col]
            factors = subset[factor_col]

            if meanormedian == 'mean':
                avg_factors = subset.groupby(time_col)[factor_col].mean()
                ci_func = bootstrap_ci_mean if bootstrap else lambda x: sem(x) * 1.96
            else:
                avg_factors = subset.groupby(time_col)[factor_col].median()
                ci_func = bootstrap_ci_median if bootstrap else lambda x: sem(x) * 1.96

            ci = subset.groupby(time_col)[factor_col].apply(ci_func)

            color = color_palette[i]
            label = 'Overall' if category is None else category

            # Exclude the first time point (time zero)
            valid_indices = avg_factors.index > 0

            ax.plot(avg_factors.index[valid_indices], avg_factors.values[valid_indices], label=label, color=color, linewidth=2.5)
            ax.fill_between(avg_factors.index[valid_indices], 
                            np.maximum((avg_factors - ci)[valid_indices], 0),  # Ensure lower bound is not below zero
                            (avg_factors + ci)[valid_indices], 
                            color=color, alpha=0.3)

        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(factor_col, fontsize=font_size, labelpad=20)
        ax.legend(fontsize=font_size, loc='upper left', bbox_to_anchor=(1.05, 1))
        if grid:
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(xmin, xmax)
        
        # Set custom or automatic y-limits
        if custom_yrange:
            ax.set_ylim(custom_yrange)
        else:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(0, ymax * 1.1)  # Add padding if using automatic limits
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to fit legend

    # Create directory for plots if it doesn't exist
    plots_dir = os.path.join(master_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate filename
    time_type = 'absolute' if absolute else 'time_zeroed'
    bootstrap_text = 'bootstrapped' if bootstrap else 'nonbootstrapped'
    multiplot_text = 'multiplot' if multiplot else 'singleplot'
    filename = f"{plots_dir}/{factor_col}_{time_type}_{meanormedian}_{bootstrap_text}_{multiplot_text}.png"
    
    # Save plot
    plt.savefig(filename, bbox_inches='tight')
    
    # Show plot if specified
    if show_plot:
        plt.show()
    else:
        plt.close()







def plot_barplots(data_df, factor_col='speed_um_s', separate_by='condition', palette='colorblind', meanormedian='mean', talk=False):
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
    sns.set_context("notebook", rc={"lines.linewidth": 2.5, "font.size": font_size, "axes.titlesize": font_size, "axes.labelsize": font_size, "xtick.labelsize": font_size, "ytick.labelsize": font_size})
    
    avg_factors_list = []
    ci_intervals = []
    
    for i, category in enumerate(unique_categories):
        subset = data_df if category is None else data_df[data_df[separate_by] == category]
        
        if meanormedian == 'mean':
            avg_factors = subset[factor_col].mean()
            ci_interval = bootstrap_ci_mean(subset[factor_col], num_samples=1000, alpha=0.05)
        else:
            avg_factors = subset[factor_col].median()
            ci_interval = bootstrap_ci_median(subset[factor_col], num_samples=1000, alpha=0.05)
        
        avg_factors_list.append(avg_factors)
        ci_intervals.append(ci_interval)
    
    categories = unique_categories if separate_by else ['Overall']
    ax.bar(categories, avg_factors_list, yerr=ci_intervals, color=color_palette, capsize=5, edgecolor='black')
    
    # Remove 'Condition_' prefix from x tick labels
    new_labels = [label.replace('Condition_', '') for label in categories]
    if talk:
        ax.set_xticklabels(new_labels, fontsize=font_size)
    else:
        ax.set_xticklabels(new_labels, fontsize=font_size, rotation=90)

    
    ax.set_ylabel(factor_col, fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tight_layout()
    
    plt.show()





def plot_violinplots(data_df, factor_col='speed_um_s', separate_by='condition', palette='colorblind', talk=False, orderin=None):
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
    sns.set_context("notebook", rc={"lines.linewidth": 2.5, "font.size": font_size, "axes.titlesize": font_size, "axes.labelsize": font_size, "xtick.labelsize": font_size, "ytick.labelsize": font_size})
    
    # Plot violin plot with custom order
    sns.violinplot(x=separate_by, y=factor_col, hue=separate_by, data=data_df, palette=color_palette, ax=ax, legend=False, alpha=0.79, order=orderin)
    
    # If orderin is provided, update x-tick labels accordingly
    if orderin is not None:
        new_labels = [label.replace('Condition_', '') for label in orderin]
    else:
        new_labels = [label.replace('Condition_', '') for label in unique_categories]

    ax.set_xticks(range(len(new_labels)))
    ax.set_xticklabels(new_labels, fontsize=font_size)

    ax.set_ylabel(factor_col, fontsize=font_size, labelpad=20)
    ax.set_xlabel(None)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tight_layout()
    
    plt.show()


def plot_metric_for_selected_particles(time_windowed_df, feature='avg_msd', num_particles=5, global_xlim=True, subplot_size=5):
    # Get unique motion classes
    motion_classes = time_windowed_df['motion_class'].unique()

    # Randomly select a set of particles for each motion class
    selected_particles = {}
    global_max_time = 0  # To store the global maximum time across selected particles
    max_feature_value = 0  # To store the global max feature value across selected particles

    for motion_class in motion_classes:
        particles = time_windowed_df[time_windowed_df['motion_class'] == motion_class]['unique_id'].unique()
        selected_particles[motion_class] = random.sample(list(particles), min(num_particles, len(particles)))
        
        # Calculate the maximum time_s for the current motion class
        for unique_id in selected_particles[motion_class]:
            data = time_windowed_df[time_windowed_df['unique_id'] == unique_id]
            global_max_time = max(global_max_time, data['time_s'].max())
            max_feature_value = max(max_feature_value, data[feature].max())

    # Add padding to the maximum feature value
    padding = 0.1
    max_feature_value *= (1 + padding)

    # Determine the total number of plots
    total_plots = sum(len(particles) for particles in selected_particles.values())

    # Set up subplots with fixed subplot size
    ncols = num_particles  # Number of particles per row
    nrows = len(motion_classes)  # One row per motion class

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(subplot_size * ncols, subplot_size * nrows))
    fig.suptitle(f'{feature} vs. Time for Selected Particles in Each Motion Class')

    # Plot each selected particle in its subplot
    plot_idx = 0
    for i, motion_class in enumerate(motion_classes):
        for j, unique_id in enumerate(selected_particles[motion_class]):
            data = time_windowed_df[time_windowed_df['unique_id'] == unique_id]
            ax = axes[i, j]  # Access subplot at row i and column j
            ax.plot(data['time_s'], data[feature], label=f'Particle {unique_id}', color=get_color(motion_class))
            ax.set_title(f'Particle {unique_id} ({motion_class})')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'{feature} ($\mu m^2$)')
            ax.grid(True)

            # Set limits for x-axis and y-axis
            if global_xlim:
                ax.set_xlim(0, global_max_time)
            else:
                ax.set_xlim(0, data['time_s'].max())

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
    colorblind_colors = plt.get_cmap('tab10')
    if motion_class == 'subdiffusive':
        return colorblind_colors(0)  # Blue
    elif motion_class == 'normal':
        return colorblind_colors(1)  # Orange
    elif motion_class == 'superdiffusive':
        return colorblind_colors(2)  # Green
    else:
        return 'black'
    

def plot_single_particle_msd(msd_lagtime_df):
    '''
    This thing basically takes 3 example particle tracks, one from each motion class
    '''


    # Ensure we have data for each motion class
    motion_classes = ['subdiffusive', 'normal', 'superdiffusive']
    
    # Randomly select one unique_id from each motion class
    selected_particles = {}
    for motion_class in motion_classes:
        particles = msd_lagtime_df[msd_lagtime_df['motion_class'] == motion_class]['unique_id'].unique()
        if len(particles) > 0:
            selected_particles[motion_class] = random.choice(particles)

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Plot MSD for each selected particle
    for motion_class, unique_id in selected_particles.items():
        data = msd_lagtime_df[msd_lagtime_df['unique_id'] == unique_id]
        plt.plot(data['lag_time'], data['msd'], label=f'{motion_class} (Particle {unique_id})', color=get_color(motion_class))

    # Set log scales
    plt.xscale('log')
    plt.yscale('log')

    # Add labels and title
    plt.xlabel('Time Lag (s)')
    plt.ylabel('MSD ($\mu m^2$)')
    plt.title('MSD vs. Time Lag for Selected Particles')
    plt.legend()
    plt.grid(True)
    plt.show()

def get_color(motion_class):
    # Assign colors using Colorblind colormap
    colorblind_colors = plt.get_cmap('tab10')
    if motion_class == 'subdiffusive':
        return colorblind_colors(0)  # Blue
    elif motion_class == 'normal':
        return colorblind_colors(1)  # Orange
    elif motion_class == 'superdiffusive':
        return colorblind_colors(2)  # Green
    else:
        return 'black'
    

# def plot_classification_pie_chart(df, colormap_name='Dark2'):
#     classification_counts = df['motion_class'].value_counts()
    
#     # Define labels and sizes
#     labels = classification_counts.index
#     sizes = classification_counts.values

#     # Get colors from the specified colormap
#     colormap = cm.get_cmap(colormap_name)
#     colors = colormap(np.linspace(0, 1, len(labels)))

#     # Plot pie chart
#     plt.figure(figsize=(7, 7))
#     plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
#     plt.title('Classification of Time Windowed Tracks')
#     plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
#     plt.show()



# def plot_classification_pie_chart(time_windowed_df, colormap_name='Dark2'):
#     # Calculate classification counts and percentages
#     classification_counts = time_windowed_df['motion_class'].value_counts()
#     total_count = classification_counts.sum()
#     percentages = classification_counts / total_count * 100
    
#     # Define labels with counts and percentages
#     labels = [f'{cls}\n{count} ({percent:.1f}%)' for cls, count, percent in zip(classification_counts.index, classification_counts.values, percentages)]

#     # Get colors from the specified colormap
#     colormap = cm.get_cmap(colormap_name)
#     colors = colormap(np.linspace(0, 1, len(labels)))

#     # Plot pie chart
#     plt.figure(figsize=(7, 7))
#     plt.pie(classification_counts, labels=labels, startangle=140, colors=colors)
#     plt.title('Classification of Time Windowed Tracks')
#     plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
#     plt.show()


def plot_classification_pie_charts(df, group_by='Location', colormap_name='Dark2', figsize=(15, 10), font_size=12, label_font_size=8):
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

    # Plot each category as a separate pie chart
    for i, category in enumerate(categories):
        ax = axes[i]
        subset_df = df[df[group_by] == category]
        classification_counts = subset_df['motion_class'].value_counts()
        total_count = classification_counts.sum()
        percentages = classification_counts / total_count * 100
        
        # Define labels for outside the pie
        outside_labels = [f'{cls} ({count})' for cls, count in zip(classification_counts.index, classification_counts.values)]
        
        # Colors for pie slices
        colors = colormap(np.linspace(0, 1, len(outside_labels)))

        # Plot pie chart with percentages inside
        wedges, texts, autotexts = ax.pie(
            classification_counts, 
            labels=outside_labels, 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=colors,
            textprops={'fontsize': label_font_size}
        )

        # Set the color and size of the percentage text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(label_font_size)

        ax.set_title(f'{category} ({total_count} tracks)', fontsize=font_size)
        ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.

    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f'Classification of Time Windowed Tracks by {group_by}', fontsize=font_size + 2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()




# def plot_boxplots(data_df, feature, x_category, font_size=12, order=None, palette='colorblind', show_plot=True, master_dir=None, grid=True, bw=False, strip=False, y_max=None, figsize=(10, 8), ):
#     """
#     Plot boxplots for the specified feature against a categorical x_category with custom order and styling options.

#     Parameters
#     ----------
#     data_df : DataFrame
#         DataFrame containing the data.
#     feature : str
#         The feature to plot on the y-axis.
#     x_category : str
#         The categorical feature to plot on the x-axis.
#     font_size : int, optional
#         Font size for the plot text. Default is 12.
#     order : list, optional
#         Specific order for the categories. Default is None.
#     palette : str, optional
#         Color palette for the plot. Default is 'colorblind'.
#     show_plot : bool, optional
#         Whether to display the plot in the notebook. Default is True.
#     master_dir : str, optional
#         Directory to save the plot. Default is None.
#     grid : bool, optional
#         Whether to display grid lines. Default is True.
#     bw : bool, optional
#         Whether to use black-and-white styling. Default is False.
#     y_max : float, optional
#         Maximum value for the y-axis. Default is None.
#     figsize : tuple, optional
#         Size of the plot (width, height) in inches. Default is (10, 8).
#     """
    
#     plt.figure(figsize=figsize)  # Use the figsize parameter to set the figure size
#     sns.set_context("notebook", rc={"xtick.labelsize": font_size, "ytick.labelsize": font_size})

#     if bw:
#         # Black-and-white styling: no fill color, black edges
#         boxplot = sns.boxplot(x=x_category, y=feature, data=data_df, linewidth=1.5, showfliers=False, color='white', order=order)
#         for patch in boxplot.patches:
#             patch.set_edgecolor('black')  # Set edge color to black
#             patch.set_linewidth(1.5)     # Set edge width
#         for element in ['boxes', 'whiskers', 'medians', 'caps']:
#             plt.setp(boxplot.artists, color='black')  # Set edge color to black
#             plt.setp(boxplot.lines, color='black')    # Set whiskers and caps to black
#     else:
#         boxplot = sns.boxplot(x=x_category, y=feature, data=data_df, palette=palette, order=order, showfliers=False)
#         boxplot.patch.set_linewidth(1.5)

    
        

#     plt.xlabel('', fontsize=font_size)
#     plt.ylabel(feature, fontsize=font_size)
#     plt.title(f'{feature} by {x_category}', fontsize=font_size)
#     plt.xticks(rotation=45)

#     if grid:
#         plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7, axis='y')  # Grid on y-axis
#         plt.gca().set_axisbelow(True)  # Ensure grid is below the plot elements

#     # Set the maximum y-axis limit if specified
#     if y_max is not None:
#         plt.ylim(top=y_max)

#     # Customize spines
#     ax = plt.gca()
#     ax.spines['top'].set_color('lightgray')
#     ax.spines['right'].set_color('lightgray')
#     ax.spines['left'].set_color('black')
#     ax.spines['bottom'].set_color('black')

#     plt.tight_layout()

#     if master_dir is None:
#         master_dir = "plots"  # Default directory

#     os.makedirs(master_dir, exist_ok=True)
#     filename = f"{master_dir}/{feature}_by_{x_category}.png"
#     plt.savefig(filename, bbox_inches='tight')

#     if show_plot:
#         plt.show()
#     else:
#         plt.close()

# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# def plot_boxplots(data_df, feature, x_category, font_size=12, order=None, palette='colorblind', show_plot=True, master_dir=None, grid=True, bw=False, strip=False, y_max=None, figsize=(10, 8), annotate_median=False):
#     """
#     Plot boxplots for the specified feature against a categorical x_category with custom order and styling options.

#     Parameters
#     ----------
#     data_df : DataFrame
#         DataFrame containing the data.
#     feature : str
#         The feature to plot on the y-axis.
#     x_category : str
#         The categorical feature to plot on the x-axis.
#     font_size : int, optional
#         Font size for the plot text. Default is 12.
#     order : list, optional
#         Specific order for the categories. Default is None.
#     palette : str, optional
#         Color palette for the plot. Default is 'colorblind'.
#     show_plot : bool, optional
#         Whether to display the plot in the notebook. Default is True.
#     master_dir : str, optional
#         Directory to save the plot. Default is None.
#     grid : bool, optional
#         Whether to display grid lines. Default is True.
#     bw : bool, optional
#         Whether to use black-and-white styling. Default is False.
#     strip : bool, optional
#         Whether to overlay a stripplot on the boxplot. Default is False.
#     y_max : float, optional
#         Maximum value for the y-axis. Default is None.
#     figsize : tuple, optional
#         Size of the plot (width, height) in inches. Default is (10, 8).
#     """

#     plt.figure(figsize=figsize)  # Use the figsize parameter to set the figure size
#     sns.set_context("notebook", rc={"xtick.labelsize": font_size, "ytick.labelsize": font_size})

#     if bw:
#         # Black-and-white styling: no fill color, black edges
#         boxplot = sns.boxplot(x=x_category, y=feature, data=data_df, linewidth=1.5, showfliers=False, color='white', order=order)
#         for patch in boxplot.patches:
#             patch.set_edgecolor('black')  # Set edge color to black
#             patch.set_linewidth(1.5)     # Set edge width
#         for element in ['boxes', 'whiskers', 'medians', 'caps']:
#             plt.setp(boxplot.artists, color='black')  # Set edge color to black
#             plt.setp(boxplot.lines, color='black')    # Set whiskers and caps to black
#     else:
#         boxplot = sns.boxplot(x=x_category, y=feature, data=data_df, palette=palette, order=order, showfliers=False)
#         boxplot.patch.set_linewidth(1.5)

#     if strip:
#         sns.stripplot(x=x_category, y=feature, data=data_df, color='black', size=0.3, order=order, jitter=True)

#     plt.xlabel('', fontsize=font_size)
#     plt.ylabel(feature, fontsize=font_size)
#     plt.title(f'{feature} by {x_category}', fontsize=font_size)
#     # plt.xticks(rotation=45)

#     if grid:
#         plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7, axis='y')  # Grid on y-axis
#         plt.gca().set_axisbelow(True)  # Ensure grid is below the plot elements

#     # Set the maximum y-axis limit if specified
#     if y_max is not None:
#         plt.ylim(top=y_max)

#     # Customize spines
#     ax = plt.gca()
#     ax.spines['top'].set_color('lightgray')
#     ax.spines['right'].set_color('lightgray')
#     ax.spines['left'].set_color('black')
#     ax.spines['bottom'].set_color('black')

#         # Annotate median values if requested
#     # Annotate median values if requested
#     # Annotate median values if requested
#     if annotate_median:
#         medians = data_df.groupby(x_category)[feature].median()
#         y_max = plt.ylim()[1]  # Get the top y-limit of the plot

#         if order is not None:
#             sorted_medians = medians.reindex(order)  # Reorder the medians based on the 'order' parameter
#         else:
#             sorted_medians = medians

#         for i, median in enumerate(sorted_medians):
#             plt.text(i, y_max * 0.95, f'{median:.2f}', 
#                      horizontalalignment='center', size=font_size, color='black', weight='bold')




#     plt.tight_layout()

#     if master_dir is None:
#         master_dir = "plots"  # Default directory

#     os.makedirs(master_dir, exist_ok=True)
#     filename = f"{master_dir}/{feature}_by_{x_category}.png"
#     plt.savefig(filename, bbox_inches='tight')

#     if show_plot:
#         plt.show()
#     else:
#         plt.close()


# from matplotlib.colors import is_color_like

# def plot_boxplots_test(data_df, feature, x_category, font_size=12, order=None, palette='colorblind', 
#                   background='white', transparent=False, show_plot=True, master_dir=None, 
#                   grid=True, bw=False, strip=False, y_max=None, figsize=(10, 8), 
#                   annotate_median=False):
#     """
#     Plot boxplots for the specified feature against a categorical x_category with custom order and styling options.

#     Parameters
#     ----------
#     data_df : DataFrame
#         DataFrame containing the data.
#     feature : str
#         The feature to plot on the y-axis.
#     x_category : str
#         The categorical feature to plot on the x-axis.
#     font_size : int, optional
#         Font size for the plot text. Default is 12.
#     order : list, optional
#         Specific order for the categories. Default is None.
#     palette : str, optional
#         Color palette for the plot. Default is 'colorblind'.
#     background : str, tuple, optional
#         Background color as color name, RGB tuple, or hex code. Default is 'white'.
#     transparent : bool, optional
#         If True, makes the plot fully transparent except for box plots and axes. Default is False.
#     show_plot : bool, optional
#         Whether to display the plot in the notebook. Default is True.
#     master_dir : str, optional
#         Directory to save the plot. Default is None.
#     grid : bool, optional
#         Whether to display grid lines. Default is True.
#     bw : bool, optional
#         Whether to use black-and-white styling. Default is False.
#     strip : bool, optional
#         Whether to overlay a stripplot on the boxplot. Default is False.
#     y_max : float, optional
#         Maximum value for the y-axis. Default is None.
#     figsize : tuple, optional
#         Size of the plot (width, height) in inches. Default is (10, 8).
#     """

#     # Validate and apply background color
#     if is_color_like(background):
#         figure_background = background
#     else:
#         print("Invalid color provided for background. Defaulting to white.")
#         figure_background = 'white'

#     # Create figure and set background color
#     fig, ax = plt.subplots(figsize=figsize, facecolor=figure_background)
#     if not transparent:
#         ax.set_facecolor(figure_background)  # Set plot area background color
#     else:
#         fig.patch.set_alpha(0)  # Make the figure background transparent
#         ax.set_facecolor((0, 0, 0, 0))  # Make the plot area transparent

#     sns.set_context("notebook", rc={"xtick.labelsize": font_size, "ytick.labelsize": font_size})

#     if bw:
#         boxplot = sns.boxplot(x=x_category, y=feature, data=data_df, linewidth=1.5, showfliers=False, color='white', order=order)
#         for patch in boxplot.patches:
#             patch.set_edgecolor('black')
#             patch.set_linewidth(1.5)
#         for element in ['boxes', 'whiskers', 'medians', 'caps']:
#             plt.setp(boxplot.artists, color='black')
#             plt.setp(boxplot.lines, color='black')
#     else:
#         boxplot = sns.boxplot(x=x_category, y=feature, data=data_df, palette=palette, order=order, showfliers=False)
#         boxplot.patch.set_linewidth(1.5)

#     if strip:
#         sns.stripplot(x=x_category, y=feature, data=data_df, color='black', size=0.3, order=order, jitter=True)

#     # Customize grid and axes based on transparency
#     if grid:
#         ax.grid(True, linestyle='--', linewidth=0.5, color='gray' if not transparent else (0, 0, 0, 0.5), alpha=0.7, axis='y')

#     plt.xlabel('', fontsize=font_size)
#     plt.ylabel(feature, fontsize=font_size)
#     plt.title(f'{feature} by {x_category}', fontsize=font_size)

#     # Set the maximum y-axis limit if specified
#     if y_max is not None:
#         plt.ylim(top=y_max)

#     # Customize spines
#     for spine in ax.spines.values():
#         spine.set_edgecolor('black')
#         spine.set_linewidth(1.5)
#         if transparent:
#             spine.set_alpha(0.5)

#     if annotate_median:
#         medians = data_df.groupby(x_category)[feature].median()
#         y_max = plt.ylim()[1]
#         sorted_medians = medians.reindex(order) if order else medians
#         for i, median in enumerate(sorted_medians):
#             plt.text(i, y_max * 0.95, f'{median:.2f}', 
#                      horizontalalignment='center', size=font_size, color='black', weight='bold')

#     plt.tight_layout()

#     if master_dir is None:
#         master_dir = "plots"
#     os.makedirs(master_dir, exist_ok=True)
#     filename = f"{master_dir}/{feature}_by_{x_category}.png"
#     plt.savefig(filename, bbox_inches='tight', transparent=transparent)  # Save with transparency if specified

#     if show_plot:
#         plt.show()
#     else:
#         plt.close()





def plot_boxplots(data_df, feature, x_category, font_size=12, order=None, palette='colorblind', 
                  background='white', transparent=False, line_color='black', show_plot=True, 
                  master_dir=None, grid=True, bw=False, strip=False, y_max=None, figsize=(10, 8), 
                  annotate_median=False):
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
        figure_background = 'white'

    # Create figure and set background color
    fig, ax = plt.subplots(figsize=figsize, facecolor=figure_background)
    if not transparent:
        ax.set_facecolor(figure_background)  # Set plot area background color
    else:
        fig.patch.set_alpha(0)  # Make the figure background transparent
        ax.set_facecolor((0, 0, 0, 0))  # Make the plot area transparent

    sns.set_context("notebook", rc={"xtick.labelsize": font_size, "ytick.labelsize": font_size})

    if bw:
        boxplot = sns.boxplot(x=x_category, y=feature, data=data_df, linewidth=1.5, 
                              showfliers=False, color='white', order=order)
        for patch in boxplot.patches:
            patch.set_edgecolor(line_color)
            patch.set_linewidth(1.5)
        for element in ['boxes', 'whiskers', 'medians', 'caps']:
            plt.setp(boxplot.artists, color=line_color)
            plt.setp(boxplot.lines, color=line_color)
    else:
        boxplot = sns.boxplot(x=x_category, y=feature, data=data_df, palette=palette, 
                              order=order, showfliers=False, linewidth=1.5)
        for patch in boxplot.patches:
            patch.set_edgecolor(line_color)
            patch.set_linewidth(1.5)
        for line in boxplot.lines:  # Apply color to all boxplot lines, including medians and quartiles
            line.set_color(line_color)
            line.set_linewidth(1.5)

    if strip:
        sns.stripplot(x=x_category, y=feature, data=data_df, color=line_color, size=3, 
                      order=order, jitter=True)

    plt.xlabel('', fontsize=font_size, color=line_color)
    plt.ylabel(feature, fontsize=font_size, color=line_color)
    # plt.title(f'{feature} by {x_category}', fontsize=font_size, color=line_color)

    # Set tick and label colors
    ax.tick_params(axis='both', which='both', color=line_color, labelcolor=line_color, labelsize=font_size)

    # Set grid and axis line colors
    if grid:
        # ax.grid(True, linestyle='--', linewidth=0.5, color=line_color if not transparent else (0, 0, 0, 0.5), alpha=0.7, axis='y')
        ax.grid(True, linestyle='--', linewidth=0.5, color=line_color, alpha=0.7, axis='y')

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
            plt.text(i, y_max * 0.965, f'{median:.2f}', 
                     horizontalalignment='center', size=font_size, color=line_color, weight='bold')

    plt.tight_layout()

    if master_dir is None:
        master_dir = "plots"
    os.makedirs(master_dir, exist_ok=True)
    filename = f"{master_dir}/{feature}_by_{x_category}.png"
    plt.savefig(filename, bbox_inches='tight', transparent=transparent)

    if show_plot:
        plt.show()
    else:
        plt.close()






# def plot_stacked_bar(df, x_category, order=None, font_size=16, colormap='Dark2', figsize=(10, 5)):
#     """
#     Plot a stacked bar chart showing the percentage of motion classes for each category on the x-axis.
    
#     Parameters
#     ----------
#     df : DataFrame
#         DataFrame containing the data with a 'motion_class' column and specified category.
#     x_category : str
#         The column name of the category to plot on the x-axis.
#     order : list, optional
#         Custom order for the categories on the x-axis. Default is None.
#     font_size : int, optional
#         Font size for the plot. Default is 16.
#     colormap : str, optional
#         Colormap for the plot. Default is 'Dark2'.
#     figsize : tuple, optional
#         Size of the plot (width, height) in inches. Default is (10, 5).
#     """
#     # Apply custom order if provided
#     if order is not None:
#         df[x_category] = pd.Categorical(df[x_category], categories=order, ordered=True)

#     # Calculate the percentage of each motion class within each category
#     percentage_data = (df.groupby([x_category, 'motion_class']).size()
#                        .unstack(fill_value=0)
#                        .apply(lambda x: x / x.sum() * 100, axis=1))

#     # Determine the unique motion classes
#     motion_classes = df['motion_class'].unique()

#     # Use a consistent colormap for the motion classes
#     if colormap == 'colorblind':
#         colors = sns.color_palette("colorblind", len(motion_classes))
#     elif colormap == 'Dark2':
#         cmap = cm.get_cmap("Dark2", len(motion_classes))
#         colors = cmap(np.linspace(0, 1, len(motion_classes)))
#     else:
#         colors = plt.get_cmap(colormap, len(motion_classes)).colors

#     # Plotting
#     ax = percentage_data.plot(kind='bar', stacked=True, figsize=figsize, color=colors)

#     # Add black outlines to the bars
#     for patch in ax.patches:
#         patch.set_edgecolor('black')

#     # Rotate x-tick labels for readability if necessary
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

#     # Annotate percentages on the bars
#     for patch in ax.patches:
#         width, height = patch.get_width(), patch.get_height()
#         x, y = patch.get_xy()
#         if height > 0:  # Only annotate if there's a height to show
#             ax.annotate(f'{height:.1f}%', (x + width / 2, y + height / 2),
#                         ha='center', va='center', fontsize=font_size, color='k')

#     # Move the legend outside the plot
#     plt.legend(title='Motion Type', bbox_to_anchor=(1.05, 1), loc='upper left',
#                title_fontsize=font_size, prop={'size': font_size})

#     plt.title('Distribution of Motion Classes', fontsize=font_size)
#     plt.xlabel('', fontsize=font_size)
#     plt.ylabel('Percentage (%)', fontsize=font_size)
#     plt.xticks(fontsize=font_size, rotation=45)
#     plt.yticks(fontsize=font_size)

#     # Add grid
#     ax.grid(True, which='both', linestyle='--', linewidth=0.5)

#     # Ensure bars are above the grid
#     for patch in ax.patches:
#         patch.set_zorder(2)

#     # Make top and right spines visible and set their color to light gray
#     ax.spines['top'].set_visible(True)
#     ax.spines['top'].set_color('gray')
#     ax.spines['right'].set_visible(True)
#     ax.spines['right'].set_color('gray')

#     plt.tight_layout()
#     plt.show()



def plot_stacked_bar(df, x_category, order=None, font_size=16, colormap='Dark2', figsize=(10, 8), 
                     background='white', transparent=False, line_color='black'):
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
        figure_background = 'none'
        axis_background = (0, 0, 0, 0)  # Transparent background for axes
    else:
        figure_background = background
        axis_background = background

    # Apply custom order if provided
    if order is not None:
        df[x_category] = pd.Categorical(df[x_category], categories=order, ordered=True)

    # Calculate the percentage of each motion class within each category
    percentage_data = (df.groupby([x_category, 'motion_class']).size()
                       .unstack(fill_value=0)
                       .apply(lambda x: x / x.sum() * 100, axis=1))

    # Determine the unique motion classes and assign colors
    motion_classes = df['motion_class'].unique()
    if colormap == 'colorblind':
        colors = sns.color_palette("colorblind", len(motion_classes))
    elif colormap == 'Dark2':
        cmap = cm.get_cmap("Dark2", len(motion_classes))
        colors = cmap(np.linspace(0, 1, len(motion_classes)))
    else:
        colors = plt.get_cmap(colormap, len(motion_classes)).colors

    # Plotting
    fig, ax = plt.subplots(figsize=figsize, facecolor=figure_background)
    ax.set_facecolor(axis_background)
    percentage_data.plot(kind='bar', stacked=True, ax=ax, color=colors, edgecolor=line_color)

    # Add black outlines to the bars
    for patch in ax.patches:
        patch.set_edgecolor(line_color)

    # Annotate percentages on the bars
    for patch in ax.patches:
        width, height = patch.get_width(), patch.get_height()
        x, y = patch.get_xy()
        if height > 0:  # Only annotate if there's a height to show
            ax.annotate(f'{height:.1f}%', (x + width / 2, y + height / 2),
                        ha='center', va='center', fontsize=font_size, color=line_color)

    # Customize text elements
    ax.set_title('Distribution of Motion Classes', fontsize=font_size, color=line_color)
    ax.set_xlabel('', fontsize=font_size, color=line_color)
    ax.set_ylabel('Percentage (%)', fontsize=font_size, color=line_color)
    ax.tick_params(axis='both', which='both', color=line_color, labelcolor=line_color, labelsize=font_size)

    # Rotate x-tick labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Move the legend outside the plot
    legend = plt.legend(title='Motion Type', bbox_to_anchor=(1.05, 1), loc='upper left',
               title_fontsize=font_size, prop={'size': font_size}, frameon=False,)
    
    # Set legend text color
    for text in legend.get_texts():
        text.set_color(line_color)

    # Set title color
    legend.get_title().set_color(line_color)

    # Add grid with line color
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color=line_color, zorder=0)

    # Customize axis spines
    for spine in ax.spines.values():
        spine.set_edgecolor(line_color)
        if transparent:
            spine.set_alpha(0.5)

    plt.tight_layout()
    plt.show()



# def plot_single_particle(particle_df, speed_thresholds, animation=True):
#     """
#     Plots the trajectory of a single particle, either as an animation or as a static image, 
#     with each segment colored according to its speed category.
    
#     Parameters:
#     - particle_df: DataFrame containing the particle's data with columns 'x', 'y', and 'speed_um_s'.
#     - speed_thresholds: List or tuple of four numbers defining the boundaries for low, medium, and high speeds.
#     - animation: Boolean, if True creates an animation, if False creates a static plot.
#     """
#     # Get the unique ID of the particle
#     unique_id = particle_df['unique_id'].unique()[0]

#     # Ensure the speed thresholds are sorted
#     speed_thresholds = sorted(speed_thresholds)
    
#     # Assign speed categories based on thresholds
#     conditions = [
#         (particle_df['speed_um_s'] >= speed_thresholds[0]) & (particle_df['speed_um_s'] < speed_thresholds[1]),
#         (particle_df['speed_um_s'] >= speed_thresholds[1]) & (particle_df['speed_um_s'] < speed_thresholds[2]),
#         (particle_df['speed_um_s'] >= speed_thresholds[2])
#     ]
#     choices = ['low', 'medium', 'high']
#     particle_df['speed_category'] = np.select(conditions, choices, default='unknown')
    
#     # Define a colormap for the speed categories
#     colormap = {
#         'low': 'blue',
#         'medium': 'green',
#         'high': 'orange'
#     }
    
#     # Calculate center and range for square plot
#     center_x = (particle_df['x'].max() + particle_df['x'].min()) / 2
#     center_y = (particle_df['y'].max() + particle_df['y'].min()) / 2
#     range_extent = max(particle_df['x'].max() - particle_df['x'].min(), particle_df['y'].max() - particle_df['y'].min()) / 2
#     range_extent *= 1.1  # Add some padding
    
#     # Create directory for saving images
#     if animation:
#         dir_name = f'{config.MASTER}visualization/particle_speedcat_{unique_id}'
#         os.makedirs(dir_name, exist_ok=True)
#     else:
#         dir_name = f'{config.MASTER}visualization/particle_speedcat_static_{unique_id}'
#         os.makedirs(dir_name, exist_ok=True)
    
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.set_title(f'Particle Trajectory with Speed Categories: {unique_id}')
#     ax.set_xlabel('X Position')
#     ax.set_ylabel('Y Position')
#     ax.set_xlim(center_x - range_extent, center_x + range_extent)
#     ax.set_ylim(center_y - range_extent, center_y + range_extent)
    
#     if animation:
#         # Create animation by iterating through the data and plotting frame by frame
#         for i in range(1, len(particle_df)):
#             ax.clear()
#             ax.set_title(f'Particle Trajectory with Speed Categories: {unique_id}')
#             ax.set_xlabel('X Position')
#             ax.set_ylabel('Y Position')
#             ax.set_xlim(center_x - range_extent, center_x + range_extent)
#             ax.set_ylim(center_y - range_extent, center_y + range_extent)

#             # Plot the trajectory up to the current point, changing colors according to speed category
#             for j in range(1, i + 1):
#                 x_values = particle_df['x'].iloc[j-1:j+1]
#                 y_values = particle_df['y'].iloc[j-1:j+1]
#                 speed_category = particle_df['speed_category'].iloc[j-1]
#                 color = colormap.get(speed_category, 'black')
#                 ax.plot(x_values, y_values, color=color, linewidth=2)

#             ax.legend(handles=[plt.Line2D([0], [0], color=color, label=f'{category.capitalize()} Speed') for category, color in colormap.items()])
#             plt_path = os.path.join(dir_name, f'frame_{i:03d}.png')
#             plt.savefig(plt_path)
#     else:
#         # Create static plot of the entire trajectory
#         for i in range(1, len(particle_df)):
#             x_values = particle_df['x'].iloc[i-1:i+1]
#             y_values = particle_df['y'].iloc[i-1:i+1]
#             speed_category = particle_df['speed_category'].iloc[i-1]
#             color = colormap.get(speed_category, 'black')
#             ax.plot(x_values, y_values, color=color, linewidth=2)

#         ax.legend(handles=[plt.Line2D([0], [0], color=color, label=f'{category.capitalize()} Speed') for category, color in colormap.items()])
#         static_path = os.path.join(dir_name, 'static_trajectory.png')
#         plt.savefig(static_path)
#         plt.show()
    
#     plt.close(fig)


# def plot_single_particle(particle_df, threshold_col, thresholds, animation=True):
#     """
#     Plots the trajectory of a single particle, either as an animation or as a static image, 
#     with each segment colored according to its speed category.
    
#     Parameters:
#     - particle_df: DataFrame containing the particle's data with columns 'x', 'y', and 'speed_um_s'.
#     - speed_thresholds: List or tuple of four numbers defining the boundaries for low, medium, and high speeds.
#     - animation: Boolean, if True creates an animation, if False creates a static plot.
#     """
#     # Get the unique ID of the particle
#     unique_id = particle_df['unique_id'].unique()[0]

#     # Ensure the speed thresholds are sorted
#     thresholds = sorted(thresholds)
    
#     # Assign speed categories based on thresholds
#     conditions = [
#         (particle_df[threshold_col] >= thresholds[0]) & (particle_df[threshold_col] < thresholds[1]),
#         (particle_df[threshold_col] >= thresholds[1]) & (particle_df[threshold_col] < thresholds[2]),
#         (particle_df[threshold_col] >= thresholds[2])
#     ]
#     choices = ['low', 'medium', 'high']
#     factorcategory = f'{threshold_col}_category'
#     particle_df[factorcategory] = np.select(conditions, choices, default='unknown')
    
#     # Define a colormap for the speed categories
#     colormap = {
#         'low': 'blue',
#         'medium': 'green',
#         'high': 'red'
#     }
    
#     # Calculate center and range for square plot
#     center_x = (particle_df['x'].max() + particle_df['x'].min()) / 2
#     center_y = (particle_df['y'].max() + particle_df['y'].min()) / 2
#     range_extent = max(particle_df['x'].max() - particle_df['x'].min(), particle_df['y'].max() - particle_df['y'].min()) / 2
#     range_extent *= 1.1  # Add some padding
    
#     # Create directory for saving images
#     if animation:
#         dir_name = f'{config.MASTER}visualization/particle_{unique_id}_cat_{threshold_col}'
#         os.makedirs(dir_name, exist_ok=True)
#     else:
#         dir_name = f'{config.MASTER}visualization'
#         os.makedirs(dir_name, exist_ok=True)
    
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.set_title(f'Particle Trajectory with Speed Categories: {unique_id}')
#     ax.set_xlabel('X Position')
#     ax.set_ylabel('Y Position')
#     ax.set_xlim(center_x - range_extent, center_x + range_extent)
#     ax.set_ylim(center_y - range_extent, center_y + range_extent)
    
#     if animation:
#         # Create animation by iterating through the data and plotting frame by frame
#         for i in range(1, len(particle_df)):
#             ax.clear()
#             ax.set_title(f'Particle Trajectory with {factorcategory} Categories: {unique_id}')
#             ax.set_xlabel('X Position')
#             ax.set_ylabel('Y Position')
#             ax.set_xlim(center_x - range_extent, center_x + range_extent)
#             ax.set_ylim(center_y - range_extent, center_y + range_extent)

#             # Plot the trajectory up to the current point, changing colors according to speed category
#             for j in range(1, i + 1):
#                 x_values = particle_df['x'].iloc[j-1:j+1]
#                 y_values = particle_df['y'].iloc[j-1:j+1]
#                 fac_category = particle_df[factorcategory].iloc[j-1]
#                 color = colormap.get(fac_category, 'black')
#                 ax.plot(x_values, y_values, color=color, linewidth=2)

#             ax.legend(handles=[plt.Line2D([0], [0], color=color, label=f'{category.capitalize()}') for category, color in colormap.items()])
#             plt_path = os.path.join(dir_name, f'frame_{i:03d}.png')
#             plt.savefig(plt_path)
#     else:
#         # Create static plot of the entire trajectory
#         for i in range(1, len(particle_df)):
#             x_values = particle_df['x'].iloc[i-1:i+1]
#             y_values = particle_df['y'].iloc[i-1:i+1]
#             fac_category = particle_df[factorcategory].iloc[i-1]
#             color = colormap.get(fac_category, 'black')
#             ax.plot(x_values, y_values, color=color, linewidth=2)

#         ax.legend(handles=[plt.Line2D([0], [0], color=color, label=f'{category.capitalize()}') for category, color in colormap.items()])
#         static_path = os.path.join(dir_name, f'static_particle_{unique_id}_cat_{threshold_col}.png')
#         plt.savefig(static_path)
#         plt.show()
    
#     plt.close(fig)


############ Keeping this version in case of ffmpeg ##################


# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# def plot_single_particle(particle_df, threshold_col, thresholds, animate=True):
#     """
#     Plots the trajectory of a single particle, either as an animation or as a static image, 
#     with each segment colored according to its speed category.
    
#     Parameters:
#     - particle_df: DataFrame containing the particle's data with columns 'x', 'y', and 'speed_um_s'.
#     - threshold_col: The column used to determine the thresholds for coloring.
#     - thresholds: List or tuple of three numbers defining the boundaries for low, medium, and high categories.
#     - animate: Boolean, if True creates an animation, if False creates a static plot.
#     """
#     # Get the unique ID of the particle
#     unique_id = particle_df['unique_id'].unique()[0]

#     # Ensure the thresholds are sorted
#     thresholds = sorted(thresholds)
    
#     # Assign categories based on thresholds
#     conditions = [
#         (particle_df[threshold_col] >= thresholds[0]) & (particle_df[threshold_col] < thresholds[1]),
#         (particle_df[threshold_col] >= thresholds[1]) & (particle_df[threshold_col] < thresholds[2]),
#         (particle_df[threshold_col] >= thresholds[2])
#     ]
#     choices = ['low', 'medium', 'high']
#     factorcategory = f'{threshold_col}_category'
#     particle_df[factorcategory] = np.select(conditions, choices, default='unknown')
    
#     # Define a colormap for the categories
#     colormap = {
#         'low': 'blue',
#         'medium': 'green',
#         'high': 'red'
#     }
    
#     # Calculate center and range for square plot
#     center_x = (particle_df['x'].max() + particle_df['x'].min()) / 2
#     center_y = (particle_df['y'].max() + particle_df['y'].min()) / 2
#     range_extent = max(particle_df['x'].max() - particle_df['x'].min(), particle_df['y'].max() - particle_df['y'].min()) / 2
#     range_extent *= 1.1  # Add some padding
    
#     # Create directory for saving video or images
#     dir_name = f'{config.MASTER}visualization/particle_{unique_id}_cat_{threshold_col}'
#     os.makedirs(dir_name, exist_ok=True)
    
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.set_title(f'Particle Trajectory with {factorcategory} Categories: {unique_id}')
#     ax.set_xlabel('X Position')
#     ax.set_ylabel('Y Position')
#     ax.set_xlim(center_x - range_extent, center_x + range_extent)
#     ax.set_ylim(center_y - range_extent, center_y + range_extent)

#     # Initial plot for animation
#     line, = ax.plot([], [], lw=2)

#     def init():
#         line.set_data([], [])
#         return line,

#     def update_plot(i):
#         x_values = particle_df['x'].iloc[:i+1]
#         y_values = particle_df['y'].iloc[:i+1]
#         colors = [colormap.get(particle_df[factorcategory].iloc[j], 'black') for j in range(i)]
#         line.set_data(x_values, y_values)
#         line.set_color(colors[-1] if colors else 'black')
#         return line,

#     if animate:
#         # Create animation
#         ani = animation.FuncAnimation(fig, update_plot, init_func=init, frames=len(particle_df), blit=True, repeat=False)
#         video_path = os.path.join(dir_name, f'particle_{unique_id}_cat_{threshold_col}.mp4')
#         ani.save(video_path, writer='ffmpeg', fps=5)
#     else:
#         # Create static plot of the entire trajectory
#         for i in range(1, len(particle_df)):
#             x_values = particle_df['x'].iloc[i-1:i+1]
#             y_values = particle_df['y'].iloc[i-1:i+1]
#             fac_category = particle_df[factorcategory].iloc[i-1]
#             color = colormap.get(fac_category, 'black')
#             ax.plot(x_values, y_values, color=color, linewidth=2)

#         ax.legend(handles=[plt.Line2D([0], [0], color=color, label=f'{category.capitalize()}') for category, color in colormap.items()])
#         static_path = os.path.join(dir_name, f'static_particle_{unique_id}_cat_{threshold_col}.png')
#         plt.savefig(static_path)
#         plt.show()
    
#     plt.close(fig)

########################## FFMPEG version above ######################

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

def plot_single_particle(particle_df, threshold_col, thresholds, animate=True):
    """
    Plots the trajectory of a single particle, either as an animation or as a static image, 
    with each segment colored according to its speed category.
    
    Parameters:
    - particle_df: DataFrame containing the particle's data with columns 'x', 'y', and 'speed_um_s'.
    - threshold_col: The column used to determine the thresholds for coloring.
    - thresholds: List or tuple of three numbers defining the boundaries for low, medium, and high categories.
    - animate: Boolean, if True creates an animation, if False creates a static plot.
    """
    # Get the unique ID of the particle
    unique_id = particle_df['unique_id'].unique()[0]

    # Ensure the thresholds are sorted
    thresholds = sorted(thresholds)
    
    # Assign categories based on thresholds
    conditions = [
        (particle_df[threshold_col] >= thresholds[0]) & (particle_df[threshold_col] < thresholds[1]),
        (particle_df[threshold_col] >= thresholds[1]) & (particle_df[threshold_col] < thresholds[2]),
        (particle_df[threshold_col] >= thresholds[2])
    ]
    choices = ['low', 'medium', 'high']
    factorcategory = f'{threshold_col}_category'
    particle_df[factorcategory] = np.select(conditions, choices, default='unknown')
    
    # Define a colormap for the categories
    # colormap = {
    #     'low': 'blue',
    #     'medium': 'green',
    #     'high': 'red'
    # }

    colormap = {
    'low': '#1F77B4',    # Hex color for 'low'
    'medium': '#FF7F0E', # Hex color for 'medium'
    'high': '#2CA02C'    # Hex color for 'high'
    }
    
    # Calculate center and range for square plot
    center_x = (particle_df['x'].max() + particle_df['x'].min()) / 2
    center_y = (particle_df['y'].max() + particle_df['y'].min()) / 2
    range_extent = max(particle_df['x'].max() - particle_df['x'].min(), particle_df['y'].max() - particle_df['y'].min()) / 2
    range_extent *= 1.1  # Add some padding
    
    # Create directory for saving video or images
    # dir_name = f'{config.MASTER}visualization/particle_{unique_id}_cat_{threshold_col}'
    # os.makedirs(dir_name, exist_ok=True)

    # if animate:
    #     dir_name = f'{config.MASTER}visualization/particle_{unique_id}_cat_{threshold_col}'
    #     os.makedirs(dir_name, exist_ok=True)
    # else:
    dir_name = f'{config.MASTER}visualization\singleparticleplots'
    os.makedirs(dir_name, exist_ok=True)

    fontsizes = 16
    plt.rcParams.update({'font.size': fontsizes})
    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
    
    ax.set_title(f'Particle Trajectory with {factorcategory} Categories: {unique_id}', fontsize=fontsizes)
    ax.set_xlabel('X Position', fontsize=fontsizes)
    ax.set_ylabel('Y Position', fontsize=fontsizes)
    ax.set_xlim(center_x - range_extent, center_x + range_extent)
    ax.set_ylim(center_y - range_extent, center_y + range_extent)

    # Adjust the font size of the axis ticks
    ax.tick_params(axis='both', which='major', labelsize=fontsizes)

    # List to store frames
    frames = []

    def update_plot(i):
        ax.clear()
        ax.set_title(f'Particle Trajectory with {factorcategory} Categories: {unique_id}', fontsize=fontsizes)
        ax.set_xlabel('X Position', fontsize=fontsizes)
        ax.set_ylabel('Y Position', fontsize=fontsizes)
        ax.set_xlim(center_x - range_extent, center_x + range_extent)
        ax.set_ylim(center_y - range_extent, center_y + range_extent)

        # Adjust the font size of the axis ticks in the update function as well
        ax.tick_params(axis='both', which='major', labelsize=fontsizes)

        # Plot the trajectory up to the current point, changing colors according to category
        for j in range(1, i + 1):
            x_values = particle_df['x'].iloc[j-1:j+1]
            y_values = particle_df['y'].iloc[j-1:j+1]
            fac_category = particle_df[factorcategory].iloc[j-1]
            color = colormap.get(fac_category, 'black')
            ax.plot(x_values, y_values, color=color, linewidth=2)

        ax.legend(handles=[plt.Line2D([0], [0], color=color, label=f'{category.capitalize()}') for category, color in colormap.items()], fontsize=fontsizes)


        # Save frame to list for GIF creation
        fig.canvas.draw()
        image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames.append(image)

    if animate:
        # Create animation frame by frame
        for i in range(1, len(particle_df)):
            update_plot(i)
        
        # Save the frames as a GIF
        gif_path = os.path.join(dir_name, f'particle_{unique_id}_cat_{threshold_col}.gif')
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=50, loop=0)


        # frames[0].save('output.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)
    else:
        # Create static plot of the entire trajectory
        for i in range(1, len(particle_df)):
            x_values = particle_df['x'].iloc[i-1:i+1]
            y_values = particle_df['y'].iloc[i-1:i+1]
            fac_category = particle_df[factorcategory].iloc[i-1]
            color = colormap.get(fac_category, 'black')
            ax.plot(x_values, y_values, color=color, linewidth=2)

        ax.legend(handles=[plt.Line2D([0], [0], color=color, label=f'{category.capitalize()}') for category, color in colormap.items()])
        static_path = os.path.join(dir_name, f'static_particle_{unique_id}_cat_{threshold_col}.png')
        plt.savefig(static_path)
        plt.show()
    
    plt.close(fig)


def plot_single_particle_wrapper(time_windowed_df, metrics_df, filter_col, low=None, high=None, condition=None, location=None, threshold_col='speed_um_s', thresholds=None, animate=False):
    """
    Wrapper function to filter dataframes, extract a single particle, and plot its track.
    
    Parameters:
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
    
    Returns:
    - None: Displays and saves the plot.
    """
    
    # Step 1: Filter the time_windowed_df and extract unique IDs
    filtered_df, unique_ids = generalized_filter(time_windowed_df, filter_col, low, high, condition, location)
    
    # Step 2: Filter the metrics_df by the extracted unique IDs
    metrics_df_filtered = metrics_df[metrics_df['unique_id'].isin(unique_ids)]
    
    # Step 3: Extract a single particle track
    single_particle_df = extract_single_particle_df(metrics_df_filtered)
    if thresholds is None:
        thresholds = [0, 10, 15, 10000]

    # Step 4: Plot the single particle track
    plot_single_particle(single_particle_df, threshold_col, thresholds, animate)

    # Optional: Save the plot 
    # plt.savefig(f'single_particle_plot_{single_particle_df["unique_id"].iloc[0]}.png')
    # plt.show()


def plot_jointplot(data_df, x_var, y_var, font_size=12, palette='colorblind', separate=None, show_plot=True, master_dir=None, 
                   grid=True, bw=False, y_min=None, y_max=None, x_min=None, x_max=None, figsize=(10, 8), order=None, 
                   kind='reg', height=7, color=None, point_size=50, small_multiples=False):
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
        ax.text(0.05, 0.95, f'$R^2 = {r_squared:.2f}$', transform=ax.transAxes, fontsize=font_size, verticalalignment='top')

    if separate is not None:
        if order:
            data_df[separate] = pd.Categorical(data_df[separate], categories=order, ordered=True)
        else:
            data_df[separate] = pd.Categorical(data_df[separate])

    if small_multiples and separate:
        unique_categories = data_df[separate].cat.categories
        colors = sns.color_palette(palette, len(unique_categories))

        for i, category in enumerate(unique_categories):
            subset = data_df[data_df[separate] == category]

            if kind == 'hex':
                g = sns.jointplot(x=x_var, y=y_var, data=subset, kind=kind, height=height, color=colors[i])
            else:
                g = sns.jointplot(x=x_var, y=y_var, data=subset, kind=kind, height=height, color=colors[i], scatter_kws={'s': point_size})

            g.fig.suptitle(f'{category}', fontsize=font_size + 2)
            g.ax_joint.set_xlabel(x_var, fontsize=font_size)
            g.ax_joint.set_ylabel(y_var, fontsize=font_size)

            g.ax_joint.set_xlim(left=x_min, right=x_max)
            g.ax_joint.set_ylim(bottom=y_min, top=y_max)
            g.ax_joint.set_xticks(np.arange(x_min, x_max + x_tick_interval, x_tick_interval))
            g.ax_joint.set_yticks(np.arange(y_min, y_max + y_tick_interval, y_tick_interval))
            add_r_squared(g.ax_joint, subset[x_var], subset[y_var])

            plt.tight_layout()

            if master_dir is None:
                master_dir = "plots"

            os.makedirs(master_dir, exist_ok=True)
            filename = f"{master_dir}/{category}_{y_var}_vs_{x_var}_jointplot.png"
            plt.savefig(filename, bbox_inches='tight')

            if show_plot:
                plt.show()
            else:
                plt.close()

    else:
        if separate:
            g = sns.FacetGrid(data_df, hue=separate, palette=palette, height=height, aspect=1.5)
            g.map(sns.regplot, x_var, y_var, scatter_kws={'s': point_size}, ci=None)
            g.add_legend()
            for ax in g.axes.flatten():
                for category in data_df[separate].unique():
                    subset = data_df[data_df[separate] == category]
                    add_r_squared(ax, subset[x_var], subset[y_var])

        else:
            g = sns.jointplot(x=x_var, y=y_var, data=data_df, kind=kind, height=height, color=color if color else sns.color_palette(palette, 1)[0], scatter_kws={'s': point_size})

        g.ax_joint.set_xlim(left=x_min, right=x_max)
        g.ax_joint.set_ylim(bottom=y_min, top=y_max)
        g.ax_joint.set_xticks(np.arange(x_min, x_max + x_tick_interval, x_tick_interval))
        g.ax_joint.set_yticks(np.arange(y_min, y_max + y_tick_interval, y_tick_interval))
        
        add_r_squared(g.ax_joint, data_df[x_var], data_df[y_var])

        g.set_axis_labels(x_var, y_var, fontsize=font_size)
        plt.suptitle(f'{y_var} vs {x_var}', fontsize=font_size, y=1.02)

        plt.tight_layout()

        if master_dir is None:
            master_dir = "plots"

        os.makedirs(master_dir, exist_ok=True)
        filename = f"{master_dir}/{y_var}_vs_{x_var}_jointplot.png"
        plt.savefig(filename, bbox_inches='tight')

        if show_plot:
            plt.show()
        else:
            plt.close()




def plot_joint_with_fit(data_df, x_var, y_var, font_size=12, palette='colorblind', separate=None, show_plot=True, master_dir=None, 
                        grid=True, bw=False, y_max=None, x_max=None, figsize=(10, 8), tick_interval=5, order=None, 
                        fit_type='linear', scatter=True, kind='reg', height=7, color=None, point_size=50):
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

    scatter_kws = {'s': point_size}

    # Apply the order if specified
    if separate and order:
        data_df[separate] = pd.Categorical(data_df[separate], categories=order, ordered=True)

    if separate:
        unique_categories = data_df[separate].cat.categories if order else data_df[separate].unique()
        colors = sns.color_palette(palette, len(unique_categories))

        for i, category in enumerate(unique_categories):
            subset = data_df[data_df[separate] == category]
            reg_color = colors[i] if color is None else color

            g = sns.jointplot(x=x_var, y=y_var, data=subset, kind=kind, height=height, color=reg_color, scatter_kws=scatter_kws)

            if fit_type == 'linear':
                sns.regplot(x=x_var, y=y_var, data=subset, scatter=scatter, color=reg_color, ci=None, ax=g.ax_joint)
            elif fit_type == 'polynomial':
                sns.regplot(x=x_var, y=y_var, data=subset, scatter=scatter, color=reg_color, ci=None, order=2, ax=g.ax_joint)
            elif fit_type == 'exponential':
                log_y = np.log(subset[y_var])
                sns.regplot(x=x_var, y=log_y, data=subset, scatter=scatter, color=reg_color, ci=None, ax=g.ax_joint)
                g.ax_joint.set_ylabel(f'log({y_var})')

            if y_max is not None:
                g.ax_joint.set_ylim(top=y_max)
            if x_max is not None:
                g.ax_joint.set_xlim(right=x_max)
                g.ax_joint.set_xticks(np.arange(0, x_max + 1, tick_interval))

            g.ax_joint.set_xlabel(x_var, fontsize=font_size)
            g.ax_joint.set_ylabel(y_var, fontsize=font_size)
            g.ax_joint.set_title(f'{category}: {y_var} vs {x_var}', fontsize=font_size, pad=20)

            plt.tight_layout()

            if master_dir is None:
                master_dir = "plots"  # Default directory

            os.makedirs(master_dir, exist_ok=True)
            filename = f"{master_dir}/{category}_{y_var}_vs_{x_var}_jointplot_fit_{fit_type}.png"
            plt.savefig(filename, bbox_inches='tight')

            if show_plot:
                plt.show()
            else:
                plt.close()

    else:
        g = sns.jointplot(x=x_var, y=y_var, data=data_df, kind=kind, height=height, color=color if color else sns.color_palette(palette, 1)[0], scatter_kws=scatter_kws)

        if fit_type == 'linear':
            sns.regplot(x=x_var, y=y_var, data=data_df, scatter=scatter, color=color if color else sns.color_palette(palette, 1)[0], ci=None, ax=g.ax_joint)
        elif fit_type == 'polynomial':
            sns.regplot(x=x_var, y=y_var, data=data_df, scatter=scatter, color=color if color else sns.color_palette(palette, 1)[0], ci=None, order=2, ax=g.ax_joint)
        elif fit_type == 'exponential':
            log_y = np.log(data_df[y_var])
            sns.regplot(x=x_var, y=log_y, data=data_df, scatter=scatter, color=color if color else sns.color_palette(palette, 1)[0], ci=None, ax=g.ax_joint)
            g.ax_joint.set_ylabel(f'log({y_var})')

        if y_max is not None:
            g.ax_joint.set_ylim(top=y_max)
        if x_max is not None:
            g.ax_joint.set_xlim(right=x_max)
            g.ax_joint.set_xticks(np.arange(0, x_max + 1, tick_interval))

        g.ax_joint.set_xlabel(x_var, fontsize=font_size)
        g.ax_joint.set_ylabel(y_var, fontsize=font_size)
        g.ax_joint.set_title(f'{y_var} vs {x_var}', fontsize=font_size, pad=20)

        plt.tight_layout()

        if master_dir is None:
            master_dir = "plots"  # Default directory

        os.makedirs(master_dir, exist_ok=True)
        filename = f"{master_dir}/{y_var}_vs_{x_var}_jointplot_fit_{fit_type}.png"
        plt.savefig(filename, bbox_inches='tight')

        if show_plot:
            plt.show()
        else:
            plt.close()




def plot_combo_hist_scatter_kde(data_df, x_var, y_var, font_size=12, palette='mako', scatter_color=".15", hist_bins=50, kde_levels=5, 
                                figsize=(6, 6), separate=None, order=None, x_min=None, x_max=None, y_min=None, y_max=None, horizontal=False):
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
            data_df[separate] = pd.Categorical(data_df[separate], categories=order, ordered=True)
        else:
            data_df[separate] = pd.Categorical(data_df[separate])

        unique_categories = data_df[separate].cat.categories
        num_categories = len(unique_categories)

        # Create subplots horizontally or vertically based on the 'horizontal' flag
        if horizontal:
            fig, axes = plt.subplots(ncols=num_categories, figsize=(figsize[0] * num_categories, figsize[1]))
        else:
            fig, axes = plt.subplots(nrows=num_categories, figsize=(figsize[0], figsize[1] * num_categories))

        if num_categories == 1:
            axes = [axes]  # Ensure axes is iterable if there's only one subplot

        for i, category in enumerate(unique_categories):
            subset = data_df[data_df[separate] == category]
            ax = axes[i]

            # Scatterplot
            sns.scatterplot(x=x_var, y=y_var, data=subset, s=5, color=scatter_color, ax=ax)

            # 2D Histogram
            sns.histplot(x=x_var, y=y_var, data=subset, bins=hist_bins, pthresh=.1, cmap=palette, ax=ax)

            # KDE Plot
            sns.kdeplot(x=x_var, y=y_var, data=subset, levels=kde_levels, color="w", linewidths=1, ax=ax)

            ax.set_xlabel(x_var, fontsize=font_size)
            ax.set_ylabel(y_var, fontsize=font_size)
            ax.set_title(f'{category}', fontsize=font_size + 2)

            ax.tick_params(axis='both', labelsize=font_size)

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
        sns.histplot(x=x_var, y=y_var, data=data_df, bins=hist_bins, pthresh=.1, cmap=palette, ax=ax)

        # KDE Plot
        sns.kdeplot(x=x_var, y=y_var, data=data_df, levels=kde_levels, color="w", linewidths=1, ax=ax)

        ax.set_xlabel(x_var, fontsize=font_size)
        ax.set_ylabel(y_var, fontsize=font_size)
        ax.set_title(f'{y_var} vs {x_var} with Hist and KDE', fontsize=font_size + 2)

        ax.tick_params(axis='both', labelsize=font_size)

        if x_min is not None and x_max is not None:
            ax.set_xlim(x_min, x_max)
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
    
        plt.tight_layout()
        plt.show()

