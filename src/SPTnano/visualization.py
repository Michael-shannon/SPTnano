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

import config




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



    


# def plot_histograms(data_df, feature, bins=100, separate=None, xlimit=None, small_multiples=False, palette='colorblind', use_kde=False, show_plot=True, master_dir=None, tick_interval=5, average='mean', order=None):
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

#     data_df = data_df.replace(0, np.nan)

#     if master_dir is None:
#         master_dir = config.master  # Use the master directory from config if not provided

#     if separate is not None and order is not None:
#         # Ensure the data is ordered according to the specified order
#         data_df[separate] = pd.Categorical(data_df[separate], categories=order, ordered=True)

#     # Use the categories attribute to maintain the specified order
#     if separate is not None:
#         unique_categories = data_df[separate].cat.categories
#     else:
#         unique_categories = [None]

#     color_palette = sns.color_palette(palette, len(unique_categories))

#     if small_multiples and separate is not None:
#         num_categories = len(unique_categories)
#         fig, axes = plt.subplots(num_categories, 1, figsize=(20, 6 * num_categories), sharex=True)
#         # multiplier = 
#         if num_categories == 1:
#             axes = [axes]  # To handle the case with only one subplot
        
#         for i, category in enumerate(unique_categories):
#             if pd.isna(category):
#                 continue
#             subset = data_df[data_df[separate] == category]
#             subsetvalues = subset[feature]
            
#             max_value = subsetvalues.max()
#             bin_edges = np.linspace(0, max_value, bins + 1)
            
#             # Plot histogram or KDE
#             if use_kde:
#                 sns.kdeplot(subsetvalues, fill=True, ax=axes[i], color=color_palette[i])
#             else:
#                 sns.histplot(subsetvalues, bins=bin_edges, kde=False, ax=axes[i], stat="percent", color=color_palette[i])
            
#             # Plot average line
#             if average == 'mean':
#                 avg_value = subsetvalues.mean()
#                 axes[i].axvline(avg_value, color='black', linestyle='--')
#                 axes[i].text(0.4, 0.6, f"Mean: {avg_value:.2f}", transform=axes[i].transAxes, fontsize=16)
#             elif average == 'median':
#                 avg_value = subsetvalues.median()
#                 axes[i].axvline(avg_value, color='black', linestyle='--')
#                 axes[i].text(0.4, 0.6, f"Median: {avg_value:.2f}", transform=axes[i].transAxes, fontsize=16)
            
#             axes[i].set_title(f'{category}', fontsize=16)
#             axes[i].tick_params(axis='both', which='major', labelsize=16)
#             axes[i].set_xlabel(f'{feature}', fontsize=16)
#             axes[i].set_ylabel('Percentage', fontsize=16)
            
#             if xlimit is not None:
#                 axes[i].set_xlim(0, xlimit)
#             else:
#                 axes[i].set_xlim(0, max_value)
        
#         plt.xlabel(f'{feature}', fontsize=16)
#         plt.tight_layout()
    
#     else:
#         plt.figure(figsize=(20, 12))
#         size = 10
#         multiplier = 2
#         sns.set_context("notebook", rc={"xtick.labelsize": size * multiplier, "ytick.labelsize": size * multiplier})
        
#         max_value = data_df[feature].max()
#         bin_edges = np.linspace(0, max_value, bins + 1)
        
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
#                 plt.text(0.4, 0.6, f"Overall Mean: {avg_value:.2f}", transform=plt.gca().transAxes, fontsize=10 * multiplier)
#             elif average == 'median':
#                 avg_value = subsetvalues.median()
#                 plt.axvline(avg_value, color='b', linestyle='--')
#                 plt.text(0.4, 0.6, f"Overall Median: {avg_value:.2f}", transform=plt.gca().transAxes, fontsize=10 * multiplier)
        
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
#                 plt.text(0.4, 0.6 - shift, f"{category}: {average.capitalize()}: {avg_value:.2f} from {number_of_tracks} tracks", transform=plt.gca().transAxes, fontsize=10 * multiplier)
        
#         plt.xlabel(f'{feature}', fontsize=size * multiplier)
#         plt.ylabel('Percentage', fontsize=size * multiplier)
#         plt.legend(title='', fontsize=size * multiplier)
#         ax = plt.gca()
#         if xlimit is not None:
#             ax.set_xlim(0, xlimit)
#         else:
#             ax.set_xlim(0, max_value)
        
#         ax.set_xticks(np.arange(0, max_value + 1, tick_interval))  # Ensure ticks are at integer intervals
#         ax.set_xlim(0, xlimit or max_value)  # Start x-axis at 0
#         ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)  # Add faint gridlines

#     # Create directory for plots if it doesn't exist
#     plots_dir = os.path.join(master_dir, 'plots')
#     os.makedirs(plots_dir, exist_ok=True)
    
#     # Generate filename
#     kde_text = 'kde' if use_kde else 'histogram'
#     average_text = f'{average}'
#     if small_multiples:
#         multitext = 'small_multiples'
#     else:
#         multitext = 'single_plot'

#     filename = f"{plots_dir}/{kde_text}_{feature}_{average_text}_{multitext}.png"
    
#     # Save plot
#     plt.savefig(filename, bbox_inches='tight')
#     print(f"Plot saved as {filename}")
    
#     # Show plot if specified
#     if show_plot:
#         plt.show()
#     else:
#         plt.close()


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_histograms(data_df, feature, bins=100, separate=None, xlimit=None, small_multiples=False, palette='colorblind', use_kde=False, show_plot=True, master_dir=None, tick_interval=5, average='mean', order=None, grid=False):
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
    """

    if master_dir is None:
        master_dir = config.master  # Use the master directory from config if not provided

    if separate is not None and order is not None:
        # Ensure the data is ordered according to the specified order
        data_df[separate] = pd.Categorical(data_df[separate], categories=order, ordered=True)

    textpositionx=  0.6
    textpositiony= 0.8

    # Use the categories attribute to maintain the specified order
    if separate is not None:
        unique_categories = data_df[separate].cat.categories
    else:
        unique_categories = [None]

    color_palette = sns.color_palette(palette, len(unique_categories))

    # Determine global maximum y-value for consistent y-axis limits
    global_max_y = 0

    if small_multiples and separate is not None:
        num_categories = len(unique_categories)
        fig, axes = plt.subplots(num_categories, 1, figsize=(20, 6 * num_categories), sharex=True)
        
        if num_categories == 1:
            axes = [axes]  # To handle the case with only one subplot
        
        for i, category in enumerate(unique_categories):
            if pd.isna(category):
                continue
            subset = data_df[data_df[separate] == category]
            subsetvalues = subset[feature]
            
            max_value = subsetvalues.max()
            bin_edges = np.linspace(0, max_value, bins + 1)
            
            # Plot histogram or KDE
            if use_kde:
                sns.kdeplot(subsetvalues, fill=True, ax=axes[i], color=color_palette[i])
                current_max_y = axes[i].get_ylim()[1]  # Get the current maximum y-value from the plot
            else:
                plot = sns.histplot(subsetvalues, bins=bin_edges, kde=False, ax=axes[i], stat="percent", color=color_palette[i])
                current_max_y = plot.get_ylim()[1]
            
            # Update global maximum y-value
            if current_max_y > global_max_y:
                global_max_y = current_max_y
            
            # Plot average line
            if average == 'mean':
                avg_value = subsetvalues.mean()
                axes[i].axvline(avg_value, color='black', linestyle='--')
                axes[i].text(textpositionx, textpositiony, f"Mean: {avg_value:.2f}", transform=axes[i].transAxes, fontsize=16)
            elif average == 'median':
                avg_value = subsetvalues.median()
                axes[i].axvline(avg_value, color='black', linestyle='--')
                axes[i].text(textpositionx, textpositiony, f"Median: {avg_value:.2f}", transform=axes[i].transAxes, fontsize=16)
            
            axes[i].set_title(f'{category}', fontsize=16)
            axes[i].tick_params(axis='both', which='major', labelsize=16)
            axes[i].set_xlabel(f'{feature}', fontsize=16)
            axes[i].set_ylabel('Percentage', fontsize=16)
            
            if xlimit is not None:
                axes[i].set_xlim(0, xlimit)
            else:
                axes[i].set_xlim(0, max_value)
        
        # Set common y-axis limits for all subplots
        for ax in axes:
            ax.set_ylim(0, global_max_y)
        
        plt.tight_layout()
    
    else:
        plt.figure(figsize=(20, 12))
        sns.set_context("notebook", rc={"xtick.labelsize": 16, "ytick.labelsize": 16})
        
        max_value = data_df[feature].max()
        bin_edges = np.linspace(0, max_value, bins + 1)
        
        if separate is None:
            subsetvalues = data_df[feature]
            
            # Plot histogram or KDE
            if use_kde:
                sns.kdeplot(subsetvalues, fill=True, alpha=0.5, color=color_palette[0])
            else:
                sns.histplot(subsetvalues, bins=bin_edges, kde=False, alpha=0.5, stat="percent", color=color_palette[0])
            
            # Plot average line
            if average == 'mean':
                avg_value = subsetvalues.mean()
                plt.axvline(avg_value, color='r', linestyle='--')
                plt.text(textpositionx, textpositiony, f"Overall Mean: {avg_value:.2f}", transform=plt.gca().transAxes, fontsize=16)
            elif average == 'median':
                avg_value = subsetvalues.median()
                plt.axvline(avg_value, color='b', linestyle='--')
                plt.text(0.4, 0.6, f"Overall Median: {avg_value:.2f}", transform=plt.gca().transAxes, fontsize=16)
        
        else:
            for i, category in enumerate(unique_categories):
                if pd.isna(category):
                    continue
                subset = data_df[data_df[separate] == category]
                subsetvalues = subset[feature]
                
                # Plot histogram or KDE
                if use_kde:
                    sns.kdeplot(subsetvalues, fill=True, label=category, alpha=0.5, color=color_palette[i])
                else:
                    sns.histplot(subsetvalues, bins=bin_edges, kde=False, label=category, alpha=0.5, stat="percent", color=color_palette[i])
                
                # Plot average line
                if average == 'mean':
                    avg_value = subsetvalues.mean()
                    plt.axvline(avg_value, color=color_palette[i], linestyle='--')
                elif average == 'median':
                    avg_value = subsetvalues.median()
                    plt.axvline(avg_value, color=color_palette[i], linestyle='--')
                
                number_of_tracks = len(subset['unique_id'].unique())
                shift = i * 0.05
                plt.text(textpositionx, textpositiony - shift, f"{category}: {average.capitalize()}: {avg_value:.2f} from {number_of_tracks} tracks", transform=plt.gca().transAxes, fontsize=16)
        
        plt.xlabel(f'{feature}', fontsize=16)
        plt.ylabel('Percentage', fontsize=16)
        plt.legend(title='', fontsize=16)
        ax = plt.gca()
        if xlimit is not None:
            ax.set_xlim(0, xlimit)
        else:
            ax.set_xlim(0, max_value)
        
        ax.set_xticks(np.arange(0, max_value + 1, tick_interval))  # Ensure ticks are at integer intervals
        ax.set_xlim(0, xlimit or max_value)  # Start x-axis at 0
        if grid:
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)  # Add faint gridlines
        else:
            print("No grid")

    # Create directory for plots if it doesn't exist
    plots_dir = os.path.join(master_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    histodir = os.path.join(plots_dir, 'histograms')
    os.makedirs(histodir, exist_ok=True)
    
    # Generate filename
    kde_text = 'kde' if use_kde else 'histogram'
    average_text = f'{average}'
    if small_multiples:
        multitext = 'small_multiples'
    else:
        multitext = 'single_plot'

    filename = f"{histodir}/{kde_text}_{feature}_{average_text}_{multitext}.png"
    
    # Save plot
    plt.savefig(filename, bbox_inches='tight')
    
    # Show plot if specified
    if show_plot:
        plt.show()
    else:
        plt.close()




        #### To be refined::: #############

import os
import pims
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from matplotlib.collections import LineCollection

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
    vis_folder = os.path.join(master_folder, 'visualization')
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
from mpl_toolkits.axes_grid1 import make_axes_locatable

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




def napari_visualize_image_with_tracks(tracked_filename, tracks_df, master_dir):
    # Get the condition from the dataframe based on the tracked filename
    condition = get_condition_from_filename(tracks_df, tracked_filename)
    
    # Construct the full file path by removing '_tracked' and adding '.tif'
    image_filename = tracked_filename.replace('_tracked', '') + '.tif'
    image_path = os.path.join(master_dir, condition, image_filename)
    
    # Load the image
    image = load_image(image_path)

    # Load the tracks
    tracks = load_tracks(tracks_df, tracked_filename)

    print(tracks.columns)

    tracks_new_df = tracks[["particle", "frame", "y", "x"]]

    # Extract x, y, and frame information for the tracks
    coords = np.array([tracks['x'].values, tracks['y'].values, tracks['frame'].values]).T
    
    # Start Napari viewer
    viewer = napari.Viewer()

    # Add image layer
    viewer.add_image(image, name=f'Raw {tracked_filename}')

    # Add tracks layer
    viewer.add_tracks(tracks_new_df, name=f'Tracks{tracked_filename}')

    napari.run()



# def napari_visualize_image_with_tracks(tracked_filename, tracks_df, master_dir):
#     # Get the condition from the dataframe based on the tracked filename
#     condition = get_condition_from_filename(tracks_df, tracked_filename)
    
#     # Construct the full file path for the image by removing '_tracked' and adding '.tif'
#     image_filename = tracked_filename.replace('_tracked', '') + '.tif'
#     image_path = os.path.join(master_dir, condition, image_filename)
    
#     # Debugging: Print the constructed file paths
#     print(f"Tracked filename: {tracked_filename}")
#     print(f"Condition: {condition}")
#     print(f"Image filename: {image_filename}")
#     print(f"Image path: {image_path}")
    
#     # Load the image
#     try:
#         image = io.imread(image_path)
#     except FileNotFoundError:
#         print(f"Error: Image file not found at path '{image_path}'")
#         return

#     # Load the tracks
#     tracks = tracks_df[tracks_df['filename'] == tracked_filename]

#     print(tracks.columns)

#     tracks_new_df = tracks[["particle", "frame", "y", "x"]]

#     # Extract x, y, and frame information for the tracks
#     coords = np.array([tracks['x'].values, tracks['y'].values, tracks['frame'].values]).T
    
#     # Start Napari viewer
#     viewer = napari.Viewer()

#     # Add image layer
#     viewer.add_image(image, name='Raw Image')

#     # Add tracks layer
#     viewer.add_tracks(tracks_new_df, name='Tracks')

#     napari.run()


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

# def plot_time_series(data_df, factor_col='speed_um_s', absolute=True, separate_by='condition', palette='colorblind', meanormedian='mean', multiplot=False, talk=False):
#     """
#     Plot time series of a specified factor, with mean as a thick line and confidence intervals as shaded areas.
    
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
#     """
    
#     if not absolute:
#         time_col = 'time_s_zeroed'
#         x_label = 'Time zeroed (s)'
#     else:
#         time_col = 'time_s'
#         x_label = 'Time (s)'

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
#                 ci = subset.groupby(time_col)[factor_col].apply(lambda x: bootstrap_ci_mean(x, num_samples=1000, alpha=0.05))
#             else:
#                 avg_factors = subset.groupby(time_col)[factor_col].median()
#                 ci = subset.groupby(time_col)[factor_col].apply(lambda x: bootstrap_ci_median(x, num_samples=1000, alpha=0.05))

#             color = color_palette[i]
#             label = category

#             ax.plot(avg_factors.index, avg_factors.values, label=label, color=color, linewidth=0.5)
#             ax.fill_between(avg_factors.index, avg_factors - ci, avg_factors + ci, color=color, alpha=0.3)
#             ax.set_xlabel(x_label, fontsize=font_size)
#             ax.set_ylabel(factor_col, fontsize=font_size, labelpad=20)
#             ax.legend(title=separate_by, fontsize=font_size, loc='upper left', bbox_to_anchor=(1, 1))
#             ax.set_title(f'Time Series of {factor_col} - {category}', fontsize=font_size)
        
#         plt.tight_layout()
#     else:
#         fig, ax = plt.subplots(figsize=fig_size)
        
#         for i, category in enumerate(unique_categories):
#             subset = data_df if category is None else data_df[data_df[separate_by] == category]
#             times = subset[time_col]
#             factors = subset[factor_col]

#             if meanormedian == 'mean':
#                 avg_factors = subset.groupby(time_col)[factor_col].mean()
#                 ci = subset.groupby(time_col)[factor_col].apply(lambda x: bootstrap_ci_mean(x, num_samples=1000, alpha=0.05))
#             else:
#                 avg_factors = subset.groupby(time_col)[factor_col].median()
#                 ci = subset.groupby(time_col)[factor_col].apply(lambda x: bootstrap_ci_median(x, num_samples=1000, alpha=0.05))

#             color = color_palette[i]
#             label = 'Overall' if category is None else category

#             ax.plot(avg_factors.index, avg_factors.values, label=label, color=color, linewidth=0.5)
#             ax.fill_between(avg_factors.index, avg_factors - ci, avg_factors + ci, color=color, alpha=0.3)

#         ax.set_xlabel(x_label, fontsize=font_size)
#         ax.set_ylabel(factor_col, fontsize=font_size, labelpad=20)
#         ax.legend(title=separate_by, fontsize=font_size, loc='upper left', bbox_to_anchor=(1.05, 1))
#         ax.set_title(f'Time Series of {factor_col}', fontsize=font_size)
#         plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to fit legend
    
#     plt.show()


def plot_time_series(data_df, factor_col='speed_um_s', absolute=True, separate_by='condition', palette='colorblind', meanormedian='mean', multiplot=False, talk=False, bootstrap=True, show_plot=True, master_dir=None):
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
    """
    max_time = data_df['time_s'].max()
    max_time_zeroed = data_df['time_s_zeroed'].max()
    xmin=0.2 #### A FIX FOR NOW
    
    if master_dir is None:
        master_dir = config.master  # Use the master directory from config if not provided

    if not absolute:
        time_col = 'time_s_zeroed'
        x_label = 'Time zeroed (s)'
        xmax = max_time_zeroed
    else:
        time_col = 'time_s'
        x_label = 'Time (s)'
        xmax = max_time

    unique_categories = data_df[separate_by].unique() if separate_by else [None]
    color_palette = sns.color_palette(palette, len(unique_categories))
    
    # Set figure size and font size based on the `talk` parameter
    if talk:
        fig_size = (40, 12)
        font_size = 35
    else:
        if multiplot and separate_by:
            fig_size = (10, 5 * len(unique_categories))
        else:
            fig_size = (5, 3)
        font_size = 14
    
    sns.set_context("notebook", rc={"lines.linewidth": 2.5, "font.size": font_size, "axes.titlesize": font_size, "axes.labelsize": font_size, "xtick.labelsize": font_size, "ytick.labelsize": font_size})
    
    if multiplot and separate_by:
        fig, axes = plt.subplots(len(unique_categories), 1, figsize=fig_size, sharex=True)
        
        for i, category in enumerate(unique_categories):
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
            ax.fill_between(avg_factors.index[valid_indices], (avg_factors - ci)[valid_indices], (avg_factors + ci)[valid_indices], color=color, alpha=0.3)
            ax.set_xlabel(x_label, fontsize=font_size)
            ax.set_ylabel(factor_col, fontsize=font_size, labelpad=20)
            ax.legend(fontsize=font_size, loc='upper left', bbox_to_anchor=(1, 1))
            # ax.set_title(f'Time Series of {factor_col} - {category}', fontsize=font_size)
            ax.set_xlim(xmin, xmax)
            # Add faint gridlines
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=fig_size)
        
        for i, category in enumerate(unique_categories):
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
            ax.fill_between(avg_factors.index[valid_indices], (avg_factors - ci)[valid_indices], (avg_factors + ci)[valid_indices], color=color, alpha=0.3)

        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(factor_col, fontsize=font_size, labelpad=20)
        ax.legend(fontsize=font_size, loc='upper left', bbox_to_anchor=(1.05, 1))
        # Add faint gridlines
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


        # ax.set_title(f'Time Series of {factor_col}', fontsize=font_size)
        # get the max value of the x

        #

        ax.set_xlim(xmin, xmax)
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
    #set xlim
   
    
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





def plot_violinplots(data_df, factor_col='speed_um_s', separate_by='condition', palette='colorblind', talk=False):
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
    
    # Plot violin plot
    sns.violinplot(x=separate_by, y=factor_col, hue=separate_by, data=data_df, palette=color_palette, ax=ax, legend=False, alpha=0.79)
    
    # Remove 'Condition_' prefix from x tick labels
    new_labels = [label.replace('Condition_', '') for label in unique_categories]
    ax.set_xticks(range(len(new_labels)))
    ax.set_xticklabels(new_labels, fontsize=font_size)

    ax.set_ylabel(factor_col, fontsize=font_size, labelpad=20)
    ax.set_xlabel(None)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tight_layout()
    
    plt.show()