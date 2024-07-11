import os
import pims
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns

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


def plot_histograms(traj_df, framerate=0.1, bins=100, coltoseparate = 'tracker', xlimit=None):
    """
    Plot histograms of track lengths in seconds for each tracker, with consistent binning.

    Parameters
    ----------
    traj_df : DataFrame
        DataFrame containing track data with columns 'tracker', 'unique_id', and 'filename'.
    framerate : float, optional
        Frame rate in seconds per frame. Default is 0.1.
    bins : int, optional
        Number of bins for the histogram. Default is 100.
    """
    plt.figure(figsize=(20, 12))
    size = 10
    multiplier = 2
    sns.set_context("notebook", rc={"xtick.labelsize": size*multiplier, "ytick.labelsize": size*multiplier})
    
    max_track_length = traj_df['unique_id'].value_counts().max() * framerate
    bin_edges = np.linspace(0, max_track_length, bins + 1)
    
    for i, tracker in enumerate(traj_df[coltoseparate].unique()):
        subset = traj_df[traj_df[coltoseparate] == tracker]
        subsetvalues = subset['unique_id'].value_counts()
        subsetvalues_seconds = subsetvalues * framerate
        
        # Calculate percentage counts
        counts, _ = np.histogram(subsetvalues_seconds, bins=bin_edges)
        percentage_counts = (counts / counts.sum()) * 100
        
        # Plot histogram
        sns.histplot(subsetvalues_seconds, bins=bin_edges, kde=True, label=tracker, alpha=0.5, stat="percent")
        
        subset_mean = subsetvalues_seconds.mean()
        subset_median = subsetvalues_seconds.median()
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
