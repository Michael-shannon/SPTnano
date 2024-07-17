import os
import scipy.io
import pandas as pd
import numpy as np
from IPython.display import Markdown, display

def generate_file_tree(startpath):
    tree = []
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        tree.append(f"{indent}- {os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            tree.append(f"{subindent}- {f}")
    return "\n".join(tree)

def display_file_tree(startpath):
    file_tree = generate_file_tree(startpath)
    display(Markdown(f"```\n{file_tree}\n```"))

# def uniqueID_microns_and_seconds(df, pixelsize_microns, time_between_frames):
#     '''Adds columns to the DataFrame with positions in microns and time in seconds'''
#     # unique id and file identifier
#     df['file_id'] = pd.Categorical(df['filename']).codes
#     df['unique_id'] = df['file_id'].astype(str) + '_' + df['particle'].astype(str)
#     #space transformations
#     df['x_um'] = df['x'] * pixelsize_microns
#     df['y_um'] = df['y'] * pixelsize_microns
#     # time transformations
#     df['frame_zeroed'] = df.groupby('unique_id')['frame'].transform(lambda x: x - x.iloc[0])
#     df['time_s'] = df['frame'] * time_between_frames
#     df['time_s_zeroed'] = df.groupby('unique_id')['time_s'].transform(lambda x: x - x.iloc[0])
#     return df

def add_unique_id(df):
    df['file_id'] = pd.Categorical(df['filename']).codes
    df['unique_id'] = df['file_id'].astype(str) + '_' + df['particle'].astype(str)
    return df

def add_microns_and_secs(df, pixelsize_microns, time_between_frames):
    '''Adds columns to the DataFrame with positions in microns and time in seconds'''
    #space transformations
    df['x_um'] = df['x'] * pixelsize_microns
    df['y_um'] = df['y'] * pixelsize_microns

    df['frame_zeroed'] = df.groupby('particle')['frame'].transform(lambda x: x - x.iloc[0])
    df['time_s'] = df['frame'] * time_between_frames
    df['time_s_zeroed'] = df.groupby('particle')['time_s'].transform(lambda x: x - x.iloc[0])
    return df



# Define a function to read .mat files and convert them to DataFrames
def read_mat_file(file_path):
    mat = scipy.io.loadmat(file_path)
    tr_data = mat['data']['tr'][0, 0]
    
    # Initialize an empty list to store the combined data
    combined_data = []

    # Iterate over the cells in tr_data and concatenate them
    for cell in tr_data[0]:
        if cell.size > 0:
            combined_data.append(cell)

    # Convert the combined data into a numpy array
    combined_data = np.vstack(combined_data)

    # Create a DataFrame with the appropriate columns
    # Adjust column names based on inspection
    df = pd.DataFrame(combined_data, columns=['x', 'y', 'frame', 'particle', 'column5', 'column6', 'column7', 'column8'])
    
    return df

# Filter stubs
def filter_stubs(df, min_time):

    '''
    Removes tracks that are shorter than 'min_time' by finding the max duration of each time_s_zeroed column and filtering on that
    Works across exposure times, because it works on converted seconds, not frames

    '''
    # Calculate the duration of each track by grouping by 'particle' and using the 'time_s' column
    track_durations = df.groupby('unique_id')['time_s_zeroed'].max() 
    # Identify particles with tracks longer than 0.2 seconds
    valid_particles = track_durations[track_durations >= min_time].index
    # Filter the dataframe to include only valid particles
    filtered_df = df[df['unique_id'].isin(valid_particles)]

    return filtered_df

def extract_single_particle_df(df, condition, unique_id):
    chosen_df = df[df['condition'] == condition]
    particle_ids = chosen_df.unique_id.unique()
    # choose a random particle from the list
    chosen_particle = np.random.choice(particle_ids)
    chosen_df = df[df['condition'] == condition]
    single_particle_df = chosen_df[chosen_df['unique_id'] == chosen_particle]
    return single_particle_df 



def filter_large_jumps(df, threshold):
    """
    Filter out entire particles with any frames showing large jumps in micrometers.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing tracking data with a 'segment_len_um' column.
    threshold : float
        Threshold for what constitutes a large jump in micrometers.

    Returns
    -------
    DataFrame
        DataFrame with particles having large jumps filtered out.
    """
    # Identify unique_ids with any large jumps
    large_jump_particles = df[df['segment_len_um'] > threshold]['unique_id'].unique()
    
    # Filter out particles with large jumps
    df_filtered = df[~df['unique_id'].isin(large_jump_particles)].copy()
    # df_filtered.drop(columns=['x_um_prev', 'y_um_prev', 'segment_len_um'], inplace=True)
    return df_filtered

def filter_high_speeds(metrics_df, speed_threshold):
    '''
    Filter based on speed instead - can be relevant if you have different exposure times and different times between frames
    '''

    # Identify unique_ids with any high speeds
    high_speed_particles = metrics_df[metrics_df['speed_um_s'] > speed_threshold]['unique_id'].unique()

    # Filter out particles with high speeds
    metrics_df_filtered = metrics_df[~metrics_df['unique_id'].isin(high_speed_particles)].copy()
    return metrics_df_filtered

