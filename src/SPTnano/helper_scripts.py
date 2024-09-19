import os
import scipy.io
import pandas as pd
import numpy as np
from IPython.display import Markdown, display
from tqdm.notebook import tqdm

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

# def read_mat_BNP_file(file_path):
#     mat = scipy.io.loadmat(file_path)
#     tr_data = mat['data']['tr'][0, 0]
    
#     # Initialize an empty list to store the combined data
#     combined_data = []

#     # Iterate over the cells in tr_data and concatenate them
#     for cell in tr_data[0]:
#         if cell.size > 0:
#             combined_data.append(cell)

#     # Convert the combined data into a numpy array
#     combined_data = np.vstack(combined_data)

#     # Create a DataFrame with the appropriate columns
#     # Adjust column names based on inspection
#     df = pd.DataFrame(combined_data, columns=['x', 'y', 'frame', 'particle', 'column5', 'column6', 'column7', 'column8'])
    
#     return df

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

# def extract_single_particle_df(df, condition, unique_id):
#     chosen_df = df[df['condition'] == condition]
#     particle_ids = chosen_df.unique_id.unique()
#     # choose a random particle from the list
#     chosen_particle = np.random.choice(particle_ids)
#     chosen_df = df[df['condition'] == condition]
#     single_particle_df = chosen_df[chosen_df['unique_id'] == chosen_particle]
#     return single_particle_df 

def extract_single_particle_df(df, unique_id=None):
    """
    Extracts a DataFrame for a single particle based on the specified condition and unique ID.
    Condition and Location can be specified as a string or as an index.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - condition (Union[str, int]): The condition to filter by, or an index to choose from unique conditions.
    - unique_id (Optional[int]): The unique ID of the particle to filter by. If None, a random particle is chosen.
    - location (Union[str, int], optional): The location to filter by, or an index to choose from unique locations. Default is None.
    
    Returns:
    - pd.DataFrame: The filtered DataFrame containing only the selected particle.
    """
    
    # # Handle the condition input (can be a string or an index)
    # unique_conditions = df['condition'].unique()
    # if isinstance(condition, int):
    #     if condition >= len(unique_conditions):
    #         raise ValueError(f"Condition index {condition} is out of bounds.")
    #     condition = unique_conditions[condition]
    
    # # Apply the condition filter
    # chosen_df = df[df['condition'] == condition]
    
    # # Handle the location input (can be a string or an index)
    # if location is not None:
    #     unique_locations = df['Location'].unique()
    #     if isinstance(location, int):
    #         if location >= len(unique_locations):
    #             raise ValueError(f"Location index {location} is out of bounds.")
    #         location = unique_locations[location]
        
    #     # Apply the location filter
    #     chosen_df = chosen_df[chosen_df['Location'] == location]
    
    # If unique_id is not provided, choose a random particle
    if unique_id is None:
        particle_ids = df['unique_id'].unique()
        unique_id = np.random.choice(particle_ids)
    
    # Filter by the chosen particle
    single_particle_df = df[df['unique_id'] == unique_id]
    
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

def sample_dataframe(df, percent_samples):
    # isolate each condition
    conditions = df.condition.unique()
    # initialize a list to store the sampled dataframes
    sampled_dfs = []
    # loop through each condition
    for condition in conditions:
        # isolate the dataframe for that condition
        condition_df = df[df.condition == condition]
        # extract a list of the unique_ids in that df
        unique_ids = condition_df.unique_id.unique()
        # sample those unique ids
        sampled_ids = np.random.choice(unique_ids, int(len(unique_ids) * percent_samples), replace=False)
        # filter the dataframe to only include those unique ids
        sampled_df = condition_df[condition_df.unique_id.isin(sampled_ids)]
        # append the sampled dataframe to the list
        sampled_dfs.append(sampled_df)
    # concatenate the list of dataframes into a single dataframe
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    # report on the sampling and the original dataframe
    print('Original dataframe contains {} tracks'.format(len(df)))
    print('Sampled dataframe contains {} tracks'.format(len(sampled_df)))
    return sampled_df

def generalized_filter(df_in, filter_col, low=None, high=None, condition=None, location=None):
    """
    Filters the input DataFrame based on a range of values in filter_col and additional conditions.

    Parameters:
    - df_in (pd.DataFrame): The input DataFrame to filter.
    - filter_col (str): The column to filter based on a range of values.
    - low (float, optional): The lower bound of the filter range (inclusive). Default is None.
    - high (float, optional): The upper bound of the filter range (inclusive). Default is None.
    - condition (Union[str, int], optional): The specific condition to filter by, or an index to choose from unique conditions. Default is None.
    - location (Union[str, int], optional): The specific location to filter by, or an index to choose from unique locations. Default is None.

    Returns:
    - pd.DataFrame: The filtered DataFrame.
    """
    
    # Start with the input DataFrame
    df_filtered = df_in.copy()
    print(f'Available filters: {df_filtered.columns}')
    
    # Handle the filtering logic for low and high values
    if low is None and high is not None:
        df_filtered = df_filtered[df_filtered[filter_col] <= high]
    elif low is not None and high is None:
        df_filtered = df_filtered[df_filtered[filter_col] >= low]
    elif low is not None and high is not None:
        df_filtered = df_filtered[(df_filtered[filter_col] >= low) & (df_filtered[filter_col] <= high)]
    
    # Handle the condition input (can be a string or an index)
    if condition is not None:
        unique_conditions = df_filtered['condition'].unique()
        if isinstance(condition, int):
            if condition >= len(unique_conditions):
                raise ValueError(f"Condition index {condition} is out of bounds.")
            condition = unique_conditions[condition]
        df_filtered = df_filtered[df_filtered['condition'] == condition]
    
    # Handle the location input (can be a string or an index)
    if location is not None:
        unique_locations = df_filtered['Location'].unique()
        if isinstance(location, int):
            if location >= len(unique_locations):
                raise ValueError(f"Location index {location} is out of bounds - if you think it should be there, check whether you have it in the condition you filtered on")
            location = unique_locations[location]
        df_filtered = df_filtered[df_filtered['Location'] == location]

    unique_ids = df_filtered['unique_id'].unique()
    
    return df_filtered, unique_ids


def optimized_assign_motion_class_by_unique_id(metrics_df, time_windowed_df, window_size=24, overlap=12):
    
    timewindowedids = time_windowed_df.unique_id.unique()
    metricsdfids = metrics_df.unique_id.unique()
    #filter the metrics_df to only include the unique_ids that are in the time_windowed_df
    metrics_df = metrics_df[metrics_df.unique_id.isin(timewindowedids)]
    # get the number of metrics_df unique_ids
    metricsdfidsafter = metrics_df.unique_id.unique()
    percentids = len(metricsdfidsafter)/len(metricsdfids) *100

    # print the percentage of unique_ids that remain
    print(f'Percentage of unique_ids that remain after removing intersection: {percentids}%')
    
    # Calculate the start and end frames for each time window in the time_windowed_df
    time_windowed_df['start_frame'] = time_windowed_df['time_window'] * (window_size - overlap)
    time_windowed_df['end_frame'] = time_windowed_df['start_frame'] + window_size
    
    # Initialize a column for the motion class
    metrics_df['motion_class'] = None
    
    # Process each unique_id separately
    for unique_id in tqdm(metrics_df['particle'].unique()):
        # Extract the subset of the data for this unique_id
        metrics_subset = metrics_df[metrics_df['particle'] == unique_id].copy()
        time_window_subset = time_windowed_df[time_windowed_df['unique_id'].str.endswith(f"_{unique_id}")].copy()
        
        if time_window_subset.empty:
            continue
        
        # Find the relevant time windows for the frames in this unique_id
        for _, window_row in time_window_subset.iterrows():
            start_frame = window_row['start_frame']
            end_frame = window_row['end_frame']
            motion_class = window_row['motion_class']
            
            # Assign the motion class to all frames within the window
            condition = (metrics_subset['frame'] >= start_frame) & (metrics_subset['frame'] < end_frame)
            metrics_df.loc[metrics_subset[condition].index, 'motion_class'] = motion_class
    
    # If any frames are still not assigned, assign the closest motion class
    undefined_rows = metrics_df['motion_class'].isnull()
    if undefined_rows.any():
        metrics_df.loc[undefined_rows, 'motion_class'] = metrics_df.loc[undefined_rows].apply(
            lambda row: time_window_subset.iloc[
                (time_window_subset['start_frame'] - row['frame']).abs().argmin()
            ]['motion_class'], axis=1
        )
    
    return metrics_df