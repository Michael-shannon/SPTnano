import os
import scipy.io
import pandas as pd
import numpy as np
from IPython.display import Markdown, display
from tqdm.notebook import tqdm

# import os
# import numpy as np
import tifffile as tiff
# import tifffile as tiff
from tifffile import TiffWriter
from nd2reader import ND2Reader
import config
import re

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
    df['particleint'] = df['particle'].astype(int)
    df['unique_id'] = df['file_id'].astype(str) + '_' + df['particleint'].astype(str)

    return df

# def add_unique_id(df):
#     df['file_id'] = pd.Categorical(df['filename']).codes
#     df['unique_id'] = df['file_id'].astype(str) + '_' + df['particle'].astype(str)
    
#     return df

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




def extract_median_dark_frame(dark_frame_directory):
    """
    Extract the median dark frame from a directory containing either ND2 or TIFF files.
    """
    # Find ND2 or TIFF files in the dark frame directory
    dark_frame_files = [f for f in os.listdir(dark_frame_directory) if f.lower().endswith(('nd2', 'tif', 'tiff'))]
    if not dark_frame_files:
        raise FileNotFoundError("No ND2 or TIFF files found in the dark frame directory.")
    
    dark_frame_path = os.path.join(dark_frame_directory, dark_frame_files[0])

    # Read the dark frame stack
    if dark_frame_path.lower().endswith('nd2'):
        with ND2Reader(dark_frame_path) as nd2_dark:
            dark_frames = [np.array(frame) for frame in nd2_dark]
    elif dark_frame_path.lower().endswith(('tif', 'tiff')):
        with tiff.TiffFile(dark_frame_path) as tif:
            dark_frames = [page.asarray() for page in tif.pages]

    # Calculate the median dark frame
    median_dark_frame = np.median(np.array(dark_frames), axis=0)

    return median_dark_frame

def apply_dark_frame_correction(input_directory, dark_frame_directory, output_directory):
    """
    Apply dark frame correction to all ND2 or TIFF files in the input directory
    using a median dark frame extracted from the dark_frame_directory.
    Saves the dark-corrected images to the output directory.
    """
    # Extract the median dark frame
    median_dark_frame = extract_median_dark_frame(dark_frame_directory)

    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Process each ND2 or TIFF file in the input directory
    for file in os.listdir(input_directory):
        if file.lower().endswith(('nd2', 'tif', 'tiff')):
            input_filepath = os.path.join(input_directory, file)
            print(f"Processing: {input_filepath}")

            # Read the image stack
            if file.lower().endswith('nd2'):
                with ND2Reader(input_filepath) as nd2_file:
                    frames = [np.array(frame) for frame in nd2_file]
            elif file.lower().endswith(('tif', 'tiff')):
                with tiff.TiffFile(input_filepath) as tif:
                    frames = [page.asarray() for page in tif.pages]

            # Subtract the median dark frame
            corrected_stack = np.array(frames) - median_dark_frame
            corrected_stack[corrected_stack < 0] = 0  # Ensure no negative values

            # Save the dark-corrected stack
            output_filepath = os.path.join(output_directory, os.path.splitext(file)[0] + '_dark_corrected.tif')
            tiff.imwrite(output_filepath, corrected_stack, photometric='minisblack')
            print(f"Saved dark-corrected file to: {output_filepath}")


def create_cut_directory(master_folder_path):
    # Define the new master directory with '_cut' suffix
    new_master_folder_path = master_folder_path + '_cut'
    if not os.path.exists(new_master_folder_path):
        os.makedirs(new_master_folder_path)

    # Define paths for 'data' folder in both master and new master directory
    data_folder_path = os.path.join(master_folder_path, 'data')
    new_data_folder_path = os.path.join(new_master_folder_path, 'data')

    # Ensure the new 'data' folder exists
    os.makedirs(new_data_folder_path, exist_ok=True)

    # Loop through all condition folders
    condition_folders = [f for f in os.listdir(data_folder_path) if f.startswith('Condition_')]
    for condition in condition_folders:
        condition_folder_path = os.path.join(data_folder_path, condition)
        new_condition_folder_path = os.path.join(new_data_folder_path, condition)

        # Ensure the new condition folder exists
        os.makedirs(new_condition_folder_path, exist_ok=True)

        # Loop through all TIFF stacks in this condition folder
        tiff_files = [f for f in os.listdir(condition_folder_path) if f.lower().endswith(('.tif', '.tiff'))]
        for tiff_file in tqdm(tiff_files, desc=f"Processing {condition}"):
            tiff_path = os.path.join(condition_folder_path, tiff_file)
            new_tiff_path = os.path.join(new_condition_folder_path, tiff_file)

            # Open the TIFF stack and remove every second frame
            with tiff.TiffFile(tiff_path) as stack:
                frames = stack.asarray()
                # Select every other frame
                cut_frames = frames[::2]

            # Save the reduced stack to the new directory
            with TiffWriter(new_tiff_path, bigtiff=True) as tif_writer:
                for frame in cut_frames:
                    tif_writer.write(frame, photometric='minisblack')

            print(f"Finished processing {tiff_file} in {condition}")




# def clean_and_split_tracks(df, segment_length_threshold=0.3, remove_short_tracks=True, min_track_length=None):
#     """
#     Clean and split tracks based on segment length thresholds. Tracks with segments exceeding the threshold
#     are split, erroneous segments are removed, and resulting subtracks are reassigned unique IDs.
    
#     Parameters:
#     - df: pd.DataFrame, input dataframe containing particle tracking data with `x_um`, `y_um`, `unique_id`, and `time_s`.
#     - segment_length_threshold: float, the maximum allowable segment length (in micrometers).
#     - remove_short_tracks: bool, whether to remove tracks shorter than the minimum track length after splitting.
#     - min_track_length: int, the minimum number of frames for a track to be considered valid after splitting.
#                         Defaults to the size of the time window if not provided.
    
#     Returns:
#     - cleaned_df: pd.DataFrame, the cleaned and split dataframe.
#     - report: dict, a summary report detailing the modifications made to the dataset.
#     """
#     # Ensure `min_track_length` is specified
#     if min_track_length is None:
#         raise ValueError("Please specify `min_track_length` to remove short tracks.")

#     original_track_count = df['unique_id'].nunique()
#     original_row_count = len(df)

#     # Calculate segment lengths
#     df = df.sort_values(by=['unique_id', 'frame'])
#     df[['x_um_prev', 'y_um_prev']] = df.groupby('unique_id')[['x_um', 'y_um']].shift(1)
#     df['segment_len_um'] = np.sqrt(
#         (df['x_um'] - df['x_um_prev'])**2 + 
#         (df['y_um'] - df['y_um_prev'])**2
#     )
#     df['segment_len_um'] = df['segment_len_um'].fillna(0)

#     # Initialize variables for splitting
#     new_tracks = []
#     new_unique_id = 0
#     split_track_count = 0
#     deleted_segments_count = 0

#     # Process each track individually
#     for unique_id, track_data in df.groupby('unique_id'):
#         current_track = []
#         previous_valid = None

#         for i, row in track_data.iterrows():
#             segment_length = row['segment_len_um']
            
#             if segment_length > segment_length_threshold:
#                 # Count deleted rows
#                 deleted_segments_count += 1

#                 # Save current valid track if exists
#                 if current_track:
#                     split_track_count += 1
#                     track_df = pd.DataFrame(current_track)
#                     track_df['unique_id'] = new_unique_id
#                     new_tracks.append(track_df)
#                     new_unique_id += 1
#                     current_track = []
#                 previous_valid = False  # Mark the current segment as invalid
#             else:
#                 if previous_valid is False:
#                     # Start a new track after invalid segments
#                     split_track_count += 1
#                     track_df = pd.DataFrame(current_track)
#                     track_df['unique_id'] = new_unique_id
#                     new_tracks.append(track_df)
#                     new_unique_id += 1
#                     current_track = []
#                 current_track.append(row)
#                 previous_valid = True

#         # Save leftover valid track if exists
#         if current_track:
#             track_df = pd.DataFrame(current_track)
#             track_df['unique_id'] = new_unique_id
#             new_tracks.append(track_df)
#             new_unique_id += 1

#     # Combine all new tracks
#     if new_tracks:
#         cleaned_df = pd.concat(new_tracks).reset_index(drop=True)

#         # Optionally remove short tracks
#         if remove_short_tracks:
#             cleaned_df['track_length'] = cleaned_df.groupby('unique_id')['unique_id'].transform('size')
#             cleaned_df = cleaned_df[cleaned_df['track_length'] >= min_track_length]
#             cleaned_df.drop(columns=['track_length'], inplace=True)
#     else:
#         cleaned_df = pd.DataFrame(columns=df.columns)  # Empty dataframe if no valid tracks remain

#     # Generate report
#     final_track_count = cleaned_df['unique_id'].nunique()
#     final_row_count = len(cleaned_df)

#     report = {
#         'original_track_count': original_track_count,
#         'final_track_count': final_track_count,
#         'tracks_split': split_track_count,
#         'deleted_segments': deleted_segments_count,
#         'original_row_count': original_row_count,
#         'final_row_count': final_row_count,
#     }

#     return cleaned_df, report


# def clean_and_split_tracks(df, segment_length_threshold=0.3, remove_short_tracks=True, min_track_length=None):
#     """
#     Clean and split tracks based on segment length thresholds. Tracks with segments exceeding the threshold
#     are split, erroneous segments are removed, and resulting subtracks are reassigned unique IDs.

#     Parameters:
#     - df: pd.DataFrame, input dataframe containing particle tracking data with `x_um`, `y_um`, `unique_id`, and `time_s`.
#     - segment_length_threshold: float, the maximum allowable segment length (in micrometers).
#     - remove_short_tracks: bool, whether to remove tracks shorter than the minimum track length after splitting.
#     - min_track_length: int, the minimum number of frames for a track to be considered valid after splitting.
#                         Defaults to the size of the time window if not provided.

#     Returns:
#     - cleaned_df: pd.DataFrame, the cleaned and split dataframe.
#     - report: dict, a summary report detailing the modifications made to the dataset.
#     """
#     if min_track_length is None:
#         raise ValueError("Please specify `min_track_length` to remove short tracks.")

#     original_track_count = df['unique_id'].nunique()
#     original_row_count = len(df)

#     # Calculate segment lengths
#     df = df.sort_values(by=['unique_id', 'frame'])
#     df[['x_um_prev', 'y_um_prev']] = df.groupby('unique_id')[['x_um', 'y_um']].shift(1)
#     df['segment_len_um'] = np.sqrt(
#         (df['x_um'] - df['x_um_prev'])**2 + 
#         (df['y_um'] - df['y_um_prev'])**2
#     )
#     df['segment_len_um'] = df['segment_len_um'].fillna(0)

#     # Assign `file_id` for consistent unique_id generation
#     df['file_id'] = pd.Categorical(df['filename']).codes

#     # Initialize variables for splitting
#     new_tracks = []
#     split_track_count = 0
#     deleted_segments_count = 0
#     next_particle_id = 0  # Used to assign new particle IDs within the same file

#     # Process each track individually
#     for unique_id, track_data in df.groupby('unique_id'):
#         current_track = []
#         previous_valid = None

#         for i, row in track_data.iterrows():
#             segment_length = row['segment_len_um']

#             if segment_length > segment_length_threshold:
#                 # Count deleted rows
#                 deleted_segments_count += 1

#                 # Save the current valid track if it exists
#                 if current_track:
#                     split_track_count += 1
#                     track_df = pd.DataFrame(current_track)
#                     track_df['particle'] = next_particle_id
#                     track_df['file_id'] = track_data['file_id'].iloc[0]  # Carry over `file_id`
#                     track_df['unique_id'] = track_df['file_id'].astype(str) + '_' + track_df['particle'].astype(str)
#                     new_tracks.append(track_df)
#                     next_particle_id += 1
#                     current_track = []
#                 previous_valid = False  # Mark the current segment as invalid
#             else:
#                 if previous_valid is False:
#                     # Start a new track after invalid segments
#                     split_track_count += 1
#                     track_df = pd.DataFrame(current_track)
#                     track_df['particle'] = next_particle_id
#                     track_df['file_id'] = track_data['file_id'].iloc[0]  # Carry over `file_id`
#                     track_df['unique_id'] = track_df['file_id'].astype(str) + '_' + track_df['particle'].astype(str) + '.0'
#                     new_tracks.append(track_df)
#                     next_particle_id += 1
#                     current_track = []
#                 current_track.append(row)
#                 previous_valid = True

#         # Save any leftover valid track
#         if current_track:
#             track_df = pd.DataFrame(current_track)
#             track_df['particle'] = next_particle_id
#             track_df['file_id'] = track_data['file_id'].iloc[0]  # Carry over `file_id`
#             track_df['unique_id'] = track_df['file_id'].astype(str) + '_' + track_df['particle'].astype(str) + '.0'
#             new_tracks.append(track_df)
#             next_particle_id += 1

#     # Combine all new tracks
#     if new_tracks:
#         cleaned_df = pd.concat(new_tracks).reset_index(drop=True)

#         # Optionally remove short tracks
#         if remove_short_tracks:
#             cleaned_df['track_length'] = cleaned_df.groupby('unique_id')['unique_id'].transform('size')
#             cleaned_df = cleaned_df[cleaned_df['track_length'] >= min_track_length]
#             cleaned_df.drop(columns=['track_length'], inplace=True)
#     else:
#         cleaned_df = pd.DataFrame(columns=df.columns)  # Empty dataframe if no valid tracks remain

#     # Generate report
#     final_track_count = cleaned_df['unique_id'].nunique()
#     final_row_count = len(cleaned_df)

#     report = {
#         'original_track_count': original_track_count,
#         'final_track_count': final_track_count,
#         'tracks_split': split_track_count,
#         'deleted_segments': deleted_segments_count,
#         'original_row_count': original_row_count,
#         'final_row_count': final_row_count,
#     }

#     return cleaned_df, report



# def clean_and_split_tracks(df, segment_length_threshold=0.3, remove_short_tracks=True, min_track_length=None):
#     """
#     Clean and split tracks based on segment length thresholds. Tracks with segments exceeding the threshold
#     are split, erroneous segments are removed, and resulting subtracks are reassigned unique IDs.

#     Parameters:
#     - df: pd.DataFrame, input dataframe containing particle tracking data with `x_um`, `y_um`, `unique_id`, and `time_s`.
#     - segment_length_threshold: float, the maximum allowable segment length (in micrometers).
#     - remove_short_tracks: bool, whether to remove tracks shorter than the minimum track length after splitting.
#     - min_track_length: int, the minimum number of frames for a track to be considered valid after splitting.

#     Returns:
#     - cleaned_df: pd.DataFrame, the cleaned and split dataframe.
#     - removed_unique_ids: list, `unique_id`s that were removed due to short length or other reasons.
#     - report: dict, a summary report detailing the modifications made to the dataset.
#     """
#     if min_track_length is None:
#         raise ValueError("Please specify `min_track_length` to remove short tracks.")

#     original_track_count = df['unique_id'].nunique()
#     original_row_count = len(df)

#     # Calculate segment lengths
#     df = df.sort_values(by=['unique_id', 'frame'])
#     df[['x_um_prev', 'y_um_prev']] = df.groupby('unique_id')[['x_um', 'y_um']].shift(1)
#     df['segment_len_um'] = np.sqrt(
#         (df['x_um'] - df['x_um_prev'])**2 + 
#         (df['y_um'] - df['y_um_prev'])**2
#     )
#     df['segment_len_um'] = df['segment_len_um'].fillna(0)

#     # Assign `file_id` for consistent unique_id generation
#     df['file_id'] = pd.Categorical(df['filename']).codes

#     # Initialize variables for splitting
#     new_tracks = []
#     split_track_count = 0
#     deleted_segments_count = 0
#     next_particle_id = 0
#     removed_unique_ids = set()  # To track removed IDs

#     # Process each track individually
#     for unique_id, track_data in df.groupby('unique_id'):
#         current_track = []
#         previous_valid = None

#         for i, row in track_data.iterrows():
#             segment_length = row['segment_len_um']

#             if segment_length > segment_length_threshold:
#                 # Count deleted rows
#                 deleted_segments_count += 1

#                 # Save the current valid track if it exists
#                 if current_track:
#                     split_track_count += 1
#                     track_df = pd.DataFrame(current_track)
#                     track_df['particle'] = next_particle_id
#                     track_df['file_id'] = track_data['file_id'].iloc[0]
#                     track_df['unique_id'] = track_df['file_id'].astype(str) + '_' + track_df['particle'].astype(str)
#                     new_tracks.append(track_df)
#                     next_particle_id += 1
#                     current_track = []
#                 previous_valid = False
#             else:
#                 if previous_valid is False:
#                     split_track_count += 1
#                     track_df = pd.DataFrame(current_track)
#                     track_df['particle'] = next_particle_id
#                     track_df['file_id'] = track_data['file_id'].iloc[0]
#                     track_df['unique_id'] = track_df['file_id'].astype(str) + '_' + track_df['particle'].astype(str)
#                     new_tracks.append(track_df)
#                     next_particle_id += 1
#                     current_track = []
#                 current_track.append(row)
#                 previous_valid = True

#         # Save any leftover valid track
#         if current_track:
#             track_df = pd.DataFrame(current_track)
#             track_df['particle'] = next_particle_id
#             track_df['file_id'] = track_data['file_id'].iloc[0]
#             track_df['unique_id'] = track_df['file_id'].astype(str) + '_' + track_df['particle'].astype(str)
#             new_tracks.append(track_df)
#             next_particle_id += 1

#     # Combine all new tracks
#     if new_tracks:
#         cleaned_df = pd.concat(new_tracks).reset_index(drop=True)

#         # Optionally remove short tracks
#         if remove_short_tracks:
#             cleaned_df['track_length'] = cleaned_df.groupby('unique_id')['unique_id'].transform('size')
#             removed_unique_ids = set(cleaned_df.loc[cleaned_df['track_length'] < min_track_length, 'unique_id'])
#             cleaned_df = cleaned_df[cleaned_df['track_length'] >= min_track_length]
#             cleaned_df.drop(columns=['track_length'], inplace=True)
#     else:
#         cleaned_df = pd.DataFrame(columns=df.columns)  # Empty dataframe if no valid tracks remain

#     # Generate report
#     final_track_count = cleaned_df['unique_id'].nunique()
#     final_row_count = len(cleaned_df)

#     report = {
#         'original_track_count': original_track_count,
#         'final_track_count': final_track_count,
#         'tracks_split': split_track_count,
#         'deleted_segments': deleted_segments_count,
#         'original_row_count': original_row_count,
#         'final_row_count': final_row_count,
#     }

#     return cleaned_df, removed_unique_ids, report


def clean_and_split_tracks(
    df, 
    segment_length_threshold=0.3, 
    remove_short_tracks=True, 
    min_track_length_seconds=None, 
    time_between_frames=0.1
):
    """
    Clean and split tracks based on segment length thresholds. Tracks with segments exceeding the threshold
    are split, erroneous segments are removed, and resulting subtracks are reassigned unique IDs.

    Parameters:
    - df: pd.DataFrame, input dataframe containing particle tracking data with `x_um`, `y_um`, `unique_id`, and `time_s`.
    - segment_length_threshold: float, the maximum allowable segment length (in micrometers).
    - remove_short_tracks: bool, whether to remove tracks shorter than the minimum track length after splitting.
    - min_track_length_seconds: float, the minimum track duration in seconds to be retained.
    - time_between_frames: float, time duration between frames in seconds (from config).

    Returns:
    - cleaned_df: pd.DataFrame, the cleaned and split dataframe.
    - removed_unique_ids: list, `unique_id`s that were removed due to short length or other reasons.
    - report: dict, a summary report detailing the modifications made to the dataset.
    """
    if min_track_length_seconds is not None:
        min_track_length_frames = int(min_track_length_seconds / time_between_frames)
    else:
        raise ValueError("Please specify `min_track_length_seconds`.")

    original_track_count = df['unique_id'].nunique()
    original_row_count = len(df)

    # Calculate segment lengths
    df = df.sort_values(by=['unique_id', 'frame'])
    df[['x_um_prev', 'y_um_prev']] = df.groupby('unique_id')[['x_um', 'y_um']].shift(1)
    df['segment_len_um'] = np.sqrt(
        (df['x_um'] - df['x_um_prev'])**2 + 
        (df['y_um'] - df['y_um_prev'])**2
    )
    df['segment_len_um'] = df['segment_len_um'].fillna(0)

    # Assign `file_id` for consistent unique_id generation
    df['file_id'] = pd.Categorical(df['filename']).codes

    # Initialize variables for splitting
    new_tracks = []
    split_track_count = 0
    deleted_segments_count = 0
    next_particle_id = 0.0
    removed_unique_ids = set()  # To track removed IDs

    # Process each track individually
    for unique_id, track_data in df.groupby('unique_id'):
        current_track = []
        previous_valid = None

        for i, row in track_data.iterrows():
            segment_length = row['segment_len_um']

            if segment_length > segment_length_threshold:
                # Count deleted rows
                deleted_segments_count += 1

                # Save the current valid track if it exists
                if current_track:
                    split_track_count += 1
                    track_df = pd.DataFrame(current_track)
                    track_df['particle'] = next_particle_id
                    track_df['file_id'] = track_data['file_id'].iloc[0]
                    # track_df['unique_id'] = track_df['file_id'].astype(str) + '_' + track_df['particle'].astype(str)
                    track_df['unique_id'] = track_df['file_id'].astype(str) + '_' + track_df['particle'].map('{:.1f}'.format)

                    # print(f'Current track particle ID: {next_particle_id}')  # Print particle ID for each track]}')
                    # print(f'Current track file ID: {track_data["file_id"].iloc[0]}')  # Print file ID for each track
                    # print(f'Current track df:')
                    # print(track_df.particle.unique())
                    new_tracks.append(track_df)
                    next_particle_id += 1.0
                    current_track = []
                previous_valid = False
            else:
                if previous_valid is False:
                    split_track_count += 1
                    track_df = pd.DataFrame(current_track)
                    track_df['particle'] = next_particle_id
                    track_df['file_id'] = track_data['file_id'].iloc[0]
                    # track_df['unique_id'] = track_df['file_id'].astype(str) + '_' + track_df['particle'].astype(str)
                    track_df['unique_id'] = track_df['file_id'].astype(str) + '_' + track_df['particle'].map('{:.1f}'.format)


                    # print(f'Current track particle ID: {next_particle_id}')  # Print particle ID for each track]}')
                    # print(f'Current track file ID: {track_data["file_id"].iloc[0]}')  # Print file ID for each track
                    # print(f'Current track df:')
                    # print(track_df.particle.unique())

                    new_tracks.append(track_df)
                    next_particle_id += 1.0
                    current_track = []
                current_track.append(row)
                previous_valid = True

        # Save any leftover valid track
        if current_track:
            track_df = pd.DataFrame(current_track)
            track_df['particle'] = next_particle_id
            track_df['file_id'] = track_data['file_id'].iloc[0]
            # track_df['unique_id'] = track_df['file_id'].astype(str) + '_' + track_df['particle'].astype(str)
            track_df['unique_id'] = track_df['file_id'].astype(str) + '_' + track_df['particle'].map('{:.1f}'.format)

            # print(f'Current track particle ID: {next_particle_id}')  # Print particle ID for each track]}')
            # print(f'Current track file ID: {track_data["file_id"].iloc[0]}')  # Print file ID for each track
            # print(f'Current track df:')
            # print(track_df.particle.unique())
            new_tracks.append(track_df)
            next_particle_id += 1.0

    # Combine all new tracks
    if new_tracks:
        cleaned_df = pd.concat(new_tracks).reset_index(drop=True)

        # Optionally remove short tracks
        if remove_short_tracks:
            cleaned_df['track_length'] = cleaned_df.groupby('unique_id')['unique_id'].transform('size')
            removed_unique_ids = set(cleaned_df.loc[cleaned_df['track_length'] < min_track_length_frames, 'unique_id'])
            cleaned_df = cleaned_df[cleaned_df['track_length'] >= min_track_length_frames]
            cleaned_df.drop(columns=['track_length'], inplace=True)
    else:
        cleaned_df = pd.DataFrame(columns=df.columns)  # Empty dataframe if no valid tracks remain

    # Generate report
    final_track_count = cleaned_df['unique_id'].nunique()
    final_row_count = len(cleaned_df)

    report = {
        'original_track_count': original_track_count,
        'final_track_count': final_track_count,
        'tracks_split': split_track_count,
        'deleted_segments': deleted_segments_count,
        'original_row_count': original_row_count,
        'final_row_count': final_row_count,
    }

    return cleaned_df, removed_unique_ids, report

def pathfixer(
    df, 
    segment_length_threshold=0.3, 
    remove_short_tracks=True, 
    min_track_length_seconds=1.0, 
    time_between_frames=0.1
):
    """
    Cleans and splits tracks based on segment length thresholds. Tracks with segments exceeding 
    the threshold are split, erroneous segments are removed, and resulting subtracks are 
    reassigned unique IDs.

    Parameters:
    - df: pd.DataFrame, input dataframe containing particle tracking data with `x_um`, `y_um`, `unique_id`, and `time_s`.
    - segment_length_threshold: float, the maximum allowable segment length (in micrometers).
    - remove_short_tracks: bool, whether to remove tracks shorter than the minimum track length after splitting.
    - min_track_length_seconds: float, the minimum track duration in seconds to be retained.
    - time_between_frames: float, time duration between frames in seconds.

    Returns:
    - cleaned_df: pd.DataFrame, the cleaned and split dataframe.
    - removed_unique_ids: list, `unique_id`s that were removed due to short length or other reasons.
    - report: dict, a summary report detailing the modifications made to the dataset.
    """
    if min_track_length_seconds is None:
        raise ValueError("Please specify `min_track_length_seconds`.")

    min_track_length_frames = int(min_track_length_seconds / time_between_frames)

    original_track_count = df['unique_id'].nunique()
    original_row_count = len(df)

    # Sort by track ID and frame order
    df = df.sort_values(by=['unique_id', 'frame']).reset_index(drop=True)

    # Compute segment lengths
    df[['x_um_prev', 'y_um_prev']] = df.groupby('unique_id')[['x_um', 'y_um']].shift(1)
    df['segment_len_um'] = np.sqrt(
        (df['x_um'] - df['x_um_prev'])**2 + 
        (df['y_um'] - df['y_um_prev'])**2
    ).fillna(0)

    # Assign `file_id` for consistent unique_id generation
    df['file_id'] = pd.Categorical(df['filename']).codes

    # Initialize variables
    new_tracks = []
    split_track_count = 0
    deleted_segments_count = 0
    removed_unique_ids = set()  
    next_particle_id = 0  # Using an integer instead of float to avoid formatting issues

    # Process each track individually
    for unique_id, track_data in df.groupby('unique_id'):
        current_track = []
        
        for _, row in track_data.iterrows():
            if row['segment_len_um'] > segment_length_threshold:
                deleted_segments_count += 1
                if current_track:
                    split_track_count += 1
                    track_df = pd.DataFrame(current_track)
                    track_df['particle'] = next_particle_id
                    track_df['file_id'] = track_data['file_id'].iloc[0]
                    track_df['unique_id'] = track_df['file_id'].astype(str) + '_' + str(next_particle_id)
                    # df['unique_id'] = df['file_id'].astype(str) + '_' + df['particle'].astype(str)
                    new_tracks.append(track_df)
                    next_particle_id += 1
                    current_track = []
            else:
                current_track.append(row)

        # Save any remaining valid track
        if current_track:
            track_df = pd.DataFrame(current_track)
            track_df['particle'] = next_particle_id
            track_df['file_id'] = track_data['file_id'].iloc[0]
            track_df['unique_id'] = track_df['file_id'].astype(str) + '_' + str(next_particle_id)
            new_tracks.append(track_df)
            next_particle_id += 1

    # Combine all new tracks
    cleaned_df = pd.concat(new_tracks, ignore_index=True) if new_tracks else pd.DataFrame(columns=df.columns)

    # Remove short tracks if enabled
    if remove_short_tracks and not cleaned_df.empty:
        cleaned_df['track_length'] = cleaned_df.groupby('unique_id')['unique_id'].transform('size')
        removed_unique_ids = set(cleaned_df.loc[cleaned_df['track_length'] < min_track_length_frames, 'unique_id'])
        cleaned_df = cleaned_df[cleaned_df['track_length'] >= min_track_length_frames]
        cleaned_df.drop(columns=['track_length'], inplace=True)

    # Generate report
    final_track_count = cleaned_df['unique_id'].nunique()
    final_row_count = len(cleaned_df)

    report = {
        'original_track_count': original_track_count,
        'final_track_count': final_track_count,
        'tracks_split': split_track_count,
        'deleted_segments': deleted_segments_count,
        'original_row_count': original_row_count,
        'final_row_count': final_row_count,
    }

    return cleaned_df, removed_unique_ids, report


# def add_motion_class(metrics_df, time_windowed_df, time_window=config.TIME_WINDOW):
#     """
#     Add a 'motion_class' column to metrics_df based on time_windowed_df with debugging.
    
#     Parameters:
#     - metrics_df: DataFrame containing the particle tracking data.
#     - time_windowed_df: DataFrame containing the motion class for time windows.
#     - time_window: Number of frames in a time window (default 6).
    
#     Returns:
#     - Updated metrics_df with 'motion_class' column added.
#     """
#     # Initialize the motion_class column with empty values
#     metrics_df['motion_class'] = ''

#     # Temporary dictionary to store motion classes per frame
#     frame_classes = {}

#     # Debugging counters
#     total_windows = 0
#     matched_windows = 0
#     unmatched_rows = set(metrics_df.index)  # Track unmatched rows

#     # Tolerance for floating-point comparison
#     tolerance = 1e-5

#     # Iterate over each unique ID in the time_windowed_df
#     for unique_id in time_windowed_df['unique_id'].unique():
#         # Filter data for the current unique_id
#         tw_df = time_windowed_df[time_windowed_df['unique_id'] == unique_id]
#         m_df = metrics_df[metrics_df['unique_id'] == unique_id]  # Match particle ID

#         if m_df.empty:
#             print(f"Warning: No matching data found in metrics_df for unique_id '{unique_id}'")
#             continue

#         for _, tw_row in tw_df.iterrows():
#             total_windows += 1
#             x_start, y_start = tw_row['x_um_start'], tw_row['y_um_start']
#             time_win_class = tw_row['motion_class']

#             # Locate the start index in metrics_df (using tolerance for floating-point comparison)
#             start_idx = m_df[
#                 (abs(m_df['x_um'] - x_start) < tolerance) & 
#                 (abs(m_df['y_um'] - y_start) < tolerance)
#             ].index

#             if not start_idx.empty:
#                 matched_windows += 1
#                 # Calculate frame range for the time window
#                 start_idx = start_idx[0]
#                 frame_range = range(start_idx, start_idx + time_window)

#                 # Assign the motion_class for this time window
#                 for frame_idx in frame_range:
#                     if frame_idx < len(metrics_df):
#                         if frame_idx in unmatched_rows:
#                             unmatched_rows.remove(frame_idx)
#                         if frame_idx not in frame_classes:
#                             frame_classes[frame_idx] = []
#                         frame_classes[frame_idx].append(time_win_class)
#             else:
#                 print(f"Warning: No matching start point found for x_start={x_start}, y_start={y_start} in unique_id '{unique_id}'")

#     # Apply majority rule to resolve overlaps
#     for frame_idx, classes in frame_classes.items():
#         # Assign the most frequent motion_class for this frame
#         metrics_df.at[frame_idx, 'motion_class'] = max(set(classes), key=classes.count)

#     # Assign 'unlabeled' to unmatched rows
#     for idx in unmatched_rows:
#         metrics_df.at[idx, 'motion_class'] = 'unlabeled'

#     # Debugging output
#     print(f"Total time windows processed: {total_windows}")
#     print(f"Matched time windows: {matched_windows}")
#     print(f"Unmatched rows assigned 'unlabeled': {len(unmatched_rows)}")

#     return metrics_df

# def add_motion_class(metrics_df, time_windowed_df, time_window=config.TIME_WINDOW):
#     """
#     Add a 'motion_class' column to metrics_df based on time_windowed_df with debugging.
    
#     Parameters:
#     - metrics_df: DataFrame containing the particle tracking data.
#     - time_windowed_df: DataFrame containing the motion class for time windows.
#     - time_window: Number of frames in a time window (default 6).
    
#     Returns:
#     - Updated metrics_df with 'motion_class' column added.
#     """
#     # Initialize the motion_class column with empty values
#     metrics_df['motion_class'] = ''

#     # Temporary dictionary to store motion classes per frame
#     frame_classes = {}

#     # Debugging counters
#     total_windows = 0
#     matched_windows = 0
#     unmatched_rows = set(metrics_df.index)  # Track unmatched rows

#     # Tolerance for floating-point comparison
#     tolerance = 1e-5

#     # Iterate over each unique ID in the time_windowed_df
#     for unique_id in time_windowed_df['unique_id'].unique():
#         # Filter data for the current unique_id
#         tw_df = time_windowed_df[time_windowed_df['unique_id'] == unique_id]
#         m_df = metrics_df[metrics_df['unique_id'] == unique_id]  # Match particle ID

#         if m_df.empty:
#             print(f"Warning: No matching data found in metrics_df for unique_id '{unique_id}'")
#             continue

#         for _, tw_row in tw_df.iterrows():
#             total_windows += 1
#             x_start, y_start = tw_row['x_um_start'], tw_row['y_um_start']
#             time_win_class = tw_row['motion_class']

#             # Locate the start index in metrics_df (using tolerance for floating-point comparison)
#             start_idx = m_df[
#                 (abs(m_df['x_um'] - x_start) < tolerance) & 
#                 (abs(m_df['y_um'] - y_start) < tolerance)
#             ].index

#             if not start_idx.empty:
#                 matched_windows += 1
#                 # Calculate frame range for the time window
#                 start_idx = start_idx[0]
#                 frame_range = range(start_idx, start_idx + time_window)
#                 # print(len(frame_range))

#                 # Assign the motion_class for this time window
#                 for frame_idx in frame_range:
#                     if frame_idx < len(metrics_df):
#                         if frame_idx in unmatched_rows:
#                             unmatched_rows.remove(frame_idx)
#                         if frame_idx not in frame_classes:
#                             frame_classes[frame_idx] = []
#                         frame_classes[frame_idx].append(time_win_class)
#             else:
#                 print(f"Warning: No matching start point found for x_start={x_start}, y_start={y_start} in unique_id '{unique_id}'")

#     # Apply majority rule to resolve overlaps
#     for frame_idx, classes in frame_classes.items():
#         # Assign the most frequent motion_class for this frame
#         metrics_df.at[frame_idx, 'motion_class'] = max(set(classes), key=classes.count)

#     # Assign 'unlabeled' to unmatched rows
#     for idx in unmatched_rows:
#         metrics_df.at[idx, 'motion_class'] = 'unlabeled'

#     # Debugging output
#     print(f"Total time windows processed: {total_windows}")
#     print(f"Matched time windows: {matched_windows}")
#     print(f"Unmatched rows assigned 'unlabeled': {len(unmatched_rows)}")

#     return metrics_df
# ########################### backup version #############################
# def add_motion_class(metrics_df, time_windowed_df, time_window=config.TIME_WINDOW): #updated to cope with the new functions on making sure there are no crazy Ds.
#     """
#     Add a 'motion_class' column to metrics_df based on time_windowed_df with debugging.
    
#     This updated function takes into account that some time_windowed_df unique_ids may have 
#     been modified by the excision process (e.g. appended with "_s"). If a unique_id from time_windowed_df 
#     is not found in metrics_df, the function uses the base unique_id (obtained by removing the '_s' suffix)
#     for matching. For each time window, the motion_class is assigned over the corresponding block of frames,
#     with overlapping assignments resolved by majority rule.
    
#     Finally, to facilitate visualization, any gaps (i.e. rows that remain empty) are filled
#     via a forward fill and backward fill so that every row receives a motion_class.
    
#     Parameters:
#       - metrics_df: DataFrame containing the instantaneous particle tracking data.
#                     Must include columns ['unique_id', 'x_um', 'y_um'].
#       - time_windowed_df: DataFrame containing time-windowed motion class information.
#                           Must include columns ['unique_id', 'x_um_start', 'y_um_start', 'motion_class'].
#       - time_window: Number of frames per time window (default: config.TIME_WINDOW).
    
#     Returns:
#       Updated metrics_df with a new 'motion_class' column.
#     """


#     # Initialize the 'motion_class' column to empty strings.
#     metrics_df['motion_class'] = ''
    
#     # Dictionary to hold lists of motion_class assignments for each frame index.
#     frame_classes = {}
    
#     # Debugging counters.
#     total_windows = 0
#     matched_windows = 0
#     # Keep track of rows which did not get assigned any value.
#     unmatched_rows = set(metrics_df.index)
    
#     # Tolerance for matching the coordinates (as in the D mapping function).
#     tolerance = 1e-5
    
#     # Loop over each unique_id in time_windowed_df.
#     for uid in tqdm(time_windowed_df['unique_id'].unique(), desc="Processing tracks"):
#         # First, try to see if the uid is present in metrics_df.
#         if uid in metrics_df['unique_id'].unique():
#             m_df = metrics_df[metrics_df['unique_id'] == uid]
#         else:
#             # If not, assume it may have an excision suffix and try the base unique_id.
#             base_uid = uid.split('_s')[0]
#             m_df = metrics_df[metrics_df['unique_id'] == base_uid]
        
#         tw_df = time_windowed_df[time_windowed_df['unique_id'] == uid]
        
#         if m_df.empty:
#             print(f"Warning: No matching data found in metrics_df for unique_id '{uid}' (or its base).")
#             continue
        
#         for _, tw_row in tw_df.iterrows():
#             total_windows += 1
#             x_start, y_start = tw_row['x_um_start'], tw_row['y_um_start']
#             time_win_class = tw_row['motion_class']
            
#             # Locate the starting frame in m_df using the coordinate tolerance.
#             start_idx_series = m_df[
#                 (abs(m_df['x_um'] - x_start) < tolerance) &
#                 (abs(m_df['y_um'] - y_start) < tolerance)
#             ].index
            
#             if not start_idx_series.empty:
#                 matched_windows += 1
#                 start_idx = start_idx_series[0]
#                 # Define the frame range for the time window.
#                 frame_range = range(start_idx, start_idx + time_window)
#                 for frame_idx in frame_range:
#                     if frame_idx < len(metrics_df):
#                         unmatched_rows.discard(frame_idx)
#                         frame_classes.setdefault(frame_idx, []).append(time_win_class)
#             else:
#                 print(f"Warning: No matching start point found for uid '{uid}' at x={x_start}, y={y_start}.")
    
#     # Resolve any overlapping assignments via majority rule.
#     for frame_idx, classes in frame_classes.items():
#         metrics_df.at[frame_idx, 'motion_class'] = max(set(classes), key=classes.count)
    
#     # Assign 'unlabeled' to any rows that remain unmatched.
#     for idx in unmatched_rows:
#         metrics_df.at[idx, 'motion_class'] = 'unlabeled'
    
#     print(f"Total time windows processed: {total_windows}")
#     print(f"Matched time windows: {matched_windows}")
#     print(f"Unmatched rows assigned 'unlabeled': {len(unmatched_rows)}")
    
#     # For visualization, fill any remaining gaps (empty strings) by forward fill then backward fill.
#     # metrics_df['motion_class'] = metrics_df['motion_class'].replace('', np.nan)
#     metrics_df['motion_class'] = metrics_df['motion_class'].replace(['', 'unlabeled'], np.nan)
#     metrics_df['motion_class'] = metrics_df['motion_class'].ffill().bfill()
    
#     return metrics_df

############### BACKUP VERSION #######################


def add_motion_class(metrics_df, time_windowed_df, time_window=config.TIME_WINDOW):
    """
    Add a 'motion_class' column to metrics_df based on time_windowed_df.
    
    - Handles split track IDs with '_s#' suffix by matching base IDs.
    - Uses each track’s index list so windows never map into the wrong track.
    - Resolves overlapping windows by majority vote.
    - Marks any truly unassigned frames 'unlabeled', then forward/backward fills.
    """
    # Work on a copy and reset index so it’s a clean RangeIndex
    df = metrics_df.copy().reset_index(drop=True)
    df['motion_class'] = np.nan

    # Accumulate assignments per row
    frame_classes = {}
    total_windows = 0
    matched_windows = 0

    # Which IDs actually exist in the instant data?
    base_ids = set(df['unique_id'].unique())
    tol = 1e-5

    # 1) Assign each window
    for uid in tqdm(time_windowed_df['unique_id'].unique(), desc="Processing motion windows"):
        track_id = uid if uid in base_ids else uid.split('_s')[0]
        tw = time_windowed_df[time_windowed_df['unique_id'] == uid]
        inst = df[df['unique_id'] == track_id]
        inst_idx = inst.index.to_list()
        if not inst_idx:
            continue

        for _, row in tw.iterrows():
            total_windows += 1
            sx, sy = row['x_um_start'], row['y_um_start']
            cls     = row['motion_class']

            # find the exact start frame in this track
            hits = inst[
                (inst['x_um'].sub(sx).abs() < tol) &
                (inst['y_um'].sub(sy).abs() < tol)
            ].index.tolist()
            if not hits:
                continue

            matched_windows += 1
            start_idx = hits[0]
            pos       = inst_idx.index(start_idx)
            # only assign within this track’s slice
            for ridx in inst_idx[pos : pos + time_window]:
                frame_classes.setdefault(ridx, []).append(cls)

    # 2) Resolve overlaps by majority vote
    for ridx, clist in frame_classes.items():
        vote = max(set(clist), key=clist.count)
        df.at[ridx, 'motion_class'] = vote

    # 3) Any rows still NaN get “unlabeled”
    unlabeled = df['motion_class'].isna()
    df.loc[unlabeled, 'motion_class'] = 'unlabeled'

    # 4) For visualization ease, treat 'unlabeled' as missing, then ffill/bfill
    df['motion_class'] = df['motion_class'].replace('unlabeled', np.nan)
    df['motion_class'] = df['motion_class'].ffill().bfill()

    print(f"Total windows:           {total_windows}")
    print(f"Matched windows:         {matched_windows}")
    print(f"Final 'unlabeled' frames: {df['motion_class'].isna().sum()} (should be 0)")

    return df


def extract_metadata(df, filename_col):
    """
    Extracts 'power' and 'cellID' from the filename column and adds them as new columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing a column with filenames.
        filename_col (str): Name of the column containing filenames.
    
    Returns:
        pd.DataFrame: Updated DataFrame with 'power' and 'cellID' columns.
    """
    def parse_filename(filename):
        match = re.search(r'power-(\d+percent)_cell-(\d+)', filename)
        if match:
            power = match.group(1)  # Extracts power (e.g., '25percent')
            cellID = int(match.group(2))  # Extracts cell number as integer
            return power, cellID
        return None, None  # If no match, return None values

    df[['power', 'cellID']] = df[filename_col].apply(lambda x: pd.Series(parse_filename(x)))

    return df


# def map_D_to_instant(metrics_df, time_windowed_df, time_window=config.TIME_WINDOW, 
#                      overlap_method='average', tolerance=1e-5):
#     """
#     Map time-windowed diffusion coefficients back to the instantaneous tracking DataFrame.
    
#     For each unique track (using 'unique_id'), the function finds the frame in metrics_df
#     corresponding to the starting point (x_um_start, y_um_start) provided in time_windowed_df.
#     It then assigns the diffusion coefficient (from 'diffusion_coefficient') over a block of frames 
#     defined by 'time_window'. If frames belong to overlapping time windows (e.g. overlap of 30 frames),
#     the final diffusion coefficient is computed according to the chosen 'overlap_method'.

#     Parameters:
#       metrics_df (pd.DataFrame): Instantaneous tracking data. Must include ['unique_id', 'x_um', 'y_um'].
#       time_windowed_df (pd.DataFrame): Time-windowed tracking data with columns 
#                                        ['unique_id', 'x_um_start', 'y_um_start', 'diffusion_coefficient'].
#       time_window (int): Number of frames that each time window covers (default is 60).
#       overlap_method (str): Method to resolve overlapping diffusion coefficient assignments:
#                             - 'average' (default): arithmetic mean of overlapping D values.
#                             - 'first': use the first assigned D value.
#                             - 'last': use the last assigned D value.
#                             - 'min': use the minimum D value.
#                             - 'max': use the maximum D value.
#                             - 'median': use the median D value.
#       tolerance (float): Tolerance for matching start coordinates (default is 1e-5).

#     Returns:
#       tuple: (updated metrics_df, report)
#              - metrics_df: The input DataFrame with a new 'diffusion_coefficient' column populated.
#              - report: A list of strings containing warnings and summary debug messages.
#     """
#     report = []  # This list will hold warnings and summary messages.
    
#     # Initialize the diffusion_coefficient column with NaN values.
#     metrics_df['diffusion_coefficient'] = np.nan
    
#     # Dictionary to hold lists of D values for each frame index.
#     frame_D = {}
    
#     # Debug counters.
#     total_windows = 0
#     matched_windows = 0
#     unmatched_rows = set(metrics_df.index)
    
#     # Wrap the outer loop with tqdm to show progress across unique tracks.
#     for unique_id in tqdm(time_windowed_df['unique_id'].unique(), desc="Processing tracks"):
#         # Extract data for the current unique_id.
#         tw_df = time_windowed_df[time_windowed_df['unique_id'] == unique_id]
#         inst_df = metrics_df[metrics_df['unique_id'] == unique_id]
        
#         if inst_df.empty:
#             report.append(f"Warning: No matching data in metrics_df for unique_id '{unique_id}'.")
#             continue
        
#         # Optionally, if you have many time windows, you can wrap the inner loop as well:
#         for _, tw_row in tqdm(tw_df.iterrows(), total=tw_df.shape[0],
#                                 desc=f"Processing windows for track {unique_id}", leave=False):
#             total_windows += 1
#             start_x = tw_row['x_um_start']
#             start_y = tw_row['y_um_start']
#             D_value = tw_row['diffusion_coefficient']
            
#             # Locate the starting frame in inst_df using tolerance for x and y.
#             start_idx_series = inst_df[
#                 (abs(inst_df['x_um'] - start_x) < tolerance) &
#                 (abs(inst_df['y_um'] - start_y) < tolerance)
#             ].index
            
#             if not start_idx_series.empty:
#                 matched_windows += 1
#                 start_idx = start_idx_series[0]
#                 # For the defined time window, assign the D_value.
#                 for frame_idx in range(start_idx, start_idx + time_window):
#                     if frame_idx < len(metrics_df):
#                         if frame_idx in unmatched_rows:
#                             unmatched_rows.remove(frame_idx)
#                         # Append this D_value; overlapping frames will have multiple values.
#                         frame_D.setdefault(frame_idx, []).append(D_value)
#             else:
#                 report.append(
#                     f"Warning: No matching start point found for unique_id '{unique_id}' at x={start_x}, y={start_y}."
#                 )
    
#     # Resolve diffusion coefficient for each frame, especially those with overlapping window assignments.
#     for frame_idx, D_values in frame_D.items():
#         if overlap_method == 'average':
#             combined_D = sum(D_values) / len(D_values)
#         elif overlap_method == 'first':
#             combined_D = D_values[0]
#         elif overlap_method == 'last':
#             combined_D = D_values[-1]
#         elif overlap_method == 'min':
#             combined_D = min(D_values)
#         elif overlap_method == 'max':
#             combined_D = max(D_values)
#         elif overlap_method == 'median':
#             combined_D = np.median(D_values)
#         else:
#             report.append(f"Warning: Unknown overlap_method '{overlap_method}'. Defaulting to average.")
#             combined_D = sum(D_values) / len(D_values)
            
#         metrics_df.at[frame_idx, 'diffusion_coefficient'] = combined_D
    
#     # Append summary debug information to the report.
#     report.append(f"Total time windows processed: {total_windows}")
#     report.append(f"Matched time windows: {matched_windows}")
#     report.append(f"Frames left unmatched (remain as NaN): {len(unmatched_rows)}")
    
#     return metrics_df, report

# def map_D_to_instant(metrics_df, time_windowed_df, time_window=config.TIME_WINDOW, # NEW ONE
#                      overlap_method='average', tolerance=1e-5):
#     """
#     Map time-windowed diffusion coefficients back to the instantaneous tracking DataFrame.
    
#     For each unique track (using 'unique_id'), the function finds the frame in metrics_df
#     corresponding to the starting point (x_um_start, y_um_start) provided in time_windowed_df.
#     It then assigns the diffusion coefficient (from 'diffusion_coefficient') over a block of frames 
#     defined by 'time_window'. If frames belong to overlapping time windows, the final diffusion coefficient 
#     is computed according to the chosen 'overlap_method'.
    
#     This version has been updated to handle cases where excision has been used to split a track 
#     (e.g. unique_id '1' becomes '1_s'). In such cases it falls back to matching the base unique_id.
#     Additionally, after mapping the D values, a new column 'D_fit_quality' is added to metrics_df. 
#     Rows with a NaN diffusion coefficient are flagged as 'bad', and others as 'good'.
    
#     Parameters:
#       metrics_df (pd.DataFrame): Instantaneous tracking data. Must include ['unique_id', 'x_um', 'y_um'].
#       time_windowed_df (pd.DataFrame): Time-windowed tracking data with columns 
#                                        ['unique_id', 'x_um_start', 'y_um_start', 'diffusion_coefficient'].
#       time_window (int): Number of frames that each time window covers (default is 60).
#       overlap_method (str): Method to resolve overlapping diffusion coefficient assignments:
#                             - 'average' (default): arithmetic mean of overlapping D values.
#                             - 'first': use the first assigned D value.
#                             - 'last': use the last assigned D value.
#                             - 'min': use the minimum D value.
#                             - 'max': use the maximum D value.
#                             - 'median': use the median D value.
#       tolerance (float): Tolerance for matching start coordinates (default is 1e-5).
    
#     Returns:
#       tuple: (updated metrics_df, report)
#              - metrics_df: The input DataFrame with a new 'diffusion_coefficient' column and 'D_fit_quality' column.
#              - report: A list of strings containing warnings and summary debug messages.
#     """
#     import numpy as np
#     from tqdm import tqdm
    
#     report = []  # Holds warnings and summary messages.
    
#     # Initialize the diffusion_coefficient column with NaN.
#     metrics_df['diffusion_coefficient'] = np.nan
    
#     # Dictionary to hold lists of D values by frame index.
#     frame_D = {}
    
#     # Debug counters.
#     total_windows = 0
#     matched_windows = 0
#     unmatched_rows = set(metrics_df.index)
    
#     # Get all unique base ids from metrics_df (assumes they are the original unique_ids).
#     base_ids = set(metrics_df['unique_id'].unique())
    
#     # Iterate over unique_ids in time_windowed_df.
#     for uid in tqdm(time_windowed_df['unique_id'].unique(), desc="Processing tracks"):
#         # If the unique_id has been modified (e.g. ends with '_s'), try to get the base id.
#         if uid not in base_ids:
#             # For instance, if uid = '1_s', then base_uid will be '1'.
#             base_uid = uid.split('_s')[0]
#             if base_uid in base_ids:
#                 uid_to_use = base_uid
#             else:
#                 uid_to_use = uid  # Fallback: no base found.
#         else:
#             uid_to_use = uid

#         # Extract data for the current track from both DataFrames.
#         tw_df = time_windowed_df[time_windowed_df['unique_id'] == uid]
#         inst_df = metrics_df[metrics_df['unique_id'] == uid_to_use]
        
#         if inst_df.empty:
#             report.append(f"Warning: No matching data in metrics_df for unique_id '{uid_to_use}' (from time_windowed_df uid '{uid}').")
#             continue
        
#         for _, tw_row in tqdm(tw_df.iterrows(), total=tw_df.shape[0],
#                                 desc=f"Processing windows for track {uid}", leave=False):
#             total_windows += 1
#             start_x = tw_row['x_um_start']
#             start_y = tw_row['y_um_start']
#             D_value = tw_row['diffusion_coefficient']
            
#             # Locate the starting frame using tolerance.
#             start_idx_series = inst_df[
#                 (abs(inst_df['x_um'] - start_x) < tolerance) &
#                 (abs(inst_df['y_um'] - start_y) < tolerance)
#             ].index
            
#             if not start_idx_series.empty:
#                 matched_windows += 1
#                 start_idx = start_idx_series[0]
#                 # For the time window, assign the D_value.
#                 for frame_idx in range(start_idx, start_idx + time_window):
#                     if frame_idx < len(metrics_df):
#                         if frame_idx in unmatched_rows:
#                             unmatched_rows.remove(frame_idx)
#                         frame_D.setdefault(frame_idx, []).append(D_value)
#             else:
#                 report.append(
#                     f"Warning: No matching start point found for unique_id '{uid_to_use}' at x={start_x}, y={start_y}."
#                 )
    
#     # Resolve diffusion coefficient for each frame.
#     for frame_idx, D_values in frame_D.items():
#         if overlap_method == 'average':
#             combined_D = sum(D_values) / len(D_values)
#         elif overlap_method == 'first':
#             combined_D = D_values[0]
#         elif overlap_method == 'last':
#             combined_D = D_values[-1]
#         elif overlap_method == 'min':
#             combined_D = min(D_values)
#         elif overlap_method == 'max':
#             combined_D = max(D_values)
#         elif overlap_method == 'median':
#             combined_D = np.median(D_values)
#         else:
#             report.append(f"Warning: Unknown overlap_method '{overlap_method}'. Defaulting to average.")
#             combined_D = sum(D_values) / len(D_values)
            
#         metrics_df.at[frame_idx, 'diffusion_coefficient'] = combined_D
    
#     # Flag rows based on whether a diffusion coefficient was assigned.
#     metrics_df['D_fit_quality'] = np.where(metrics_df['diffusion_coefficient'].isna(), 'bad', 'good')
    
#     report.append(f"Total time windows processed: {total_windows}")
#     report.append(f"Matched time windows: {matched_windows}")
#     report.append(f"Frames left unmatched (remain as NaN): {len(unmatched_rows)}")
    
#     return metrics_df, report

# def map_D_to_instant(metrics_df, time_windowed_df, time_window=config.TIME_WINDOW, 
#                      overlap_method='average', tolerance=1e-5):
#     """
#     Map time-windowed diffusion coefficients back to the instantaneous tracking DataFrame.
    
#     For each unique track (using 'unique_id'), the function finds the frame in metrics_df
#     corresponding to the starting point (x_um_start, y_um_start) provided in time_windowed_df.
#     It then assigns the diffusion coefficient (from 'diffusion_coefficient') over a block of frames 
#     defined by 'time_window'. If frames belong to overlapping time windows, the final diffusion coefficient 
#     is computed according to the chosen 'overlap_method'.
    
#     This version handles cases where excision has been used to split a track (e.g. unique_id '1' becomes '1_s')
#     by matching the base unique_id. After mapping, a new column 'D_fit_quality' is added:
#       - Rows with an assigned D are flagged "good"
#       - Rows with no assigned D (i.e. NaN) are flagged "bad"
    
#     Finally, instead of linear interpolation, any remaining NaN gaps in 'diffusion_coefficient'
#     are filled by repeating the neighbor values as follows:
#       - For each contiguous block of NaNs, determine the "previous" good value (just before the block)
#         and the "next" good value (just after the block).
#       - Split the gap (if more than one frame) at the midpoint: assign the first half of the gap
#         the previous good value, and the second half the next good value.
#       - For a gap of one frame, assign the previous good value if available; otherwise, assign the next.
    
#     Parameters:
#       metrics_df (pd.DataFrame): Instantaneous tracking data. Must include ['unique_id', 'x_um', 'y_um'].
#       time_windowed_df (pd.DataFrame): Time-windowed tracking data with columns 
#                                        ['unique_id', 'x_um_start', 'y_um_start', 'diffusion_coefficient'].
#       time_window (int): Number of frames that each time window covers (default is 60).
#       overlap_method (str): Method to resolve overlapping diffusion coefficient assignments:
#                             - 'average' (default): arithmetic mean of overlapping D values.
#                             - 'first': use the first assigned D value.
#                             - 'last': use the last assigned D value.
#                             - 'min': use the minimum D value.
#                             - 'max': use the maximum D value.
#                             - 'median': use the median D value.
#       tolerance (float): Tolerance for matching start coordinates (default is 1e-5).
    
#     Returns:
#       tuple: (updated metrics_df, report)
#              - metrics_df: The input DataFrame with new columns 'diffusion_coefficient' (with gaps filled)
#                            and 'D_fit_quality' (indicating the original mapping quality).
#              - report: A list of strings containing warnings and summary messages.
#     """
#     import numpy as np
#     from tqdm import tqdm
#     import pandas as pd

#     report = []  # Will hold warnings and summary messages.
    
#     # Initialize diffusion_coefficient column as NaN.
#     metrics_df['diffusion_coefficient'] = np.nan
    
#     # Dictionary to hold lists of D values for each frame index.
#     frame_D = {}
    
#     # Debug counters.
#     total_windows = 0
#     matched_windows = 0
#     unmatched_rows = set(metrics_df.index)
    
#     # Get the set of base unique_ids present in metrics_df.
#     base_ids = set(metrics_df['unique_id'].unique())
    
#     # Loop over unique_ids in time_windowed_df.
#     for uid in tqdm(time_windowed_df['unique_id'].unique(), desc="Processing tracks"):
#         # If uid has an excision suffix (e.g. '1_s'), remove it and use the base unique_id.
#         if uid not in base_ids:
#             base_uid = uid.split('_s')[0]
#             uid_to_use = base_uid if base_uid in base_ids else uid
#         else:
#             uid_to_use = uid
        
#         # Select corresponding data from time_windowed_df and metrics_df.
#         tw_df = time_windowed_df[time_windowed_df['unique_id'] == uid]
#         inst_df = metrics_df[metrics_df['unique_id'] == uid_to_use]
        
#         if inst_df.empty:
#             report.append(f"Warning: No matching data in metrics_df for unique_id '{uid_to_use}' (from time_windowed_df uid '{uid}').")
#             continue
        
#         for _, tw_row in tqdm(tw_df.iterrows(), total=tw_df.shape[0],
#                                 desc=f"Processing windows for track {uid}", leave=False):
#             total_windows += 1
#             start_x = tw_row['x_um_start']
#             start_y = tw_row['y_um_start']
#             D_value = tw_row['diffusion_coefficient']
            
#             # Locate the starting frame in inst_df using x and y tolerances.
#             start_idx_series = inst_df[
#                 (abs(inst_df['x_um'] - start_x) < tolerance) &
#                 (abs(inst_df['y_um'] - start_y) < tolerance)
#             ].index
            
#             if not start_idx_series.empty:
#                 matched_windows += 1
#                 start_idx = start_idx_series[0]
#                 # For frames from start_idx to start_idx+time_window, assign D_value.
#                 for frame_idx in range(start_idx, start_idx + time_window):
#                     if frame_idx < len(metrics_df):
#                         unmatched_rows.discard(frame_idx)
#                         frame_D.setdefault(frame_idx, []).append(D_value)
#             else:
#                 report.append(
#                     f"Warning: No matching start point found for unique_id '{uid_to_use}' at x={start_x}, y={start_y}."
#                 )
    
#     # For each frame index that has at least one D value assigned,
#     # resolve the diffusion coefficient using the chosen overlap_method.
#     for frame_idx, D_values in frame_D.items():
#         if overlap_method == 'average':
#             combined_D = sum(D_values) / len(D_values)
#         elif overlap_method == 'first':
#             combined_D = D_values[0]
#         elif overlap_method == 'last':
#             combined_D = D_values[-1]
#         elif overlap_method == 'min':
#             combined_D = min(D_values)
#         elif overlap_method == 'max':
#             combined_D = max(D_values)
#         elif overlap_method == 'median':
#             combined_D = np.median(D_values)
#         else:
#             report.append(f"Warning: Unknown overlap_method '{overlap_method}'. Defaulting to average.")
#             combined_D = sum(D_values) / len(D_values)
            
#         metrics_df.at[frame_idx, 'diffusion_coefficient'] = combined_D
    
#     # Create the D_fit_quality flag: "good" if D is assigned, "bad" if NaN.
#     metrics_df['D_fit_quality'] = np.where(metrics_df['diffusion_coefficient'].isna(), 'bad', 'good')
    
#     # --- Custom "repetition" fill for missing diffusion_coefficient values ---
#     # Rather than interpolate linearly, we fill contiguous NaN blocks by repeating the last
#     # good value for the first half of the gap and the next good value for the second half.
#     D_series = metrics_df['diffusion_coefficient']
#     nan_mask = D_series.isna()
#     nan_indices = np.where(nan_mask)[0]
    
#     if len(nan_indices) > 0:
#         groups = []
#         group_start = nan_indices[0]
#         prev_idx = nan_indices[0]
#         for ix in nan_indices[1:]:
#             if ix == prev_idx + 1:
#                 prev_idx = ix
#             else:
#                 groups.append((group_start, prev_idx))
#                 group_start = ix
#                 prev_idx = ix
#         groups.append((group_start, prev_idx))
    
#         for (start, end) in groups:
#             gap_length = end - start + 1
#             # Get previous good value (if exists).
#             if start > 0:
#                 prev_good = metrics_df.iloc[start - 1]['diffusion_coefficient']
#             else:
#                 prev_good = None
#             # Get next good value (if exists).
#             if end < len(metrics_df) - 1:
#                 next_good = metrics_df.iloc[end + 1]['diffusion_coefficient']
#             else:
#                 next_good = None
                
#             # Decide on fill values.
#             if gap_length == 1:
#                 # For a single missing frame, use previous good if available; else next good.
#                 fill_val = prev_good if prev_good is not None else next_good
#                 metrics_df.iat[start, metrics_df.columns.get_loc('diffusion_coefficient')] = fill_val
#             else:
#                 half = gap_length // 2
#                 # First half of gap.
#                 if prev_good is not None:
#                     metrics_df.loc[start:(start+half-1), 'diffusion_coefficient'] = prev_good
#                 else:
#                     # If no previous good, use next good.
#                     metrics_df.loc[start:(start+half-1), 'diffusion_coefficient'] = next_good
#                 # Second half of gap.
#                 if next_good is not None:
#                     metrics_df.loc[(start+half):end, 'diffusion_coefficient'] = next_good
#                 else:
#                     # If no next good, use previous good.
#                     metrics_df.loc[(start+half):end, 'diffusion_coefficient'] = prev_good
    
#     report.append(f"Total time windows processed: {total_windows}")
#     report.append(f"Matched time windows: {matched_windows}")
#     report.append(f"Frames left unmatched before gap filling: {len(unmatched_rows)}")
    
#     return metrics_df, report


def map_D_to_instant(metrics_df, time_windowed_df,
                     time_window=config.TIME_WINDOW,
                     overlap_method='average',
                     tolerance=1e-5):
    """
    Map time-windowed diffusion coefficients back to the instantaneous tracking DataFrame.
    1) Reset index to ensure no NaN labels.
    2) Assign each window’s D strictly within its track’s own indices.
    3) Resolve overlaps by the chosen method.
    4) Extend the first/last window’s D to the track ends.
    5) Half‑gap fill any remaining NaNs.
    """
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    # 1) Reset the index so we have a clean RangeIndex
    metrics_df = metrics_df.copy().reset_index(drop=True)
    metrics_df['diffusion_coefficient'] = np.nan

    frame_D = {}         # row → list of D values
    total_windows = 0
    matched_windows = 0

    # All track IDs present
    base_ids = set(metrics_df['unique_id'].unique())

    # 2) Assign
    for uid in tqdm(time_windowed_df['unique_id'].unique(), desc="Mapping windows"):
        tid = uid if uid in base_ids else uid.split('_s')[0]
        tw = time_windowed_df[time_windowed_df['unique_id'] == uid]
        inst = metrics_df[metrics_df['unique_id'] == tid]
        inst_idx = inst.index.to_list()
        if not inst_idx:
            continue

        for _, w in tw.iterrows():
            total_windows += 1
            sx, sy = w['x_um_start'], w['y_um_start']
            Dval    = w['diffusion_coefficient']

            # locate the exact start row
            hits = inst[
                (inst['x_um'].sub(sx).abs() < tolerance) &
                (inst['y_um'].sub(sy).abs() < tolerance)
            ].index.tolist()
            if not hits:
                continue

            matched_windows += 1
            pos = inst_idx.index(hits[0])
            for ridx in inst_idx[pos:pos + time_window]:
                frame_D.setdefault(ridx, []).append(Dval)

    # 3) Resolve overlaps
    for ridx, Dlist in frame_D.items():
        if overlap_method == 'first':
            val = Dlist[0]
        elif overlap_method == 'last':
            val = Dlist[-1]
        elif overlap_method == 'min':
            val = min(Dlist)
        elif overlap_method == 'max':
            val = max(Dlist)
        elif overlap_method == 'median':
            val = np.median(Dlist)
        else:
            val = sum(Dlist) / len(Dlist)
        metrics_df.at[ridx, 'diffusion_coefficient'] = val

    # 4) Extend first/last window D to track ends
    for tid in base_ids:
        idxs = metrics_df.index[metrics_df['unique_id'] == tid].tolist()
        sub  = metrics_df.loc[idxs, 'diffusion_coefficient']
        non  = sub.dropna()
        if non.empty:
            continue
        first_i, last_i = non.index.min(), non.index.max()
        first_val = metrics_df.at[first_i, 'diffusion_coefficient']
        last_val  = metrics_df.at[last_i, 'diffusion_coefficient']
        for i in idxs:
            if pd.isna(metrics_df.at[i, 'diffusion_coefficient']):
                metrics_df.at[i, 'diffusion_coefficient'] = (
                    first_val if i < first_i else last_val
                )

    # 5) Half‑gap fill any remaining NaNs
    Dcol = metrics_df['diffusion_coefficient']
    mask = Dcol.isna().to_numpy()
    if mask.any():
        nan_idx = np.where(mask)[0]
        groups = []
        s = nan_idx[0]; p = s
        for i in nan_idx[1:]:
            if i == p + 1:
                p = i
            else:
                groups.append((s, p)); s = i; p = i
        groups.append((s, p))

        for (start, end) in groups:
            length = end - start + 1
            prev_val = metrics_df.iloc[start-1]['diffusion_coefficient'] if start>0 else None
            next_val = metrics_df.iloc[end+1]['diffusion_coefficient'] if end<len(Dcol)-1 else None

            if length == 1:
                fill = prev_val if prev_val is not None else next_val
                metrics_df.iat[start, metrics_df.columns.get_loc('diffusion_coefficient')] = fill
            else:
                half = length // 2
                # first half
                if prev_val is not None:
                    metrics_df.loc[start:start+half-1, 'diffusion_coefficient'] = prev_val
                else:
                    metrics_df.loc[start:start+half-1, 'diffusion_coefficient'] = next_val
                # second half
                if next_val is not None:
                    metrics_df.loc[start+half:end, 'diffusion_coefficient'] = next_val
                else:
                    metrics_df.loc[start+half:end, 'diffusion_coefficient'] = prev_val

    # quality flag & report
    metrics_df['D_fit_quality'] = np.where(
        metrics_df['diffusion_coefficient'].isna(), 'bad', 'good'
    )
    unmatched = int(metrics_df['diffusion_coefficient'].isna().sum())
    report = [
        f"Total windows:           {total_windows}",
        f"Matched windows:         {matched_windows}",
        f"Frames still unmatched:  {unmatched}",
    ]
    return metrics_df, report





