import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import polars as pl
import scipy.io

# import os
# import numpy as np
import tifffile as tiff
from IPython.display import Markdown, display
from nd2reader import ND2Reader

# import tifffile as tiff
from tifffile import TiffWriter
from tqdm.notebook import tqdm

from . import config


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


def generate_file_tree(startpath):
    tree = []
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = " " * 4 * (level)
        tree.append(f"{indent}- {os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
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
    df["file_id"] = pd.Categorical(df["filename"]).codes
    df["particleint"] = df["particle"].astype(int)
    df["unique_id"] = df["file_id"].astype(str) + "_" + df["particleint"].astype(str)

    return df


# def add_unique_id(df):
#     df['file_id'] = pd.Categorical(df['filename']).codes
#     df['unique_id'] = df['file_id'].astype(str) + '_' + df['particle'].astype(str)

#     return df


def add_microns_and_secs(df, pixelsize_microns, time_between_frames):
    """Adds columns to the DataFrame with positions in microns and time in seconds"""
    # space transformations
    df["x_um"] = df["x"] * pixelsize_microns
    df["y_um"] = df["y"] * pixelsize_microns

    df["frame_zeroed"] = df.groupby("particle")["frame"].transform(
        lambda x: x - x.iloc[0]
    )
    df["time_s"] = df["frame"] * time_between_frames
    df["time_s_zeroed"] = df.groupby("particle")["time_s"].transform(
        lambda x: x - x.iloc[0]
    )
    return df


# Define a function to read .mat files and convert them to DataFrames
def read_mat_file(file_path):
    mat = scipy.io.loadmat(file_path)
    tr_data = mat["data"]["tr"][0, 0]

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
    df = pd.DataFrame(
        combined_data,
        columns=[
            "x",
            "y",
            "frame",
            "particle",
            "column5",
            "column6",
            "column7",
            "column8",
        ],
    )

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
    """
    Removes tracks that are shorter than 'min_time' by finding the max duration of each time_s_zeroed column and filtering on that
    Works across exposure times, because it works on converted seconds, not frames

    """
    # Calculate the duration of each track by grouping by 'particle' and using the 'time_s' column
    track_durations = df.groupby("unique_id")["time_s_zeroed"].max()
    # Identify particles with tracks longer than 0.2 seconds
    valid_particles = track_durations[track_durations >= min_time].index
    # Filter the dataframe to include only valid particles
    filtered_df = df[df["unique_id"].isin(valid_particles)]

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

    Parameters
    - df (pd.DataFrame): The input DataFrame.
    - condition (Union[str, int]): The condition to filter by, or an index to choose from unique conditions.
    - unique_id (Optional[int]): The unique ID of the particle to filter by. If None, a random particle is chosen.
    - location (Union[str, int], optional): The location to filter by, or an index to choose from unique locations. Default is None.

    Returns
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
        particle_ids = df["unique_id"].unique()
        unique_id = np.random.choice(particle_ids)

    # Filter by the chosen particle
    single_particle_df = df[df["unique_id"] == unique_id]

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
    large_jump_particles = df[df["segment_len_um"] > threshold]["unique_id"].unique()

    # Filter out particles with large jumps
    df_filtered = df[~df["unique_id"].isin(large_jump_particles)].copy()
    # df_filtered.drop(columns=['x_um_prev', 'y_um_prev', 'segment_len_um'], inplace=True)
    return df_filtered


def filter_high_speeds(metrics_df, speed_threshold):
    """
    Filter based on speed instead - can be relevant if you have different exposure times and different times between frames
    """
    # Identify unique_ids with any high speeds
    high_speed_particles = metrics_df[metrics_df["speed_um_s"] > speed_threshold][
        "unique_id"
    ].unique()

    # Filter out particles with high speeds
    metrics_df_filtered = metrics_df[
        ~metrics_df["unique_id"].isin(high_speed_particles)
    ].copy()
    return metrics_df_filtered


def sample_dataframe(df, percent_samples, seed=42):
    """
    Sample a percentage of unique_ids from each condition.
    
    Parameters:
    -----------
    df : pd.DataFrame or pl.DataFrame
        Input dataframe with 'condition' and 'unique_id' columns
    percent_samples : float
        Percentage of unique_ids to sample (0.0 to 1.0)
    seed : int, optional
        Random seed for reproducibility (default: 42)
        
    Returns:
    --------
    pd.DataFrame or pl.DataFrame
        Sampled dataframe (same type as input)
    """
    # Check if it's a polars dataframe
    try:
        import polars as pl
        is_polars = isinstance(df, pl.DataFrame)
    except ImportError:
        is_polars = False
    
    if is_polars:
        # Polars implementation
        # Get unique condition-unique_id pairs
        cond_id = df.select(['condition', 'unique_id']).unique()
        
        # Sample unique_ids per condition
        sampled_cond_ids = (
            cond_id
            .group_by('condition')
            .agg([
                pl.col('unique_id').count().alias('group_count')
            ])
            .with_columns([
                (pl.col('group_count') * percent_samples).ceil().cast(pl.Int64).clip(1).alias('n_sample')
            ])
            .join(cond_id, on="condition")
            .group_by('condition')
            .agg([
                pl.col('unique_id').sample(n=pl.first('n_sample'), seed=seed).alias('unique_id')
            ])
            .explode('unique_id')
        )
        
        # Filter original df to only sampled unique_ids per condition
        sampled_df = df.join(
            sampled_cond_ids.select(['condition', 'unique_id']),
            on=['condition', 'unique_id'],
            how='inner'
        )
        
        # Report
        print(f"Original dataframe contains {len(df)} rows")
        print(f"Sampled dataframe contains {len(sampled_df)} rows")
        
    else:
        # Pandas implementation (original logic)
        np.random.seed(seed)
        
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
            n_samples = max(1, int(len(unique_ids) * percent_samples))
            sampled_ids = np.random.choice(
                unique_ids, n_samples, replace=False
            )
            # filter the dataframe to only include those unique ids
            sampled_df = condition_df[condition_df.unique_id.isin(sampled_ids)]
            # append the sampled dataframe to the list
            sampled_dfs.append(sampled_df)
        # concatenate the list of dataframes into a single dataframe
        sampled_df = pd.concat(sampled_dfs, ignore_index=True)
        # report on the sampling and the original dataframe
        print(f"Original dataframe contains {len(df)} rows")
        print(f"Sampled dataframe contains {len(sampled_df)} rows")
    
    return sampled_df


def generalized_filter(
    df_in, filter_col, low=None, high=None, condition=None, location=None
):
    """
    Filters the input DataFrame based on a range of values in filter_col and additional conditions.

    Parameters
    - df_in (pd.DataFrame): The input DataFrame to filter.
    - filter_col (str): The column to filter based on a range of values.
    - low (float, optional): The lower bound of the filter range (inclusive). Default is None.
    - high (float, optional): The upper bound of the filter range (inclusive). Default is None.
    - condition (Union[str, int], optional): The specific condition to filter by, or an index to choose from unique conditions. Default is None.
    - location (Union[str, int], optional): The specific location to filter by, or an index to choose from unique locations. Default is None.

    Returns
    - pd.DataFrame: The filtered DataFrame.

    """
    # Start with the input DataFrame
    df_filtered = df_in.copy()
    print(f"Available filters: {df_filtered.columns}")

    # Handle the filtering logic for low and high values
    if low is None and high is not None:
        df_filtered = df_filtered[df_filtered[filter_col] <= high]
    elif low is not None and high is None:
        df_filtered = df_filtered[df_filtered[filter_col] >= low]
    elif low is not None and high is not None:
        df_filtered = df_filtered[
            (df_filtered[filter_col] >= low) & (df_filtered[filter_col] <= high)
        ]

    # Handle the condition input (can be a string or an index)
    if condition is not None:
        unique_conditions = df_filtered["condition"].unique()
        if isinstance(condition, int):
            if condition >= len(unique_conditions):
                raise ValueError(f"Condition index {condition} is out of bounds.")
            condition = unique_conditions[condition]
        df_filtered = df_filtered[df_filtered["condition"] == condition]

    # Handle the location input (can be a string or an index)
    if location is not None:
        location_col = _get_location_column(df_filtered)
        unique_locations = df_filtered[location_col].unique()
        if isinstance(location, int):
            if location >= len(unique_locations):
                raise ValueError(
                    f"Location index {location} is out of bounds - if you think it should be there, check whether you have it in the condition you filtered on"
                )
            location = unique_locations[location]
        df_filtered = df_filtered[df_filtered[location_col] == location]

    unique_ids = df_filtered["unique_id"].unique()

    return df_filtered, unique_ids


def optimized_assign_motion_class_by_unique_id(
    metrics_df, time_windowed_df, window_size=24, overlap=12
):
    timewindowedids = time_windowed_df.unique_id.unique()
    metricsdfids = metrics_df.unique_id.unique()
    # filter the metrics_df to only include the unique_ids that are in the time_windowed_df
    metrics_df = metrics_df[metrics_df.unique_id.isin(timewindowedids)]
    # get the number of metrics_df unique_ids
    metricsdfidsafter = metrics_df.unique_id.unique()
    percentids = len(metricsdfidsafter) / len(metricsdfids) * 100

    # print the percentage of unique_ids that remain
    print(
        f"Percentage of unique_ids that remain after removing intersection: {percentids}%"
    )

    # Calculate the start and end frames for each time window in the time_windowed_df
    time_windowed_df["start_frame"] = time_windowed_df["time_window"] * (
        window_size - overlap
    )
    time_windowed_df["end_frame"] = time_windowed_df["start_frame"] + window_size

    # Initialize a column for the motion class
    metrics_df["motion_class"] = None

    # Process each unique_id separately
    for unique_id in tqdm(metrics_df["particle"].unique()):
        # Extract the subset of the data for this unique_id
        metrics_subset = metrics_df[metrics_df["particle"] == unique_id].copy()
        time_window_subset = time_windowed_df[
            time_windowed_df["unique_id"].str.endswith(f"_{unique_id}")
        ].copy()

        if time_window_subset.empty:
            continue

        # Find the relevant time windows for the frames in this unique_id
        for _, window_row in time_window_subset.iterrows():
            start_frame = window_row["start_frame"]
            end_frame = window_row["end_frame"]
            motion_class = window_row["motion_class"]

            # Assign the motion class to all frames within the window
            condition = (metrics_subset["frame"] >= start_frame) & (
                metrics_subset["frame"] < end_frame
            )
            metrics_df.loc[metrics_subset[condition].index, "motion_class"] = (
                motion_class
            )

    # If any frames are still not assigned, assign the closest motion class
    undefined_rows = metrics_df["motion_class"].isnull()
    if undefined_rows.any():
        metrics_df.loc[undefined_rows, "motion_class"] = metrics_df.loc[
            undefined_rows
        ].apply(
            lambda row: time_window_subset.iloc[
                (time_window_subset["start_frame"] - row["frame"]).abs().argmin()
            ]["motion_class"],
            axis=1,
        )

    return metrics_df


def extract_median_dark_frame(dark_frame_directory):
    """
    Extract the median dark frame from a directory containing either ND2 or TIFF files.
    """
    # Find ND2 or TIFF files in the dark frame directory
    dark_frame_files = [
        f
        for f in os.listdir(dark_frame_directory)
        if f.lower().endswith(("nd2", "tif", "tiff"))
    ]
    if not dark_frame_files:
        raise FileNotFoundError(
            "No ND2 or TIFF files found in the dark frame directory."
        )

    dark_frame_path = os.path.join(dark_frame_directory, dark_frame_files[0])

    # Read the dark frame stack
    if dark_frame_path.lower().endswith("nd2"):
        with ND2Reader(dark_frame_path) as nd2_dark:
            dark_frames = [np.array(frame) for frame in nd2_dark]
    elif dark_frame_path.lower().endswith(("tif", "tiff")):
        with tiff.TiffFile(dark_frame_path) as tif:
            dark_frames = [page.asarray() for page in tif.pages]

    # Calculate the median dark frame
    median_dark_frame = np.median(np.array(dark_frames), axis=0)

    return median_dark_frame


def apply_dark_frame_correction(
    input_directory, dark_frame_directory, output_directory
):
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
        if file.lower().endswith(("nd2", "tif", "tiff")):
            input_filepath = os.path.join(input_directory, file)
            print(f"Processing: {input_filepath}")

            # Read the image stack
            if file.lower().endswith("nd2"):
                with ND2Reader(input_filepath) as nd2_file:
                    frames = [np.array(frame) for frame in nd2_file]
            elif file.lower().endswith(("tif", "tiff")):
                with tiff.TiffFile(input_filepath) as tif:
                    frames = [page.asarray() for page in tif.pages]

            # Subtract the median dark frame
            corrected_stack = np.array(frames) - median_dark_frame
            corrected_stack[corrected_stack < 0] = 0  # Ensure no negative values

            # Save the dark-corrected stack
            output_filepath = os.path.join(
                output_directory, os.path.splitext(file)[0] + "_dark_corrected.tif"
            )
            tiff.imwrite(output_filepath, corrected_stack, photometric="minisblack")
            print(f"Saved dark-corrected file to: {output_filepath}")


def create_cut_directory(master_folder_path):
    # Define the new master directory with '_cut' suffix
    new_master_folder_path = master_folder_path + "_cut"
    if not os.path.exists(new_master_folder_path):
        os.makedirs(new_master_folder_path)

    # Define paths for 'data' folder in both master and new master directory
    data_folder_path = os.path.join(master_folder_path, "data")
    new_data_folder_path = os.path.join(new_master_folder_path, "data")

    # Ensure the new 'data' folder exists
    os.makedirs(new_data_folder_path, exist_ok=True)

    # Loop through all condition folders
    condition_folders = [
        f for f in os.listdir(data_folder_path) if f.startswith("Condition_")
    ]
    for condition in condition_folders:
        condition_folder_path = os.path.join(data_folder_path, condition)
        new_condition_folder_path = os.path.join(new_data_folder_path, condition)

        # Ensure the new condition folder exists
        os.makedirs(new_condition_folder_path, exist_ok=True)

        # Loop through all TIFF stacks in this condition folder
        tiff_files = [
            f
            for f in os.listdir(condition_folder_path)
            if f.lower().endswith((".tif", ".tiff"))
        ]
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
                    tif_writer.write(frame, photometric="minisblack")

            print(f"Finished processing {tiff_file} in {condition}")


def clean_and_split_tracks(
    df,
    segment_length_threshold=0.3,
    remove_short_tracks=True,
    min_track_length_seconds=None,
    time_between_frames=0.1,
):
    """
    Clean and split tracks based on segment length thresholds. Tracks with segments exceeding the threshold
    are split, erroneous segments are removed, and resulting subtracks are reassigned unique IDs.

    Parameters
    - df: pd.DataFrame, input dataframe containing particle tracking data with `x_um`, `y_um`, `unique_id`, and `time_s`.
    - segment_length_threshold: float, the maximum allowable segment length (in micrometers).
    - remove_short_tracks: bool, whether to remove tracks shorter than the minimum track length after splitting.
    - min_track_length_seconds: float, the minimum track duration in seconds to be retained.
    - time_between_frames: float, time duration between frames in seconds (from config).

    Returns
    - cleaned_df: pd.DataFrame, the cleaned and split dataframe.
    - removed_unique_ids: list, `unique_id`s that were removed due to short length or other reasons.
    - report: dict, a summary report detailing the modifications made to the dataset.

    """
    if min_track_length_seconds is not None:
        min_track_length_frames = int(min_track_length_seconds / time_between_frames)
    else:
        raise ValueError("Please specify `min_track_length_seconds`.")

    original_track_count = df["unique_id"].nunique()
    original_row_count = len(df)

    # Calculate segment lengths
    df = df.sort_values(by=["unique_id", "frame"])
    df[["x_um_prev", "y_um_prev"]] = df.groupby("unique_id")[["x_um", "y_um"]].shift(1)
    df["segment_len_um"] = np.sqrt(
        (df["x_um"] - df["x_um_prev"]) ** 2 + (df["y_um"] - df["y_um_prev"]) ** 2
    )
    df["segment_len_um"] = df["segment_len_um"].fillna(0)

    # Assign `file_id` for consistent unique_id generation
    df["file_id"] = pd.Categorical(df["filename"]).codes

    # Initialize variables for splitting
    new_tracks = []
    split_track_count = 0
    deleted_segments_count = 0
    next_particle_id = 0.0
    removed_unique_ids = set()  # To track removed IDs

    # Process each track individually
    for unique_id, track_data in df.groupby("unique_id"):
        current_track = []
        previous_valid = None

        for i, row in track_data.iterrows():
            segment_length = row["segment_len_um"]

            if segment_length > segment_length_threshold:
                # Count deleted rows
                deleted_segments_count += 1

                # Save the current valid track if it exists
                if current_track:
                    split_track_count += 1
                    track_df = pd.DataFrame(current_track)
                    track_df["particle"] = next_particle_id
                    track_df["file_id"] = track_data["file_id"].iloc[0]
                    # track_df['unique_id'] = track_df['file_id'].astype(str) + '_' + track_df['particle'].astype(str)
                    track_df["unique_id"] = (
                        track_df["file_id"].astype(str)
                        + "_"
                        + track_df["particle"].map("{:.1f}".format)
                    )

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
                    track_df["particle"] = next_particle_id
                    track_df["file_id"] = track_data["file_id"].iloc[0]
                    # track_df['unique_id'] = track_df['file_id'].astype(str) + '_' + track_df['particle'].astype(str)
                    track_df["unique_id"] = (
                        track_df["file_id"].astype(str)
                        + "_"
                        + track_df["particle"].map("{:.1f}".format)
                    )

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
            track_df["particle"] = next_particle_id
            track_df["file_id"] = track_data["file_id"].iloc[0]
            # track_df['unique_id'] = track_df['file_id'].astype(str) + '_' + track_df['particle'].astype(str)
            track_df["unique_id"] = (
                track_df["file_id"].astype(str)
                + "_"
                + track_df["particle"].map("{:.1f}".format)
            )

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
            cleaned_df["track_length"] = cleaned_df.groupby("unique_id")[
                "unique_id"
            ].transform("size")
            removed_unique_ids = set(
                cleaned_df.loc[
                    cleaned_df["track_length"] < min_track_length_frames, "unique_id"
                ]
            )
            cleaned_df = cleaned_df[
                cleaned_df["track_length"] >= min_track_length_frames
            ]
            cleaned_df.drop(columns=["track_length"], inplace=True)
    else:
        cleaned_df = pd.DataFrame(
            columns=df.columns
        )  # Empty dataframe if no valid tracks remain

    # Generate report
    final_track_count = cleaned_df["unique_id"].nunique()
    final_row_count = len(cleaned_df)

    report = {
        "original_track_count": original_track_count,
        "final_track_count": final_track_count,
        "tracks_split": split_track_count,
        "deleted_segments": deleted_segments_count,
        "original_row_count": original_row_count,
        "final_row_count": final_row_count,
    }

    return cleaned_df, removed_unique_ids, report


def pathfixer(
    df,
    segment_length_threshold=0.3,
    remove_short_tracks=True,
    min_track_length_seconds=1.0,
    time_between_frames=0.1,
):
    """
    Cleans and splits tracks based on segment length thresholds. Tracks with segments exceeding
    the threshold are split, erroneous segments are removed, and resulting subtracks are
    reassigned unique IDs.

    Parameters
    - df: pd.DataFrame, input dataframe containing particle tracking data with `x_um`, `y_um`, `unique_id`, and `time_s`.
    - segment_length_threshold: float, the maximum allowable segment length (in micrometers).
    - remove_short_tracks: bool, whether to remove tracks shorter than the minimum track length after splitting.
    - min_track_length_seconds: float, the minimum track duration in seconds to be retained.
    - time_between_frames: float, time duration between frames in seconds.

    Returns
    - cleaned_df: pd.DataFrame, the cleaned and split dataframe.
    - removed_unique_ids: list, `unique_id`s that were removed due to short length or other reasons.
    - report: dict, a summary report detailing the modifications made to the dataset.

    """
    if min_track_length_seconds is None:
        raise ValueError("Please specify `min_track_length_seconds`.")

    min_track_length_frames = int(min_track_length_seconds / time_between_frames)

    original_track_count = df["unique_id"].nunique()
    original_row_count = len(df)

    # Sort by track ID and frame order
    df = df.sort_values(by=["unique_id", "frame"]).reset_index(drop=True)

    # Compute segment lengths
    df[["x_um_prev", "y_um_prev"]] = df.groupby("unique_id")[["x_um", "y_um"]].shift(1)
    df["segment_len_um"] = np.sqrt(
        (df["x_um"] - df["x_um_prev"]) ** 2 + (df["y_um"] - df["y_um_prev"]) ** 2
    ).fillna(0)

    # Assign `file_id` for consistent unique_id generation
    df["file_id"] = pd.Categorical(df["filename"]).codes

    # Initialize variables
    new_tracks = []
    split_track_count = 0
    deleted_segments_count = 0
    removed_unique_ids = set()
    next_particle_id = 0  # Using an integer instead of float to avoid formatting issues

    # Process each track individually
    for unique_id, track_data in df.groupby("unique_id"):
        current_track = []

        for _, row in track_data.iterrows():
            if row["segment_len_um"] > segment_length_threshold:
                deleted_segments_count += 1
                if current_track:
                    split_track_count += 1
                    track_df = pd.DataFrame(current_track)
                    track_df["particle"] = next_particle_id
                    track_df["file_id"] = track_data["file_id"].iloc[0]
                    track_df["unique_id"] = (
                        track_df["file_id"].astype(str) + "_" + str(next_particle_id)
                    )
                    # df['unique_id'] = df['file_id'].astype(str) + '_' + df['particle'].astype(str)
                    new_tracks.append(track_df)
                    next_particle_id += 1
                    current_track = []
            else:
                current_track.append(row)

        # Save any remaining valid track
        if current_track:
            track_df = pd.DataFrame(current_track)
            track_df["particle"] = next_particle_id
            track_df["file_id"] = track_data["file_id"].iloc[0]
            track_df["unique_id"] = (
                track_df["file_id"].astype(str) + "_" + str(next_particle_id)
            )
            new_tracks.append(track_df)
            next_particle_id += 1

    # Combine all new tracks
    cleaned_df = (
        pd.concat(new_tracks, ignore_index=True)
        if new_tracks
        else pd.DataFrame(columns=df.columns)
    )

    # Remove short tracks if enabled
    if remove_short_tracks and not cleaned_df.empty:
        cleaned_df["track_length"] = cleaned_df.groupby("unique_id")[
            "unique_id"
        ].transform("size")
        removed_unique_ids = set(
            cleaned_df.loc[
                cleaned_df["track_length"] < min_track_length_frames, "unique_id"
            ]
        )
        cleaned_df = cleaned_df[cleaned_df["track_length"] >= min_track_length_frames]
        cleaned_df.drop(columns=["track_length"], inplace=True)

    # Generate report
    final_track_count = cleaned_df["unique_id"].nunique()
    final_row_count = len(cleaned_df)

    report = {
        "original_track_count": original_track_count,
        "final_track_count": final_track_count,
        "tracks_split": split_track_count,
        "deleted_segments": deleted_segments_count,
        "original_row_count": original_row_count,
        "final_row_count": final_row_count,
    }

    return cleaned_df, removed_unique_ids, report


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
    df["motion_class"] = np.nan

    # Accumulate assignments per row
    frame_classes = {}
    total_windows = 0
    matched_windows = 0

    # Which IDs actually exist in the instant data?
    base_ids = set(df["unique_id"].unique())
    tol = 1e-5

    # 1) Assign each window
    for uid in tqdm(
        time_windowed_df["unique_id"].unique(), desc="Processing motion windows"
    ):
        track_id = uid if uid in base_ids else uid.split("_s")[0]
        tw = time_windowed_df[time_windowed_df["unique_id"] == uid]
        inst = df[df["unique_id"] == track_id]
        inst_idx = inst.index.to_list()
        if not inst_idx:
            continue

        for _, row in tw.iterrows():
            total_windows += 1
            sx, sy = row["x_um_start"], row["y_um_start"]
            cls = row["motion_class"]

            # find the exact start frame in this track
            hits = inst[
                (inst["x_um"].sub(sx).abs() < tol) & (inst["y_um"].sub(sy).abs() < tol)
            ].index.tolist()
            if not hits:
                continue

            matched_windows += 1
            start_idx = hits[0]
            pos = inst_idx.index(start_idx)
            # only assign within this track’s slice
            for ridx in inst_idx[pos : pos + time_window]:
                frame_classes.setdefault(ridx, []).append(cls)

    # 2) Resolve overlaps by majority vote
    for ridx, clist in frame_classes.items():
        vote = max(set(clist), key=clist.count)
        df.at[ridx, "motion_class"] = vote

    # 3) Any rows still NaN get “unlabeled”
    unlabeled = df["motion_class"].isna()
    df.loc[unlabeled, "motion_class"] = "unlabeled"

    # 4) For visualization ease, treat 'unlabeled' as missing, then ffill/bfill
    df["motion_class"] = df["motion_class"].replace("unlabeled", np.nan)
    df["motion_class"] = df["motion_class"].ffill().bfill()

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
        match = re.search(r"power-(\d+percent)_cell-(\d+)", filename)
        if match:
            power = match.group(1)  # Extracts power (e.g., '25percent')
            cellID = int(match.group(2))  # Extracts cell number as integer
            return power, cellID
        return None, None  # If no match, return None values

    df[["power", "cellID"]] = df[filename_col].apply(
        lambda x: pd.Series(parse_filename(x))
    )

    return df


def map_D_to_instant(
    metrics_df,
    time_windowed_df,
    time_window=config.TIME_WINDOW,
    overlap_method="average",
    tolerance=1e-5,
):
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
    metrics_df["diffusion_coefficient"] = np.nan

    frame_D = {}  # row → list of D values
    total_windows = 0
    matched_windows = 0

    # All track IDs present
    base_ids = set(metrics_df["unique_id"].unique())

    # 2) Assign
    for uid in tqdm(time_windowed_df["unique_id"].unique(), desc="Mapping windows"):
        tid = uid if uid in base_ids else uid.split("_s")[0]
        tw = time_windowed_df[time_windowed_df["unique_id"] == uid]
        inst = metrics_df[metrics_df["unique_id"] == tid]
        inst_idx = inst.index.to_list()
        if not inst_idx:
            continue

        for _, w in tw.iterrows():
            total_windows += 1
            sx, sy = w["x_um_start"], w["y_um_start"]
            Dval = w["diffusion_coefficient"]

            # locate the exact start row
            hits = inst[
                (inst["x_um"].sub(sx).abs() < tolerance)
                & (inst["y_um"].sub(sy).abs() < tolerance)
            ].index.tolist()
            if not hits:
                continue

            matched_windows += 1
            pos = inst_idx.index(hits[0])
            for ridx in inst_idx[pos : pos + time_window]:
                frame_D.setdefault(ridx, []).append(Dval)

    # 3) Resolve overlaps
    for ridx, Dlist in frame_D.items():
        if overlap_method == "first":
            val = Dlist[0]
        elif overlap_method == "last":
            val = Dlist[-1]
        elif overlap_method == "min":
            val = min(Dlist)
        elif overlap_method == "max":
            val = max(Dlist)
        elif overlap_method == "median":
            val = np.median(Dlist)
        else:
            val = sum(Dlist) / len(Dlist)
        metrics_df.at[ridx, "diffusion_coefficient"] = val

    # 4) Extend first/last window D to track ends
    for tid in base_ids:
        idxs = metrics_df.index[metrics_df["unique_id"] == tid].tolist()
        sub = metrics_df.loc[idxs, "diffusion_coefficient"]
        non = sub.dropna()
        if non.empty:
            continue
        first_i, last_i = non.index.min(), non.index.max()
        first_val = metrics_df.at[first_i, "diffusion_coefficient"]
        last_val = metrics_df.at[last_i, "diffusion_coefficient"]
        for i in idxs:
            if pd.isna(metrics_df.at[i, "diffusion_coefficient"]):
                metrics_df.at[i, "diffusion_coefficient"] = (
                    first_val if i < first_i else last_val
                )

    # 5) Half‑gap fill any remaining NaNs
    Dcol = metrics_df["diffusion_coefficient"]
    mask = Dcol.isna().to_numpy()
    if mask.any():
        nan_idx = np.where(mask)[0]
        groups = []
        s = nan_idx[0]
        p = s
        for i in nan_idx[1:]:
            if i == p + 1:
                p = i
            else:
                groups.append((s, p))
                s = i
                p = i
        groups.append((s, p))

        for start, end in groups:
            length = end - start + 1
            prev_val = (
                metrics_df.iloc[start - 1]["diffusion_coefficient"]
                if start > 0
                else None
            )
            next_val = (
                metrics_df.iloc[end + 1]["diffusion_coefficient"]
                if end < len(Dcol) - 1
                else None
            )

            if length == 1:
                fill = prev_val if prev_val is not None else next_val
                metrics_df.iat[
                    start, metrics_df.columns.get_loc("diffusion_coefficient")
                ] = fill
            else:
                half = length // 2
                # first half
                if prev_val is not None:
                    metrics_df.loc[
                        start : start + half - 1, "diffusion_coefficient"
                    ] = prev_val
                else:
                    metrics_df.loc[
                        start : start + half - 1, "diffusion_coefficient"
                    ] = next_val
                # second half
                if next_val is not None:
                    metrics_df.loc[start + half : end, "diffusion_coefficient"] = (
                        next_val
                    )
                else:
                    metrics_df.loc[start + half : end, "diffusion_coefficient"] = (
                        prev_val
                    )

    # quality flag & report
    metrics_df["D_fit_quality"] = np.where(
        metrics_df["diffusion_coefficient"].isna(), "bad", "good"
    )
    unmatched = int(metrics_df["diffusion_coefficient"].isna().sum())
    report = [
        f"Total windows:           {total_windows}",
        f"Matched windows:         {matched_windows}",
        f"Frames still unmatched:  {unmatched}",
    ]
    return metrics_df, report


def map_windowed_to_instant(
    metrics_df: pd.DataFrame,
    windowed_df: pd.DataFrame,
    window_size: int,
    overlap: int,
    field_cols: list,
    track_col: str = "unique_id",
    frame_col: str = "frame",
    window_col: str = "time_window",
) -> pd.DataFrame:
    """
    Broadcast selected windowed_df columns back onto each frame in metrics_df.
    - Non-float columns (e.g. 'cluster', 'motion_class'): majority vote
    - Float columns: mean over overlapping windows
    """
    # 1) Sort and index metrics_df
    md = (
        metrics_df.sort_values([track_col, frame_col], ignore_index=False)
        .reset_index()
        .rename(columns={"index": "__orig_idx"})
    )
    md["__frame_idx"] = md.groupby(track_col).cumcount()
    N = len(md)

    # 2) Map (track, frame_idx) to absolute row
    idx_map = md[[track_col, "__frame_idx"]].copy()
    idx_map["__row"] = idx_map.index

    # 3) Prepare windows and compute start rows
    step = window_size - overlap
    w = windowed_df[[track_col, window_col] + field_cols].copy()
    w["__start_idx"] = w[window_col] * step
    w = w.merge(
        idx_map,
        left_on=[track_col, "__start_idx"],
        right_on=[track_col, "__frame_idx"],
        how="left",
    ).dropna(subset=["__row"])
    w["__row"] = w["__row"].astype(int)

    # 4) Build coverage indices
    M = len(w)
    T = window_size
    starts = w["__row"].values.reshape(M, 1)
    offsets = np.arange(T).reshape(1, T)
    idxs = starts + offsets  # (M,T)
    valid = (idxs >= 0) & (idxs < N)
    flat_idxs = idxs[valid]

    result = md.copy()

    # 5) Scatter-and-reduce for each field
    for fld in field_cols:
        vals = w[fld].values.reshape(M, 1)
        flat_vals = np.repeat(vals, T, axis=1)[valid]
        if pd.api.types.is_float_dtype(windowed_df[fld]):
            # mean aggregation
            sum_arr = np.zeros(N, dtype=float)
            count_arr = np.zeros(N, dtype=int)
            np.add.at(sum_arr, flat_idxs, flat_vals.astype(float))
            np.add.at(count_arr, flat_idxs, 1)
            # arr = sum_arr / np.where(count_arr>0, count_arr, 1)
            arr = sum_arr / count_arr.astype(
                float
            )  # this will give inf where count_arr==0
            arr[count_arr == 0] = np.nan  # mask out those inf’s into NaN
        else:
            # majority vote
            ser = pd.Series(flat_vals, index=flat_idxs)
            mode = ser.groupby(level=0).agg(lambda s: Counter(s).most_common(1)[0][0])
            arr = np.full(N, np.nan, dtype=object)
            arr[mode.index.values.astype(int)] = mode.values
        result[fld] = arr

    # 6) Forward/backward fill within each track using transform for alignment
    for fld in field_cols:
        filled = result.groupby(track_col)[fld].transform(lambda s: s.ffill().bfill())
        # cast back to original dtype
        if pd.api.types.is_float_dtype(result[fld]):
            result[fld] = filled.astype(float)
        else:
            result[fld] = filled.astype(result[fld].dtype)

    # 7) Restore original ordering
    return (
        result.sort_values("__orig_idx")
        .drop(columns=["__orig_idx", "__frame_idx"])
        .reset_index(drop=True)
    )


def parse_filename_metadata(filename):
    """
    Parse filename to extract metadata components.
    Expected format: cell-{cell}_type-{type}_mol-{mol}_group-{group}_{data_type}.csv
    
    Parameters
    ----------
    filename : str
        Filename to parse
        
    Returns
    -------
    tuple
        (cell, type, mol, group, data_type) extracted from filename
    """
    # Remove .csv extension and split by underscores
    base_name = filename.replace('.csv', '')
    parts = base_name.split('_')
    
    metadata = {}
    data_type = parts[-1]  # instant or windowed
    
    # Parse the metadata parts
    for part in parts[:-1]:  # exclude the last part (data_type)
        if '-' in part:
            key, value = part.split('-', 1)
            metadata[key] = value
    
    return metadata['cell'], metadata['type'], metadata['mol'], metadata['group'], data_type


def create_identifier(cell, type_val, mol, group):
    """
    Create identifier from first letters of cell, type, mol, group.
    
    Parameters
    ----------
    cell : str
        Cell type
    type_val : str
        Type value
    mol : str
        Molecule type
    group : str
        Group identifier
        
    Returns
    -------
    str
        Four-character lowercase identifier
    """
    return f"{cell[0].lower()}{type_val[0].lower()}{mol[0].lower()}{group[0].lower()}"


def create_master_dataframes(dataframe_directory, listoffiles):
    """
    Create master instant and windowed dataframes from CSV files with metadata.
    
    Parameters
    ----------
    dataframe_directory : str
        Directory containing the CSV files
    listoffiles : list
        List of filenames to process
        
    Returns
    -------
    tuple
        (master_instant_df, master_windowed_df, summary_instant, summary_windowed)
    """
    # Initialize lists to store dataframes
    instant_dataframes = []
    windowed_dataframes = []

    print("Processing files...")

    for filename in listoffiles:
        print(f"Processing: {filename}")
        
        # Parse metadata from filename
        cell, type_val, mol, group, data_type = parse_filename_metadata(filename)
        
        # Create identifier
        identifier = create_identifier(cell, type_val, mol, group)
        
        # Load the CSV file
        file_path = os.path.join(dataframe_directory, filename)
        df = pd.read_csv(file_path)
        
        # Add metadata columns
        df['cell'] = cell
        df['type'] = type_val
        df['mol'] = mol
        df['group'] = group
        df['identifier'] = identifier
        
        # Modify unique_id if it exists
        if 'unique_id' in df.columns:
            df['unique_id'] = identifier + '_' + df['unique_id'].astype(str)
        else:
            print(f"Warning: 'unique_id' column not found in {filename}")
        
        # Add to appropriate list based on data type
        if data_type == 'instant':
            instant_dataframes.append(df)
        elif data_type == 'windowed':
            windowed_dataframes.append(df)
        
        print(f"  - Cell: {cell}, Type: {type_val}, Mol: {mol}, Group: {group}")
        print(f"  - Identifier: {identifier}")
        print(f"  - Data type: {data_type}")
        print(f"  - Shape: {df.shape}")
        print()

    print(f"Found {len(instant_dataframes)} instant dataframes")
    print(f"Found {len(windowed_dataframes)} windowed dataframes")

    # Create master dataframes
    print("\nCreating master dataframes...")
    
    master_instant_df = None
    master_windowed_df = None
    summary_instant = None
    summary_windowed = None
    
    # Create master instant dataframe
    if instant_dataframes:
        master_instant_df = pd.concat(instant_dataframes, ignore_index=True)
        print(f"Master instant dataframe created with shape: {master_instant_df.shape}")
        
        # Display summary of instant data
        print("\nInstant dataframe summary:")
        print("Unique identifiers:", master_instant_df['identifier'].unique())
        print("Combinations:")
        summary_instant = master_instant_df.groupby(['cell', 'type', 'mol', 'group', 'identifier']).size().reset_index(name='count')
        print(summary_instant)
    else:
        print("No instant dataframes found!")

    print()

    # Create master windowed dataframe  
    if windowed_dataframes:
        master_windowed_df = pd.concat(windowed_dataframes, ignore_index=True)
        print(f"Master windowed dataframe created with shape: {master_windowed_df.shape}")
        
        # Display summary of windowed data
        print("\nWindowed dataframe summary:")
        print("Unique identifiers:", master_windowed_df['identifier'].unique())
        print("Combinations:")
        summary_windowed = master_windowed_df.groupby(['cell', 'type', 'mol', 'group', 'identifier']).size().reset_index(name='count')
        print(summary_windowed)
    else:
        print("No windowed dataframes found!")

    print("\n✓ Master dataframes created successfully!")
    
    return master_instant_df, master_windowed_df, summary_instant, summary_windowed


# ==================== POLARS-BASED FUNCTIONS ====================

def extract_location(filename):
    """
    Extract location from filename using regex pattern.
    
    Parameters
    ----------
    filename : str
        Filename to extract location from
        
    Returns
    -------
    str
        Location in uppercase, or "Unknown" if not found
    """
    match = re.search(r"loc-(\w+)(?:_|$)", filename)
    if match:
        return match.group(1).upper()  # Convert to uppercase
    return "Unknown"  # Default value if no location is found


def extract_metadata_from_condition(condition):
    """
    Extract metadata components from condition string.
    
    Parameters
    ----------
    condition : str
        Condition string containing metadata like "cell-neuron_type-wildtype_mol-HTT_geno-Q175_"
        
    Returns
    -------
    dict
        Dictionary with extracted metadata: {'cell': ..., 'type': ..., 'mol': ..., 'geno': ...}
    """
    metadata = {}
    # Split by underscore and look for key-value pairs
    parts = condition.split('_')
    for part in parts:
        if '-' in part:
            key, value = part.split('-', 1)
            metadata[key] = value
    
    return metadata


def extract_metadata_from_foldername(foldername):
    """
    Extract group metadata from folder name.
    
    Parameters
    ----------
    foldername : str
        Folder name containing group info like "group-HTT77_something" 
        
    Returns
    -------
    str
        Group value (only up to the underscore) or "Unknown" if not found
    """
    # Look for group-XXX pattern in the folder name
    match = re.search(r"group-([^_]+)", foldername)
    if match:
        return match.group(1)
    return "Unknown"


def create_identifier_polars(cell, type_val, mol, geno, group):
    """
    Create identifier from first letters of cell, type, mol, geno, group.
    
    Parameters
    ----------
    cell : str
        Cell type
    type_val : str
        Type value
    mol : str
        Molecule type
    geno : str
        Genotype
    group : str
        Group identifier
        
    Returns
    -------
    str
        Five-character lowercase identifier
    """
    parts = [cell, type_val, mol, geno, group]
    identifier = ''.join([str(part)[0].lower() if part and str(part) != 'Unknown' else 'x' for part in parts])
    return identifier


def load_dataframe_with_metadata_polars(folder_path, dataframe_name, parent_folder_path):
    """
    Load a single dataframe with metadata extraction using Polars.
    
    Parameters
    ----------
    folder_path : str
        Path to folder containing the CSV file (the saved_data directory)
    dataframe_name : str
        Name of the CSV file to load (without .csv extension)
    parent_folder_path : str
        Full path to the parent folder (for compatibility with existing code)
        
    Returns
    -------
    pl.DataFrame or None
        Loaded dataframe with metadata columns, or None if file not found
    """
    csv_path = os.path.join(folder_path, f"{dataframe_name}.csv")
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return None
    
    try:
        # Load with Polars
        df = pl.read_csv(csv_path)
        
        # Add foldername column (full path with trailing backslash for compatibility)
        folder_name_with_slash = parent_folder_path + ('' if parent_folder_path.endswith(('\\', '/')) else '\\')
        df = df.with_columns(pl.lit(folder_name_with_slash).alias("foldername"))
        
        # Extract group from folder basename
        folder_basename = os.path.basename(parent_folder_path.rstrip('\\/'))
        group = extract_metadata_from_foldername(folder_basename)
        df = df.with_columns(pl.lit(group).alias("group"))
        
        # Extract metadata from condition column if it exists
        if "condition" in df.columns:
            # Get a sample condition value to extract metadata keys
            sample_condition = df.select("condition").head(1).to_pandas().iloc[0, 0]
            metadata_keys = extract_metadata_from_condition(sample_condition)
            
            # Add metadata columns
            for key in ['cell', 'type', 'mol', 'geno']:
                if key in metadata_keys:
                    # Use regex to extract each metadata field
                    pattern = f"{key}-([^_]+)"
                    df = df.with_columns(
                        pl.col("condition")
                        .str.extract(pattern, 1)
                        .fill_null("Unknown")
                        .alias(key)
                    )
                else:
                    df = df.with_columns(pl.lit("Unknown").alias(key))
        else:
            # Add default values if condition column doesn't exist
            for key in ['cell', 'type', 'mol', 'geno']:
                df = df.with_columns(pl.lit("Unknown").alias(key))
        
        # Extract location from filename if it exists
        if "filename" in df.columns:
            df = df.with_columns(
                pl.col("filename")
                .map_elements(extract_location, return_dtype=pl.Utf8)
                .alias("location")
            )
        else:
            df = df.with_columns(pl.lit("Unknown").alias("location"))
        
        # Create identifier
        df = df.with_columns(
            pl.struct(["cell", "type", "mol", "geno", "group"])
            .map_elements(
                lambda x: create_identifier_polars(
                    x["cell"], x["type"], x["mol"], x["geno"], x["group"]
                ),
                return_dtype=pl.Utf8
            )
            .alias("identifier")
        )
        
        # Modify unique_id if it exists
        if "unique_id" in df.columns:
            df = df.with_columns(
                (pl.col("identifier") + "_" + pl.col("unique_id").cast(pl.Utf8)).alias("unique_id")
            )
        
        return df
        
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def create_master_dataframes_polars(master_dir, instant_df_name="instant_df_processed", windowed_df_name="time_windowed_df_processed"):
    """
    Create master instant and windowed dataframes from all folders in master_dir using Polars.
    
    Parameters
    ----------
    master_dir : str
        Path to master directory containing subdirectories with saved_data folders
    instant_df_name : str, default "instant_df_processed"
        Name of the instant dataframe CSV file (without .csv extension)
    windowed_df_name : str, default "time_windowed_df_processed"
        Name of the windowed dataframe CSV file (without .csv extension)
        
    Returns
    -------
    tuple
        (master_instant_df, master_windowed_df) - Polars DataFrames
    """
    print(f"Scanning master directory: {master_dir}")
    
    instant_dataframes = []
    windowed_dataframes = []
    
    # Scan all subdirectories in master_dir
    for item in os.listdir(master_dir):
        item_path = os.path.join(master_dir, item)
        if os.path.isdir(item_path):
            # Look for saved_data subdirectory
            saved_data_path = os.path.join(item_path, "saved_data")
            if os.path.exists(saved_data_path) and os.path.isdir(saved_data_path):
                print(f"Processing folder: {item}")
                
                # Load instant dataframe - pass the full parent folder path for compatibility
                instant_df = load_dataframe_with_metadata_polars(
                    saved_data_path, instant_df_name, item_path
                )
                if instant_df is not None:
                    instant_dataframes.append(instant_df)
                    print(f"  ✓ Loaded {instant_df_name}.csv ({instant_df.shape[0]} rows)")
                
                # Load windowed dataframe - pass the full parent folder path for compatibility
                windowed_df = load_dataframe_with_metadata_polars(
                    saved_data_path, windowed_df_name, item_path
                )
                if windowed_df is not None:
                    windowed_dataframes.append(windowed_df)
                    print(f"  ✓ Loaded {windowed_df_name}.csv ({windowed_df.shape[0]} rows)")
            else:
                print(f"  ⚠ No saved_data directory found in {item}")
    
    # Concatenate dataframes
    master_instant_df = None
    master_windowed_df = None
    
    if instant_dataframes:
        print(f"\nConcatenating {len(instant_dataframes)} instant dataframes...")
        master_instant_df = pl.concat(instant_dataframes, how="vertical")
        print(f"Master instant dataframe shape: {master_instant_df.shape}")
    else:
        print("No instant dataframes found!")
    
    if windowed_dataframes:
        print(f"\nConcatenating {len(windowed_dataframes)} windowed dataframes...")
        master_windowed_df = pl.concat(windowed_dataframes, how="vertical")
        print(f"Master windowed dataframe shape: {master_windowed_df.shape}")
    else:
        print("No windowed dataframes found!")
    
    return master_instant_df, master_windowed_df


def save_master_dataframes_polars(master_instant_df, master_windowed_df, save_dir, 
                                  instant_filename="master_instant_df", 
                                  windowed_filename="master_windowed_df",
                                  save_parquet=True, save_csv=True):
    """
    Save master dataframes in Parquet and/or CSV format.
    
    Parameters
    ----------
    master_instant_df : pl.DataFrame
        Master instant dataframe
    master_windowed_df : pl.DataFrame
        Master windowed dataframe
    save_dir : str
        Directory to save the files
    instant_filename : str, default "master_instant_df"
        Base filename for instant dataframe (without extension)
    windowed_filename : str, default "master_windowed_df"
        Base filename for windowed dataframe (without extension)
    save_parquet : bool, default True
        Whether to save in Parquet format
    save_csv : bool, default True
        Whether to save in CSV format
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if master_instant_df is not None:
        if save_parquet:
            parquet_path = os.path.join(save_dir, f"{instant_filename}.parquet")
            master_instant_df.write_parquet(parquet_path)
            print(f"✓ Saved instant dataframe to: {parquet_path}")
        
        if save_csv:
            csv_path = os.path.join(save_dir, f"{instant_filename}.csv")
            master_instant_df.write_csv(csv_path)
            print(f"✓ Saved instant dataframe to: {csv_path}")
    
    if master_windowed_df is not None:
        if save_parquet:
            parquet_path = os.path.join(save_dir, f"{windowed_filename}.parquet")
            master_windowed_df.write_parquet(parquet_path)
            print(f"✓ Saved windowed dataframe to: {parquet_path}")
        
        if save_csv:
            csv_path = os.path.join(save_dir, f"{windowed_filename}.csv")
            master_windowed_df.write_csv(csv_path)
            print(f"✓ Saved windowed dataframe to: {csv_path}")


def load_master_dataframes_polars(save_dir, 
                                  instant_filename="master_instant_df.parquet", 
                                  windowed_filename="master_windowed_df.parquet"):
    """
    Load previously saved master dataframes.
    
    Parameters
    ----------
    save_dir : str
        Directory containing the saved files
    instant_filename : str, default "master_instant_df.parquet"
        Filename for instant dataframe
    windowed_filename : str, default "master_windowed_df.parquet"
        Filename for windowed dataframe
        
    Returns
    -------
    tuple
        (master_instant_df, master_windowed_df) - Polars DataFrames or None if not found
    """
    master_instant_df = None
    master_windowed_df = None
    
    instant_path = os.path.join(save_dir, instant_filename)
    if os.path.exists(instant_path):
        master_instant_df = pl.read_parquet(instant_path)
        print(f"✓ Loaded instant dataframe from: {instant_path}")
    else:
        print(f"⚠ Instant dataframe not found: {instant_path}")
    
    windowed_path = os.path.join(save_dir, windowed_filename)
    if os.path.exists(windowed_path):
        master_windowed_df = pl.read_parquet(windowed_path)
        print(f"✓ Loaded windowed dataframe from: {windowed_path}")
    else:
        print(f"⚠ Windowed dataframe not found: {windowed_path}")
    
    return master_instant_df, master_windowed_df


def print_dataframe_summary_polars(df, df_name):
    """
    Print a summary of the dataframe including shape, columns, and unique values in key columns.
    
    Parameters
    ----------
    df : pl.DataFrame
        Dataframe to summarize
    df_name : str
        Name of the dataframe for printing
    """
    if df is None:
        print(f"{df_name}: None")
        return
    
    print(f"\n{df_name} Summary:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns}")
    
    # Print unique values for key metadata columns
    key_columns = ['foldername', 'cell', 'type', 'mol', 'geno', 'group', 'identifier', 'location']
    for col in key_columns:
        if col in df.columns:
            unique_values = df.select(col).unique().to_pandas()[col].tolist()
            print(f"{col}: {unique_values}")


def convert_polars_to_pandas(df_polars):
    """
    Convert Polars DataFrame to Pandas DataFrame for compatibility with existing functions.
    
    Parameters
    ----------
    df_polars : pl.DataFrame
        Polars DataFrame to convert
        
    Returns
    -------
    pd.DataFrame
        Converted Pandas DataFrame
    """
    return df_polars.to_pandas() if df_polars is not None else None


def load_dataframes_by_pattern_polars(base_directory, instant_df_name="instant_df_processed", windowed_df_name="time_windowed_df_processed"):
    """
    Load dataframes using the exact same pattern as the old method, but with Polars.
    Mimics the original folder scanning approach for maximum compatibility.
    
    Parameters
    ----------
    base_directory : str
        Base directory containing folders with saved_data subdirectories
    instant_df_name : str, default "instant_df_processed"
        Name of the instant dataframe CSV file (without .csv extension)
    windowed_df_name : str, default "time_windowed_df_processed"
        Name of the windowed dataframe CSV file (without .csv extension)
        
    Returns
    -------
    dict
        Dictionary with keys like 'HTT_metrics_df', 'HTT_time_windowed_df', etc.
        depending on what folders are found
    """
    # Store the paths of the folders contained in base_directory
    folders = [f.path for f in os.scandir(base_directory) if f.is_dir()]
    
    # Add /saved_data to the path
    folders = [f + '/saved_data' for f in folders]
    
    dataframes = {}
    
    # Process each folder
    for folder in folders:
        # Get the folder directory name but remove the '/saved_data' part
        folder_name = os.path.dirname(folder) + '\\'
        
        try:
            if 'HTTinEScells' in folder:
                instant_df = pl.read_csv(os.path.join(folder, f'{instant_df_name}.csv'))
                windowed_df = pl.read_csv(os.path.join(folder, f'{windowed_df_name}.csv'))
                
                # Add foldername as a column to the dataframes (exactly like the old method)
                instant_df = instant_df.with_columns(pl.lit(folder_name).alias("foldername"))
                windowed_df = windowed_df.with_columns(pl.lit(folder_name).alias("foldername"))
                
                dataframes['HTT_metrics_df'] = instant_df
                dataframes['HTT_time_windowed_df'] = windowed_df
                print(f"✓ Loaded HTT dataframes from {folder}")
                
            elif 'DyneininEScells' in folder:
                instant_df = pl.read_csv(os.path.join(folder, f'{instant_df_name}.csv'))
                windowed_df = pl.read_csv(os.path.join(folder, f'{windowed_df_name}.csv'))
                
                instant_df = instant_df.with_columns(pl.lit(folder_name).alias("foldername"))
                windowed_df = windowed_df.with_columns(pl.lit(folder_name).alias("foldername"))
                
                dataframes['Dynein_metrics_df'] = instant_df
                dataframes['Dynein_time_windowed_df'] = windowed_df
                print(f"✓ Loaded Dynein dataframes from {folder}")
                
            elif 'KinesinEScells' in folder:
                instant_df = pl.read_csv(os.path.join(folder, f'{instant_df_name}.csv'))
                windowed_df = pl.read_csv(os.path.join(folder, f'{windowed_df_name}.csv'))
                
                instant_df = instant_df.with_columns(pl.lit(folder_name).alias("foldername"))
                windowed_df = windowed_df.with_columns(pl.lit(folder_name).alias("foldername"))
                
                dataframes['Kinesin_metrics_df'] = instant_df
                dataframes['Kinesin_time_windowed_df'] = windowed_df
                print(f"✓ Loaded Kinesin dataframes from {folder}")
                
            elif 'MyosinEScells' in folder:
                instant_df = pl.read_csv(os.path.join(folder, f'{instant_df_name}.csv'))
                windowed_df = pl.read_csv(os.path.join(folder, f'{windowed_df_name}.csv'))
                
                instant_df = instant_df.with_columns(pl.lit(folder_name).alias("foldername"))
                windowed_df = windowed_df.with_columns(pl.lit(folder_name).alias("foldername"))
                
                dataframes['Myosin_metrics_df'] = instant_df
                dataframes['Myosin_time_windowed_df'] = windowed_df
                print(f"✓ Loaded Myosin dataframes from {folder}")
                
        except Exception as e:
            print(f"Error loading from {folder}: {e}")
    
    return dataframes


def split_by_diffusion_threshold(data_df, feature="diffusion_coefficient", threshold=None, 
                                 log_scale=True, log_base=10):
    """
    Split dataframe into two based on a threshold value for diffusion coefficient.
    Mimics the log calculation logic from the histograms function.
    Supports both pandas and polars dataframes with automatic detection.
    
    Parameters
    -----------
    data_df : pandas.DataFrame or polars.DataFrame
        Input dataframe containing the feature to split on
    feature : str, default="diffusion_coefficient"
        Column name to apply threshold to
    threshold : float
        Threshold value to split on, IN LOG SCALE if log_scale=True. 
        If None, uses median of the (log-transformed) feature
    log_scale : bool, default=True
        Whether to calculate log values before applying threshold
    log_base : int, default=10
        Base for logarithm calculation (10, 2, or any other number)
    
    Returns
    --------
    full_df_with_threshold : pandas.DataFrame or polars.DataFrame
        Full original dataframe with added 'simple_threshold' column ('low'/'high')
    low_windowed_df : pandas.DataFrame or polars.DataFrame
        Dataframe containing values below threshold with 'simple_threshold' column
    high_windowed_df : pandas.DataFrame or polars.DataFrame
        Dataframe containing values at or above threshold with 'simple_threshold' column
    threshold_used : float
        The actual threshold value used for splitting (in log scale if log_scale=True)
    """
    
    # Detect dataframe type and handle accordingly
    is_polars = hasattr(data_df, 'schema')  # polars DataFrames have schema attribute
    
    if is_polars:
        import polars as pl
        # Make a copy to avoid modifying original dataframe
        df_copy = data_df.clone()
        # Check if feature exists in dataframe
        if feature not in df_copy.columns:
            raise ValueError(f"Feature '{feature}' not found in dataframe columns")
    else:
        # Assume pandas
        # Make a copy to avoid modifying original dataframe
        df_copy = data_df.copy()
        # Check if feature exists in dataframe
        if feature not in df_copy.columns:
            raise ValueError(f"Feature '{feature}' not found in dataframe columns")
    
    # Handle log scale transformation (same logic as histograms function)
    if log_scale:
        new_feature = "log_" + feature
        
        if is_polars:
            # Check for positive values in polars
            if (df_copy.select(pl.col(feature) <= 0).sum().item() > 0):
                raise ValueError(f"All values of {feature} must be positive for log scale.")
            
            # Create log transformed column in polars
            if log_base == 10:
                df_copy = df_copy.with_columns(pl.col(feature).log10().alias(new_feature))
            elif log_base == 2:
                df_copy = df_copy.with_columns(pl.col(feature).log().alias(new_feature) / np.log(2))
            else:
                df_copy = df_copy.with_columns(pl.col(feature).log().alias(new_feature) / np.log(log_base))
        else:
            # Pandas version
            if (df_copy[feature] <= 0).any():
                raise ValueError(f"All values of {feature} must be positive for log scale.")
            
            if log_base == 10:
                df_copy[new_feature] = np.log10(df_copy[feature])
            elif log_base == 2:
                df_copy[new_feature] = np.log2(df_copy[feature])
            else:
                df_copy[new_feature] = np.log(df_copy[feature]) / np.log(log_base)
        
        feature_to_split = new_feature
        
        # Threshold is already in log scale - use directly
        if threshold is not None:
            threshold_log = threshold
            # Convert to original scale for reporting
            threshold_original = log_base ** threshold_log
        else:
            # Use median of log-transformed values
            if is_polars:
                threshold_log = df_copy.select(pl.col(feature_to_split).median()).item()
            else:
                threshold_log = df_copy[feature_to_split].median()
            # Convert to original scale for reporting
            threshold_original = log_base ** threshold_log
    else:
        feature_to_split = feature
        threshold_log = threshold
        threshold_original = threshold
        if threshold is None:
            if is_polars:
                threshold_log = df_copy.select(pl.col(feature_to_split).median()).item()
            else:
                threshold_log = df_copy[feature_to_split].median()
            threshold_original = threshold_log
    
    # Add simple_threshold column and split the dataframe
    if is_polars:
        # Add simple_threshold column
        df_copy = df_copy.with_columns(
            pl.when(pl.col(feature_to_split) < threshold_log)
            .then(pl.lit("low"))
            .otherwise(pl.lit("high"))
            .alias("simple_threshold")
        )
        
        # Split the dataframe
        low_windowed_df = df_copy.filter(pl.col("simple_threshold") == "low")
        high_windowed_df = df_copy.filter(pl.col("simple_threshold") == "high")
        full_df_with_threshold = df_copy
    else:
        # Pandas version
        # Add simple_threshold column
        df_copy['simple_threshold'] = np.where(
            df_copy[feature_to_split] < threshold_log, 
            'low', 
            'high'
        )
        
        # Split the dataframe
        low_windowed_df = df_copy[df_copy['simple_threshold'] == 'low'].copy()
        high_windowed_df = df_copy[df_copy['simple_threshold'] == 'high'].copy()
        full_df_with_threshold = df_copy
    
    # Print summary statistics
    if is_polars:
        total_count = df_copy.height
        low_count = low_windowed_df.height
        high_count = high_windowed_df.height
    else:
        total_count = len(df_copy)
        low_count = len(low_windowed_df)
        high_count = len(high_windowed_df)
    
    print(f"Split summary for {feature}:")
    print(f"  Dataframe type: {'Polars' if is_polars else 'Pandas'}")
    print(f"  Total records: {total_count}")
    if log_scale:
        print(f"  Log{log_base} threshold: {threshold_log:.4f}")
        print(f"  Original scale threshold: {threshold_original:.4f}")
    else:
        print(f"  Threshold used: {threshold_log:.4f}")
    print(f"  Low group (< threshold): {low_count} ({low_count/total_count*100:.1f}%)")
    print(f"  High group (>= threshold): {high_count} ({high_count/total_count*100:.1f}%)")
    print(f"  Added 'simple_threshold' column with values: 'low', 'high'")
    
    return full_df_with_threshold, low_windowed_df, high_windowed_df, threshold_log


def analyze_feature_variance_pca(
    df,
    feature_columns,
    n_components=None,
    scale_data=True,
    group_by=None,
    plot=True,
    figsize=(12, 5),
    return_transformed=False
):
    """
    Perform PCA analysis to understand feature variance and contribution.
    
    This helps identify which features contribute most to variance in your data,
    useful for understanding which motion parameters are most informative for
    distinguishing track behaviors.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the features to analyze.
    feature_columns : list of str
        List of column names to include in PCA analysis.
    n_components : int, optional
        Number of principal components to compute. If None, uses min(n_features, n_samples).
        Default is None.
    scale_data : bool, optional
        Whether to standardize features before PCA. Highly recommended when features
        have different scales. Default is True.
    group_by : str, optional
        If provided, performs separate PCA for each group and plots comparison.
        Default is None.
    plot : bool, optional
        Whether to generate visualization plots. Default is True.
    figsize : tuple, optional
        Figure size for plots. Default is (12, 5).
    return_transformed : bool, optional
        Whether to return the PCA-transformed data. Default is False.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'pca': fitted PCA object(s)
        - 'explained_variance_ratio': variance explained by each component
        - 'cumulative_variance': cumulative variance explained
        - 'components': principal component loadings (feature contributions)
        - 'feature_importance': absolute contribution of each feature to variance
        - 'transformed_data': PCA-transformed data (if return_transformed=True)
    
    Examples
    --------
    >>> # Basic PCA on motion features
    >>> features = ['diffusion_coefficient', 'avg_speed_um_s', 'cum_displacement_um', 
    ...             'radius_gyration', 'straightness']
    >>> results = analyze_feature_variance_pca(df, features)
    
    >>> # PCA by experimental condition
    >>> results = analyze_feature_variance_pca(df, features, group_by='condition')
    
    >>> # Get transformed data for clustering
    >>> results = analyze_feature_variance_pca(df, features, n_components=3, 
    ...                                         return_transformed=True)
    >>> pca_features = results['transformed_data']
    
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Ensure feature_columns is a list
    if isinstance(feature_columns, str):
        feature_columns = [feature_columns]
    
    # Check if all columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")
    
    # Remove rows with NaN in feature columns
    df_clean = df[feature_columns].dropna()
    n_samples = len(df_clean)
    n_features = len(feature_columns)
    
    print(f"📊 PCA Analysis on {n_features} features, {n_samples:,} samples")
    print(f"Features: {', '.join(feature_columns)}")
    
    if n_components is None:
        n_components = min(n_features, n_samples)
    
    # Scale data if requested
    if scale_data:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean)
        print("✓ Data standardized (mean=0, std=1)")
    else:
        X_scaled = df_clean.values
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate feature importance (sum of absolute loadings across components)
    feature_importance = np.abs(pca.components_).sum(axis=0)
    feature_importance = feature_importance / feature_importance.sum() * 100
    
    # Sort features by importance
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance_%': feature_importance
    }).sort_values('importance_%', ascending=False)
    
    print(f"\n🎯 Variance Explained:")
    for i, var in enumerate(pca.explained_variance_ratio_[:5]):
        print(f"   PC{i+1}: {var*100:.2f}%")
    print(f"   Total (first {min(5, n_components)} PCs): {pca.explained_variance_ratio_[:5].sum()*100:.2f}%")
    
    print(f"\n🔍 Feature Importance (contribution to overall variance):")
    for _, row in importance_df.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance_%']:.2f}%")
    
    # Plotting
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: Scree plot (variance explained by each PC)
        ax = axes[0]
        pcs = np.arange(1, len(pca.explained_variance_ratio_) + 1)
        ax.bar(pcs, pca.explained_variance_ratio_ * 100, alpha=0.7, color='steelblue')
        ax.plot(pcs, np.cumsum(pca.explained_variance_ratio_) * 100, 
                'ro-', linewidth=2, markersize=6, label='Cumulative')
        ax.set_xlabel('Principal Component', fontsize=11)
        ax.set_ylabel('Variance Explained (%)', fontsize=11)
        ax.set_title('Scree Plot', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 2: Feature loadings heatmap (first 3-5 PCs)
        ax = axes[1]
        n_pcs_to_show = min(5, n_components)
        loadings = pca.components_[:n_pcs_to_show, :]
        sns.heatmap(loadings, 
                    xticklabels=feature_columns,
                    yticklabels=[f'PC{i+1}' for i in range(n_pcs_to_show)],
                    cmap='RdBu_r', center=0, annot=False, 
                    cbar_kws={'label': 'Loading'}, ax=ax, vmin=-1, vmax=1)
        ax.set_title('Feature Loadings', fontsize=12, fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        
        # Plot 3: Feature importance bar plot
        ax = axes[2]
        importance_sorted = importance_df.sort_values('importance_%', ascending=True)
        ax.barh(importance_sorted['feature'], importance_sorted['importance_%'], 
                alpha=0.7, color='coral')
        ax.set_xlabel('Contribution to Variance (%)', fontsize=11)
        ax.set_title('Feature Importance', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()
    
    # Prepare results
    results = {
        'pca': pca,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'components': pca.components_,
        'feature_importance': importance_df,
        'n_components': n_components,
        'feature_columns': feature_columns
    }
    
    if return_transformed:
        # Return transformed data as DataFrame with original index
        pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        transformed_df = pd.DataFrame(X_pca, columns=pca_cols, index=df_clean.index)
        results['transformed_data'] = transformed_df
    
    return results


def extract_top_features_per_pc(pca_results, n_features=5):
    """
    Extract the top contributing features for each principal component.
    
    This shows which features drive each PC specifically, helping you understand
    what each dimension of variation represents biologically.
    
    Parameters
    ----------
    pca_results : dict
        Results dictionary from analyze_feature_variance_pca()
    n_features : int, optional
        Number of top features to extract per PC. Default is 5.
    
    Returns
    -------
    dict
        Dictionary mapping PC names to DataFrames with top features and their loadings.
        Also prints a formatted summary.
    
    Examples
    --------
    >>> # After running PCA
    >>> pca_results = spt.analyze_feature_variance_pca(df, features)
    >>> top_features_per_pc = spt.extract_top_features_per_pc(pca_results, n_features=3)
    
    """
    import pandas as pd
    
    components = pca_results['components']
    feature_names = pca_results['feature_columns']
    explained_var = pca_results['explained_variance_ratio']
    n_pcs = components.shape[0]
    
    print("=" * 80)
    print("🔍 TOP FEATURES PER PRINCIPAL COMPONENT")
    print("=" * 80)
    
    pc_feature_dict = {}
    
    for pc_idx in range(min(n_pcs, 10)):  # Show up to 10 PCs
        loadings = components[pc_idx, :]
        
        # Get absolute values for ranking but keep signs for interpretation
        abs_loadings = np.abs(loadings)
        top_indices = np.argsort(abs_loadings)[-n_features:][::-1]
        
        # Create DataFrame for this PC
        top_features_df = pd.DataFrame({
            'feature': [feature_names[i] for i in top_indices],
            'loading': [loadings[i] for i in top_indices],
            'abs_loading': [abs_loadings[i] for i in top_indices]
        })
        
        pc_name = f'PC{pc_idx + 1}'
        pc_feature_dict[pc_name] = top_features_df
        
        # Print formatted output
        print(f"\n📊 {pc_name} (explains {explained_var[pc_idx]*100:.2f}% of variance)")
        print("-" * 80)
        
        for idx, row in top_features_df.iterrows():
            direction = "+" if row['loading'] > 0 else "-"
            print(f"   {idx+1}. {row['feature']:<30} {direction} {abs(row['loading']):.3f}")
    
    print("\n" + "=" * 80)
    print("💡 INTERPRETATION GUIDE:")
    print("   - Positive (+) loading: feature increases with PC score")
    print("   - Negative (-) loading: feature decreases with PC score")
    print("   - Higher absolute value = stronger contribution")
    print("   - PCs are orthogonal (independent dimensions of variation)")
    print("=" * 80)
    
    return pc_feature_dict


def analyze_feature_relationships(
    df,
    features,
    group_by=None,
    method='all',
    figsize=(12, 10),
    save_path=None,
    transparent_background=False,
    line_color='black',
    text_color='black',
    export_format='png'
):
    """
    Analyze relationships between features to guide feature selection for clustering/thresholding.
    
    Helps identify:
    - Redundant features (high correlation)
    - Independent features (low correlation, high mutual information)
    - Features with complementary information
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data containing features
    features : list of str
        Feature column names to analyze
    group_by : str, optional
        Column name to color by groups (e.g., 'mol'). Default None.
    method : str, optional
        Analysis type: 'correlation', 'mutual_info', 'both', or 'all'. Default 'all'.
    figsize : tuple, optional
        Figure size. Default (12, 10).
    save_path : str, optional
        Path to save figure. Default None.
    transparent_background : bool, optional
        If True, save with transparent background. Default False.
    line_color : str, optional
        Color for axes, ticks, labels. Default 'black'.
    text_color : str, optional
        Color for text elements. Default 'black'.
    export_format : str, optional
        File format ('png' or 'svg'). Default 'png'.
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'correlation_matrix': correlation matrix
        - 'mutual_info_matrix': mutual information matrix (if computed)
        - 'redundant_pairs': list of highly correlated feature pairs
        - 'independent_pairs': list of weakly correlated pairs
    
    Examples
    --------
    >>> results = spt.analyze_feature_relationships(df, features, group_by='mol')
    >>> print("Redundant pairs:", results['redundant_pairs'])
    
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.feature_selection import mutual_info_regression
    from scipy.stats import spearmanr
    
    # Create a clean dataframe with only the features
    feature_df = df[features].copy()
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    results = {}
    
    # 1. CORRELATION MATRIX
    if method in ['correlation', 'both', 'all']:
        corr_matrix = feature_df.corr(method='spearman')
        results['correlation_matrix'] = corr_matrix
        
        # Find redundant pairs (|correlation| > 0.8)
        redundant_pairs = []
        independent_pairs = []
        
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    redundant_pairs.append((features[i], features[j], corr_val))
                elif abs(corr_val) < 0.3:
                    independent_pairs.append((features[i], features[j], corr_val))
        
        results['redundant_pairs'] = redundant_pairs
        results['independent_pairs'] = independent_pairs
        
        # Plot correlation matrix
        if method in ['correlation', 'all']:
            # Set up background
            figure_background = 'none' if transparent_background else 'white'
            axis_background = (0, 0, 0, 0) if transparent_background else 'white'
            
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor(figure_background)
            ax.set_facecolor(axis_background)
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Spearman Correlation'},
                xticklabels=features,
                yticklabels=features,
                ax=ax
            )
            
            # Style text and axes
            ax.set_title('Feature Correlation Matrix\n(High values = redundant, Low = independent)',
                        fontsize=14, fontweight='bold', pad=20, color=text_color)
            ax.tick_params(colors=line_color, which='both')
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            plt.setp(ax.get_xticklabels(), color=text_color, rotation=45, ha='right')
            plt.setp(ax.get_yticklabels(), color=text_color, rotation=0)
            
            # Style colorbar
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_tick_params(color=line_color)
            cbar.ax.yaxis.label.set_color(text_color)
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=text_color)
            
            plt.tight_layout()
            
            if save_path:
                save_file = f"{save_path}/correlation_matrix.{export_format}"
                plt.savefig(save_file, dpi=300, bbox_inches='tight', 
                           transparent=transparent_background)
                print(f"Saved: {save_file}")
            plt.show()
    
    # 2. MUTUAL INFORMATION (captures nonlinear relationships)
    if method in ['mutual_info', 'both', 'all']:
        print("Computing mutual information (this may take a moment)...")
        
        n_features = len(features)
        mi_matrix = np.zeros((n_features, n_features))
        
        for i, feat_i in enumerate(features):
            for j, feat_j in enumerate(features):
                if i == j:
                    mi_matrix[i, j] = np.nan  # Self-MI is not meaningful
                elif i < j:
                    # Compute MI
                    X = feature_df[feat_i].values.reshape(-1, 1)
                    y = feature_df[feat_j].values
                    mi = mutual_info_regression(X, y, random_state=42)[0]
                    mi_matrix[i, j] = mi
                    mi_matrix[j, i] = mi  # Symmetric
        
        results['mutual_info_matrix'] = pd.DataFrame(
            mi_matrix, 
            index=features, 
            columns=features
        )
        
        # Plot MI matrix
        if method in ['mutual_info', 'all']:
            # Set up background
            figure_background = 'none' if transparent_background else 'white'
            axis_background = (0, 0, 0, 0) if transparent_background else 'white'
            
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor(figure_background)
            ax.set_facecolor(axis_background)
            
            mask = np.triu(np.ones_like(mi_matrix, dtype=bool))
            
            sns.heatmap(
                mi_matrix,
                mask=mask,
                annot=True,
                fmt='.2f',
                cmap='YlOrRd',
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Mutual Information'},
                xticklabels=features,
                yticklabels=features,
                ax=ax
            )
            
            # Style text and axes
            ax.set_title('Mutual Information Matrix\n(Captures nonlinear dependencies)',
                        fontsize=14, fontweight='bold', pad=20, color=text_color)
            ax.tick_params(colors=line_color, which='both')
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            plt.setp(ax.get_xticklabels(), color=text_color, rotation=45, ha='right')
            plt.setp(ax.get_yticklabels(), color=text_color, rotation=0)
            
            # Style colorbar
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_tick_params(color=line_color)
            cbar.ax.yaxis.label.set_color(text_color)
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=text_color)
            
            plt.tight_layout()
            
            if save_path:
                save_file = f"{save_path}/mutual_info_matrix.{export_format}"
                plt.savefig(save_file, dpi=300, bbox_inches='tight',
                           transparent=transparent_background)
                print(f"Saved: {save_file}")
            plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("FEATURE RELATIONSHIP ANALYSIS")
    print("="*80)
    
    if 'redundant_pairs' in results:
        print(f"\n🔴 REDUNDANT PAIRS (|r| > 0.8): {len(results['redundant_pairs'])}")
        print("   (Consider removing one from each pair)")
        for feat1, feat2, corr in sorted(results['redundant_pairs'], 
                                         key=lambda x: abs(x[2]), reverse=True)[:10]:
            print(f"   {feat1:35s} <-> {feat2:35s} | r = {corr:+.3f}")
        
        print(f"\n✅ INDEPENDENT PAIRS (|r| < 0.3): {len(results['independent_pairs'])}")
        print("   (Good candidates for multi-parameter thresholding)")
        for feat1, feat2, corr in results['independent_pairs'][:10]:
            print(f"   {feat1:35s} <-> {feat2:35s} | r = {corr:+.3f}")
    
    print("="*80)
    
    return results


def detect_multimodality(
    df,
    features,
    group_by=None,
    n_bins=50,
    figsize=(15, 8),
    save_path=None,
    transparent_background=False,
    line_color='black',
    text_color='black',
    export_format='png'
):
    """
    Detect bimodal/multimodal distributions in features (good thresholding candidates).
    
    Uses Hartigan's dip test and visual inspection to identify features with
    multiple peaks, which are ideal for setting thresholds.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data containing features
    features : list of str
        Feature column names to test
    group_by : str, optional
        Column to separate groups. Default None.
    n_bins : int, optional
        Number of histogram bins. Default 50.
    figsize : tuple, optional
        Figure size. Default (15, 8).
    save_path : str, optional
        Path to save figure. Default None.
    transparent_background : bool, optional
        If True, save with transparent background. Default False.
    line_color : str, optional
        Color for axes, ticks, labels, grid. Default 'black'.
    text_color : str, optional
        Color for text elements. Default 'black'.
    export_format : str, optional
        File format ('png' or 'svg'). Default 'png'.
    
    Returns
    -------
    results : dict
        Dictionary with:
        - 'bimodality_scores': coefficient for each feature (>0.555 = bimodal)
        - 'recommended_features': features likely multimodal
    
    Examples
    --------
    >>> results = spt.detect_multimodality(df, features, group_by='mol')
    >>> print("Best for thresholding:", results['recommended_features'])
    
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # Clean data
    feature_df = df[features].copy()
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    results = {
        'bimodality_scores': {},
        'skewness': {},
        'kurtosis': {},
        'recommended_features': []
    }
    
    # Calculate bimodality coefficient for each feature
    # BC = (skew^2 + 1) / (kurtosis + 3*(n-1)^2/(n-2)/(n-3))
    # BC > 0.555 suggests bimodal or multimodal
    
    print("\n" + "="*80)
    print("MULTIMODALITY DETECTION")
    print("="*80)
    print("\nBimodality Coefficient (BC):")
    print("  BC > 0.555 = likely bimodal/multimodal (GOOD for thresholding)")
    print("  BC < 0.555 = likely unimodal")
    print("-"*80)
    
    for feat in features:
        data = feature_df[feat].dropna()
        
        if len(data) < 10:
            continue
        
        # Calculate statistics
        skew = stats.skew(data)
        kurt = stats.kurtosis(data, fisher=True)  # Excess kurtosis
        n = len(data)
        
        # Bimodality coefficient
        numerator = skew**2 + 1
        denominator = kurt + 3 * (n - 1)**2 / ((n - 2) * (n - 3))
        bc = numerator / denominator
        
        results['bimodality_scores'][feat] = bc
        results['skewness'][feat] = skew
        results['kurtosis'][feat] = kurt
        
        # Recommend if BC > 0.555
        if bc > 0.555:
            results['recommended_features'].append(feat)
            status = "✅ MULTIMODAL"
        else:
            status = "   unimodal"
        
        print(f"{status} | {feat:35s} | BC = {bc:.3f} | skew = {skew:+.2f} | kurt = {kurt:+.2f}")
    
    # Plot histograms for top candidates
    recommended = results['recommended_features']
    if len(recommended) > 0:
        n_plots = min(len(recommended), 9)
        n_cols = min(3, n_plots)
        n_rows = int(np.ceil(n_plots / n_cols))
        
        # Set up background
        figure_background = 'none' if transparent_background else 'white'
        axis_background = (0, 0, 0, 0) if transparent_background else 'white'
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.patch.set_facecolor(figure_background)
        
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_plots > 1 else [axes]
        
        for idx, feat in enumerate(recommended[:n_plots]):
            ax = axes[idx]
            ax.set_facecolor(axis_background)
            
            if group_by and group_by in df.columns:
                # Plot by group
                for group in sorted(df[group_by].unique()):
                    group_data = df[df[group_by] == group][feat].dropna()
                    ax.hist(group_data, bins=n_bins, alpha=0.5, label=str(group))
                legend = ax.legend()
                plt.setp(legend.get_texts(), color=text_color)
            else:
                # Single histogram
                ax.hist(feature_df[feat].dropna(), bins=n_bins, alpha=0.7, color='steelblue')
            
            bc = results['bimodality_scores'][feat]
            ax.set_title(f"{feat}\nBC = {bc:.3f}", fontsize=10, fontweight='bold', color=text_color)
            ax.set_xlabel('', color=text_color)
            ax.set_ylabel('Count', color=text_color)
            ax.grid(alpha=0.3, color=line_color)
            
            # Style spines and ticks
            for spine in ax.spines.values():
                spine.set_edgecolor(line_color)
            ax.tick_params(colors=line_color, which='both')
            plt.setp(ax.get_xticklabels(), color=text_color)
            plt.setp(ax.get_yticklabels(), color=text_color)
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle('Multimodal Feature Distributions (Top Candidates)', 
                     fontsize=14, fontweight='bold', y=1.00, color=text_color)
        plt.tight_layout()
        
        if save_path:
            save_file = f"{save_path}/multimodality_analysis.{export_format}"
            plt.savefig(save_file, dpi=300, bbox_inches='tight',
                       transparent=transparent_background)
            print(f"Saved: {save_file}")
        
        plt.show()
    
    print("="*80)
    print(f"\n✅ Found {len(recommended)} features with multimodal distributions")
    print("   These are ideal candidates for threshold-based separation!")
    print("="*80)
    
    return results


def analyze_feature_separation_power(
    df,
    features,
    reference_feature,
    group_by,
    n_top=10,
    figsize=(12, 6),
    save_path=None,
    transparent_background=False,
    line_color='black',
    text_color='black',
    export_format='png'
):
    """
    Identify which features provide the most complementary information to a reference
    feature (e.g., diffusion coefficient) for separating groups.
    
    This helps answer: "Which features should I combine with D to threshold my data?"
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data containing features
    features : list of str
        Candidate feature columns
    reference_feature : str
        Reference feature (e.g., 'D' or 'avg_D')
    group_by : str
        Column defining groups (e.g., 'mol')
    n_top : int, optional
        Number of top features to return. Default 10.
    figsize : tuple, optional
        Figure size. Default (12, 6).
    save_path : str, optional
        Path to save figure. Default None.
    transparent_background : bool, optional
        If True, save with transparent background. Default False.
    line_color : str, optional
        Color for axes, ticks, labels, grid. Default 'black'.
    text_color : str, optional
        Color for text elements. Default 'black'.
    export_format : str, optional
        File format ('png' or 'svg'). Default 'png'.
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'separation_scores': score for each feature
        - 'top_features': top N features by separation power
        - 'orthogonality_scores': how independent from reference
    
    Examples
    --------
    >>> results = spt.analyze_feature_separation_power(
    ...     df, features, reference_feature='D', group_by='mol'
    ... )
    >>> print("Best to combine with D:", results['top_features'])
    
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_score
    from scipy.stats import spearmanr
    
    # Clean data
    clean_df = df[[reference_feature, group_by] + features].copy()
    clean_df = clean_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    results = {
        'separation_scores': {},
        'orthogonality_scores': {},
        'combined_scores': {}
    }
    
    # Get group labels
    groups = clean_df[group_by].values
    
    print("\n" + "="*80)
    print(f"FEATURE SEPARATION POWER ANALYSIS")
    print(f"Reference: {reference_feature} | Groups: {group_by}")
    print("="*80)
    print("\nScoring each feature based on:")
    print("  1. How well it separates groups when combined with reference")
    print("  2. How independent it is from reference (orthogonality)")
    print("-"*80)
    
    for feat in features:
        if feat == reference_feature:
            continue
        
        # 1. Calculate separation with reference + feature (2D silhouette)
        X_2d = clean_df[[reference_feature, feat]].values
        
        try:
            sil_2d = silhouette_score(X_2d, groups, metric='euclidean')
        except:
            sil_2d = 0.0
        
        # 2. Calculate orthogonality (independence from reference)
        corr, _ = spearmanr(clean_df[reference_feature], clean_df[feat])
        orthogonality = 1 - abs(corr)  # 1 = perfectly independent, 0 = perfectly correlated
        
        # 3. Combined score (weighted)
        combined = 0.6 * sil_2d + 0.4 * orthogonality
        
        results['separation_scores'][feat] = sil_2d
        results['orthogonality_scores'][feat] = orthogonality
        results['combined_scores'][feat] = combined
        
        print(f"{feat:35s} | Sep: {sil_2d:.3f} | Orth: {orthogonality:.3f} | Combined: {combined:.3f}")
    
    # Get top features
    sorted_features = sorted(results['combined_scores'].items(), 
                            key=lambda x: x[1], reverse=True)
    results['top_features'] = [feat for feat, score in sorted_features[:n_top]]
    
    # Plot
    # Set up background
    figure_background = 'none' if transparent_background else 'white'
    axis_background = (0, 0, 0, 0) if transparent_background else 'white'
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor(figure_background)
    
    # Plot 1: Separation scores
    ax = axes[0]
    ax.set_facecolor(axis_background)
    
    sorted_sep = sorted(results['separation_scores'].items(), 
                       key=lambda x: x[1], reverse=True)[:n_top]
    features_sorted = [f for f, s in sorted_sep]
    scores_sorted = [s for f, s in sorted_sep]
    
    ax.barh(range(len(features_sorted)), scores_sorted, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(features_sorted)))
    ax.set_yticklabels(features_sorted, fontsize=9)
    ax.set_xlabel('Silhouette Score (2D)', fontsize=11, color=text_color)
    ax.set_title(f'Separation Power with {reference_feature}', fontsize=12, fontweight='bold', color=text_color)
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x', color=line_color)
    
    # Style spines and ticks
    for spine in ax.spines.values():
        spine.set_edgecolor(line_color)
    ax.tick_params(colors=line_color, which='both')
    plt.setp(ax.get_xticklabels(), color=text_color)
    plt.setp(ax.get_yticklabels(), color=text_color)
    
    # Plot 2: Combined scores
    ax = axes[1]
    ax.set_facecolor(axis_background)
    
    features_top = [f for f, s in sorted_features[:n_top]]
    scores_top = [s for f, s in sorted_features[:n_top]]
    
    ax.barh(range(len(features_top)), scores_top, color='forestgreen', alpha=0.7)
    ax.set_yticks(range(len(features_top)))
    ax.set_yticklabels(features_top, fontsize=9)
    ax.set_xlabel('Combined Score', fontsize=11, color=text_color)
    ax.set_title('Overall Ranking (Separation + Orthogonality)', fontsize=12, fontweight='bold', color=text_color)
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x', color=line_color)
    
    # Style spines and ticks
    for spine in ax.spines.values():
        spine.set_edgecolor(line_color)
    ax.tick_params(colors=line_color, which='both')
    plt.setp(ax.get_xticklabels(), color=text_color)
    plt.setp(ax.get_yticklabels(), color=text_color)
    
    plt.tight_layout()
    
    if save_path:
        save_file = f"{save_path}/separation_power_analysis.{export_format}"
        plt.savefig(save_file, dpi=300, bbox_inches='tight',
                   transparent=transparent_background)
        print(f"Saved: {save_file}")
    
    plt.show()
    
    print("="*80)
    print(f"\n✅ TOP {n_top} FEATURES TO COMBINE WITH {reference_feature}:")
    for idx, feat in enumerate(results['top_features'], 1):
        score = results['combined_scores'][feat]
        sep = results['separation_scores'][feat]
        orth = results['orthogonality_scores'][feat]
        print(f"  {idx:2d}. {feat:35s} | Score: {score:.3f} (Sep: {sep:.3f}, Orth: {orth:.3f})")
    print("="*80)
    
    return results


def analyze_pca_by_group(
    df,
    feature_columns,
    group_by,
    n_components=3,
    n_top_features=10,
    scale_data=True,
    figsize=(18, 6),
    save_path=None,
    transparent_background=False,
    line_color='black',
    text_color='black',
    export_format='png'
):
    """
    Run PCA separately for each group to identify group-specific feature contributions.
    
    This reveals whether different molecules/conditions have different dominant features
    driving their variance.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data containing features
    feature_columns : list of str
        Feature column names to include in PCA
    group_by : str
        Column name to separate groups (e.g., 'mol')
    n_components : int, optional
        Number of PCs to compute. Default 3.
    n_top_features : int, optional
        Number of top features to display per PC. Default 10.
    scale_data : bool, optional
        Whether to standardize features before PCA. Default True.
    figsize : tuple, optional
        Figure size. Default (18, 6).
    save_path : str, optional
        Path to save figure. Default None.
    transparent_background : bool, optional
        If True, save with transparent background. Default False.
    line_color : str, optional
        Color for axes, ticks, labels. Default 'black'.
    text_color : str, optional
        Color for text elements. Default 'black'.
    export_format : str, optional
        File format ('png' or 'svg'). Default 'png'.
    
    Returns
    -------
    results : dict
        Dictionary with keys for each group, containing PCA results
    
    Examples
    --------
    >>> results = spt.analyze_pca_by_group(
    ...     df, features, group_by='mol', n_components=3
    ... )
    
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Get unique groups
    groups = sorted(df[group_by].unique())
    n_groups = len(groups)
    
    results = {}
    
    print("\n" + "="*80)
    print(f"GROUP-WISE PCA ANALYSIS: {group_by}")
    print("="*80)
    
    # Run PCA for each group
    for group in groups:
        print(f"\n{'─'*80}")
        print(f"GROUP: {group}")
        print(f"{'─'*80}")
        
        # Filter data for this group
        group_df = df[df[group_by] == group][feature_columns].copy()
        group_df = group_df.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(group_df) < 10:
            print(f"  ⚠️  Skipping {group}: insufficient data (n={len(group_df)})")
            continue
        
        print(f"  Sample size: {len(group_df)}")
        
        # Scale if requested
        if scale_data:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(group_df)
        else:
            X_scaled = group_df.values
        
        # Run PCA
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        
        # Store results
        results[group] = {
            'pca': pca,
            'components': pca.components_,
            'explained_variance': pca.explained_variance_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'feature_columns': feature_columns,
            'n_samples': len(group_df)
        }
        
        # Print variance explained
        print(f"\n  Variance Explained:")
        for i, var_ratio in enumerate(pca.explained_variance_ratio_):
            cumsum = pca.explained_variance_ratio_[:i+1].sum()
            print(f"    PC{i+1}: {var_ratio*100:5.2f}%  (cumulative: {cumsum*100:5.2f}%)")
        
        # Print top features per PC
        print(f"\n  Top {n_top_features} Features per PC:")
        for pc_idx in range(n_components):
            loadings = pca.components_[pc_idx, :]
            abs_loadings = np.abs(loadings)
            top_indices = np.argsort(abs_loadings)[-n_top_features:][::-1]
            
            print(f"\n    PC{pc_idx+1}:")
            for rank, idx in enumerate(top_indices, 1):
                feat_name = feature_columns[idx]
                loading = loadings[idx]
                print(f"      {rank:2d}. {feat_name:35s} | {loading:+.3f}")
    
    # Visualization: Feature importance heatmaps for each group
    if len(results) > 0:
        # Set up background
        figure_background = 'none' if transparent_background else 'white'
        axis_background = (0, 0, 0, 0) if transparent_background else 'white'
        
        fig, axes = plt.subplots(1, n_groups, figsize=figsize)
        fig.patch.set_facecolor(figure_background)
        
        if n_groups == 1:
            axes = [axes]
        
        for ax_idx, group in enumerate(groups):
            if group not in results:
                axes[ax_idx].axis('off')
                continue
            
            ax = axes[ax_idx]
            ax.set_facecolor(axis_background)
            
            # Get loadings matrix (PCs x features)
            components = results[group]['components'][:n_components, :]
            
            # Get top features across all PCs for this group
            abs_components = np.abs(components)
            max_abs_per_feature = abs_components.max(axis=0)
            top_feature_indices = np.argsort(max_abs_per_feature)[-n_top_features:][::-1]
            
            # Create heatmap data
            heatmap_data = components[:, top_feature_indices].T  # features x PCs
            top_feature_names = [feature_columns[i] for i in top_feature_indices]
            
            # Plot heatmap
            im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', 
                          vmin=-1, vmax=1)
            
            # Set ticks
            ax.set_xticks(range(n_components))
            ax.set_xticklabels([f'PC{i+1}' for i in range(n_components)])
            ax.set_yticks(range(len(top_feature_names)))
            ax.set_yticklabels(top_feature_names, fontsize=8)
            
            # Title with variance explained
            var_expl = results[group]['explained_variance_ratio']
            title_str = f"{group}\n"
            title_str += " | ".join([f"PC{i+1}: {v*100:.1f}%" for i, v in enumerate(var_expl)])
            ax.set_title(title_str, fontsize=11, fontweight='bold', color=text_color, pad=10)
            
            # Style
            ax.tick_params(colors=line_color, which='both')
            plt.setp(ax.get_xticklabels(), color=text_color)
            plt.setp(ax.get_yticklabels(), color=text_color)
            
            for spine in ax.spines.values():
                spine.set_edgecolor(line_color)
            
            # Colorbar (only for last subplot)
            if ax_idx == n_groups - 1:
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Loading', color=text_color)
                cbar.ax.yaxis.set_tick_params(color=line_color)
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=text_color)
        
        fig.suptitle(f'Feature Loadings by {group_by} (Top {n_top_features} Features)', 
                    fontsize=14, fontweight='bold', y=0.98, color=text_color)
        plt.tight_layout()
        
        if save_path:
            save_file = f"{save_path}/pca_by_{group_by}.{export_format}"
            plt.savefig(save_file, dpi=300, bbox_inches='tight',
                       transparent=transparent_background)
            print(f"\nSaved: {save_file}")
        
        plt.show()
    
    print("\n" + "="*80)
    print("✅ Group-wise PCA complete!")
    print("="*80)
    
    return results


def plot_pc_feature_importance(pca_results, n_pcs=3, figsize=(15, 4)):
    """
    Create feature importance bar plots for each principal component separately.
    
    Shows which features have the strongest loadings on each PC, helping to
    interpret what biological/physical pattern each dimension represents.
    
    Parameters
    ----------
    pca_results : dict
        Results dictionary from analyze_feature_variance_pca()
    n_pcs : int, optional
        Number of PCs to plot. Default is 3.
    figsize : tuple, optional
        Figure size. Default is (15, 4).
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    
    Examples
    --------
    >>> pca_results = spt.analyze_feature_variance_pca(df, features)
    >>> fig = spt.plot_pc_feature_importance(pca_results, n_pcs=3)
    
    """
    import matplotlib.pyplot as plt
    
    components = pca_results['components']
    feature_names = pca_results['feature_columns']
    explained_var = pca_results['explained_variance_ratio']
    
    n_pcs = min(n_pcs, len(components))
    
    fig, axes = plt.subplots(1, n_pcs, figsize=figsize)
    if n_pcs == 1:
        axes = [axes]
    
    for pc_idx in range(n_pcs):
        ax = axes[pc_idx]
        
        # Get loadings for this PC
        loadings = components[pc_idx, :]
        
        # Sort by absolute loading
        abs_loadings = np.abs(loadings)
        sorted_indices = np.argsort(abs_loadings)
        
        # Get top features (all of them, sorted)
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_loadings = loadings[sorted_indices]
        
        # Create horizontal bar plot
        colors = ['#d62728' if l < 0 else '#2ca02c' for l in sorted_loadings]
        ax.barh(range(len(sorted_features)), sorted_loadings, color=colors, alpha=0.7)
        
        # Styling
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features, fontsize=9)
        ax.set_xlabel('Loading', fontsize=11)
        ax.set_title(f'PC{pc_idx + 1} ({explained_var[pc_idx]*100:.1f}% variance)',
                    fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(alpha=0.3, axis='x')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ca02c', alpha=0.7, label='Positive (+)'),
            Patch(facecolor='#d62728', alpha=0.7, label='Negative (-)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def cluster_pca_kmeans(
    df,
    pc_cols=['PC1', 'PC2', 'PC3'],
    n_clusters=3,
    cluster_col_name='PC_cluster',
    random_state=42
):
    """
    Perform K-means clustering on PCA coordinates and add cluster labels to dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing PC coordinates
    pc_cols : list of str, optional
        PC column names to use. Default ['PC1', 'PC2', 'PC3'].
    n_clusters : int, optional
        Number of clusters. Default 3.
    cluster_col_name : str, optional
        Name for the cluster column. Default 'PC_cluster'.
    random_state : int, optional
        Random seed for reproducibility. Default 42.
    
    Returns
    -------
    df_with_clusters : pandas.DataFrame
        Original dataframe with added cluster column
    cluster_info : dict
        Dictionary with clustering information:
        - 'kmeans': fitted KMeans object
        - 'centers': cluster centers
        - 'inertia': within-cluster sum of squares
        - 'silhouette': silhouette score
    
    Examples
    --------
    >>> df_clustered, info = spt.cluster_pca_kmeans(
    ...     df, pc_cols=['PC1', 'PC2', 'PC3'], n_clusters=3
    ... )
    >>> print(f"Silhouette score: {info['silhouette']:.3f}")
    
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import pandas as pd
    import numpy as np
    
    # Extract PC coordinates
    X = df[pc_cols].values
    
    # Remove any NaN/inf values
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
    X_clean = X[valid_mask]
    
    if len(X_clean) < n_clusters:
        raise ValueError(f"Not enough valid samples ({len(X_clean)}) for {n_clusters} clusters")
    
    print(f"\n{'='*80}")
    print(f"K-MEANS CLUSTERING ON PCA SPACE")
    print(f"{'='*80}")
    print(f"PC columns: {pc_cols}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Valid samples: {len(X_clean)} / {len(df)}")
    
    # Perform K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels_clean = kmeans.fit_predict(X_clean)
    
    # Calculate metrics
    silhouette = silhouette_score(X_clean, cluster_labels_clean)
    
    print(f"\nClustering Results:")
    print(f"  Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")
    print(f"  Silhouette score: {silhouette:.3f}")
    print(f"    (closer to 1 = better defined clusters)")
    
    # Assign labels back to original dataframe
    df_with_clusters = df.copy()
    cluster_labels_full = np.full(len(df), -1, dtype=int)  # -1 for invalid samples
    cluster_labels_full[valid_mask] = cluster_labels_clean
    df_with_clusters[cluster_col_name] = cluster_labels_full
    
    # Print cluster sizes
    print(f"\nCluster sizes:")
    for cluster_id in range(n_clusters):
        count = np.sum(cluster_labels_clean == cluster_id)
        percent = 100 * count / len(cluster_labels_clean)
        print(f"  Cluster {cluster_id}: {count:6d} ({percent:5.1f}%)")
    
    if np.sum(cluster_labels_full == -1) > 0:
        print(f"  Invalid:    {np.sum(cluster_labels_full == -1):6d} (excluded)")
    
    # Store results
    cluster_info = {
        'kmeans': kmeans,
        'centers': kmeans.cluster_centers_,
        'inertia': kmeans.inertia_,
        'silhouette': silhouette,
        'n_clusters': n_clusters,
        'pc_cols': pc_cols,
        'cluster_col': cluster_col_name
    }
    
    print(f"\n✅ Cluster labels added as column: '{cluster_col_name}'")
    print(f"{'='*80}\n")
    
    return df_with_clusters, cluster_info


def cluster_pca_interactive_box(df, pc_cols=['PC1', 'PC2', 'PC3'], color_by=None):
    """
    Interactive 3D plot with lasso selection for manual cluster definition.
    
    Uses Plotly for interactive selection. Selected points can be extracted
    for manual cluster assignment.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing PC coordinates
    pc_cols : list of str, optional
        PC column names to use. Default ['PC1', 'PC2', 'PC3'].
    color_by : str, optional
        Column to color points by. Default None.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure
    
    Notes
    -----
    After displaying the figure, use the lasso/box select tools to highlight points.
    To extract selected indices, you'll need to use plotly's selection callbacks.
    
    Examples
    --------
    >>> fig = spt.cluster_pca_interactive_box(df, color_by='mol')
    >>> fig.show()
    
    """
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd
    
    print("\n" + "="*80)
    print("INTERACTIVE 3D CLUSTER SELECTOR")
    print("="*80)
    print("Instructions:")
    print("  1. Use the 'Box Select' or 'Lasso Select' tool in the plotly toolbar")
    print("  2. Drag to select points in 3D space")
    print("  3. Selected points will be highlighted")
    print("  4. To define boxes programmatically, use the box ranges instead")
    print("="*80 + "\n")
    
    # Extract coordinates
    x = df[pc_cols[0]].values
    y = df[pc_cols[1]].values
    z = df[pc_cols[2]].values
    
    # Set up colors
    if color_by and color_by in df.columns:
        color_vals_raw = df[color_by].values
        
        # Check if categorical or numeric
        if pd.api.types.is_numeric_dtype(color_vals_raw):
            # Numeric - use as is with colorscale
            color_vals = color_vals_raw
            use_colorscale = True
            hover_text = [f"{color_by}: {val}<br>{pc_cols[0]}: {x[i]:.2f}<br>{pc_cols[1]}: {y[i]:.2f}<br>{pc_cols[2]}: {z[i]:.2f}"
                         for i, val in enumerate(color_vals_raw)]
        else:
            # Categorical - convert to numeric codes for coloring
            categories = pd.Categorical(color_vals_raw)
            color_vals = categories.codes
            use_colorscale = True
            
            # Create hover text with category names
            hover_text = [f"{color_by}: {color_vals_raw[i]}<br>{pc_cols[0]}: {x[i]:.2f}<br>{pc_cols[1]}: {y[i]:.2f}<br>{pc_cols[2]}: {z[i]:.2f}"
                         for i in range(len(x))]
    else:
        color_vals = 'steelblue'
        use_colorscale = False
        hover_text = [f"{pc_cols[0]}: {x[i]:.2f}<br>{pc_cols[1]}: {y[i]:.2f}<br>{pc_cols[2]}: {z[i]:.2f}"
                     for i in range(len(x))]
    
    # Create figure
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=color_vals,
            colorscale='Viridis' if use_colorscale else None,
            opacity=0.6,
            colorbar=dict(title=color_by) if color_by and use_colorscale else None,
            showscale=use_colorscale
        ),
        text=hover_text,
        hoverinfo='text'
    )])
    
    fig.update_layout(
        title="Interactive 3D PCA - Use Box/Lasso Select to Define Clusters",
        scene=dict(
            xaxis_title=pc_cols[0],
            yaxis_title=pc_cols[1],
            zaxis_title=pc_cols[2]
        ),
        width=1000,
        height=800
    )
    
    return fig


def define_box_clusters(df, pc_cols=['PC1', 'PC2', 'PC3'], box_definitions=None, 
                        cluster_col_name='PC_cluster'):
    """
    Define clusters by manually specifying 3D box boundaries.
    
    This is useful after visually inspecting the 3D PCA plot to manually
    define cluster regions.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing PC coordinates
    pc_cols : list of str, optional
        PC column names to use. Default ['PC1', 'PC2', 'PC3'].
    box_definitions : list of dict, optional
        List of box definitions. Each dict should have:
        - 'PC1': [min, max]
        - 'PC2': [min, max]  
        - 'PC3': [min, max]
        Example:
        [
            {'PC1': [-2, 0], 'PC2': [-1, 1], 'PC3': [-1, 1]},  # Cluster 0
            {'PC1': [0, 2], 'PC2': [-1, 1], 'PC3': [-1, 1]}    # Cluster 1
        ]
    cluster_col_name : str, optional
        Name for the cluster column. Default 'PC_cluster'.
    
    Returns
    -------
    df_with_clusters : pandas.DataFrame
        Original dataframe with added cluster column
    
    Examples
    --------
    >>> boxes = [
    ...     {'PC1': [-3, -1], 'PC2': [-2, 2], 'PC3': [-2, 2]},
    ...     {'PC1': [-1, 1], 'PC2': [-2, 2], 'PC3': [-2, 2]},
    ...     {'PC1': [1, 3], 'PC2': [-2, 2], 'PC3': [-2, 2]}
    ... ]
    >>> df_clustered = spt.define_box_clusters(df, box_definitions=boxes)
    
    """
    import numpy as np
    
    if box_definitions is None:
        raise ValueError("Must provide box_definitions. See docstring for format.")
    
    df_with_clusters = df.copy()
    cluster_labels = np.full(len(df), -1, dtype=int)  # -1 = unassigned
    
    print(f"\n{'='*80}")
    print(f"MANUAL BOX-BASED CLUSTERING")
    print(f"{'='*80}")
    print(f"Number of boxes: {len(box_definitions)}")
    
    for cluster_id, box in enumerate(box_definitions):
        # Create mask for points within this box
        mask = np.ones(len(df), dtype=bool)
        
        for pc_col in pc_cols:
            if pc_col in box:
                pc_min, pc_max = box[pc_col]
                pc_vals = df[pc_col].values
                mask &= (pc_vals >= pc_min) & (pc_vals <= pc_max)
        
        # Assign cluster label
        cluster_labels[mask] = cluster_id
        n_in_cluster = np.sum(mask)
        
        print(f"\nCluster {cluster_id}:")
        for pc_col in pc_cols:
            if pc_col in box:
                print(f"  {pc_col}: [{box[pc_col][0]:.2f}, {box[pc_col][1]:.2f}]")
        print(f"  → {n_in_cluster} points assigned")
    
    # Handle unassigned points
    n_unassigned = np.sum(cluster_labels == -1)
    if n_unassigned > 0:
        print(f"\n⚠️  {n_unassigned} points not assigned to any cluster (label = -1)")
    
    df_with_clusters[cluster_col_name] = cluster_labels
    
    print(f"\n✅ Cluster labels added as column: '{cluster_col_name}'")
    print(f"{'='*80}\n")
    
    return df_with_clusters


def plot_pca_3d(
    df,
    pc_cols=['PC1', 'PC2', 'PC3'],
    color_by=None,
    small_multiples=False,
    density_mode=None,
    figsize=(10, 8),
    title=None,
    alpha=0.6,
    s=20,
    save_path=None,
    show_plot=True,
    elev=30,
    azim=45,
    contour_levels=5,
    density_bins=20,
    xlim=None,
    ylim=None,
    zlim=None,
    transparent_background=False,
    line_color='black',
    text_color='black',
    export_format='png',
    contour_cmap='viridis',
    cluster_centers=None,
    draw_hulls=False,
    hull_alpha=0.1,
    threshold_lines=None
):
    """
    Create 3D scatter plots of principal components with optional density visualization.
    
    Visualize tracks in PC space to identify natural clusters and behavioral subtypes.
    Supports density-based visualization to highlight high-density regions.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing PC scores (columns PC1, PC2, PC3) and optional grouping variables.
    pc_cols : list of str, optional
        Column names for the three PCs to plot. Default is ['PC1', 'PC2', 'PC3'].
    color_by : str, optional
        Column name to use for color coding points. If None, all points same color.
        Default is None.
    small_multiples : bool, optional
        If True and color_by is specified, creates separate 3D plots for each category.
        If False, plots all categories in one 3D plot with different colors.
        Default is False.
    density_mode : str or None, optional
        How to visualize density in 3D space:
        - None: Standard scatter plot (default)
        - 'size': Point size scales with local density
        - 'alpha': Point transparency scales with local density
        - 'contour': Add 3D isosurface contours at density levels
        Default is None.
    figsize : tuple, optional
        Figure size. Default is (10, 8).
    title : str, optional
        Overall title for the plot(s). Default is None.
    alpha : float, optional
        Point transparency (0-1). Ignored if density_mode='alpha'. Default is 0.6.
    s : int, optional
        Base point size. Scales with density if density_mode='size'. Default is 20.
    save_path : str, optional
        Path to save the plot (PNG format). Default is None (no save).
    show_plot : bool, optional
        Whether to display the plot. Default is True.
    elev : float, optional
        Elevation angle for 3D view. Default is 30.
    azim : float, optional
        Azimuth angle for 3D view. Default is 45.
    contour_levels : int, optional
        Number of density contour levels if density_mode='contour'. Default is 5.
    density_bins : int, optional
        Number of bins per dimension for density calculation. Default is 20.
    xlim : tuple, optional
        X-axis limits (xmin, xmax). Default is None (auto).
    ylim : tuple, optional
        Y-axis limits (ymin, ymax). Default is None (auto).
    zlim : tuple, optional
        Z-axis limits (zmin, zmax). Default is None (auto).
    transparent_background : bool, optional
        If True, makes the background transparent. Useful for presentations
        and publications. Default is False.
    line_color : str, optional
        Color for axis lines, ticks, and grid. Default is 'black'.
        Use 'white' for dark backgrounds in presentations.
    text_color : str, optional
        Color for axis labels and title. Default is 'black'.
        Use 'white' for dark backgrounds in presentations.
    export_format : str, optional
        File format to export ('png' or 'svg'). Default is 'png'.
    contour_cmap : str, optional
        Colormap for density contours when density_mode='contour'.
        Default is 'viridis'. Options: 'viridis', 'plasma', 'inferno', 'magma', etc.
    cluster_centers : numpy.ndarray, optional
        Array of cluster centers (N x 3) to plot as large red stars.
        Useful for visualizing K-means cluster centers. Default is None.
    draw_hulls : bool, optional
        If True and color_by is specified, draws 3D convex hulls around each cluster.
        Makes cluster boundaries crystal clear. Default is False.
    hull_alpha : float, optional
        Transparency of convex hull surfaces (0-1). Default is 0.1.
    threshold_lines : dict, optional
        Dictionary with PC names as keys and threshold values as values.
        Draws dotted planes/edges at these thresholds to show octant boundaries.
        Example: {'PC1': 0.5, 'PC2': -0.3, 'PC3': 1.2}
        Default is None (no threshold lines).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    
    Examples
    --------
    >>> # Standard scatter plot colored by molecule
    >>> fig = spt.plot_pca_3d(pca_df, color_by='mol')
    
    >>> # Density visualization with point sizes
    >>> fig = spt.plot_pca_3d(pca_df, color_by='mol', density_mode='size')
    
    >>> # 3D contour surfaces showing density
    >>> fig = spt.plot_pca_3d(pca_df, density_mode='contour', contour_levels=3)
    
    >>> # Publication-ready with transparent background and white text
    >>> fig = spt.plot_pca_3d(pca_df, color_by='mol', transparent_background=True,
    ...                       line_color='white', text_color='white', export_format='svg',
    ...                       save_path='./plots/')
    
    >>> # Small multiples by condition
    >>> fig = spt.plot_pca_3d(pca_df, color_by='condition', small_multiples=True)
    
    >>> # With threshold boundaries to show octants
    >>> thresholds = {'PC1': 0.5, 'PC2': -0.3, 'PC3': 1.2}
    >>> fig = spt.plot_pca_3d(pca_df, color_by='PC_cluster', threshold_lines=thresholds)
    
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.stats import gaussian_kde
    from scipy.ndimage import gaussian_filter
    
    # Check if PC columns exist
    missing_cols = [col for col in pc_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"PC columns not found in dataframe: {missing_cols}")
    
    # Remove NaN values
    plot_df = df[pc_cols + ([color_by] if color_by else [])].dropna()
    
    def calculate_local_density(x, y, z, bins=20):
        """Calculate local density for each point using 3D histogram."""
        # Create 3D histogram
        H, edges = np.histogramdd(
            np.vstack([x, y, z]).T,
            bins=bins
        )
        
        # Smooth the histogram
        H_smooth = gaussian_filter(H, sigma=1.0)
        
        # Find which bin each point belongs to
        x_idx = np.digitize(x, edges[0]) - 1
        y_idx = np.digitize(y, edges[1]) - 1
        z_idx = np.digitize(z, edges[2]) - 1
        
        # Clip to valid range
        x_idx = np.clip(x_idx, 0, bins - 1)
        y_idx = np.clip(y_idx, 0, bins - 1)
        z_idx = np.clip(z_idx, 0, bins - 1)
        
        # Get density for each point
        densities = H_smooth[x_idx, y_idx, z_idx]
        
        return densities
    
    def draw_convex_hull(ax, points, color, alpha_hull=0.1):
        """Draw 3D convex hull around a set of points."""
        from scipy.spatial import ConvexHull
        
        if len(points) < 4:
            # Need at least 4 points for a 3D convex hull
            return
        
        try:
            hull = ConvexHull(points)
            
            # Draw the hull faces
            for simplex in hull.simplices:
                triangle = points[simplex]
                # Create a triangular surface
                ax.plot_trisurf(
                    triangle[:, 0], triangle[:, 1], triangle[:, 2],
                    color=color, alpha=alpha_hull, shade=True,
                    edgecolor=color, linewidth=0.5
                )
        except Exception as e:
            # Convex hull can fail for degenerate cases
            print(f"Warning: Could not draw convex hull: {e}")
    
    def add_density_contours(ax, x, y, z, levels=5, bins=20, alpha_contour=0.3, cmap_name='viridis'):
        """Add 3D isosurface contours showing density."""
        from skimage import measure
        
        # Create 3D histogram
        H, edges = np.histogramdd(
            np.vstack([x, y, z]).T,
            bins=bins
        )
        
        # Smooth the histogram
        H_smooth = gaussian_filter(H, sigma=1.5)
        
        # Normalize
        H_smooth = H_smooth / H_smooth.max()
        
        # Create meshgrid for the histogram
        x_centers = (edges[0][:-1] + edges[0][1:]) / 2
        y_centers = (edges[1][:-1] + edges[1][1:]) / 2
        z_centers = (edges[2][:-1] + edges[2][1:]) / 2
        
        # Get colormap
        cmap = plt.cm.get_cmap(cmap_name)
        
        # Draw isosurfaces at different density levels
        density_levels = np.linspace(H_smooth.max() * 0.2, H_smooth.max() * 0.8, levels)
        
        for level_idx, level in enumerate(density_levels):
            try:
                # Use marching cubes to find isosurface
                verts, faces, _, _ = measure.marching_cubes(H_smooth, level=level)
                
                # Scale vertices to actual data coordinates
                verts_scaled = np.zeros_like(verts)
                verts_scaled[:, 0] = x_centers[0] + verts[:, 0] * (x_centers[-1] - x_centers[0]) / bins
                verts_scaled[:, 1] = y_centers[0] + verts[:, 1] * (y_centers[-1] - y_centers[0]) / bins
                verts_scaled[:, 2] = z_centers[0] + verts[:, 2] * (z_centers[-1] - z_centers[0]) / bins
                
                # Plot the surface
                ax.plot_trisurf(
                    verts_scaled[:, 0],
                    verts_scaled[:, 1],
                    faces,
                    verts_scaled[:, 2],
                    alpha=alpha_contour,
                    color=cmap(level_idx / levels),
                    shade=True,
                    linewidth=0
                )
            except (ValueError, RuntimeError):
                # Skip if marching cubes fails
                pass
    
    def draw_threshold_boundaries(ax, thresholds_dict, pc_cols_list, data_ranges, line_color_val='gray'):
        """Draw wireframe cube boundaries at threshold values."""
        t1 = thresholds_dict[pc_cols_list[0]]
        t2 = thresholds_dict[pc_cols_list[1]]
        t3 = thresholds_dict[pc_cols_list[2]]
        
        # Get data ranges for drawing planes
        x_range = data_ranges[0]
        y_range = data_ranges[1]
        z_range = data_ranges[2]
        
        # Draw 3 planes (one per threshold)
        # PC1 threshold plane (vertical plane perpendicular to x-axis)
        y_plane = np.linspace(y_range[0], y_range[1], 10)
        z_plane = np.linspace(z_range[0], z_range[1], 10)
        Y_plane, Z_plane = np.meshgrid(y_plane, z_plane)
        X_plane = np.full_like(Y_plane, t1)
        ax.plot_wireframe(X_plane, Y_plane, Z_plane, color=line_color_val, 
                         alpha=0.3, linestyle='--', linewidth=1)
        
        # PC2 threshold plane (vertical plane perpendicular to y-axis)
        x_plane = np.linspace(x_range[0], x_range[1], 10)
        z_plane = np.linspace(z_range[0], z_range[1], 10)
        X_plane, Z_plane = np.meshgrid(x_plane, z_plane)
        Y_plane = np.full_like(X_plane, t2)
        ax.plot_wireframe(X_plane, Y_plane, Z_plane, color=line_color_val, 
                         alpha=0.3, linestyle='--', linewidth=1)
        
        # PC3 threshold plane (horizontal plane perpendicular to z-axis)
        x_plane = np.linspace(x_range[0], x_range[1], 10)
        y_plane = np.linspace(y_range[0], y_range[1], 10)
        X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
        Z_plane = np.full_like(X_plane, t3)
        ax.plot_wireframe(X_plane, Y_plane, Z_plane, color=line_color_val, 
                         alpha=0.3, linestyle='--', linewidth=1)
    
    # Set backgrounds
    figure_background = 'none' if transparent_background else 'white'
    axis_background = (0, 0, 0, 0) if transparent_background else 'white'
    
    # MATPLOTLIB VERSION (Static)
    if color_by is not None and small_multiples:
        # Small multiples
        categories = sorted(plot_df[color_by].unique())
        n_cats = len(categories)
        
        n_cols = min(3, n_cats)
        n_rows = int(np.ceil(n_cats / n_cols))
        
        fig = plt.figure(figsize=(figsize[0] * n_cols, figsize[1] * n_rows))
        fig.patch.set_facecolor(figure_background)
        
        for idx, cat in enumerate(categories):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
            ax.set_facecolor(axis_background)
            cat_df = plot_df[plot_df[color_by] == cat]
            
            x = cat_df[pc_cols[0]].values
            y = cat_df[pc_cols[1]].values
            z = cat_df[pc_cols[2]].values
            
            # Apply density visualization if requested
            if density_mode == 'size':
                densities = calculate_local_density(x, y, z, bins=density_bins)
                sizes = s + (densities / densities.max()) * s * 3
                ax.scatter(x, y, z, s=sizes, alpha=alpha)
            elif density_mode == 'alpha':
                densities = calculate_local_density(x, y, z, bins=density_bins)
                alphas = 0.2 + (densities / densities.max()) * 0.8
                ax.scatter(x, y, z, s=s, c=densities, cmap='viridis', alpha=alphas)
            elif density_mode == 'contour':
                ax.scatter(x, y, z, s=s, alpha=alpha)
                add_density_contours(ax, x, y, z, levels=contour_levels, bins=density_bins, cmap_name=contour_cmap)
            else:
                ax.scatter(x, y, z, s=s, alpha=alpha)
            
            ax.set_xlabel(pc_cols[0], fontsize=10, color=text_color)
            ax.set_ylabel(pc_cols[1], fontsize=10, color=text_color)
            ax.set_zlabel(pc_cols[2], fontsize=10, color=text_color)
            ax.set_title(f"{color_by}: {cat}", fontsize=11, fontweight='bold', color=text_color)
            ax.view_init(elev=elev, azim=azim)
            
            # Set axis limits if provided
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            if zlim is not None:
                ax.set_zlim(zlim)
            
            # Style axes
            ax.tick_params(colors=line_color, labelsize=8)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor(line_color)
            ax.yaxis.pane.set_edgecolor(line_color)
            ax.zaxis.pane.set_edgecolor(line_color)
            ax.grid(alpha=0.3, color=line_color)
            
            # Draw threshold boundaries if provided
            if threshold_lines is not None:
                data_ranges = [
                    (cat_df[pc_cols[0]].min(), cat_df[pc_cols[0]].max()),
                    (cat_df[pc_cols[1]].min(), cat_df[pc_cols[1]].max()),
                    (cat_df[pc_cols[2]].min(), cat_df[pc_cols[2]].max())
                ]
                draw_threshold_boundaries(ax, threshold_lines, pc_cols, data_ranges, 
                                        line_color_val=line_color)
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', color=text_color)
            
    else:
        # Single plot
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor(figure_background)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(axis_background)
        
        x = plot_df[pc_cols[0]].values
        y = plot_df[pc_cols[1]].values
        z = plot_df[pc_cols[2]].values
        
        if color_by is not None and color_by in plot_df.columns:
            # Color by category using colorblind-friendly palette
            categories = sorted(plot_df[color_by].unique())
            
            # Colorblind-friendly palette (Wong 2011 + extras)
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
            
            # Use colorblind palette, cycle if needed
            colors = [colorblind_colors[i % len(colorblind_colors)] for i in range(len(categories))]
            
            for cat_idx, (cat, color) in enumerate(zip(categories, colors)):
                cat_df = plot_df[plot_df[color_by] == cat]
                cat_x = cat_df[pc_cols[0]].values
                cat_y = cat_df[pc_cols[1]].values
                cat_z = cat_df[pc_cols[2]].values
                
                # Apply density visualization if requested
                if density_mode == 'size':
                    densities = calculate_local_density(cat_x, cat_y, cat_z, bins=density_bins)
                    sizes = s + (densities / densities.max()) * s * 3
                    ax.scatter(cat_x, cat_y, cat_z, s=sizes, alpha=alpha, label=str(cat), c=color)
                elif density_mode == 'alpha':
                    densities = calculate_local_density(cat_x, cat_y, cat_z, bins=density_bins)
                    alphas = 0.2 + (densities / densities.max()) * 0.8
                    for i in range(len(cat_x)):
                        ax.scatter(cat_x[i], cat_y[i], cat_z[i], s=s, alpha=alphas[i], c=color)
                    # Add one point for legend
                    ax.scatter([], [], [], s=s, alpha=alpha, label=str(cat), c=color)
                elif density_mode == 'contour':
                    ax.scatter(cat_x, cat_y, cat_z, s=s, alpha=alpha, label=str(cat), c=color)
                else:
                    ax.scatter(cat_x, cat_y, cat_z, s=s, alpha=alpha, label=str(cat), c=color)
                
                # Draw convex hull around this cluster if requested
                if draw_hulls and len(cat_x) >= 4:
                    points = np.column_stack([cat_x, cat_y, cat_z])
                    draw_convex_hull(ax, points, color, alpha_hull=hull_alpha)
            
            # Add contours for all data if requested
            if density_mode == 'contour':
                add_density_contours(ax, x, y, z, levels=contour_levels, 
                                   bins=density_bins, cmap_name=contour_cmap)
            
            ax.legend(title=color_by, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Single color
            if density_mode == 'size':
                densities = calculate_local_density(x, y, z, bins=density_bins)
                sizes = s + (densities / densities.max()) * s * 3
                ax.scatter(x, y, z, s=sizes, alpha=alpha, c='steelblue')
            elif density_mode == 'alpha':
                densities = calculate_local_density(x, y, z, bins=density_bins)
                alphas = 0.2 + (densities / densities.max()) * 0.8
                scatter = ax.scatter(x, y, z, s=s, c=densities, cmap='viridis', alpha=alphas)
                plt.colorbar(scatter, ax=ax, label='Density', shrink=0.5, pad=0.1)
            elif density_mode == 'contour':
                ax.scatter(x, y, z, s=s, alpha=alpha, c='steelblue')
                # Note: contour_per_cluster only works with color_by specified
                add_density_contours(ax, x, y, z, levels=contour_levels, bins=density_bins, cmap_name=contour_cmap)
            else:
                ax.scatter(x, y, z, s=s, alpha=alpha, c='steelblue')
        
        # Plot cluster centers if provided
        if cluster_centers is not None:
            ax.scatter(cluster_centers[:, 0], 
                      cluster_centers[:, 1], 
                      cluster_centers[:, 2],
                      c='red', s=200, alpha=1.0, marker='*', 
                      edgecolors='black', linewidths=2,
                      label='Cluster Centers', zorder=1000)
        
        ax.set_xlabel(pc_cols[0], fontsize=11, color=text_color)
        ax.set_ylabel(pc_cols[1], fontsize=11, color=text_color)
        ax.set_zlabel(pc_cols[2], fontsize=11, color=text_color)
        ax.set_title(title if title else f'3D PCA: {pc_cols[0]} vs {pc_cols[1]} vs {pc_cols[2]}',
                    fontsize=12, fontweight='bold', color=text_color)
        ax.view_init(elev=elev, azim=azim)
        
        # Set axis limits if provided
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if zlim is not None:
            ax.set_zlim(zlim)
        
        # Style axes
        ax.tick_params(colors=line_color)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(line_color)
        ax.yaxis.pane.set_edgecolor(line_color)
        ax.zaxis.pane.set_edgecolor(line_color)
        ax.grid(alpha=0.3, color=line_color)
        
        # Draw threshold boundaries if provided
        if threshold_lines is not None:
            data_ranges = [
                (plot_df[pc_cols[0]].min(), plot_df[pc_cols[0]].max()),
                (plot_df[pc_cols[1]].min(), plot_df[pc_cols[1]].max()),
                (plot_df[pc_cols[2]].min(), plot_df[pc_cols[2]].max())
            ]
            draw_threshold_boundaries(ax, threshold_lines, pc_cols, data_ranges, 
                                    line_color_val=line_color)
    
    plt.tight_layout()
    
    if save_path:
        ext = export_format.lower()
        if ext not in ['png', 'svg']:
            print("Invalid export format specified. Defaulting to 'png'.")
            ext = 'png'
        
        # Check if save_path is a directory or a full file path
        if os.path.isdir(save_path) or (not os.path.exists(save_path) and not save_path.endswith(('.png', '.svg'))):
            # It's a directory - generate filename automatically
            save_dir = save_path
            
            # Build descriptive filename
            filename_parts = ['pca_3d', f"{pc_cols[0]}_{pc_cols[1]}_{pc_cols[2]}"]
            if color_by is not None:
                filename_parts.append(f"by_{color_by}")
            if small_multiples:
                filename_parts.append("multiples")
            if density_mode:
                filename_parts.append(density_mode)
            filename_parts.append(f"elev{elev}_azim{azim}")
            
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
        
        plt.savefig(
            save_path,
            transparent=transparent_background,
            dpi=150,
            format=ext,
            bbox_inches='tight'
        )
        
        # SVG post-processing
        if ext == 'svg':
            try:
                import re
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
        
        print(f"3D plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def select_pc_thresholds_interactive(
    df,
    pc_cols=['PC1', 'PC2', 'PC3'],
    color_by='mol',
    n_bins=50,
    continuous_update=False
):
    """
    Interactively select threshold values for each PC using ipywidgets sliders.
    
    Creates interactive sliders with checkboxes. Uncheck to skip threshold on a PC.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing PC coordinates
    pc_cols : list of str, optional
        PC column names. Default ['PC1', 'PC2', 'PC3'].
    color_by : str, optional
        Column to separate histograms by. Default 'mol'.
    n_bins : int, optional
        Number of histogram bins. Default 50.
    continuous_update : bool, optional
        If True, updates plot while dragging (creates duplicates but more interactive).
        If False, updates only when releasing slider (cleaner but less responsive).
        Default False.
    
    Returns
    -------
    widget_container : object
        Interactive widget container with sliders, checkboxes, and pc_cols attributes.
    
    Examples
    --------
    >>> # Create widget (updates on release - clean)
    >>> widget = spt.select_pc_thresholds_interactive(pca_df, color_by='mol')
    
    >>> # More responsive (updates while dragging - may duplicate)
    >>> widget = spt.select_pc_thresholds_interactive(pca_df, continuous_update=True)
    
    >>> # After adjusting, get thresholds:
    >>> thresholds = spt.extract_thresholds_from_widget(widget)
    
    Notes
    -----
    - Requires ipywidgets: pip install ipywidgets
    - Drag sliders to adjust thresholds
    - UNCHECK "Use threshold" box to skip that PC (use full range)
    - Red line shows active threshold
    
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from ipywidgets import interact, FloatSlider, Checkbox
    import numpy as np
    
    print("\n" + "="*80)
    print("INTERACTIVE PC THRESHOLD SELECTION")
    print("="*80)
    print("Instructions:")
    print("  1. Drag sliders to adjust thresholds")
    print("  2. UNCHECK 'Use threshold' to skip that PC (use full range)")
    print("  3. Red line = active threshold")
    print(f"  4. Update mode: {'CONTINUOUS (while dragging)' if continuous_update else 'ON RELEASE (cleaner)'}")
    print("  5. After adjusting, run: thresholds = extract_thresholds_from_widget(widget)")
    print("="*80 + "\n")
    
    # Store data and ranges
    pc_data = {}
    pc_ranges = {}
    categories = None
    
    for pc_col in pc_cols:
        pc_data[pc_col] = df[pc_col].dropna()
        pc_ranges[pc_col] = (pc_data[pc_col].min(), pc_data[pc_col].max())
    
    if color_by and color_by in df.columns:
        categories = sorted(df[color_by].unique())
    
    # Create sliders and checkboxes
    sliders = {}
    checkboxes = {}
    
    for pc_col in pc_cols:
        sliders[pc_col] = FloatSlider(
            value=pc_data[pc_col].median(),
            min=pc_ranges[pc_col][0],
            max=pc_ranges[pc_col][1],
            step=(pc_ranges[pc_col][1] - pc_ranges[pc_col][0]) / 200,
            description=f'{pc_col}:',
            continuous_update=continuous_update,
            readout_format='.3f',
            style={'description_width': '80px'},
            layout={'width': '500px'}
        )
        
        checkboxes[pc_col] = Checkbox(
            value=True,
            description='Use threshold',
            style={'description_width': 'initial'},
            layout={'width': '150px'}
        )
    
    # Create update function
    def update_plot(PC1_threshold, PC1_use, PC2_threshold, PC2_use, PC3_threshold, PC3_use):
        # Get current values
        threshold_vals = [PC1_threshold, PC2_threshold, PC3_threshold]
        use_flags = [PC1_use, PC2_use, PC3_use]
        
        thresholds = {}
        for pc_col, threshold_val, use_flag in zip(pc_cols, threshold_vals, use_flags):
            if use_flag:
                thresholds[pc_col] = threshold_val
            else:
                thresholds[pc_col] = None
        
        # Create subplots
        titles = []
        for pc_col in pc_cols:
            if thresholds[pc_col] is not None:
                titles.append(f'{pc_col} (threshold: {thresholds[pc_col]:.3f})')
            else:
                titles.append(f'{pc_col} (Full range - no threshold)')
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=titles,
            horizontal_spacing=0.12
        )
        
        # Plot histograms
        for idx, pc_col in enumerate(pc_cols, 1):
            if categories:
                for cat in categories:
                    cat_data = df[df[color_by] == cat][pc_col].dropna()
                    fig.add_trace(
                        go.Histogram(
                            x=cat_data,
                            nbinsx=n_bins,
                            name=str(cat),
                            opacity=0.6,
                            showlegend=(idx == 1)
                        ),
                        row=1, col=idx
                    )
            else:
                fig.add_trace(
                    go.Histogram(
                        x=pc_data[pc_col],
                        nbinsx=n_bins,
                        name=pc_col,
                        opacity=0.7,
                        marker_color='steelblue',
                        showlegend=False
                    ),
                    row=1, col=idx
                )
            
            # Add threshold line if active
            if thresholds[pc_col] is not None:
                fig.add_vline(
                    x=thresholds[pc_col],
                    line_dash="dash",
                    line_color="red",
                    line_width=3,
                    row=1, col=idx
                )
            
            # Update axes
            fig.update_xaxes(title_text=pc_col, row=1, col=idx)
            fig.update_yaxes(title_text="Count", row=1, col=idx)
        
        # Update layout
        fig.update_layout(
            height=450,
            showlegend=True,
            barmode='overlay',
            title_text="Interactive PC Threshold Selection",
            title_font_size=13
        )
        
        fig.show()
    
    # Create interactive widget
    widget = interact(
        update_plot,
        PC1_threshold=sliders[pc_cols[0]],
        PC1_use=checkboxes[pc_cols[0]],
        PC2_threshold=sliders[pc_cols[1]],
        PC2_use=checkboxes[pc_cols[1]],
        PC3_threshold=sliders[pc_cols[2]],
        PC3_use=checkboxes[pc_cols[2]]
    )
    
    # Store sliders and checkboxes for retrieval
    widget.sliders = sliders
    widget.checkboxes = checkboxes
    widget.pc_cols = pc_cols
    
    print("✅ Widget created! Drag sliders and check/uncheck boxes.")
    print("   After adjusting, run: thresholds = extract_thresholds_from_widget(widget)\n")
    
    return widget


def extract_thresholds_from_widget(widget):
    """
    Extract threshold values from the interactive widget after adjustments.
    
    Parameters
    ----------
    widget : ipywidgets.interact
        Widget returned by select_pc_thresholds_interactive()
    
    Returns
    -------
    thresholds : dict
        Dictionary with PC names as keys and threshold values (or None if unchecked).
    
    Examples
    --------
    >>> widget = spt.select_pc_thresholds_interactive(pca_df)
    >>> # ... adjust sliders and checkboxes ...
    >>> thresholds = spt.extract_thresholds_from_widget(widget)
    
    """
    # Get values from sliders and checkboxes (stored as dicts)
    thresholds = {}
    for pc_col in widget.pc_cols:
        if widget.checkboxes[pc_col].value:
            thresholds[pc_col] = widget.sliders[pc_col].value
        else:
            thresholds[pc_col] = None
    
    print("\n" + "="*80)
    print("EXTRACTED THRESHOLDS:")
    for pc, val in thresholds.items():
        if val is not None:
            print(f"  {pc}: {val:.3f}")
        else:
            print(f"  {pc}: None (full range - no threshold)")
    print("="*80 + "\n")
    
    return thresholds


def assign_octant_clusters(
    df,
    pc_cols=['PC1', 'PC2', 'PC3'],
    thresholds=None,
    octant_to_cluster_map=None,
    cluster_col_name='PC_clusters_manual'
):
    """
    Assign octant IDs (0-7) based on PC thresholds, add to dataframe as clusters.
    
    Each threshold splits a PC into two regions:
    - Low: [min_PC, threshold)
    - High: [threshold, max_PC]
    
    With 3 PCs, this creates 8 octants (2^3 combinations of low/high).
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing PC coordinates
    pc_cols : list of str, optional
        PC column names. Default ['PC1', 'PC2', 'PC3'].
    thresholds : dict
        Dictionary with PC names as keys and threshold values as values.
        Example: {'PC1': 0.5, 'PC2': -0.3, 'PC3': 1.2}
    octant_to_cluster_map : dict, optional
        Maps octant IDs (0-7) to cluster IDs. If None, octant ID is used directly.
        Example: {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2}
        This groups octants into fewer clusters.
    cluster_col_name : str, optional
        Name for the cluster column. Default 'PC_clusters_manual'.
    
    Returns
    -------
    df_with_clusters : pandas.DataFrame
        Original dataframe with added cluster column
    
    Examples
    --------
    >>> # Simple: 8 octants as 8 clusters
    >>> thresholds = {'PC1': 0.5, 'PC2': -0.3, 'PC3': 1.2}
    >>> df_clustered = spt.assign_octant_clusters(pca_df, thresholds=thresholds)
    
    >>> # Advanced: Map 8 octants to 3 clusters
    >>> octant_map = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2}
    >>> df_clustered = spt.assign_octant_clusters(
    ...     pca_df, thresholds=thresholds, octant_to_cluster_map=octant_map
    ... )
    
    Notes
    -----
    Octant encoding (binary, where 1=high, 0=low):
    - Octant 0 (000): PC1 low, PC2 low, PC3 low
    - Octant 1 (001): PC1 low, PC2 low, PC3 high
    - Octant 2 (010): PC1 low, PC2 high, PC3 low
    - ... up to Octant 7 (111): all high
    
    """
    import pandas as pd
    import numpy as np
    
    if thresholds is None:
        raise ValueError("Must provide thresholds dictionary")
    
    df_with_clusters = df.copy()
    
    # Extract threshold values
    t1 = thresholds[pc_cols[0]]
    t2 = thresholds[pc_cols[1]]
    t3 = thresholds[pc_cols[2]]
    
    print("\n" + "="*80)
    print("ASSIGNING PC-BASED CLUSTERS")
    print("="*80)
    print(f"Thresholds:")
    
    # Display thresholds (handle None)
    if t1 is not None:
        print(f"  {pc_cols[0]}: {t1:.3f}  [Low: [{df[pc_cols[0]].min():.3f}, {t1:.3f}), High: [{t1:.3f}, {df[pc_cols[0]].max():.3f}]]")
    else:
        print(f"  {pc_cols[0]}: None (full range - no split)")
    
    if t2 is not None:
        print(f"  {pc_cols[1]}: {t2:.3f}  [Low: [{df[pc_cols[1]].min():.3f}, {t2:.3f}), High: [{t2:.3f}, {df[pc_cols[1]].max():.3f}]]")
    else:
        print(f"  {pc_cols[1]}: None (full range - no split)")
    
    if t3 is not None:
        print(f"  {pc_cols[2]}: {t3:.3f}  [Low: [{df[pc_cols[2]].min():.3f}, {t3:.3f}), High: [{t3:.3f}, {df[pc_cols[2]].max():.3f}]]")
    else:
        print(f"  {pc_cols[2]}: None (full range - no split)")
    
    print("-"*80)
    
    # Compute octant ID for each point (handling None thresholds)
    # If threshold is None, treat all as "low" (0) - no split
    if t1 is not None:
        pc1_high = (df[pc_cols[0]] >= t1).astype(int)
    else:
        pc1_high = np.zeros(len(df), dtype=int)
    
    if t2 is not None:
        pc2_high = (df[pc_cols[1]] >= t2).astype(int)
    else:
        pc2_high = np.zeros(len(df), dtype=int)
    
    if t3 is not None:
        pc3_high = (df[pc_cols[2]] >= t3).astype(int)
    else:
        pc3_high = np.zeros(len(df), dtype=int)
    
    octant_ids = pc1_high * 4 + pc2_high * 2 + pc3_high * 1
    
    # Calculate actual number of clusters based on non-None thresholds
    n_active_thresholds = sum([t1 is not None, t2 is not None, t3 is not None])
    n_actual_octants = 2 ** n_active_thresholds
    print(f"\nActive thresholds: {n_active_thresholds}")
    print(f"Number of resulting regions: {n_actual_octants}")
    
    # Map to clusters if provided, otherwise use octant IDs directly
    if octant_to_cluster_map is not None:
        cluster_ids = octant_ids.map(octant_to_cluster_map)
        df_with_clusters[cluster_col_name] = cluster_ids
        
        print("\n8 Octants mapped to clusters:")
        for oct_id in range(8):
            if oct_id in octant_to_cluster_map:
                cluster = octant_to_cluster_map[oct_id]
                count = np.sum(octant_ids == oct_id)
                percent = 100 * count / len(df) if len(df) > 0 else 0
                
                # Decode octant
                p1_status = "high" if (oct_id & 4) else "low"
                p2_status = "high" if (oct_id & 2) else "low"
                p3_status = "high" if (oct_id & 1) else "low"
                
                print(f"  Octant {oct_id} ({p1_status}/{p2_status}/{p3_status}) → Cluster {cluster} | {count:6d} points ({percent:5.1f}%)")
        
        # Cluster summary
        print("\nFinal Cluster Distribution:")
        for cluster_id in sorted(df_with_clusters[cluster_col_name].unique()):
            count = np.sum(df_with_clusters[cluster_col_name] == cluster_id)
            percent = 100 * count / len(df_with_clusters)
            print(f"  Cluster {cluster_id}: {count:6d} points ({percent:5.1f}%)")
    else:
        # No mapping - use octant IDs as cluster IDs, but skip empty octants
        df_with_clusters[cluster_col_name] = octant_ids
        
        print("\nOctants before removing empty ones:")
        non_empty_octants = []
        for oct_id in range(8):
            count = np.sum(octant_ids == oct_id)
            percent = 100 * count / len(df) if len(df) > 0 else 0
            
            # Decode octant
            p1_status = "high" if (oct_id & 4) else "low"
            p2_status = "high" if (oct_id & 2) else "low"
            p3_status = "high" if (oct_id & 1) else "low"
            
            if count > 0:
                non_empty_octants.append(oct_id)
                print(f"  ✓ Octant {oct_id} ({p1_status}/{p2_status}/{p3_status}) | {count:6d} points ({percent:5.1f}%)")
            else:
                print(f"  ✗ Octant {oct_id} ({p1_status}/{p2_status}/{p3_status}) | EMPTY (will be removed)")
        
        # Renumber non-empty octants to sequential cluster IDs
        if len(non_empty_octants) < 8:
            print(f"\n🔄 Renumbering {len(non_empty_octants)} non-empty octants to sequential cluster IDs...")
            octant_to_new_id = {old_id: new_id for new_id, old_id in enumerate(non_empty_octants)}
            df_with_clusters[cluster_col_name] = df_with_clusters[cluster_col_name].map(octant_to_new_id)
            
            print("\nFinal Cluster Distribution (after removing empty octants):")
            for new_id, old_oct_id in enumerate(non_empty_octants):
                count = np.sum(df_with_clusters[cluster_col_name] == new_id)
                percent = 100 * count / len(df_with_clusters)
                
                # Decode original octant
                p1_status = "high" if (old_oct_id & 4) else "low"
                p2_status = "high" if (old_oct_id & 2) else "low"
                p3_status = "high" if (old_oct_id & 1) else "low"
                
                print(f"  Cluster {new_id} (was Octant {old_oct_id}: {p1_status}/{p2_status}/{p3_status}) | {count:6d} points ({percent:5.1f}%)")
        else:
            print("\n✅ All octants contain points - no renumbering needed")
    
    print("="*80)
    print(f"✅ Cluster column '{cluster_col_name}' added to dataframe")
    print("="*80 + "\n")
    
    return df_with_clusters


def merge_clusters(
    df,
    merge_groups,
    cluster_col='PC_clusters_manual',
    new_cluster_col=None
):
    """
    Merge multiple clusters into fewer clusters after visualization.
    
    This is useful for combining octants/clusters that you decide should be 
    grouped together after looking at the 3D PCA plot.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing cluster assignments
    merge_groups : list of lists
        Each sublist contains cluster IDs that should be merged together.
        Example: [[0, 1, 3], [2, 4]] merges 0,1,3 into cluster 0 and 2,4 into cluster 1.
    cluster_col : str, optional
        Name of the existing cluster column. Default 'PC_clusters_manual'.
    new_cluster_col : str, optional
        Name for the new cluster column. If None, overwrites existing column.
        Default None (overwrites).
    
    Returns
    -------
    df_merged : pandas.DataFrame
        DataFrame with updated cluster assignments
    
    Examples
    --------
    >>> # After visualizing, decide to merge clusters
    >>> # Say clusters 0, 1, 3 look similar, and 2, 4 look similar
    >>> df_merged = spt.merge_clusters(
    ...     pca_df_clustered,
    ...     merge_groups=[[0, 1, 3], [2, 4]],
    ...     cluster_col='PC_clusters_manual'
    ... )
    >>> # Result: 0,1,3 → cluster 0; 2,4 → cluster 1
    
    >>> # Keep merged clusters in a new column
    >>> df_merged = spt.merge_clusters(
    ...     pca_df_clustered,
    ...     merge_groups=[[0, 1], [2, 3, 4]],
    ...     cluster_col='PC_clusters_manual',
    ...     new_cluster_col='PC_clusters_merged'
    ... )
    
    Notes
    -----
    - Clusters not mentioned in merge_groups are left as-is (given new sequential IDs)
    - New cluster IDs are assigned sequentially starting from 0
    
    """
    import pandas as pd
    import numpy as np
    
    print("\n" + "="*80)
    print("MERGING CLUSTERS")
    print("="*80)
    
    df_merged = df.copy()
    
    # Get current cluster column
    if cluster_col not in df_merged.columns:
        raise ValueError(f"Cluster column '{cluster_col}' not found in dataframe")
    
    old_clusters = df_merged[cluster_col].values
    unique_old = sorted(df_merged[cluster_col].unique())
    
    print(f"Current clusters in '{cluster_col}': {unique_old}")
    print(f"Merge groups specified: {merge_groups}")
    print("-"*80)
    
    # Build mapping from old cluster ID to new cluster ID
    old_to_new = {}
    new_cluster_id = 0
    
    # First, assign new IDs to merged groups
    for group in merge_groups:
        print(f"Merge group {new_cluster_id}: {group} → new cluster {new_cluster_id}")
        for old_id in group:
            if old_id not in unique_old:
                print(f"  ⚠️  Warning: cluster {old_id} not found in data (will be ignored)")
            else:
                old_to_new[old_id] = new_cluster_id
        new_cluster_id += 1
    
    # Then assign new IDs to any remaining clusters (not in merge_groups)
    all_merged = set([item for group in merge_groups for item in group])
    remaining = [c for c in unique_old if c not in all_merged]
    
    if remaining:
        print(f"\nClusters not in merge groups (will keep as separate clusters):")
        for old_id in remaining:
            print(f"  Cluster {old_id} → new cluster {new_cluster_id}")
            old_to_new[old_id] = new_cluster_id
            new_cluster_id += 1
    
    # Apply mapping
    target_col = new_cluster_col if new_cluster_col else cluster_col
    df_merged[target_col] = pd.Series(old_clusters).map(old_to_new).values
    
    # Summary
    print("\n" + "="*80)
    print("FINAL CLUSTER DISTRIBUTION:")
    print("="*80)
    for new_id in sorted(df_merged[target_col].unique()):
        count = np.sum(df_merged[target_col] == new_id)
        percent = 100 * count / len(df_merged)
        
        # Find which old clusters contributed to this new cluster
        old_ids = [old for old, new in old_to_new.items() if new == new_id]
        print(f"  Cluster {new_id} (from old clusters {old_ids}): {count:6d} points ({percent:5.1f}%)")
    
    print("="*80)
    print(f"✅ Merged clusters saved to column '{target_col}'")
    print("="*80 + "\n")
    
    return df_merged


def center_scale_data(
    df,
    columns,
    method='standard',
    group_by=None,
    copy=True
):
    """
    Center and scale data using various scaling methods from scikit-learn.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to scale.
    columns : list of str
        List of column names to scale.
    method : str, optional
        Scaling method to use. Options are:
        - 'standard': StandardScaler (mean=0, std=1) [default]
        - 'minmax': MinMaxScaler (scales to [0, 1])
        - 'robust': RobustScaler (uses median and IQR, robust to outliers)
        - 'maxabs': MaxAbsScaler (scales to [-1, 1], preserves sparsity)
        - 'quantile': QuantileTransformer (uniform or gaussian distribution)
        - 'power': PowerTransformer (makes data more Gaussian)
        - 'normalize': Normalizer (scales each sample to unit norm)
        - 'standard_minmax': StandardScaler then MinMaxScaler (center then bound to [0,1])
    group_by : str or list of str, optional
        Column name(s) to group by before scaling. If provided, scaling is done
        independently for each group. Default is None.
    copy : bool, optional
        Whether to return a copy of the dataframe. Default is True.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with scaled columns. Original columns are replaced with scaled values.
    
    Examples
    --------
    >>> # Standard scaling
    >>> df_scaled = center_scale_data(df, ['x_um', 'y_um'], method='standard')
    
    >>> # MinMax scaling by group
    >>> df_scaled = center_scale_data(df, ['speed'], method='minmax', group_by='condition')
    
    >>> # Robust scaling (good for data with outliers)
    >>> df_scaled = center_scale_data(df, features_list, method='robust')
    
    """
    from sklearn.preprocessing import (
        StandardScaler,
        MinMaxScaler,
        RobustScaler,
        MaxAbsScaler,
        QuantileTransformer,
        PowerTransformer,
        Normalizer
    )
    
    if copy:
        df = df.copy()
    
    # Initialize the scaler based on method
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler(),
        'maxabs': MaxAbsScaler(),
        'quantile': QuantileTransformer(),
        'power': PowerTransformer(),
        'normalize': Normalizer(),
        'standard_minmax': 'combined'  # Special case
    }
    
    if method not in scalers:
        raise ValueError(
            f"method must be one of {list(scalers.keys())}, got '{method}'"
        )
    
    # Ensure columns is a list
    if isinstance(columns, str):
        columns = [columns]
    
    # Check if all columns exist
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")
    
    # Scale data
    if group_by is None:
        # Scale all data at once
        if method == 'standard_minmax':
            # First apply StandardScaler, then MinMaxScaler
            scaler1 = StandardScaler()
            df[columns] = scaler1.fit_transform(df[columns])
            scaler2 = MinMaxScaler()
            df[columns] = scaler2.fit_transform(df[columns])
        else:
            scaler = scalers[method]
            df[columns] = scaler.fit_transform(df[columns])
    else:
        # Scale by group
        if isinstance(group_by, str):
            group_by = [group_by]
        
        for group_vals, group_df in df.groupby(group_by):
            if method == 'standard_minmax':
                # First apply StandardScaler, then MinMaxScaler
                scaler1 = StandardScaler()
                df.loc[group_df.index, columns] = scaler1.fit_transform(group_df[columns])
                scaler2 = MinMaxScaler()
                df.loc[group_df.index, columns] = scaler2.fit_transform(df.loc[group_df.index, columns])
            else:
                scaler = scalers[method]
                df.loc[group_df.index, columns] = scaler.fit_transform(group_df[columns])
    
    return df


def balance_classes_for_pca(
    df,
    class_column,
    unique_id_column='unique_id',
    method='downsample',
    target_n_tracks=None,
    random_state=42,
    return_weights=False,
    stratify_within=None
):
    """
    Balance classes by sampling whole tracks (unique_ids) for PCA or other analyses.
    
    This function addresses class imbalance by balancing the number of TRACKS per class,
    not individual datapoints. Tracks remain intact (not split). Useful for PCA where
    large classes can dominate variance.
    
    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        DataFrame containing tracking data with unique_id and class columns.
    class_column : str
        Column name containing class labels to balance (e.g., 'mol', 'condition').
    unique_id_column : str, optional
        Column name containing unique track identifiers. Default 'unique_id'.
    method : str, optional
        Balancing method. Options:
        - 'downsample': Randomly sample tracks from each class to match smallest class (default)
        - 'stratified': Sample a fixed proportion of tracks from each class
        - 'weighted': Return sample weights for each track (for weighted PCA)
        Default 'downsample'.
    target_n_tracks : int, optional
        Target number of tracks per class. 
        - For 'downsample': if None, matches smallest class. If specified, samples that many.
        - For 'stratified': if None, uses 50% of tracks. If specified, samples that proportion
          as a fraction (e.g., 0.5 for 50%).
        - For 'weighted': ignored.
        Default None.
    random_state : int, optional
        Random seed for reproducibility. Default 42.
    return_weights : bool, optional
        If True, returns sample weights column. Useful for weighted PCA.
        Only meaningful for method='weighted'. Default False.
    stratify_within : str, optional
        Additional column to stratify within each class (e.g., 'replicate').
        Ensures balanced sampling across subgroups. Default None.
    
    Returns
    -------
    df_balanced : pd.DataFrame or pl.DataFrame
        Balanced dataframe (same type as input) containing sampled tracks.
        If return_weights=True, includes 'sample_weight' column.
    report : dict
        Dictionary with balancing statistics:
        - 'method': balancing method used
        - 'original_n_tracks_per_class': dict of class -> n_tracks before balancing
        - 'original_n_datapoints_per_class': dict of class -> n_datapoints before
        - 'balanced_n_tracks_per_class': dict of class -> n_tracks after balancing
        - 'balanced_n_datapoints_per_class': dict of class -> n_datapoints after
        - 'sampling_rate_per_class': dict of class -> fraction of tracks kept
        - 'sampled_unique_ids': list of track IDs included in balanced sample
        - 'weights_per_class': dict of class -> weight (if method='weighted')
    
    Examples
    --------
    >>> # Downsample to smallest class (HTT77 has 602 tracks)
    >>> df_balanced, report = balance_classes_for_pca(df, class_column='mol')
    >>> # Each mol now has 602 tracks
    
    >>> # Downsample to specific number of tracks
    >>> df_balanced, report = balance_classes_for_pca(
    ...     df, class_column='mol', method='downsample', target_n_tracks=1000
    ... )
    
    >>> # Stratified sampling - keep 30% of tracks from each class
    >>> df_balanced, report = balance_classes_for_pca(
    ...     df, class_column='mol', method='stratified', target_n_tracks=0.3
    ... )
    
    >>> # Weighted approach for PCA (returns all data with weights)
    >>> df_weighted, report = balance_classes_for_pca(
    ...     df, class_column='mol', method='weighted', return_weights=True
    ... )
    >>> # Use df_weighted['sample_weight'] in sklearn PCA with sample_weight parameter
    
    >>> # Stratify within replicates to ensure balanced replicate representation
    >>> df_balanced, report = balance_classes_for_pca(
    ...     df, class_column='mol', stratify_within='replicate'
    ... )
    
    >>> # Apply same balancing to multiple dataframes (e.g., instant + windowed)
    >>> instant_balanced, report = balance_classes_for_pca(instant_df, class_column='mol')
    >>> sampled_ids = report['sampled_unique_ids']
    >>> windowed_balanced = windowed_df[windowed_df['unique_id'].isin(sampled_ids)]
    >>> # Now both dataframes have the exact same tracks!
    
    Notes
    -----
    - Balancing is done at the TRACK level (by unique_id), not datapoint level
    - All datapoints belonging to a sampled track are included
    - For PCA, balancing prevents large classes from dominating variance
    - 'downsample': reduces variance (fewer samples) but removes bias
    - 'stratified': flexible sampling rate, good for exploring different amounts
    - 'weighted': keeps all data but assigns weights (requires PCA implementation support)
    
    """
    import numpy as np
    import pandas as pd
    
    # Detect if polars or pandas
    try:
        import polars as pl
        is_polars = isinstance(df, pl.DataFrame)
    except ImportError:
        is_polars = False
    
    # Convert polars to pandas for processing (will convert back at end)
    if is_polars:
        df_pandas = df.to_pandas()
    else:
        df_pandas = df.copy()
    
    # Validate inputs
    if class_column not in df_pandas.columns:
        raise ValueError(f"class_column '{class_column}' not found in dataframe")
    if unique_id_column not in df_pandas.columns:
        raise ValueError(f"unique_id_column '{unique_id_column}' not found in dataframe")
    if stratify_within is not None and stratify_within not in df_pandas.columns:
        raise ValueError(f"stratify_within '{stratify_within}' not found in dataframe")
    
    print("\n" + "="*80)
    print("CLASS BALANCING FOR PCA (by whole tracks)")
    print("="*80)
    print(f"Class column: {class_column}")
    print(f"Track ID column: {unique_id_column}")
    print(f"Method: {method}")
    if stratify_within:
        print(f"Stratifying within: {stratify_within}")
    print("-"*80)
    
    # Get original statistics
    original_stats = {}
    classes = sorted(df_pandas[class_column].unique())
    
    print("\n📊 ORIGINAL DATA (before balancing):")
    print(f"{'Class':<15} {'n_tracks':<12} {'n_datapoints':<15} {'avg_pts/track':<15}")
    print("-"*80)
    
    for cls in classes:
        cls_df = df_pandas[df_pandas[class_column] == cls]
        n_tracks = cls_df[unique_id_column].nunique()
        n_datapoints = len(cls_df)
        avg_pts = n_datapoints / n_tracks if n_tracks > 0 else 0
        
        original_stats[cls] = {
            'n_tracks': n_tracks,
            'n_datapoints': n_datapoints,
            'avg_pts_per_track': avg_pts
        }
        
        print(f"{str(cls):<15} {n_tracks:<12} {n_datapoints:<15} {avg_pts:<15.1f}")
    
    # Determine balancing strategy
    if method == 'downsample':
        # Downsample to target number of tracks per class
        if target_n_tracks is None:
            # Match smallest class
            target_n_tracks = min([s['n_tracks'] for s in original_stats.values()])
            print(f"\n🎯 Target: {target_n_tracks} tracks per class (matching smallest class)")
        else:
            print(f"\n🎯 Target: {target_n_tracks} tracks per class (user-specified)")
        
        # Check if target is feasible
        min_tracks = min([s['n_tracks'] for s in original_stats.values()])
        if target_n_tracks > min_tracks:
            print(f"⚠️  Warning: target_n_tracks ({target_n_tracks}) > smallest class ({min_tracks})")
            print(f"   Setting target to {min_tracks} (smallest class)")
            target_n_tracks = min_tracks
        
        np.random.seed(random_state)
        sampled_dfs = []
        
        for cls in classes:
            cls_df = df_pandas[df_pandas[class_column] == cls]
            unique_ids = cls_df[unique_id_column].unique()
            
            if stratify_within:
                # Stratified sampling within class
                sampled_ids = []
                for stratum_val in cls_df[stratify_within].unique():
                    stratum_ids = cls_df[cls_df[stratify_within] == stratum_val][unique_id_column].unique()
                    n_to_sample = int(np.ceil(len(stratum_ids) / len(unique_ids) * target_n_tracks))
                    n_to_sample = min(n_to_sample, len(stratum_ids))
                    sampled_ids.extend(np.random.choice(stratum_ids, n_to_sample, replace=False))
                
                # Trim to exact target if we oversampled due to ceiling
                if len(sampled_ids) > target_n_tracks:
                    sampled_ids = np.random.choice(sampled_ids, target_n_tracks, replace=False)
            else:
                # Simple random sampling
                sampled_ids = np.random.choice(unique_ids, target_n_tracks, replace=False)
            
            sampled_df = cls_df[cls_df[unique_id_column].isin(sampled_ids)]
            sampled_dfs.append(sampled_df)
        
        df_balanced = pd.concat(sampled_dfs, ignore_index=True)
        
    elif method == 'stratified':
        # Sample a proportion of tracks from each class
        if target_n_tracks is None:
            proportion = 0.5  # Default 50%
            print(f"\n🎯 Target: {proportion*100:.0f}% of tracks from each class (default)")
        else:
            if target_n_tracks > 1.0:
                raise ValueError(f"For stratified sampling, target_n_tracks should be a proportion (0-1), got {target_n_tracks}")
            proportion = target_n_tracks
            print(f"\n🎯 Target: {proportion*100:.0f}% of tracks from each class")
        
        np.random.seed(random_state)
        sampled_dfs = []
        
        for cls in classes:
            cls_df = df_pandas[df_pandas[class_column] == cls]
            unique_ids = cls_df[unique_id_column].unique()
            n_to_sample = max(1, int(len(unique_ids) * proportion))
            
            if stratify_within:
                # Stratified sampling within class
                sampled_ids = []
                for stratum_val in cls_df[stratify_within].unique():
                    stratum_ids = cls_df[cls_df[stratify_within] == stratum_val][unique_id_column].unique()
                    n_stratum_sample = max(1, int(len(stratum_ids) * proportion))
                    n_stratum_sample = min(n_stratum_sample, len(stratum_ids))
                    sampled_ids.extend(np.random.choice(stratum_ids, n_stratum_sample, replace=False))
            else:
                # Simple random sampling
                sampled_ids = np.random.choice(unique_ids, n_to_sample, replace=False)
            
            sampled_df = cls_df[cls_df[unique_id_column].isin(sampled_ids)]
            sampled_dfs.append(sampled_df)
        
        df_balanced = pd.concat(sampled_dfs, ignore_index=True)
        
    elif method == 'weighted':
        # Assign weights inversely proportional to class size (by n_tracks)
        print(f"\n🎯 Assigning weights inversely proportional to class track counts")
        print("   (Use these weights in sklearn PCA's fit() with sample_weight parameter)")
        
        # Calculate class weights (inverse of track count)
        total_tracks = sum([s['n_tracks'] for s in original_stats.values()])
        n_classes = len(classes)
        class_weights = {}
        
        for cls in classes:
            n_tracks_cls = original_stats[cls]['n_tracks']
            # Weight = (total_tracks / n_classes) / n_tracks_cls
            # This makes each class contribute equally
            class_weights[cls] = (total_tracks / n_classes) / n_tracks_cls
        
        # Assign weights to each datapoint based on its class
        df_balanced = df_pandas.copy()
        df_balanced['sample_weight'] = df_balanced[class_column].map(class_weights)
        
        # For track-level weights, also store per unique_id
        # (useful if you want to aggregate to track level before PCA)
        df_balanced['track_weight'] = df_balanced['sample_weight']
        
    else:
        raise ValueError(f"method must be 'downsample', 'stratified', or 'weighted', got '{method}'")
    
    # Get balanced statistics
    balanced_stats = {}
    
    print("\n" + "="*80)
    print("📊 BALANCED DATA (after balancing):")
    print(f"{'Class':<15} {'n_tracks':<12} {'n_datapoints':<15} {'avg_pts/track':<15} {'sampling_rate':<15}")
    print("-"*80)
    
    for cls in classes:
        cls_df_balanced = df_balanced[df_balanced[class_column] == cls]
        n_tracks = cls_df_balanced[unique_id_column].nunique()
        n_datapoints = len(cls_df_balanced)
        avg_pts = n_datapoints / n_tracks if n_tracks > 0 else 0
        sampling_rate = n_tracks / original_stats[cls]['n_tracks']
        
        balanced_stats[cls] = {
            'n_tracks': n_tracks,
            'n_datapoints': n_datapoints,
            'avg_pts_per_track': avg_pts,
            'sampling_rate': sampling_rate
        }
        
        print(f"{str(cls):<15} {n_tracks:<12} {n_datapoints:<15} {avg_pts:<15.1f} {sampling_rate:<15.2%}")
    
    if method == 'weighted':
        print("\n⚖️  WEIGHTS PER CLASS:")
        for cls in classes:
            weight = class_weights[cls]
            print(f"  {str(cls):<15} weight = {weight:.4f}")
        print("\n   → Each datapoint has 'sample_weight' and 'track_weight' columns")
        print("   → Use in PCA: pca.fit(X, sample_weight=df['sample_weight'])")
    
    # Summary statistics
    print("\n" + "="*80)
    print("📈 SUMMARY:")
    total_tracks_orig = sum([s['n_tracks'] for s in original_stats.values()])
    total_tracks_balanced = sum([s['n_tracks'] for s in balanced_stats.values()])
    total_pts_orig = sum([s['n_datapoints'] for s in original_stats.values()])
    total_pts_balanced = sum([s['n_datapoints'] for s in balanced_stats.values()])
    
    print(f"  Total tracks:      {total_tracks_orig:,} → {total_tracks_balanced:,} ({total_tracks_balanced/total_tracks_orig:.1%} retained)")
    print(f"  Total datapoints:  {total_pts_orig:,} → {total_pts_balanced:,} ({total_pts_balanced/total_pts_orig:.1%} retained)")
    
    if method in ['downsample', 'stratified']:
        # Check balance quality
        n_tracks_per_class = [s['n_tracks'] for s in balanced_stats.values()]
        balance_ratio = max(n_tracks_per_class) / min(n_tracks_per_class) if min(n_tracks_per_class) > 0 else np.inf
        print(f"  Balance ratio:     {balance_ratio:.2f} (1.0 = perfect balance)")
        
        if balance_ratio < 1.1:
            print("  ✅ Excellent balance!")
        elif balance_ratio < 1.5:
            print("  ✓  Good balance")
        else:
            print("  ⚠️  Moderate imbalance remains")
    
    print("="*80 + "\n")
    
    # Get list of sampled unique_ids for filtering other dataframes
    sampled_unique_ids = df_balanced[unique_id_column].unique().tolist()
    
    # Prepare report
    report = {
        'method': method,
        'original_n_tracks_per_class': {cls: s['n_tracks'] for cls, s in original_stats.items()},
        'original_n_datapoints_per_class': {cls: s['n_datapoints'] for cls, s in original_stats.items()},
        'balanced_n_tracks_per_class': {cls: s['n_tracks'] for cls, s in balanced_stats.items()},
        'balanced_n_datapoints_per_class': {cls: s['n_datapoints'] for cls, s in balanced_stats.items()},
        'sampling_rate_per_class': {cls: s['sampling_rate'] for cls, s in balanced_stats.items()},
        'sampled_unique_ids': sampled_unique_ids,  # List of all sampled track IDs
    }
    
    if method == 'weighted':
        report['weights_per_class'] = class_weights
    
    print(f"💾 Sampled {len(sampled_unique_ids)} unique tracks (IDs stored in report['sampled_unique_ids'])")
    print("   Use these to filter other dataframes to match this balanced sample!")
    
    # Convert back to polars if needed
    if is_polars:
        df_balanced = pl.from_pandas(df_balanced)
    
    return df_balanced, report
