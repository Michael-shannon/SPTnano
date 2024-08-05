################# This is the base original version for building new stuff into #################

# import pandas as pd
# import numpy as np

# class ParticleMetrics:
#     def __init__(self, df):
#         self.df = df
#         self.metrics_df = self.df.copy()
        
#     def calculate_distances(self):
#         """
#         Calculate the distances between consecutive frames for each particle in micrometers.
#         """
#         self.metrics_df = self.metrics_df.sort_values(by=['unique_id', 'frame'])
#         self.metrics_df[['x_um_prev', 'y_um_prev']] = self.metrics_df.groupby('unique_id')[['x_um', 'y_um']].shift(1)
#         self.metrics_df['segment_len_um'] = np.sqrt(
#             (self.metrics_df['x_um'] - self.metrics_df['x_um_prev'])**2 + 
#             (self.metrics_df['y_um'] - self.metrics_df['y_um_prev'])**2
#         )
#         # Fill NaN values with 0
#         self.metrics_df['segment_len_um'] = self.metrics_df['segment_len_um'].fillna(0)
#         return self.metrics_df

#     def calculate_speeds(self):
#         """
#         Calculate the speed between consecutive frames for each particle in micrometers per second.
#         """
#         self.metrics_df[['time_s_prev']] = self.metrics_df.groupby('unique_id')[['time_s']].shift(1)
#         self.metrics_df['delta_time_s'] = self.metrics_df['time_s'] - self.metrics_df['time_s_prev']
#         self.metrics_df['speed_um_s'] = self.metrics_df['segment_len_um'] / self.metrics_df['delta_time_s']
#         # Fill NaN and infinite values with 0
#         self.metrics_df['speed_um_s'] = self.metrics_df['speed_um_s'].replace([np.inf, -np.inf], np.nan).fillna(0)
#         return self.metrics_df

#     def calculate_all_features(self):
#         """
#         Calculate all features for the particle tracking data.
#         This method will call all individual feature calculation methods.
#         """
#         # Calculate distances between consecutive frames
#         self.calculate_distances()

#         # Calculate speeds between consecutive frames
#         self.calculate_speeds()
        
#         # Placeholder for additional feature calculations
#         # self.calculate_feature_X()
#         # self.calculate_feature_Y()
        
#         # Cleanup step to remove temporary columns
#         self.cleanup()
        
#         return self.metrics_df

#     def cleanup(self):
#         """
#         Cleanup the dataframe by dropping unnecessary columns after all features are calculated.
#         """
#         self.metrics_df.drop(columns=['x_um_prev', 'y_um_prev', 'time_s_prev', 'delta_time_s'], inplace=True)

#     def get_metrics_df(self):
#         """
#         Return the dataframe with calculated metrics.
#         """
#         return self.metrics_df
    
############# This is the base original version for building new stuff into #################

############################### DEVELOPING NEW FEATURES #########################################




import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import scipy.optimize
import re


# class ParticleMetrics:
#     def __init__(self, df):
#         self.df = df.copy()
#         self.df['Location'] = self.df['filename'].apply(self.extract_location)  # Add Location column
#         self.metrics_df = self.df.copy()
#         self.time_averaged_df = pd.DataFrame(columns=[
#             'x_um_start', 'y_um_start', 'x_um_end', 'y_um_end', 
#             'particle', 'condition', 'filename', 'file_id', 'unique_id',
#             'avg_msd', 'n_frames', 'total_time_s', 'Location', 
#             'diffusion_coefficient', 'anomalous_exponent', 'motion_class'  # Add columns for diffusion coefficient, anomalous exponent, and motion class
#         ])
#         self.time_windowed_df = pd.DataFrame(columns=[
#             'time_window', 'x_um_start', 'y_um_start', 'x_um_end', 'y_um_end',
#             'particle', 'condition', 'filename', 'file_id', 'unique_id',
#             'avg_msd', 'n_frames', 'total_time_s', 'Location',
#             'diffusion_coefficient', 'anomalous_exponent', 'motion_class'  # Add columns for diffusion coefficient, anomalous exponent, and motion class
#         ])

#     @staticmethod
#     def extract_location(filename):
#         match = re.match(r'loc-(\w{2})_', filename)
#         if match:
#             return match.group(1)
#         return 'Unknown'  # Default value if no location is found

#     @staticmethod
#     def msd_model(t, D, alpha):
#         return 4 * D * t**alpha
    
#     def calculate_distances(self):
#         """
#         Calculate the distances between consecutive frames for each particle in micrometers.
#         """
#         self.metrics_df = self.metrics_df.sort_values(by=['unique_id', 'frame'])
#         self.metrics_df[['x_um_prev', 'y_um_prev']] = self.metrics_df.groupby('unique_id')[['x_um', 'y_um']].shift(1)
#         self.metrics_df['segment_len_um'] = np.sqrt(
#             (self.metrics_df['x_um'] - self.metrics_df['x_um_prev'])**2 + 
#             (self.metrics_df['y_um'] - self.metrics_df['y_um_prev'])**2
#         )
#         # Fill NaN values with 0
#         self.metrics_df['segment_len_um'] = self.metrics_df['segment_len_um'].fillna(0)
#         return self.metrics_df

#     def calculate_speeds(self):
#         """
#         Calculate the speed between consecutive frames for each particle in micrometers per second.
#         """
#         self.metrics_df[['time_s_prev']] = self.metrics_df.groupby('unique_id')[['time_s']].shift(1)
#         self.metrics_df['delta_time_s'] = self.metrics_df['time_s'] - self.metrics_df['time_s_prev']
#         self.metrics_df['speed_um_s'] = self.metrics_df['segment_len_um'] / self.metrics_df['delta_time_s']
#         # Fill NaN and infinite values with 0
#         self.metrics_df['speed_um_s'] = self.metrics_df['speed_um_s'].replace([np.inf, -np.inf], np.nan).fillna(0)
#         return self.metrics_df


    
#     def calculate_directions(self):
#         """
#         Calculate the direction of motion between consecutive frames for each particle in radians.
#         """
#         self.metrics_df['direction_rad'] = np.arctan2(
#             self.metrics_df['y_um'] - self.metrics_df['y_um_prev'],
#             self.metrics_df['x_um'] - self.metrics_df['x_um_prev']
#         )
#         # Fill NaN values with 0
#         self.metrics_df['direction_rad'] = self.metrics_df['direction_rad'].fillna(0)
#         return self.metrics_df

    
#     def calculate_accelerations(self):
#         """
#         Calculate the acceleration between consecutive frames for each particle in micrometers per second squared.
#         """
#         self.metrics_df[['speed_um_s_prev']] = self.metrics_df.groupby('unique_id')[['speed_um_s']].shift(1)
#         self.metrics_df['acceleration_um_s2'] = (self.metrics_df['speed_um_s'] - self.metrics_df['speed_um_s_prev']) / self.metrics_df['delta_time_s']
#         # Fill NaN and infinite values with 0
#         self.metrics_df['acceleration_um_s2'] = self.metrics_df['acceleration_um_s2'].replace([np.inf, -np.inf], np.nan).fillna(0)
#         return self.metrics_df

#     def calculate_jerk(self):
#         """
#         Calculate the jerk between consecutive frames for each particle in micrometers per second cubed.
#         """
#         self.metrics_df[['acceleration_um_s2_prev']] = self.metrics_df.groupby('unique_id')[['acceleration_um_s2']].shift(1)
#         self.metrics_df['jerk_um_s3'] = (self.metrics_df['acceleration_um_s2'] - self.metrics_df['acceleration_um_s2_prev']) / self.metrics_df['delta_time_s']
#         # Fill NaN and infinite values with 0
#         self.metrics_df['jerk_um_s3'] = self.metrics_df['jerk_um_s3'].replace([np.inf, -np.inf], np.nan).fillna(0)
#         return self.metrics_df

#     def calculate_normalized_curvature(self):
#         """
#         Calculate the curvature normalized by distance between consecutive frames for each particle.
#         """
#         self.metrics_df[['direction_rad_prev']] = self.metrics_df.groupby('unique_id')[['direction_rad']].shift(1)
#         self.metrics_df['normalized_curvature'] = (self.metrics_df['direction_rad'] - self.metrics_df['direction_rad_prev']) / self.metrics_df['segment_len_um']
#         # Fill NaN and infinite values with 0
#         self.metrics_df['normalized_curvature'] = self.metrics_df['normalized_curvature'].replace([np.inf, -np.inf], np.nan).fillna(0)
#         return self.metrics_df
    
#     def calculate_angle_normalized_curvature(self):
#         """
#         Calculate the curvature (change in direction) normalized to the range [-pi, pi] between consecutive frames for each particle.
#         """
#         self.metrics_df[['direction_rad_prev']] = self.metrics_df.groupby('unique_id')[['direction_rad']].shift(1)
#         self.metrics_df['angle_normalized_curvature'] = self.metrics_df['direction_rad'] - self.metrics_df['direction_rad_prev']
#         # Normalize curvature to the range [-pi, pi]
#         self.metrics_df['angle_normalized_curvature'] = (self.metrics_df['angle_normalized_curvature'] + np.pi) % (2 * np.pi) - np.pi
#         # Fill NaN values with 0
#         self.metrics_df['angle_normalized_curvature'] = self.metrics_df['angle_normalized_curvature'].fillna(0)
#         return self.metrics_df

#     def calculate_net_displacement(self):
#         """
#         Calculate the net displacement from the starting point for each particle in micrometers.
#         """
#         self.metrics_df[['x_um_start', 'y_um_start']] = self.metrics_df.groupby('unique_id')[['x_um', 'y_um']].transform('first')
#         self.metrics_df['net_displacement_um'] = np.sqrt(
#             (self.metrics_df['x_um'] - self.metrics_df['x_um_start'])**2 + 
#             (self.metrics_df['y_um'] - self.metrics_df['y_um_start'])**2
#         )
#         return self.metrics_df

#     def calculate_instantaneous_diffusion_coefficient(self):
#         """
#         Calculate the instantaneous diffusion coefficient for each particle in square micrometers per second.
#         """
#         self.metrics_df['instant_diff_coeff'] = self.metrics_df['segment_len_um']**2 / (4 * self.metrics_df['delta_time_s'])
#         # Fill NaN and infinite values with 0
#         self.metrics_df['instant_diff_coeff'] = self.metrics_df['instant_diff_coeff'].replace([np.inf, -np.inf], np.nan).fillna(0)
#         return self.metrics_df
    
#     def calculate_instantaneous_velocity(self):
#         """
#         Calculate the instantaneous velocity between consecutive frames for each particle in micrometers per second.
#         """
#         self.metrics_df['instant_velocity_x_um_s'] = (self.metrics_df['x_um'] - self.metrics_df['x_um_prev']) / self.metrics_df['delta_time_s']
#         self.metrics_df['instant_velocity_y_um_s'] = (self.metrics_df['y_um'] - self.metrics_df['y_um_prev']) / self.metrics_df['delta_time_s']
#         # Fill NaN and infinite values with 0
#         self.metrics_df['instant_velocity_x_um_s'] = self.metrics_df['instant_velocity_x_um_s'].replace([np.inf, -np.inf], np.nan).fillna(0)
#         self.metrics_df['instant_velocity_y_um_s'] = self.metrics_df['instant_velocity_y_um_s'].replace([np.inf, -np.inf], np.nan).fillna(0)
#         return self.metrics_df


#     def calculate_msd_for_track(self, track_data, max_lagtime):
#         """
#         Calculate the MSD for a single track.
#         Parameters:
#         - track_data: DataFrame containing the track data
#         - max_lagtime: maximum number of frames to consider for lag times
#         Returns:
#         - avg_msd: average MSD for the track
#         - D: diffusion coefficient
#         - alpha: anomalous exponent
#         - motion_class: classified motion type
#         """
#         n_frames = len(track_data)
#         msd_values = np.zeros(max_lagtime)
#         counts = np.zeros(max_lagtime)

#         for lag in range(1, max_lagtime + 1):
#             if lag < n_frames:
#                 displacements = (track_data[['x_um', 'y_um']].iloc[lag:].values - track_data[['x_um', 'y_um']].iloc[:-lag].values) ** 2
#                 squared_displacements = np.sum(displacements, axis=1)
#                 msd_values[lag - 1] = np.mean(squared_displacements)
#                 counts[lag - 1] = len(squared_displacements)
#             else:
#                 break

#         avg_msd = np.mean(msd_values)  # Calculate the average MSD for the track (units: μm²)

#         # Calculate total time in seconds for the track
#         total_time_s = (track_data['time_s'].iloc[-1] - track_data['time_s'].iloc[0])
#         lag_times = np.arange(1, max_lagtime + 1) * (total_time_s / (n_frames - 1))
#         popt, _ = scipy.optimize.curve_fit(self.msd_model, lag_times, msd_values[:max_lagtime])
#         D, alpha = popt[0], popt[1]

#         # Classify the type of motion
#         if alpha < 1:
#             motion_class = 'subdiffusive'
#         elif alpha > 1:
#             motion_class = 'superdiffusive'
#         else:
#             motion_class = 'normal'

#         return avg_msd, D, alpha, motion_class

#     def produce_time_averaged_df(self, max_lagtime=None):
#         """
#         Produce the time-averaged DataFrame.
#         Parameters:
#         - max_lagtime: maximum number of frames to consider for lag times
#         """
#         if max_lagtime is None:
#             max_lagtime = self.calculate_default_max_lagtime()

#         time_averaged_list = []

#         for unique_id, track_data in tqdm(self.metrics_df.groupby('unique_id'), desc="Producing Time-Averaged DataFrame"):
#             avg_msd, D, alpha, motion_class = self.calculate_msd_for_track(track_data, max_lagtime)

#             # Calculate total time in seconds for the track
#             total_time_s = (track_data['time_s'].iloc[-1] - track_data['time_s'].iloc[0])

#             # Add track-level summary information to time_averaged_df
#             start_row = track_data.iloc[0]
#             end_row = track_data.iloc[-1]
#             track_summary = pd.DataFrame({
#                 'x_um_start': [start_row['x_um']],
#                 'y_um_start': [start_row['y_um']],
#                 'x_um_end': [end_row['x_um']],
#                 'y_um_end': [end_row['y_um']],
#                 'particle': [start_row['particle']],
#                 'condition': [start_row['condition']],
#                 'filename': [start_row['filename']],
#                 'file_id': [start_row['file_id']],
#                 'unique_id': [unique_id],
#                 'avg_msd': [avg_msd],  # Add the average MSD (units: μm²)
#                 'n_frames': [len(track_data)],  # Add the number of frames
#                 'total_time_s': [total_time_s],  # Add the total time in seconds
#                 'Location': [start_row['Location']],  # Add the Location
#                 'diffusion_coefficient': [D],  # Add the diffusion coefficient
#                 'anomalous_exponent': [alpha],  # Add the anomalous exponent
#                 'motion_class': [motion_class],  # Add the motion class
#                 # Placeholder for additional metrics
#                 # 'additional_metric': None,
#             })

#             time_averaged_list.append(track_summary)

#         self.time_averaged_df = pd.concat(time_averaged_list).reset_index(drop=True)

#     def calculate_time_windowed_metrics(self, window_size=None, overlap=None):
#         """
#         Calculate metrics for each time window.
#         Parameters:
#         - window_size: size of the time window in frames
#         - overlap: number of overlapping frames between windows
#         """
#         if window_size is None:
#             window_size = self.calculate_default_window_size()
#         if overlap is None:
#             overlap = int(window_size / 2)  # Default overlap is half the window size

#         # Print the calculated or provided window size and overlap
#         print(f"Using window size: {window_size} frames: please note, tracks shorter than the window size will be skipped")
#         print(f"Using overlap: {overlap} frames")

#         windowed_list = []

#         for unique_id, track_data in tqdm(self.metrics_df.groupby('unique_id'), desc="Calculating Time-Windowed Metrics"):
#             n_frames = len(track_data)

#             for start in range(0, n_frames - window_size + 1, window_size - overlap):
#                 end = start + window_size
#                 window_data = track_data.iloc[start:end]

#                 if len(window_data) < window_size:
#                     continue

#                 # Calculate metrics for the window
#                 avg_msd, D, alpha, motion_class = self.calculate_msd_for_track(window_data, max_lagtime=min(100, int(window_size / 2)))

#                 # Calculate total time in seconds for the window
#                 total_time_s = (window_data['time_s'].iloc[-1] - window_data['time_s'].iloc[0])

#                 # Add window-level summary information to time_windowed_df
#                 start_row = window_data.iloc[0]
#                 end_row = window_data.iloc[-1]
#                 window_summary = pd.DataFrame({
#                     'time_window': [start // (window_size - overlap)],
#                     'x_um_start': [start_row['x_um']],
#                     'y_um_start': [start_row['y_um']],
#                     'x_um_end': [end_row['x_um']],
#                     'y_um_end': [end_row['y_um']],
#                     'particle': [start_row['particle']],
#                     'condition': [start_row['condition']],
#                     'filename': [start_row['filename']],
#                     'file_id': [start_row['file_id']],
#                     'unique_id': [unique_id],
#                     'avg_msd': [avg_msd],  # Add the average MSD (units: μm²)
#                     'n_frames': [window_size],  # Add the number of frames
#                     'total_time_s': [total_time_s],  # Add the total time in seconds
#                     'Location': [start_row['Location']],  # Add the Location
#                     'diffusion_coefficient': [D],  # Add the diffusion coefficient
#                     'anomalous_exponent': [alpha],  # Add the anomalous exponent
#                     'motion_class': [motion_class],  # Add the motion class
#                     # Placeholder for additional metrics
#                     # 'additional_metric': None,
#                 })

#                 windowed_list.append(window_summary)

#         self.time_windowed_df = pd.concat(windowed_list).reset_index(drop=True)

#     def calculate_metrics_for_window(self, window_data):
#         """
#         Calculate metrics for a given window of data.
#         """
#         n_frames = len(window_data)
#         max_lagtime = min(100, int(n_frames / 2))  # Example: use half the length of the window or 100, whichever is smaller

#         msd_values = np.zeros(max_lagtime)

#         for lag in range(1, max_lagtime + 1):
#             if lag < n_frames:
#                 displacements = (window_data[['x_um', 'y_um']].iloc[lag:].values - window_data[['x_um', 'y_um']].iloc[:-lag].values) ** 2
#                 squared_displacements = np.sum(displacements, axis=1)
#                 msd_values[lag - 1] = np.mean(squared_displacements)
#             else:
#                 break

#         avg_msd = np.mean(msd_values)  # Calculate the average MSD for the window (units: μm²)

#         # Fit MSD data to determine the type of motion and extract parameters
#         total_time_s = (window_data['time_s'].iloc[-1] - window_data['time_s'].iloc[0])
#         lag_times = np.arange(1, max_lagtime + 1) * (total_time_s / (n_frames - 1))
#         popt, _ = scipy.optimize.curve_fit(self.msd_model, lag_times, msd_values[:max_lagtime])
#         D, alpha = popt[0], popt[1]

#         # Classify the type of motion
#         if alpha < 1:
#             motion_class = 'subdiffusive'
#         elif alpha > 1:
#             motion_class = 'superdiffusive'
#         else:
#             motion_class = 'normal'

#         return avg_msd, D, alpha, motion_class

#     def calculate_default_window_size(self):
#         """
#         Calculate the default window size based on the average length of tracks in the dataset.
#         """
#         avg_track_length = self.metrics_df.groupby('unique_id').size().mean()
#         default_window_size = int(avg_track_length / 2)  # Example: use half the average length of the tracks
#         return default_window_size

#     def calculate_all_features(self, max_lagtime=None, calculate_time_windowed=False):
#         """
#         Calculate all features for the particle tracking data.
#         This method will call all individual feature calculation methods.
#         Parameters:
#         - max_lagtime: maximum number of frames to consider for lag times
#         - calculate_time_windowed: boolean flag to indicate if time-windowed metrics should be calculated
#         """
#         # Calculate default max lag time if not provided
#         if max_lagtime is None:
#             max_lagtime = self.calculate_default_max_lagtime()

#         # Calculate distances between consecutive frames
#         self.calculate_distances()

#         # Calculate speeds between consecutive frames
#         self.calculate_speeds()
        
#         # Calculate directions of motion between consecutive frames
#         self.calculate_directions()
        
#         # Calculate accelerations between consecutive frames
#         self.calculate_accelerations()
        
#         # Calculate jerks between consecutive frames
#         self.calculate_jerk()
        
#         # Calculate curvatures between consecutive frames
#         self.calculate_normalized_curvature()
#         self.calculate_angle_normalized_curvature()

#         # Calculate instantaneous diffusion coefficient between consecutive frames
#         self.calculate_instantaneous_diffusion_coefficient()
        
#         # Calculate net displacement
#         self.calculate_net_displacement()

#         # Calculate MSD for each track and aggregate
#         self.calculate_msd_for_all_tracks(max_lagtime)

#         # Calculate instantaneous velocity
#         self.calculate_instantaneous_velocity()
        
#         # Calculate time-windowed metrics if requested
#         if calculate_time_windowed:
#             self.calculate_time_windowed_metrics()
        
#         # Cleanup step to remove temporary columns
#         self.cleanup()
        
#         return self.metrics_df

#     def get_time_averaged_df(self):
#         """
#         Return the DataFrame with time-averaged metrics.
#         """
#         return self.time_averaged_df

#     def get_time_windowed_df(self):
#         """
#         Return the DataFrame with time-windowed metrics.
#         """
#         return self.time_windowed_df

#     def calculate_default_max_lagtime(self):
#         """
#         Calculate the default maximum lag time based on the shortest track in the dataset.
#         """
#         min_track_length = self.metrics_df.groupby('unique_id').size().min()
#         default_max_lagtime = min(100, int(min_track_length / 2))  # Example: use half the length of the shortest track or 100, whichever is smaller
#         return default_max_lagtime

#     def calculate_msd_for_all_tracks(self, max_lagtime):
#         """
#         Calculate the MSD for all tracks and produce the time-averaged DataFrame.
#         Parameters:
#         - max_lagtime: maximum number of frames to consider for lag times
#         """
#         self.produce_time_averaged_df(max_lagtime)

#     def cleanup(self):
#         """
#         Cleanup the dataframe by dropping unnecessary columns after all features are calculated.
#         """
#         self.metrics_df.drop(columns=[
#             'x_um_prev', 'y_um_prev', 'time_s_prev', 'delta_time_s', 
#             'speed_um_s_prev', 'acceleration_um_s2_prev', 'direction_rad_prev',
#             'instant_velocity_x_um_s', 'instant_velocity_y_um_s',
#             ], inplace=True)






##############################################################################################
############################## NEW ONE BEGIN ########################################################

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import scipy.optimize
import re

class ParticleMetrics:
    def __init__(self, df):
        self.df = df.copy()
        self.df['Location'] = self.df['filename'].apply(self.extract_location)  # Add Location column
        self.metrics_df = self.df.copy()
        self.time_averaged_df = pd.DataFrame(columns=[
            'x_um_start', 'y_um_start', 'x_um_end', 'y_um_end', 
            'particle', 'condition', 'filename', 'file_id', 'unique_id',
            'avg_msd', 'n_frames', 'total_time_s', 'Location', 
            'diffusion_coefficient', 'anomalous_exponent', 'motion_class',
            'avg_speed_um_s', 'avg_acceleration_um_s2', 'avg_jerk_um_s3',
            'avg_normalized_curvature', 'avg_angle_normalized_curvature'  # Add average normalized curvature and angle normalized curvature
        ])
        self.time_windowed_df = pd.DataFrame(columns=[
            'time_window', 'x_um_start', 'y_um_start', 'x_um_end', 'y_um_end',
            'particle', 'condition', 'filename', 'file_id', 'unique_id',
            'avg_msd', 'n_frames', 'total_time_s', 'Location',
            'diffusion_coefficient', 'anomalous_exponent', 'motion_class',
            'avg_speed_um_s', 'avg_acceleration_um_s2', 'avg_jerk_um_s3',
            'avg_normalized_curvature', 'avg_angle_normalized_curvature'  # Add average normalized curvature and angle normalized curvature
        ])

    @staticmethod
    def extract_location(filename):
        match = re.match(r'loc-(\w{2})_', filename)
        if match:
            return match.group(1)
        return 'Unknown'  # Default value if no location is found

    @staticmethod
    def msd_model(t, D, alpha):
        return 4 * D * t**alpha
    
    def calculate_distances(self):
        """
        Calculate the distances between consecutive frames for each particle in micrometers.
        """
        self.metrics_df = self.metrics_df.sort_values(by=['unique_id', 'frame'])
        self.metrics_df[['x_um_prev', 'y_um_prev']] = self.metrics_df.groupby('unique_id')[['x_um', 'y_um']].shift(1)
        self.metrics_df['segment_len_um'] = np.sqrt(
            (self.metrics_df['x_um'] - self.metrics_df['x_um_prev'])**2 + 
            (self.metrics_df['y_um'] - self.metrics_df['y_um_prev'])**2
        )
        # Fill NaN values with 0
        self.metrics_df['segment_len_um'] = self.metrics_df['segment_len_um'].fillna(0)
        return self.metrics_df

    def calculate_speeds(self):
        """
        Calculate the speed between consecutive frames for each particle in micrometers per second.
        """
        self.metrics_df[['time_s_prev']] = self.metrics_df.groupby('unique_id')[['time_s']].shift(1)
        self.metrics_df['delta_time_s'] = self.metrics_df['time_s'] - self.metrics_df['time_s_prev']
        self.metrics_df['speed_um_s'] = self.metrics_df['segment_len_um'] / self.metrics_df['delta_time_s']
        # Fill NaN and infinite values with 0
        self.metrics_df['speed_um_s'] = self.metrics_df['speed_um_s'].replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.metrics_df

    def calculate_directions(self):
        """
        Calculate the direction of motion between consecutive frames for each particle in radians.
        """
        self.metrics_df['direction_rad'] = np.arctan2(
            self.metrics_df['y_um'] - self.metrics_df['y_um_prev'],
            self.metrics_df['x_um'] - self.metrics_df['x_um_prev']
        )
        # Fill NaN values with 0
        self.metrics_df['direction_rad'] = self.metrics_df['direction_rad'].fillna(0)
        return self.metrics_df

    def calculate_accelerations(self):
        """
        Calculate the acceleration between consecutive frames for each particle in micrometers per second squared.
        """
        self.metrics_df[['speed_um_s_prev']] = self.metrics_df.groupby('unique_id')[['speed_um_s']].shift(1)
        self.metrics_df['acceleration_um_s2'] = (self.metrics_df['speed_um_s'] - self.metrics_df['speed_um_s_prev']) / self.metrics_df['delta_time_s']
        # Fill NaN and infinite values with 0
        self.metrics_df['acceleration_um_s2'] = self.metrics_df['acceleration_um_s2'].replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.metrics_df

    def calculate_jerk(self):
        """
        Calculate the jerk between consecutive frames for each particle in micrometers per second cubed.
        """
        self.metrics_df[['acceleration_um_s2_prev']] = self.metrics_df.groupby('unique_id')[['acceleration_um_s2']].shift(1)
        self.metrics_df['jerk_um_s3'] = (self.metrics_df['acceleration_um_s2'] - self.metrics_df['acceleration_um_s2_prev']) / self.metrics_df['delta_time_s']
        # Fill NaN and infinite values with 0
        self.metrics_df['jerk_um_s3'] = self.metrics_df['jerk_um_s3'].replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.metrics_df

    def calculate_normalized_curvature(self):
        """
        Calculate the curvature normalized by distance between consecutive frames for each particle.
        """
        self.metrics_df[['direction_rad_prev']] = self.metrics_df.groupby('unique_id')[['direction_rad']].shift(1)
        self.metrics_df['normalized_curvature'] = (self.metrics_df['direction_rad'] - self.metrics_df['direction_rad_prev']) / self.metrics_df['segment_len_um']
        # Fill NaN and infinite values with 0
        self.metrics_df['normalized_curvature'] = self.metrics_df['normalized_curvature'].replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.metrics_df
    
    def calculate_angle_normalized_curvature(self):
        """
        Calculate the curvature (change in direction) normalized to the range [-pi, pi] between consecutive frames for each particle.
        """
        self.metrics_df[['direction_rad_prev']] = self.metrics_df.groupby('unique_id')[['direction_rad']].shift(1)
        self.metrics_df['angle_normalized_curvature'] = self.metrics_df['direction_rad'] - self.metrics_df['direction_rad_prev']
        # Normalize curvature to the range [-pi, pi]
        self.metrics_df['angle_normalized_curvature'] = (self.metrics_df['angle_normalized_curvature'] + np.pi) % (2 * np.pi) - np.pi
        # Fill NaN values with 0
        self.metrics_df['angle_normalized_curvature'] = self.metrics_df['angle_normalized_curvature'].fillna(0)
        return self.metrics_df

    def calculate_net_displacement(self):
        """
        Calculate the net displacement from the starting point for each particle in micrometers.
        """
        self.metrics_df[['x_um_start', 'y_um_start']] = self.metrics_df.groupby('unique_id')[['x_um', 'y_um']].transform('first')
        self.metrics_df['net_displacement_um'] = np.sqrt(
            (self.metrics_df['x_um'] - self.metrics_df['x_um_start'])**2 + 
            (self.metrics_df['y_um'] - self.metrics_df['y_um_start'])**2
        )
        return self.metrics_df

    def calculate_instantaneous_diffusion_coefficient(self):
        """
        Calculate the instantaneous diffusion coefficient for each particle in square micrometers per second.
        """
        self.metrics_df['instant_diff_coeff'] = self.metrics_df['segment_len_um']**2 / (4 * self.metrics_df['delta_time_s'])
        # Fill NaN and infinite values with 0
        self.metrics_df['instant_diff_coeff'] = self.metrics_df['instant_diff_coeff'].replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.metrics_df
    
    def calculate_instantaneous_velocity(self):
        """
        Calculate the instantaneous velocity between consecutive frames for each particle in micrometers per second.
        """
        self.metrics_df['instant_velocity_x_um_s'] = (self.metrics_df['x_um'] - self.metrics_df['x_um_prev']) / self.metrics_df['delta_time_s']
        self.metrics_df['instant_velocity_y_um_s'] = (self.metrics_df['y_um'] - self.metrics_df['y_um_prev']) / self.metrics_df['delta_time_s']
        # Fill NaN and infinite values with 0
        self.metrics_df['instant_velocity_x_um_s'] = self.metrics_df['instant_velocity_x_um_s'].replace([np.inf, -np.inf], np.nan).fillna(0)
        self.metrics_df['instant_velocity_y_um_s'] = self.metrics_df['instant_velocity_y_um_s'].replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.metrics_df

    def calculate_msd_for_track(self, track_data, max_lagtime):
        """
        Calculate the MSD for a single track.
        Parameters:
        - track_data: DataFrame containing the track data
        - max_lagtime: maximum number of frames to consider for lag times
        Returns:
        - avg_msd: average MSD for the track
        - D: diffusion coefficient
        - alpha: anomalous exponent
        - motion_class: classified motion type
        """
        n_frames = len(track_data)
        msd_values = np.zeros(max_lagtime)
        counts = np.zeros(max_lagtime)

        for lag in range(1, max_lagtime + 1):
            if lag < n_frames:
                displacements = (track_data[['x_um', 'y_um']].iloc[lag:].values - track_data[['x_um', 'y_um']].iloc[:-lag].values) ** 2
                squared_displacements = np.sum(displacements, axis=1)
                msd_values[lag - 1] = np.mean(squared_displacements)
                counts[lag - 1] = len(squared_displacements)
            else:
                break

        avg_msd = np.mean(msd_values)  # Calculate the average MSD for the track (units: μm²)

        # Calculate total time in seconds for the track
        total_time_s = (track_data['time_s'].iloc[-1] - track_data['time_s'].iloc[0])
        lag_times = np.arange(1, max_lagtime + 1) * (total_time_s / (n_frames - 1))
        popt, _ = scipy.optimize.curve_fit(self.msd_model, lag_times, msd_values[:max_lagtime])
        D, alpha = popt[0], popt[1]

        # Classify the type of motion
        if alpha < 1:
            motion_class = 'subdiffusive'
        elif alpha > 1:
            motion_class = 'superdiffusive'
        else:
            motion_class = 'normal'

        return avg_msd, D, alpha, motion_class

    def produce_time_averaged_df(self, max_lagtime=None):
        """
        Produce the time-averaged DataFrame.
        Parameters:
        - max_lagtime: maximum number of frames to consider for lag times
        """
        if max_lagtime is None:
            max_lagtime = self.calculate_default_max_lagtime()

        time_averaged_list = []

        for unique_id, track_data in tqdm(self.metrics_df.groupby('unique_id'), desc="Producing Time-Averaged DataFrame"):
            avg_msd, D, alpha, motion_class = self.calculate_msd_for_track(track_data, max_lagtime)

            # Calculate total time in seconds for the track
            total_time_s = (track_data['time_s'].iloc[-1] - track_data['time_s'].iloc[0])

            # Calculate average instantaneous metrics for the track
            avg_speed = track_data['speed_um_s'].mean()
            avg_acceleration = track_data['acceleration_um_s2'].mean()
            avg_jerk = track_data['jerk_um_s3'].mean()
            avg_norm_curvature = track_data['normalized_curvature'].mean()
            avg_angle_norm_curvature = track_data['angle_normalized_curvature'].mean()

            # Add track-level summary information to time_averaged_df
            start_row = track_data.iloc[0]
            end_row = track_data.iloc[-1]
            track_summary = pd.DataFrame({
                'x_um_start': [start_row['x_um']],
                'y_um_start': [start_row['y_um']],
                'x_um_end': [end_row['x_um']],
                'y_um_end': [end_row['y_um']],
                'particle': [start_row['particle']],
                'condition': [start_row['condition']],
                'filename': [start_row['filename']],
                'file_id': [start_row['file_id']],
                'unique_id': [unique_id],
                'avg_msd': [avg_msd],  # Add the average MSD (units: μm²)
                'n_frames': [len(track_data)],  # Add the number of frames
                'total_time_s': [total_time_s],  # Add the total time in seconds
                'Location': [start_row['Location']],  # Add the Location
                'diffusion_coefficient': [D],  # Add the diffusion coefficient
                'anomalous_exponent': [alpha],  # Add the anomalous exponent
                'motion_class': [motion_class],  # Add the motion class
                'avg_speed_um_s': [avg_speed],  # Add average speed
                'avg_acceleration_um_s2': [avg_acceleration],  # Add average acceleration
                'avg_jerk_um_s3': [avg_jerk],  # Add average jerk
                'avg_normalized_curvature': [avg_norm_curvature],  # Add average normalized curvature
                'avg_angle_normalized_curvature': [avg_angle_norm_curvature],  # Add average angle normalized curvature
            })

            time_averaged_list.append(track_summary)

        self.time_averaged_df = pd.concat(time_averaged_list).reset_index(drop=True)

    def calculate_time_windowed_metrics(self, window_size=None, overlap=None):
        """
        Calculate metrics for each time window.
        Parameters:
        - window_size: size of the time window in frames
        - overlap: number of overlapping frames between windows
        """
        if window_size is None:
            window_size = self.calculate_default_window_size()
        if overlap is None:
            overlap = int(window_size / 2)  # Default overlap is half the window size

        # Print the calculated or provided window size and overlap
        print(f"Using window size: {window_size} frames: please note, tracks shorter than the window size will be skipped")
        print(f"Using overlap: {overlap} frames")

        windowed_list = []

        for unique_id, track_data in tqdm(self.metrics_df.groupby('unique_id'), desc="Calculating Time-Windowed Metrics"):
            n_frames = len(track_data)

            for start in range(0, n_frames - window_size + 1, window_size - overlap):
                end = start + window_size
                window_data = track_data.iloc[start:end]

                if len(window_data) < window_size:
                    continue

                # Calculate metrics for the window
                avg_msd, D, alpha, motion_class = self.calculate_msd_for_track(window_data, max_lagtime=min(100, int(window_size / 2)))

                # Calculate total time in seconds for the window
                total_time_s = (window_data['time_s'].iloc[-1] - window_data['time_s'].iloc[0])

                # Calculate average instantaneous metrics for the window
                avg_speed = window_data['speed_um_s'].mean()
                avg_acceleration = window_data['acceleration_um_s2'].mean()
                avg_jerk = window_data['jerk_um_s3'].mean()
                avg_norm_curvature = window_data['normalized_curvature'].mean()
                avg_angle_norm_curvature = window_data['angle_normalized_curvature'].mean()

                # Add window-level summary information to time_windowed_df
                start_row = window_data.iloc[0]
                end_row = window_data.iloc[-1]
                window_summary = pd.DataFrame({
                    'time_window': [start // (window_size - overlap)],
                    'x_um_start': [start_row['x_um']],
                    'y_um_start': [start_row['y_um']],
                    'x_um_end': [end_row['x_um']],
                    'y_um_end': [end_row['y_um']],
                    'particle': [start_row['particle']],
                    'condition': [start_row['condition']],
                    'filename': [start_row['filename']],
                    'file_id': [start_row['file_id']],
                    'unique_id': [unique_id],
                    'avg_msd': [avg_msd],  # Add the average MSD (units: μm²)
                    'n_frames': [window_size],  # Add the number of frames
                    'total_time_s': [total_time_s],  # Add the total time in seconds
                    'Location': [start_row['Location']],  # Add the Location
                    'diffusion_coefficient': [D],  # Add the diffusion coefficient
                    'anomalous_exponent': [alpha],  # Add the anomalous exponent
                    'motion_class': [motion_class],  # Add the motion class
                    'avg_speed_um_s': [avg_speed],  # Add average speed
                    'avg_acceleration_um_s2': [avg_acceleration],  # Add average acceleration
                    'avg_jerk_um_s3': [avg_jerk],  # Add average jerk
                    'avg_normalized_curvature': [avg_norm_curvature],  # Add average normalized curvature
                    'avg_angle_normalized_curvature': [avg_angle_norm_curvature],  # Add average angle normalized curvature
                })

                windowed_list.append(window_summary)

        self.time_windowed_df = pd.concat(windowed_list).reset_index(drop=True)

    def calculate_metrics_for_window(self, window_data):
        """
        Calculate metrics for a given window of data.
        """
        n_frames = len(window_data)
        max_lagtime = min(100, int(n_frames / 2))  # Example: use half the length of the window or 100, whichever is smaller

        msd_values = np.zeros(max_lagtime)

        for lag in range(1, max_lagtime + 1):
            if lag < n_frames:
                displacements = (window_data[['x_um', 'y_um']].iloc[lag:].values - window_data[['x_um', 'y_um']].iloc[:-lag].values) ** 2
                squared_displacements = np.sum(displacements, axis=1)
                msd_values[lag - 1] = np.mean(squared_displacements)
            else:
                break

        avg_msd = np.mean(msd_values)  # Calculate the average MSD for the window (units: μm²)

        # Fit MSD data to determine the type of motion and extract parameters
        total_time_s = (window_data['time_s'].iloc[-1] - window_data['time_s'].iloc[0])
        lag_times = np.arange(1, max_lagtime + 1) * (total_time_s / (n_frames - 1))
        popt, _ = scipy.optimize.curve_fit(self.msd_model, lag_times, msd_values[:max_lagtime])
        D, alpha = popt[0], popt[1]

        # Classify the type of motion
        if alpha < 1:
            motion_class = 'subdiffusive'
        elif alpha > 1:
            motion_class = 'superdiffusive'
        else:
            motion_class = 'normal'

        return avg_msd, D, alpha, motion_class

    def calculate_default_window_size(self):
        """
        Calculate the default window size based on the average length of tracks in the dataset.
        """
        avg_track_length = self.metrics_df.groupby('unique_id').size().mean()
        default_window_size = int(avg_track_length / 2)  # Example: use half the average length of the tracks
        return default_window_size

    def calculate_all_features(self, max_lagtime=None, calculate_time_windowed=False):
        """
        Calculate all features for the particle tracking data.
        This method will call all individual feature calculation methods.
        Parameters:
        - max_lagtime: maximum number of frames to consider for lag times
        - calculate_time_windowed: boolean flag to indicate if time-windowed metrics should be calculated
        """
        # Calculate default max lag time if not provided
        if max_lagtime is None:
            max_lagtime = self.calculate_default_max_lagtime()

        # Calculate distances between consecutive frames
        self.calculate_distances()

        # Calculate speeds between consecutive frames
        self.calculate_speeds()
        
        # Calculate directions of motion between consecutive frames
        self.calculate_directions()
        
        # Calculate accelerations between consecutive frames
        self.calculate_accelerations()
        
        # Calculate jerks between consecutive frames
        self.calculate_jerk()
        
        # Calculate curvatures between consecutive frames
        self.calculate_normalized_curvature()
        self.calculate_angle_normalized_curvature()

        # Calculate instantaneous diffusion coefficient between consecutive frames
        self.calculate_instantaneous_diffusion_coefficient()
        
        # Calculate net displacement
        self.calculate_net_displacement()

        # Calculate MSD for each track and aggregate
        self.calculate_msd_for_all_tracks(max_lagtime)

        # Calculate instantaneous velocity
        self.calculate_instantaneous_velocity()
        
        # Calculate time-windowed metrics if requested
        if calculate_time_windowed:
            self.calculate_time_windowed_metrics()
        
        # Cleanup step to remove temporary columns
        self.cleanup()
        
        return self.metrics_df

    def get_time_averaged_df(self):
        """
        Return the DataFrame with time-averaged metrics.
        """
        return self.time_averaged_df

    def get_time_windowed_df(self):
        """
        Return the DataFrame with time-windowed metrics.
        """
        return self.time_windowed_df

    def calculate_default_max_lagtime(self):
        """
        Calculate the default maximum lag time based on the shortest track in the dataset.
        """
        min_track_length = self.metrics_df.groupby('unique_id').size().min()
        default_max_lagtime = min(100, int(min_track_length / 2))  # Example: use half the length of the shortest track or 100, whichever is smaller
        return default_max_lagtime

    def calculate_msd_for_all_tracks(self, max_lagtime):
        """
        Calculate the MSD for all tracks and produce the time-averaged DataFrame.
        Parameters:
        - max_lagtime: maximum number of frames to consider for lag times
        """
        self.produce_time_averaged_df(max_lagtime)

    def cleanup(self):
        """
        Cleanup the dataframe by dropping unnecessary columns after all features are calculated.
        """
        self.metrics_df.drop(columns=[
            'x_um_prev', 'y_um_prev', 'time_s_prev', 'delta_time_s', 
            'speed_um_s_prev', 'acceleration_um_s2_prev', 'direction_rad_prev',
            'instant_velocity_x_um_s', 'instant_velocity_y_um_s',
        ], inplace=True)


































##############################################################################################
############################## NEW ONE END ########################################################

############## DEVELOPING NEW FEATURES #############################









































########### This version is for SPOTon ############ Probably, for that, its better to create a totally new class that just does it itself, without the need to remake it ##############

# import pandas as pd
# import numpy as np
# import lmfit

# class ParticleMetrics:
#     def __init__(self, df):
#         self.df = df
#         self.metrics_df = self.df.copy()

#     def calculate_distances(self):
#         """
#         Calculate the distances between consecutive frames for each particle in micrometers.
#         """
#         self.metrics_df = self.metrics_df.sort_values(by=['unique_id', 'frame'])
#         self.metrics_df[['x_um_prev', 'y_um_prev']] = self.metrics_df.groupby('unique_id')[['x_um', 'y_um']].shift(1)
#         self.metrics_df['segment_len_um'] = np.sqrt(
#             (self.metrics_df['x_um'] - self.metrics_df['x_um_prev'])**2 + 
#             (self.metrics_df['y_um'] - self.metrics_df['y_um_prev'])**2
#         )
#         # Fill NaN values with 0
#         self.metrics_df['segment_len_um'] = self.metrics_df['segment_len_um'].fillna(0)
#         return self.metrics_df

#     def calculate_speeds(self):
#         """
#         Calculate the speed between consecutive frames for each particle in micrometers per second.
#         """
#         self.metrics_df[['time_s_prev']] = self.metrics_df.groupby('unique_id')[['time_s']].shift(1)
#         self.metrics_df['delta_time_s'] = self.metrics_df['time_s'] - self.metrics_df['time_s_prev']
#         self.metrics_df['speed_um_s'] = self.metrics_df['segment_len_um'] / self.metrics_df['delta_time_s']
#         # Fill NaN and infinite values with 0
#         self.metrics_df['speed_um_s'] = self.metrics_df['speed_um_s'].replace([np.inf, -np.inf], np.nan).fillna(0)
#         return self.metrics_df

#     def calculate_jump_length_distribution(self):
#         """
#         Calculate the jump length distribution for each particle.
#         """
#         self.metrics_df = self.calculate_distances()  # Ensure distances are calculated
#         self.metrics_df['jump_length'] = self.metrics_df.groupby('unique_id')['segment_len_um'].shift(-1)
#         # Removing NaN values in the jump_length column
#         self.metrics_df = self.metrics_df.dropna(subset=['jump_length'])
#         return self.metrics_df

#     def fit_jump_length_distribution(self):
#         """
#         Fit the jump length distribution to a 2-state model.
#         """
#         def wrapped_jump_length_2states(x, D_free, D_bound, F_bound, sigma):
#             """Wrapper for the main fit function assuming global variables"""
#             y1 = (1 - F_bound) * (x / (2 * (D_free + sigma**2))) * np.exp(-(x**2) / (4 * (D_free + sigma**2)))
#             y2 = F_bound * (x / (2 * (D_bound + sigma**2))) * np.exp(-(x**2) / (4 * (D_bound + sigma**2)))
#             return y1 + y2

#         model = lmfit.Model(wrapped_jump_length_2states, independent_vars=['x'])
        
#         # Setting initial parameter values and bounds
#         params = model.make_params(D_free=1.0, D_bound=0.1, F_bound=0.5, sigma=0.05)
#         params['D_free'].set(min=0.0)
#         params['D_bound'].set(min=0.0)
#         params['F_bound'].set(min=0.0, max=1.0)
#         params['sigma'].set(min=0.0)

#         # Prepare the data for fitting
#         y, x_edges = np.histogram(self.metrics_df['jump_length'], bins=30, density=True)
#         x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        
#         # Remove zero values to avoid NaNs in log scale
#         x_centers = x_centers[y > 0]
#         y = y[y > 0]

#         # Fit the model to the data
#         result = model.fit(y, params, x=x_centers)
        
#         if result.success:
#             return result.best_values
#         else:
#             print("Fit was not successful. Check the data and initial parameter guesses.")
#             return {}

#     def calculate_all_features(self):
#         """
#         Calculate all features for the particle tracking data.
#         This method will call all individual feature calculation methods.
#         """
#         # Calculate distances between consecutive frames
#         self.calculate_distances()

#         # Calculate speeds between consecutive frames
#         self.calculate_speeds()
        
#         # Calculate jump length distribution
#         self.calculate_jump_length_distribution()
        
#         # Fit jump length distribution to model
#         fit_results = self.fit_jump_length_distribution()
#         print(f"Best fit parameters: {fit_results}")
        
#         # Cleanup step to remove temporary columns
#         self.cleanup()
        
#         return self.metrics_df

#     def cleanup(self):
#         """
#         Cleanup the dataframe by dropping unnecessary columns after all features are calculated.
#         """
#         self.metrics_df.drop(columns=['x_um_prev', 'y_um_prev', 'time_s_prev', 'delta_time_s'], inplace=True)

#     def get_metrics_df(self):
#         """
#         Return the dataframe with calculated metrics.
#         """
#         return self.metrics_df

########## THis is the end of the spoton version ###########








