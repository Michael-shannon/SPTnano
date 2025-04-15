




import numpy as np
import pandas as pd
from scipy.stats import vonmises
from tqdm.notebook import tqdm
import scipy.optimize
import re
import config

class ParticleMetrics:
    def __init__(self, df, time_between_frames=None, tolerance=None):
        self.df = df.copy()
        self.df['Location'] = self.df['filename'].apply(self.extract_location)  # Add Location column
        self.metrics_df = self.df.copy()

        # Retrieve time_between_frames from config if not provided
        self.time_between_frames = time_between_frames if time_between_frames is not None else config.TIME_BETWEEN_FRAMES
        
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

        self.msd_lagtime_df = pd.DataFrame(columns=[
            'unique_id', 'time_window', 'lag_time', 'msd', 'diffusion_coefficient', 'anomalous_exponent'
        ])

                # Calculate tolerance if not provided
        self.tolerance = tolerance or self.calculate_tolerance()

    def calculate_tolerance(self):
        # Determine tolerance from existing data if available
        if len(self.msd_lagtime_df) > 0:
            alpha_std = self.msd_lagtime_df['anomalous_exponent'].std()
            return alpha_std / 2  # Use half of the standard deviation
        else:
            return 0.1  # Default value if no previous data is available


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
        Skip calculation if 'segment_len_um' already exists.
        """
        if 'segment_len_um' not in self.metrics_df.columns:
            self.metrics_df = self.metrics_df.sort_values(by=['unique_id', 'frame'])
            self.metrics_df[['x_um_prev', 'y_um_prev']] = self.metrics_df.groupby('unique_id')[['x_um', 'y_um']].shift(1)
            self.metrics_df['segment_len_um'] = np.sqrt(
                (self.metrics_df['x_um'] - self.metrics_df['x_um_prev'])**2 + 
                (self.metrics_df['y_um'] - self.metrics_df['y_um_prev'])**2
            )
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
        # self.metrics_df['normalized_curvature_deg'] = np.degrees(self.metrics_df['normalized_curvature'])
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
        # Convert columns from radians to degrees

        # self.metrics_df['angle_normalized_curvature_deg'] = np.degrees(self.metrics_df['angle_normalized_curvature'])
        return self.metrics_df
    
    def calculate_vonmises_kappa(self, window_data):
        """
        Fit von Mises distributions to the turning angles and absolute directions
        in the given window data.
        
        Parameters:
        - window_data: DataFrame containing tracking data for the window.
        
        Returns:
        - kappa_turning: Estimated concentration parameter for turning angles.
        - kappa_absolute: Estimated concentration parameter for absolute directions.
        """
        
 

        # Use the pre-computed columns:
        # For turning angles, we are using "angle_normalized_curvature"
        turning_angles = window_data['angle_normalized_curvature'].values
        # For absolute directions, use "direction_rad"
        absolute_angles = window_data['direction_rad'].values

        # Fit the von Mises distribution for turning angles.
        try:
            # Optionally you can set fscale=1 to fix the scale parameter.
            params_turning = vonmises.fit(turning_angles, fscale=1)
            kappa_turning = params_turning[1]
        except Exception as e:
            print(f"Von Mises fit for turning angles failed: {e}")
            kappa_turning = np.nan

        # Fit the von Mises distribution for absolute directions.
        try:
            params_absolute = vonmises.fit(absolute_angles, fscale=1)
            kappa_absolute = params_absolute[1]
        except Exception as e:
            print(f"Von Mises fit for absolute angles failed: {e}")
            kappa_absolute = np.nan

        return kappa_turning, kappa_absolute

    

    # def calculate_persistence_length(self, track_data):
    #     """
    #     Calculate the persistence length for a given track.
    #     Parameters:
    #     - track_data: DataFrame containing the track data.
    #     Returns:
    #     - persistence_length: Calculated persistence length for the track.
    #     """
    #     directions = track_data['direction_rad']
    #     # Calculate directional correlation: <cos(theta_i - theta_j)>
    #     direction_diffs = directions.diff().dropna()
    #     correlation = np.cos(direction_diffs).mean()
    #         # Check if correlation is greater than zero before calculating log
    #     if correlation > 0:
    #         persistence_length = -1 / np.log(correlation)
    #     else:
    #         persistence_length = np.nan  # Assign NaN if the correlation is zero or negative
    

    #     # persistence_length = -1 / np.log(correlation) if correlation != 0 else np.nan
    #     return persistence_length
    
    def calculate_persistence_length(self, track_data): #updated 3-6-2025
        """
        Calculate the persistence length for a given track (or time window)
        in physical units. If the directional correlation is non-positive, returns 0.
        
        Parameters:
        - track_data: DataFrame containing the track data (or time window data).
        
        Returns:
        - persistence_length: Calculated persistence length for the track in physical units.
        """
        directions = track_data['direction_rad']
        # Compute the differences in direction between consecutive frames.
        direction_diffs = directions.diff().dropna()
        # Calculate the average cosine of these differences.
        correlation = np.cos(direction_diffs).mean()
        
        if correlation > 0:
            # Persistence length in "step" units.
            persistence_length_steps = -1 / np.log(correlation)
            # Calculate the average segment length (in physical units, e.g., micrometers) over this window.
            avg_segment_length = track_data['segment_len_um'].mean()
            # Scale to obtain the persistence length in physical units.
            persistence_length = persistence_length_steps * avg_segment_length
        else:
            # Instead of returning NaN, return 0.
            persistence_length = 0

        return persistence_length
    


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
    

    def calculate_cum_displacement(self): #ADDED 3_19_2025
        # Ensure that segment lengths are computed
        self.calculate_distances()
        # Compute cumulative (total) displacement for each particle
        self.metrics_df['cum_displacement_um'] = self.metrics_df.groupby('unique_id')['segment_len_um'].cumsum()
        return self.metrics_df
    
    # def calculate_cumulative_displacement(df):
    #     """
    #     Calculates the cumulative displacement (total distance traveled) for each particle.
        
    #     Parameters:
    #     - df: DataFrame containing columns 'unique_id', 'frame', 'x_um', and 'y_um'.
        
    #     Returns:
    #     - df: DataFrame with a new column 'cumulative_displacement_um', which is the
    #         cumulative sum of distances (segment lengths) between consecutive frames.
    #     """
    #     # Ensure the data is sorted by unique_id and frame
    #     df = df.sort_values(by=['unique_id', 'frame']).copy()
        
    #     # Compute segment lengths if not already present
    #     if 'segment_len_um' not in df.columns:
    #         # Calculate previous positions for each particle
    #         df[['x_um_prev', 'y_um_prev']] = df.groupby('unique_id')[['x_um', 'y_um']].shift(1)
    #         # Compute the Euclidean distance between consecutive positions
    #         df['segment_len_um'] = ((df['x_um'] - df['x_um_prev'])**2 +
    #                                 (df['y_um'] - df['y_um_prev'])**2)**0.5
    #         # Replace NaN for the first frame of each particle with 0
    #         df['segment_len_um'] = df['segment_len_um'].fillna(0)
        
    #     # Compute the cumulative sum of segment lengths for each particle
    #     df['cumulative_displacement_um'] = df.groupby('unique_id')['segment_len_um'].cumsum()
    #     return df


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
    
    # def calculate_msd_for_track(self, track_data, store_msd=False, time_window=None):
    #     n_frames = len(track_data)
    #     if n_frames < 3:
    #         print(f"Skipping unique_id {track_data['unique_id'].iloc[0]}: insufficient frames ({n_frames})")
    #         return np.nan, np.nan, np.nan, 'unlabeled'

    #     # Use all possible lag times (frames - 1)
    #     lag_times = np.arange(1, n_frames) * self.time_between_frames
    #     msd_values = np.zeros(len(lag_times))

    #     for lag in range(1, len(lag_times) + 1):
    #         displacements = (track_data[['x_um', 'y_um']].iloc[lag:].values -
    #                         track_data[['x_um', 'y_um']].iloc[:-lag].values) ** 2
    #         squared_displacements = np.sum(displacements, axis=1)
    #         msd_values[lag - 1] = np.mean(squared_displacements)

    #     # print(f"DEBUG: Unique ID {track_data['unique_id'].iloc[0]}") #USEFUL FOR DEBUGGING!
    #     # print(f"  - Total frames: {n_frames}")
    #     # print(f"  - Total lag times: {len(lag_times)}")
    #     # print(f"  - Lag times: {lag_times}")
    #     # print(f"  - MSD values: {msd_values}")

    #     # Check if sufficient lag times are available
    #     if len(lag_times) < 3:
    #         print(f"Skipping unique_id {track_data['unique_id'].iloc[0]}: insufficient lag times ({len(lag_times)})")
    #         return np.nan, np.nan, np.nan, 'unlabeled'

    #     try:
    #         popt, _ = scipy.optimize.curve_fit(self.msd_model, lag_times, msd_values)
    #         D, alpha = popt[0], popt[1]
    #     except RuntimeError:
    #         print(f"Curve fitting failed for unique_id {track_data['unique_id'].iloc[0]}")
    #         return np.nan, np.nan, np.nan, 'unlabeled'

    #     # print(f"  - Diffusion coefficient (D): {D:.4f}")
    #     # print(f"  - Anomalous exponent (alpha): {alpha:.4f}")

    #     # Classify motion
    #     if alpha < 1 - self.tolerance:
    #         motion_class = 'subdiffusive'
    #     elif alpha > 1 + self.tolerance:
    #         motion_class = 'superdiffusive'
    #     else:
    #         motion_class = 'normal'

    #     # print(f"  - Motion class: {motion_class}")

    #     # Store MSD values if requested
    #     if store_msd and time_window is not None:
    #         msd_records = [
    #             {
    #                 'unique_id': track_data['unique_id'].iloc[0],
    #                 'time_window': time_window,
    #                 'lag_time': lag,
    #                 'msd': msd,
    #                 'diffusion_coefficient': D,
    #                 'anomalous_exponent': alpha
    #             }
    #             for lag, msd in zip(lag_times, msd_values)
    #         ]
    #         if msd_records:
    #             msd_records_df = pd.DataFrame(msd_records)
    #             self.msd_lagtime_df = pd.concat([self.msd_lagtime_df, msd_records_df], ignore_index=True)

    #     return np.mean(msd_values), D, alpha, motion_class

########### THIS VERSION REMOVED 4-15-2025 BELOW ################

    # def calculate_msd_for_track(self, track_data, store_msd=False, time_window=None, #ADDED 3_19_2025
    #                             use_bounds=False, max_D=None, allow_partial_window=True,
    #                             min_window_size=60):
    #     """
    #     Calculate the MSD for a given track (or window of a track) and fit the MSD model.
        
    #     Parameters:
    #     - track_data: DataFrame for the track or window.
    #     - store_msd: If True, store individual lag time MSD records.
    #     - time_window: An identifier for the time window (if applicable).
    #     - use_bounds: Whether to apply parameter bounds in curve fitting.
    #     - max_D: Upper bound for the diffusion coefficient if using bounds.
    #     - allow_partial_window: If False and this is a window (time_window is not None),
    #         skip windows with fewer than min_window_size frames.
    #     - min_window_size: Minimum number of frames required in a window when not allowing partial windows.
        
    #     Returns:
    #     A tuple of (avg_msd, D, alpha, motion_class)
    #     """
    #     n_frames = len(track_data)
        
    #     # For full tracks, require at least 3 frames.
    #     if time_window is None:
    #         if n_frames < 3:
    #             print(f"Skipping unique_id {track_data['unique_id'].iloc[0]}: insufficient frames ({n_frames})")
    #             return np.nan, np.nan, np.nan, 'unlabeled'
    #     else:
    #         # For time windows, enforce minimum window size unless allowed.
    #         if not allow_partial_window and n_frames < min_window_size:
    #             print(f"Skipping unique_id {track_data['unique_id'].iloc[0]} for time window {time_window}: "
    #                 f"insufficient frames ({n_frames} < required {min_window_size})")
    #             return np.nan, np.nan, np.nan, 'unlabeled'
        
    #     # Compute lag times and corresponding MSD values.
    #     lag_times = np.arange(1, n_frames) * self.time_between_frames
    #     msd_values = np.zeros(len(lag_times))
    #     for lag in range(1, len(lag_times) + 1):
    #         displacements = (track_data[['x_um', 'y_um']].iloc[lag:].values -
    #                         track_data[['x_um', 'y_um']].iloc[:-lag].values) ** 2
    #         squared_displacements = np.sum(displacements, axis=1)
    #         msd_values[lag - 1] = np.mean(squared_displacements)
        
    #     if len(lag_times) < 3:
    #         print(f"Skipping unique_id {track_data['unique_id'].iloc[0]}: insufficient lag times ({len(lag_times)})")
    #         return np.nan, np.nan, np.nan, 'unlabeled'
        
    #     # Fit the MSD model.
    #     try:
    #         if use_bounds:
    #             popt, _ = scipy.optimize.curve_fit(self.msd_model, lag_times, msd_values,
    #                                             bounds=([0, -np.inf], [max_D, np.inf]))
    #         else:
    #             popt, _ = scipy.optimize.curve_fit(self.msd_model, lag_times, msd_values)
    #         D, alpha = popt[0], popt[1]
    #     except RuntimeError:
    #         print(f"Curve fitting failed for unique_id {track_data['unique_id'].iloc[0]}")
    #         return np.nan, np.nan, np.nan, 'unlabeled'
        
    #     # Classify the motion type.
    #     if alpha < 1 - self.tolerance:
    #         motion_class = 'subdiffusive'
    #     elif alpha > 1 + self.tolerance:
    #         motion_class = 'superdiffusive'
    #     else:
    #         motion_class = 'normal'
        
    #     # Post-hoc filtering: if D is hitting the upper bound, mark the classification as "overestimated".
    #     if max_D is not None and D >= max_D * 0.99: #had to change this 4-14-2025 because it was not enjoying the 'None' thing
    #         motion_class = f"overestimated_{motion_class}"
        
    #     # Optionally store the per-lag MSD values.
    #     if store_msd and time_window is not None:
    #         msd_records = [
    #             {
    #                 'unique_id': track_data['unique_id'].iloc[0],
    #                 'time_window': time_window,
    #                 'lag_time': lag,
    #                 'msd': msd,
    #                 'diffusion_coefficient': D,
    #                 'anomalous_exponent': alpha
    #             }
    #             for lag, msd in zip(lag_times, msd_values)
    #         ]
    #         if msd_records:
    #             msd_records_df = pd.DataFrame(msd_records)
    #             self.msd_lagtime_df = pd.concat([self.msd_lagtime_df, msd_records_df], ignore_index=True)
        
    #     return np.mean(msd_values), D, alpha, motion_class

############### THIS VERSION REMOVED 4-15-2025 ABOVE ################################

### NEW ONE BELOW ###

    # def calculate_msd_for_track(self, track_data, store_msd=False, time_window=None, 
    #                             use_bounds=False, max_D=None, allow_partial_window=True,
    #                             min_window_size=60, **kwargs):
    #     """
    #     Calculate the MSD for a given track (or window) and fit the MSD model.
        
    #     Parameters:
    #     - track_data: DataFrame for the track/window.
    #     - store_msd: If True, store the individual per-lag MSD records.
    #     - time_window: Identifier for the time window (if applicable).
    #     - use_bounds: Whether to enforce parameter bounds in curve fitting.
    #     - max_D: Upper bound for the diffusion coefficient (if bounds are used).
    #     - allow_partial_window: If False (for windows), skip windows with fewer than min_window_size frames.
    #     - min_window_size: Minimum number of frames required for a window.
        
    #     Additional **kwargs:
    #     - r2_threshold: The minimum acceptable R² value (on log–log data) for a good fit (default: 0.95).
    #     - bad_fit_strategy: Strategy for handling bad fits. Options are:
    #         "remove_track" (remove entire track),
    #         "excise_window" (excise just this window, splitting the track),
    #         "flag" (keep the window but mark it with a flag).
    #         Default is "flag".
        
    #     Returns:
    #     A tuple of (avg_msd, D, alpha, motion_class) where motion_class may include
    #     a bad-fit flag if applicable.
    #     """
    #     import numpy as np
    #     import scipy.optimize
        
    #     n_frames = len(track_data)
        
    #     # For full tracks, require at least 3 frames.
    #     if time_window is None:
    #         if n_frames < 3:
    #             print(f"Skipping unique_id {track_data['unique_id'].iloc[0]}: insufficient frames ({n_frames})")
    #             return np.nan, np.nan, np.nan, 'unlabeled'
    #     else:
    #         # For time windows, enforce minimum window size if not allowing partials.
    #         if not allow_partial_window and n_frames < min_window_size:
    #             print(f"Skipping unique_id {track_data['unique_id'].iloc[0]} for time window {time_window}: "
    #                 f"insufficient frames ({n_frames} < required {min_window_size})")
    #             return np.nan, np.nan, np.nan, 'unlabeled'
        
    #     # Compute lag times and MSD values.
    #     lag_times = np.arange(1, n_frames) * self.time_between_frames
    #     msd_values = np.zeros(len(lag_times))
    #     for lag in range(1, len(lag_times) + 1):
    #         displacements = (track_data[['x_um', 'y_um']].iloc[lag:].values -
    #                         track_data[['x_um', 'y_um']].iloc[:-lag].values) ** 2
    #         squared_displacements = np.sum(displacements, axis=1)
    #         msd_values[lag - 1] = np.mean(squared_displacements)
        
    #     if len(lag_times) < 3:
    #         print(f"Skipping unique_id {track_data['unique_id'].iloc[0]}: insufficient lag times ({len(lag_times)})")
    #         return np.nan, np.nan, np.nan, 'unlabeled'
        
    #     # Fit the MSD model.
    #     try:
    #         if use_bounds:
    #             popt, _ = scipy.optimize.curve_fit(self.msd_model, lag_times, msd_values,
    #                                             bounds=([0, -np.inf], [max_D, np.inf]))
    #         else:
    #             popt, _ = scipy.optimize.curve_fit(self.msd_model, lag_times, msd_values)
    #         D, alpha = popt[0], popt[1]
    #     except RuntimeError:
    #         print(f"Curve fitting failed for unique_id {track_data['unique_id'].iloc[0]}")
    #         return np.nan, np.nan, np.nan, 'bad_fit_removed_track'
        
    #     # Compute predicted MSD values from the fitted model.
    #     predicted_msd = self.msd_model(lag_times, D, alpha)
        
    #     # Compute R² on log–log data to assess the goodness of fit.
    #     # (Avoid zeros by assuming msd_values > 0.)
    #     log_obs = np.log10(msd_values)
    #     log_pred = np.log10(predicted_msd)
    #     ss_res = np.sum((log_obs - log_pred) ** 2)
    #     ss_tot = np.sum((log_obs - np.mean(log_obs)) ** 2)
    #     r_squared = 1 - ss_res / ss_tot
        
    #     # Read additional parameters.
    #     r2_threshold = kwargs.get("r2_threshold", 0.95)
    #     bad_fit_strategy = kwargs.get("bad_fit_strategy", "flag")
        
    #     # If the fit is poor, handle it according to the chosen strategy.
    #     if r_squared < r2_threshold:
    #         if bad_fit_strategy == "remove_track":
    #             print(f"Bad fit for unique_id {track_data['unique_id'].iloc[0]}: R² = {r_squared:.3f}. Removing entire track.")
    #             return np.nan, np.nan, np.nan, 'bad_fit_removed_track'
    #         elif bad_fit_strategy == "excise_window":
    #             print(f"Bad fit for unique_id {track_data['unique_id'].iloc[0]} time window {time_window}: R² = {r_squared:.3f}. Excising this window.")
    #             return np.nan, np.nan, np.nan, 'excised_bad_fit'
    #         elif bad_fit_strategy == "flag":
    #             print(f"Bad fit for unique_id {track_data['unique_id'].iloc[0]} time window {time_window}: R² = {r_squared:.3f}. Flagging this window.")
    #             # Append a flag to the motion class based on alpha.
    #             if alpha < 1 - self.tolerance:
    #                 motion_class = 'bad_fit_flagged_subdiffusive'
    #             elif alpha > 1 + self.tolerance:
    #                 motion_class = 'bad_fit_flagged_superdiffusive'
    #             else:
    #                 motion_class = 'bad_fit_flagged_normal'
    #             return np.mean(msd_values), D, alpha, motion_class
        
    #     # Normal motion classification.
    #     if alpha < 1 - self.tolerance:
    #         motion_class = 'subdiffusive'
    #     elif alpha > 1 + self.tolerance:
    #         motion_class = 'superdiffusive'
    #     else:
    #         motion_class = 'normal'
        
    #     # Post-hoc filtering: if D is hitting the upper bound, mark as "overestimated".
    #     if max_D is not None and D >= max_D * 0.99:
    #         motion_class = f"overestimated_{motion_class}"
        
    #     # Optionally store the per-lag MSD values.
    #     if store_msd and time_window is not None:
    #         msd_records = [
    #             {
    #                 'unique_id': track_data['unique_id'].iloc[0],
    #                 'time_window': time_window,
    #                 'lag_time': lag,
    #                 'msd': msd,
    #                 'diffusion_coefficient': D,
    #                 'anomalous_exponent': alpha
    #             }
    #             for lag, msd in zip(lag_times, msd_values)
    #         ]
    #         if msd_records:
    #             msd_records_df = pd.DataFrame(msd_records)
    #             self.msd_lagtime_df = pd.concat([self.msd_lagtime_df, msd_records_df], ignore_index=True)
        
    #     return np.mean(msd_values), D, alpha, motion_class
    
########### NEW ONE ABOVE ####################################
############ V3 below ########################### 
 ##################### GOLDEN below #########################################
    # def msd_normal(self, t, D):
    #     """
    #     Normal diffusion MSD model in 2D.
    #     For example, for 2-dimensional diffusion:
    #         MSD(t) = 4 * D * t
    #     """
    #     return 4 * D * t

    # def calculate_msd_for_track(self, track_data, store_msd=False, time_window=None, 
    #                             use_bounds=False, allow_partial_window=True,
    #                             min_window_size=60, **kwargs):
    #     """
    #     Calculate the MSD for a given track (or window) and fit the MSD model.
        
    #     Parameters:
    #     - track_data: DataFrame for the track or window.
    #     - store_msd: If True, store the per-lag MSD records.
    #     - time_window: An identifier for the time window (if applicable).
    #     - use_bounds: (Legacy parameter; now unused.)
    #     - allow_partial_window: If False and this is a window (time_window is not None),
    #                             skip windows with fewer than min_window_size frames.
    #     - min_window_size: Minimum number of frames required in a window.
        
    #     Additional **kwargs:
    #     - fit_method: "r2_threshold" (default) or "model_selection".  
    #                     In "r2_threshold" mode the goodness-of-fit is checked by an R² threshold.
    #                     In "model_selection" mode both a normal (alpha=1) and an anomalous model 
    #                     (alpha free) are fit and compared via AIC.
    #     - r2_threshold: R² threshold for a good fit for normal diffusion (default: 0.95).
    #     - anomalous_r2_threshold: R² threshold for anomalous diffusion (default: 0.80).
    #     - bad_fit_strategy: Strategy to handle a bad fit. Options are:
    #         "remove_track" (remove entire track),
    #         "excise_window" (excise this window and later split the track),
    #         "flag" (keep the window but mark it as flagged).
    #         Default is "flag".
        
    #     Returns:
    #     A tuple (avg_msd, D, alpha, motion_class) where motion_class may have markers for bad fits.
    #     """
    #     import numpy as np
    #     import scipy.optimize

    #     n_frames = len(track_data)
        
    #     # Check frame requirements.
    #     if time_window is None:
    #         if n_frames < 3:
    #             print(f"Skipping unique_id {track_data['unique_id'].iloc[0]}: insufficient frames ({n_frames})")
    #             return np.nan, np.nan, np.nan, 'unlabeled'
    #     else:
    #         if not allow_partial_window and n_frames < min_window_size:
    #             print(f"Skipping unique_id {track_data['unique_id'].iloc[0]} for time window {time_window}: "
    #                 f"insufficient frames ({n_frames} < required {min_window_size})")
    #             return np.nan, np.nan, np.nan, 'unlabeled'
        
    #     # Compute lag times and observed MSD values.
    #     lag_times = np.arange(1, n_frames) * self.time_between_frames
    #     msd_values = np.zeros(len(lag_times))
    #     for lag in range(1, len(lag_times) + 1):
    #         displacements = (track_data[['x_um', 'y_um']].iloc[lag:].values -
    #                         track_data[['x_um', 'y_um']].iloc[:-lag].values) ** 2
    #         squared_displacements = np.sum(displacements, axis=1)
    #         msd_values[lag - 1] = np.mean(squared_displacements)
        
    #     if len(lag_times) < 3:
    #         print(f"Skipping unique_id {track_data['unique_id'].iloc[0]}: insufficient lag times ({len(lag_times)})")
    #         return np.nan, np.nan, np.nan, 'unlabeled'
        
    #     # Get fitting options from kwargs.
    #     fit_method = kwargs.get("fit_method", "r2_threshold")
    #     r2_threshold = kwargs.get("r2_threshold", 0.95)
    #     anomalous_r2_threshold = kwargs.get("anomalous_r2_threshold", 0.80)
    #     bad_fit_strategy = kwargs.get("bad_fit_strategy", "flag")
        
    #     # Prepare the log10 of observed MSD for later comparison.
    #     log_obs = np.log10(msd_values)
    #     n = len(lag_times)
        
    #     if fit_method == "model_selection":
    #         # ---- Fit anomalous model (with free alpha) ----
    #         try:
    #             popt_anom, _ = scipy.optimize.curve_fit(self.msd_model, lag_times, msd_values)
    #             D_anom, alpha_anom = popt_anom[0], popt_anom[1]
    #             predicted_anom = self.msd_model(lag_times, D_anom, alpha_anom)
    #             log_pred_anom = np.log10(predicted_anom)
    #             rss_anom = np.sum((log_obs - log_pred_anom) ** 2)
    #         except RuntimeError:
    #             print(f"Curve fitting (anomalous) failed for unique_id {track_data['unique_id'].iloc[0]}")
    #             return np.nan, np.nan, np.nan, 'bad_fit_removed_track'
            
    #         # ---- Fit normal model (with alpha fixed = 1) ----
    #         try:
    #             popt_norm, _ = scipy.optimize.curve_fit(self.msd_normal, lag_times, msd_values)
    #             D_norm = popt_norm[0]
    #             predicted_norm = self.msd_normal(lag_times, D_norm)
    #             log_pred_norm = np.log10(predicted_norm)
    #             rss_norm = np.sum((log_obs - log_pred_norm) ** 2)
    #         except RuntimeError:
    #             print(f"Curve fitting (normal) failed for unique_id {track_data['unique_id'].iloc[0]}")
    #             # Fallback to the anomalous model if normal fit fails.
    #             D_norm, rss_norm = np.nan, np.inf
            
    #         # ---- Compute AIC for both models ----
    #         # AIC = 2*k + n*ln(RSS/n)
    #         aic_anom = 2 * 2 + n * np.log(rss_anom / n)
    #         aic_norm = 2 * 1 + n * np.log(rss_norm / n)
            
    #         # Select model with lower AIC.
    #         if aic_norm < aic_anom:
    #             chosen_model = "normal"
    #             D = D_norm
    #             alpha = 1.0
    #             chosen_rss = rss_norm
    #         else:
    #             chosen_model = "anomalous"
    #             D = D_anom
    #             alpha = alpha_anom
    #             chosen_rss = rss_anom
            
    #         # Compute R² from chosen model.
    #         ss_tot = np.sum((log_obs - np.mean(log_obs)) ** 2)
    #         chosen_r2 = 1 - chosen_rss / ss_tot
            
    #         # Decide which R² threshold applies.
    #         if chosen_model == "normal":
    #             threshold = r2_threshold
    #         else:
    #             threshold = anomalous_r2_threshold
            
    #         # If the fit is too poor, apply the bad-fit strategy.
    #         if chosen_r2 < threshold:
    #             if bad_fit_strategy == "remove_track":
    #                 print(f"Bad fit (model_selection) for unique_id {track_data['unique_id'].iloc[0]} (R² = {chosen_r2:.3f}). Removing entire track.")
    #                 return np.nan, np.nan, np.nan, 'bad_fit_removed_track'
    #             elif bad_fit_strategy == "excise_window":
    #                 print(f"Bad fit (model_selection) for unique_id {track_data['unique_id'].iloc[0]} time window {time_window} (R² = {chosen_r2:.3f}). Excising this window.")
    #                 return np.nan, np.nan, np.nan, 'excised_bad_fit'
    #             elif bad_fit_strategy == "flag":
    #                 print(f"Bad fit (model_selection) for unique_id {track_data['unique_id'].iloc[0]} time window {time_window} (R² = {chosen_r2:.3f}). Flagging this window.")
    #                 if chosen_model == "anomalous":
    #                     if alpha < 1 - self.tolerance:
    #                         motion_class = 'bad_fit_flagged_subdiffusive'
    #                     elif alpha > 1 + self.tolerance:
    #                         motion_class = 'bad_fit_flagged_superdiffusive'
    #                     else:
    #                         motion_class = 'bad_fit_flagged_normal'
    #                 else:
    #                     motion_class = 'bad_fit_flagged_normal'
    #                 return np.mean(msd_values), D, alpha, motion_class
    #         else:
    #             # If fit is acceptable, classify.
    #             if chosen_model == "normal":
    #                 motion_class = "normal"
    #             else:
    #                 if alpha < 1 - self.tolerance:
    #                     motion_class = "subdiffusive"
    #                 elif alpha > 1 + self.tolerance:
    #                     motion_class = "superdiffusive"
    #                 else:
    #                     motion_class = "normal"
    #         avg_msd = np.mean(msd_values)
        
    #     else:
    #         # ---- Fall back to R² threshold method (default) ----
    #         try:
    #             popt, _ = scipy.optimize.curve_fit(self.msd_model, lag_times, msd_values)
    #             D, alpha = popt[0], popt[1]
    #             predicted = self.msd_model(lag_times, D, alpha)
    #         except RuntimeError:
    #             print(f"Curve fitting failed for unique_id {track_data['unique_id'].iloc[0]}")
    #             return np.nan, np.nan, np.nan, 'bad_fit_removed_track'
            
    #         log_pred = np.log10(predicted)
    #         ss_res = np.sum((log_obs - log_pred) ** 2)
    #         ss_tot = np.sum((log_obs - np.mean(log_obs)) ** 2)
    #         chosen_r2 = 1 - ss_res / ss_tot
            
    #         # Use different thresholds based on alpha.
    #         if (alpha < 1 - self.tolerance) or (alpha > 1 + self.tolerance):
    #             threshold = anomalous_r2_threshold
    #         else:
    #             threshold = r2_threshold
            
    #         if chosen_r2 < threshold:
    #             if bad_fit_strategy == "remove_track":
    #                 print(f"Bad fit for unique_id {track_data['unique_id'].iloc[0]} (R² = {chosen_r2:.3f}). Removing entire track.")
    #                 return np.nan, np.nan, np.nan, 'bad_fit_removed_track'
    #             elif bad_fit_strategy == "excise_window":
    #                 print(f"Bad fit for unique_id {track_data['unique_id'].iloc[0]} time window {time_window} (R² = {chosen_r2:.3f}). Excising this window.")
    #                 return np.nan, np.nan, np.nan, 'excised_bad_fit'
    #             elif bad_fit_strategy == "flag":
    #                 print(f"Bad fit for unique_id {track_data['unique_id'].iloc[0]} time window {time_window} (R² = {chosen_r2:.3f}). Flagging this window.")
    #                 if alpha < 1 - self.tolerance:
    #                     motion_class = 'bad_fit_flagged_subdiffusive'
    #                 elif alpha > 1 + self.tolerance:
    #                     motion_class = 'bad_fit_flagged_superdiffusive'
    #                 else:
    #                     motion_class = 'bad_fit_flagged_normal'
    #                 return np.mean(msd_values), D, alpha, motion_class
    #         else:
    #             if alpha < 1 - self.tolerance:
    #                 motion_class = 'subdiffusive'
    #             elif alpha > 1 + self.tolerance:
    #                 motion_class = 'superdiffusive'
    #             else:
    #                 motion_class = 'normal'
    #         avg_msd = np.mean(msd_values)
        
    #     # Optionally store per-lag MSD records.
    #     if store_msd and (time_window is not None):
    #         import pandas as pd
    #         msd_records = [{
    #             'unique_id': track_data['unique_id'].iloc[0],
    #             'time_window': time_window,
    #             'lag_time': lag,
    #             'msd': msd,
    #             'diffusion_coefficient': D,
    #             'anomalous_exponent': alpha
    #         } for lag, msd in zip(lag_times, msd_values)]
    #         if msd_records:
    #             msd_records_df = pd.DataFrame(msd_records)
    #             self.msd_lagtime_df = pd.concat([self.msd_lagtime_df, msd_records_df], ignore_index=True)
        
    #     return avg_msd, D, alpha, motion_class
    



    # def calculate_time_windowed_metrics(self, window_size=None, overlap=None, filter_metrics_df=False, **kwargs):
    #     """
    #     Calculate metrics for time windows across all tracks.
        
    #     Any additional keyword arguments (kwargs) are passed to calculate_msd_for_track,
    #     including those controlling fit_method (e.g. "r2_threshold" vs "model_selection")
    #     and the bad-fit handling options.
        
    #     For bad fits, three strategies are available via 'bad_fit_strategy':
    #     - "remove_track": Remove all windows from the track.
    #     - "excise_window": Excise (skip) the offending window. If an excision occurs,
    #                         subsequent windows are given a new unique ID with the suffix '_s' (for split).
    #     - "flag": Keep the window but flag it (e.g. 'bad_fit_flagged_normal').
        
    #     Parameters:
    #     - window_size: Number of frames per window.
    #     - overlap: Number of overlapping frames.
    #     - filter_metrics_df: If True, update self.metrics_df to only include frames in windows.
    #     - **kwargs: Additional options to be forwarded to calculate_msd_for_track.
    #     """
    #     from tqdm import tqdm
    #     import numpy as np
    #     import pandas as pd
        
    #     if window_size is None:
    #         window_size = self.calculate_default_window_size()
    #     if overlap is None:
    #         overlap = int(window_size / 2)
        
    #     print("Note: Partial windows (below the window size) are not included in the time-windowed metrics.")
        
    #     windowed_list = []
    #     included_frames = set()
        
    #     for unique_id, track_data in tqdm(self.metrics_df.groupby('unique_id'), desc="Calculating Time-Windowed Metrics"):
    #         n_frames = len(track_data)
    #         track_removed = False
    #         current_unique_id = unique_id  # This may change if a bad window is excised.
            
    #         for start in range(0, n_frames - window_size + 1, window_size - overlap):
    #             end = start + window_size
    #             window_data = track_data.iloc[start:end].copy()
    #             included_frames.update(window_data['frame'].values)
                
    #             # Compute displacements.
    #             start_row = window_data.iloc[0]
    #             end_row = window_data.iloc[-1]
    #             net_disp = np.sqrt((end_row['x_um'] - start_row['x_um'])**2 +
    #                             (end_row['y_um'] - start_row['y_um'])**2)
                
    #             window_data['x_um_prev'] = window_data['x_um'].shift(1)
    #             window_data['y_um_prev'] = window_data['y_um'].shift(1)
    #             window_data['segment_len_um'] = np.sqrt(
    #                 (window_data['x_um'] - window_data['x_um_prev'])**2 +
    #                 (window_data['y_um'] - window_data['y_um_prev'])**2
    #             ).fillna(0)
    #             cum_disp = window_data['segment_len_um'].sum()
                
    #             # Call our MSD fitting function with the provided kwargs.
    #             avg_msd, D, alpha, motion_class = self.calculate_msd_for_track(
    #                 window_data,
    #                 store_msd=True,
    #                 time_window=start // (window_size - overlap),
    #                 **kwargs
    #             )
                
    #             # Handle bad-fit outcomes.
    #             if motion_class == 'bad_fit_removed_track':
    #                 print(f"Removing entire track for unique_id {unique_id} due to a bad fit.")
    #                 track_removed = True
    #                 break
    #             elif motion_class == 'excised_bad_fit':
    #                 if current_unique_id == unique_id:
    #                     current_unique_id = f"{unique_id}_s"
    #                     print(f"Track {unique_id}: first bad window excised; subsequent windows will have unique_id '{current_unique_id}'.")
    #                 continue  # Skip this window.
                
    #             total_time_s = window_data['time_s'].iloc[-1] - window_data['time_s'].iloc[0]
    #             avg_speed = window_data['speed_um_s'].mean()
    #             avg_acceleration = window_data['acceleration_um_s2'].mean()
    #             avg_jerk = window_data['jerk_um_s3'].mean()
    #             avg_norm_curvature = window_data['normalized_curvature'].mean()
    #             avg_angle_norm_curvature = window_data['angle_normalized_curvature'].mean()
    #             persistence_length = self.calculate_persistence_length(window_data)
                
    #             window_summary = pd.DataFrame({
    #                 'time_window': [start // (window_size - overlap)],
    #                 'x_um_start': [start_row['x_um']],
    #                 'y_um_start': [start_row['y_um']],
    #                 'x_um_end': [end_row['x_um']],
    #                 'y_um_end': [end_row['y_um']],
    #                 'particle': [start_row['particle']],
    #                 'condition': [start_row['condition']],
    #                 'filename': [start_row['filename']],
    #                 'file_id': [start_row['file_id']],
    #                 'unique_id': [current_unique_id],
    #                 'avg_msd': [avg_msd],
    #                 'n_frames': [window_size],
    #                 'total_time_s': [total_time_s],
    #                 'Location': [start_row['Location']],
    #                 'diffusion_coefficient': [D],
    #                 'anomalous_exponent': [alpha],
    #                 'motion_class': [motion_class],
    #                 'avg_speed_um_s': [avg_speed],
    #                 'avg_acceleration_um_s2': [avg_acceleration],
    #                 'avg_jerk_um_s3': [avg_jerk],
    #                 'avg_normalized_curvature': [avg_norm_curvature],
    #                 'avg_angle_normalized_curvature': [avg_angle_norm_curvature],
    #                 'persistence_length': [persistence_length],
    #                 'net_displacement_um': [net_disp],
    #                 'cum_displacement_um': [cum_disp],
    #                 'bad_fit_flag': [motion_class.startswith("bad_fit_flagged")]
    #             })
    #             windowed_list.append(window_summary)
            
    #         if track_removed:
    #             print(f"Skipping unique_id {unique_id} entirely due to a bad fit in one or more windows.")
    #             continue
        
    #     self.time_windowed_df = pd.concat(windowed_list).reset_index(drop=True)
        
    #     if filter_metrics_df:
    #         print("Filtering metrics_df to only include frames within time windows...")
    #         print(f"Initial number of frames: {len(self.metrics_df)}")
    #         self.metrics_df = self.metrics_df[self.metrics_df['frame'].isin(included_frames)].copy()
    #         print(f"Remaining frames after filtering: {len(self.metrics_df)}")

####### V3 ABOVE #############
    ############# V4 below ###############################

    def msd_normal(self, t, D):
        """
        Normal diffusion MSD model in 2D.
        For 2-dimensional diffusion:
            MSD(t) = 4 * D * t
        """
        return 4 * D * t


    def calculate_msd_for_track(self, track_data, store_msd=False, time_window=None, 
                                use_bounds=False, allow_partial_window=True,
                                min_window_size=60, **kwargs):
        """
        Calculate the MSD for a given track (or window) and fit the MSD model.
        
        Parameters:
        - track_data: DataFrame for the track or window.
        - store_msd: If True, store the per-lag MSD records.
        - time_window: Identifier for the time window (if applicable).
        - use_bounds: (Legacy; unused) Whether to enforce parameter bounds.
        - allow_partial_window: If False, skip windows with fewer than min_window_size frames.
        - min_window_size: Minimum number of frames required in a window.
        
        Additional keyword arguments:
        - fit_method: "r2_threshold" (default) or "model_selection".
                        In "r2_threshold" mode the goodness-of-fit is assessed via an R² threshold.
                        In "model_selection" mode both a normal model (α fixed = 1) and an anomalous model
                        (α free) are fitted and compared via AIC.
        - r2_threshold: R² threshold for a good fit for the normal model (default: 0.95).
        - anomalous_r2_threshold: R² threshold for the anomalous model (default: 0.80).
        - bad_fit_strategy: How to handle a poor fit:
                "remove_track" (remove the entire track),
                "excise_window" (skip the offending window and, if it’s the first such instance in the track,
                                    later windows will get a new unique ID),
                "flag" (keep the window but mark it as flagged).
                Default is "flag".
        - use_ci: (Boolean) If True, use the confidence interval for α (based on parameter uncertainty)
                    rather than a fixed tolerance for diffusion type classification. Default is False.
        - ci_multiplier: Multiplier for the standard error for constructing a confidence interval (default: 1.96).
        
        Returns:
        A tuple: (avg_msd, D, alpha, motion_class)
                where motion_class is a string (e.g. "normal", "subdiffusive", "superdiffusive", or flagged/excised notes).
        """
        import numpy as np
        import scipy.optimize

        n_frames = len(track_data)
        
        # Check that there are enough frames.
        if time_window is None:
            if n_frames < 3:
                print(f"Skipping unique_id {track_data['unique_id'].iloc[0]}: insufficient frames ({n_frames}).")
                return np.nan, np.nan, np.nan, 'unlabeled'
        else:
            if not allow_partial_window and n_frames < min_window_size:
                print(f"Skipping unique_id {track_data['unique_id'].iloc[0]} for time window {time_window}: insufficient frames ({n_frames} < {min_window_size}).")
                return np.nan, np.nan, np.nan, 'unlabeled'
        
        # Compute lag times and observed MSD values.
        lag_times = np.arange(1, n_frames) * self.time_between_frames
        msd_values = np.zeros(len(lag_times))
        for lag in range(1, len(lag_times) + 1):
            displacements = (track_data[['x_um', 'y_um']].iloc[lag:].values -
                            track_data[['x_um', 'y_um']].iloc[:-lag].values) ** 2
            msd_values[lag - 1] = np.mean(np.sum(displacements, axis=1))
        
        if len(lag_times) < 3:
            print(f"Skipping unique_id {track_data['unique_id'].iloc[0]}: insufficient lag times ({len(lag_times)}).")
            return np.nan, np.nan, np.nan, 'unlabeled'
        
        # Get options from kwargs.
        fit_method = kwargs.get("fit_method", "r2_threshold")
        r2_threshold = kwargs.get("r2_threshold", 0.95)
        anomalous_r2_threshold = kwargs.get("anomalous_r2_threshold", 0.80)
        bad_fit_strategy = kwargs.get("bad_fit_strategy", "flag")
        use_ci_flag = kwargs.get("use_ci", False)
        ci_multiplier = kwargs.get("ci_multiplier", 1.96)  # Approximately 95% CI if errors are normal.
        
        log_obs = np.log10(msd_values)
        n = len(lag_times)
        
        # ----- MODEL_SELECTION BRANCH -----
        if fit_method == "model_selection":
            # Fit the anomalous model (free α) first.
            try:
                popt_anom, pcov_anom = scipy.optimize.curve_fit(self.msd_model, lag_times, msd_values)
                D_anom, alpha_anom = popt_anom[0], popt_anom[1]
                predicted_anom = self.msd_model(lag_times, D_anom, alpha_anom)
                log_pred_anom = np.log10(predicted_anom)
                rss_anom = np.sum((log_obs - log_pred_anom) ** 2)
            except RuntimeError:
                print(f"Curve fitting (anomalous) failed for unique_id {track_data['unique_id'].iloc[0]}.")
                return np.nan, np.nan, np.nan, 'bad_fit_removed_track'
            
            # Fit the normal model (α fixed at 1) using msd_normal.
            try:
                popt_norm, _ = scipy.optimize.curve_fit(self.msd_normal, lag_times, msd_values)
                D_norm = popt_norm[0]
                predicted_norm = self.msd_normal(lag_times, D_norm)
                log_pred_norm = np.log10(predicted_norm)
                rss_norm = np.sum((log_obs - log_pred_norm) ** 2)
            except RuntimeError:
                print(f"Curve fitting (normal) failed for unique_id {track_data['unique_id'].iloc[0]}.")
                # Use the anomalous model if normal fit fails.
                D_norm, rss_norm = np.nan, np.inf
            
            # Compute AIC values.
            # For the anomalous model, k = 2; for normal, k = 1.
            aic_anom = 2 * 2 + n * np.log(rss_anom / n)
            aic_norm = 2 * 1 + n * np.log(rss_norm / n)
            
            # Calculate Akaike weights.
            aic_values = {"anomalous": aic_anom, "normal": aic_norm}
            min_aic = min(aic_values.values())
            delta_aics = {model: (aic_values[model] - min_aic) for model in aic_values}
            weights = {model: np.exp(-0.5 * delta_aics[model]) for model in delta_aics}
            sum_weights = sum(weights.values())
            akaike_weights = {model: weights[model] / sum_weights for model in weights}
            # For debugging:
            print(f"Akaike weights: {akaike_weights}")
            
            # Choose the model with the lower AIC.
            if aic_norm < aic_anom:
                chosen_model = "normal"
                D = D_norm
                alpha = 1.0
                chosen_rss = rss_norm
                chosen_covar = None  # α is fixed.
            else:
                chosen_model = "anomalous"
                D = D_anom
                alpha = alpha_anom
                chosen_rss = rss_anom
                chosen_covar = pcov_anom
            
            # Compute R² in log–log space.
            ss_tot = np.sum((log_obs - np.mean(log_obs)) ** 2)
            chosen_r2 = 1 - chosen_rss / ss_tot
            
            # Decide which R² threshold applies.
            threshold = r2_threshold if chosen_model == "normal" else anomalous_r2_threshold
            
            # Check if the chosen model’s fit is acceptable.
            if chosen_r2 < threshold:
                if bad_fit_strategy == "remove_track":
                    print(f"Bad fit (model_selection) for unique_id {track_data['unique_id'].iloc[0]} (R² = {chosen_r2:.3f}). Removing track.")
                    return np.nan, np.nan, np.nan, 'bad_fit_removed_track'
                elif bad_fit_strategy == "excise_window":
                    print(f"Bad fit (model_selection) for unique_id {track_data['unique_id'].iloc[0]} time window {time_window} (R² = {chosen_r2:.3f}). Excising window.")
                    return np.nan, np.nan, np.nan, 'excised_bad_fit'
                elif bad_fit_strategy == "flag":
                    print(f"Bad fit (model_selection) for unique_id {track_data['unique_id'].iloc[0]} time window {time_window} (R² = {chosen_r2:.3f}). Flagging window.")
                    motion_class = ('bad_fit_flagged_normal' if chosen_model=="normal"
                                    else ('bad_fit_flagged_subdiffusive' if alpha < 1 - (self.tolerance if not use_ci_flag else 0) 
                                        else 'bad_fit_flagged_superdiffusive'))
                    return np.mean(msd_values), D, alpha, motion_class
            
            # Now, determine diffusion type using either fixed tolerance or CI.
            if chosen_model == "normal":
                motion_class = "normal"
            else:
                if use_ci_flag and (chosen_covar is not None):
                    # Calculate standard error of α.
                    try:
                        alpha_err = np.sqrt(np.diag(chosen_covar))[1]
                    except Exception as e:
                        print(f"Could not extract uncertainty: {e}; reverting to fixed tolerance.")
                        use_ci_flag = False
                    if use_ci_flag:
                        alpha_lower = alpha - ci_multiplier * alpha_err
                        alpha_upper = alpha + ci_multiplier * alpha_err
                        print(f"Confidence interval for α: [{alpha_lower:.3f}, {alpha_upper:.3f}]")
                        if alpha_lower <= 1 <= alpha_upper:
                            motion_class = "normal"
                        elif alpha_upper < 1:
                            motion_class = "subdiffusive"
                        elif alpha_lower > 1:
                            motion_class = "superdiffusive"
                        else:
                            motion_class = "ambiguous"
                    else:
                        # Fallback to fixed tolerance.
                        if alpha < 1 - self.tolerance:
                            motion_class = "subdiffusive"
                        elif alpha > 1 + self.tolerance:
                            motion_class = "superdiffusive"
                        else:
                            motion_class = "normal"
                else:
                    # Use fixed tolerance as a default.
                    if alpha < 1 - self.tolerance:
                        motion_class = "subdiffusive"
                    elif alpha > 1 + self.tolerance:
                        motion_class = "superdiffusive"
                    else:
                        motion_class = "normal"
            
            avg_msd = np.mean(msd_values)
        
        # -------- FALLBACK: r2_threshold METHOD --------
        else:
            try:
                popt, pcov = scipy.optimize.curve_fit(self.msd_model, lag_times, msd_values)
                D, alpha = popt[0], popt[1]
                predicted = self.msd_model(lag_times, D, alpha)
            except RuntimeError:
                print(f"Curve fitting failed for unique_id {track_data['unique_id'].iloc[0]}.")
                return np.nan, np.nan, np.nan, 'bad_fit_removed_track'
            
            log_pred = np.log10(predicted)
            ss_res = np.sum((log_obs - log_pred) ** 2)
            ss_tot = np.sum((log_obs - np.mean(log_obs)) ** 2)
            chosen_r2 = 1 - ss_res / ss_tot
            
            threshold = anomalous_r2_threshold if (alpha < 1 - self.tolerance or alpha > 1 + self.tolerance) else r2_threshold
            
            if chosen_r2 < threshold:
                if bad_fit_strategy == "remove_track":
                    print(f"Bad fit for unique_id {track_data['unique_id'].iloc[0]} (R² = {chosen_r2:.3f}). Removing track.")
                    return np.nan, np.nan, np.nan, 'bad_fit_removed_track'
                elif bad_fit_strategy == "excise_window":
                    print(f"Bad fit for unique_id {track_data['unique_id'].iloc[0]} time window {time_window} (R² = {chosen_r2:.3f}). Excising window.")
                    return np.nan, np.nan, np.nan, 'excised_bad_fit'
                elif bad_fit_strategy == "flag":
                    print(f"Bad fit for unique_id {track_data['unique_id'].iloc[0]} time window {time_window} (R² = {chosen_r2:.3f}). Flagging window.")
                    if alpha < 1 - self.tolerance:
                        motion_class = 'bad_fit_flagged_subdiffusive'
                    elif alpha > 1 + self.tolerance:
                        motion_class = 'bad_fit_flagged_superdiffusive'
                    else:
                        motion_class = 'bad_fit_flagged_normal'
                    return np.mean(msd_values), D, alpha, motion_class
            else:
                if alpha < 1 - self.tolerance:
                    motion_class = 'subdiffusive'
                elif alpha > 1 + self.tolerance:
                    motion_class = 'superdiffusive'
                else:
                    motion_class = 'normal'
            avg_msd = np.mean(msd_values)
        
        # Optionally store per-lag MSD records.
        if store_msd and (time_window is not None):
            import pandas as pd
            msd_records = [{
                'unique_id': track_data['unique_id'].iloc[0],
                'time_window': time_window,
                'lag_time': lag,
                'msd': msd,
                'diffusion_coefficient': D,
                'anomalous_exponent': alpha
            } for lag, msd in zip(lag_times, msd_values)]
            if msd_records:
                msd_records_df = pd.DataFrame(msd_records)
                self.msd_lagtime_df = pd.concat([self.msd_lagtime_df, msd_records_df], ignore_index=True)
        
        return avg_msd, D, alpha, motion_class


    def calculate_time_windowed_metrics(self, window_size=None, overlap=None, filter_metrics_df=False, **kwargs):
        """
        Calculate metrics for time windows across all tracks.
        
        Any additional keyword arguments are passed to calculate_msd_for_track,
        including those controlling fit_method (e.g. "r2_threshold" vs "model_selection")
        and the bad-fit handling options.
        
        For bad fits, three strategies are available via 'bad_fit_strategy':
        - "remove_track": Remove all windows from the track.
        - "excise_window": Excise (skip) the offending window. If an excision occurs,
                            subsequent windows are given a new unique ID with the suffix '_s'.
        - "flag": Keep the window but flag it (e.g. 'bad_fit_flagged_normal').
        
        Parameters:
        - window_size: Number of frames per window.
        - overlap: Number of overlapping frames.
        - filter_metrics_df: If True, update self.metrics_df to only include frames in windows.
        - **kwargs: Additional options to pass to calculate_msd_for_track.
        """
        from tqdm import tqdm
        import numpy as np
        import pandas as pd
        
        if window_size is None:
            window_size = self.calculate_default_window_size()
        if overlap is None:
            overlap = int(window_size / 2)
        
        print("Note: Partial windows (below the window size) are not included in the time-windowed metrics.")
        
        windowed_list = []
        included_frames = set()
        
        for unique_id, track_data in tqdm(self.metrics_df.groupby('unique_id'), desc="Calculating Time-Windowed Metrics"):
            n_frames = len(track_data)
            track_removed = False
            current_unique_id = unique_id  # May change if a bad window is excised.
            
            for start in range(0, n_frames - window_size + 1, window_size - overlap):
                end = start + window_size
                window_data = track_data.iloc[start:end].copy()
                included_frames.update(window_data['frame'].values)
                
                # Compute displacements.
                start_row = window_data.iloc[0]
                end_row = window_data.iloc[-1]
                net_disp = np.sqrt((end_row['x_um'] - start_row['x_um'])**2 +
                                (end_row['y_um'] - start_row['y_um'])**2)
                
                window_data['x_um_prev'] = window_data['x_um'].shift(1)
                window_data['y_um_prev'] = window_data['y_um'].shift(1)
                window_data['segment_len_um'] = np.sqrt(
                    (window_data['x_um'] - window_data['x_um_prev'])**2 +
                    (window_data['y_um'] - window_data['y_um_prev'])**2
                ).fillna(0)
                cum_disp = window_data['segment_len_um'].sum()
                
                # Call the MSD fitting function.
                avg_msd, D, alpha, motion_class = self.calculate_msd_for_track(
                    window_data,
                    store_msd=True,
                    time_window=start // (window_size - overlap),
                    **kwargs
                )
                
                if motion_class == 'bad_fit_removed_track':
                    print(f"Removing entire track for unique_id {unique_id} due to a bad fit.")
                    track_removed = True
                    break
                elif motion_class == 'excised_bad_fit':
                    if current_unique_id == unique_id:
                        current_unique_id = f"{unique_id}_s"
                        print(f"Track {unique_id}: first bad window excised; subsequent windows will have unique_id '{current_unique_id}'.")
                    continue  # Skip this window.
                
                total_time_s = window_data['time_s'].iloc[-1] - window_data['time_s'].iloc[0]
                avg_speed = window_data['speed_um_s'].mean()
                avg_acceleration = window_data['acceleration_um_s2'].mean()
                avg_jerk = window_data['jerk_um_s3'].mean()
                avg_norm_curvature = window_data['normalized_curvature'].mean()
                avg_angle_norm_curvature = window_data['angle_normalized_curvature'].mean()
                persistence_length = self.calculate_persistence_length(window_data)
                
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
                    'unique_id': [current_unique_id],
                    'avg_msd': [avg_msd],
                    'n_frames': [window_size],
                    'total_time_s': [total_time_s],
                    'Location': [start_row['Location']],
                    'diffusion_coefficient': [D],
                    'anomalous_exponent': [alpha],
                    'motion_class': [motion_class],
                    'avg_speed_um_s': [avg_speed],
                    'avg_acceleration_um_s2': [avg_acceleration],
                    'avg_jerk_um_s3': [avg_jerk],
                    'avg_normalized_curvature': [avg_norm_curvature],
                    'avg_angle_normalized_curvature': [avg_angle_norm_curvature],
                    'persistence_length': [persistence_length],
                    'net_displacement_um': [net_disp],
                    'cum_displacement_um': [cum_disp],
                    'bad_fit_flag': [motion_class.startswith("bad_fit_flagged")]
                })

                # --- NEW: Compute von Mises kappa values for this window ---
                kappa_turning, kappa_absolute = self.calculate_vonmises_kappa(window_data)
                
                # Append the kappa values to the window summary DataFrame
                window_summary['kappa_turning'] = kappa_turning
                window_summary['kappa_absolute'] = kappa_absolute



                windowed_list.append(window_summary)
            
            if track_removed:
                print(f"Skipping unique_id {unique_id} entirely due to a bad fit in one or more windows.")
                continue
        
        self.time_windowed_df = pd.concat(windowed_list).reset_index(drop=True)
        
        if filter_metrics_df:
            print("Filtering metrics_df to only include frames within time windows...")
            print(f"Initial number of frames: {len(self.metrics_df)}")
            self.metrics_df = self.metrics_df[self.metrics_df['frame'].isin(included_frames)].copy()
            print(f"Remaining frames after filtering: {len(self.metrics_df)}")



    ############### V4 above ################################

   

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

        ##### old version 2-19-2025 #### BELOW

    # def calculate_time_windowed_metrics(self, window_size=None, overlap=None, filter_metrics_df=False):
    #     """
    #     Calculate metrics for time windows across all tracks.
    #     Parameters:
    #     - window_size: Number of frames in each window
    #     - overlap: Number of frames overlapping between consecutive windows
    #     - filter_metrics_df: Whether to filter the metrics_df to only include frames within time windows
    #     """
    #     if window_size is None:
    #         window_size = self.calculate_default_window_size()
    #     if overlap is None:
    #         overlap = int(window_size / 2)

    #     print("Note: Partial windows (below the window size) are not included in the time-windowed metrics.")

    #     windowed_list = []
    #     included_frames = set()  # Track frames included in windows

    #     for unique_id, track_data in tqdm(self.metrics_df.groupby('unique_id'), desc="Calculating Time-Windowed Metrics"):
    #         n_frames = len(track_data)
    #         for start in range(0, n_frames - window_size + 1, window_size - overlap):
    #             end = start + window_size
    #             window_data = track_data.iloc[start:end]

    #             included_frames.update(window_data['frame'].values)  # Track included frames

    #             avg_msd, D, alpha, motion_class = self.calculate_msd_for_track(
    #                 window_data, store_msd=True, time_window=start // (window_size - overlap)
    #             )

    #             total_time_s = (window_data['time_s'].iloc[-1] - window_data['time_s'].iloc[0])
    #             avg_speed = window_data['speed_um_s'].mean()
    #             avg_acceleration = window_data['acceleration_um_s2'].mean()
    #             avg_jerk = window_data['jerk_um_s3'].mean()
    #             avg_norm_curvature = window_data['normalized_curvature'].mean()
    #             avg_angle_norm_curvature = window_data['angle_normalized_curvature'].mean()
    #             persistence_length = self.calculate_persistence_length(window_data)

    #             start_row = window_data.iloc[0]
    #             end_row = window_data.iloc[-1]
    #             window_summary = pd.DataFrame({
    #                 'time_window': [start // (window_size - overlap)],
    #                 'x_um_start': [start_row['x_um']],
    #                 'y_um_start': [start_row['y_um']],
    #                 'x_um_end': [end_row['x_um']],
    #                 'y_um_end': [end_row['y_um']],
    #                 'particle': [start_row['particle']],
    #                 'condition': [start_row['condition']],
    #                 'filename': [start_row['filename']],
    #                 'file_id': [start_row['file_id']],
    #                 'unique_id': [unique_id],
    #                 'avg_msd': [avg_msd],
    #                 'n_frames': [window_size],
    #                 'total_time_s': [total_time_s],
    #                 'Location': [start_row['Location']],
    #                 'diffusion_coefficient': [D],
    #                 'anomalous_exponent': [alpha],
    #                 'motion_class': [motion_class],
    #                 'avg_speed_um_s': [avg_speed],
    #                 'avg_acceleration_um_s2': [avg_acceleration],
    #                 'avg_jerk_um_s3': [avg_jerk],
    #                 'avg_normalized_curvature': [avg_norm_curvature],
    #                 'avg_angle_normalized_curvature': [avg_angle_norm_curvature],
    #                 'persistence_length': [persistence_length],
    #             })

    #             windowed_list.append(window_summary)

    #     self.time_windowed_df = pd.concat(windowed_list).reset_index(drop=True)

    #     # Add frame filtering logic if requested
    #     if filter_metrics_df:
    #         print("Filtering metrics_df to only include frames within time windows...")
    #         print(f"Initial number of frames: {len(self.metrics_df)}")
    #         self.metrics_df = self.metrics_df[self.metrics_df['frame'].isin(included_frames)].copy()
    #         print(f"Remaining frames after filtering: {len(self.metrics_df)}")

                ##### old version 2-19-2025 #### ## ABOVE

    ############################ old-new version 4-14-2025 below ############################

    # def calculate_time_windowed_metrics(self, window_size=None, overlap=None, filter_metrics_df=False, **kwargs):
    #     """
    #     Calculate metrics for time windows across all tracks.
        
    #     Parameters:
    #     - window_size: Number of frames in each window.
    #     - overlap: Number of frames overlapping between consecutive windows.
    #     - filter_metrics_df: Whether to filter the metrics_df to only include frames within time windows.
    #     """
    #     if window_size is None:
    #         window_size = self.calculate_default_window_size()
    #     if overlap is None:
    #         overlap = int(window_size / 2)

    #     print("Note: Partial windows (below the window size) are not included in the time-windowed metrics.")

    #     windowed_list = []
    #     included_frames = set()  # Track frames included in windows

    #     # Process each track based on unique_id
    #     for unique_id, track_data in tqdm(self.metrics_df.groupby('unique_id'), desc="Calculating Time-Windowed Metrics"):
    #         n_frames = len(track_data)
    #         # Slide a window across each track
    #         for start in range(0, n_frames - window_size + 1, window_size - overlap):
    #             end = start + window_size
    #             # Make a copy to avoid modifying the original track_data
    #             window_data = track_data.iloc[start:end].copy()

    #             # Track included frames in the window
    #             included_frames.update(window_data['frame'].values)

    #             # Calculate net displacement for the window (start to end straight-line distance)
    #             start_row = window_data.iloc[0]
    #             end_row = window_data.iloc[-1]
    #             net_disp = np.sqrt((end_row['x_um'] - start_row['x_um'])**2 + (end_row['y_um'] - start_row['y_um'])**2)

    #             # Calculate cumulative displacement for the window:
    #             # 1. Shift coordinates to get previous frame positions.
    #             window_data['x_um_prev'] = window_data['x_um'].shift(1)
    #             window_data['y_um_prev'] = window_data['y_um'].shift(1)
    #             # 2. Compute segment length between consecutive frames.
    #             window_data['segment_len_um'] = np.sqrt(
    #                 (window_data['x_um'] - window_data['x_um_prev'])**2 +
    #                 (window_data['y_um'] - window_data['y_um_prev'])**2
    #             )
    #             # Replace NaN from the shift with 0.
    #             window_data['segment_len_um'] = window_data['segment_len_um'].fillna(0)
    #             # 3. Sum the segment lengths to get the cumulative displacement.
    #             cum_disp = window_data['segment_len_um'].sum()

    #             # Calculate MSD and motion features for this window
    #             avg_msd, D, alpha, motion_class = self.calculate_msd_for_track(
    #                 window_data, store_msd=True, time_window=start // (window_size - overlap), **kwargs
    #             )

    #             total_time_s = window_data['time_s'].iloc[-1] - window_data['time_s'].iloc[0]
    #             avg_speed = window_data['speed_um_s'].mean()
    #             avg_acceleration = window_data['acceleration_um_s2'].mean()
    #             avg_jerk = window_data['jerk_um_s3'].mean()
    #             avg_norm_curvature = window_data['normalized_curvature'].mean()
    #             avg_angle_norm_curvature = window_data['angle_normalized_curvature'].mean()
    #             persistence_length = self.calculate_persistence_length(window_data)

    #             # Create a summary DataFrame for the window including the new displacement metrics
    #             window_summary = pd.DataFrame({
    #                 'time_window': [start // (window_size - overlap)],
    #                 'x_um_start': [start_row['x_um']],
    #                 'y_um_start': [start_row['y_um']],
    #                 'x_um_end': [end_row['x_um']],
    #                 'y_um_end': [end_row['y_um']],
    #                 'particle': [start_row['particle']],
    #                 'condition': [start_row['condition']],
    #                 'filename': [start_row['filename']],
    #                 'file_id': [start_row['file_id']],
    #                 'unique_id': [unique_id],
    #                 'avg_msd': [avg_msd],
    #                 'n_frames': [window_size],
    #                 'total_time_s': [total_time_s],
    #                 'Location': [start_row['Location']],
    #                 'diffusion_coefficient': [D],
    #                 'anomalous_exponent': [alpha],
    #                 'motion_class': [motion_class],
    #                 'avg_speed_um_s': [avg_speed],
    #                 'avg_acceleration_um_s2': [avg_acceleration],
    #                 'avg_jerk_um_s3': [avg_jerk],
    #                 'avg_normalized_curvature': [avg_norm_curvature],
    #                 'avg_angle_normalized_curvature': [avg_angle_norm_curvature],
    #                 'persistence_length': [persistence_length],
    #                 'net_displacement_um': [net_disp],       # New: Net displacement for the window
    #                 'cum_displacement_um': [cum_disp]          # New: Cumulative displacement for the window
    #             })

    #             windowed_list.append(window_summary)

    #     # Concatenate all window summaries into a single DataFrame
    #     self.time_windowed_df = pd.concat(windowed_list).reset_index(drop=True)

    #     # Optionally filter the original metrics_df to only include frames that were part of the windows
    #     if filter_metrics_df:
    #         print("Filtering metrics_df to only include frames within time windows...")
    #         print(f"Initial number of frames: {len(self.metrics_df)}")
    #         self.metrics_df = self.metrics_df[self.metrics_df['frame'].isin(included_frames)].copy()
    #         print(f"Remaining frames after filtering: {len(self.metrics_df)}")

############################ old-new version 4-14-2025 ABOVE ############################
    ############################### BOUNDS AND STUFF BELOW #################################### 

    # def calculate_time_windowed_metrics(self, window_size=None, overlap=None, filter_metrics_df=False, **kwargs):
    #     """
    #     Calculate metrics for time windows across all tracks.
        
    #     Additional keyword arguments (kwargs) are forwarded to calculate_msd_for_track
    #     (including r2_threshold and bad_fit_strategy).
        
    #     For handling "bad fits", three strategies are available via 'bad_fit_strategy':
    #     - "remove_track": Remove all windows for the track.
    #     - "excise_window": Skip (excise) the offending window and, if an excision occurs,
    #         assign a new unique ID (by appending "_excised") to subsequent windows in this track.
    #     - "flag": Keep the window but add a flag in the motion_class (e.g., 'bad_fit_flagged_normal').
        
    #     Parameters:
    #     - window_size: Number of frames per window.
    #     - overlap: Number of frames overlapping between windows.
    #     - filter_metrics_df: Whether to filter self.metrics_df to only include frames in windows.
    #     - **kwargs: extra parameters to pass to calculate_msd_for_track.
    #     """
    #     # from tqdm import tqdm  # for progress reporting
    #     # import numpy as np
    #     # import pandas as pd
        
    #     if window_size is None:
    #         window_size = self.calculate_default_window_size()
    #     if overlap is None:
    #         overlap = int(window_size / 2)
        
    #     print("Note: Partial windows (below the window size) are not included in the time-windowed metrics.")
        
    #     windowed_list = []
    #     included_frames = set()  # For optional filtering of original metrics_df.
        
    #     # Process each track by unique_id.
    #     for unique_id, track_data in tqdm(self.metrics_df.groupby('unique_id'), desc="Calculating Time-Windowed Metrics"):
    #         n_frames = len(track_data)
    #         # Flags for handling excision and track removal.
    #         track_removed = False
    #         # current_unique_id can change if we excise a window.
    #         current_unique_id = unique_id
            
    #         # Loop over windows for this track.
    #         for start in range(0, n_frames - window_size + 1, window_size - overlap):
    #             end = start + window_size
    #             window_data = track_data.iloc[start:end].copy()
    #             included_frames.update(window_data['frame'].values)
                
    #             # Calculate displacements.
    #             start_row = window_data.iloc[0]
    #             end_row = window_data.iloc[-1]
    #             net_disp = np.sqrt((end_row['x_um'] - start_row['x_um'])**2 +
    #                             (end_row['y_um'] - start_row['y_um'])**2)
    #             # Cumulative displacement.
    #             window_data['x_um_prev'] = window_data['x_um'].shift(1)
    #             window_data['y_um_prev'] = window_data['y_um'].shift(1)
    #             window_data['segment_len_um'] = np.sqrt(
    #                 (window_data['x_um'] - window_data['x_um_prev'])**2 +
    #                 (window_data['y_um'] - window_data['y_um_prev'])**2
    #             ).fillna(0)
    #             cum_disp = window_data['segment_len_um'].sum()
                
    #             # Calculate MSD and motion features; pass along **kwargs.
    #             avg_msd, D, alpha, motion_class = self.calculate_msd_for_track(
    #                 window_data, 
    #                 store_msd=True, 
    #                 time_window=start // (window_size - overlap),
    #                 **kwargs
    #             )
                
    #             # Handle bad-fit strategies.
    #             if motion_class == 'bad_fit_removed_track':
    #                 # Remove the entire track by breaking out of the window loop.
    #                 print(f"Removing entire track for unique_id {unique_id} due to bad fit.")
    #                 track_removed = True
    #                 break
    #             elif motion_class == 'excised_bad_fit':
    #                 # Excise this window (do not add it) and update the unique_id if not already split.
    #                 if current_unique_id == unique_id:
    #                     current_unique_id = f"{unique_id}_excised"
    #                     print(f"Track {unique_id}: first bad window excised; subsequent windows will have unique_id '{current_unique_id}'.")
    #                 # Skip adding this window.
    #                 continue
    #             else:
    #                 # For the "flag" strategy, motion_class may include a prefix "bad_fit_flagged".
    #                 bad_fit_flag = motion_class.startswith("bad_fit_flagged")
                
    #             total_time_s = window_data['time_s'].iloc[-1] - window_data['time_s'].iloc[0]
    #             avg_speed = window_data['speed_um_s'].mean()
    #             avg_acceleration = window_data['acceleration_um_s2'].mean()
    #             avg_jerk = window_data['jerk_um_s3'].mean()
    #             avg_norm_curvature = window_data['normalized_curvature'].mean()
    #             avg_angle_norm_curvature = window_data['angle_normalized_curvature'].mean()
    #             persistence_length = self.calculate_persistence_length(window_data)
                
    #             # Create window summary.
    #             window_summary = pd.DataFrame({
    #                 'time_window': [start // (window_size - overlap)],
    #                 'x_um_start': [start_row['x_um']],
    #                 'y_um_start': [start_row['y_um']],
    #                 'x_um_end': [end_row['x_um']],
    #                 'y_um_end': [end_row['y_um']],
    #                 'particle': [start_row['particle']],
    #                 'condition': [start_row['condition']],
    #                 'filename': [start_row['filename']],
    #                 'file_id': [start_row['file_id']],
    #                 'unique_id': [current_unique_id],
    #                 'avg_msd': [avg_msd],
    #                 'n_frames': [window_size],
    #                 'total_time_s': [total_time_s],
    #                 'Location': [start_row['Location']],
    #                 'diffusion_coefficient': [D],
    #                 'anomalous_exponent': [alpha],
    #                 'motion_class': [motion_class],
    #                 'avg_speed_um_s': [avg_speed],
    #                 'avg_acceleration_um_s2': [avg_acceleration],
    #                 'avg_jerk_um_s3': [avg_jerk],
    #                 'avg_normalized_curvature': [avg_norm_curvature],
    #                 'avg_angle_normalized_curvature': [avg_angle_norm_curvature],
    #                 'persistence_length': [persistence_length],
    #                 'net_displacement_um': [net_disp],
    #                 'cum_displacement_um': [cum_disp],
    #                 'bad_fit_flag': [bad_fit_flag if 'bad_fit_flag' in locals() else False]
    #             })
                
    #             windowed_list.append(window_summary)
            
    #         if track_removed:
    #             print(f"Skipping unique_id {unique_id} entirely due to a bad fit in one or more windows.")
    #             continue  # Skip adding any windows from this track.
        
    #     # Concatenate all window summaries.
    #     self.time_windowed_df = pd.concat(windowed_list).reset_index(drop=True)
        
    #     if filter_metrics_df:
    #         print("Filtering metrics_df to only include frames within time windows...")
    #         print(f"Initial number of frames: {len(self.metrics_df)}")
    #         self.metrics_df = self.metrics_df[self.metrics_df['frame'].isin(included_frames)].copy()
    #         print(f"Remaining frames after filtering: {len(self.metrics_df)}")

        
    ############################### BOUNDS AND STUFF ABOVE #################################### 

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

        self.calculate_cum_displacement() #ADDED 3_19_2025

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


################################### DEV VERSION BELOW #############################################