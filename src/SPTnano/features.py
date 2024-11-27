




import numpy as np
import pandas as pd
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
    
    # def calculate_distances(self):
    #     """
    #     Calculate the distances between consecutive frames for each particle in micrometers.
    #     Skip calculation if 'segment_len_um' already exists.
    #     """
    #     self.metrics_df = self.metrics_df.sort_values(by=['unique_id', 'frame'])
    #     self.metrics_df[['x_um_prev', 'y_um_prev']] = self.metrics_df.groupby('unique_id')[['x_um', 'y_um']].shift(1)
    #     self.metrics_df['segment_len_um'] = np.sqrt(
    #         (self.metrics_df['x_um'] - self.metrics_df['x_um_prev'])**2 + 
    #         (self.metrics_df['y_um'] - self.metrics_df['y_um_prev'])**2
    #     )
    #     # Fill NaN values with 0
    #     self.metrics_df['segment_len_um'] = self.metrics_df['segment_len_um'].fillna(0)
    #     return self.metrics_df
    
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
    

    def calculate_persistence_length(self, track_data):
        """
        Calculate the persistence length for a given track.
        Parameters:
        - track_data: DataFrame containing the track data.
        Returns:
        - persistence_length: Calculated persistence length for the track.
        """
        directions = track_data['direction_rad']
        # Calculate directional correlation: <cos(theta_i - theta_j)>
        direction_diffs = directions.diff().dropna()
        correlation = np.cos(direction_diffs).mean()
            # Check if correlation is greater than zero before calculating log
        if correlation > 0:
            persistence_length = -1 / np.log(correlation)
        else:
            persistence_length = np.nan  # Assign NaN if the correlation is zero or negative
    

        # persistence_length = -1 / np.log(correlation) if correlation != 0 else np.nan
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



    def calculate_msd_for_track(self, track_data, max_lagtime, store_msd=False, time_window=None):
        n_frames = len(track_data)
        msd_values = np.zeros(max_lagtime)
        lag_times = np.zeros(max_lagtime)

        for lag in range(1, max_lagtime + 1):
            if lag < n_frames:
                displacements = (track_data[['x_um', 'y_um']].iloc[lag:].values - track_data[['x_um', 'y_um']].iloc[:-lag].values) ** 2
                squared_displacements = np.sum(displacements, axis=1)
                msd_values[lag - 1] = np.mean(squared_displacements)
                lag_times[lag - 1] = lag * self.time_between_frames
            else:
                break

        avg_msd = np.mean(msd_values)

        # Calculate total time in seconds for the track
        total_time_s = (track_data['time_s'].iloc[-1] - track_data['time_s'].iloc[0])
        lag_times = np.arange(1, max_lagtime + 1) * (total_time_s / (n_frames - 1))
        popt, _ = scipy.optimize.curve_fit(self.msd_model, lag_times, msd_values[:max_lagtime])
        D, alpha = popt[0], popt[1]

        # Print the chosen tolerance for consistency
        # print(f"Using consistent tolerance: {self.tolerance}") #took this out for now

        # Classify the type of motion with consistent tolerance
        if alpha < 1 - self.tolerance:
            motion_class = 'subdiffusive'
        elif alpha > 1 + self.tolerance:
            motion_class = 'superdiffusive'
        else:
            motion_class = 'normal'

        # Store detailed MSD values and lag times if requested
        if store_msd and time_window is not None:
            msd_records = [
                {
                    'unique_id': track_data['unique_id'].iloc[0],
                    'time_window': time_window,
                    'lag_time': lag,
                    'msd': msd,
                    'diffusion_coefficient': D,
                    'anomalous_exponent': alpha
                }
                for lag, msd in zip(lag_times, msd_values)
            ]
            # self.msd_lagtime_df = pd.concat([self.msd_lagtime_df, pd.DataFrame(msd_records)], ignore_index=True)
            # Only concatenate if msd_records is not empty
            if msd_records:
                msd_records_df = pd.DataFrame(msd_records)
                self.msd_lagtime_df = pd.concat([self.msd_lagtime_df, msd_records_df], ignore_index=True)

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
        if window_size is None:
            window_size = self.calculate_default_window_size()
        if overlap is None:
            overlap = int(window_size / 2)

        windowed_list = []

        for unique_id, track_data in tqdm(self.metrics_df.groupby('unique_id'), desc="Calculating Time-Windowed Metrics"):
            n_frames = len(track_data)

            for start in range(0, n_frames - window_size + 1, window_size - overlap):
                end = start + window_size
                window_data = track_data.iloc[start:end]

                if len(window_data) < window_size:
                    continue

                # Calculate metrics for the window and store MSD details
                avg_msd, D, alpha, motion_class = self.calculate_msd_for_track(
                    window_data, max_lagtime=min(100, int(window_size / 2)), store_msd=True, time_window=start // (window_size - overlap)
                )

                # Calculate total time in seconds for the window
                total_time_s = (window_data['time_s'].iloc[-1] - window_data['time_s'].iloc[0])

                # Calculate average instantaneous metrics for the window
                avg_speed = window_data['speed_um_s'].mean()
                avg_acceleration = window_data['acceleration_um_s2'].mean()
                avg_jerk = window_data['jerk_um_s3'].mean()
                avg_norm_curvature = window_data['normalized_curvature'].mean()
                avg_angle_norm_curvature = window_data['angle_normalized_curvature'].mean()
                persistence_length = self.calculate_persistence_length(window_data)

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
                })

                windowed_list.append(window_summary)

        self.time_windowed_df = pd.concat(windowed_list).reset_index(drop=True)

        # Use the instance variable for frame duration
        self.time_windowed_df['time_s'] = self.time_windowed_df['time_window'] * (window_size - overlap) * self.time_between_frames


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


################################### DEV VERSION BELOW #############################################