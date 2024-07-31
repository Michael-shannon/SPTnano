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

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import re
import scipy.optimize



class ParticleMetrics:
    def __init__(self, df):
        self.df = df
        self.df['Location'] = self.df['filename'].apply(self.extract_location)  # Add Location column
        self.metrics_df = self.df.copy()
        self.time_averaged_df = pd.DataFrame(columns=[
            'x_um_start', 'y_um_start', 'x_um_end', 'y_um_end', 
            'particle', 'condition', 'filename', 'file_id', 'unique_id',
            'avg_msd', 'n_frames', 'total_time_s', 'Location', 
            'diffusion_coefficient', 'anomalous_exponent', 'motion_class'  # Add columns for diffusion coefficient, anomalous exponent, and motion class
        ])

    @staticmethod
    def extract_location(filename):
        match = re.match(r'loc-(\w{2})_', filename)
        if match:
            return match.group(1)
        return 'Unknown'  # Default value if no location is found
        
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
    
    def calculate_default_max_lagtime(self):
        """
        Calculate the default maximum lag time based on the shortest track in the dataset.
        """
        min_track_length = self.metrics_df.groupby('unique_id').size().min()
        default_max_lagtime = min(100, int(min_track_length / 2))  # Example: use half the length of the shortest track or 100, whichever is smaller
        return default_max_lagtime
    
    @staticmethod
    def msd_model(t, D, alpha):
        return 4 * D * t**alpha
    
    def get_time_averaged_df(self):
        """
        Return the DataFrame with time-averaged metrics.
        """
        return self.time_averaged_df
    

    def calculate_msd(self, max_lagtime):
        """
        Calculate the time-averaged MSD for each track and aggregate across all tracks.
        Parameters:
        - max_lagtime: maximum number of frames to consider for lag times
        """
        msd_list = []

        for unique_id, track_data in tqdm(self.metrics_df.groupby('unique_id'), desc="Calculating MSD"):
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
            msd_list.append(pd.DataFrame({
                'unique_id': unique_id,
                'lag_time': np.arange(1, max_lagtime + 1),
                'msd': msd_values,
                'count': counts
            }))

            # Calculate total time in seconds for the track
            total_time_s = (track_data['time_s'].iloc[-1] - track_data['time_s'].iloc[0])

            # Fit MSD data to determine the type of motion and extract parameters
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
                'n_frames': [n_frames],  # Add the number of frames
                'total_time_s': [total_time_s],  # Add the total time in seconds
                'Location': [start_row['Location']],  # Add the Location
                'diffusion_coefficient': [D],  # Add the diffusion coefficient
                'anomalous_exponent': [alpha],  # Add the anomalous exponent
                'motion_class': [motion_class],  # Add the motion class
                # Placeholder for additional metrics
                # 'additional_metric': None,
            })

            if self.time_averaged_df.empty:
                self.time_averaged_df = track_summary
            else:
                self.time_averaged_df = pd.concat([self.time_averaged_df, track_summary], ignore_index=True)

        self.msd_df = pd.concat(msd_list).reset_index(drop=True)
        return self.msd_df





    
    def calculate_all_features(self, max_lagtime=None):
        """
        Calculate all features for the particle tracking data.
        This method will call all individual feature calculation methods.
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

        # Calculate net displacement
        self.calculate_net_displacement()


        # Calculate instantaneous diffusion coefficient between consecutive frames
        self.calculate_instantaneous_diffusion_coefficient()

        # Calculate instantaneous velocity
        self.calculate_instantaneous_velocity()
        
        # Calculate MSD for each track and aggregate
        self.calculate_msd(max_lagtime)
        
        # Cleanup step to remove temporary columns
        self.cleanup()
        
        return self.metrics_df

    def cleanup(self):
        """
        Cleanup the dataframe by dropping unnecessary columns after all features are calculated.
        """
        self.metrics_df.drop(columns=[
            'x_um_prev', 'y_um_prev', 'time_s_prev', 'delta_time_s', 
            'speed_um_s_prev', 'acceleration_um_s2_prev', 'direction_rad_prev',
            'instant_velocity_x_um_s', 'instant_velocity_y_um_s',
            ], inplace=True)

    def get_metrics_df(self):
        """
        Return the dataframe with calculated metrics.
        """
        return self.metrics_df
    ###################
































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








