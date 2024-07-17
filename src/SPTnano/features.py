import pandas as pd
import numpy as np

class ParticleMetrics:
    def __init__(self, df):
        self.df = df
        self.metrics_df = self.df.copy()
        
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

    def calculate_all_features(self):
        """
        Calculate all features for the particle tracking data.
        This method will call all individual feature calculation methods.
        """
        # Calculate distances between consecutive frames
        self.calculate_distances()

        # Calculate speeds between consecutive frames
        self.calculate_speeds()
        
        # Placeholder for additional feature calculations
        # self.calculate_feature_X()
        # self.calculate_feature_Y()
        
        # Cleanup step to remove temporary columns
        self.cleanup()
        
        return self.metrics_df

    def cleanup(self):
        """
        Cleanup the dataframe by dropping unnecessary columns after all features are calculated.
        """
        self.metrics_df.drop(columns=['x_um_prev', 'y_um_prev', 'time_s_prev', 'delta_time_s'], inplace=True)

    def get_metrics_df(self):
        """
        Return the dataframe with calculated metrics.
        """
        return self.metrics_df
    







