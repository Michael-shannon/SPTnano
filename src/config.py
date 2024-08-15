# config.py

# Define global variables
MASTER = 'D:/NEURONS_HTT_JULY_2024/100x_20ms_analyze/'
# MASTER = 'D:/NEURONS_HTT_JULY_2024/60x_20_ms_analyze/'
# D:\NEURONS_HTT_JULY_2024\60x_20_ms_analyze
SAVED_DATA = MASTER + 'saved_data/'

# Other configurations can be added here
PIXELSIZE_MICRONS = 0.07
TIME_BETWEEN_FRAMES = 0.02

# PIXELSIZE_MICRONS = 0.107
# TIME_BETWEEN_FRAMES = 0.02


ORDEROFCONDITIONS = ['Condition_20H20S','Condition_77H20S','Condition_150H20S']
######### For 60 x 10 ms #################
# pixelsize_microns = 0.107
# time_between_frames = 0.01


# FEATURES

FEATURES = ['speed_um_s', 'direction_rad', 'acceleration_um_s2',
       'jerk_um_s3', 'normalized_curvature', 'angle_normalized_curvature',
       'instant_diff_coeff']


print("Config module loaded. Master directory is:", MASTER)