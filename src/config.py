# config.py

# Define global variables
MASTER = 'path/to/top/level/directory/'  # Update this path to your master directory
SAVED_DATA = MASTER + 'saved_data/'

# Other configurations can be added here
PIXELSIZE_MICRONS = 0.065
TIME_BETWEEN_FRAMES = 0.01 #0.1

TIME_WINDOW = 60#60 #60 #6
OVERLAP = 30#30 #30 #3

ORDEROFCONDITIONS = ['Condition_RUES2_kinesin','Condition_HTTKO_kinesin']

# FEATURES

FEATURES = ['speed_um_s', 'direction_rad', 'acceleration_um_s2',
       'jerk_um_s3', 'normalized_curvature', 'angle_normalized_curvature',
       'instant_diff_coeff']

FEATURES2 = ['speed_um_s', 'direction_rad', 'acceleration_um_s2',
       'jerk_um_s3', 'normalized_curvature', 'angle_normalized_curvature',
       'instant_diff_coeff', 'motion_class', 'diffusion_coefficient']


print("Config module loaded. Master directory is:", MASTER)

