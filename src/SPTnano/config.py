# config.py - SPTnano package configuration

import os

# Define global variables - users can modify these directly
MASTER = 'D:/data_transformer_input/'
SAVED_DATA = MASTER + 'saved_data/'

# Other configurations
PIXELSIZE_MICRONS = 0.065
TIME_BETWEEN_FRAMES = 0.01

TIME_WINDOW = 60
OVERLAP = 30

# Order of conditions
ORDEROFCONDITIONS = [
    'Condition_freehalo_cort',
    'Condition_20H20S_cort',
    'Condition_77H20S_cort',
    'Condition_20H77S_cort'
]

# Features
FEATURES = [
    'speed_um_s', 
    'direction_rad', 
    'acceleration_um_s2',
    'jerk_um_s3', 
    'normalized_curvature', 
    'angle_normalized_curvature',
    'instant_diff_coeff'
]

FEATURES2 = [
    'speed_um_s', 
    'direction_rad', 
    'acceleration_um_s2',
    'jerk_um_s3', 
    'normalized_curvature', 
    'angle_normalized_curvature',
    'instant_diff_coeff', 
    'motion_class', 
    'diffusion_coefficient'
]

# TensorBoard configuration
TENSORBOARD_LOGS = os.path.join(SAVED_DATA, 'tensorboard_logs')

# Enhanced Analysis Parameters for Transformer Training
ANALYSIS_PARAMS = {
    # Data processing
    'min_track_length': 60,
    'pixel_size': 0.065,  # μm per pixel
    'frame_rate': 100,    # Hz
    
    # Traditional analysis
    'window_size': 60,
    'n_clusters_traditional': 5,
    
    # Enhanced transformer parameters
    'transformer_params': {
        'single_scale': {
            'window_size': 60,
            'overlap': 30,
            'epochs': 25,
            'batch_size': 64,
            'augmentation_strategy': 'comprehensive'
        },
        'multi_scale': {
            'scales': [
                {'window_size': 30, 'overlap': 15},   # Rapid dynamics
                {'window_size': 60, 'overlap': 30},   # Behavioral states  
                {'window_size': 120, 'overlap': 60},  # Persistent patterns
                {'window_size': 240, 'overlap': 120}  # Long-range transport
            ],
            'epochs': 20,
            'batch_size': 64,
            'augmentation_strategy': 'measurement_noise'
        }
    },
    
    # Training management
    'training_params': {
        'use_tensorboard': True,
        'save_models': True,
        'checkpoint_every': 5,
        'interruption_protection': True,
        'session_name': 'htt_enhanced_analysis',
        'use_scheduler': True
    },
    
    # Clustering
    'n_clusters_transformer': 5,
    
    # Visualization
    'figsize': (12, 8),
    'dpi': 100
}

# Create necessary directories
os.makedirs(SAVED_DATA, exist_ok=True)
os.makedirs(TENSORBOARD_LOGS, exist_ok=True)

print("Config module loaded. Master directory is:", MASTER) 