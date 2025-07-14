# config.py

import os

# Define global variables
# MASTER = 'path/to/top/level/directory/'  # Update this path to your master directory
# MASTER = 'D:/4_26_2025_Kinesininneurons_cort/'
# MASTER = 'D:/data_transformer_input/'
MASTER = 'D:/Reanalzye_April2025/FIGURE2_Neurons_WTHTTinRegionsofNeurons/2_25_2025_CorticalNeuron_20H20S_FreeHalo_20H77S_77H20S_analyze/'

# MASTER = 'D:/Reanalzye_April2025/FIGURE2_Neurons_WTHTTinRegionsofNeurons/2_25_2025_CorticalNeuron_20H20S_FreeHalo_20H77S_77H20S_analyze/'
SAVED_DATA = MASTER + 'saved_data/'

# Other configurations can be added here
PIXELSIZE_MICRONS = 0.065
TIME_BETWEEN_FRAMES = 0.01 #0.1

TIME_WINDOW = 60#60 #60 #6
OVERLAP = 30#30 #30 #3

# ORDEROFCONDITIONS = ['Condition_RUES2_kinesin','Condition_HTTKO_kinesin']
# ORDEROFCONDITIONS = ['Condition_mol-kinesin_geno-RUES2_type-ES_loc-ES','Condition_mol-HTT_geno-20H20S_type-ES_loc-ES', 'Condition_mol-kinesin_geno-HTTKO_type-ES_loc-ES']

ORDEROFCONDITIONS = ['Condition_freehalo_cort','Condition_20H20S_cort','Condition_77H20S_cort','Condition_20H77S_cort']

# FEATURES

FEATURES = ['speed_um_s', 'direction_rad', 'acceleration_um_s2',
       'jerk_um_s3', 'normalized_curvature', 'angle_normalized_curvature',
       'instant_diff_coeff']

FEATURES2 = ['speed_um_s', 'direction_rad', 'acceleration_um_s2',
       'jerk_um_s3', 'normalized_curvature', 'angle_normalized_curvature',
       'instant_diff_coeff', 'motion_class', 'diffusion_coefficient']

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
    
    # 🔥 NEW: Enhanced transformer parameters
    'transformer_params': {
        'single_scale': {
            'window_size': 60,
            'overlap': 30,
            'epochs': 25,
            'batch_size': 64,
            'augmentation_strategy': 'comprehensive'  # 🔥 Enhanced augmentation!
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
            'augmentation_strategy': 'measurement_noise'  # 🔥 Realistic noise simulation
        }
    },
    
    # 🔥 NEW: Training management
    'training_params': {
        'use_tensorboard': True,           # Real-time monitoring
        'save_models': True,               # Auto-save trained models
        'checkpoint_every': 5,             # Save checkpoint every N epochs
        'interruption_protection': True,   # Handle Ctrl+C gracefully
        'session_name': 'htt_enhanced_analysis',  # Training session identifier
        'use_scheduler': True             # 🔥 Adaptive learning rate scheduler (set True to enable)
    },
    
    # Clustering
    'n_clusters_transformer': 5,
    
    # Visualization
    'figsize': (12, 8),
    'dpi': 100
}

# Create tensorboard directory if it doesn't exist
os.makedirs(TENSORBOARD_LOGS, exist_ok=True)

print("Config module loaded. Master directory is:", MASTER) 