# config.py - SPTnano package configuration

import os

# Define global variables - users can modify these directly

MASTER = 'D:/TRANSFORMER_DEVELOPMENT/'

SAVED_DATA = MASTER + "saved_data/"

# Other configurations
PIXELSIZE_MICRONS = 0.065
TIME_BETWEEN_FRAMES = 0.01

TIME_WINDOW = 60
OVERLAP = 30

# Order of conditions
ORDEROFCONDITIONS = [
    "Condition_freehalo_cort",
    "Condition_20H20S_cort",
    "Condition_77H20S_cort",
    "Condition_20H77S_cort",
]

# Features
FEATURES = [
    "speed_um_s",
    "direction_rad",
    "acceleration_um_s2",
    "jerk_um_s3",
    "normalized_curvature",
    "angle_normalized_curvature",
    "instant_diff_coeff",
]

FEATURES2 = [
    "speed_um_s",
    "direction_rad",
    "acceleration_um_s2",
    "jerk_um_s3",
    "normalized_curvature",
    "angle_normalized_curvature",
    "instant_diff_coeff",
    "motion_class",
    "diffusion_coefficient",
]

# TensorBoard configuration
TENSORBOARD_LOGS = os.path.join(SAVED_DATA, "tensorboard_logs")

# Enhanced Analysis Parameters for Transformer Training
ANALYSIS_PARAMS = {
    # Data processing
    "min_track_length": 60,
    "pixel_size": 0.065,  # μm per pixel
    "frame_rate": 100,  # Hz
    # Traditional analysis
    "window_size": 60,
    "n_clusters_traditional": 5,
        # Data splitting configuration
        "split_params": {
            "condition_factors": ["mol"],  # Factors to create class balance labels (default: just molecule type)
            "test_split": 0.2,
            "val_split": 0.1,  # Reduced since we're using fixed cells for test
            "split_strategy": "fixed_cells",  # Fixed number of cells per condition for test set
            "cells_per_condition": 6,  # Number of cells per condition in test set
            "random_seed": 42,
        },
    # Enhanced transformer parameters
    "transformer_params": {
        "single_scale": {
            "window_size": 60,
            "overlap": 30,
            "epochs": 25,
            "batch_size": 64,
            "augmentation_strategy": "measurement_noise",
        },
        "multi_scale": {
            "scales": [
                {"window_size": 30, "overlap": 15},  # Rapid dynamics
                {"window_size": 60, "overlap": 30},  # Behavioral states
                {"window_size": 120, "overlap": 60},  # Persistent patterns
                {"window_size": 240, "overlap": 120},  # Long-range transport
            ],
            "epochs": 20,
            "batch_size": 64,
            "augmentation_strategy": "measurement_noise",
        },
    },
    # Training management
    "training_params": {
        "use_tensorboard": True,
        "save_models": True,
        "checkpoint_every": 5,
        "interruption_protection": True,
        "session_name": "htt_enhanced_analysis",
        "use_scheduler": True,
    },
    # Clustering
    "n_clusters_transformer": 5,
    # Visualization
    "figsize": (12, 8),
    "dpi": 100,
}

# =============================================================================
# TRANSFORMER GRID SEARCH CONFIGURATION
# =============================================================================
# Configuration for architecture × temperature grid search training

# Model architectures to train
TRANSFORMER_ARCHITECTURES = [
    {'name': 'med64_h4_ff128_L2',  'embed_dim': 64,  'num_heads': 4, 'ff_dim': 128, 'num_layers': 2},  # Baseline
    {'name': 'med64_h4_ff256_L3',  'embed_dim': 64,  'num_heads': 4, 'ff_dim': 256, 'num_layers': 3},  # Deeper
    {'name': 'med128_h4_ff256_L2', 'embed_dim': 128, 'num_heads': 4, 'ff_dim': 256, 'num_layers': 2},  # Bigger embed
    {'name': 'med128_h8_ff512_L3', 'embed_dim': 128, 'num_heads': 8, 'ff_dim': 512, 'num_layers': 3},  # Full capacity
]

# Temperature values for contrastive loss
TRANSFORMER_TEMPERATURES = [0.2, 0.5]  # Low (sharp) and Mid (balanced) - MATCHES NOTEBOOK

# Training hyperparameters
TRANSFORMER_TRAINING = {
    "batch_size": 256,
    "epochs": 100,
    "learning_rate": 1e-4,
    "window_size": 60,
    "overlap": 30,
    "min_track_length": 60,
    
    # Loss settings
    "use_adjacent_subwindow": False,  # MATCHES NOTEBOOK (was True)
    "adjacent_subwindow_weight": 0.5,
    "adjacent_temperature": 0.7,
    "subwindow_size": 10,
    "mask_same_track_negatives": True,
    
    # Augmentation (optimized from AUGMENTATION_DOCUMENTATION.md)
    "augmentation_type": "shuffle_scale_angle",  # 3-seg shuffle + 10-50% scale + 1-10° angle
    "noise_strength": 0.012,  # Not used for shuffle_scale_angle, kept for compatibility
    "scale_strength": 0.3,    # Not used for shuffle_scale_angle, kept for compatibility
    
    # Checkpointing & Early Stopping
    "save_best_model": True,
    "checkpoint_interval": 5,
    "early_stopping_patience": 15,
    "use_tensorboard": True,
    "use_scheduler": True,
    
    # Data loading (for WSL2/Linux - use multiprocessing)
    "num_workers": 8,  # Optimal for most systems (4-8 is usually best, 14 was too high causing slowdown)
    "pin_memory": True,  # Faster CPU→GPU transfer
}

# Data paths (will be auto-detected for WSL2)
# NOTE: Everything is on D: drive now (models, checkpoints, logs, splits, etc.)
TRANSFORMER_DATA = {
    "data_drive": "F:",  # Windows drive letter (for INPUT data files only)
    "data_dir": "Analyzed/HIERARCHICAL_GATES_20260119_102840",
    "instant_df_name": "instant_df_hierarchical_gates.parquet",
    "windowed_df_name": "windowed_df_hierarchical_gates.parquet",
    "output_drive": "D:",  # Drive for ALL OUTPUTS (models, checkpoints, logs, splits)
    "splits_dir": "TRANSFORMER_DEVELOPMENT/saved_data/models/data_splits_withheirarchalgates/",  # Directory containing data_splits.pkl
    "splits_drive": "D:",  # Drive where splits are stored
}

# Create necessary directories
os.makedirs(SAVED_DATA, exist_ok=True)
os.makedirs(TENSORBOARD_LOGS, exist_ok=True)

print("Config module loaded. Master directory is:", MASTER)
