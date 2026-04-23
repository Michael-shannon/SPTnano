# config.py - SPTnano package configuration

import os

# Define global variables - users can modify these directly

MASTER = 'D:/TRANSFORMER_DEVELOPMENT/'
# MASTER = 'Z:/mshannon/2026/Feb/2_6_2026_Hap40KD_HTTinES_20_77_150_analyze/'

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
# --- Original grid (completed) ---
# TRANSFORMER_ARCHITECTURES = [
#     {'name': 'med64_h4_ff128_L2',  'embed_dim': 64,  'num_heads': 4, 'ff_dim': 128, 'num_layers': 2},  # Baseline
#     {'name': 'med64_h4_ff256_L3',  'embed_dim': 64,  'num_heads': 4, 'ff_dim': 256, 'num_layers': 3},  # Deeper
#     {'name': 'med128_h4_ff256_L2', 'embed_dim': 128, 'num_heads': 4, 'ff_dim': 256, 'num_layers': 2},  # Bigger embed
#     {'name': 'med128_h8_ff512_L3', 'embed_dim': 128, 'num_heads': 8, 'ff_dim': 512, 'num_layers': 3},  # Full capacity
# ]
# TRANSFORMER_TEMPERATURES = [0.2, 0.5]
# --- New models to train ---
TRANSFORMER_ARCHITECTURES = [
    {'name': 'med32_h8_ff512_L3',  'embed_dim': 32,  'num_heads': 8, 'ff_dim': 512, 'num_layers': 3},  # Compact embed, wide FF
]

# Temperature values for contrastive loss
TRANSFORMER_TEMPERATURES = [0.1, 0.2]  # Sharp and Low

# Training hyperparameters
# NOTE: "epochs" is the default target for grid search. To continue training a
# specific model beyond this, use the --epochs flag on the command line:
#   python train_transformer_single_wsl2.py --model-name med128_h8_ff512_L3_t0.2 --epochs 200
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
    "augmentation_type": "shuffle_scale_angle",  # segment shuffle + scale + 1-10° angle
    "noise_strength": 0.012,  # Not used for shuffle_scale_angle, kept for compatibility
    "scale_strength": 0.3,    # Not used for shuffle_scale_angle, kept for compatibility
    # shuffle_scale_angle only: valid_prefix = segment only real frames (uses n_valid_frames);
    # legacy = three equal segments over full max_seq_len (e.g. 20+20+20 for T=60).
    "shuffle_segmentation_mode": "valid_prefix",
    "shuffle_segment_length_frames": 10,
    
    # Checkpointing & Early Stopping
    "save_best_model": True,
    "checkpoint_interval": 5,
    "early_stopping_patience": 15,
    "use_tensorboard": True,
    "use_scheduler": True,
    # ReduceLROnPlateau: LR is multiplied by scheduler_factor when the monitored
    # loss stops improving for scheduler_patience consecutive epochs (same as contrastive trainers).
    # Supervised script passes val CE loss into step(); contrastive code paths use val loss similarly.
    "scheduler_patience": 5,
    "scheduler_factor": 0.5,
    "scheduler_min_lr": 0.0,
    
    # Data loading (for WSL2/Linux - use multiprocessing)
    "num_workers": 8,  # Optimal for most systems (4-8 is usually best, 14 was too high causing slowdown)
    "pin_memory": True,  # Faster CPU→GPU transfer
}

# Data paths (will be auto-detected for WSL2)
# Full instant/windowed parquets (see notebooks saving to full_dataframes_*).
TRANSFORMER_DATA = {
    "data_drive": "D:",
    "data_dir": "TRANSFORMER_DEVELOPMENT/full_dataframes_3_29_2026",
    "instant_df_name": "instant_df.parquet",
    "windowed_df_name": "windowed_df.parquet",
    "output_drive": "D:",  # Drive for ALL OUTPUTS (models, checkpoints, logs, splits)
    "splits_dir": "TRANSFORMER_DEVELOPMENT/data_splits_3_29_2026",
    "splits_drive": "D:",
}

# Supervised training on final_population.
#
# Shared by: train_transformer_supervised_wsl2.py (64-D), train_transformer_supervised_6d_wsl2.py,
# train_transformer_supervised_32d_wsl2.py, train_transformer_supervised_4d_wsl2.py,
# train_transformer_supervised_focal_supcon_wsl2.py, train_transformer_supervised_focal_supcon_32d_wsl2.py,
# run_focal_supcon_four_presets_wsl2.py (subprocess), run_supervised_three_stage_wsl2.py (subprocess).
# Output subfolder names: ``TRANSFORMER_SUPERVISED_WSL2_OUTPUT_SUBDIRS`` (window_uid default) and
# ``TRANSFORMER_SUPERVISED_WSL2_OUTPUT_SUBDIRS_SLIDING`` (``--legacy-sliding-windows``); see ``supervised_wsl2_output_subdirs``.
#
# Augmentation, batch size, LR, windowing, workers, checkpoint interval, and early stopping come from
# TRANSFORMER_TRAINING. Default epoch target when scripts omit --epochs: TRANSFORMER_SUPERVISED["epochs"].
#
# Paths below are authoritative; data layout matches TRANSFORMER_DATA (same splits / full_dataframes tree).
TRANSFORMER_SUPERVISED = {
    "epochs": 200,
    "label_column": "final_population",
    "models_subdir": "models_supervised_population",
    "tensorboard_subdir": "tensorboard_logs_supervised_population",
    # Directory containing data_splits.pkl (train_transformer_supervised_wsl2.py).
    "splits_dir": "TRANSFORMER_DEVELOPMENT/data_splits_3_29_2026",
    "splits_drive": "D:",
    # data_splits.pkl may store stale absolute paths; substring remaps after F→D / WSL normalization.
    "splits_embedded_path_replacements": [
        (
            "Analyzed/ALL_DATA_BEING_USED/data_splits_3_20_2026",
            "TRANSFORMER_DEVELOPMENT/data_splits_3_29_2026",
        ),
        (
            "TRANSFORMER_DEVELOPMENT/data_splits_3_20_2026",
            "TRANSFORMER_DEVELOPMENT/data_splits_3_29_2026",
        ),
    ],
    # Labels: must match columns in windowed_df from full_dataframes_*.
    "windowed_parquet_path": "D:/TRANSFORMER_DEVELOPMENT/full_dataframes_3_29_2026/windowed_df.parquet",
}

# Output folder names under {output_drive}/TRANSFORMER_DEVELOPMENT/saved_data/ for supervised WSL
# variants (scripts override models_subdir / tensorboard_subdir with these so runs do not overwrite).
# Default 64-D run uses TRANSFORMER_SUPERVISED["models_subdir"] / ["tensorboard_subdir"] above.
#
# *window_uid* layout: separate from older sliding-window runs (same arch names, different data alignment).
# Use ``supervised_wsl2_output_subdirs(legacy_sliding_windows=True)`` for the previous folder names.
TRANSFORMER_SUPERVISED_WSL2_OUTPUT_SUBDIRS = {
    "models_6d": "models_supervised_population_6d_window_uid",
    "tensorboard_6d": "tensorboard_logs_supervised_population_6d_window_uid",
    "models_32d": "models_supervised_population_32d_window_uid",
    "tensorboard_32d": "tensorboard_logs_supervised_population_32d_window_uid",
    "models_4d": "models_supervised_population_4d_window_uid",
    "tensorboard_4d": "tensorboard_logs_supervised_population_4d_window_uid",
    "models_focal_supcon": "models_supervised_population_focal_supcon_window_uid",
    "tensorboard_focal_supcon": "tensorboard_logs_supervised_population_focal_supcon_window_uid",
    "models_focal_supcon_32d": "models_supervised_population_focal_supcon_32d_window_uid",
    "tensorboard_focal_supcon_32d": "tensorboard_logs_supervised_population_focal_supcon_32d_window_uid",
}

# Previous layout (sliding 60-frame windows, etc.) — used when training scripts pass --legacy-sliding-windows.
TRANSFORMER_SUPERVISED_WSL2_OUTPUT_SUBDIRS_SLIDING = {
    "models_6d": "models_supervised_population_6d",
    "tensorboard_6d": "tensorboard_logs_supervised_population_6d",
    "models_32d": "models_supervised_population_32d",
    "tensorboard_32d": "tensorboard_logs_supervised_population_32d",
    "models_4d": "models_supervised_population_4d",
    "tensorboard_4d": "tensorboard_logs_supervised_population_4d",
    "models_focal_supcon": "models_supervised_population_focal_supcon",
    "tensorboard_focal_supcon": "tensorboard_logs_supervised_population_focal_supcon",
    "models_focal_supcon_32d": "models_supervised_population_focal_supcon_32d",
    "tensorboard_focal_supcon_32d": "tensorboard_logs_supervised_population_focal_supcon_32d",
}


def supervised_wsl2_output_subdirs(legacy_sliding_windows: bool = False) -> dict[str, str]:
    """Subfolders under ``saved_data/`` for 6d/32d/focal supervised WSL2 scripts.

    Default (``legacy_sliding_windows=False``): window_uid + pad/mask pipeline — ``*_window_uid`` dirs.
    With ``legacy_sliding_windows=True``: original names (sliding-window extractor checkpoints).
    """
    if legacy_sliding_windows:
        return dict(TRANSFORMER_SUPERVISED_WSL2_OUTPUT_SUBDIRS_SLIDING)
    return dict(TRANSFORMER_SUPERVISED_WSL2_OUTPUT_SUBDIRS)


def supervised_wsl2_output_subdirs_tagged(
    legacy_sliding_windows: bool = False,
    output_tag: str | None = None,
) -> dict[str, str]:
    """Same as :func:`supervised_wsl2_output_subdirs`, but append ``_<output_tag>`` to each folder name.

    Use this (via training script ``--output-tag``) so new runs do not overwrite existing checkpoints
    under the default ``*_window_uid`` directories.
    """
    base = supervised_wsl2_output_subdirs(legacy_sliding_windows)
    if output_tag is None:
        return base
    tag = str(output_tag).strip()
    if not tag:
        return base
    for c in " /\\:":
        tag = tag.replace(c, "_")
    return {k: f"{v}_{tag}" for k, v in base.items()}


def shuffle_scale_angle_kwargs_from_cli(
    shuffle_segmentation_mode: str | None = None,
    shuffle_segment_length_frames: int | None = None,
) -> tuple[str, int]:
    """Resolve ``shuffle_scale_angle`` options: CLI overrides :data:`TRANSFORMER_TRAINING`."""
    mode = str(TRANSFORMER_TRAINING.get("shuffle_segmentation_mode", "valid_prefix")).lower().strip()
    slen = int(TRANSFORMER_TRAINING.get("shuffle_segment_length_frames", 10))
    if shuffle_segmentation_mode is not None:
        m = str(shuffle_segmentation_mode).lower().strip()
        if m not in ("valid_prefix", "legacy"):
            raise ValueError(
                f"shuffle_segmentation_mode must be 'valid_prefix' or 'legacy', got {shuffle_segmentation_mode!r}"
            )
        mode = m
    if shuffle_segment_length_frames is not None:
        slen = max(1, int(shuffle_segment_length_frames))
    return mode, slen


# Backbone for train_transformer_supervised_wsl2.py only (3 runs: CE / weights / stratified).
TRANSFORMER_SUPERVISED_ARCHITECTURE = {
    "name": "supervised64_h8_ff512_L3",
    "embed_dim": 64,
    "num_heads": 8,
    "ff_dim": 512,
    "num_layers": 3,
}

# Create necessary directories
os.makedirs(SAVED_DATA, exist_ok=True)
os.makedirs(TENSORBOARD_LOGS, exist_ok=True)

print("Config module loaded. Master directory is:", MASTER)
