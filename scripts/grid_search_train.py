#!/usr/bin/env python
"""
Grid Search Training Script for SPTnano Transformer Models
===========================================================

This script trains 16 different Transformer architectures for SPT data analysis.
It supports:
- Parallel data loading (num_workers > 0) for fast training
- Resume from checkpoint
- Early stopping
- Best model tracking
- TensorBoard logging
- Automatic skipping of already-trained models

Usage:
------
    # Train all models from scratch:
    python grid_search_train.py

    # Resume training model 0 from checkpoint, then continue with rest:
    python grid_search_train.py --resume 0

    # Train only a specific model:
    python grid_search_train.py --model 5

    # Train models 3-7:
    python grid_search_train.py --start 3 --end 8

    # Use different number of workers:
    python grid_search_train.py --workers 4

Author: Auto-generated for SPTnano project
"""

import argparse
import gc
import glob
import json
import os
import pickle
import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    """DataLoader with background thread prefetching - works on Windows!
    
    Uses threading instead of multiprocessing, so it avoids Windows pickle issues.
    Prefetches batches in a background thread while GPU processes current batch.
    """
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=4)

# Add src to path if running from scripts folder
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import polars as pl

from SPTnano.transformer import (
    TransformerMotionEncoder,
    TimeAwareMotionTrainer,
    TimeAwareTrajectoryDataset,
    create_smart_train_val_test_split,
    move_tracks_to_test_by_window_uids,
)


# =============================================================================
# CONFIGURATION - Edit these to match your setup
# =============================================================================

# Auto-detect if running on Windows or Linux (WSL2)
import platform
IS_LINUX = platform.system() == "Linux"

# Paths - automatically converted for WSL2
if IS_LINUX:
    # WSL2: Windows drives are mounted at /mnt/
    SAVE_DIR = "/mnt/f/Analyzed/FIXED_WINDOW_UIDS_20251030_155925"
    MODELS_BASE_DIR = "/mnt/f/TRANSFORMER_DEVELOPMENT/saved_data/models/"
    TENSORBOARD_BASE_DIR = "/mnt/f/TRANSFORMER_DEVELOPMENT/saved_data/tensorboard_logs/"
    print("🐧 Running on Linux/WSL2 - using /mnt/ paths")
else:
    # Windows paths
    DRIVE_LETTER = "F:"
    SAVE_DIR = f"{DRIVE_LETTER}/Analyzed/FIXED_WINDOW_UIDS_20251030_155925"
    MODELS_BASE_DIR = f"{DRIVE_LETTER}/TRANSFORMER_DEVELOPMENT/saved_data/models/"
    TENSORBOARD_BASE_DIR = f"{DRIVE_LETTER}/TRANSFORMER_DEVELOPMENT/saved_data/tensorboard_logs/"

SPLITS_PATH = os.path.join(MODELS_BASE_DIR, "data_splits.pkl")

# Curated ground truth window UIDs (for moving to test set)
CHOSEN_WINDOW_UIDS = {
    "transport": [
        'eekrw_131_7338_R1_1_2849.0_2908.0',
        'eekrw_132_7896_R1_10_301.0_360.0', 'eekrw_132_7896_R1_11_331.0_390.0', 'eekrw_132_7896_R1_9_271.0_330.0',
        'eekrw_72_27796_R1_4_2092.0_2151.0', 'eekrw_72_27796_R1_5_2122.0_2181.0',
        'eekrw_64_26280_R1_0_2401.0_2461.0',
        'eekrw_119_3224_R1_5_151.0_210.0', 'eekrw_119_3224_R1_6_181.0_240.0',
        'eekrw_122_4897_R1_0_3896.0_3955.0', 'eekrw_122_4897_R1_1_3926.0_3985.0', 'eekrw_122_4897_R1_2_3956.0_4015.0',
        'eekrw_124_5697_R1_0_822.0_882.0', 'eekrw_124_5697_R1_1_853.0_912.0',
        'eekrw_124_5718_R1_1_1226.0_1285.0',
        'eekrw_64_26137_R1_0_3723.0_3784.0',
        'eekrw_63_26099_R1_0_2180.0_2239.0',
        'eekrw_123_5096_R1_0_1923.0_1983.0'
    ],
    "bound": [
        'eemrw_33_3980_R1_10_1876.0_1936.0', 'eemrw_33_3980_R1_8_1816.0_1875.0', 'eemrw_33_3980_R1_9_1846.0_1905.0',
        'eekrw_64_26165_R1_0_4231.0_4294.0', 'eekrw_64_26165_R1_1_4263.0_4327.0',
        'eemrw_33_3513_R1_16_3418.0_3477.0', 'eemrw_33_3513_R1_17_3448.0_3507.0', 'eemrw_33_3513_R1_18_3478.0_3537.0',
        'eemrw_42_5444_R1_23_996.0_1055.0', 'eemrw_42_5444_R1_24_1026.0_1085.0', 'eemrw_42_5444_R1_25_1056.0_1115.0',
        'eemrw_34_4506_R1_0_2043.0_2106.0', 'eemrw_34_4506_R1_1_2077.0_2137.0',
        'eekrw_77_28613_R1_35_3643.0_3702.0', 'eekrw_77_28613_R1_36_3673.0_3732.0', 'eekrw_77_28613_R1_37_3703.0_3762.0',
        'eemrw_34_4188_R1_1_5333.0_5392.0',
        'eemrw_33_3399_R1_3_2262.0_2322.0',
        'eemrw_42_5293_R1_23_691.0_750.0', 'eemrw_42_5293_R1_24_721.0_780.0', 'eemrw_42_5293_R1_25_751.0_810.0',
        'eekrw_66_26462_R1_10_3647.0_3706.0', 'eekrw_66_26462_R1_11_3677.0_3736.0', 'eekrw_66_26462_R1_12_3707.0_3766.0',
        'eeh2h_eeh2x_22_3800_R2_R2_1_5583.0_5642.0', 'eeh2h_eeh2x_22_3800_R2_R2_2_5613.0_5672.0', 'eeh2h_eeh2x_22_3800_R2_R2_3_5643.0_5702.0'
    ],
    "transient": [
        'eeh2h_38_12054_R1_0_2432.0_2493.0',
        'eekrw_123_5265_R1_1_5141.0_5200.0',
        'eekrw_73_28118_R1_0_1708.0_1768.0', 'eekrw_73_28118_R1_1_1739.0_1798.0', 'eekrw_73_28118_R1_2_1769.0_1828.0',
        'eeh2h_30_10132_R1_0_2059.0_2118.0',
        'eemrw_37_4775_R1_0_2880.0_2950.0', 'eemrw_37_4775_R1_1_2919.0_2983.0',
        'eemrw_34_4032_R1_0_2811.0_2872.0',
        'eemrw_33_3474_R1_0_2638.0_2699.0',
        'eemrw_40_5120_R1_0_3095.0_3155.0',
        'eeh2h_eeh2x_31_7315_R2_R2_0_5427.0_5492.0',
        'eeh2h_eeh2x_4_10006_R1_R1_0_5714.0_5773.0'
    ]
}

# Model architecture grid (4 selected combinations)
# Model 0: Baseline (smallest) - already trained
# Model 1: Deeper + more FF capacity
# Model 2: Bigger embedding
# Model 3: Maximum capacity (largest)
MODEL_CONFIGS = [
    {'name': 'e64_h4_ff128_L2',  'embed_dim': 64,  'num_heads': 4, 'ff_dim': 128, 'num_layers': 2},  # ~67K params
    {'name': 'e64_h4_ff256_L3',  'embed_dim': 64,  'num_heads': 4, 'ff_dim': 256, 'num_layers': 3},  # ~130K params
    {'name': 'e128_h4_ff256_L2', 'embed_dim': 128, 'num_heads': 4, 'ff_dim': 256, 'num_layers': 2},  # ~200K params
    {'name': 'e128_h8_ff512_L3', 'embed_dim': 128, 'num_heads': 8, 'ff_dim': 512, 'num_layers': 3},  # ~540K params
]

# Training hyperparameters
WINDOW_SIZE = 60
OVERLAP = 30
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 1e-4
TEMPERATURE = 0.5

# Loss configuration
USE_ADJACENT_SUBWINDOW = True
ADJACENT_SUBWINDOW_WEIGHT = 0.5
ADJACENT_TEMPERATURE = 0.7
SUBWINDOW_SIZE = 10
MASK_SAME_TRACK_NEGATIVES = True

# Augmentation
AUGMENTATION_TYPE = "combined"
NOISE_STRENGTH = 0.012
SCALE_STRENGTH = 0.2

# Checkpointing & Early Stopping
CHECKPOINT_INTERVAL = 5
EARLY_STOPPING_PATIENCE = 15
SAVE_BEST_MODEL = True

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_raw_data():
    """Load raw parquet data files."""
    print(f"📂 Loading raw data from: {SAVE_DIR}", flush=True)
    
    instant_df = pl.read_parquet(os.path.join(SAVE_DIR, 'master_instant_df_FIXED.parquet'))
    windowed_df = pl.read_parquet(os.path.join(SAVE_DIR, 'master_windowed_df_FIXED.parquet'))
    
    # Update drive letters if needed
    if IS_LINUX:
        # Convert Windows paths to WSL2 paths
        instant_df = instant_df.with_columns(
            pl.col('foldername').str.replace('Z:', '/mnt/z').str.replace('F:', '/mnt/f')
        )
        windowed_df = windowed_df.with_columns(
            pl.col('foldername').str.replace('Z:', '/mnt/z').str.replace('F:', '/mnt/f')
        )
    else:
        instant_df = instant_df.with_columns(
            pl.col('foldername').str.replace('Z:', 'F:')
        )
        windowed_df = windowed_df.with_columns(
            pl.col('foldername').str.replace('Z:', 'F:')
        )
    
    # Remove bad fits
    print(f"   Before filtering: {windowed_df.shape[0]:,} windows", flush=True)
    windowed_df = windowed_df.filter(pl.col('bad_fit_flag') == False)
    print(f"   After filtering: {windowed_df.shape[0]:,} windows", flush=True)
    
    print(f"\n✅ Data loaded:", flush=True)
    print(f"   instant_df: {len(instant_df):,} rows", flush=True)
    print(f"   windowed_df: {len(windowed_df):,} rows", flush=True)
    
    return instant_df, windowed_df


def create_and_save_splits():
    """Create train/val/test splits and save them."""
    print("\n" + "=" * 70, flush=True)
    print("📊 CREATING DATA SPLITS", flush=True)
    print("=" * 70, flush=True)
    
    # Load raw data
    instant_df, windowed_df = load_raw_data()
    
    # Create splits - THIS CAN TAKE A FEW MINUTES on large datasets!
    print("\n📊 Creating train/val/test split...", flush=True)
    print("   ⏳ This may take 2-5 minutes for 51M rows...", flush=True)
    
    try:
        train_df, val_df, test_df, split_info = create_smart_train_val_test_split(
            instant_df,
            val_split=0.15,
            test_split=0.35,
            split_strategy="fixed_cells",
            random_seed=42,
            balancing_features=["mol"],
            cells_per_condition=6,
            save_dir=SAVE_DIR,
            save_test=True,
        )
    except Exception as e:
        print(f"\n❌ Error during split creation: {e}", flush=True)
        raise
    
    print(f"\n📊 Initial split complete:", flush=True)
    print(f"   Train: {len(train_df):,} points", flush=True)
    print(f"   Val:   {len(val_df):,} points", flush=True)
    print(f"   Test:  {len(test_df):,} points", flush=True)
    
    # Move curated ground truth tracks to test
    print("\n🔄 Moving curated ground truth tracks to test set...", flush=True)
    train_df, val_df, test_df, move_info = move_tracks_to_test_by_window_uids(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        window_uids_to_move=CHOSEN_WINDOW_UIDS,
        windowed_df=windowed_df,
    )
    
    print(f"\n📊 Move Summary:", flush=True)
    print(f"   Tracks moved from train: {move_info['from_train']}", flush=True)
    print(f"   Tracks moved from val: {move_info['from_val']}", flush=True)
    print(f"   Already in test: {move_info['already_in_test']}", flush=True)
    print(f"   Window UIDs found: {len(move_info['window_uids_found'])} / {len(move_info['window_uids_found']) + len(move_info['window_uids_not_found'])}", flush=True)
    
    # Save splits
    print(f"\n💾 Saving splits to: {SPLITS_PATH}", flush=True)
    os.makedirs(MODELS_BASE_DIR, exist_ok=True)
    
    split_data = {
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'split_config': {
            'window_size': WINDOW_SIZE,
            'overlap': OVERLAP,
            'random_seed': 42,
            'split_strategy': 'fixed_cells',
            'balancing_features': ['mol']
        }
    }
    
    with open(SPLITS_PATH, 'wb') as f:
        pickle.dump(split_data, f)
    
    print(f"\n✅ Splits saved to: {SPLITS_PATH}", flush=True)
    print(f"   Train: {len(train_df):,} points", flush=True)
    print(f"   Val:   {len(val_df):,} points", flush=True)
    print(f"   Test:  {len(test_df):,} points", flush=True)
    
    return train_df, val_df, test_df


def load_data_splits():
    """Load pre-computed train/val/test splits."""
    if not os.path.exists(SPLITS_PATH):
        raise FileNotFoundError(
            f"No saved splits found at {SPLITS_PATH}.\n"
            f"Run with --create-splits first to create and save the splits."
        )
    
    print(f"📂 Loading data splits from: {SPLITS_PATH}", flush=True)
    with open(SPLITS_PATH, 'rb') as f:
        split_data = pickle.load(f)
    
    train_df = split_data['train_df']
    val_df = split_data['val_df']
    test_df = split_data['test_df']
    
    print(f"   Train: {len(train_df):,} points", flush=True)
    print(f"   Val:   {len(val_df):,} points", flush=True)
    print(f"   Test:  {len(test_df):,} points", flush=True)
    
    return train_df, val_df, test_df


def create_dataloaders(train_df, val_df, num_workers=0):
    """Create datasets and dataloaders with background thread prefetching.
    
    Uses prefetch_generator (threading-based) instead of multiprocessing.
    This works on Windows without pickle errors!
    """
    print(f"\n📊 Creating datasets with background prefetching...", flush=True)
    
    train_dataset = TimeAwareTrajectoryDataset(
        train_df, window_size=WINDOW_SIZE, overlap=OVERLAP, min_track_length=WINDOW_SIZE
    )
    val_dataset = TimeAwareTrajectoryDataset(
        val_df, window_size=WINDOW_SIZE, overlap=OVERLAP, min_track_length=WINDOW_SIZE
    )
    
    # Create dataloaders with DataLoaderX (thread-based prefetching)
    # This uses a background thread to prefetch batches - works on Windows!
    train_loader = DataLoaderX(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,  # Keep at 0, prefetching happens via threading
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoaderX(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    print(f"   Train: {len(train_dataset):,} window pairs → {len(train_loader):,} batches", flush=True)
    print(f"   Val:   {len(val_dataset):,} window pairs → {len(val_loader):,} batches", flush=True)
    print(f"   🧵 Background thread prefetching enabled (max_prefetch=4)", flush=True)
    
    return train_loader, val_loader


def is_model_trained(model_name):
    """Check if a model has already been fully trained."""
    model_dir = os.path.join(MODELS_BASE_DIR, model_name)
    final_path = os.path.join(model_dir, "final_model.pt")
    return os.path.exists(final_path)


def get_latest_checkpoint(model_name):
    """Find the checkpoint with the MOST epochs trained (look inside, not at filename!)."""
    checkpoint_dir = os.path.join(MODELS_BASE_DIR, model_name, "checkpoints")
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt"))
    
    if not checkpoints:
        return None, 0
    
    # Find checkpoint with most training history (fixes bug where filenames are wrong)
    print("📂 Scanning checkpoints for actual epoch counts...")
    best_checkpoint = None
    best_epoch_count = 0
    for cp_path in checkpoints:
        cp_data = torch.load(cp_path, weights_only=False)
        epoch_count = len(cp_data.get('train_losses', []))
        filename_epoch = int(cp_path.split('_')[-1].split('.')[0])
        print(f"   {os.path.basename(cp_path)}: filename={filename_epoch}, actual={epoch_count} epochs")
        if epoch_count > best_epoch_count:
            best_epoch_count = epoch_count
            best_checkpoint = cp_path
    
    print(f"✅ Selected: {os.path.basename(best_checkpoint)} with {best_epoch_count} epochs")
    return best_checkpoint, best_epoch_count


def train_model(model_index, train_loader, val_loader, resume_from_checkpoint=False):
    """Train a single model."""
    config = MODEL_CONFIGS[model_index]
    model_name = config['name']
    embed_dim = config['embed_dim']
    num_heads = config['num_heads']
    ff_dim = config['ff_dim']
    num_layers = config['num_layers']
    
    # Paths
    model_dir = os.path.join(MODELS_BASE_DIR, model_name)
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    tensorboard_dir = os.path.join(TENSORBOARD_BASE_DIR, model_name)
    best_model_path = os.path.join(model_dir, "best_model.pt")
    final_model_path = os.path.join(model_dir, "final_model.pt")
    config_path = os.path.join(model_dir, "config.json")
    
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print(f"🏗️ MODEL {model_index + 1}/16: {model_name}")
    print(f"   Architecture: {embed_dim}d, {num_heads} heads, {ff_dim} FF, {num_layers} layers")
    print("=" * 70)
    
    # Check for resume
    start_epoch = 0
    checkpoint_data = None
    
    if resume_from_checkpoint:
        checkpoint_path, start_epoch = get_latest_checkpoint(model_name)
        if checkpoint_path:
            print(f"📥 Resuming from checkpoint: epoch {start_epoch}")
            checkpoint_data = torch.load(checkpoint_path, weights_only=False)
        else:
            print("⚠️ No checkpoint found, starting from scratch")
            resume_from_checkpoint = False
    
    # Save config
    config_dict = {
        "model_index": model_index,
        "model_name": model_name,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "ff_dim": ff_dim,
        "num_layers": num_layers,
        "window_size": WINDOW_SIZE,
        "overlap": OVERLAP,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "temperature": TEMPERATURE,
        "use_adjacent_subwindow": USE_ADJACENT_SUBWINDOW,
        "adjacent_subwindow_weight": ADJACENT_SUBWINDOW_WEIGHT,
        "adjacent_temperature": ADJACENT_TEMPERATURE,
        "subwindow_size": SUBWINDOW_SIZE,
        "mask_same_track": MASK_SAME_TRACK_NEGATIVES,
        "augmentation_type": AUGMENTATION_TYPE,
        "noise_strength": NOISE_STRENGTH,
        "scale_strength": SCALE_STRENGTH,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Create model
    model = TransformerMotionEncoder(
        input_dim=3,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {n_params:,}")
    
    # NOTE: DO NOT clear TensorBoard event files!
    # TensorBoard handles multiple event files correctly by merging them.
    # We use epoch_offset to write at correct step numbers, so new data
    # will continue from where we left off.
    
    # Create trainer with epoch_offset for correct numbering when resuming
    trainer = TimeAwareMotionTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        lr=LEARNING_RATE,
        device=DEVICE,
        temperature=TEMPERATURE,
        use_augmentation=True,
        augmentation_type=AUGMENTATION_TYPE,
        noise_strength=NOISE_STRENGTH,
        scale_strength=SCALE_STRENGTH,
        use_adjacent_subwindow=USE_ADJACENT_SUBWINDOW,
        adjacent_subwindow_weight=ADJACENT_SUBWINDOW_WEIGHT,
        adjacent_temperature=ADJACENT_TEMPERATURE,
        subwindow_size=SUBWINDOW_SIZE,
        temporal_weight=0.0,
        use_within_window_consistency=False,
        mask_same_track=MASK_SAME_TRACK_NEGATIVES,
        save_best_model=SAVE_BEST_MODEL,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        use_tensorboard=True,
        tensorboard_log_dir=tensorboard_dir,
        use_scheduler=True,
        epoch_offset=start_epoch,  # For correct TensorBoard/checkpoint numbering
    )
    
    # Restore state if resuming
    if resume_from_checkpoint and checkpoint_data:
        trainer.model.load_state_dict(checkpoint_data['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint_data and trainer.scheduler:
            trainer.scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        
        # Restore loss history
        trainer.train_losses = checkpoint_data.get('train_losses', [])
        trainer.val_losses = checkpoint_data.get('val_losses', [])
        trainer.temporal_losses = checkpoint_data.get('temporal_losses', [])
        trainer.augmentation_losses = checkpoint_data.get('augmentation_losses', [])
        trainer.within_window_losses = checkpoint_data.get('within_window_losses', [])
        trainer.adjacent_subwindow_losses = checkpoint_data.get('adjacent_subwindow_losses', [])
        trainer.val_temporal_losses = checkpoint_data.get('val_temporal_losses', [])
        trainer.val_augmentation_losses = checkpoint_data.get('val_augmentation_losses', [])
        trainer.val_within_window_losses = checkpoint_data.get('val_within_window_losses', [])
        trainer.val_adjacent_subwindow_losses = checkpoint_data.get('val_adjacent_subwindow_losses', [])
        
        # Restore best model tracking
        trainer.best_val_loss = checkpoint_data.get('best_val_loss', float('inf'))
        trainer.best_epoch = checkpoint_data.get('best_epoch', -1)
        trainer.best_model_state = checkpoint_data.get('best_model_state', None)
        
        # Restore early stopping counter
        trainer.epochs_without_improvement = checkpoint_data.get('epochs_without_improvement', 0)
        
        # NOTE: No need to replay to TensorBoard!
        # The old event files are preserved, and new training will write at
        # correct step numbers (epoch_offset + local_epoch). TensorBoard
        # automatically merges multiple event files.
        print(f"📊 TensorBoard: Keeping existing logs, new data will append at epoch {start_epoch}+")
    
    # Train
    remaining_epochs = EPOCHS - start_epoch
    print(f"\n🚀 Training for {remaining_epochs} epochs (starting from {start_epoch + 1})...")
    trainer.train(epochs=remaining_epochs)
    
    # Save models
    trainer.save_model(final_model_path)
    if trainer.best_epoch >= 0 and SAVE_BEST_MODEL:
        trainer.restore_best_model()
        trainer.save_model(best_model_path)
    
    # Results
    result = {
        'model_index': model_index,
        'model_name': model_name,
        'n_params': n_params,
        'final_train_loss': trainer.train_losses[-1] if trainer.train_losses else None,
        'final_val_loss': trainer.val_losses[-1] if trainer.val_losses else None,
        'best_val_loss': trainer.best_val_loss if trainer.best_epoch >= 0 else None,
        'best_epoch': trainer.best_epoch + 1 if trainer.best_epoch >= 0 else None,
        'epochs_trained': len(trainer.train_losses),
    }
    
    print(f"\n💾 Saved to: {model_dir}/")
    print(f"   Final val loss: {result['final_val_loss']:.4f}" if result['final_val_loss'] else "")
    if result['best_val_loss']:
        print(f"   Best val loss:  {result['best_val_loss']:.4f} (epoch {result['best_epoch']})")
    
    # Cleanup
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Grid Search Training for SPTnano Transformer Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python grid_search_train.py --create-splits    # Create splits first (only needed once!)
  python grid_search_train.py --resume 0 --skip-trained  # Resume model 0, then continue
  python grid_search_train.py --skip-trained     # Train all, skip already finished
  python grid_search_train.py --model 5          # Train only model 5
  python grid_search_train.py --start 3 --end 8  # Train models 3-7
  python grid_search_train.py --workers 4        # Use 4 workers for data loading
  python grid_search_train.py --list             # List all 16 models and status
        """
    )
    
    parser.add_argument('--resume', type=int, default=None, metavar='N',
                        help='Resume training model N from checkpoint')
    parser.add_argument('--model', type=int, default=None, metavar='N',
                        help='Train only model N (0-15)')
    parser.add_argument('--start', type=int, default=0, metavar='N',
                        help='Start from model N (default: 0)')
    parser.add_argument('--end', type=int, default=16, metavar='N',
                        help='End at model N, exclusive (default: 16)')
    parser.add_argument('--workers', type=int, default=0, metavar='N',
                        help='Number of data loading workers (default: 0 for Windows compatibility)')
    parser.add_argument('--skip-trained', action='store_true',
                        help='Skip models that already have final_model.pt')
    parser.add_argument('--list', action='store_true',
                        help='List all model configurations and exit')
    parser.add_argument('--create-splits', action='store_true',
                        help='Create and save train/val/test splits from raw data, then exit')
    
    args = parser.parse_args()
    
    # Create splits and exit
    if args.create_splits:
        create_and_save_splits()
        print("\n✅ Splits created! Now run again without --create-splits to train.")
        return
    
    # List models and exit
    if args.list:
        print("\n📋 Model Configurations:")
        print("-" * 60)
        print(f"{'Idx':<4} {'Name':<20} {'Dim':<5} {'Heads':<6} {'FF':<5} {'Layers':<6}")
        print("-" * 60)
        for i, cfg in enumerate(MODEL_CONFIGS):
            status = "✅ trained" if is_model_trained(cfg['name']) else ""
            print(f"{i:<4} {cfg['name']:<20} {cfg['embed_dim']:<5} {cfg['num_heads']:<6} {cfg['ff_dim']:<5} {cfg['num_layers']:<6} {status}")
        return
    
    # Print header
    print("=" * 70)
    print("🧬 SPTnano Grid Search Training")
    print(f"   Device: {DEVICE}")
    print(f"   Workers: {args.workers}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Load data
    train_df, val_df, test_df = load_data_splits()
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(train_df, val_df, num_workers=args.workers)
    
    # Determine which models to train
    if args.model is not None:
        model_indices = [args.model]
    else:
        model_indices = list(range(args.start, args.end))
    
    # Track results
    results = []
    
    # Resume a specific model first if requested
    if args.resume is not None:
        print(f"\n🔄 Resuming model {args.resume} from checkpoint...")
        result = train_model(args.resume, train_loader, val_loader, resume_from_checkpoint=True)
        results.append(result)
        
        # Remove from list so we don't train it again
        if args.resume in model_indices:
            model_indices.remove(args.resume)
    
    # Train remaining models
    for model_idx in model_indices:
        model_name = MODEL_CONFIGS[model_idx]['name']
        
        # Skip if already trained
        if args.skip_trained and is_model_trained(model_name):
            print(f"\n⏭️ Skipping model {model_idx} ({model_name}) - already trained")
            continue
        
        result = train_model(model_idx, train_loader, val_loader, resume_from_checkpoint=False)
        results.append(result)
    
    # Print summary
    print("\n" + "=" * 70)
    print("🎉 GRID SEARCH COMPLETE!")
    print("=" * 70)
    
    if results:
        print("\n📊 Results Summary:")
        print("-" * 70)
        print(f"{'Model':<20} {'Params':<12} {'Best Val':<12} {'Best Epoch':<12}")
        print("-" * 70)
        
        # Sort by best val loss
        sorted_results = sorted(results, key=lambda x: x['best_val_loss'] or float('inf'))
        for r in sorted_results:
            best_val = f"{r['best_val_loss']:.4f}" if r['best_val_loss'] else "N/A"
            best_ep = str(r['best_epoch']) if r['best_epoch'] else "N/A"
            print(f"{r['model_name']:<20} {r['n_params']:,}  {best_val:<12} {best_ep:<12}")
        
        # Save results to CSV
        import pandas as pd
        results_df = pd.DataFrame(results)
        results_path = os.path.join(MODELS_BASE_DIR, "grid_search_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\n📄 Results saved to: {results_path}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    # This guard is required for Windows multiprocessing
    main()
