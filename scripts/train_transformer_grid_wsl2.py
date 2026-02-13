#!/usr/bin/env python3
"""
WSL2 Training Script for Transformer Grid Search
================================================

This script trains transformer models with proper Linux multiprocessing for optimal
GPU utilization. Designed to run on WSL2 (Linux) with full multiprocessing support.

Features:
- Proper multiprocessing (num_workers > 0) for parallel data loading
- Automatic WSL2 path conversion (Windows drives → /mnt/)
- Same training logic as notebook
- Resume from checkpoint support
- Automatic model skipping

Usage:
------
    # Train all models (auto mode - skips completed, resumes incomplete):
    python train_transformer_grid_wsl2.py

    # Resume specific model:
    python train_transformer_grid_wsl2.py --resume 0

    # Train specific models only:
    python train_transformer_grid_wsl2.py --models 0 1 2

    # Start fresh (ignore checkpoints):
    python train_transformer_grid_wsl2.py --fresh

    # Adjust number of data loading workers:
    python train_transformer_grid_wsl2.py --workers 16

Requirements:
-------------
- WSL2 (Linux environment)
- CUDA-capable GPU
- All dependencies installed in WSL2 environment
"""

import argparse
import gc
import glob
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add src to path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import polars as pl

from SPTnano.config import (
    TRANSFORMER_ARCHITECTURES,
    TRANSFORMER_TEMPERATURES,
    TRANSFORMER_TRAINING,
    TRANSFORMER_DATA,
)
from SPTnano.transformer import (
    TransformerMotionEncoder,
    TimeAwareMotionTrainer,
    TimeAwareTrajectoryDataset,
    create_smart_train_val_test_split,
)


# =============================================================================
# PATH CONVERSION FOR WSL2
# =============================================================================

def convert_wsl_path(windows_path):
    """
    Convert Windows path to WSL2 path.
    
    Examples:
        F:/Data/file.parquet → /mnt/f/Data/file.parquet
        F:\\Data\\file.parquet → /mnt/f/Data/file.parquet
    """
    if platform.system() != "Linux":
        return windows_path  # Already on Windows
    
    # Normalize path separators
    path = windows_path.replace("\\", "/")
    
    # Convert drive letter to /mnt/ format
    if len(path) >= 2 and path[1] == ":":
        drive_letter = path[0].lower()
        rest = path[2:]  # Remove ":/" or ":\\"
        return f"/mnt/{drive_letter}{rest}"
    
    return path


def get_data_paths():
    """Get data paths, converting to WSL2 format if needed."""
    data_drive = TRANSFORMER_DATA["data_drive"]  # F: for input data
    output_drive = TRANSFORMER_DATA.get("output_drive", "D:")  # D: for ALL outputs
    data_dir = TRANSFORMER_DATA["data_dir"]
    splits_dir = TRANSFORMER_DATA["splits_dir"]  # Directory containing data_splits.pkl
    splits_drive = TRANSFORMER_DATA.get("splits_drive", output_drive)  # Use output_drive for splits
    
    # Build Windows paths
    # Input data (from F:)
    data_base = f"{data_drive}/{data_dir}"
    instant_df_path = f"{data_base}/{TRANSFORMER_DATA['instant_df_name']}"
    windowed_df_path = f"{data_base}/{TRANSFORMER_DATA['windowed_df_name']}"
    
    # ALL OUTPUTS go to D: drive (models, checkpoints, logs, splits)
    models_base = f"{output_drive}/TRANSFORMER_DEVELOPMENT/saved_data/models"
    tensorboard_base = f"{output_drive}/TRANSFORMER_DEVELOPMENT/saved_data/tensorboard_logs"
    splits_full_path = f"{splits_drive}/{splits_dir}"
    
    # Convert to WSL2 if on Linux
    if platform.system() == "Linux":
        instant_df_path = convert_wsl_path(instant_df_path)
        windowed_df_path = convert_wsl_path(windowed_df_path)
        models_base = convert_wsl_path(models_base)
        tensorboard_base = convert_wsl_path(tensorboard_base)
        splits_full_path = convert_wsl_path(splits_full_path)
        print("🐧 Running on WSL2/Linux - paths converted to /mnt/ format")
    else:
        print("🪟 Running on Windows - using native paths")
    
    return {
        "instant_df_path": instant_df_path,
        "windowed_df_path": windowed_df_path,
        "models_base_dir": models_base,
        "tensorboard_base_dir": tensorboard_base,
        "splits_path": splits_full_path,
    }


# =============================================================================
# MODEL CONFIG GENERATION
# =============================================================================

def generate_model_configs():
    """Generate model configs from architectures × temperatures."""
    model_configs = []
    for arch in TRANSFORMER_ARCHITECTURES:
        for temp in TRANSFORMER_TEMPERATURES:
            name = f"{arch['name']}_t{temp}"
            config = {
                'name': name,
                'embed_dim': arch['embed_dim'],
                'num_heads': arch['num_heads'],
                'ff_dim': arch['ff_dim'],
                'num_layers': arch['num_layers'],
                'temperature': temp,
            }
            model_configs.append(config)
    return model_configs


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data_splits(splits_path):
    """Load pre-computed train/val/test splits."""
    import pickle
    
    # Handle case where splits_path is a directory containing data_splits.pkl
    if os.path.isdir(splits_path):
        pickle_file = os.path.join(splits_path, "data_splits.pkl")
        if not os.path.exists(pickle_file):
            # List what's actually in the directory
            try:
                contents = os.listdir(splits_path)
                raise FileNotFoundError(
                    f"Split directory found but data_splits.pkl not found: {pickle_file}\n"
                    f"Directory contents: {contents}"
                )
            except PermissionError:
                raise FileNotFoundError(
                    f"Split directory found but data_splits.pkl not found: {pickle_file}\n"
                    f"Permission denied accessing directory"
                )
        splits_path = pickle_file
    elif os.path.isfile(splits_path):
        # It's already a file, use it directly
        pass
    else:
        # Path doesn't exist - provide helpful error
        parent_dir = os.path.dirname(splits_path) if os.path.dirname(splits_path) else splits_path
        if os.path.exists(parent_dir):
            try:
                contents = os.listdir(parent_dir)
                raise FileNotFoundError(
                    f"Split path not found: {splits_path}\n"
                    f"Parent directory exists. Contents: {contents}"
                )
            except PermissionError:
                raise FileNotFoundError(
                    f"Split path not found: {splits_path}\n"
                    f"Parent directory exists but permission denied"
                )
        else:
            raise FileNotFoundError(
                f"Split path not found: {splits_path}\n"
                f"Parent directory also doesn't exist: {parent_dir}\n"
                f"Check if F: drive is mounted in WSL2: ls /mnt/f"
            )
    
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Split file not found: {splits_path}")
    
    with open(splits_path, 'rb') as f:
        split_data = pickle.load(f)
    
    split_config = split_data.get('split_config', {})
    is_polars = split_config.get('is_polars', True)
    
    # Convert paths from F: to D: drive (if they point to F:)
    # This handles the case where pickle was created with F: paths but we're using D: now
    splits_drive = TRANSFORMER_DATA.get("splits_drive", "D:")
    
    def convert_path(path):
        """Convert F: drive paths to D: drive paths and fix .pkl in directory name."""
        if not isinstance(path, str):
            return path
        
        original_path = path
        
        # Fix: Remove .pkl from directory name if present (old pickle had wrong path)
        # e.g., data_splits_withheirarchalgates.pkl/train_df.parquet → data_splits_withheirarchalgates/train_df.parquet
        path = path.replace(".pkl\\", "\\").replace(".pkl/", "/")
        
        # Replace F: with D: (handle all formats)
        if "F:" in path or "/mnt/f" in path.lower():
            # Replace F: with D:
            path = path.replace("F:", splits_drive)
            path = path.replace("F:\\", f"{splits_drive}\\")
            path = path.replace("f:", splits_drive.lower())
            path = path.replace("f:\\", f"{splits_drive.lower()}\\")
            # Handle WSL2 paths
            path = path.replace("/mnt/f/", f"/mnt/{splits_drive.lower().replace(':', '')}/")
            path = path.replace("/mnt/F/", f"/mnt/{splits_drive.lower().replace(':', '')}/")
        
        # Convert Windows path to WSL2 if on Linux
        if platform.system() == "Linux":
            if path.startswith("D:") or path.startswith("D:\\") or path.startswith("d:") or path.startswith("d:\\"):
                path = convert_wsl_path(path)
        
        # Debug: print conversion if path changed
        if original_path != path:
            print(f"   🔄 Converted path: {original_path[:60]}... → {path[:60]}...")
        
        return path
    
    train_path = convert_path(split_config['train_path'])
    val_path = convert_path(split_config['val_path'])
    test_path = convert_path(split_config['test_path'])
    
    # Load from parquet files
    train_df = pl.read_parquet(train_path)
    val_df = pl.read_parquet(val_path)
    test_df = pl.read_parquet(test_path)
    
    print(f"✅ Loaded splits: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")
    
    return train_df, val_df, test_df


def create_dataloaders(train_df, val_df, num_workers=8):
    """
    Create dataloaders with proper multiprocessing for Linux/WSL2.
    
    On Linux, we can use num_workers > 0 for true parallel data loading.
    This is much faster than Windows threading approach.
    """
    print(f"\n📊 Creating datasets with multiprocessing (num_workers={num_workers})...")
    
    window_size = TRANSFORMER_TRAINING["window_size"]
    overlap = TRANSFORMER_TRAINING["overlap"]
    min_track_length = TRANSFORMER_TRAINING["min_track_length"]
    batch_size = TRANSFORMER_TRAINING["batch_size"]
    
    train_dataset = TimeAwareTrajectoryDataset(
        train_df, window_size=window_size, overlap=overlap, min_track_length=min_track_length
    )
    val_dataset = TimeAwareTrajectoryDataset(
        val_df, window_size=window_size, overlap=overlap, min_track_length=min_track_length
    )
    
    # Use proper multiprocessing on Linux
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=TRANSFORMER_TRAINING["pin_memory"],
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=TRANSFORMER_TRAINING["pin_memory"],
        persistent_workers=True if num_workers > 0 else False,
    )
    
    print(f"   ✅ Train: {len(train_dataset):,} pairs → {len(train_loader):,} batches")
    print(f"   ✅ Val:   {len(val_dataset):,} pairs → {len(val_loader):,} batches")
    print(f"   🚀 Multiprocessing enabled: {num_workers} workers per dataloader")
    
    return train_loader, val_loader


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def is_model_complete(model_dir):
    """Check if model training is complete."""
    final_path = os.path.join(model_dir, "final_model.pt")
    return os.path.exists(final_path)


def find_best_checkpoint(checkpoint_dir):
    """Find checkpoint with most epochs."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt"))
    if not checkpoints:
        return None, 0
    
    best_checkpoint = None
    best_epoch_count = 0
    
    for cp_path in checkpoints:
        try:
            cp_data = torch.load(cp_path, map_location='cpu', weights_only=False)
            epoch_count = len(cp_data.get('train_losses', []))
            if epoch_count > best_epoch_count:
                best_epoch_count = epoch_count
                best_checkpoint = cp_path
        except Exception as e:
            print(f"   ⚠️  Could not load {os.path.basename(cp_path)}: {e}")
            continue
    
    return best_checkpoint, best_epoch_count


def train_model(model_index, config, train_loader, val_loader, paths, args):
    """Train a single model."""
    model_name = config['name']
    model_dir = os.path.join(paths['models_base_dir'], model_name)
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    tensorboard_dir = os.path.join(paths['tensorboard_base_dir'], model_name)
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    best_model_path = os.path.join(model_dir, "best_model.pt")
    final_model_path = os.path.join(model_dir, "final_model.pt")
    config_path = os.path.join(model_dir, "config.json")
    
    print("\n" + "=" * 80)
    print(f"🏗️ MODEL {model_index + 1}/{len(generate_model_configs())}: {model_name}")
    print(f"   Architecture: {config['embed_dim']}d, {config['num_heads']} heads, "
          f"{config['ff_dim']} FF, {config['num_layers']} layers")
    print(f"   Temperature: {config['temperature']}")
    print("=" * 80)
    
    # Check for existing checkpoint
    resume_epoch = 0
    checkpoint_to_load = None
    
    if args.mode in ["auto", "resume"]:
        checkpoint_to_load, resume_epoch = find_best_checkpoint(checkpoint_dir)
        if checkpoint_to_load:
            max_epochs = TRANSFORMER_TRAINING["epochs"]
            if resume_epoch < max_epochs:
                print(f"📂 Found checkpoint: {os.path.basename(checkpoint_to_load)} ({resume_epoch} epochs)")
            else:
                print(f"✅ Model already completed {max_epochs} epochs")
                checkpoint_to_load = None
    
    # Save config
    config_dict = {
        "model_index": model_index,
        "model_name": model_name,
        "embed_dim": config['embed_dim'],
        "num_heads": config['num_heads'],
        "ff_dim": config['ff_dim'],
        "num_layers": config['num_layers'],
        **TRANSFORMER_TRAINING,
        "temperature": config['temperature'],
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Create model
    model = TransformerMotionEncoder(
        input_dim=3,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        num_layers=config['num_layers'],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {n_params:,}")
    
    # Load checkpoint if resuming
    if checkpoint_to_load:
        checkpoint = torch.load(checkpoint_to_load, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✅ Loaded model state from checkpoint")
    
    # Create trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = TimeAwareMotionTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        lr=TRANSFORMER_TRAINING["learning_rate"],
        device=device,
        temperature=config['temperature'],
        use_augmentation=True,
        augmentation_type=TRANSFORMER_TRAINING["augmentation_type"],
        noise_strength=TRANSFORMER_TRAINING["noise_strength"],
        scale_strength=TRANSFORMER_TRAINING["scale_strength"],
        use_adjacent_subwindow=TRANSFORMER_TRAINING["use_adjacent_subwindow"],
        adjacent_subwindow_weight=TRANSFORMER_TRAINING["adjacent_subwindow_weight"],
        adjacent_temperature=TRANSFORMER_TRAINING["adjacent_temperature"],
        subwindow_size=TRANSFORMER_TRAINING["subwindow_size"],
        temporal_weight=0.0,
        use_within_window_consistency=False,
        mask_same_track=TRANSFORMER_TRAINING["mask_same_track_negatives"],
        save_best_model=TRANSFORMER_TRAINING["save_best_model"],
        checkpoint_interval=TRANSFORMER_TRAINING["checkpoint_interval"],
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=TRANSFORMER_TRAINING["early_stopping_patience"],
        use_tensorboard=TRANSFORMER_TRAINING["use_tensorboard"],
        tensorboard_log_dir=tensorboard_dir,
        use_scheduler=TRANSFORMER_TRAINING["use_scheduler"],
        epoch_offset=resume_epoch,
    )
    
    # Restore trainer state if resuming
    if checkpoint_to_load:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and trainer.scheduler:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore loss history
        trainer.train_losses = checkpoint.get('train_losses', [])
        trainer.val_losses = checkpoint.get('val_losses', [])
        trainer.temporal_losses = checkpoint.get('temporal_losses', [])
        trainer.augmentation_losses = checkpoint.get('augmentation_losses', [])
        trainer.within_window_losses = checkpoint.get('within_window_losses', [])
        trainer.adjacent_subwindow_losses = checkpoint.get('adjacent_subwindow_losses', [])
        trainer.val_temporal_losses = checkpoint.get('val_temporal_losses', [])
        trainer.val_augmentation_losses = checkpoint.get('val_augmentation_losses', [])
        trainer.val_within_window_losses = checkpoint.get('val_within_window_losses', [])
        trainer.val_adjacent_subwindow_losses = checkpoint.get('val_adjacent_subwindow_losses', [])
        
        # Restore best model tracking
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        trainer.best_epoch = checkpoint.get('best_epoch', -1)
        trainer.best_model_state = checkpoint.get('best_model_state', None)
        trainer.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        
        print(f"   ✅ Restored trainer state (optimizer, scheduler, loss history)")
        print(f"   📊 Continuing from epoch {resume_epoch}")
    
    # Train
    max_epochs = TRANSFORMER_TRAINING["epochs"]
    remaining_epochs = max_epochs - resume_epoch
    
    if remaining_epochs > 0:
        trainer.train(epochs=remaining_epochs)
    else:
        print(f"   ⚠️  Model already completed {max_epochs} epochs, skipping training")
    
    # Save models
    trainer.save_model(final_model_path)
    if trainer.best_epoch >= 0 and TRANSFORMER_TRAINING["save_best_model"]:
        trainer.restore_best_model()
        trainer.save_model(best_model_path)
    
    # Collect results before cleanup
    result = {
        'model_index': model_index,
        'model_name': model_name,
        'n_params': n_params,
        'final_val_loss': trainer.val_losses[-1] if trainer.val_losses else None,
        'best_val_loss': trainer.best_val_loss if trainer.best_epoch >= 0 else None,
        'best_epoch': trainer.best_epoch + 1 if trainer.best_epoch >= 0 else None,
        'epochs_trained': len(trainer.train_losses),
    }
    
    # Print results
    print(f"\n💾 Saved to: {model_dir}/")
    if trainer.val_losses:
        print(f"   Final val loss: {trainer.val_losses[-1]:.4f}")
    if trainer.best_epoch >= 0:
        print(f"   Best val loss:  {trainer.best_val_loss:.4f} (epoch {trainer.best_epoch + 1})")
    
    # Clean up
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train transformer models with WSL2 multiprocessing support"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "resume", "fresh", "specific"],
        default="auto",
        help="Training mode: auto (skip completed, resume incomplete), "
             "resume (resume specific model), fresh (start all from scratch), "
             "specific (train only specified models)",
    )
    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        help="Model index to resume (only used with --mode resume)",
    )
    parser.add_argument(
        "--models",
        type=int,
        nargs="+",
        default=[],
        help="Specific model indices to train (only used with --mode specific)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of data loading workers (default: from config.py)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start all models from scratch (ignores checkpoints)",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        default=True,  # Launch by default
        help="Launch TensorBoard automatically (default: True)",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_false",
        dest="tensorboard",
        help="Don't launch TensorBoard",
    )
    
    args = parser.parse_args()
    
    # Override mode if --fresh is used
    if args.fresh:
        args.mode = "fresh"
    
    # Get paths
    paths = get_data_paths()
    
    # Launch TensorBoard if requested
    tensorboard_process = None
    if args.tensorboard:
        print("=" * 80)
        print("📊 Launching TensorBoard...")
        tensorboard_dir = paths['tensorboard_base_dir']
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        try:
            # Launch TensorBoard in background
            tensorboard_process = subprocess.Popen(
                ['tensorboard', '--logdir', tensorboard_dir, '--port', '6006', '--host', '0.0.0.0'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            time.sleep(2)  # Give it a moment to start
            
            # Check if it's still running (didn't crash immediately)
            if tensorboard_process.poll() is None:
                print(f"   ✅ TensorBoard launched successfully!")
                print(f"   📍 Access at: http://localhost:6006")
                print(f"   📂 Log directory: {tensorboard_dir}")
            else:
                # Process exited immediately (probably error)
                stdout, stderr = tensorboard_process.communicate()
                print(f"   ⚠️  TensorBoard failed to start: {stderr.decode()[:200]}")
                print(f"   💡 You can manually launch: tensorboard --logdir {tensorboard_dir}")
                tensorboard_process = None
        except FileNotFoundError:
            print(f"   ⚠️  TensorBoard not found. Install with: pip install tensorboard")
            print(f"   💡 Or launch manually: tensorboard --logdir {tensorboard_dir}")
        except Exception as e:
            print(f"   ⚠️  Could not launch TensorBoard: {e}")
            print(f"   💡 You can manually launch: tensorboard --logdir {tensorboard_dir}")
    
    # Load data
    print("=" * 80)
    print("📂 Loading data splits...")
    train_df, val_df, test_df = load_data_splits(paths['splits_path'])
    
    # Create dataloaders
    num_workers = args.workers if args.workers is not None else TRANSFORMER_TRAINING["num_workers"]
    train_loader, val_loader = create_dataloaders(train_df, val_df, num_workers=num_workers)
    
    # Generate model configs
    model_configs = generate_model_configs()
    num_models = len(model_configs)
    
    print("\n" + "=" * 80)
    print(f"🚀 TRAINING GRID: {num_models} Models")
    print(f"   Mode: {args.mode}")
    print(f"   Data loading workers: {num_workers}")
    print("=" * 80)
    
    # Determine which models to train
    models_to_train = []
    models_to_skip = []
    
    if args.mode == "specific":
        models_to_train = [i for i in args.models if 0 <= i < num_models]
        models_to_skip = [i for i in range(num_models) if i not in models_to_train]
        print(f"📌 Training specific models: {models_to_train}")
        
    elif args.mode == "fresh":
        models_to_train = list(range(num_models))
        print(f"🔄 Starting all models from scratch (ignoring existing checkpoints)")
        
    elif args.mode == "resume":
        if args.resume < 0 or args.resume >= num_models:
            raise ValueError(f"Resume index {args.resume} out of range [0, {num_models-1}]")
        models_to_train = [args.resume]
        models_to_skip = [i for i in range(num_models) if i != args.resume]
        print(f"🔄 Resuming model {args.resume}")
        
    else:  # "auto" mode
        for i, cfg in enumerate(model_configs):
            model_dir = os.path.join(paths['models_base_dir'], cfg['name'])
            if is_model_complete(model_dir):
                models_to_skip.append(i)
            else:
                models_to_train.append(i)
        
        if models_to_skip:
            print(f"✅ Skipping {len(models_to_skip)} completed models: {models_to_skip}")
        if models_to_train:
            print(f"📋 Training {len(models_to_train)} models: {models_to_train}")
    
    print("=" * 80)
    
    # Train models
    results = []
    for model_index in models_to_train:
        config = model_configs[model_index]
        result = train_model(model_index, config, train_loader, val_loader, paths, args)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 80)
    if results:
        print(f"\n📊 Trained {len(results)} model(s):")
        for r in results:
            print(f"   {r['model_name']}: val_loss={r['final_val_loss']:.4f if r['final_val_loss'] else 'N/A'}")
    
    # Note about TensorBoard
    if tensorboard_process and tensorboard_process.poll() is None:
        print(f"\n📊 TensorBoard is still running at: http://localhost:6006")
        print(f"   Press Ctrl+C in this terminal to stop TensorBoard, or:")
        print(f"   Run: kill {tensorboard_process.pid}")
    
    # Note about TensorBoard
    if tensorboard_process and tensorboard_process.poll() is None:
        print(f"\n📊 TensorBoard is still running at: http://localhost:6006")
        print(f"   Press Ctrl+C in this terminal to stop TensorBoard, or:")
        print(f"   Run: kill {tensorboard_process.pid}")


if __name__ == "__main__":
    main()
