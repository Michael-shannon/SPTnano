"""
HPC.py - High Performance Computing script for parallelizing SPTnano time-windowed metrics calculation

This script provides functionality to:
1. Chunk trajectory data by unique_id (tracks)
2. Process chunks in parallel using joblib
3. Concatenate results back together
4. Support both local testing and cluster execution

Usage:
    # As a script:
    python -m SPTnano.HPC --input_file instant_df.csv --output_dir ./output --n_jobs 8 --chunk_size 1000
    
    # In a notebook:
    from SPTnano.HPC import parallel_time_windowed_metrics
"""

import argparse
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Parallel processing
from joblib import Parallel, delayed
import multiprocessing as mp

# SPTnano imports - now we're inside the package
from . import config
from .features import ParticleMetrics


def create_chunks_by_unique_id(df: pd.DataFrame, chunk_size: int = 1000) -> List[pd.DataFrame]:
    """
    Split DataFrame into chunks based on unique_id (tracks).
    
    Args:
        df: Input DataFrame containing trajectory data
        chunk_size: Target number of unique tracks per chunk
        
    Returns:
        List of DataFrame chunks
    """
    unique_ids = df['unique_id'].unique()
    total_tracks = len(unique_ids)
    
    print(f"Total tracks to process: {total_tracks:,}")
    print(f"Target chunk size: {chunk_size} tracks")
    
    # Create chunks of unique_ids
    chunks = []
    for i in range(0, len(unique_ids), chunk_size):
        chunk_ids = unique_ids[i:i + chunk_size]
        chunk_df = df[df['unique_id'].isin(chunk_ids)].copy()
        chunks.append(chunk_df)
    
    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        n_tracks = chunk['unique_id'].nunique()
        n_points = len(chunk)
        print(f"  Chunk {i+1}: {n_tracks} tracks, {n_points:,} points")
    
    return chunks


def process_chunk(chunk_df: pd.DataFrame, 
                  chunk_id: int,
                  time_between_frames: float,
                  processing_params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Process a single chunk of trajectory data.
    
    Args:
        chunk_df: DataFrame chunk to process
        chunk_id: Identifier for this chunk
        time_between_frames: Time between frames in seconds
        processing_params: Dictionary of parameters for calculate_time_windowed_metrics
        
    Returns:
        Tuple of (instant_df_chunk, time_windowed_df_chunk, chunk_id)
    """
    print(f"Processing chunk {chunk_id} with {chunk_df['unique_id'].nunique()} tracks...")
    start_time = time.time()
    
    try:
        # Create ParticleMetrics object for this chunk
        particle_metrics = ParticleMetrics(chunk_df, time_between_frames)
        
        # Calculate time-windowed metrics
        particle_metrics.calculate_time_windowed_metrics(**processing_params)
        
        # Get results
        instant_df_chunk = particle_metrics.metrics_df
        time_windowed_df_chunk = particle_metrics.get_time_windowed_df()
        
        processing_time = time.time() - start_time
        print(f"✓ Chunk {chunk_id} completed in {processing_time:.1f}s")
        
        return instant_df_chunk, time_windowed_df_chunk, chunk_id
        
    except Exception as e:
        print(f"❌ Error processing chunk {chunk_id}: {str(e)}")
        return None, None, chunk_id


def estimate_chunk_size(df: pd.DataFrame, target_memory_gb: float = 4.0) -> int:
    """
    Estimate appropriate chunk size based on memory constraints.
    
    Args:
        df: Input DataFrame
        target_memory_gb: Target memory usage per chunk in GB
        
    Returns:
        Recommended chunk size (number of tracks)
    """
    # Estimate memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    total_tracks = df['unique_id'].nunique()
    memory_per_track = memory_mb / total_tracks
    
    target_memory_mb = target_memory_gb * 1024
    estimated_chunk_size = int(target_memory_mb / memory_per_track)
    
    # Apply reasonable bounds
    min_chunk_size = 100
    max_chunk_size = 5000
    chunk_size = max(min_chunk_size, min(max_chunk_size, estimated_chunk_size))
    
    print(f"Memory estimation:")
    print(f"  Total data: {memory_mb:.1f} MB")
    print(f"  Memory per track: {memory_per_track:.2f} MB")
    print(f"  Target memory per chunk: {target_memory_gb:.1f} GB")
    print(f"  Recommended chunk size: {chunk_size} tracks")
    
    return chunk_size


def parallel_time_windowed_metrics(input_file: str,
                                   output_dir: str,
                                   n_jobs: int = -1,
                                   chunk_size: int = None,
                                   time_between_frames: float = 0.01,
                                   **metric_params) -> Tuple[str, str]:
    """
    Calculate time-windowed metrics in parallel across multiple CPU cores.
    
    Args:
        input_file: Path to input CSV file containing instant_df
        output_dir: Directory to save output files
        n_jobs: Number of parallel jobs (-1 for all cores)
        chunk_size: Number of tracks per chunk (auto-estimated if None)
        time_between_frames: Time between frames in seconds
        **metric_params: Parameters for calculate_time_windowed_metrics
        
    Returns:
        Tuple of (instant_df_output_path, time_windowed_df_output_path)
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("SPTnano Parallel Time-Windowed Metrics Calculator")
    print("=" * 60)
    
    # Load data
    print(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df):,} trajectory points")
    print(f"✓ Total tracks: {df['unique_id'].nunique():,}")
    
    # Estimate chunk size if not provided
    if chunk_size is None:
        chunk_size = estimate_chunk_size(df)
    
    # Create chunks
    chunks = create_chunks_by_unique_id(df, chunk_size)
    
    # Set up parallel processing
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    print(f"Using {n_jobs} CPU cores for processing")
    
    # Default parameters for time-windowed metrics
    default_params = {
        'window_size': 60,
        'overlap': 30,
        'allow_partial_window': False,
        'min_window_size': 60,
        'fit_method': "r2_threshold",
        'bad_fit_strategy': 'flag',
        'r2_threshold': 0.000001,
        'anomalous_r2_threshold': 0.000001,
        'use_ci': True,
        'ci_multiplier': 1.96,
        'entropy_bins': 18,
        'pausing_speed_threshold': 0.1
    }
    
    # Update with any provided parameters
    processing_params = {**default_params, **metric_params}
    
    print("\nProcessing parameters:")
    for key, value in processing_params.items():
        print(f"  {key}: {value}")
    
    # Process chunks in parallel
    print(f"\nStarting parallel processing...")
    start_time = time.time()
    
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(process_chunk)(
            chunk, 
            i, 
            time_between_frames, 
            processing_params
        ) for i, chunk in enumerate(chunks)
    )
    
    processing_time = time.time() - start_time
    print(f"\n✓ All chunks processed in {processing_time:.1f}s")
    
    # Concatenate results
    print("Concatenating results...")
    instant_dfs = []
    time_windowed_dfs = []
    
    for instant_chunk, time_windowed_chunk, chunk_id in results:
        if instant_chunk is not None and time_windowed_chunk is not None:
            instant_dfs.append(instant_chunk)
            time_windowed_dfs.append(time_windowed_chunk)
        else:
            print(f"⚠️  Chunk {chunk_id} failed - skipping")
    
    # Combine all results
    final_instant_df = pd.concat(instant_dfs, ignore_index=True)
    final_time_windowed_df = pd.concat(time_windowed_dfs, ignore_index=True)
    
    print(f"✓ Final instant_df: {len(final_instant_df):,} points")
    print(f"✓ Final time_windowed_df: {len(final_time_windowed_df):,} windows")
    
    # Save results
    instant_output = os.path.join(output_dir, 'instant_df_processed.csv')
    time_windowed_output = os.path.join(output_dir, 'time_windowed_df_processed.csv')
    
    final_instant_df.to_csv(instant_output, index=False)
    final_time_windowed_df.to_csv(time_windowed_output, index=False)
    
    print(f"✓ Results saved:")
    print(f"  Instant DF: {instant_output}")
    print(f"  Time-windowed DF: {time_windowed_output}")
    
    return instant_output, time_windowed_output


def test_local_with_data(df: pd.DataFrame, 
                         saved_data_dir: str, 
                         sample_fraction: float = 0.1, 
                         n_jobs: int = 2):
    """
    Test the parallel processing locally with provided DataFrame.
    
    Args:
        df: Input DataFrame to test with
        saved_data_dir: Directory to save test files
        sample_fraction: Fraction of data to use for testing
        n_jobs: Number of jobs for local testing
    """
    print("=" * 60)
    print("LOCAL TESTING MODE - Using Provided DataFrame")
    print("=" * 60)
    
    # Take a sample for testing
    unique_ids = df['unique_id'].unique()
    sample_size = max(1, int(len(unique_ids) * sample_fraction))
    test_ids = np.random.choice(unique_ids, size=sample_size, replace=False)
    test_df = df[df['unique_id'].isin(test_ids)]
    
    print(f"Test dataset: {len(test_df):,} points, {len(test_ids)} tracks")
    
    # Save test data
    test_file = os.path.join(saved_data_dir, 'test_instant_df.csv')
    test_df.to_csv(test_file, index=False)
    print(f"✓ Test data saved to: {test_file}")
    
    # Run parallel processing
    output_dir = os.path.join(saved_data_dir, 'test_output')
    
    instant_output, windowed_output = parallel_time_windowed_metrics(
        input_file=test_file,
        output_dir=output_dir,
        n_jobs=n_jobs,
        chunk_size=25,  # Very small chunks for testing
        time_between_frames=0.01
    )
    
    print("✓ Local test completed successfully!")
    print(f"✓ Results available at:")
    print(f"  - {instant_output}")
    print(f"  - {windowed_output}")
    
    return instant_output, windowed_output


def test_local(sample_fraction: float = 0.1, n_jobs: int = 2):
    """
    Test the parallel processing locally with a small sample of data.
    
    Args:
        sample_fraction: Fraction of data to use for testing
        n_jobs: Number of jobs for local testing
    """
    print("=" * 60)
    print("LOCAL TESTING MODE")
    print("=" * 60)
    
    # Load SPTnano config
    saved_data = config.SAVED_DATA
    
    # Load sample data
    sample_file = os.path.join(saved_data, 'sample_df.csv')
    if not os.path.exists(sample_file):
        print(f"Sample file not found: {sample_file}")
        print("Please ensure sample_df.csv exists in your saved_data directory")
        return
    
    print(f"Loading sample data from: {sample_file}")
    df = pd.read_csv(sample_file)
    
    return test_local_with_data(df, saved_data, sample_fraction, n_jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel SPTnano time-windowed metrics calculation")
    
    parser.add_argument("--input_file", type=str, help="Path to input CSV file")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs (-1 for all cores)")
    parser.add_argument("--chunk_size", type=int, default=None, help="Number of tracks per chunk")
    parser.add_argument("--time_between_frames", type=float, default=0.01, help="Time between frames in seconds")
    
    # Processing parameters
    parser.add_argument("--window_size", type=int, default=60, help="Window size in frames")
    parser.add_argument("--overlap", type=int, default=30, help="Overlap between windows")
    parser.add_argument("--r2_threshold", type=float, default=0.000001, help="R² threshold for fit quality")
    
    # Test mode
    parser.add_argument("--test", action="store_true", help="Run local test with sample data")
    parser.add_argument("--test_fraction", type=float, default=0.1, help="Fraction of data for testing")
    
    args = parser.parse_args()
    
    if args.test:
        test_local(sample_fraction=args.test_fraction, n_jobs=2)
    else:
        if not args.input_file:
            print("Error: --input_file is required when not in test mode")
            parser.print_help()
            exit(1)
        
        # Extract metric parameters
        metric_params = {
            'window_size': args.window_size,
            'overlap': args.overlap,
            'r2_threshold': args.r2_threshold
        }
        
        parallel_time_windowed_metrics(
            input_file=args.input_file,
            output_dir=args.output_dir,
            n_jobs=args.n_jobs,
            chunk_size=args.chunk_size,
            time_between_frames=args.time_between_frames,
            **metric_params
        ) 