# SPTnano HPC Parallel Processing

This directory contains scripts for parallelizing SPTnano's `calculate_time_windowed_metrics` function across multiple CPU cores, both locally and on HPC clusters.

## Files

- **`HPC.py`** - Main parallel processing script
- **`example_usage.py`** - Usage examples for different scenarios
- **`slurm_job.sh`** - SLURM batch script template for cluster usage
- **`HPC_README.md`** - This documentation

## Quick Start

### 1. Test Locally (Recommended First Step)

```bash
# Test with a small sample of your data
python -m SPTnano.HPC --test --test_fraction 0.05

# Or run the example script
python example_usage.py
```

### 2. Process Small Dataset Locally

```bash
python -m SPTnano.HPC \
    --input_file /path/to/your/instant_df.csv \
    --output_dir ./output \
    --n_jobs 4 \
    --chunk_size 500
```

### 3. Process Large Dataset on Cluster

```bash
# Edit slurm_job.sh with your paths and cluster settings
sbatch slurm_job.sh
```

## How It Works

### Chunking Strategy

The script splits your data by `unique_id` (individual tracks) into chunks:

- **Automatic sizing**: Estimates optimal chunk size based on memory usage
- **Manual sizing**: Use `--chunk_size` to specify number of tracks per chunk
- **Memory efficient**: Each chunk is processed independently

### Parallel Processing

- Uses **joblib** for reliable parallel processing
- Works on both local machines and HPC clusters
- Supports both multiprocessing and distributed computing
- Robust error handling and progress reporting

### Output

The script produces two files:
- `instant_df_processed.csv` - Updated instant DataFrame with window UIDs
- `time_windowed_df_processed.csv` - Time-windowed metrics

## Command Line Options

### Required
- `--input_file` - Path to input CSV file containing instant_df

### Optional
- `--output_dir` - Output directory (default: `./output`)
- `--n_jobs` - Number of parallel jobs (-1 for all cores)
- `--chunk_size` - Number of tracks per chunk (auto-estimated if not provided)
- `--time_between_frames` - Time between frames in seconds (default: 0.01)

### Processing Parameters
- `--window_size` - Window size in frames (default: 60)
- `--overlap` - Overlap between windows (default: 30)
- `--r2_threshold` - R² threshold for fit quality (default: 0.000001)

### Testing
- `--test` - Run local test with sample data
- `--test_fraction` - Fraction of data for testing (default: 0.1)

## Examples

### Local Testing
```bash
# Quick test with 5% of sample data
python -m SPTnano.HPC --test --test_fraction 0.05

# Test with 2 CPU cores
python -m SPTnano.HPC --test --test_fraction 0.1
```

### Small Dataset (Local)
```bash
python -m SPTnano.HPC \
    --input_file data/sample_df.csv \
    --output_dir ./results \
    --n_jobs 8 \
    --chunk_size 1000
```

### Large Dataset (Cluster)
```bash
python -m SPTnano.HPC \
    --input_file /scratch/user/instant_df_large.csv \
    --output_dir /scratch/user/results \
    --n_jobs -1 \
    --chunk_size 2000 \
    --window_size 60 \
    --overlap 30
```

## Performance Tips

### Chunk Size Selection
- **Small chunks (100-500)**: Better load balancing, more overhead
- **Large chunks (2000-5000)**: Less overhead, potential memory issues
- **Auto-estimation**: Let the script estimate based on ~4GB per chunk

### CPU Usage
- **Local**: Use `--n_jobs 4` or `--n_jobs 8` to leave cores for other tasks
- **Cluster**: Use `--n_jobs -1` to use all allocated cores

### Memory Considerations
- Each chunk loads independently into memory
- Default estimation targets ~4GB per chunk
- Monitor memory usage with `htop` or cluster monitoring tools

## Cluster Usage (SLURM)

### 1. Edit slurm_job.sh
```bash
# Update these variables:
WORK_DIR="/path/to/your/SPTnano"
INPUT_FILE="/path/to/your/instant_df.csv"
OUTPUT_DIR="/path/to/your/output"

# Update SLURM settings:
#SBATCH --account=your_account_name
#SBATCH --partition=your_partition
#SBATCH --mail-user=your_email@institution.edu
```

### 2. Submit Job
```bash
sbatch slurm_job.sh
```

### 3. Monitor Progress
```bash
# Check job status
squeue -u $USER

# Monitor output
tail -f sptnano_parallel_JOBID.out
```

## Troubleshooting

### Common Issues

**Import Error**: Ensure SPTnano is in your Python path
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/SPTnano/src"
```

**Memory Error**: Reduce chunk size
```bash
python -m SPTnano.HPC --chunk_size 500  # Instead of default
```

**Slow Performance**: Check if I/O bound
- Use local storage instead of network storage
- Consider using fewer, larger chunks

### Error Recovery

The script is designed to handle chunk failures gracefully:
- Failed chunks are skipped with warning messages
- Successful chunks are still concatenated
- Check output logs for specific error details

## Integration with Notebooks

You can also call the parallel processing from within a Jupyter notebook:

```python
from HPC import parallel_time_windowed_metrics

# Process data
instant_output, windowed_output = parallel_time_windowed_metrics(
    input_file='data/instant_df.csv',
    output_dir='./results',
    n_jobs=4,
    chunk_size=1000
)

# Load results
instant_df = pd.read_csv(instant_output)
time_windowed_df = pd.read_csv(windowed_output)
```

## Requirements

- Python 3.8+
- joblib
- pandas
- numpy
- SPTnano package

Install dependencies:
```bash
pip install joblib pandas numpy
```

## Support

For issues or questions:
1. Check the error logs in the output directory
2. Try running with `--test` mode first
3. Reduce chunk size if memory issues occur
4. Ensure all file paths are correct and accessible 