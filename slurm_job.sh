#!/bin/bash
#SBATCH --job-name=sptnano_parallel
#SBATCH --account=your_account_name
#SBATCH --partition=your_partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=sptnano_parallel_%j.out
#SBATCH --error=sptnano_parallel_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@institution.edu

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start Time: $(date)"

# Load necessary modules (adjust for your cluster)
module load python/3.12
module load gcc/11.2.0

# Activate your conda environment
source activate nanoSPT

# Set up paths
WORK_DIR="/path/to/your/SPTnano"
INPUT_FILE="/path/to/your/instant_df.csv"
OUTPUT_DIR="/path/to/your/output"

# Change to working directory
cd $WORK_DIR

# Print environment information
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the parallel processing
echo "Starting parallel processing..."
python -m SPTnano.HPC \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --n_jobs $SLURM_CPUS_PER_TASK \
    --chunk_size 2000 \
    --time_between_frames 0.01 \
    --window_size 60 \
    --overlap 30 \
    --r2_threshold 0.000001

# Check exit status
if [ $? -eq 0 ]; then
    echo "Processing completed successfully!"
    echo "Output files saved to: $OUTPUT_DIR"
else
    echo "Processing failed!"
    exit 1
fi

echo "End Time: $(date)"
echo "Job completed." 