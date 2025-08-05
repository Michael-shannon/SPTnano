#!/bin/bash
#SBATCH --job-name=sptnano_parallel
#SBATCH --account=briv_hotel_bank
#SBATCH --partition=hpc_bigmem_a
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=512G
#SBATCH --time=1-0:0:0
#SBATCH --output=sptnano_parallel_%j.out
#SBATCH --error=sptnano_parallel_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mshannon@rockefeller.edu

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start Time: $(date)"

# Activate your conda environment (created from environment_HPC.yml)


source "/ru-auth/local/home/mshannon/miniforge3/etc/profile.d/conda.sh"
eval "$(mamba shell hook --shell bash)"
mamba activate SPTnano_HPC


# Set up paths
export WORK_DIR="/lustre/fs4/briv_lab/store/mshannon/SPTnano"
export INPUT_FILE="/lustre/fs4/briv_lab/scratch/mshannon/2_25_2025_CorticalNeuron_20H20S_FreeHalo_20H77S_77H20S_analyze/saved_data/instant_df.csv"
export OUTPUT_DIR="/lustre/fs4/briv_lab/scratch/mshannon/2_25_2025_CorticalNeuron_20H20S_FreeHalo_20H77S_77H20S_analyze/saved_data/HPC_output"

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