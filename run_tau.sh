#!/bin/bash
#SBATCH --job-name=taufactor_mpi
#SBATCH --partition=shodhp
#SBATCH --nodes=1
#SBATCH --gres=gpu:0             # CPU Only - keep GPUs free
#SBATCH --ntasks-per-node=192    # Use remaining 192 CPUs
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=tau_log.out
#SBATCH --error=tau_log.err

echo "--- Taufactor Analysis Started on $(hostname) ---"
date

# 1. Activate Environment
source $HOME/miniconda/bin/activate env

# 2. Navigate to Project Root
cd $SLURM_SUBMIT_DIR

# 3. Run MPI Script
# Pointing to the specific folder defined in your config.yaml
# Since generation ran inside 'src', the folder is 'src/output_parameter_sweep'
TARGET_DIR="src/output_parameter_sweep"

echo "Processing files in: $TARGET_DIR"

mpirun -n 192 --bind-to none python src/run_tau.py --input_dir $TARGET_DIR

echo "--- Analysis Complete ---"
date