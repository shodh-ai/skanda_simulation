#!/bin/bash
#SBATCH --job-name=str_gen
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=200
#SBATCH --mem=0
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/gen_log.out
#SBATCH --error=logs/gen_log.err

echo "--- MPI Generation Started on $(hostname) ---"
date

source $HOME/miniconda/bin/activate gen_env
cd $SLURM_SUBMIT_DIR
mkdir -p logs
set -e

# --- CONFIGURATION FOR MPI STEPS ---
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# 4. Run MPI Script
mpirun --bind-to none python generate.py --n 1000000

echo "--- Generation Complete ---"
date