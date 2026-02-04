#!/bin/bash
#SBATCH --job-name=pybamm_mpi
#SBATCH --partition=shodhp
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=192
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=pybamm_log.out
#SBATCH --error=pybamm_log.err

echo "--- PyBaMM MPI Simulation Started on $(hostname) ---"
date

# 1. Activate Environment
source $HOME/miniconda/bin/activate env

# 2. Navigate to Project Root
cd $SLURM_SUBMIT_DIR

# 3. Define Input Files (Relative to project root)
# The script expects files in src or relative paths
TAU_CSV="src/output_parameter_sweep/taufactor_results.csv"
PARAMS_CSV="src/master_parameters.csv"
OUTPUT_DIR="src/final_pybamm_output"

echo "Using Taufactor Data: $TAU_CSV"
echo "Using Parameters: $PARAMS_CSV"

# 4. Run MPI Script
# --bind-to none is critical for oversubscribing or flexible scheduling
mpirun -n 192 --bind-to none python src/run_pybamm.py \
    --tau_csv $TAU_CSV \
    --params_csv $PARAMS_CSV \
    --output_dir $OUTPUT_DIR

echo "--- Simulation Complete ---"
date