#!/bin/bash
#SBATCH --job-name=microstruct_sim
#SBATCH --partition=shodhp
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100-SXM4:8
#SBATCH --ntasks-per-node=192
#SBATCH --time=04:00:00
#SBATCH --output=simulation_%j.out
#SBATCH --error=simulation_%j.err

echo "Job started on $(hostname) at $(date)"

# 1. Activate Environment
source $HOME/miniconda/bin/activate env

# 2. Set Library Paths
export AMGX_DIR=$SLURM_SUBMIT_DIR/AMGX
export LD_LIBRARY_PATH=$AMGX_DIR/build:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd $SLURM_SUBMIT_DIR/src

# 3. Verify PyAMGX loads
python -c "import pyamgx; print('PyAMGX loaded successfully')"

LOG_FILE="gpu_usage_log.csv"
echo "Starting GPU monitoring to: $LOG_FILE"

# Run nvidia-smi query every 1 second (-l 1) in the background (&)
nvidia-smi --query-gpu=timestamp,name,pstate,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw \
           --format=csv -l 1 > $LOG_FILE &

# Save the Process ID (PID) of the monitor so we can kill it later
MONITOR_PID=$!

# 4. Run the Script
# Your python script will detect all 8 GPUs automatically
echo "Running Simulation..."
python -u run_fipy.py --config config.yml

kill $MONITOR_PID

echo "Job finished at $(date)"
echo "GPU Logs saved to $LOG_FILE"