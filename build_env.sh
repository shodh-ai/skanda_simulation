#!/bin/bash
#SBATCH --job-name=build_amgx
#SBATCH --partition=shodhp
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100-SXM4:8
#SBATCH --ntasks-per-node=192
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=build_log.out
#SBATCH --error=build_log.err

# Stop on first error
set -e 

echo "--- Starting Build Process on $(hostname) ---"

source $HOME/miniconda/bin/activate env

echo "Compiling AMGX..."
cd $SLURM_SUBMIT_DIR/AMGX
rm -rf build
mkdir -p build && cd build

cmake .. \
  -DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_CUDA_FLAGS="-I$CONDA_PREFIX/include" \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_BUILD_TYPE=Release

make -j64

echo "Compiling PyAMGX..."
cd $SLURM_SUBMIT_DIR/pyamgx
rm -rf build dist *.egg-info

export AMGX_DIR=$SLURM_SUBMIT_DIR/AMGX
export CUDA_HOME=$CONDA_PREFIX

pip install -e . --no-build-isolation

echo "--- Build Success! ---"