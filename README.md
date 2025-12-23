# Battery Simulation Pipeline

Generates 3D microstructures using physics-based FiPy simulation with optional GPU acceleration, calculates transport properties, and runs PyBaMM battery simulations.

## Features

- **Structure Generation**: Physics-based convection-diffusion-evaporation (FiPy) or Gaussian Random Field
- **GPU Acceleration**: PyAMGX for FiPy linear solver, multi-GPU parallel processing
- **Transport Properties**: Porosity, tortuosity, effective diffusivity (via taufactor)
- **Battery Simulation**: PyBaMM DFN/SPM models

## Quick Start

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python main.py

# Multi-GPU (all available)
python main_multigpu.py

# Multi-GPU (specific GPUs)
python main_multigpu.py --gpus 0,1,2,3,4,5,6 --num-samples 700
```

## Installation

### 1. Python Environment

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 2. GPU Acceleration (Optional)

To enable GPU-accelerated FiPy solving via PyAMGX:

#### Step 2.1: Install CUDA Toolkit via Conda

```bash
# Install miniconda if not present
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
export PATH="$HOME/miniconda3/bin:$PATH"

# Install CUDA toolkit and dependencies
conda install -c nvidia cuda-toolkit cuda-nvcc cuda-libraries-dev nvtx -y
```

#### Step 2.2: Build AMGX from Source

```bash
cd ~/shodh

# Clone AMGX
git clone --recursive https://github.com/NVIDIA/AMGX.git
cd AMGX

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. \
  -DCMAKE_CUDA_COMPILER=$HOME/miniconda3/bin/nvcc \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_CUDA_FLAGS="-I$HOME/miniconda3/include" \
  -DCMAKE_BUILD_TYPE=Release

# Build (takes ~10-15 minutes)
make -j8
```

This creates `libamgxsh.so` in `~/shodh/AMGX/build/`.

#### Step 2.3: Install pyamgx (Python Wrapper)

```bash
cd ~/shodh

# Clone pyamgx
git clone https://github.com/shwina/pyamgx.git
cd pyamgx

# Set AMGX_DIR to AMGX source directory (not build!)
export AMGX_DIR=$HOME/shodh/AMGX

# Activate your venv
source ~/shodh/skanda_simulation/env/bin/activate

# Install without build isolation
pip install -e . --no-build-isolation
```

#### Step 2.4: Set Library Paths

Add to your `~/.bashrc` or run before executing:

```bash
export LD_LIBRARY_PATH=$HOME/shodh/AMGX/build:$HOME/miniconda3/lib:$LD_LIBRARY_PATH
```

#### Step 2.5: Verify Installation

```python
import pyamgx
pyamgx.initialize()
print("PyAMGX initialized successfully!")
```
## Performance

Benchmarked on NVIDIA H200 GPUs with 128³ volume, 20 time steps:

| Metric | Time |
|--------|------|
| Structure generation (per sample) | ~240s |
| Transport calculation (per sample) | ~0.25s |
| Total per sample | ~240s |
