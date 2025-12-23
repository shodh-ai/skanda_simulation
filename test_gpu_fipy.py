#!/usr/bin/env python
"""
GPU vs CPU Benchmark for FiPy Physics Solver

Usage: python test_gpu_fipy.py [cuda_device] [volume_size]
  cuda_device: GPU device ID (default: 0)
  volume_size: Volume dimension, e.g. 64 for 64³ (default: 64)
"""
import os
import sys
import time
import subprocess
import threading

# === CUDA Setup (must be done before GPU imports) ===
def _setup_cuda(cuda_device=0):
    """Setup CUDA library paths and device."""
    lib_paths = [
        os.path.expanduser("~/shodh/AMGX/build"),
        os.path.expanduser("~/miniconda3/lib"),
    ]
    paths_to_add = [p for p in lib_paths if os.path.exists(p)]
    
    if paths_to_add:
        current = os.environ.get("LD_LIBRARY_PATH", "")
        new_paths = os.pathsep.join(paths_to_add)
        os.environ["LD_LIBRARY_PATH"] = f"{new_paths}{os.pathsep}{current}" if current else new_paths
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

# Parse arguments
CUDA_DEVICE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
VOLUME_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 64

_setup_cuda(CUDA_DEVICE)

import numpy as np
from structure import generate_physics_structure, FIPY_AVAILABLE, PYAMGX_AVAILABLE


# === GPU Monitor ===
class GPUMonitor:
    """Monitor GPU utilization and memory in background thread."""
    
    def __init__(self, device_id=0, interval=0.05):
        self.device_id = device_id
        self.interval = interval
        self.running = False
        self.thread = None
        self.samples = []
        self.baseline_memory = 0
    
    def _query_gpu(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits", f"--id={self.device_id}"],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                return int(parts[0]), int(parts[1])
        except Exception:
            pass
        return 0, 0
    
    def _monitor_loop(self):
        while self.running:
            self.samples.append(self._query_gpu())
            time.sleep(self.interval)
    
    def start(self):
        self.running = True
        self.samples = []
        _, self.baseline_memory = self._query_gpu()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        
        if not self.samples:
            return None
        
        utils = [s[0] for s in self.samples]
        mems = [s[1] for s in self.samples]
        return {
            "max_util": max(utils),
            "avg_util": sum(utils) / len(utils),
            "baseline_mem": self.baseline_memory,
            "peak_mem": max(mems),
            "mem_increase": max(mems) - self.baseline_memory,
        }


# === Benchmark ===
def run_benchmark(shape, use_gpu, monitor=None):
    """Run FiPy physics simulation and measure time."""
    params = {
        "target_porosity": 0.40,
        "drying_intensity": 0.3,
        "time_steps": 20,
        "dt": 0.1,
        "velocity_scale": 0.6,
        "diff_pore": 1.0,
        "diff_solid": 0.1,
    }
    
    if monitor:
        monitor.start()
    
    start = time.time()
    result = generate_physics_structure(shape, params, seed=42, use_gpu_fipy=use_gpu)
    elapsed = time.time() - start
    
    gpu_stats = monitor.stop() if monitor else None
    porosity = np.mean(result)
    
    return elapsed, porosity, gpu_stats


def main():
    shape = (VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE)
    
    print("=" * 60)
    print("FiPy GPU vs CPU Benchmark")
    print("=" * 60)
    print(f"Volume:     {shape} ({VOLUME_SIZE**3:,} cells)")
    print(f"GPU Device: {CUDA_DEVICE}")
    print(f"FiPy:       {'Available' if FIPY_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"PyAMGX:     {'Available' if PYAMGX_AVAILABLE else 'NOT AVAILABLE'}")
    print("=" * 60)
    
    if not FIPY_AVAILABLE:
        print("ERROR: FiPy is required")
        return
    
    # GPU benchmark
    gpu_time, gpu_porosity, gpu_stats = None, None, None
    if PYAMGX_AVAILABLE:
        print("\n[GPU] Running...")
        monitor = GPUMonitor(device_id=CUDA_DEVICE)
        gpu_time, gpu_porosity, gpu_stats = run_benchmark(shape, use_gpu=True, monitor=monitor)
        print(f"[GPU] Time: {gpu_time:.2f}s | Porosity: {gpu_porosity:.4f}")
        
        if gpu_stats:
            print(f"      Max Util: {gpu_stats['max_util']}% | "
                  f"Avg Util: {gpu_stats['avg_util']:.1f}% | "
                  f"Memory: +{gpu_stats['mem_increase']} MB")
    else:
        print("\n[GPU] Skipped (PyAMGX not available)")
    
    # CPU benchmark
    print("\n[CPU] Running...")
    cpu_time, cpu_porosity, _ = run_benchmark(shape, use_gpu=False)
    print(f"[CPU] Time: {cpu_time:.2f}s | Porosity: {cpu_porosity:.4f}")
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if gpu_time and cpu_time:
        speedup = cpu_time / gpu_time
        print(f"CPU Time:    {cpu_time:.2f}s")
        print(f"GPU Time:    {gpu_time:.2f}s")
        print(f"Speedup:     {speedup:.2f}x")
        print(f"Match:       {'YES' if abs(cpu_porosity - gpu_porosity) < 1e-6 else 'NO'}")
    else:
        print(f"CPU Time:    {cpu_time:.2f}s")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
