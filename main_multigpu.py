#!/usr/bin/env python
"""
Multi-GPU Battery Simulation Pipeline

Distributes samples across multiple GPUs for parallel processing.

Usage:
  python main_multigpu.py                    # Use all available GPUs
  python main_multigpu.py --gpus 0,1,2,3     # Use specific GPUs
  python main_multigpu.py --gpus 0,1,2,3,4,5,6 --num-samples 700
"""
import os
import sys
import yaml
import pandas as pd
import json
import time
import argparse
import subprocess
from multiprocessing import Pool, Manager
from functools import partial
import numpy as np


def load_config(path="config.yaml"):
    """Load the YAML configuration file."""
    if not os.path.exists(path):
        print(f"Error: Configuration file '{path}' not found.")
        sys.exit(1)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_worker(worker_args):
    """
    Worker function that runs on a specific GPU.
    Each worker processes a subset of sample indices.
    """
    gpu_id, sample_indices, config, img_dir, base_seed = worker_args
    
    # Set GPU for this process BEFORE importing GPU libraries
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Now import GPU-dependent modules
    import structure
    import transport
    
    results = []
    rng = np.random.default_rng(base_seed)
    
    # Pre-generate all porosities to maintain reproducibility
    trans_conf = config["transport"]
    min_eps = trans_conf.get("min_epsilon", 0.3)
    max_eps = trans_conf.get("max_epsilon", 0.6)
    
    # Generate porosities for all samples up to max index
    max_idx = max(sample_indices) + 1
    all_porosities = rng.uniform(min_eps, max_eps, size=max_idx)
    
    for i in sample_indices:
        current_target_porosity = all_porosities[i]
        config["structure"]["target_porosity"] = current_target_porosity
        
        run_data = {
            "id": i,
            "gpu_id": gpu_id,
            "seed_used": base_seed + i,
            "input_psd": config["structure"]["psd_power"],
            "input_porosity": current_target_porosity,
            "voxel_size_um": config["structure"]["voxel_size_um"],
        }
        
        try:
            # Structure generation with timing
            t0 = time.perf_counter()
            file_path, binary_vol = structure.generate_structure(i, config, img_dir)
            structure_time = time.perf_counter() - t0
            run_data["tiff_path"] = file_path
            run_data["structure_time_s"] = round(structure_time, 3)
            
            # Transport calculation with timing
            t0 = time.perf_counter()
            trans_props = transport.calculate_properties(vol_data=binary_vol)
            transport_time = time.perf_counter() - t0
            run_data.update(trans_props)
            run_data["transport_time_s"] = round(transport_time, 3)
            
            run_data["status"] = "success"
            
        except Exception as e:
            run_data["status"] = "failed"
            run_data["error_msg"] = str(e)
        
        results.append(run_data)
    
    return results


def get_available_gpus():
    """Get list of available GPU IDs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        gpus = [int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
        return gpus
    except Exception:
        return [0]  # Default to GPU 0


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Battery Simulation")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs (e.g., 0,1,2,3). Default: all available")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Override number of samples from config")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Determine GPUs to use
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    else:
        gpu_ids = get_available_gpus()
    
    num_gpus = len(gpu_ids)
    
    # Get parameters
    num_samples = args.num_samples or config["general"]["num_samples"]
    base_seed = config["general"]["base_seed"]
    out_dir = config["general"]["output_dir"]
    img_dir = os.path.join(out_dir, "tiff_stacks")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    # Save run configuration
    with open(os.path.join(out_dir, "run_config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    # Distribute samples across GPUs
    all_indices = list(range(num_samples))
    chunks = [[] for _ in range(num_gpus)]
    for idx, sample_idx in enumerate(all_indices):
        chunks[idx % num_gpus].append(sample_idx)
    
    # Print run info
    print("=" * 60)
    print("Multi-GPU Battery Simulation Pipeline")
    print("=" * 60)
    print(f"Total Samples:    {num_samples}")
    print(f"GPUs:             {gpu_ids}")
    print(f"Samples per GPU:  ~{num_samples // num_gpus}")
    print(f"Output:           {out_dir}")
    print("=" * 60)
    
    # Prepare worker arguments
    worker_args = [
        (gpu_ids[i], chunks[i], config.copy(), img_dir, base_seed)
        for i in range(num_gpus)
        if chunks[i]  # Only if there are samples for this GPU
    ]
    
    # Run workers in parallel
    start_time = time.perf_counter()
    
    print(f"\nLaunching {len(worker_args)} workers...")
    with Pool(processes=len(worker_args)) as pool:
        all_results = pool.map(run_worker, worker_args)
    
    total_time = time.perf_counter() - start_time
    
    # Flatten and sort results
    results_list = []
    for worker_results in all_results:
        results_list.extend(worker_results)
    results_list.sort(key=lambda x: x["id"])
    
    # Save results
    df = pd.DataFrame(results_list)
    final_csv = os.path.join(out_dir, "final_results.csv")
    df.to_csv(final_csv, index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    # Per-GPU stats
    for gpu_id in gpu_ids:
        gpu_df = df[df["gpu_id"] == gpu_id]
        if len(gpu_df) > 0:
            struct_avg = gpu_df["structure_time_s"].mean() if "structure_time_s" in gpu_df else 0
            trans_avg = gpu_df["transport_time_s"].mean() if "transport_time_s" in gpu_df else 0
            print(f"GPU {gpu_id}: {len(gpu_df)} samples, struct={struct_avg:.3f}s avg, trans={trans_avg:.3f}s avg")
    
    print("-" * 60)
    
    # Overall stats
    if "structure_time_s" in df.columns:
        print(f"Structure:  {df['structure_time_s'].mean():.3f}s avg, {df['structure_time_s'].sum():.2f}s total")
    if "transport_time_s" in df.columns:
        print(f"Transport:  {df['transport_time_s'].mean():.3f}s avg, {df['transport_time_s'].sum():.2f}s total")
    
    success_count = len(df[df["status"] == "success"])
    print(f"\nTotal wall time:  {total_time:.2f}s")
    print(f"Throughput:       {num_samples / total_time:.2f} samples/sec")
    print(f"Success rate:     {success_count}/{num_samples} ({100*success_count/num_samples:.1f}%)")
    print(f"\nResults saved to: {final_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()

